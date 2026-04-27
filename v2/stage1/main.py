import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
from datetime import datetime
import yaml
import logging
import random
from utils.preprocessing import preprocess_data

from utils import load_args, load_yaml_param_settings, seed_everything, get_logger
# from utils import save_hyperparameters
from models.model import Network
from models.loss import CombinedPINNLoss
from models.pinn_dataset import PINNDataset
from modules.trainer_pinn import PINNTrainer


class PINNModelManager:
    def __init__(self, config):
        self.config = config
        self.epochs = self.config['train']['epochs']
        self.batch_size = self.config['train']['batch_size']
        self.hidden_dim = self.config['model']['hidden_dim']
        self.depth = self.config['model']['depth']
        self.n_bins = self.config['model']['n_bins']
        self.std = self.config['model']['augment_std']
        self.mask = self.config['model']['masking_ratio']
        self.lr = self.config['train']['lr']
        self.saving_path = self.config['train']['saving_path']
        self.model_saving_strategy = self.config['train']['model_saving_strategy']
        self.seed = self.config['train']['seed']
        self.num_workers = self.config['train']['num_workers']
        self.device = self.config['train']['device']

        self.pin_memory = self.config['train']['pin_memory']
        self.persist_workers = self.config['train']['persist_workers']
        
        # PINN specific parameters
        self.lambda_mean_reversion = self.config['pinn']['lambda_mean_reversion']
        
        # Experiment folder path - to be set later in main
        self.experiment_save_dir = None
        
        # Logger setup
        self.logger = get_logger(__name__)
        
        # Trainer initialization
        self.trainer = None

    def set_experiment_save_dir(self, save_dir):
        """Set experiment save directory"""
        self.experiment_save_dir = save_dir

    def load_data(self, file_path):
        # Data loading logic
        train_data_frame = pd.read_csv(file_path)

        # Use preprocess_data (original method)
        train_data = preprocess_data(train_data_frame, file_path, self.logger)
            
        # Load data with PINNDataset
        train_dataset = PINNDataset(
            self.config, 
            data=train_data,
            monthly_returns_path=self.config['pinn']['monthly_returns_path'],
            current_file_path=file_path
        )
        
        # Worker initialization function for reproducibility
        def worker_init_fn(worker_id):
            worker_seed = self.seed + worker_id
            np.random.seed(worker_seed)
            random.seed(worker_seed)
            torch.manual_seed(worker_seed)

        train_loader = DataLoader(train_dataset, 
                                batch_size=self.batch_size, 
                                shuffle=True, 
                                num_workers=self.num_workers, 
                                drop_last=True, 
                                pin_memory=self.pin_memory,
                                persistent_workers=self.persist_workers,
                                worker_init_fn=worker_init_fn)
        
        return train_loader, train_dataset
    
    def train_and_inference(self, file_path):
        # Load data and dataset
        train_loader, train_dataset = self.load_data(file_path)

        # Load Physics-Informed Neural Network model
        model = Network(cluster_num = self.config['model']['cluster_num'],
                        dim_in_out = self.config['model']['n_bins'],
                        num_features = self.config['model']['num_features'],
                        hidden_dim = self.config['model']['hidden_dim'],
                        depth = self.config['model']['depth'],
                        heads = self.config['model']['heads'],
                        pre_norm = self.config['model']['pre_norm'],
                        use_simple_rmsnorm = self.config['model']['use_simple_rmsnorm'],
                        cls_init=self.config['model']['cls_init'],
                        dropout_mask = self.config['model']['dropout_mask'],
                        predict_ou_params = self.config['model']['predict_ou_params']
        )
        
        optimizer = AdamW(model.parameters(), lr = self.lr)
        num_training_steps = self.epochs * len(train_loader)
        
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                    num_warmup_steps= self.config['train']['warmup_steps'], 
                                                    num_training_steps=num_training_steps)
            
        #  Use Physics-Informed CombinedPINNLoss
        criterion = CombinedPINNLoss(
            lambda_instance=1.0,
            lambda_cluster=1.0,
            lambda_mean_reversion=self.lambda_mean_reversion,
            dt=1.0/12.0,  # Changed to monthly dt (1/12)
            warmup_steps=int(len(train_loader) * self.epochs *
                           self.config['pinn']['mr_warmup_ratio']),
            min_cluster_size=self.config['pinn']['min_cluster_size'],
            use_daily=False,  # Changed to use monthly data
            use_predicted_params=self.config['pinn']['use_predicted_params']
        )

        # GPU/CPU configuration
        model = model.to(self.device)

        if self.device != 'cpu':
            self.logger.info(f"Using single GPU: {self.device}")

        criterion = criterion.to(self.device)

        # Reset step count at the start of monthly training
        criterion.reset_step_count()

        # Create Trainer
        self.trainer = PINNTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            logger=self.logger,
            train_dataset=train_dataset,
            scheduler=scheduler,
            config=self.config,
            use_accelerator=False,
            accelerator=None
        )

        # Configure progress bar
        max_train_steps = self.epochs * len(train_loader)

        # Execute training
        progress_bar = tqdm(range(max_train_steps), position=1)
        self.trainer.fit(train_loader, self.epochs, progress_bar=progress_bar)

        # Save model
        month_name = os.path.splitext(os.path.basename(file_path))[0]

        os.makedirs(f"{self.experiment_save_dir}/models", exist_ok=True)
        model_save_path = f"{self.experiment_save_dir}/models/{month_name}.pt"
        self.trainer.save_model(model_save_path)

        # Save loss plots
        self.trainer.save_loss_plots(self.experiment_save_dir, month_name)

        # Perform inference
        predictions, embeddings, cluster_probs, ou_params = self.cluster_inference(file_path, model, self.config, train_dataset)

        # Reset loss tracking for next month's training
        self.trainer.reset_loss_tracking()
        
        return predictions

    def cluster_inference(self, file_path, model, config, train_dataset):
        # Use the same data
        test_data_frame = pd.read_csv(file_path)

        # Use preprocess_data (same as training)
        test_data = preprocess_data(test_data_frame, file_path, self.logger)
            
        test_dataset = PINNDataset(
            self.config, 
            data=test_data, 
            is_train=False,
            scaler=train_dataset.get_scaler(),
            bin_edges=train_dataset.get_bin_edges(),
            clip_bounds=train_dataset.get_clip_bounds(),
            num_imputers=train_dataset.get_num_imputers(),
            monthly_returns_path=self.config['pinn']['monthly_returns_path'],
            current_file_path=file_path
        )
            
        test_loader = DataLoader(test_dataset, 
                                 batch_size=self.batch_size, 
                                 shuffle=False,
                                 num_workers=self.num_workers,
                                 drop_last=False,
                                 pin_memory=self.pin_memory,)

        # Perform inference - Collect Embedding, Cluster Probabilities, OU Parameters
        embeddings, cluster_probs, ou_params = self.trainer.inference(test_loader, return_cluster=True, tta_iterations=5)
        clusters = np.argmax(cluster_probs, axis=1)

        # No outlier
        clusters = clusters + 1

        # Generate basic predictions
        predictions = pd.DataFrame({
            'firms': test_data_frame['PERMNO'],
            'clusters': clusters, 
            'MOM1': test_data_frame['MOM1']
        })

        month_name = os.path.splitext(os.path.basename(file_path))[0]

        # Result save directory - using experiment_save_dir
        # 1. Save existing clustering results
        cluster_save_path = f"{self.experiment_save_dir}/clustering/{month_name}.csv"
        os.makedirs(os.path.dirname(cluster_save_path), exist_ok=True)
        predictions.to_csv(cluster_save_path, index=True)  # Changed to index=True
        self.logger.info(f"Clustering results saved to {cluster_save_path}")

        # 2. Save Embeddings
        embeddings_save_path = f"{self.experiment_save_dir}/embeddings/{month_name}.npy"
        os.makedirs(os.path.dirname(embeddings_save_path), exist_ok=True)
        np.save(embeddings_save_path, embeddings)
        self.logger.info(f"Embeddings saved to {embeddings_save_path}")

        # 3. Save Cluster Probabilities
        probs_save_path = f"{self.experiment_save_dir}/probabilities/{month_name}.npy"
        os.makedirs(os.path.dirname(probs_save_path), exist_ok=True)
        np.save(probs_save_path, cluster_probs)
        self.logger.info(f"Cluster probabilities saved to {probs_save_path}")

        # 4. Save OU Parameters
        if ou_params is not None:
            ou_params_save_path = f"{self.experiment_save_dir}/ou_parameters/{month_name}.npz"
            os.makedirs(os.path.dirname(ou_params_save_path), exist_ok=True)
            np.savez(ou_params_save_path, 
                    mu=ou_params['mu'], 
                    sigma=ou_params['sigma'], 
                    theta=ou_params['theta'])
            self.logger.info(f"OU parameters saved to {ou_params_save_path}")

        # 5. Save integrated results (for Trading)
        trading_data = pd.DataFrame({
            'firms': test_data_frame['PERMNO'],
            'clusters': clusters,
            'MOM1': test_data_frame['MOM1']
        })

        # Add OU parameters to DataFrame
        if ou_params is not None:
            # Sample-wise weighted OU parameters (existing)
            trading_data['mu'] = ou_params['mu']
            trading_data['sigma'] = ou_params['sigma']
            trading_data['theta'] = ou_params['theta']

            # Also add cluster-wise fixed OU parameters
            # Store fixed parameters for the primary cluster of each sample
            primary_cluster_indices = clusters - 1  # clusters are 1-based, indices are 0-based

            # Map cluster-wise fixed parameters to sample-wise
            # Note: Here we get the parameters of the primary cluster each sample belongs to
            cluster_fixed_mu = np.zeros(len(trading_data))
            cluster_fixed_sigma = np.zeros(len(trading_data))
            cluster_fixed_theta = np.zeros(len(trading_data))

            # Actual cluster-wise fixed parameters are difficult to obtain at inference time
            # Instead, use the average of sample-wise parameters for each cluster
            for cluster_id in range(1, cluster_probs.shape[1] + 1):
                cluster_mask = (clusters == cluster_id)
                if np.sum(cluster_mask) > 0:
                    # Average parameters of samples belonging to this cluster
                    cluster_fixed_mu[cluster_mask] = ou_params['mu'][cluster_mask].mean()
                    cluster_fixed_sigma[cluster_mask] = ou_params['sigma'][cluster_mask].mean()
                    cluster_fixed_theta[cluster_mask] = ou_params['theta'][cluster_mask].mean()

            trading_data['cluster_mu'] = cluster_fixed_mu
            trading_data['cluster_sigma'] = cluster_fixed_sigma
            trading_data['cluster_theta'] = cluster_fixed_theta

        # Also add cluster probabilities (optional)
        for i in range(cluster_probs.shape[1]):
            trading_data[f'prob_cluster_{i+1}'] = cluster_probs[:, i]
            
        trading_save_path = f"{self.experiment_save_dir}/trading_data/{month_name}.csv"
        os.makedirs(os.path.dirname(trading_save_path), exist_ok=True)
        trading_data.to_csv(trading_save_path, index=False)
        self.logger.info(f"Trading data saved to {trading_save_path}")

        return predictions, embeddings, cluster_probs, ou_params


def main(start_year: int = None, start_month: int = None):
    # Load configuration
    args = load_args()
    config = load_yaml_param_settings(args.config)
    
    # Seed everything
    seed_everything(config['train']['seed'])
    
    # Create manager
    manager = PINNModelManager(config)
    
    # Get file list
    file_dir = "./data/monthly"
    files = sorted(glob(f"{file_dir}/*.csv"))

    # Set training start year/month (from config if no parameter, default if not in config)
    if start_year is None or start_month is None:
        # Try to get from data.start_month in YYYY-MM format
        data_start_month = config['data']['start_month']
        if '-' in data_start_month:
            year_str, month_str = data_start_month.split('-')
            if start_year is None:
                start_year = int(year_str)
            if start_month is None:
                start_month = int(month_str)
        else:
            # Fallback to old behavior
            if start_year is None:
                start_year = config['train']['start_year']
            if start_month is None:
                start_month = config['train']['start_month']
    start_date_str = f"{start_year}{start_month:02d}"

    # Filter by extracting year/month from filename (YYYY-MM.csv format)
    filtered_files = []
    for file_path in files:
        filename = os.path.basename(file_path)
        # Filename format: YYYY-MM.csv
        import re
        date_match = re.search(r'(\d{4})-(\d{2})\.csv', filename)
        if date_match:
            year = int(date_match.group(1))
            month = int(date_match.group(2))
            file_date_str = f"{year}{month:02d}"
            if file_date_str >= start_date_str:
                filtered_files.append(file_path)
        else:
            # Include if date cannot be found
            filtered_files.append(file_path)

    print(f"Found {len(files)} total files, {len(filtered_files)} files from {start_year}-{start_month:02d} onwards")

    # Set experiment save directory
    save_dir = config['train']['saving_path']
    os.makedirs(save_dir, exist_ok=True)
    manager.set_experiment_save_dir(save_dir)
    print(f"Saving results to: {save_dir}")

    # Process each file
    for file_path in filtered_files:
        print(f"\nProcessing: {file_path}")
        manager.train_and_inference(file_path)
        
if __name__ == "__main__":
    main(start_year=1999, start_month=12) 