"""
Download CRSP + Compustat data from WRDS for ORCA reproduction.

Produces two pickle files that ORCA's make_data.py expects:
  - ORCA/data/raw_data/inter_permno_crsp.pkl      (daily CRSP)
  - ORCA/data/raw_data/inter_permno_firm_characteristics.pkl  (quarterly Compustat)

Usage:
  python v3/01_download_wrds_data.py
  (will prompt for WRDS password)
"""
import wrds
import pandas as pd
import os

WRDS_USERNAME = 'YOUR_WRDS_USERNAME'  # Replace with your WRDS username
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'ORCA', 'data', 'raw_data')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Connecting to WRDS...")
db = wrds.Connection(wrds_username=WRDS_USERNAME)

# ============================================================
# 1. CRSP Daily Stock File (SKIP if already downloaded)
# ============================================================
crsp_path = os.path.join(OUTPUT_DIR, 'inter_permno_crsp.pkl')
if os.path.exists(crsp_path):
    print(f"\n1/3: CRSP already exists at {crsp_path}, skipping...")
else:
    print("\n1/3: Downloading CRSP daily data (1999-2023)...")
    crsp_query = """
        SELECT a.permno, a.permco, a.date, a.ret, a.prc, a.vol, a.shrout,
               a.openprc, a.askhi, a.bidlo,
               b.ticker, b.comnam, b.cusip, b.siccd, b.shrcd, b.exchcd
        FROM crsp.dsf AS a
        LEFT JOIN crsp.dsenames AS b
            ON a.permno = b.permno
            AND b.namedt <= a.date
            AND a.date <= b.nameendt
        WHERE a.date BETWEEN '1999-01-01' AND '2023-12-31'
            AND b.shrcd IN (10, 11)
            AND b.exchcd IN (1, 2, 3)
    """
    crsp = db.raw_sql(crsp_query)
    print(f"  CRSP daily: {crsp.shape[0]:,} rows, {crsp['permno'].nunique():,} unique PERMNOs")
    crsp.columns = [c.upper() if c != 'date' else c for c in crsp.columns]
    crsp.to_pickle(crsp_path)
    print(f"  Saved to {crsp_path}")

# ============================================================
# 2. Compustat Quarterly (via CCM merged link)
#    Firm characteristics for ORCA's 12 accounting variables
# ============================================================
print("\n2/3: Downloading Compustat quarterly + CCM link (1999-2023)...")
compustat_query = """
    SELECT a.gvkey, a.datadate, b.lpermno, b.lpermco,
           a.tic, a.cusip, a.conm,
           a.atq, a.ltq, a.dlcq, a.dlttq, a.seqq, a.ceqq, a.cheq,
           a.saleq, a.cogsq, a.niq, a.oiadpq, a.ibq, a.piq, a.txtq, a.dpq,
           a.prccq, a.prchq, a.prclq, a.epspxq, a.epsfiq, a.cshoq,
           c.sic, c.naics, c.gsector, c.gind, c.gsubind
    FROM comp.fundq AS a
    INNER JOIN crsp.ccmxpf_lnkhist AS b
        ON a.gvkey = b.gvkey
        AND b.linktype IN ('LU', 'LC')
        AND b.linkprim IN ('P', 'C')
        AND (b.linkdt <= a.datadate OR b.linkdt IS NULL)
        AND (a.datadate <= b.linkenddt OR b.linkenddt IS NULL)
    LEFT JOIN comp.company AS c
        ON a.gvkey = c.gvkey
    WHERE a.datadate BETWEEN '1998-01-01' AND '2023-12-31'
        AND a.indfmt = 'INDL'
        AND a.datafmt = 'STD'
        AND a.popsrc = 'D'
        AND a.consol = 'C'
"""
compustat = db.raw_sql(compustat_query)
print(f"  Compustat quarterly: {compustat.shape[0]:,} rows, {compustat['lpermno'].nunique():,} unique PERMNOs")

char_path = os.path.join(OUTPUT_DIR, 'inter_permno_firm_characteristics.pkl')
compustat.to_pickle(char_path)
print(f"  Saved to {char_path}")

db.close()

print(f"\n{'='*60}")
print(f"  Done! Raw data saved to {OUTPUT_DIR}")
print(f"  CRSP: {crsp.shape}")
print(f"  Compustat: {compustat.shape}")
print(f"{'='*60}")
print(f"\nNext step: cd ORCA && python data/make_data.py")
