# %% [markdown]
# # Phase 1 - Data pipeline and business  understanding 
# 
# Goal: create a clean analytical foundation and churn analysis using KKbox dataset
# - Main objectives: Load and validate datasets
# - Understand table relationships
# - Audit data quality
# - Explore churn behavior
# - Build initial business observation

# %%
import pandas as pd
import numpy as np
import gc
import warnings
from pathlib import Path
warnings.filterwarnings("ignore")
#import of libraries


# %%
Raw = Path("../data/raw")
MEMBERS_PATH = Raw / "members_v3.csv"
TRAIN1_PATH = Raw / "train.csv"
TRAIN2_PATH = Raw / "train_v2.csv"
TRANSACTIONS_PATH1 = Raw / "transactions.csv"
TRANSACTIONS_PATH2 = Raw / "transactions_v2.csv"
USER_LOGS_PATH = Raw / "user_logs.csv"
USER_LOGS_V2_PATH = Raw / "user_logs_v2.csv"
TRAIN_PATHS = [TRAIN1_PATH, TRAIN2_PATH]
TRANSACTIONS_PATHS = [TRANSACTIONS_PATH1, TRANSACTIONS_PATH2]

REFERENCE_DATE   = pd.Timestamp("2017-04-01")
CHUNK_SIZE       = 2_000_000
SCALE_POS_WEIGHT = 10
EXPECTED_USERS   = 1_082_190
EXPECTED_CHURN   = 0.0915

PROCESSED = Path("../data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

train1 = pd.read_csv(TRAIN1_PATH)
train2 = pd.read_csv(TRAIN2_PATH)
train = pd.concat([train1, train2], ignore_index=True).drop_duplicates('msno').reset_index(drop=True)
print(f"Train after dedup: {train.shape[0]:,} users, churn rate: {train['is_churn'].mean():.2%}")

transactions_old = pd.read_csv(TRANSACTIONS_PATH1)
transactions_new = pd.read_csv(TRANSACTIONS_PATH2)
transactions = pd.concat([transactions_old, transactions_new], ignore_index=True)
del transactions_old, transactions_new
gc.collect()
print(f"Transactions loaded: {transactions.shape} — staging copies freed")

# %%
all_paths = [MEMBERS_PATH, TRAIN1_PATH, TRAIN2_PATH,
             TRANSACTIONS_PATH1, TRANSACTIONS_PATH2,
             USER_LOGS_PATH, USER_LOGS_V2_PATH]
for p in all_paths:
    status = "✓" if p.exists() else "✗ MISSING"
    print(f"{status}  {p.name}")

# %%
members = pd.read_csv(MEMBERS_PATH)

members['registration_init_time'] = pd.to_datetime(
    members['registration_init_time'].astype(str),
    format='%Y%m%d',
    errors='coerce'
)

birth_valid = members['bd'].between(1940, 2006)
members.loc[~birth_valid, 'bd'] = np.nan

print(f"Members loaded: {members.shape}")
print(f"registration_init_time dtype: {members['registration_init_time'].dtype}")
print(f"bd: {birth_valid.sum():,} valid, {(~birth_valid).sum():,} set to NaN")



# %%
reg_dates = members.set_index('msno')['registration_init_time'].to_dict()
reg_dates_s = pd.Series(reg_dates, dtype='datetime64[ns]')
print(f"Registration lookup built: {len(reg_dates_s):,} users")

# %%
customer_txn_count = (
    transactions.groupby("msno")
    .size()
    .reset_index(name="txn_count")
)

# %%
chunk_accum = []
total_chunks = 0

for filepath in [USER_LOGS_PATH, USER_LOGS_V2_PATH]:
    print(f"\nProcessing: {filepath.name}")
    for chunk in pd.read_csv(filepath, chunksize=CHUNK_SIZE):
        total_chunks += 1

        # Date parsing
        chunk['date'] = pd.to_datetime(
            chunk['date'].astype(str), format='%Y%m%d', errors='coerce'
        )
        chunk = chunk.dropna(subset=['date'])

        # Cap impossible values
        chunk['total_secs'] = chunk['total_secs'].clip(upper=86_400)
        chunk['num_unq']    = chunk['num_unq'].clip(lower=0, upper=10_000)

        # Completion rate
        plays = chunk[['num_25','num_50','num_75','num_985','num_100']].sum(axis=1)
        plays = plays.replace(0, np.nan)
        chunk['completion_rate'] = (chunk['num_100'] / plays).fillna(0.0)

        # First-week tag
        chunk['reg_date']       = chunk['msno'].map(reg_dates_s)
        chunk['days_since_reg'] = (chunk['date'] - chunk['reg_date']).dt.days
        is_first_week           = chunk['days_since_reg'].between(0, 6)

        # Aggregate all rows
        agg_all = chunk.groupby('msno', sort=False).agg(
            date_max       = ('date',            'max'),
            date_count     = ('date',            'count'),
            num_unq_sum    = ('num_unq',         'sum'),
            total_secs_sum = ('total_secs',      'sum'),
            completion_sum = ('completion_rate', 'sum'),
        ).reset_index()

        # Aggregate first-week rows
        fw = chunk[is_first_week].groupby('msno', sort=False).agg(
            fw_num_unq_sum = ('num_unq', 'sum'),
            fw_day_count   = ('date',    'count'),
        ).reset_index()

        agg_all = agg_all.merge(fw, on='msno', how='left')
        agg_all['fw_num_unq_sum'] = agg_all['fw_num_unq_sum'].fillna(0)
        agg_all['fw_day_count']   = agg_all['fw_day_count'].fillna(0)

        chunk_accum.append(agg_all)

        if total_chunks % 5 == 0:
            print(f"  Chunk {total_chunks} done")

print(f"\nTotal chunks processed: {total_chunks}")

# %%
print("combining chunks")
ul_raw = pd.concat(chunk_accum, ignore_index=True)
del chunk_accum
gc.collect()
#collapse to one row per user
ul_agg = ul_raw.groupby('msno', sort=False).agg(
    last_active_date = ('date_max', 'max'),
    activity_day_count = ('date_count', 'sum'),
    num_unq_sum = ('num_unq_sum', 'sum'),
    total_secs_sum = ('total_secs_sum', 'sum'),
    completion_sum = ('completion_sum', 'sum'),
    fw_num_unq_sum = ('fw_num_unq_sum', 'sum'),
    fw_day_count = ('fw_day_count', 'sum'),
).reset_index()
del ul_raw
gc.collect()
#compute metrics
ul_agg['avg_num_unq']         = (ul_agg['num_unq_sum']   / ul_agg['activity_day_count']).round(2)
ul_agg['avg_total_secs']      = (ul_agg['total_secs_sum'] / ul_agg['activity_day_count']).round(1)
ul_agg['avg_completion_rate'] = (ul_agg['completion_sum'] / ul_agg['activity_day_count']).round(4)
ul_agg['first_week_depth'] = (
    ul_agg['fw_num_unq_sum'] / ul_agg['fw_day_count'].replace(0, np.nan)
).fillna(0).round(2)
assert ul_agg['msno'].nunique() == ul_agg.shape[0],"Duplicate msno in ul_agg"
assert ul_agg['avg_total_secs'].max() <= 86_400, "clipping err"
print(f"ul_agg ready: {ul_agg.shape}")
print(ul_agg[['avg_num_unq', 'avg_total_secs', 'avg_completion_rate', 'first_week_depth']].describe())


# %%
#
#
master = train.copy()
before = master.shape[0]
master = master.merge(members, on='msno', how='left')
assert master.shape[0] == before, "Members join changed row count"
master = master.merge(transactions_features, on='msno', how='left')
assert master.shape[0] == before, "Transactions join changed row count"
master = master.merge(
    ul_agg[['msno', 'last_active_date', 'activity_day_count',
            'avg_num_unq', 'avg_total_secs', 'avg_completion_rate', 'first_week_depth']],
    on='msno', how='left'
)
assert master.shape[0] == before, "User logs join changed row count"
#flag missing
master['has_tx'] = master['total_paid'].notna().astype(int)
master['has_logs'] = master['avg_num_unq'].notna().astype(int)
#derived time features
master['tenure_days']           = (REFERENCE_DATE - master['registration_init_time']).dt.days
master['days_since_last_active'] = (REFERENCE_DATE - master['last_active_date']).dt.days
master['revenue_per_day'] = (
    master['total_paid'] / master['tenure_days'].replace(0, np.nan)
).round(4)
print(f"Master dataset ready: {master.shape}")
print(f"has_tx=1: {master['has_tx'].sum():,} users have transaction data")
print(f"has_logs=1: {master['has_logs'].sum():,} users have log data")



# %%
print("=" * 50)
print("Data Pipeline Initialized")
print("=" * 50)
#Row count check
assert master.shape[0] == EXPECTED_USERS, \
    f"FAIL: {master.shape[0]:,} rows is not equal to expected {EXPECTED_USERS:,}"
print(f"check Row count: {master.shape[0]:,}")

#every id must be unique
assert master['msno'].nunique() == master.shape[0], \
    f"FAIL: Duplicate msno found"
print(f"correct, every msno is unique: {master['msno'].nunique():,}")
#churn rate check (must be within 0.2% of expected)
assert abs(master['is_churn'].mean() - EXPECTED_CHURN) < 0.002, \
    f"FAIL: Churn rate {master['is_churn'].mean():.4%} is not within 0.2% of expected {EXPECTED_CHURN:.4%}"
print(f"correct, churn rate is within 0.2% of expected: {master['is_churn'].mean():.2%}")

#Check of negative tenure_days
assert (master['tenure_days'].fillna(0) >= 0).all(), \
    f"FAIL: Negative tenure_days found"
print(f"correct, no negative tenure_days found")

#null rates
null_rates = master.isnull().mean().sort_values(ascending=False)
for col, rate in null_rates[null_rates > 0].items():
    print(f"WARNING: {col} has {rate:.1%} ")

print("Data pipeline completed successfully.")

# %%
MASTER_PATH = PROCESSED / "master.csv"
master.to_csv(MASTER_PATH, index=False)
verify = pd.read_csv(MASTER_PATH, usecols=[ 'msno', 'is_churn' ])
assert verify.shape[0] == EXPECTED_USERS, "save verification failed"
print(f"master.csv saved")
print(f"Rows: {master.shape[0]:,}, Columns: {master.shape[1]:,}")
print(f" Size: {MASTER_PATH.stat().st_size / 1e6:.1f} MB  ")
print(f"{MASTER_PATH}")

# %%
members.head()

# %%
members.shape

# %%
members.info()

# %%
members.isnull().sum()

# %% [markdown]
# ## Initial observation: members dataset
# - Dataset contains large amount of rows
# - Gender of 65.4% of members is unknown
# 

# %%

print(f"Train data loaded: {train.shape}")

# %%
train.head()


# %%
train["is_churn"].value_counts()

# %%
churn_rate = train["is_churn"].mean()
print(f"Churn Rate: {churn_rate:.2%}")

# %%
overlap = set(train1['msno']) & set(train2['msno'])
print(f"Users in both train files: {len(overlap):,}")
combined_dedup = pd.concat([train1, train2]).drop_duplicates('msno')
print(f"After dedup: {combined_dedup.shape[0]:,} users")
print(f"Churn rate after dedup: {combined_dedup['is_churn'].mean():.4f}")

# %% [markdown]
# ## Observation - Churn labels
# - Churn rate = 9,15%, relatively small, only small fraction churned
# - dataset is imbalanced
# - prediction models may bias

# %%
train["is_churn"].value_counts()
train["is_churn"].value_counts(normalize=True)

# %%

try:
    print(f"Combined transactions data loaded: {transactions.shape}")
    print(f"From {transactions['msno'].nunique():,} unique members")
    print(f"Source counts: {len(transactions_old):,} + {len(transactions_new):,} -> {len(transactions):,}")
except NameError:
    print("transactions not defined yet. Reloading now...")
    transactions_old = pd.read_csv(TRANSACTIONS_PATH1)
    transactions_new = pd.read_csv(TRANSACTIONS_PATH2)
    transactions = pd.concat([transactions_old, transactions_new], ignore_index=True)
    print(f"Reloaded combined transactions data: {transactions.shape}")
    print(f"From {transactions['msno'].nunique():,} unique members")
    print(f"Source counts: {len(transactions_old):,} + {len(transactions_new):,} -> {len(transactions):,}")

# %%
transactions.head()

# %%
transactions.shape

# %%
transactions.info()

# %%
transactions.isnull().sum()

# %%
members["msno"].nunique()

# %%

print(f"Total transactions rows: {transactions.shape[0]}")

# %%
transactions["msno"].nunique()

# %%
sample = pd.read_csv(USER_LOGS_PATH, nrows=1000)
print(sample.head())
sample.shape

# %%
transactions["transaction_date"] = pd.to_datetime(transactions["transaction_date"], format="%Y%m%d", errors="coerce")
transactions["membership_expire_date"] = pd.to_datetime(transactions["membership_expire_date"], format="%Y%m%d", errors="coerce")

# %%
LP = (
    transactions.groupby("msno")["transaction_date"].max().reset_index(name="last_transaction_date")
)

# %%
LTV = (
    transactions.groupby("msno")["actual_amount_paid"]
    .sum()
    .reset_index(name="total_paid")
)  


# %%
avg_duration = (
    transactions.groupby("msno")["payment_plan_days"]
    .mean()
    .reset_index(name="avg_plan_duration")
)  

# %%
auto_renew_history= (
    transactions.groupby("msno")["is_auto_renew"]
    .max()
    .reset_index(name="auto_renew_history")
)

# %%
cancel_count = (
    transactions.groupby("msno")["is_cancel"]
    .sum()
    .reset_index(name="cancel_count")
)
cancel_count.head()
cancel_count.shape
cancel_count.info()
cancel_count["cancel_count"].describe()


# %%
last_expire = (
    transactions.groupby("msno")["membership_expire_date"]
    .max()
    .reset_index(name="last_expire_date")
)

# %%
transactions_features = LP
transactions_features = transactions_features.merge(LTV, on="msno", how="left")
transactions_features = transactions_features.merge(avg_duration, on="msno", how="left")
transactions_features = transactions_features.merge(auto_renew_history, on="msno", how="left")
transactions_features = transactions_features.merge(cancel_count, on="msno", how="left")
transactions_features = transactions_features.merge(last_expire, on="msno", how="left")
transactions_features = transactions_features.merge(customer_txn_count, on="msno", how="left")

assert transactions_features['msno'].nunique() == transactions_features.shape[0], "Duplicate msno — merge went wrong"
assert transactions_features.shape[0] == 2_426_143, f"Wrong row count: {transactions_features.shape[0]}"
print(f"✓ transactions_features ready: {transactions_features.shape}")
print(transactions_features.dtypes)



# %%
transactions_features.head()
transactions_features.shape
transactions_features.info()


# %%



