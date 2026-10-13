<<<<<<< HEAD
# DROID 2.0 — Azure real-data forecaster

Only real Azure Public Dataset V2 CPU-reading shards are accepted. Add `.csv.gz` files to `data/azure/`, then run `python droid.py --train`. Synthetic fallback and missing-value imputation are disabled. See the project-root README for the integrated Azure-only simulation. Scaling decisions are simulated, not applied to real Azure resources.
=======
# DROID v4 - predictive auto-scaling with human approval

## Setup (once)
    pip install -r requirements.txt

## Real data (recommended)
1. Open https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetLinksV2.txt
2. Download ONE file named vm_cpu_readings-file-N-of-195.csv.gz (about 1 GB). Do not unzip it.
3. Put it in the folder data/azure/
Without it the code runs on SYNTHETIC data and says so everywhere.

## Run
    python droid.py --train            # loads data, trains, measures accuracy, builds the scorecard
    streamlit run dashboard.py         # approval dashboard (Approve / Reject buttons)
    python droid.py --run --cycles 24  # optional terminal version (add --auto to approve everything)

## What is real and what is simulated
Real: CPU history, accuracy numbers (on hours the model never trained on), confidence (from past errors).
Simulated: the cloud. Scaling changes an instance count; CPU% = work / instances (work stays the same).

### Azure dataset size and training requirements

You do **not** need all 195 Azure V2 shards for an initial experiment. Put a small batch of consecutive numbered `*vm_cpu_readings*` shards in `data/azure/`. The loader samples a bounded set of VM IDs before aggregation to reduce memory use. This lightweight prototype uses the current hour plus the prior two hours to forecast one hour ahead, so it can be tested with a smaller batch. It requires at least 16 consecutive hourly observations for time-ordered train/calibration/test splits; with shards averaging around four hours, try four consecutive shards total (your existing shard 1 plus shards 2–4). Coverage varies, so the code will tell you if more are needed. This short horizon is for pipeline experimentation; use a longer trace and longer horizon before making claims about daily patterns or production performance. With too little data, training stops with a clear message instead of crashing or silently using stale synthetic metrics.
>>>>>>> 77c54d848864e79b54eb776101b2e551df56761b
