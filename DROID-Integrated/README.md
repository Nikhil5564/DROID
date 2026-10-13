<<<<<<< HEAD
# DROID — Azure-only, real-data mode

This version is intentionally restricted to the **real Azure Public Dataset CPU trace**. It does not fall back to synthetic data, does not fill missing readings by interpolation/forward-fill/back-fill, and does not generate AWS/GCP/Kubernetes/on-prem scenarios.

## What is real and what is simulated?
- **Real:** CPU readings loaded from the Azure Public Dataset V2 `.csv.gz` shards you provide; training/test splits and forecast errors measured against held-out Azure readings.
- **Simulated:** only the number of instances and the estimated effect of changing capacity. This is an explicit modeling assumption, not real Azure resource telemetry or an API action. No Azure resources are created, deleted, or modified.
- **Not present in this workflow:** fabricated CPU series, fabricated memory/request/latency values, and other cloud-provider labels.

## 1. Add Azure data
Put the original compressed Azure CPU-reading shard files in `Droid2.0/data/azure/`. Keep `.csv.gz` files compressed. Do not edit or synthesize missing values.

## 2. Install dependencies
From this project root in the VS Code terminal:

```cmd
python -m pip install -r Droid2.0\requirements.txt
```

## 3. Train/evaluate on Azure data only

```cmd
python Droid2.0\droid.py --train
```

Training stops with an error if real Azure shards are missing or there are not enough complete real traces. Check:

```cmd
type Droid2.0\data\source.txt
```

Expected source: `Azure Public Dataset (real trace)`.

## 4. Run Azure-only simulation

```cmd
python integrated_run.py --cycles 6
```

Approve or reject each proposed simulated scaling action when prompted. To auto-approve all simulated actions:

```cmd
python integrated_run.py --cycles 6 --auto
```

The default run evaluates one real Azure VM trace. To select a particular VM column, use its exact ID:

```cmd
python integrated_run.py --cycles 6 --vm "YOUR_EXACT_AZURE_VM_ID"
```

Output: `azure_decisions.csv`. Columns distinguish raw observed/forecast CPU from projected utilization estimates that depend on the instance-count assumption.

## Future datasets
Additional datasets must be added through an explicit dataset loader/source configuration and must retain source metadata. A missing/invalid dataset must stop with an error; it must never silently substitute synthetic data.

=======
Dataset source: Azure Public Dataset, Dataset V2 — VM CPU Readings. 
The dataset is distributed across multiple compressed CSV shards, each accessible through a direct download URL.
Link:https://github.com/Azure/AzurePublicDataset/blob/master/AzurePublicDatasetLinksV2.txt
Download only the files with the given link:https://github.com/Azure/AzurePublicDataset/releases/download/dataset-v2/trace_data_vm_cpu_readings_vm_cpu_readings-file-N-of-195.csv.gz 
(where N = from 1 ..... 20 etc)

**BEFORE TRAINING THE MODEL**


**MAKE SURE ALL THE .CSV.GZ FILES ARE DIRECTLY UPLOADED IN THE DIRECTORY /data/azure WITHOUT EXTRACTING THE FILE**(for Droid 2.0 version)
>>>>>>> 77c54d848864e79b54eb776101b2e551df56761b
