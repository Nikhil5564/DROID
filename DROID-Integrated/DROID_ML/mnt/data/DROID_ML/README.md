# DROID ML Prototype

Implements the proposed architecture: Collector -> XGBoost Forecaster -> Recommender -> Human Approval -> Execution Adapter -> feedback.

## Install
`pip install -r requirements.txt`

## Run all stages
`python droid_ml.py --generate --train --run --cycles 5`

## Non-interactive safe simulation
`python droid_ml.py --generate --train --run --cycles 5 --noninteractive`

## Outputs
- `data/droid_dataset.csv`
- `models/xgb_cpu.json`
- `models/metrics.json`
- `logs/audit.jsonl`

No real cloud API is called. The execution adapter is simulated.
