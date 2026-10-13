"""Azure-only DROID runner. Uses real Azure CPU traces; only capacity changes are simulated."""
from __future__ import annotations
import argparse, importlib.util, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
DROID2 = ROOT / "Droid2.0"
sys.path.insert(0, str(DROID2))
spec = importlib.util.spec_from_file_location("droid_v4", DROID2 / "droid.py")
v4 = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = v4
spec.loader.exec_module(v4)


def main(cycles: int, auto: bool, vm_name: str | None):
    art = v4.load_artifacts()
    if art is None:
        raise SystemExit(
            "No compatible real-Azure model artifacts. Put real Azure .csv.gz files in "
            "Droid2.0/data/azure/ and run: python Droid2.0/droid.py --train"
        )
    model, calib, wide, meta = art
    if meta.get("source") != "Azure Public Dataset (real trace)":
        raise SystemExit("Refusing to run: model source is not verified real Azure data.")
    if not len(wide.columns):
        raise SystemExit("No Azure VM traces are available.")

    # One real Azure VM trace at a time. No AWS/GCP/Kubernetes/on-prem provider aliases.
    if vm_name:
        if vm_name not in wide.columns:
            raise SystemExit(f"VM '{vm_name}' not found in Azure trace. Available IDs: {', '.join(map(str, wide.columns[:10]))} ...")
        vm = vm_name
    else:
        vm = str(wide.columns[0])
    series = wide[vm].to_numpy(dtype=float)
    test_start = int(meta.get("test_start_hour", v4.HISTORY - 1))
    start = max(v4.HISTORY - 1, min(test_start, len(series) - 2))
    end = min(len(series) - v4.HORIZON, start + max(1, cycles))
    if end <= start:
        raise SystemExit(f"Not enough real Azure trace hours for a run. Available hours: {len(series)}.")

    # Only instance count is simulated. The raw trace and forecast are never altered or invented.
    instances = v4.BASE_INSTANCES
    last_scaled_hour = -999
    history = []
    residuals = np.asarray(calib["res"], dtype=float)
    print("\nDROID — AZURE-ONLY TRACE RUN")
    print("=" * 76)
    print(f"Dataset: {meta['source']} | real trace hours: {len(series)} | selected VM: {vm}")
    print(f"Forecast: {v4.HORIZON} hour ahead | history: {v4.HISTORY} hours")
    print("No synthetic fallback, no imputed missing readings, and no other provider scenarios.")
    print("IMPORTANT: instance-count changes are simulated; no Azure resources are modified.\n")

    for cycle, t in enumerate(range(start, end), start=1):
        raw_actual = float(series[t])
        raw_forecast = float(v4.predict(model, series, t))
        # Explicit modeling assumption: trace CPU is normalized to BASE_INSTANCES.
        projected_actual = min(100.0, raw_actual * v4.BASE_INSTANCES / instances)
        projected_forecast = min(100.0, raw_forecast * v4.BASE_INSTANCES / instances)
        lower_raw = max(0.0, raw_forecast + float(calib["q10"]))
        upper_raw = max(lower_raw, raw_forecast + float(calib["q90"]))
        lower = min(100.0, lower_raw * v4.BASE_INSTANCES / instances)
        upper = min(100.0, upper_raw * v4.BASE_INSTANCES / instances)
        conf_up = float(np.mean(raw_forecast + residuals >= v4.UP_AT * instances / v4.BASE_INSTANCES))
        conf_down = float(np.mean(raw_forecast + residuals <= v4.DOWN_SAFE * (instances - 1) / v4.BASE_INSTANCES)) if instances > v4.MIN_INST else 0.0
        cooldown = (t - last_scaled_hour) <= v4.COOLDOWN
        action, confidence = "NO_ACTION", None
        reason = f"Projected utilization estimate {projected_forecast:.1f}% (interval {lower:.1f}-{upper:.1f}%)."
        if instances < v4.MAX_INST and projected_actual >= v4.UP_AT:
            action, confidence = "SCALE_UP", 1.0
            reason = f"Projected current utilization estimate {projected_actual:.1f}% reaches the {v4.UP_AT:.0f}% threshold."
        elif instances < v4.MAX_INST and not cooldown and conf_up >= v4.CONF_MIN and upper >= v4.UP_AT:
            action, confidence = "SCALE_UP", conf_up
            reason = f"Forecast interval reaches the {v4.UP_AT:.0f}% scale-up threshold."
        elif instances > v4.MIN_INST and not cooldown and conf_down >= v4.CONF_MIN and projected_actual * instances / (instances - 1) <= v4.DOWN_SAFE and upper < v4.DOWN_SAFE:
            action, confidence = "SCALE_DOWN", conf_down
            reason = f"Forecast supports a simulated one-instance reduction."
        elif cooldown:
            reason = f"Cooldown active. {reason}"

        print(f"\nCycle {cycle}/{end-start} (trace hour {t})")
        print(f"  Azure VM={vm} raw_trace_cpu={raw_actual:.2f}% raw_forecast={raw_forecast:.2f}% projected_util_estimate={projected_forecast:.2f}% [{lower:.2f}, {upper:.2f}] simulated_instances={instances} -> {action}")
        approved = False
        if action != "NO_ACTION":
            approved = auto or input(f"    {reason}\n    Approve simulated {action}? [y/N]: ").strip().lower() == "y"
        before = instances
        if approved:
            instances = min(v4.MAX_INST, instances + 1) if action == "SCALE_UP" else max(v4.MIN_INST, instances - 1)
            last_scaled_hour = t
            status = "APPROVED_SIMULATED"
        elif action != "NO_ACTION":
            status = "REJECTED"
        else:
            status = "NO_ACTION"
        history.append({
            "cycle": cycle, "trace_hour": int(t), "provider": "Azure", "vm": str(vm),
            "source": meta["source"], "raw_trace_cpu": round(raw_actual, 4),
            "raw_forecast_cpu": round(raw_forecast, 4),
            "projected_current_util_estimate": round(projected_actual, 4),
            "projected_forecast_util_estimate": round(projected_forecast, 4),
            "lower_estimate": round(lower, 4), "upper_estimate": round(upper, 4),
            "confidence": None if confidence is None else round(confidence, 4),
            "action": action, "status": status, "instances_before_simulated": before,
            "instances_after_simulated": instances, "reason": reason,
            "capacity_effect_is_assumption": True,
        })

    out = pd.DataFrame(history)
    output = ROOT / "azure_decisions.csv"
    out.to_csv(output, index=False)
    print(f"\nSaved Azure-only decision history: {output}")
    print("\nStatus counts:")
    print(out["status"].value_counts().to_string())
    print("\nThis is a real-data trace evaluation with simulated capacity decisions, not live Azure autoscaling.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DROID against real Azure CPU trace only")
    parser.add_argument("--cycles", type=int, default=6)
    parser.add_argument("--auto", action="store_true", help="approve every recommendation in simulation only")
    parser.add_argument("--vm", type=str, default=None, help="exact Azure VM ID/column to evaluate; defaults to first complete VM")
    args = parser.parse_args()
    main(args.cycles, args.auto, args.vm)
