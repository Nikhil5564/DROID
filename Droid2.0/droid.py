"""
DROID v4 - predictive auto-scaling with a human approval step.

Pipeline:  real CPU history -> forecast -> recommendation -> human approval -> simulated scaling

What is REAL here:
  * the CPU history (Azure Public Dataset, if you put a file in data/azure/)
  * the model accuracy numbers (measured on later hours the model never trained on)
  * the confidence score (measured from the model's own past errors, see recommend())
What is a SIMULATION (say so in your pitch):
  * the cloud itself. "Scaling" changes a number, using one stated assumption:
    the work stays the same and is split evenly, so CPU% = work / instances.

Commands:
  python droid.py --train              # load data, train, measure, compare policies
  python droid.py --run --cycles 24    # terminal demo (add --auto to approve everything)
  streamlit run dashboard.py           # the approval dashboard
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

BASE = Path(__file__).resolve().parent
DATA, MODELS, LOGS = BASE / "data", BASE / "models", BASE / "logs"
AZURE_DIR = DATA / "azure"
for folder in (DATA, MODELS, LOGS, AZURE_DIR):
    folder.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- policy (YOUR choices)
# These are operator rules, not things the model learns. Change them freely.
BASE_INSTANCES = 4     # the trace is treated as the load of a service running on 4 instances
MIN_INST, MAX_INST = 2, 8
UP_AT = 80.0           # utilization (%) we do not want to exceed
DOWN_SAFE = 70.0       # after removing an instance, utilization must stay under this (%)
OVERLOAD_AT = 90.0     # an hour at or above this counts as "overloaded" in the scorecard
REACTIVE_DOWN = 30.0   # the plain reactive baseline scales down below this (%)
CONF_MIN = 0.80        # minimum confidence for a forecast-driven action
COOLDOWN = 2           # hours after a scaling action before another forecast-driven one
PROVISION_DELAY = 1    # hours before a newly requested instance is ready
HORIZON = 1            # lightweight experiment: forecast CPU one hour ahead
HISTORY = 3             # use only the current hour and two prior hours; no 24h lag

FEATURES = ["cpu_now", "lag1", "lag2", "mean3", "max3", "hsin", "hcos"]


# ================================================================ 1. DATA
def _azure_file_order(path: str) -> tuple[int, str]:
    """Sort numbered Azure shards numerically (file-2 before file-10)."""
    match = re.search(r"file-(\d+)-of-\d+", Path(path).name)
    return (int(match.group(1)) if match else 10**9, Path(path).name)


def load_azure_hourly(min_coverage: float = 0.80, max_vms: int = 80) -> pd.DataFrame | None:
    """Load a small deterministic sample of Azure VMs without pivoting the full dataset.

    Candidate VM IDs are sampled from the first shard, then only those IDs are
    aggregated across the available shards. This keeps RAM use bounded. Shards
    should be consecutive for the resulting hourly series to be meaningful.
    """
    files = sorted(glob.glob(str(AZURE_DIR / "*vm_cpu_readings*")), key=_azure_file_order)
    if not files:
        return None

    # A candidate pool larger than max_vms lets us choose VMs with the best
    # coverage across the available time window without retaining all VM IDs.
    candidate_limit = max(2000, max_vms * 25)
    candidates: set[str] = set()
    rng = np.random.default_rng(7)
    first = files[0]
    try:
        reader = pd.read_csv(
            first, header=None, names=["ts", "vm", "min", "max", "avg"],
            usecols=["vm"], chunksize=250_000,
        )
        for chunk in reader:
            ids = chunk["vm"].dropna().astype(str).unique()
            remaining = candidate_limit - len(candidates)
            if remaining <= 0:
                break
            if len(ids) > remaining:
                ids = rng.choice(ids, size=remaining, replace=False)
            candidates.update(str(v) for v in ids)
            if len(candidates) >= candidate_limit:
                break
    except Exception as exc:
        raise SystemExit(f"Could not read Azure shard {Path(first).name}: {exc}") from exc

    if len(candidates) < 10:
        raise SystemExit(
            "Found an Azure file, but could not identify enough VM IDs. "
            "Check that the file is a valid Azure CPU-reading CSV.GZ."
        )

    parts = []
    for path in files:
        reader = pd.read_csv(
            path, header=None, names=["ts", "vm", "min", "max", "avg"],
            usecols=["ts", "vm", "avg"], chunksize=250_000,
        )
        for chunk in reader:
            chunk["vm"] = chunk["vm"].astype("string")
            chunk = chunk[chunk["vm"].isin(candidates)]
            if chunk.empty:
                continue
            chunk["ts"] = pd.to_numeric(chunk["ts"], errors="coerce")
            chunk["avg"] = pd.to_numeric(chunk["avg"], errors="coerce")
            chunk = chunk.dropna(subset=["ts", "avg"])
            if chunk.empty:
                continue
            chunk = chunk[np.isfinite(chunk["ts"]) & np.isfinite(chunk["avg"])]
            if chunk.empty:
                continue
            chunk["ts"] = chunk["ts"].astype("int64")
            chunk["hour"] = chunk["ts"] // 3600  # Azure timestamps are seconds.
            parts.append(chunk.groupby(["vm", "hour"])["avg"].agg(["sum", "count"]))

    if not parts:
        raise SystemExit("Azure files were found, but no valid CPU readings were parsed.")

    both = pd.concat(parts).groupby(level=[0, 1]).sum()
    hourly = (both["sum"] / both["count"]).rename("cpu").reset_index()
    wide = hourly.pivot(index="hour", columns="vm", values="cpu").sort_index()
    first_hour, last_hour = int(wide.index.min()), int(wide.index.max())
    wide = wide.reindex(range(first_hour, last_hour + 1))

    coverage = wide.notna().mean()
    keep = coverage[coverage >= min_coverage].sort_values(ascending=False).index
    if len(keep) < min(10, max_vms):
        best = float(coverage.max()) if len(coverage) else 0.0
        raise SystemExit(
            f"Only {len(keep)} sampled VMs cover at least {min_coverage:.0%} of the "
            f"{len(wide)}-hour window (best coverage: {best:.0%}). Add more consecutive "
            "Azure shards or lower the coverage threshold in load_azure_hourly()."
        )

    wide = wide.loc[:, list(keep[:max_vms])]
    # Fill only short gaps; VMs with long gaps are excluded rather than inventing data.
    wide = wide.interpolate(axis=0, limit=2, limit_area="inside")
    wide = wide.ffill(limit=2).bfill(limit=2).dropna(axis=1)
    if wide.shape[1] < min(10, max_vms):
        raise SystemExit(
            f"Only {wide.shape[1]} VMs remain after filling short gaps. "
            "Add more consecutive Azure shards and retry."
        )
    wide.index = range(len(wide))
    return wide.astype(float)


def synth_hourly(n_vms: int = 60, hours: int = 720, seed: int = 7) -> pd.DataFrame:
    """FALLBACK ONLY: invented daily-cycle data so the code runs without the Azure file."""
    rng = np.random.default_rng(seed)
    t = np.arange(hours)
    cols = {}
    for i in range(n_vms):
        base, phase = rng.uniform(8, 55), rng.uniform(0, 24)
        amp = rng.uniform(0.1, 0.7) * base
        weekend = 1 - rng.uniform(0, 0.3) * (((t // 24) % 7) >= 5)
        noise = np.zeros(hours)
        for k in range(1, hours):
            noise[k] = 0.7 * noise[k - 1] + rng.normal(0, 2.0)
        x = (base + amp * np.sin(2 * np.pi * (t - phase) / 24)) * weekend + noise
        for _ in range(rng.integers(0, 4)):
            s = rng.integers(30, hours - 5)
            x[s:s + 3] += rng.uniform(15, 35)
        cols[f"synth-{i:02d}"] = np.clip(x, 1, 100)
    return pd.DataFrame(cols)


def prepare() -> tuple[pd.DataFrame, str]:
    wide = load_azure_hourly()
    source = "Azure Public Dataset (real trace)"
    if wide is None:
        wide = synth_hourly()
        source = "SYNTHETIC fallback (no Azure file found in data/azure/)"
    wide.to_csv(DATA / "hourly.csv", index=False)
    (DATA / "source.txt").write_text(source, encoding="utf-8")
    return wide, source


# ================================================================ 2. FEATURES + MODEL
def feature_row(x: np.ndarray, t: int) -> list[float]:
    """What the model is told at hour t. Only the past (x[t] and earlier) is used."""
    h = t % 24
    w = x[t - 2:t + 1]
    return [x[t], x[t - 1], x[t - 2], w.mean(), w.max(),
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)]


def build_rows(wide: pd.DataFrame, t_from: int, t_to: int):
    X, y, base = [], [], []
    for vm in wide.columns:
        x = wide[vm].to_numpy(dtype=float)
        for t in range(max(t_from, HISTORY - 1), min(t_to, len(x) - HORIZON)):
            X.append(feature_row(x, t))
            y.append(x[t + 1:t + 1 + HORIZON].mean())
            base.append(x[t - HISTORY + 1:t + 1].mean())  # simple recent-history baseline
    # XGBoost accepts numeric feature columns, not pandas object/string columns.
    X = pd.DataFrame(X, columns=FEATURES).apply(pd.to_numeric, errors="coerce")
    y = np.asarray(y, dtype=np.float64)
    base = np.asarray(base, dtype=np.float64)
    numeric_X = X.to_numpy(dtype=np.float64)
    valid = np.isfinite(numeric_X).all(axis=1) & np.isfinite(y) & np.isfinite(base)
    X = X.loc[valid].astype(np.float32).reset_index(drop=True)
    return X, y[valid].astype(np.float32), base[valid].astype(np.float32)


def train() -> dict:
    wide, source = prepare()
    n = len(wide)
    # Short-horizon prototype: 3-hour feature window, 1-hour target.
    # This lowers the data requirement but is not a substitute for a longer trace
    # when evaluating daily patterns or production-like forecasting.
    min_hours = 16
    if n < min_hours:
        raise SystemExit(
            f"Not enough hourly history to train DROID: found {n} hours across "
            f"{wide.shape[1]} VMs, but at least {min_hours} consecutive hours are required. "
            "Your current Azure shard appears to cover only a few hours. Keep it, then add "
            "a small batch of consecutive numbered Azure shards (you do NOT need all 195) "
            "and run python droid.py --train again."
        )

    a, b = int(0.6 * n), int(0.8 * n)  # train | calibrate | test, in TIME order
    Xtr, ytr, _ = build_rows(wide, HISTORY - 1, a - HORIZON)
    Xca, yca, _ = build_rows(wide, a, b - HORIZON)
    Xte, yte, seasonal = build_rows(wide, b, n)
    for split_name, X, y in (("training", Xtr, ytr), ("calibration", Xca, yca), ("test", Xte, yte)):
        if len(X) == 0 or len(y) == 0:
            raise SystemExit(
                f"The {split_name} split is empty ({n} total hours). Add more consecutive "
                "Azure shards; this prototype needs enough data for its 3-hour history and 1-hour forecast."
            )

    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05,
                         subsample=0.85, colsample_bytree=0.9, random_state=42)
    model.fit(Xtr, ytr)

    residuals = yca - model.predict(Xca)
    q10, q90 = np.percentile(residuals, [10, 90])
    pred = model.predict(Xte)
    inside = (yte >= pred + q10) & (yte <= pred + q90)
    metrics = {
        "source": source, "vms": int(wide.shape[1]), "hours": int(n), "test_start_hour": int(b),
        "history_hours": HISTORY, "horizon_hours": HORIZON, "feature_set": FEATURES,
        "train_rows": len(Xtr), "calibration_rows": len(Xca), "test_rows": len(Xte),
        "model_mae": round(float(np.mean(np.abs(yte - pred))), 3),
        "persistence_mae": round(float(np.mean(np.abs(yte - Xte["cpu_now"].to_numpy()))), 3),
        "yesterday_mae": round(float(np.mean(np.abs(yte - seasonal))), 3),  # legacy key; this is the recent-history mean baseline
        "band_80_coverage": round(float(inside.mean()), 3),
    }
    model.save_model(MODELS / "xgb_cpu.json")
    (MODELS / "calibration.json").write_text(json.dumps(
        {"q10": float(q10), "q90": float(q90), "residuals": [round(float(r), 3) for r in residuals]}))
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print("\nMODEL CHECK (on hours the model never saw)")
    print(json.dumps(metrics, indent=2))

    calib = {"res": np.asarray(residuals), "q10": float(q10), "q90": float(q90)}
    comparison = compare(model, calib, wide, b)
    return {"metrics": metrics, "comparison": comparison}


def load_artifacts():
    needed = [MODELS / "xgb_cpu.json", MODELS / "calibration.json", MODELS / "metrics.json", DATA / "hourly.csv"]
    if not all(p.exists() for p in needed):
        return None
    model = XGBRegressor()
    model.load_model(MODELS / "xgb_cpu.json")
    c = json.loads((MODELS / "calibration.json").read_text())
    calib = {"res": np.asarray(c["residuals"]), "q10": c["q10"], "q90": c["q90"]}
    wide = pd.read_csv(DATA / "hourly.csv")
    meta = json.loads((MODELS / "metrics.json").read_text())
    source_path = DATA / "source.txt"
    if source_path.exists():
        current_source = source_path.read_text(encoding="utf-8").strip()
        if (meta.get("source") != current_source or int(meta.get("hours", -1)) != len(wide)
                or int(meta.get("horizon_hours", -1)) != HORIZON
                or meta.get("feature_set") != FEATURES):
            # Avoid presenting old synthetic metrics/model as if they were trained on new data.
            return None
    return model, calib, wide, meta


# ================================================================ 3. SIMULATED INFRASTRUCTURE
def util_of(demand: float, n: int) -> float:
    """CPU% on n instances when the work equals `demand` CPU% on BASE_INSTANCES."""
    return float(min(100.0, demand * BASE_INSTANCES / n))


class Workload:
    def __init__(self, name: str, series: np.ndarray):
        self.name, self.x = name, series
        self.n, self.incoming, self.last, self.snooze = BASE_INSTANCES, [], -99, -99

    def tick(self, t: int):
        ready = [e for e in self.incoming if e[0] <= t]
        self.n += sum(d for _, d in ready)
        self.incoming = [e for e in self.incoming if e[0] > t]

    def util(self, t: int) -> float:
        return util_of(self.x[t], self.n)

    @property
    def pending_up(self) -> bool:
        return any(d > 0 for _, d in self.incoming)

    def apply(self, action: str, t: int):
        if action == "SCALE_UP":
            self.incoming.append((t + PROVISION_DELAY, +1))
        elif action == "SCALE_DOWN":
            self.n -= 1
        self.last = t


# ================================================================ 4. FORECAST -> RECOMMENDATION
def predict(model, x: np.ndarray, t: int) -> float:
    row = pd.DataFrame([feature_row(x, t)], columns=FEATURES)
    return float(np.clip(model.predict(row)[0], 0, 100))


def recommend(model, calib: dict, w: Workload, t: int) -> dict:
    """
    Turn a forecast into SCALE_UP / SCALE_DOWN / NO_ACTION, with a confidence that is MEASURED:

      confidence = the share of the model's past forecast errors (on held-out data) for which
                   the decision would still have been right.

    Example: we want to scale down if the next-6h load stays under a limit. For each past error r
    we ask "if the truth had been forecast + r, would it still be under the limit?" and count the yeses.
    """
    n, x = w.n, w.x
    p = predict(model, x, t)                       # forecast of demand, one hour ahead
    res = calib["res"]
    util_now = w.util(t)
    f_util = util_of(p, n)
    lo_u, hi_u = util_of(max(0, p + calib["q10"]), n), util_of(min(100, p + calib["q90"]), n)
    cool = (t - w.last) <= COOLDOWN

    conf_up = float(np.mean(p + res >= UP_AT * n / BASE_INSTANCES))
    conf_dn = float(np.mean(p + res <= DOWN_SAFE * (n - 1) / BASE_INSTANCES)) if n > MIN_INST else 0.0

    action, conf = "NO_ACTION", None
    reason = f"Forecast next {HORIZON}h on {n} instances is about {f_util:.0f}% (likely {lo_u:.0f}-{hi_u:.0f}%): inside the safe range."

    if n < MAX_INST and not w.pending_up and util_now >= UP_AT:
        action, conf = "SCALE_UP", 1.0
        reason = f"Utilization is already {util_now:.0f}% (limit {UP_AT:.0f}%) on {n} instances."
    elif n < MAX_INST and not w.pending_up and not cool and conf_up >= CONF_MIN:
        action, conf = "SCALE_UP", conf_up
        reason = (f"Forecast next {HORIZON}h on {n} instances: {f_util:.0f}% (likely {lo_u:.0f}-{hi_u:.0f}%), "
                  f"limit {UP_AT:.0f}%. Confidence {conf_up:.0%}.")
    elif n > MIN_INST and not w.pending_up and not cool and conf_dn >= CONF_MIN \
            and util_now * n / (n - 1) <= DOWN_SAFE:
        action, conf = "SCALE_DOWN", conf_dn
        reason = (f"Forecast next {HORIZON}h on {n} instances: {f_util:.0f}% (likely {lo_u:.0f}-{hi_u:.0f}%). "
                  f"With {n - 1} instances about {util_of(p, n - 1):.0f}%, safe limit {DOWN_SAFE:.0f}%. "
                  f"Confidence {conf_dn:.0%}.")
    elif cool:
        reason = f"Cooldown: scaled {t - w.last}h ago. Forecast next {HORIZON}h is about {f_util:.0f}%."

    return {"t": t, "workload": w.name, "instances": n, "util_now": round(util_now, 1),
            "forecast_util": round(f_util, 1), "lower": round(lo_u, 1), "upper": round(hi_u, 1),
            "conf_up": round(conf_up, 3), "conf_down": round(conf_dn, 3),
            "action": action, "confidence": None if conf is None else round(conf, 3), "reason": reason}


# ================================================================ 5. APPROVAL LOOP (shared by terminal + dashboard)
def audit(event: str, payload: dict):
    with (LOGS / "audit.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps({"timestamp": datetime.now().isoformat(timespec="seconds"),
                            "event": event, "payload": payload}) + "\n")


class Sim:
    """Replays real workloads hour by hour. A human approves or rejects every recommendation."""

    def __init__(self, model, calib, series: dict[str, np.ndarray], t0: int):
        self.model, self.calib, self.t = model, calib, t0
        self.w = {k: Workload(k, v) for k, v in series.items()}
        self.pending: dict[str, dict] = {}
        self.history: list[dict] = []
        self.obs: list[dict] = []

    @property
    def now(self) -> int:
        return self.t - 1

    def advance(self):
        for name, rec in list(self.pending.items()):           # undecided requests expire
            self._log(rec, "EXPIRED")
            self.pending.pop(name)
        t = self.t
        for name, w in self.w.items():
            w.tick(t)
            self.obs.append({"t": t, "workload": name, "util": round(w.util(t), 1), "instances": w.n})
            if t < w.snooze:
                continue
            rec = recommend(self.model, self.calib, w, t)
            if rec["action"] == "NO_ACTION":
                self._log(rec, "NO_ACTION")
            else:
                self.pending[name] = rec
        self.t += 1

    def decide(self, name: str, approve: bool):
        rec = self.pending.pop(name, None)
        if rec is None:
            return
        w = self.w[name]
        if approve:
            w.apply(rec["action"], self.now)
        else:
            w.snooze = self.now + 2
        self._log(rec, "APPROVED" if approve else "REJECTED")

    def _log(self, rec: dict, status: str):
        row = {**rec, "status": status}
        self.history.append(row)
        audit("DECISION", row)


def make_sim(model, calib, wide: pd.DataFrame, meta: dict, n_workloads: int = 5) -> Sim:
    """Demo workloads = the 5 VMs with the biggest swings in the test period (a labelled demo choice)."""
    b = meta["test_start_hour"]
    top = wide.iloc[b:].std().sort_values(ascending=False).head(n_workloads).index
    series = {f"vm-{str(c)[:8]}": wide[c].to_numpy() for c in top}
    return Sim(model, calib, series, t0=b)


# ================================================================ 6. SCORECARD: does prediction actually help?
def simulate(policy, model, calib, x: np.ndarray, t0: int, t1: int, name: str = "vm") -> dict:
    w = Workload(name, x)
    overload = inst_hours = actions = 0
    for t in range(t0, t1):
        w.tick(t)
        u = w.util(t)
        overload += u >= OVERLOAD_AT
        inst_hours += w.n
        action = policy(model, calib, w, t)
        if action != "NO_ACTION":
            w.apply(action, t)
            actions += 1
    return {"overload_hours": int(overload), "instance_hours": int(inst_hours), "actions": int(actions)}


def static_policy(model, calib, w, t):
    return "NO_ACTION"


def reactive_policy(model, calib, w, t):
    u, cool = w.util(t), (t - w.last) <= COOLDOWN
    if u >= UP_AT and w.n < MAX_INST and not w.pending_up:
        return "SCALE_UP"
    if u <= REACTIVE_DOWN and w.n > MIN_INST and not cool and not w.pending_up:
        return "SCALE_DOWN"
    return "NO_ACTION"


def droid_policy(model, calib, w, t):
    return recommend(model, calib, w, t)["action"]      # assumes the operator approves everything


def compare(model, calib, wide: pd.DataFrame, b: int) -> dict:
    policies = {"static (4 instances)": static_policy, "reactive thresholds": reactive_policy,
                "DROID (forecast + confidence)": droid_policy}
    out = {}
    for label, pol in policies.items():
        tot = {"overload_hours": 0, "instance_hours": 0, "actions": 0}
        for vm in wide.columns:
            r = simulate(pol, model, calib, wide[vm].to_numpy(), b, len(wide), str(vm))
            for k in tot:
                tot[k] += r[k]
        out[label] = tot
    base = out["static (4 instances)"]["instance_hours"]
    for label, r in out.items():
        r["cost_vs_static_pct"] = round(100 * (r["instance_hours"] / base - 1), 1)
    (MODELS / "comparison.json").write_text(json.dumps(out, indent=2))
    print(f"\nSCORECARD - {wide.shape[1]} VMs x {len(wide) - b} held-out hours each (simulated scaling)")
    print(pd.DataFrame(out).T.to_string())
    return out


# ================================================================ 7. TERMINAL DEMO
def run_demo(cycles: int, auto: bool):
    art = load_artifacts()
    if art is None:
        raise SystemExit("Run `python droid.py --train` first.")
    model, calib, wide, meta = art
    sim = make_sim(model, calib, wide, meta)
    print(f"\nDATA: {meta['source']}\n")
    for _ in range(cycles):
        sim.advance()
        print(f"hour {sim.now}")
        for name, rec in list(sim.pending.items()):
            print(f"  {name}: {rec['action']} - {rec['reason']}")
            ok = auto or input("    approve? [y/N] ").strip().lower() == "y"
            sim.decide(name, ok)
            print("    ->", "APPROVED" if ok else "REJECTED")
    done = pd.DataFrame(sim.history)
    print("\n", done["status"].value_counts().to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--run", action="store_true")
    ap.add_argument("--cycles", type=int, default=24)
    ap.add_argument("--auto", action="store_true", help="approve every recommendation")
    args = ap.parse_args()
    if args.train:
        train()
    if args.run:
        run_demo(args.cycles, args.auto)
    if not (args.train or args.run):
        ap.print_help()
