from __future__ import annotations
import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# PATHS
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
MODELS = BASE / "models"
LOGS = BASE / "logs"
PLOTS = BASE / "plots"
for folder in (DATA, MODELS, LOGS, PLOTS):
    folder.mkdir(parents=True, exist_ok=True)


# CONFIGURATION
PROVIDERS = [
    "AWS",
    "Azure",
    "GCP",
    "Kubernetes",
    "On-prem",
]
SCALE_UP_THRESHOLD = 85.0
SCALE_DOWN_THRESHOLD = 30.0
CONFIDENCE_THRESHOLD = 0.80
COOLDOWN_CYCLES = 2
CAPACITY_EFFECT = 0.82
FEATURES = [
    "provider_id",
    "hour",
    "cpu",
    "memory",
    "requests",
    "latency",
    "instances",
]


# DATA CLASSES
@dataclass
class Forecast:
    provider: str
    predicted_cpu: float
    lower: float
    upper: float
    confidence: float
@dataclass
class Decision:
    cycle: int
    provider: str
    current_cpu: float
    forecast: float
    lower: float
    upper: float
    confidence: float
    action: str
    reason: str
    risk: str
    approved: bool
    instances_before: int
    instances_after: int


# AUDIT LOGGER
class Audit:
    FILE = LOGS / "audit.jsonl"
    @classmethod
    def write(cls, event, payload):
        record = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "payload": payload,
        }
        with cls.FILE.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


# DATASET GENERATION
def generate_dataset(rows=6000, seed=42):
    rng = np.random.default_rng(seed)
    records = []
    rows_per_provider = rows // len(PROVIDERS)
    for provider_id, provider in enumerate(PROVIDERS):
        base_cpu = 50 + provider_id * 4
        cpu = base_cpu
        for t in range(rows_per_provider):
            hour = t % 24
            daily_pattern = (
                12 * np.sin(2 * np.pi * hour / 24)
            )
            trend = 0.008 * t
            noise = rng.normal(0, 1.5)
            workload_target = (
                base_cpu
                + daily_pattern
                + trend
                + noise
            )
            cpu = (
                0.72 * cpu
                + 0.28 * workload_target
            )
            cpu = float(np.clip(cpu, 10, 96))
            memory = float(
                np.clip(
                    35 + 0.48 * cpu + rng.normal(0, 2.5),
                    10,
                    98,
                )
            )
            requests = max(
                100,
                400
                + 14 * cpu
                + 120 * np.sin(2 * np.pi * hour / 24)
                + rng.normal(0, 40),
            )
            latency = max(
                10,
                25 + 0.75 * cpu + rng.normal(0, 4),
            )
            next_cpu = (
                0.82 * cpu
                + 0.12 * (base_cpu + daily_pattern)
                + 0.04 * (requests / 20)
                + rng.normal(0, 0.8)
            )
            next_cpu = float(np.clip(next_cpu, 0, 100))
            records.append(
                {
                    "provider": provider,
                    "provider_id": provider_id,
                    "hour": hour,
                    "cpu": cpu,
                    "memory": memory,
                    "requests": requests,
                    "latency": latency,
                    "instances": 4,
                    "target_cpu": next_cpu,
                }
            )
    df = pd.DataFrame(records)
    output = DATA / "droid_dataset.csv"
    df.to_csv(output, index=False)
    print(f"Dataset generated: {output}")
    return df

# TRAINING
def train_model():
    dataset = DATA / "droid_dataset.csv"
    if not dataset.exists():
        print("Dataset not found. Generate it first.")
        return None
    df = pd.read_csv(dataset)
    split = int(len(df) * 0.80)
    train = df.iloc[:split]
    test = df.iloc[split:]
    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.9,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=4,
    )
    model.fit(
        train[FEATURES],
        train["target_cpu"],
    )
    prediction = model.predict(
        test[FEATURES]
    )
    mae = mean_absolute_error(
        test["target_cpu"],
        prediction,
    )
    rmse = np.sqrt(
        mean_squared_error(
            test["target_cpu"],
            prediction,
        )
    )
    r2 = r2_score(
        test["target_cpu"],
        prediction,
    )
    model_path = MODELS / "xgb_cpu.json"
    model.save_model(model_path)
    metrics = {
        "mae_cpu_percent": round(float(mae), 4),
        "rmse_cpu_percent": round(float(rmse), 4),
        "r2": round(float(r2), 4),
        "train_rows": len(train),
        "test_rows": len(test),
    }
    with (MODELS / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(metrics, f, indent=2)
    print("\nMODEL PERFORMANCE")
    print("=" * 60)
    print(json.dumps(metrics, indent=2))

    # Plot model performance
    try:
        import matplotlib.pyplot as plt

        n = min(300, len(test))

        plt.figure(figsize=(12, 5))
        plt.plot(
            test["target_cpu"].iloc[:n],
            label="Actual",
        )
        plt.plot(
            prediction[:n],
            label="Predicted",
        )
        plt.xlabel("Test sample")
        plt.ylabel("CPU %")
        plt.title("DROID - Actual vs Predicted CPU")
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            PLOTS / "actual_vs_predicted.png",
            dpi=160,
        )
        plt.close()

    except Exception as error:
        print("Plot skipped:", error)

    return model

# INFRASTRUCTURE SIMULATOR
class InfrastructureSimulator:
    def __init__(self):
        self.state = {}
        for index, provider in enumerate(PROVIDERS):
            self.state[provider] = {
                "cpu": (
                    58
                    + index * 3
                    + random.uniform(-4, 4)
                ),
                "memory": 55 + random.uniform(-5, 5),
                "requests": 900 + random.uniform(-100, 150),
                "instances": 4,
            }
    def observe(self, provider):
        state = self.state[provider]
        demand_change = random.uniform(-2, 5)
        state["cpu"] += (
            demand_change
            + random.uniform(-2.5, 2.5)
        )
        state["cpu"] = float(
            np.clip(
                state["cpu"],
                15,
                96,
            )
        )
        state["memory"] = float(
            np.clip(
                35
                + 0.5 * state["cpu"]
                + random.uniform(-3, 3),
                10,
                98,
            )
        )
        state["requests"] = max(
            100,
            state["requests"]
            * random.uniform(0.96, 1.06),
        )
        state["latency"] = max(
            10,
            25
            + 0.75 * state["cpu"]
            + random.uniform(-4, 6),
        )
        return state.copy()
    def scale_up(self, provider):
        state = self.state[provider]
        before = state["instances"]
        state["instances"] += 1
        # Additional capacity reduces utilization.
        state["cpu"] *= CAPACITY_EFFECT
        state["cpu"] = float(
            np.clip(
                state["cpu"],
                10,
                96,
            )
        )
        return before, state["instances"]
    def scale_down(self, provider):
        state = self.state[provider]
        if state["instances"] <= 2:
            return (
                state["instances"],
                state["instances"],
            )
        before = state["instances"]
        state["instances"] -= 1
        state["cpu"] /= CAPACITY_EFFECT
        state["cpu"] = float(
            np.clip(
                state["cpu"],
                10,
                96,
            )
        )
        return before, state["instances"]


# DROID DECISION ENGINE
class DROID:
    def __init__(
        self,
        model,
        interactive=True,
    ):
        self.model = model
        self.interactive = interactive
        self.infrastructure = (
            InfrastructureSimulator()
        )
        self.last_scale_cycle = {
            provider: -999
            for provider in PROVIDERS
        }
        self.history = []
        # Validation RMSE is used as a practical
        # prototype uncertainty estimate.
        self.rmse = self.load_rmse()
    @staticmethod
    def load_rmse():
        metrics_path = MODELS / "metrics.json"
        if not metrics_path.exists():
            return 2.0
        try:
            metrics = json.loads(
                metrics_path.read_text(
                    encoding="utf-8"
                )
            )

            return float(
                metrics.get(
                    "rmse_cpu_percent",
                    2.0,
                )
            )
        except Exception:
            return 2.0
    def forecast(
        self,
        provider,
        state,
    ):
        provider_id = (
            PROVIDERS.index(provider)
        )
        input_data = pd.DataFrame(
            [
                {
                    "provider_id": provider_id,
                    "hour": datetime.now().hour,
                    "cpu": state["cpu"],
                    "memory": state["memory"],
                    "requests": state["requests"],
                    "latency": state["latency"],
                    "instances": state["instances"],
                }
            ]
        )
        prediction = float(
            self.model.predict(
                input_data
            )[0]
        )
        prediction = float(
            np.clip(
                prediction,
                0,
                100,
            )
        )
        margin = max(
            2.0,
            1.2 * self.rmse,
        )
        confidence = 0.93
        return Forecast(
            provider=provider,
            predicted_cpu=round(
                prediction,
                2,
            ),
            lower=round(
                max(
                    0,
                    prediction - margin,
                ),
                2,
            ),
            upper=round(
                min(
                    100,
                    prediction + margin,
                ),
                2,
            ),
            confidence=confidence,
        )

    def decide(
        self,
        cycle,
        provider,
        state,
        forecast,
    ):

        cycles_since_scale = (
            cycle
            - self.last_scale_cycle[provider]
        )

        cooldown = (
            cycles_since_scale
            <= COOLDOWN_CYCLES
        )

        overloaded = (
            state["cpu"] >= SCALE_UP_THRESHOLD
            or forecast.upper >= SCALE_UP_THRESHOLD
        )

        safe_to_scale = (
            forecast.confidence
            >= CONFIDENCE_THRESHOLD
            and not cooldown
        )

        if overloaded and safe_to_scale:

            return (
                "SCALE_UP",
                "Current/forecast load is at or above the 85% capacity threshold.",
                "HIGH",
            )

        if (
            forecast.predicted_cpu
            <= SCALE_DOWN_THRESHOLD
            and state["instances"] > 2
            and not cooldown
        ):

            return (
                "SCALE_DOWN",
                "Forecast load is low and capacity can be reduced.",
                "MEDIUM",
            )

        if cooldown:

            reason = (
                f"Scaling cooldown active "
                f"({COOLDOWN_CYCLES} cycles)."
            )

        elif overloaded:

            reason = (
                "Forecast confidence is below policy threshold."
            )

        else:

            reason = (
                "Load remains within the operating range."
            )

        return (
            "NO_ACTION",
            reason,
            "LOW",
        )

    def approval(
        self,
        provider,
        action,
    ):

        if action == "NO_ACTION":
            return False

        if not self.interactive:
            return False

        answer = input(
            f"   Approve {action} for "
            f"{provider}? [y/N]: "
        )

        return answer.strip().lower() == "y"

    def run(
        self,
        cycles=10,
    ):

        print()
        print("DROID ML DECISION ENGINE")
        print("=" * 76)

        for cycle in range(
            1,
            cycles + 1,
        ):

            print(
                f"\nCycle {cycle}/{cycles}"
            )

            for provider in PROVIDERS:

                state = (
                    self.infrastructure.observe(
                        provider
                    )
                )

                forecast = (
                    self.forecast(
                        provider,
                        state,
                    )
                )

                (
                    action,
                    reason,
                    risk,
                ) = self.decide(
                    cycle,
                    provider,
                    state,
                    forecast,
                )

                before = state["instances"]
                after = before
                approved = False

                print(
                    f" {provider:<10} "
                    f"CPU {state['cpu']:5.1f}% -> "
                    f"forecast {forecast.predicted_cpu:5.1f}% "
                    f"[{forecast.lower:.1f}-"
                    f"{forecast.upper:.1f}] | "
                    f"{action:<10} | "
                    f"conf {forecast.confidence:.2f}"
                )

                if action != "NO_ACTION":

                    approved = self.approval(
                        provider,
                        action,
                    )

                    if approved:

                        if action == "SCALE_UP":

                            before, after = (
                                self.infrastructure.scale_up(
                                    provider
                                )
                            )

                        elif action == "SCALE_DOWN":

                            before, after = (
                                self.infrastructure.scale_down(
                                    provider
                                )
                            )

                        self.last_scale_cycle[
                            provider
                        ] = cycle

                        print(
                            f"   APPROVED -> "
                            f"simulated instances: "
                            f"{before} -> {after}"
                        )

                    else:

                        print(
                            "   NOT APPROVED -> "
                            "no infrastructure change"
                        )

                decision = Decision(
                    cycle=cycle,
                    provider=provider,
                    current_cpu=round(
                        state["cpu"],
                        2,
                    ),
                    forecast=forecast.predicted_cpu,
                    lower=forecast.lower,
                    upper=forecast.upper,
                    confidence=forecast.confidence,
                    action=action,
                    reason=reason,
                    risk=risk,
                    approved=approved,
                    instances_before=before,
                    instances_after=after,
                )

                self.history.append(
                    asdict(decision)
                )

                Audit.write(
                    "DECISION",
                    asdict(decision),
                )

        history_path = (
            DATA / "decision_history.csv"
        )

        pd.DataFrame(
            self.history
        ).to_csv(
            history_path,
            index=False,
        )

        self.make_feedback_plot()

        print()
        print("Saved:")
        print(
            f"  {history_path}"
        )
        print(
            f"  {MODELS / 'xgb_cpu.json'}"
        )
        print(
            f"  {MODELS / 'metrics.json'}"
        )
        print(
            f"  {Audit.FILE}"
        )
        print(
            f"  {PLOTS / 'provider_cpu_cycles.png'}"
        )

    def make_feedback_plot(self):

        try:

            import matplotlib.pyplot as plt

            history = pd.DataFrame(
                self.history
            )

            plt.figure(
                figsize=(12, 5)
            )

            for provider in PROVIDERS:

                data = history[
                    history["provider"]
                    == provider
                ]

                plt.plot(
                    data["cycle"],
                    data["current_cpu"],
                    marker="o",
                    label=provider,
                )

            plt.axhline(
                SCALE_UP_THRESHOLD,
                linestyle="--",
                label="Scale-up threshold",
            )

            plt.xlabel("Cycle")
            plt.ylabel("CPU %")
            plt.title(
                "DROID Provider CPU Feedback"
            )
            plt.legend()
            plt.tight_layout()

            plt.savefig(
                PLOTS
                / "provider_cpu_cycles.png",
                dpi=160,
            )

            plt.close()

        except Exception as error:

            print(
                "Feedback plot skipped:",
                error,
            )

# STREAMLIT DASHBOARD
def dashboard():
    import streamlit as st
    st.set_page_config(
        page_title="DROID Control Center",
        page_icon="🤖",
        layout="wide",
    )
    st.title(
        "🤖 DROID CONTROL CENTER"
    )
    st.caption(
        "Decision & Recommendation Infrastructure Orchestration"
    )
    history_path = (
        DATA / "decision_history.csv"
    )
    metrics_path = (
        MODELS / "metrics.json"
    )
    audit_path = Audit.FILE
    if not history_path.exists():
        st.warning(
            "No decision history found. "
            "Run the DROID engine first."
        )
        st.stop()
    history = pd.read_csv(
        history_path
    )
    metrics = {}
    if metrics_path.exists():

        metrics = json.loads(
            metrics_path.read_text(
                encoding="utf-8"
            )
        )

    latest_cycle = int(
        history["cycle"].max()
    )

    latest = history[
        history["cycle"]
        == latest_cycle
    ]

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric(
        "Current Cycle",
        latest_cycle,
    )

    c2.metric(
        "Providers",
        len(latest),
    )

    c3.metric(
        "Scale-Up",
        int(
            (
                history["action"]
                == "SCALE_UP"
            ).sum()
        ),
    )

    c4.metric(
        "Approved",
        int(
            history["approved"]
            .astype(bool)
            .sum()
        ),
    )

    c5.metric(
        "Decisions",
        len(history),
    )

    st.divider()

    # Provider cards
    st.subheader(
        "Infrastructure Status"
    )

    columns = st.columns(
        len(latest)
    )

    for column, (_, row) in zip(
        columns,
        latest.iterrows(),
    ):

        action = row["action"]

        if action == "SCALE_UP":
            status = "🔴 SCALE UP"

        elif action == "SCALE_DOWN":
            status = "🟡 SCALE DOWN"

        else:
            status = "🟢 NORMAL"

        with column:

            st.markdown(
                f"### {row['provider']}"
            )

            st.metric(
                "CPU",
                f"{row['current_cpu']:.1f}%",
            )

            st.metric(
                "Forecast",
                f"{row['forecast']:.1f}%",
            )

            st.write(status)

            st.caption(
                f"Confidence: "
                f"{row['confidence']:.2f}"
            )

    st.divider()

    # Forecast chart
    st.subheader(
        "Current vs Forecast CPU"
    )

    chart = latest[
        [
            "provider",
            "current_cpu",
            "forecast",
        ]
    ].set_index(
        "provider"
    )

    st.line_chart(
        chart
    )

    # Provider cycle chart
    st.subheader(
        "Feedback Across Cycles"
    )

    selected_provider = st.selectbox(
        "Provider",
        PROVIDERS,
    )

    provider_data = history[
        history["provider"]
        == selected_provider
    ].set_index(
        "cycle"
    )

    st.line_chart(
        provider_data[
            [
                "current_cpu",
                "forecast",
                "lower",
                "upper",
            ]
        ]
    )

    # Model performance
    st.subheader(
        "ML Model Performance"
    )

    a, b, c, d = st.columns(4)

    a.metric(
        "MAE",
        f"{metrics.get('mae_cpu_percent', 0):.3f}%",
    )

    b.metric(
        "RMSE",
        f"{metrics.get('rmse_cpu_percent', 0):.3f}%",
    )

    c.metric(
        "R²",
        f"{metrics.get('r2', 0):.4f}",
    )

    d.metric(
        "Test Rows",
        f"{metrics.get('test_rows', 0):,}",
    )

    st.info(
        "Metrics are from the synthetic prototype dataset."
    )

    # Decisions
    st.subheader(
        "Decision History"
    )

    st.dataframe(
        history.sort_values(
            ["cycle", "provider"],
            ascending=[False, True],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Reasoning
    st.subheader(
        "Recommendation Reasoning"
    )

    actions = history[
        history["action"]
        != "NO_ACTION"
    ]

    for _, row in actions.tail(
        10
    ).iloc[::-1].iterrows():

        approved = (
            "APPROVED"
            if bool(row["approved"])
            else "NOT APPROVED"
        )

        with st.expander(
            f"Cycle {int(row['cycle'])} | "
            f"{row['provider']} | "
            f"{row['action']} | "
            f"{approved}"
        ):

            st.write(
                f"**Reason:** {row['reason']}"
            )

            st.write(
                f"**Risk:** {row['risk']}"
            )

            st.write(
                f"**CPU:** "
                f"{row['current_cpu']:.1f}% → "
                f"**Forecast:** "
                f"{row['forecast']:.1f}%"
            )

            st.write(
                f"**Instances:** "
                f"{int(row['instances_before'])} → "
                f"{int(row['instances_after'])}"
            )

    # Audit
    st.subheader(
        "Audit Trail"
    )

    if audit_path.exists():

        lines = (
            audit_path.read_text(
                encoding="utf-8"
            ).splitlines()
        )

        for line in lines[-10:][::-1]:

            try:

                st.json(
                    json.loads(line)
                )

            except Exception:

                st.code(line)

    st.divider()

    st.caption(
        "DROID prototype • Cloud execution is simulated • "
        "Human approval controls consequential actions."
    )


# ============================================================
# CLI
# ============================================================

def cli():

    parser = argparse.ArgumentParser(
        description="Complete DROID ML Decision Engine"
    )

    parser.add_argument(
        "--generate",
        action="store_true",
        help="Generate synthetic training data",
    )

    parser.add_argument(
        "--train",
        action="store_true",
        help="Train XGBoost model",
    )

    parser.add_argument(
        "--run",
        action="store_true",
        help="Run DROID decision engine",
    )

    parser.add_argument(
        "--cycles",
        type=int,
        default=10,
        help="Number of decision cycles",
    )

    parser.add_argument(
        "--noninteractive",
        action="store_true",
        help="Reject consequential actions automatically",
    )

    args = parser.parse_args()

    random.seed(42)
    np.random.seed(42)

    if args.generate:
        generate_dataset()

    if args.train:
        train_model()

    if args.run:

        model_path = (
            MODELS / "xgb_cpu.json"
        )

        if not model_path.exists():

            print(
                "No trained model found. "
                "Run --generate --train first."
            )

            return

        model = XGBRegressor()

        model.load_model(
            model_path
        )

        engine = DROID(
            model=model,
            interactive=(
                not args.noninteractive
            ),
        )

        engine.run(
            args.cycles
        )

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        if get_script_run_ctx() is not None:
            dashboard()
        else:
            cli()

    except Exception:
        cli()