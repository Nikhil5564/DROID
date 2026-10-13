"""DROID operator dashboard. Run with:  streamlit run dashboard.py"""
import json

import pandas as pd
import streamlit as st

import droid

st.set_page_config(page_title="DROID Control Center", page_icon="🤖", layout="wide")


@st.cache_resource
def load():
    return droid.load_artifacts()


art = load()
if art is None:
    st.error("No trained model yet. In a terminal run:  python droid.py --train")
    st.stop()
model, calib, wide, meta = art

if "sim" not in st.session_state:
    st.session_state.sim = droid.make_sim(model, calib, wide, meta)
sim = st.session_state.sim


def reset():
    st.session_state.sim = droid.make_sim(model, calib, wide, meta)


st.title("🤖 DROID Control Center")
st.caption("Forecast the load, recommend an action, and let a human approve it. Cloud actions are simulated.")

<<<<<<< HEAD
if meta["source"] == "Azure Public Dataset (real trace)":
    st.success(f"Data: {meta['source']} — {meta['vms']} real Azure VM traces, {meta['hours']} consecutive hours")
else:
    st.error(f"Unexpected data source: {meta['source']}. Azure-only mode refuses other sources.")
    st.stop()
=======
if meta["source"].startswith("Azure"):
    st.success(f"Data: {meta['source']} - {meta['vms']} VMs, {meta['hours']} hours")
else:
    st.warning(f"Data: {meta['source']}. Do not present these numbers as real results.")
>>>>>>> 77c54d848864e79b54eb776101b2e551df56761b

# ---- how good is the forecast? (measured on hours the model never saw)
st.subheader("Forecast quality (held-out hours)")
c1, c2, c3, c4 = st.columns(4)
<<<<<<< HEAD
c1.metric("DROID deployed MAE", f"{meta['model_mae']:.2f} pts")
c2.metric("Persistence MAE", f"{meta['persistence_mae']:.2f} pts")
c3.metric("Recent 3-hour mean MAE", f"{meta.get('recent_mean_mae', meta.get('yesterday_mae', float('nan'))):.2f} pts")
c4.metric("Prediction-band coverage", f"{meta['band_80_coverage']:.0%}")
selected_forecaster = meta.get("selected_forecaster", "xgboost (existing trained artifact)")
xgb_mae = meta.get("xgboost_mae", meta["model_mae"])
st.caption(f"Selected forecaster: {selected_forecaster}. Raw XGBoost MAE on the same held-out data: {xgb_mae:.2f} pts. When retrained, selection uses only the calibration split. Lower MAE is better; results from this short trace are experimental, not production validation.")
=======
c1.metric("DROID error (MAE)", f"{meta['model_mae']:.2f} pts")
c2.metric("'Same as now' error", f"{meta['persistence_mae']:.2f} pts")
c3.metric("Recent 3-hour mean error", f"{meta['yesterday_mae']:.2f} pts")
c4.metric("80% band really covers", f"{meta['band_80_coverage']:.0%}")
st.caption(f"Lower error is better. This lightweight prototype uses a {meta.get('history_hours', 3)}-hour history to forecast {meta.get('horizon_hours', 1)} hour ahead. Results from a short trace are experimental, not production validation.")
>>>>>>> 77c54d848864e79b54eb776101b2e551df56761b

st.divider()

# ---- controls
b1, b2, b3 = st.columns([1, 1, 6])
<<<<<<< HEAD
trace_end = min(len(w.x) for w in sim.w.values()) - droid.HORIZON if sim.w else sim.t
finished = sim.t >= trace_end
b1.button("Advance 1 hour ▶", on_click=sim.advance, type="primary", disabled=finished)
b2.button("Reset", on_click=reset)
b3.markdown(f"**Replay hour:** {sim.now}")
if finished:
    st.info("End of the available real Azure trace. Reset to replay the held-out segment again.")
=======
b1.button("Advance 1 hour ▶", on_click=sim.advance, type="primary")
b2.button("Reset", on_click=reset)
b3.markdown(f"**Replay hour:** {sim.now}")
>>>>>>> 77c54d848864e79b54eb776101b2e551df56761b

# ---- the approval queue: the centerpiece
st.subheader("Recommendations waiting for approval")
if not sim.pending:
    st.info("Nothing waiting. Press 'Advance 1 hour' to move time forward.")
for name, rec in list(sim.pending.items()):
    with st.container(border=True):
        left, mid, right = st.columns([6, 1, 1])
        icon = "🔴" if rec["action"] == "SCALE_UP" else "🟡"
        left.markdown(f"### {icon} {name}: {rec['action'].replace('_', ' ')}")
        left.write(rec["reason"])
        left.caption(f"Confidence {rec['confidence']:.0%} · now {rec['util_now']:.0f}% on "
                     f"{rec['instances']} instances · forecast {rec['forecast_util']:.0f}%")
        mid.button("Approve ✅", key=f"ok_{name}_{rec['t']}", on_click=sim.decide, args=(name, True))
        right.button("Reject ❌", key=f"no_{name}_{rec['t']}", on_click=sim.decide, args=(name, False))

# ---- fleet status + history charts
st.subheader("Fleet")
<<<<<<< HEAD
cols = st.columns(max(1, len(sim.w)))
for col, (name, w) in zip(cols, sim.w.items()):
    last = [o for o in sim.obs if o.get("workload") == name]
    col.metric(
        name,
        f"{w.n} instances",
        f"{last[-1]['util']:.0f}% CPU" if last else None,
        delta_color="off",
    )

st.subheader("Simulation charts")
if sim.obs:
    obs = pd.DataFrame(sim.obs)
    required = {"t", "workload", "util", "instances"}
    missing = required - set(obs.columns)
    if missing:
        st.warning(f"Charts cannot render. Missing columns: {sorted(missing)}")
    else:
        obs["t"] = pd.to_numeric(obs["t"], errors="coerce")
        obs["util"] = pd.to_numeric(obs["util"], errors="coerce")
        obs["instances"] = pd.to_numeric(obs["instances"], errors="coerce")
        obs = obs.dropna(subset=["t", "workload"])
        if obs.empty:
            st.info("No valid simulation observations to plot yet.")
        else:
            a, b = st.columns(2)
            cpu = obs.pivot_table(
                index="t", columns="workload", values="util", aggfunc="mean"
            ).sort_index()
            instances = obs.pivot_table(
                index="t", columns="workload", values="instances", aggfunc="last"
            ).sort_index()
            with a:
                st.markdown("**CPU utilization (%)**")
                if not cpu.empty:
                    st.line_chart(cpu, height=300)
                else:
                    st.info("No CPU utilization data to display.")
            with b:
                st.markdown("**Instances over time**")
                if not instances.empty:
                    st.line_chart(instances, height=300)
                else:
                    st.info("No instance history to display.")
else:
    st.info("Charts appear after the simulation records observations. Press 'Advance 1 hour'.")

# ---- scorecard: does prediction help?
st.subheader("Scorecard: reactive rules vs DROID")
cmp_path = droid.MODELS / "comparison.json"
if cmp_path.exists():
    try:
        comparison = json.loads(cmp_path.read_text(encoding="utf-8"))
        if not comparison:
            st.warning("comparison.json exists but contains no results.")
        else:
            table = pd.DataFrame(comparison).T
            expected = [
                "Overloaded hours", "Instance-hours (cost)",
                "Scaling actions", "Cost vs static (%)"
            ]
            if table.shape[1] == len(expected):
                table.columns = expected
            st.dataframe(table, use_container_width=True)
            st.markdown("**Scorecard charts**")
            numeric = table.copy()
            for column in numeric.columns:
                numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
            ca, cb = st.columns(2)
            with ca:
                st.markdown("**Instance-hours (lower is cheaper)**")
                if "Instance-hours (cost)" in numeric.columns:
                    values = numeric[["Instance-hours (cost)"]].dropna()
                    if not values.empty:
                        st.bar_chart(values, height=300)
                    else:
                        st.info("No instance-hour values available.")
                else:
                    st.warning("Instance-hours column not found in comparison.json.")
            with cb:
                st.markdown("**Overloaded hours (lower is safer)**")
                if "Overloaded hours" in numeric.columns:
                    values = numeric[["Overloaded hours"]].dropna()
                    if not values.empty:
                        st.bar_chart(values, height=300)
                    else:
                        st.info("No overload values available.")
                else:
                    st.warning("Overloaded-hours column not found in comparison.json.")
            if "Scaling actions" in numeric.columns:
                st.markdown("**Scaling actions**")
                values = numeric[["Scaling actions"]].dropna()
                if not values.empty:
                    st.bar_chart(values, height=260)
            st.caption(
                "Same held-out hours, same simulated capacity rules, all VMs. "
                "DROID assumes every recommendation is approved. Capacity changes are simulated."
            )
    except (OSError, json.JSONDecodeError, ValueError, TypeError) as exc:
        st.error(f"Could not load scorecard from {cmp_path}: {exc}")
else:
    st.warning(f"Scorecard file not found: {cmp_path}")
=======
cols = st.columns(len(sim.w))
for col, (name, w) in zip(cols, sim.w.items()):
    last = [o for o in sim.obs if o["workload"] == name]
    col.metric(name, f"{w.n} instances", f"{last[-1]['util']:.0f}% CPU" if last else None, delta_color="off")

if sim.obs:
    obs = pd.DataFrame(sim.obs)
    a, b = st.columns(2)
    a.markdown("**CPU utilization (%)**")
    a.line_chart(obs.pivot(index="t", columns="workload", values="util"))
    b.markdown("**Instances**")
    b.line_chart(obs.pivot(index="t", columns="workload", values="instances"))

# ---- the scorecard: does prediction help?
st.subheader("Scorecard: reactive rules vs DROID")
cmp_path = droid.MODELS / "comparison.json"
if cmp_path.exists():
    table = pd.DataFrame(json.loads(cmp_path.read_text())).T
    table.columns = ["Overloaded hours", "Instance-hours (cost)", "Scaling actions", "Cost vs static (%)"]
    st.dataframe(table)
    st.caption("Same held-out hours, same simulated capacity rules, all VMs. DROID row assumes every "
               "recommendation is approved. Capacity changes are simulated.")
>>>>>>> 77c54d848864e79b54eb776101b2e551df56761b

# ---- history + audit
st.subheader("Decision history")
if sim.history:
    hist = pd.DataFrame(sim.history)
    hist = hist[hist["status"] != "NO_ACTION"]
    st.dataframe(hist[["t", "workload", "action", "status", "confidence", "reason"]].iloc[::-1],
<<<<<<< HEAD
                 use_container_width=True, hide_index=True)
=======
                 hide_index=True)
>>>>>>> 77c54d848864e79b54eb776101b2e551df56761b

st.subheader("Audit trail (last 10)")
log = droid.LOGS / "audit.jsonl"
if log.exists():
<<<<<<< HEAD
    audit_rows = []
    for line in log.read_text(encoding="utf-8").splitlines()[-10:][::-1]:
        try:
            event = json.loads(line)
            payload = event.get("payload", {})
            audit_rows.append({
                "time": event.get("timestamp", ""),
                "event": event.get("event", ""),
                "VM": payload.get("workload", ""),
                "hour": payload.get("t", ""),
                "action": payload.get("action", ""),
                "status": payload.get("status", ""),
                "reason": payload.get("reason", ""),
            })
        except (json.JSONDecodeError, TypeError):
            continue
    if audit_rows:
        st.dataframe(pd.DataFrame(audit_rows), use_container_width=True, hide_index=True)
    else:
        st.caption("No readable audit events yet. Advance the replay to create events.")
else:
    st.caption("No audit events yet. Advance the replay to create events.")
=======
    for line in log.read_text(encoding="utf-8").splitlines()[-10:][::-1]:
        st.json(json.loads(line), expanded=False)
>>>>>>> 77c54d848864e79b54eb776101b2e551df56761b
