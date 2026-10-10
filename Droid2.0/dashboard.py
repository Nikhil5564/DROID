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

if meta["source"].startswith("Azure"):
    st.success(f"Data: {meta['source']} - {meta['vms']} VMs, {meta['hours']} hours")
else:
    st.warning(f"Data: {meta['source']}. Do not present these numbers as real results.")

# ---- how good is the forecast? (measured on hours the model never saw)
st.subheader("Forecast quality (held-out hours)")
c1, c2, c3, c4 = st.columns(4)
c1.metric("DROID error (MAE)", f"{meta['model_mae']:.2f} pts")
c2.metric("'Same as now' error", f"{meta['persistence_mae']:.2f} pts")
c3.metric("Recent 3-hour mean error", f"{meta['yesterday_mae']:.2f} pts")
c4.metric("80% band really covers", f"{meta['band_80_coverage']:.0%}")
st.caption(f"Lower error is better. This lightweight prototype uses a {meta.get('history_hours', 3)}-hour history to forecast {meta.get('horizon_hours', 1)} hour ahead. Results from a short trace are experimental, not production validation.")

st.divider()

# ---- controls
b1, b2, b3 = st.columns([1, 1, 6])
b1.button("Advance 1 hour ▶", on_click=sim.advance, type="primary")
b2.button("Reset", on_click=reset)
b3.markdown(f"**Replay hour:** {sim.now}")

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

# ---- history + audit
st.subheader("Decision history")
if sim.history:
    hist = pd.DataFrame(sim.history)
    hist = hist[hist["status"] != "NO_ACTION"]
    st.dataframe(hist[["t", "workload", "action", "status", "confidence", "reason"]].iloc[::-1],
                 hide_index=True)

st.subheader("Audit trail (last 10)")
log = droid.LOGS / "audit.jsonl"
if log.exists():
    for line in log.read_text(encoding="utf-8").splitlines()[-10:][::-1]:
        st.json(json.loads(line), expanded=False)
