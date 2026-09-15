"""
app.py
======
CDDSM — Streamlit dashboard.

Wires simulation_core.py (ODE + coupled PDE) and pinn_model.py (fast
spatial-field approximation) into a single interface for Dr. S: enter
Tier 1 assay data, get a RANGE of estimates (ODE-only vs PDE-coupled),
never a single false-precision number.

PINN usage note: the PINN is used for INSTANT spatial-field
visualization, not for the headline Cmax/Tmax/AUC numbers -- those
always come from the validated ODE/PDE solvers. The PINN fills in the
gut-wall concentration profile between boundary values taken from a
fast ODE-only solve, avoiding a live ~3min coupled-PDE training run
inside the interactive session.
"""

import os
import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go

from simulation_core import (
    Tier1AssayInputs, Tier3PhysiologicalConstants,
    get_all_estimates, run_ode_simulation, compute_tier2_parameters,
)
from pinn_model import load_model, hard_constrained_output, Normalizer

st.set_page_config(page_title="CDDSM", page_icon="💊", layout="wide")

# =====================================================================
# DARK THEME
# =====================================================================
st.markdown("""
<style>
    .stApp { background-color: #0e0e17; color: #e8e8f0; }
    section[data-testid="stSidebar"] { background-color: #14141f; }
    h1, h2, h3 { color: #f0f0fa; }
    .ring-wrap { display: flex; flex-direction: column; align-items: center; padding: 8px; }
    .ring {
        width: 110px; height: 110px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        margin-bottom: 8px;
    }
    .ring-inner {
        width: 86px; height: 86px; border-radius: 50%; background: #14141f;
        display: flex; flex-direction: column; align-items: center; justify-content: center;
    }
    .ring-value { font-size: 1.05rem; font-weight: 700; color: #f0f0fa; }
    .ring-unit { font-size: 0.65rem; color: #9a9ab0; }
    .ring-label { font-size: 0.85rem; color: #c8c8dc; font-weight: 600; }
    .flag-box {
        background-color: #1c1c2e; border-left: 3px solid #ff6b6b;
        padding: 10px 14px; border-radius: 4px; font-size: 0.85rem; color: #d8d8ea;
    }
</style>
""", unsafe_allow_html=True)


def ring_gauge(label, value, max_value, unit, color, fmt="{:.2f}"):
    pct = 0 if not max_value else max(0, min(100, 100 * value / max_value))
    html = f"""
    <div class="ring-wrap">
      <div class="ring" style="background: conic-gradient({color} {pct}%, #2a2a3d {pct}% 100%);">
        <div class="ring-inner">
          <div class="ring-value">{fmt.format(value)}</div>
          <div class="ring-unit">{unit}</div>
        </div>
      </div>
      <div class="ring-label">{label}</div>
    </div>
    """
    return html


# =====================================================================
# CACHED PINN LOAD — never retrain live inside the app
# =====================================================================
@st.cache_resource
def get_pinn():
    if os.path.exists("pinn_weights.pt"):
        return load_model("pinn_weights.pt")
    return None, None


pinn_model, pinn_normalizer = get_pinn()

# =====================================================================
# SIDEBAR — Tier 1 assay inputs ONLY. No derived parameter is ever a
# user slider (F, ka, ke, Fh are computed, never entered directly).
# =====================================================================
st.sidebar.title("💊 CDDSM")
st.sidebar.caption("Compartmental Drug Delivery Simulation Model")
st.sidebar.markdown("---")
st.sidebar.subheader("Tier 1 — In Vitro Assay Inputs")

drug_status = st.sidebar.selectbox("Drug status", ["known", "new"])
smiles = st.sidebar.text_input("SMILES", value="CC(=O)NC1=CC=C(O)C=C1",
                                help="Used for acid/base/neutral classification")
dose_mg = st.sidebar.number_input("Dose (mg)", min_value=1.0, value=500.0, step=50.0)
dissolution_rate = st.sidebar.number_input("Dissolution rate (mg/min/cm²)",
                                            min_value=0.01, value=2.5, step=0.1)
papp = st.sidebar.number_input("Papp / human Peff (×10⁻³ cm/s)", min_value=0.001,
                                value=2.0, step=0.1) * 1e-3
fu = st.sidebar.slider("Fraction unbound (fu)", 0.01, 1.0, 0.9)
logp = st.sidebar.number_input("logP", value=0.46, step=0.1)
clint = st.sidebar.number_input("Microsomal CLint (µL/min/mg)", min_value=0.1, value=8.0, step=0.5)
pka = st.sidebar.number_input("pKa", value=9.4, step=0.1)

st.sidebar.markdown("---")
scenario = st.sidebar.selectbox("Scenario", ["reversible", "one_way_sink", "dual_elimination"],
                                 help="Blood↔Tissue distribution/elimination topology")
t_end_h = st.sidebar.slider("Simulation window (h)", 2.0, 48.0, 24.0)
run_clicked = st.sidebar.button("▶ Run Simulation", use_container_width=True)

if not pinn_model:
    st.sidebar.markdown(
        '<div class="flag-box">No pinn_weights.pt found — spatial PINN view disabled. '
        'Run pinn_model.py once to generate it.</div>', unsafe_allow_html=True)

# =====================================================================
# MAIN — run + cache in session state, so plots survive reruns that
# aren't triggered by the button (e.g. widget interaction elsewhere)
# =====================================================================
st.title("CDDSM Dashboard")
st.caption("A range of estimates, not a single authoritative number — for pre-clinical decision support.")

if run_clicked:
    t1 = Tier1AssayInputs(
        smiles=smiles or None, dose_mg=dose_mg, drug_status=drug_status,
        dissolution_rate_mg_min_cm2=dissolution_rate, papp_cm_s=papp,
        fu=fu, logp=logp, clint_ul_min_mg=clint, pka=pka,
    )
    t3 = Tier3PhysiologicalConstants()
    with st.spinner("Solving ODE + coupled PDE..."):
        result = get_all_estimates(t1, t3, scenario=scenario, t_end_h=t_end_h)
    st.session_state["result"] = result
    st.session_state["t1"] = t1
    st.session_state["t3"] = t3
    st.session_state["scenario"] = scenario

if "result" not in st.session_state:
    st.info("Enter Tier 1 assay data in the sidebar and click **Run Simulation**.")
    st.stop()

result = st.session_state["result"]
t1 = st.session_state["t1"]
t3 = st.session_state["t3"]
scenario = st.session_state["scenario"]
params = result["tier2_parameters"]
ode_trace = result["ode"]["trace"]
ode_m = result["ode"]["metrics"]
pde_m = result["pde"]["metrics"]

# --- flag r_outer if it hasn't been overridden from the unsourced default ---
if abs(t3.r_outer_cm - 0.06) < 1e-9:
    st.markdown(
        '<div class="flag-box">⚠️ Gut wall thickness (r_outer) is an unsourced placeholder '
        '(~600 µm). PDE-coupled results below should be read as directional, not literature-locked.</div>',
        unsafe_allow_html=True)
    st.write("")

# =====================================================================
# RING GAUGE METRICS
# =====================================================================
st.subheader("Key Estimates (ODE layer)")
cols = st.columns(5)
with cols[0]:
    st.markdown(ring_gauge("Cmax", ode_m["Cmax_mg_L"], max(ode_m["Cmax_mg_L"] * 1.3, 1),
                            "mg/L", "#4ecdc4"), unsafe_allow_html=True)
with cols[1]:
    st.markdown(ring_gauge("Tmax", ode_m["Tmax_h"], max(t_end_h, 1),
                            "h", "#ffd166"), unsafe_allow_html=True)
with cols[2]:
    st.markdown(ring_gauge("AUC", ode_m["AUC_mg_h_L"], max(ode_m["AUC_mg_h_L"] * 1.3, 1),
                            "mg·h/L", "#ff6b6b"), unsafe_allow_html=True)
with cols[3]:
    st.markdown(ring_gauge("F (bioavail.)", params.F * 100, 100,
                            "%", "#a78bfa", fmt="{:.1f}"), unsafe_allow_html=True)
with cols[4]:
    st.markdown(ring_gauge("Fh (hepatic)", params.Fh * 100, 100,
                            "%", "#60a5fa", fmt="{:.1f}"), unsafe_allow_html=True)

st.caption(f"PDE-coupled comparison — Cmax {pde_m['Cmax_mg_L']:.3f} mg/L, "
           f"Tmax {pde_m['Tmax_h']:.2f} h, AUC {pde_m['AUC_mg_h_L']:.3f} mg·h/L. "
           f"Drug classified as **{params.drug_type}** from SMILES.")

# =====================================================================
# 5-GRAPH VISUALIZATION LAYER
# 4 individual (dissolution, absorption, distribution, elimination)
# + 1 merged view, per the established app requirement.
# =====================================================================
DARK_LAYOUT = dict(
    template="plotly_dark", paper_bgcolor="#0e0e17", plot_bgcolor="#14141f",
    font=dict(color="#e8e8f0"), margin=dict(l=40, r=20, t=40, b=40),
)
t = ode_trace["t_h"]

g1, g2 = st.columns(2)
with g1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ode_trace["solid_mg"], name="Solid", line=dict(color="#ff6b6b", width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=ode_trace["dissolved_mg"], name="Dissolved", line=dict(color="#4ecdc4", width=2.5)))
    fig.update_layout(title="1. Dissolution", xaxis_title="Time (h)", yaxis_title="Mass (mg)", **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with g2:
    absorption_rate_mg_h = params.ka_per_h * ode_trace["dissolved_mg"] * params.F
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=absorption_rate_mg_h, name="Absorption rate",
                              line=dict(color="#ffd166", width=2.5), fill="tozeroy"))
    fig.update_layout(title="2. Absorption (rate into blood)", xaxis_title="Time (h)",
                       yaxis_title="mg/h", **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

g3, g4 = st.columns(2)
with g3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ode_trace["blood_conc_mg_L"] * params.V1_L, name="Blood",
                              line=dict(color="#60a5fa", width=2.5)))
    fig.add_trace(go.Scatter(x=t, y=ode_trace["tissue_mg"], name="Tissue", line=dict(color="#a78bfa", width=2.5)))
    fig.update_layout(title="3. Distribution (Blood ↔ Tissue)", xaxis_title="Time (h)",
                       yaxis_title="Mass (mg)", **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with g4:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=ode_trace["eliminated_mg"], name="Eliminated",
                              line=dict(color="#f87171", width=2.5), fill="tozeroy"))
    fig.update_layout(title="4. Elimination", xaxis_title="Time (h)", yaxis_title="Mass (mg)", **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

st.markdown("#### 5. Merged — all compartments")
fig = go.Figure()
fig.add_trace(go.Scatter(x=t, y=ode_trace["solid_mg"], name="Solid", line=dict(color="#ff6b6b")))
fig.add_trace(go.Scatter(x=t, y=ode_trace["dissolved_mg"], name="Dissolved", line=dict(color="#4ecdc4")))
fig.add_trace(go.Scatter(x=t, y=ode_trace["blood_conc_mg_L"] * params.V1_L, name="Blood", line=dict(color="#60a5fa")))
fig.add_trace(go.Scatter(x=t, y=ode_trace["tissue_mg"], name="Tissue", line=dict(color="#a78bfa")))
fig.add_trace(go.Scatter(x=t, y=ode_trace["eliminated_mg"], name="Eliminated", line=dict(color="#f87171")))
fig.update_layout(title="All compartments over time", xaxis_title="Time (h)", yaxis_title="Mass (mg)",
                   height=420, **DARK_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

# =====================================================================
# PINN SPATIAL FIELD — the C(r,t) view the ODE layer cannot produce.
# Boundary trajectories come from the FAST ode-only solve (not the
# expensive coupled PDE) -- the PINN fills in the interior instantly.
# =====================================================================
st.markdown("---")
st.subheader("Gut-wall spatial concentration field")

if pinn_model is None:
    st.info("PINN weights not found — run `python pinn_model.py` once to enable this view.")
else:
    D_physical = t3.diffusion_coefficient_cm2_s
    peff_physical = t1.papp_cm_s

    out_of_range = not (pinn_normalizer.D_lo <= D_physical <= pinn_normalizer.D_hi and
                         pinn_normalizer.peff_lo <= peff_physical <= pinn_normalizer.peff_hi)
    if out_of_range:
        st.markdown(
            '<div class="flag-box">⚠️ This drug\'s D/Peff falls outside the PINN\'s trained range — '
            'the field below is an extrapolation, not a validated interpolation. Treat as illustrative only.</div>',
            unsafe_allow_html=True)

    V_lumen_cm3 = np.pi * t3.r_intestine_inner_cm ** 2 * t3.intestine_length_cm
    C_lumen_t = ode_trace["dissolved_mg"] / V_lumen_cm3
    C_blood_t = (ode_trace["blood_conc_mg_L"]) / 1000.0  # mg/L -> mg/cm^3 (=mg/mL)

    n_r_viz, n_t_viz = 30, 60
    r_grid = np.linspace(pinn_normalizer.r_inner, pinn_normalizer.r_outer, n_r_viz)
    t_idx = np.linspace(0, len(t) - 1, n_t_viz).astype(int)
    t_grid = t[t_idx]

    R_norm = np.repeat(pinn_normalizer.norm_r(r_grid), n_t_viz)
    T_norm = np.tile(pinn_normalizer.norm_t(t_grid), n_r_viz)
    D_norm_val = float(np.clip(pinn_normalizer.norm_D(D_physical), 0, 1))
    P_norm_val = float(np.clip(pinn_normalizer.norm_peff(peff_physical), 0, 1))
    D_norm = np.full_like(R_norm, D_norm_val)
    P_norm = np.full_like(R_norm, P_norm_val)
    c_in_vals = np.tile(C_lumen_t[t_idx], n_r_viz)
    c_out_vals = np.tile(C_blood_t[t_idx], n_r_viz)

    with torch.no_grad():
        def col(v):
            return torch.tensor(v, dtype=torch.float32).view(-1, 1)
        C_pred = hard_constrained_output(
            pinn_model, col(R_norm), col(T_norm), col(D_norm), col(P_norm),
            col(c_in_vals), col(c_out_vals),
        ).numpy().reshape(n_r_viz, n_t_viz)
        C_pred = np.clip(C_pred, 0.0, None)  # concentration cannot be negative;
                                              # guards a small numerical artifact seen in testing

    fig = go.Figure(data=go.Heatmap(
        z=C_pred, x=t_grid, y=r_grid, colorscale="Turbo",
        colorbar=dict(title="mg/mL"),
    ))
    fig.update_layout(title="Concentration across gut wall depth (PINN, instant inference)",
                       xaxis_title="Time (h)", yaxis_title="Radial position (cm)", **DARK_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("PINN trained on a parametric D/Peff sweep — ~2-4% mean relative error against the "
               "full coupled solver on held-out drugs (see project validation notes). Used here for "
               "instant spatial visualization only; headline Cmax/Tmax/AUC above always come from the "
               "validated ODE/PDE solvers, never the PINN.")
