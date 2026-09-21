# CDDSM — Compartmental Drug Delivery Simulation Model

An open-source, physics-based pharmacokinetic (PK) simulation tool for oral drug delivery. Given a compound's raw in vitro assay data, CDDSM estimates the blood concentration–time curve (Cmax, Tmax, AUC) that compound would likely produce in a human — **before** any animal trial — using validated physics equations rather than a black-box ML model.

CDDSM is a screening / pre-clinical decision-support tool, not a diagnostic or clinical one. It answers *"how much drug reaches the bloodstream, how fast, and for how long"* — not whether that exposure is safe or effective. See [Scope](#scope-what-this-is-and-isnt) below.

---

## Why this exists

A DMPK (Drug Metabolism and Pharmacokinetics) scientist typically has a panel of in vitro assay results for a candidate compound — dissolution rate, Caco-2/PAMPA permeability, plasma protein binding, microsomal intrinsic clearance, pKa — long before any animal or human study. Each assay measures one property in isolation. CDDSM's job is to combine them into a single, dynamic, time-resolved answer: what would this compound's actual exposure profile look like in a person, given how all of these properties interact simultaneously.

Enterprise tools that do this (GastroPlus, Simcyp) exist, but are commercial, closed-source, and often black boxes. CDDSM is a transparent, from-scratch implementation of the same class of physics, open for inspection, modification, and — importantly — honest about exactly where it's strong and where it isn't.

---

## Architecture

CDDSM uses a **three-tier parameter system** so that nothing a user enters is ever a free-floating guess:

| Tier | What it is | Examples |
|---|---|---|
| **Tier 1** | Raw in vitro assay inputs — the only things a user ever types in | Dissolution rate, Papp, fu, logP, CLint, pKa, SMILES |
| **Tier 2** | PK parameters *derived* from Tier 1 via named literature equations | ka, ke, F, Vd, Fh — never sliders |
| **Tier 3** | Fixed physiological constants (population-average anatomy, not drug-specific) | Intestinal radius, hepatic blood flow, central compartment volume |

### Two independent physics layers

- **ODE layer** (`simulation_core.py`) — a compartmental model: `Solid → Dissolved → Blood ↔ Tissue → Eliminated`, solved via `scipy.integrate.solve_ivp`. Fast, and the primary source of Cmax/Tmax/AUC.
- **PDE layer** (`simulation_core.py`) — cylindrical Fick's 2nd Law solved across the gut wall, giving a full concentration field C(r, t) — position *and* time — not just a single blood-concentration number. Coupled directly to the ODE system (shared boundary conditions), not run as two disconnected simulations.

Both layers are reported side by side by `get_all_estimates()` — CDDSM deliberately gives a **range**, not a single false-precision number.

### Key Tier 2 equations implemented

- **Noyes-Whitney** — dissolution rate constant
- **Amidon/Sinko relation** (`ka = 2·Peff / R`) — permeability → absorption rate constant
- **Well-stirred liver model** — CLint → hepatic clearance (CLh) and first-pass survival fraction (Fh), computed, not a static literature multiplier
- **Henderson-Hasselbalch** (pH-partition hypothesis) — implemented (`fraction_unionized`), currently **not wired into the active absorption calculation** by design; see [Known Limitations](#known-limitations)

### Three absorption/elimination scenarios

`reversible` (Blood ↔ Tissue, bidirectional), `one_way_sink` (Tissue as an irreversible sink), `dual_elimination` (elimination from both Blood and Tissue) — selectable per simulation.

### PINN (`pinn_model.py`)

A physics-informed neural network trained to approximate the PDE's spatial concentration field C(r, t), conditioned on (diffusion coefficient, Peff) so it generalizes across a family of drugs rather than one fixed case. Key design choices:

- **Hard-constrained Dirichlet boundaries** — architecturally guaranteed (not a soft penalty loss), via a multiplicative ansatz that forces boundary values to be exactly satisfied regardless of network output
- Ground truth generated from the *same* coupled ODE+PDE solver used in `simulation_core.py` — one source of physics, not two implementations that could drift apart
- **~2-4% mean relative error** on held-out synthetic drugs — used only for **instant spatial-field visualization** in the dashboard; it has **zero effect** on the headline Cmax/Tmax/AUC numbers, which always come directly from the validated ODE/PDE solve

### Dashboard (`app.py`)

Streamlit app: Tier-1-only input sidebar, drug presets (see below), ring-gauge metrics, 5-graph layout (dissolution, absorption, distribution, elimination, merged), and a PINN-powered gut-wall concentration heatmap.

---

## Validated against real drugs

Four drugs, deliberately chosen for pharmacological diversity rather than picking easy cases — weak acid vs. weak base, high vs. low permeability, light vs. heavy protein binding:

| Drug | Class | Permeability | Protein binding | Literature source |
|---|---|---|---|---|
| **Paracetamol** | Weak acid (pKa 9.4) | High | Light (fu 0.9) | Prescott 1980 |
| **Ibuprofen** | Weak acid (pKa 4.4) | High | Heavy (fu 0.01) | Comparative bioavailability study |
| **Atenolol** | Weak base (pKa 9.6) | **Low** (BCS III) | Light (fu 0.97) | Dahlgren et al. 2016 — direct human regional intestinal perfusion |
| **Diclofenac** | Weak acid (pKa 4.0) | High | Very heavy (fu 0.006) | FDA label (DailyMed) + Chen 2015 |

Current model accuracy (Cmax, ODE layer, vs. literature):

| Drug | Cmax error |
|---|---|
| Paracetamol | −12.7% |
| Ibuprofen | −92.4% |
| Atenolol | −71.0% |
| Diclofenac | −50.2% |

**This spread is disclosed deliberately, not hidden.** Paracetamol validates well. The other three — all substantially ionized at physiological gut pH — are under-predicted. This is a documented finding, not an oversight: see below.

---

## Known limitations

Stated plainly, because a model that hides its own weak points is less useful than one that doesn't:

- **pH-dependent ionization is implemented but not currently active.** A regional (duodenum → jejunum → ileum → colon) pH-aware absorption model was built and tested; it empirically made Paracetamol's validation *worse* while not meaningfully fixing the ionization-heavy drugs, so it was rolled back in favor of the simpler model above. The code (`fraction_unionized`, `regional_permeability_factor`) is kept in place, unused, as a documented re-entry point.
- **Gut wall thickness (`r_outer`) is an unsourced placeholder.** Two literature search attempts did not find a clean, citable value for the diffusion-relevant mucosal thickness; kept as an explicitly flagged sub-millimeter estimate rather than presented as validated.
- **Several Tier 1 defaults for the 4 reference drugs remain unverified**, specifically: dissolution rate and CLint for most drugs, and Papp for Ibuprofen/Diclofenac. Where real literature values were found and differ from the assumption, this is noted in-code (e.g. Atenolol's Papp was corrected from an initial guess after finding a direct human perfusion study).
- **Scope is PK only — no PD.** CDDSM predicts *exposure* (how much drug, how fast, for how long), not *effect* (whether that exposure is therapeutic or toxic). It cannot and does not determine a therapeutic window; that requires pharmacodynamic and toxicology data outside this tool's scope by design.
- **Not full PBPK.** Tissue distribution is lumped into a single peripheral compartment, not resolved organ-by-organ (liver, kidney, fat, etc. individually). This matches what a DMPK scientist's typical pre-clinical assay panel can actually parameterize — going further would mean fabricating organ-specific partition coefficients rather than deriving them.

---

## Scope: what this is and isn't

| CDDSM is | CDDSM is not |
|---|---|
| A PK (pharmacokinetics) simulator | A PD (pharmacodynamics) or safety/efficacy tool |
| A pre-clinical screening/triage aid | A replacement for animal or clinical studies |
| A simplified, lumped compartmental + PDE model | Full PBPK (organ-by-organ) |
| Built for a DMPK-type user with a standard ADME assay panel | A tool requiring data no such panel would ever produce |

---

## Installation

```bash
pip install numpy scipy rdkit torch streamlit plotly
```

## Usage

Run in this order — `app.py` depends on `pinn_weights.pt`, which `pinn_model.py` produces:

```bash
python simulation_core.py    # runs the 4-drug validation suite
python pinn_model.py         # trains the PINN, saves pinn_weights.pt (few minutes)
streamlit run app.py         # launches the dashboard at localhost:8501
```

## Project structure

```
CDDSM/
├── simulation_core.py   # Tier 1/2/3 parameters, ODE + coupled PDE physics engine, drug validation registry
├── pinn_model.py         # Physics-informed neural network for the spatial concentration field
├── app.py                 # Streamlit dashboard
└── pinn_weights.pt        # Generated by pinn_model.py — trained PINN weights
```

## Tech stack

Python · NumPy · SciPy (`solve_ivp`) · RDKit (SMILES structural classification) · PyTorch (PINN) · Streamlit · Plotly

## Author

Anwesha Sarkar



Check it out.
https://compartmental-drug-delivery-simulation-model-dfjncruzq8pbxjs3k.streamlit.app/
