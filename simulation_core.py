"""
simulation_core.py
===================
CDDSM (Compartmental Drug Delivery Simulation Model)
Physics engine: ODE compartmental layer + PDE spatial-diffusion layer.

ARCHITECTURE — three-tier parameter model
------------------------------------------
Tier 1  Raw in vitro assay inputs      -> what a DMPK scientist (Dr. S) has on the bench
Tier 2  Derived PK parameters          -> computed from Tier 1 via named literature equations
Tier 3  Fixed physiological constants  -> population-average anatomy, not drug-specific

Design principle: F, ka, ke, Fh are NEVER user sliders. They are computed
from Tier 1 assay data via Tier 2 equations. See compute_tier2_parameters().

Validation reference: Paracetamol (Prescott 1980; PMC4165439;
Fernandez-Lastra et al. 2003; Fredlund et al. 2017).
"""

from dataclasses import dataclass, field
from typing import Literal, Optional
import numpy as np
from scipy.integrate import solve_ivp

try:
    from rdkit import Chem
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


# =====================================================================
# TIER 1 — RAW ASSAY INPUTS
# =====================================================================

@dataclass
class Tier1AssayInputs:
    """Raw in vitro measurements available BEFORE a preclinical (animal)
    trial. This is the only tier a user should ever type values into.

    drug_status:
      "known" -> literature-anchor values may back-fill missing fields
      "new"   -> every field must come from Dr. S's own assays
    """
    smiles: Optional[str] = None
    dose_mg: float = 500.0
    drug_status: Literal["known", "new"] = "new"

    dissolution_rate_mg_min_cm2: Optional[float] = None   # intrinsic dissolution (USP app / rotating disk)
    solubility_ph: dict = field(default_factory=lambda: {"1.2": None, "4.5": None, "6.8": None})  # mg/mL
    papp_cm_s: Optional[float] = None                      # Caco-2 / PAMPA apparent permeability
    fu: Optional[float] = None                             # fraction unbound, plasma protein binding
    logp: Optional[float] = None
    clint_ul_min_mg: Optional[float] = None                # microsomal/hepatocyte intrinsic clearance
    pka: Optional[float] = None


# =====================================================================
# TIER 3 — FIXED PHYSIOLOGICAL CONSTANTS
# =====================================================================

@dataclass
class Tier3PhysiologicalConstants:
    """Population-average constants. None of these can be supplied by
    any in vitro assay -- they are anatomy, not chemistry."""

    r_intestine_inner_cm: float = 1.75       # Helander & Fandriks 2014, small intestine lumen radius
    intestine_length_cm: float = 280.0       # effective absorptive length, small intestine

    # r_outer_cm — OPEN ITEM, NOT FULLY SOURCED.
    # Literature gives conflicting length scales depending on which
    # structure is meant: plicae circulares are mm-scale, the
    # unstirred water layer is reported ~45 um for high-permeability
    # solutes (Lennernas 1997), while mucosa+submucosa histology is
    # generally sub-mm. Using a sub-mm placeholder here pending a
    # deliberate, cited decision -- treat as unvalidated.
    r_outer_cm: float = 0.06                 # ~600 um placeholder, NOT literature-locked

    gut_transit_time_h: float = 3.5
    Qh_L_per_h: float = 90.0                 # hepatic blood flow
    Qt_L_per_h: float = 300.0                # lumped peripheral tissue blood flow
    # V1 is the pharmacokinetic "central compartment" volume, NOT literal
    # blood/plasma volume (~5L) -- it represents blood plus rapidly
    # equilibrating tissue that clinical Cmax/AUC data is referenced
    # against. 20L approximates typical central-compartment population-PK
    # estimates. Kept architecturally separate from Vd (peripheral+central).
    V1_L: float = 20.0
    body_weight_kg: float = 70.0
    gut_ph_by_region: dict = field(
        default_factory=lambda: {"stomach": 1.5, "duodenum": 6.0, "jejunum": 6.8, "ileum": 7.4}
    )
    # GI transit time split across the three absorptive small-intestine
    # regions (duodenum/jejunum/ileum) -- illustrative split of the total
    # gut_transit_time_h weighted toward jejunum+ileum length, NOT
    # individually literature-sourced per region. Flagged like r_outer.
    gi_region_transit_h: dict = field(
        default_factory=lambda: {"duodenum": 0.75, "jejunum": 4.5, "ileum": 5.25}
    )
    # Relative share of total absorptive capacity per region (sums to 1.0)
    # -- jejunum weighted highest (largest absorptive surface area from
    # villi/microvilli density), duodenum lowest (short segment despite
    # high per-unit-area absorption). Illustrative, not individually
    # literature-sourced.
    gi_region_absorption_weight: dict = field(
        default_factory=lambda: {"duodenum": 0.10, "jejunum": 0.55, "ileum": 0.35}
    )
    diffusion_coefficient_cm2_s: float = 1e-6  # typical small-molecule aqueous/tissue diffusion coefficient
    # Real membrane-permeation studies show the IONIZED form of a drug
    # still crosses lipid bilayers, just 12-155x slower than the
    # unionized form (not zero, as the naive pH-partition hypothesis
    # assumes). CALIBRATED (not purely literature-derived) against 4
    # known drugs (Paracetamol/Ibuprofen/Diclofenac/Atenolol) toward the
    # more permissive end of that range -- disclosed explicitly as a
    # calibration choice, not an independently sourced number. Even so,
    # ionization-limited acids/bases (Ibuprofen, Diclofenac, Atenolol)
    # remain under-predicted -- a real, diagnosed model limitation, not
    # fully closed by this calibration. See KNOWN_DRUGS validation.
    ionized_permeability_ratio: float = 0.08


# =====================================================================
# STRUCTURAL CLASSIFICATION (SMILES -> acid / base / neutral)
# =====================================================================

_ACID_SMARTS = ["[CX3](=O)[OX2H1]", "[$([OX2H]-c)]", "[SX4](=O)(=O)[OX2H]"]
_BASE_SMARTS = ["[NX3;H2,H1,H0;!$(NC=O);!$(N=*)]", "[nX3;H1]"]


def classify_drug_type(smiles: str) -> Literal["acid", "base", "neutral", "unknown"]:
    """RDKit SMARTS-based acid/base/neutral classification.
    Falls back to 'unknown' if RDKit is unavailable or SMILES is invalid.
    Determines which Henderson-Hasselbalch form applies in Tier 2.
    """
    if not RDKIT_AVAILABLE or not smiles:
        return "unknown"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "unknown"

    is_acid = any(mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in _ACID_SMARTS)
    is_base = any(mol.HasSubstructMatch(Chem.MolFromSmarts(p)) for p in _BASE_SMARTS)

    if is_acid and not is_base:
        return "acid"
    if is_base and not is_acid:
        return "base"
    if is_acid and is_base:
        return "acid"  # ampholyte -- treat dominant ionizable group as acid by convention here
    return "neutral"


def fraction_unionized(pH: float, pka: float, drug_type: str) -> float:
    """Henderson-Hasselbalch fraction of drug in unionized (more
    permeable) form. Unionized form is what crosses membranes fastest.

    Weak ACID (HA <-> H+ + A-): pH = pKa + log([A-]/[HA])
      -> fraction unionized (HA) = 1 / (1 + 10^(pH - pKa))
    Weak BASE (BH+ <-> B + H+): pH = pKa + log([B]/[BH+])
      -> fraction unionized (B)  = 1 / (1 + 10^(pKa - pH))
    (These two branches were swapped in an earlier version of this
    function -- caught when first actually wired into compute_ka() and
    tested against Paracetamol, a weak acid with pKa >> gut pH, which
    should be ~99.75% unionized, not ~0.25%.)
    """
    if drug_type == "acid":
        return 1.0 / (1.0 + 10 ** (pH - pka))
    elif drug_type == "base":
        return 1.0 / (1.0 + 10 ** (pka - pH))
    return 1.0  # neutral drugs: fully "unionized" by definition


# =====================================================================
# TIER 2 — DERIVED PK PARAMETERS (Tier 1 -> named equations -> Tier 2)
# =====================================================================

def compute_k_dissolution(t1: Tier1AssayInputs, particle_surface_area_cm2: float = 15.0) -> float:
    """Noyes-Whitney: dM/dt = A * k * (Cs - C). Returns a first-order
    rate constant (1/h) approximating dissolution under sink conditions.
    particle_surface_area_cm2 default (15 cm^2) approximates a typical
    micronized immediate-release oral solid dose -- NOT drug-specific,
    flag alongside r_outer as an assumption to revisit per formulation."""
    if t1.dissolution_rate_mg_min_cm2 is None:
        return 0.5  # conservative literature-anchor fallback for "known" drugs
    rate_mg_min = t1.dissolution_rate_mg_min_cm2 * particle_surface_area_cm2
    k_per_min = rate_mg_min / max(t1.dose_mg, 1e-6)
    return k_per_min * 60.0  # -> 1/h


def compute_ka(t1: Tier1AssayInputs, t3: Tier3PhysiologicalConstants, drug_type: str = "unknown") -> float:
    """Amidon/Sinko relation: ka = 2 * Peff / R_intestine.
    Converts Caco-2/PAMPA Papp (human-scaled) directly into a first-order
    absorption rate constant. This is the SAME Peff that drives the PDE
    flux boundary condition -- the connector between the two model layers.

    Returns the BASE rate constant representing absorptive capacity
    across the whole intestine (pure Amidon-Sinko). pH-dependent
    ionization correction is NOT applied here right now -- deliberately
    deferred (see learnings-and-approach.md): a regional per-pH version
    was built and tested, but empirically made Paracetamol's validation
    worse without meaningfully fixing the ionization-heavy drugs, so it
    was rolled back in favor of this simpler version while that's
    revisited. fraction_unionized()/regional_permeability_factor() are
    kept below, unused, as the re-entry point for that work later."""
    if t1.papp_cm_s is None:
        return 1.0  # 1/h literature-anchor fallback
    p_eff = t1.papp_cm_s  # assume already human-scaled; a Caco-2->human
                           # regression could be inserted here if Papp is raw Caco-2
    ka_per_s = 2.0 * p_eff / t3.r_intestine_inner_cm
    return ka_per_s * 3600.0


def regional_permeability_factor(pH: float, pka: float, drug_type: str, t3: Tier3PhysiologicalConstants) -> float:
    """NOT CURRENTLY CALLED -- kept as the re-entry point for pH-aware
    absorption modeling later (see compute_ka's docstring for why it
    was rolled back). Blended permeability factor at a SPECIFIC gut
    region's pH: unionized fraction at full rate, ionized fraction at a
    reduced (not zero) rate via t3.ionized_permeability_ratio."""
    if pka is None or drug_type not in ("acid", "base"):
        return 1.0
    f_un = fraction_unionized(pH, pka, drug_type)
    f_ion = 1.0 - f_un
    return f_un + f_ion * t3.ionized_permeability_ratio


def compute_fu(t1: Tier1AssayInputs) -> float:
    return t1.fu if t1.fu is not None else 0.9  # fallback: assume mostly unbound


def compute_kp(t1: Tier1AssayInputs) -> float:
    """Simplified logP-based tissue:plasma partition coefficient.
    A full Rodgers-Rowland model would add ionization + tissue
    composition terms; this is the load-bearing simplification to
    revisit if Blood<->Tissue kinetics need more fidelity."""
    if t1.logp is None:
        return 1.0
    # NOTE: for small, hydrophilic drugs (low logP), Vd is often driven
    # by total-body-water distribution rather than lipid partitioning --
    # the previous coefficients (0.5, -1.0) suppressed Kp too aggressively
    # for exactly this case (validated against paracetamol, logP=0.46,
    # real Vd~0.9 L/kg). This is still a simplification of a full
    # Rodgers-Rowland tissue-composition model, flagged for revisit.
    return float(np.clip(10 ** (0.3 * t1.logp + 0.1), 0.1, 20.0))


def compute_hepatic_clearance(t1: Tier1AssayInputs, t3: Tier3PhysiologicalConstants):
    """Well-stirred liver model: CLh = Qh * fu * CLint / (Qh + fu * CLint).
    Fh = 1 - CLh / Qh.
    This REPLACES a literature-multiplier Fh with a computed value --
    the concrete upgrade identified during architecture review."""
    fu = compute_fu(t1)
    if t1.clint_ul_min_mg is None:
        return 0.5 * t3.Qh_L_per_h, 0.5  # fallback CLh, Fh if no CLint assay available

    # scale microsomal CLint (uL/min/mg protein) to whole-liver clearance (L/h).
    # Standard scaling factors: ~40 mg microsomal protein/g liver, ~1500-1800 g liver (70 kg adult).
    mg_protein_per_g_liver = 40.0
    liver_mass_g = 1500.0 + (t3.body_weight_kg - 70.0) * 10.0
    clint_scaled_ul_min = t1.clint_ul_min_mg * mg_protein_per_g_liver * liver_mass_g
    clint_L_h = clint_scaled_ul_min * 60.0 / 1e6  # uL/min -> L/h

    CLh = (t3.Qh_L_per_h * fu * clint_L_h) / (t3.Qh_L_per_h + fu * clint_L_h)
    Fh = 1.0 - (CLh / t3.Qh_L_per_h)
    return CLh, float(np.clip(Fh, 0.01, 1.0))


def compute_vd(t1: Tier1AssayInputs, t3: Tier3PhysiologicalConstants) -> float:
    """Total apparent volume of distribution, corrected for fu and Kp.
    Kept explicitly separate from V1 (central compartment volume) --
    this is the fix for the historical V1/Vd double-counting bug."""
    fu = compute_fu(t1)
    kp = compute_kp(t1)
    v_tissue_L = 40.0  # approx total body water outside plasma, 70kg adult
    return t3.V1_L + kp * fu * v_tissue_L


@dataclass
class Tier2DerivedParameters:
    drug_type: str
    ka_per_h: float
    k_dissolution_per_h: float
    ke_per_h: float
    kp: float
    fu: float
    CLh_L_h: float
    Fh: float
    Fa: float
    Fg: float
    F: float
    Vd_L: float
    V1_L: float
    p_eff_cm_s: float


def compute_tier2_parameters(t1: Tier1AssayInputs, t3: Tier3PhysiologicalConstants) -> Tier2DerivedParameters:
    """Single entry point: Tier 1 assay data + Tier 3 constants -> every
    Tier 2 parameter the ODE/PDE layers need. Nothing here is a user slider."""
    drug_type = classify_drug_type(t1.smiles) if t1.smiles else "unknown"

    ka = compute_ka(t1, t3, drug_type=drug_type)
    k_diss = compute_k_dissolution(t1)
    fu = compute_fu(t1)
    kp = compute_kp(t1)
    CLh, Fh = compute_hepatic_clearance(t1, t3)
    Vd = compute_vd(t1, t3)
    # ke = CLh / V1, NOT CLh / Vd. Elimination physically occurs from the
    # blood (central) compartment, whose volume is V1 -- Vd is the total
    # apparent distribution volume including peripheral tissue and is
    # used for k12/k21 (Kp) partitioning only. Conflating the two here
    # is the exact V1/Vd class of bug already fixed once elsewhere in
    # this project; caught during Paracetamol recalibration.
    ke = CLh / t3.V1_L if t3.V1_L > 0 else 0.1

    # Fa (fraction absorbed), back to a separate empirical multiplier --
    # the regional GI-transit model (where Fa emerged from mass balance)
    # was rolled back (see compute_ka's docstring), so F needs Fa applied
    # explicitly again, same as the original single-compartment design.
    p_eff = t1.papp_cm_s if t1.papp_cm_s is not None else 5e-5
    Fa = float(np.clip(1.0 - np.exp(-p_eff / 1e-4), 0.05, 0.99))

    Fg = 0.9  # gut-wall (CYP3A4/enterocyte) first-pass -- literature-anchor
              # placeholder; upgrade path is an intestinal Clint assay, which
              # is rarely available pre-clinical, so this stays Tier-3-like.

    F = Fa * Fg * Fh

    return Tier2DerivedParameters(
        drug_type=drug_type, ka_per_h=ka, k_dissolution_per_h=k_diss,
        ke_per_h=ke, kp=kp, fu=fu, CLh_L_h=CLh, Fh=Fh, Fa=Fa, Fg=Fg, F=F,
        Vd_L=Vd, V1_L=t3.V1_L, p_eff_cm_s=p_eff,
    )


# =====================================================================
# ODE LAYER — 4-compartment chain
# Solid -> Dissolved -> Blood <-> Tissue -> Eliminated
# =====================================================================

Scenario = Literal["reversible", "one_way_sink", "dual_elimination"]


def ode_system(t, y, params: Tier2DerivedParameters, dose_mg: float, scenario: Scenario):
    """Simple single-compartment absorption model: Solid -> Dissolved ->
    Blood <-> Tissue -> Eliminated. Regional GI-transit + per-pH
    absorption was built and tested (see compute_ka docstring for why
    it was rolled back) -- this is the deliberately simpler version,
    kept as the active default while that's revisited."""
    solid, dissolved, blood, tissue, eliminated = y

    k_diss = params.k_dissolution_per_h
    ka = params.ka_per_h
    ke = params.ke_per_h
    kp = params.kp

    # distribution rate constants derived from Kp (k12 out, k21 back in)
    k12 = 0.5 * kp
    k21 = 0.5

    d_solid = -k_diss * solid
    d_dissolved = k_diss * solid - ka * dissolved

    if scenario == "reversible":
        d_blood = (params.F * ka * dissolved) - k12 * blood + k21 * tissue - ke * blood
        d_tissue = k12 * blood - k21 * tissue
        d_eliminated = ke * blood

    elif scenario == "one_way_sink":
        # tissue acts as an irreversible sink -- no return flow (k21 = 0)
        d_blood = (params.F * ka * dissolved) - k12 * blood - ke * blood
        d_tissue = k12 * blood
        d_eliminated = ke * blood

    elif scenario == "dual_elimination":
        # elimination occurs from BOTH blood (renal/hepatic) and tissue
        # (e.g. local tissue metabolism) -- adds a second elimination path
        ke_tissue = 0.3 * ke
        d_blood = (params.F * ka * dissolved) - k12 * blood + k21 * tissue - ke * blood
        d_tissue = k12 * blood - k21 * tissue - ke_tissue * tissue
        d_eliminated = ke * blood + ke_tissue * tissue

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    return [d_solid, d_dissolved, d_blood, d_tissue, d_eliminated]


def run_ode_simulation(t1: Tier1AssayInputs, t3: Tier3PhysiologicalConstants,
                        scenario: Scenario = "reversible", t_end_h: float = 24.0,
                        n_points: int = 500):
    params = compute_tier2_parameters(t1, t3)
    y0 = [t1.dose_mg, 0.0, 0.0, 0.0, 0.0]
    t_eval = np.linspace(0, t_end_h, n_points)

    sol = solve_ivp(
        ode_system, [0, t_end_h], y0, t_eval=t_eval,
        args=(params, t1.dose_mg, scenario), method="RK45", rtol=1e-6, atol=1e-9,
    )

    conc_blood_mg_L = sol.y[2] / params.V1_L
    return {
        "t_h": sol.t,
        "solid_mg": sol.y[0], "dissolved_mg": sol.y[1],
        "blood_conc_mg_L": conc_blood_mg_L,
        "tissue_mg": sol.y[3], "eliminated_mg": sol.y[4],
        "params": params,
    }


# =====================================================================
# PDE-COUPLED LAYER — a SINGLE ODE+PDE system, not two disconnected runs.
#
# The gut wall is discretized radially: dC/dt = D*(d2C/dr2 + (1/r)dC/dr).
# Its boundaries are not fixed constants -- they are tied to the live
# state of the two things they physically touch:
#   C(r_inner, t) = dissolved_mass(t) / V_lumen   (lumen side, Dirichlet)
#   C(r_outer, t) = blood_mass(t) / V1            (blood side, Dirichlet)
# The diffusive flux at r_outer IS the physical absorption rate feeding
# the blood compartment -- replacing the ODE-only layer's ka*dissolved
# shortcut with an actual solved concentration gradient.
#
# Bioavailability note: Fa (fraction absorbed) is NOT applied here as a
# multiplier -- it emerges physically from the diffusion itself (a
# low-permeability drug produces a shallow gradient and low flux on its
# own). Only Fg (gut-wall metabolism) and Fh (hepatic first-pass) are
# applied to the physical flux before it reaches systemic blood, to
# avoid double-counting Fa.
# =====================================================================

def run_pde_coupled_simulation(t1: Tier1AssayInputs, t3: Tier3PhysiologicalConstants,
                                params: Tier2DerivedParameters, scenario: Scenario,
                                dose_mg: float, n_r: int = 25, t_end_h: float = 24.0,
                                n_points: int = 400):
    D = t3.diffusion_coefficient_cm2_s * 3600.0        # cm^2/s -> cm^2/h
    r_inner = t3.r_intestine_inner_cm
    r_outer = r_inner + t3.r_outer_cm
    r = np.linspace(r_inner, r_outer, n_r)
    dr = r[1] - r[0]

    V_lumen_cm3 = np.pi * r_inner ** 2 * t3.intestine_length_cm
    A_inner_cm2 = 2 * np.pi * r_inner * t3.intestine_length_cm
    A_outer_cm2 = 2 * np.pi * r_outer * t3.intestine_length_cm
    presystemic_survival = params.Fg * params.Fh   # Fa deliberately excluded, see note above

    n_interior = n_r - 2
    # state layout: [solid, dissolved_mass, C_interior(n_interior), blood_mass, tissue_mg, eliminated_mg]

    def rhs(t, y):
        solid, dissolved_mass = y[0], y[1]
        C_interior = y[2:2 + n_interior]
        blood_mass, tissue_mg, eliminated_mg = y[2 + n_interior:]

        C_lumen = dissolved_mass / V_lumen_cm3                    # mg/cm^3
        C_blood = (blood_mass / params.V1_L) / 1000.0             # mg/L -> mg/cm^3
        C_full = np.empty(n_r)
        C_full[0], C_full[-1] = C_lumen, C_blood
        C_full[1:-1] = C_interior

        dC_interior = np.zeros(n_interior)
        for idx in range(n_interior):
            i = idx + 1
            d2C = (C_full[i + 1] - 2 * C_full[i] + C_full[i - 1]) / dr ** 2
            dC = (C_full[i + 1] - C_full[i - 1]) / (2 * dr)
            dC_interior[idx] = D * (d2C + (1.0 / r[i]) * dC)

        J_in = -D * (C_full[1] - C_full[0]) / dr          # mg/(cm^2 h), lumen -> wall
        J_out = -D * (C_full[-1] - C_full[-2]) / dr        # mg/(cm^2 h), wall -> blood
        mass_into_wall = J_in * A_inner_cm2                # mg/h leaving the lumen
        mass_into_blood = max(J_out, 0.0) * A_outer_cm2 * presystemic_survival  # mg/h entering blood

        d_solid = -params.k_dissolution_per_h * solid
        d_dissolved = params.k_dissolution_per_h * solid - mass_into_wall

        k12 = 0.5 * params.kp
        k21 = 0.5
        ke = params.ke_per_h

        if scenario == "reversible":
            d_blood = mass_into_blood - k12 * blood_mass + k21 * tissue_mg - ke * blood_mass
            d_tissue = k12 * blood_mass - k21 * tissue_mg
            d_elim = ke * blood_mass
        elif scenario == "one_way_sink":
            d_blood = mass_into_blood - k12 * blood_mass - ke * blood_mass
            d_tissue = k12 * blood_mass
            d_elim = ke * blood_mass
        elif scenario == "dual_elimination":
            ke_tissue = 0.3 * ke
            d_blood = mass_into_blood - k12 * blood_mass + k21 * tissue_mg - ke * blood_mass
            d_tissue = k12 * blood_mass - k21 * tissue_mg - ke_tissue * tissue_mg
            d_elim = ke * blood_mass + ke_tissue * tissue_mg
        else:
            raise ValueError(f"Unknown scenario: {scenario}")

        return np.concatenate(([d_solid, d_dissolved], dC_interior, [d_blood, d_tissue, d_elim]))

    y0 = np.concatenate(([dose_mg, 0.0], np.zeros(n_interior), [0.0, 0.0, 0.0]))
    t_eval = np.linspace(0, t_end_h, n_points)
    sol = solve_ivp(rhs, [0, t_end_h], y0, t_eval=t_eval, method="BDF", rtol=1e-6, atol=1e-9)

    blood_mass = sol.y[2 + n_interior]
    pde_blood_conc_mg_L = blood_mass / params.V1_L

    # Reconstruct the FULL spatial field C(r, t), not just interior nodes.
    # mg/cm^3 and mg/mL are numerically identical (1 cm^3 = 1 mL), so no
    # conversion is needed -- this is the position x time concentration
    # field that the ODE-only layer cannot produce by construction.
    dissolved_mass_t = sol.y[1]
    blood_mass_t = sol.y[2 + n_interior]
    C_lumen_t = dissolved_mass_t / V_lumen_cm3
    C_blood_t = (blood_mass_t / params.V1_L) / 1000.0
    C_full_mg_mL = np.empty((n_r, len(sol.t)))
    C_full_mg_mL[0, :] = C_lumen_t
    C_full_mg_mL[-1, :] = C_blood_t
    C_full_mg_mL[1:-1, :] = sol.y[2:2 + n_interior]

    return {
        "t_h": sol.t, "r_cm": r,
        "solid_mg": sol.y[0], "dissolved_mg": sol.y[1],
        "C_wall_interior": sol.y[2:2 + n_interior],
        "C_full_mg_mL": C_full_mg_mL,          # shape (n_r, n_timepoints) -- position x time
        "pde_blood_conc_mg_L": pde_blood_conc_mg_L,
        "tissue_mg": sol.y[3 + n_interior], "eliminated_mg": sol.y[4 + n_interior],
    }


def get_spatial_concentration_field(t1: Tier1AssayInputs, t3: Tier3PhysiologicalConstants = None,
                                     scenario: Scenario = "reversible", t_end_h: float = 24.0,
                                     n_r: int = 25, n_points: int = 400) -> dict:
    """Clean entry point for the C(r, t) spatial field -- the thing the
    ODE-only layer cannot produce, and the thing pinn_model.py's finite
    -difference ground truth and app.py's spatial heatmap both need.

    Returns
    -------
    r_cm        : (n_r,)            radial position, lumen-side to blood-side
    t_h         : (n_points,)       simulation time
    C_mg_mL     : (n_r, n_points)   concentration at each (position, time)
    r_inner_cm / r_outer_cm         wall boundaries, for axis labeling
    blood_conc_mg_L : (n_points,)   same as C_mg_mL[-1,:], convenience alias
    """
    t3 = t3 or Tier3PhysiologicalConstants()
    params = compute_tier2_parameters(t1, t3)
    pde_result = run_pde_coupled_simulation(t1, t3, params, scenario, t1.dose_mg,
                                             n_r=n_r, t_end_h=t_end_h, n_points=n_points)
    return {
        "r_cm": pde_result["r_cm"],
        "t_h": pde_result["t_h"],
        "C_mg_mL": pde_result["C_full_mg_mL"],
        "r_inner_cm": t3.r_intestine_inner_cm,
        "r_outer_cm": t3.r_intestine_inner_cm + t3.r_outer_cm,
        "blood_conc_mg_L": pde_result["pde_blood_conc_mg_L"],
    }


# =====================================================================
# METRICS + COMBINED REPORTING
# =====================================================================

def compute_metrics(t_h: np.ndarray, conc_mg_L: np.ndarray) -> dict:
    cmax = float(np.max(conc_mg_L))
    tmax = float(t_h[np.argmax(conc_mg_L)])
    trapezoid_fn = getattr(np, "trapezoid", None) or np.trapz  # NumPy 2.x renamed trapz -> trapezoid
    auc = float(trapezoid_fn(conc_mg_L, t_h))
    return {"Cmax_mg_L": cmax, "Tmax_h": tmax, "AUC_mg_h_L": auc}


def get_all_estimates(t1: Tier1AssayInputs, t3: Tier3PhysiologicalConstants = None,
                       scenario: Scenario = "reversible", t_end_h: float = 24.0) -> dict:
    """Top-level entry point. Returns BOTH ODE-only and PDE-coupled
    results explicitly -- Dr. S sees a range to inform her preclinical
    decision, never a single false-precision number."""
    t3 = t3 or Tier3PhysiologicalConstants()

    ode_result = run_ode_simulation(t1, t3, scenario=scenario, t_end_h=t_end_h)
    params = ode_result["params"]
    pde_result = run_pde_coupled_simulation(t1, t3, params, scenario, t1.dose_mg, t_end_h=t_end_h)

    ode_metrics = compute_metrics(ode_result["t_h"], ode_result["blood_conc_mg_L"])
    pde_metrics = compute_metrics(pde_result["t_h"], pde_result["pde_blood_conc_mg_L"])

    return {
        "tier2_parameters": ode_result["params"],
        "ode": {"trace": ode_result, "metrics": ode_metrics},
        "pde": {"trace": pde_result, "metrics": pde_metrics},
        "scenario": scenario,
    }


# =====================================================================
# VALIDATION — Paracetamol reference case
# =====================================================================

PARACETAMOL_LITERATURE = {
    "Tmax_h": 0.75,     # Prescott 1980 / PMC4165439 consensus, oral solution/immediate-release
    "Cmax_mg_L": 10.0,  # approx literature value specifically for a 500mg oral dose
    # AUC_mg_h_L: FLAGGED FOR RE-VERIFICATION. Non-compartmental theory
    # (AUC = F*Dose/CL) with this model's own F and CLh predicts ~17
    # mg.h/L at 500mg -- close to the simulated 15.5. A target of 45
    # mg.h/L looks scaled for a ~1000mg dose rather than 500mg; confirm
    # against the source (Fernandez-Lastra 2003 / Fredlund 2017) before
    # trusting this number over the model.
    "AUC_mg_h_L": 45.0,
}


# =====================================================================
# KNOWN-DRUG VALIDATION REGISTRY
# 4 drugs, chosen to stress-test different parts of the model (weak
# acid vs weak base, high vs low permeability/BCS class, heavy vs light
# protein binding). Physicochemical values (pKa, logP, protein binding)
# are well-established textbook figures. Papp values are now
# literature-sourced per drug (see individual comments below) rather
# than guessed. Clinical Cmax/Tmax/AUC targets are sourced from primary
# studies where noted; flagged inline where a number is still approximate.
#
# KNOWN LIMITATIONS -- documented deliberately rather than hidden:
#
# 1. REGRESSION vs. the simpler single-region model: the regional
#    GI-transit rebuild below (duodenum->jejunum->ileum->colon, each
#    with its own pH and absorption rate) is architecturally more
#    correct -- Fa is now an EMERGENT result of mass balance (drug
#    reaching the colon unabsorbed is genuinely lost) rather than a
#    separately guessed multiplier, which fixed a real double-counting
#    bug. But empirically, it made Paracetamol's validation WORSE
#    (Cmax error -12.6% -> -39.9%) while not meaningfully fixing the
#    ionization-heavy drugs. Kept anyway per explicit decision to favor
#    the more physically honest architecture over the better-fitting
#    number, and to document this finding rather than chase it further.
#
# 2. PERSISTENT FAILURE on ionization-heavy drugs: Ibuprofen, Atenolol,
#    and Diclofenac all remain ~90%+ under-predicted on Cmax/AUC even
#    after: (a) correcting Papp to literature-sourced values, (b)
#    blending ionized/unionized permeability instead of a hard gate,
#    (c) calibrating region transit time and the ionized-permeability
#    ratio against all 4 drugs jointly. This points to a genuine,
#    documented limitation of first-order single-compartment-per-region
#    absorption models for strongly pH/ionization-limited drugs --
#    the same reason production tools (GastroPlus's ACAT model) use
#    7-9 sequential sub-compartments per region instead of 1. That
#    level of granularity was judged out of scope for this rebuild.
# =====================================================================

KNOWN_DRUGS = {
    "paracetamol": {  # weak acid, pKa >> gut pH -> near-fully unionized (mild test of pH-partition fix)
        "inputs": Tier1AssayInputs(
            smiles="CC(=O)NC1=CC=C(O)C=C1", dose_mg=500.0, drug_status="known",
            dissolution_rate_mg_min_cm2=2.5, papp_cm_s=2.0e-3, fu=0.9,
            logp=0.46, clint_ul_min_mg=8.0, pka=9.4,
        ),
        "literature": {"Tmax_h": 0.75, "Cmax_mg_L": 10.0, "AUC_mg_h_L": 45.0},
    },
    "ibuprofen": {  # weak acid, pKa ~4.4 -> substantially IONIZED at gut pH -> real test of the pH-partition fix
        "inputs": Tier1AssayInputs(
            smiles="CC(C)Cc1ccc(cc1)C(C)C(=O)O", dose_mg=400.0, drug_status="known",
            dissolution_rate_mg_min_cm2=0.5, papp_cm_s=1.5e-3, fu=0.01,  # ~99% protein bound
            logp=3.5, clint_ul_min_mg=15.0, pka=4.4,
        ),
        # SOURCED: single 400mg dose, mean+-SD Cmax=26.1+-4.7 ug/ml,
        # Tmax=1.98+-0.56h, AUC0-inf=107.8+-28.6 ug.h/ml (free-acid
        # formulation, healthy volunteers) -- academia.edu bioavailability study
        "literature": {"Tmax_h": 1.98, "Cmax_mg_L": 26.1, "AUC_mg_h_L": 107.8},
    },
    "atenolol": {  # weak base, pKa~9.6, LOW permeability (BCS III) -> tests the ka/Papp-limited regime
        "inputs": Tier1AssayInputs(
            smiles="CC(C)NCC(O)COC1=CC=C(CC(N)=O)C=C1", dose_mg=50.0, drug_status="known",
            # SOURCED: median human jejunal Peff = 0.45e-4 cm/s (Dahlgren
            # et al. 2016, PubMed 27504798 -- direct regional intestinal
            # perfusion in 14 healthy volunteers). Corrects an earlier
            # unsourced guess of 2.0e-5.
            dissolution_rate_mg_min_cm2=3.0, papp_cm_s=4.5e-5, fu=0.97,
            logp=0.16, clint_ul_min_mg=2.0, pka=9.6,
        ),
        # SOURCED: peak blood levels at 2-4h, oral F~50% (absorption is
        # FAST but INCOMPLETE -- not slow), elimination t1/2 ~6-7h
        # (multiple consistent sources: EJCP 1978 review, FDA label,
        # DailyMed). Cmax scaled from a 100mg-dose study (0.85-0.98 mg/L)
        # down to this 50mg case (~0.4-0.5 mg/L, linear approximation).
        "literature": {"Tmax_h": 3.0, "Cmax_mg_L": 0.45, "AUC_mg_h_L": 3.5},
    },
    "diclofenac": {  # weak acid, pKa~4.0, VERY heavy protein binding -> tests fu at an extreme
        "inputs": Tier1AssayInputs(
            smiles="OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", dose_mg=50.0, drug_status="known",
            dissolution_rate_mg_min_cm2=1.0, papp_cm_s=1.5e-3, fu=0.006,  # ~99.4% protein bound
            logp=4.5, clint_ul_min_mg=30.0, pka=4.0,
        ),
        # PARTIALLY SOURCED: Cmax and Tmax from a randomized crossover
        # trial (50mg potassium tablet, fasting) -- Cmax 1160+-452 ng/mL,
        # median tmax 0.50h (Chen et al. 2015, Headache journal). AUC
        # NOT individually confirmed -- original guess (4.0) is unverified.
        "literature": {"Tmax_h": 0.50, "Cmax_mg_L": 1.16, "AUC_mg_h_L": 4.0},
    },
}


def paracetamol_reference_inputs() -> Tier1AssayInputs:
    """Kept for backward compatibility with existing calls -- now just a
    pointer into KNOWN_DRUGS, single source of truth either way."""
    return KNOWN_DRUGS["paracetamol"]["inputs"]


def validate_against_paracetamol():
    """Kept for backward compatibility -- delegates to the general
    validator, single drug."""
    return validate_known_drugs(drug_names=["paracetamol"])["paracetamol"]


def validate_known_drugs(drug_names=None, scenario: str = "reversible", t_end_h: float = 24.0):
    """Runs get_all_estimates() for every drug in KNOWN_DRUGS (or a
    given subset) and prints model-vs-literature comparison for each.
    Returns a dict of {drug_name: result}."""
    names = drug_names or list(KNOWN_DRUGS.keys())
    all_results = {}

    for name in names:
        spec = KNOWN_DRUGS[name]
        t1 = spec["inputs"]
        result = get_all_estimates(t1, scenario=scenario, t_end_h=t_end_h)
        metrics = result["ode"]["metrics"]
        all_results[name] = result

        print(f"=== {name.capitalize()} Validation (ODE layer) ===")
        for key, lit_val in spec["literature"].items():
            model_val = metrics[key]
            pct_err = 100 * (model_val - lit_val) / lit_val if lit_val else float("nan")
            print(f"  {key:12s}  model={model_val:8.3f}   lit={lit_val:8.3f}   err={pct_err:+.1f}%")
        p = result["tier2_parameters"]
        print(f"  [drug_type={p.drug_type}  ka={p.ka_per_h:.3f}/h  F={p.F:.3f}  Fh={p.Fh:.3f}]")
        print()

    return all_results


# =====================================================================
# WIRING — all functions above are dead code unless called from here
# =====================================================================

if __name__ == "__main__":
    print("CDDSM simulation_core.py -- validating all 4 known drugs\n")
    validate_known_drugs()
