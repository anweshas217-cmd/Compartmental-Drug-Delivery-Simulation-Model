"""
pinn_model.py
=============
CDDSM — Physics-Informed Neural Network for the gut-wall diffusion field.

WHAT THIS SOLVES
-----------------
The same PDE as simulation_core.py's coupled PDE layer:
    dC/dt = D * (d2C/dr2 + (1/r) dC/dr),   r in [r_inner, r_outer]
    C(r_inner, t) = lumen boundary condition
    C(r_outer, t) = blood boundary condition
but approximated by a neural network C_hat(r, t; D, Peff) instead of a
finite-difference grid, so that once trained it can be evaluated
instantly for a NEW drug's (D, Peff) pair without re-solving from scratch.

GROUND TRUTH SOURCE — wired to simulation_core.py, not a separate PDE.
Ground truth trajectories come from
simulation_core.get_spatial_concentration_field(), i.e. the exact same
coupled ODE+PDE solver validated against Paracetamol. This closes a real
gap: previously the PINN's finite-difference data and the main PDE layer
were two independent implementations that could silently drift apart.

HARD DIRICHLET CONSTRAINT (architecturally guaranteed, not a soft
penalty loss term) — see hard_constrained_output() below. This was
identified as outperforming soft-penalty training, especially for
early-time concentration behavior, and is preserved in this rebuild.

PARAMETRIC SWEEP EXTENSION — the network is conditioned on (D, Peff) as
explicit inputs, not just (r, t). This is what makes the "train once,
fine-tune fast on a new drug" workflow possible: a new drug's D/Peff
values can be evaluated directly, or the pretrained weights can be
lightly fine-tuned on a handful of that drug's true collocation points.
"""

import numpy as np
import torch
import torch.nn as nn

from simulation_core import (
    Tier1AssayInputs, Tier3PhysiologicalConstants,
    get_spatial_concentration_field, compute_tier2_parameters,
)

torch.manual_seed(0)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =====================================================================
# NORMALIZATION HELPERS
# Networks train far better on O(1) inputs than on raw physical units
# (r ~ 1.75 cm, t ~ 0-24h, D ~ 1e-6 cm^2/s, Peff ~ 1e-4-1e-2 cm/s all
# live on wildly different scales).
# =====================================================================

class Normalizer:
    def __init__(self, r_inner, r_outer, t_end_h, D_range, peff_range):
        self.r_inner, self.r_outer = r_inner, r_outer
        self.t_end_h = t_end_h
        self.D_lo, self.D_hi = D_range
        self.peff_lo, self.peff_hi = peff_range

    def norm_r(self, r):
        return (r - self.r_inner) / (self.r_outer - self.r_inner)  # -> [0, 1]

    def norm_t(self, t):
        return t / self.t_end_h  # -> [0, 1] (approx)

    def norm_D(self, D):
        return (np.log10(D) - np.log10(self.D_lo)) / (np.log10(self.D_hi) - np.log10(self.D_lo))

    def norm_peff(self, p):
        return (np.log10(p) - np.log10(self.peff_lo)) / (np.log10(self.peff_hi) - np.log10(self.peff_lo))


# =====================================================================
# NETWORK — hard-constrained Dirichlet boundary
# =====================================================================

class PINN(nn.Module):
    """Inputs: (r_norm, t_norm, D_norm, Peff_norm) each in ~[0,1].
    Output: normalized concentration C_hat.

    Hard boundary constraint (architectural, not a penalty term):
        C_hat(r, t) = (1 - r_norm) * C_inner_bc(t) + r_norm * C_outer_bc(t)
                      + r_norm * (1 - r_norm) * NN(r, t, D, Peff)
    The linear interpolation term exactly satisfies both boundary values
    by construction. The NN term is multiplied by r_norm*(1-r_norm),
    which is exactly zero at both boundaries -- so no matter what the
    network outputs, the boundary conditions CANNOT be violated. This
    is strictly stronger than a soft L2 penalty on boundary residuals.
    """

    def __init__(self, hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(4, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, r_norm, t_norm, D_norm, peff_norm, c_inner_bc, c_outer_bc):
        x = torch.cat([r_norm, t_norm, D_norm, peff_norm], dim=1)
        raw = self.net(x)
        bump = r_norm * (1.0 - r_norm)
        interp = (1.0 - r_norm) * c_inner_bc + r_norm * c_outer_bc
        return interp + bump * raw


def hard_constrained_output(model, r_norm, t_norm, D_norm, peff_norm, c_inner_bc, c_outer_bc):
    """Thin wrapper kept as a named entry point -- makes the hard-
    constraint property visible at call sites, not just inside forward()."""
    return model(r_norm, t_norm, D_norm, peff_norm, c_inner_bc, c_outer_bc)


# =====================================================================
# GROUND TRUTH GENERATION — parametric sweep across D / Peff
# Wired directly to simulation_core.py's coupled solver.
# =====================================================================

def generate_training_trajectories(n_drugs: int = 12, t_end_h: float = 8.0,
                                    n_r: int = 12, n_points: int = 40,
                                    D_range=(1e-7, 1e-5), peff_range=(1e-5, 5e-3),
                                    seed: int = 0):
    """Sweeps synthetic (D, Peff) pairs across physiologically realistic
    ranges and generates ground-truth C(r,t) fields for each via
    simulation_core's coupled solver. This is the extension flagged as
    'not yet implemented' in the project log -- it's what lets the PINN
    generalize to a NEW drug instead of only ever reproducing Paracetamol.
    """
    rng = np.random.default_rng(seed)
    log_D = rng.uniform(np.log10(D_range[0]), np.log10(D_range[1]), n_drugs)
    log_peff = rng.uniform(np.log10(peff_range[0]), np.log10(peff_range[1]), n_drugs)
    D_samples = 10 ** log_D
    peff_samples = 10 ** log_peff

    trajectories = []
    t3_template = Tier3PhysiologicalConstants()
    for D, peff in zip(D_samples, peff_samples):
        t3 = Tier3PhysiologicalConstants(diffusion_coefficient_cm2_s=D)
        t1 = Tier1AssayInputs(
            smiles=None, dose_mg=500.0, drug_status="new",
            dissolution_rate_mg_min_cm2=2.5, papp_cm_s=peff,
            fu=0.9, logp=1.0, clint_ul_min_mg=8.0, pka=7.0,
        )
        field = get_spatial_concentration_field(t1, t3, scenario="reversible",
                                                  t_end_h=t_end_h, n_r=n_r, n_points=n_points)
        trajectories.append({"D": D, "peff": peff, **field})

    return trajectories, t3_template


def trajectories_to_training_tensors(trajectories, normalizer: Normalizer,
                                      skew_early_time: bool = True, skew_power: float = 2.5):
    """Flattens the (drug, r, t) grid into training tuples. Early-time
    sampling is skewed (denser near t=0) because concentration gradients
    are steepest right after dosing -- uniform time sampling under-
    represents exactly the region the model struggles with most."""
    rs, ts, Ds, Ps, Cs, c_inner, c_outer = [], [], [], [], [], [], []

    for traj in trajectories:
        r_arr, t_arr, C = traj["r_cm"], traj["t_h"], traj["C_mg_mL"]
        n_r, n_t = C.shape

        if skew_early_time:
            u = np.linspace(0, 1, n_t)
            t_idx_frac = u ** skew_power  # bias toward 0 (early time)
            t_indices = (t_idx_frac * (n_t - 1)).astype(int)
        else:
            t_indices = np.arange(n_t)

        for ti in t_indices:
            for ri in range(n_r):
                rs.append(normalizer.norm_r(r_arr[ri]))
                ts.append(normalizer.norm_t(t_arr[ti]))
                Ds.append(normalizer.norm_D(traj["D"]))
                Ps.append(normalizer.norm_peff(traj["peff"]))
                Cs.append(C[ri, ti])
                c_inner.append(C[0, ti])
                c_outer.append(C[-1, ti])

    def col(v):
        return torch.tensor(np.array(v), dtype=torch.float32, device=DEVICE).view(-1, 1)

    return col(rs), col(ts), col(Ds), col(Ps), col(Cs), col(c_inner), col(c_outer)


# =====================================================================
# PDE RESIDUAL LOSS (physics-informed term, via autograd)
# =====================================================================

def pde_residual_loss(model, r_norm, t_norm, D_norm, peff_norm, c_inner_bc, c_outer_bc,
                       D_physical, r_inner_cm, r_outer_cm, t_end_h):
    """Computes the Fick's-law residual at collocation points via
    automatic differentiation, enforced as a physics loss term IN
    ADDITION TO the hard boundary constraint (which handles the
    boundary values themselves, not the interior physics)."""
    r_norm = r_norm.clone().requires_grad_(True)
    t_norm = t_norm.clone().requires_grad_(True)

    C_hat = hard_constrained_output(model, r_norm, t_norm, D_norm, peff_norm, c_inner_bc, c_outer_bc)

    dC_dt_norm = torch.autograd.grad(C_hat, t_norm, grad_outputs=torch.ones_like(C_hat),
                                      create_graph=True)[0]
    dC_dr_norm = torch.autograd.grad(C_hat, r_norm, grad_outputs=torch.ones_like(C_hat),
                                      create_graph=True)[0]
    d2C_dr2_norm = torch.autograd.grad(dC_dr_norm, r_norm, grad_outputs=torch.ones_like(dC_dr_norm),
                                        create_graph=True)[0]

    # de-normalize derivatives via chain rule: r_norm = (r-r_inner)/(r_outer-r_inner), t_norm = t/t_end
    r_scale = (r_outer_cm - r_inner_cm)
    dC_dt = dC_dt_norm / t_end_h
    dC_dr = dC_dr_norm / r_scale
    d2C_dr2 = d2C_dr2_norm / (r_scale ** 2)

    r_physical = r_norm.detach() * r_scale + r_inner_cm
    residual = dC_dt - D_physical * (d2C_dr2 + (1.0 / r_physical) * dC_dr)
    return torch.mean(residual ** 2)


# =====================================================================
# TRAINING LOOP
# =====================================================================

def train_pinn(n_epochs: int = 2000, n_drugs: int = 12, t_end_h: float = 8.0,
                D_range=(1e-7, 1e-5), peff_range=(1e-5, 5e-3), verbose_every: int = 200,
                n_r: int = 12, n_points: int = 40):
    trajectories, t3_template = generate_training_trajectories(
        n_drugs=n_drugs, t_end_h=t_end_h, D_range=D_range, peff_range=peff_range,
        n_r=n_r, n_points=n_points,
    )
    normalizer = Normalizer(
        r_inner=t3_template.r_intestine_inner_cm,
        r_outer=t3_template.r_intestine_inner_cm + t3_template.r_outer_cm,
        t_end_h=t_end_h, D_range=D_range, peff_range=peff_range,
    )
    r_n, t_n, D_n, P_n, C_true, c_in, c_out = trajectories_to_training_tensors(trajectories, normalizer)

    model = PINN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs, eta_min=1e-5)

    # representative D for the physics-residual term: use the sweep's
    # geometric mean as a fixed constant per batch, since D also varies
    # per-sample via D_norm fed into the network itself.
    D_geomean = float(np.sqrt(D_range[0] * D_range[1]))

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        C_pred = hard_constrained_output(model, r_n, t_n, D_n, P_n, c_in, c_out)
        data_loss = torch.mean((C_pred - C_true) ** 2)

        physics_loss = pde_residual_loss(
            model, r_n, t_n, D_n, P_n, c_in, c_out,
            D_physical=D_geomean * 3600.0,  # cm^2/s -> cm^2/h, matches simulation_core convention
            r_inner_cm=normalizer.r_inner, r_outer_cm=normalizer.r_outer, t_end_h=t_end_h,
        )

        loss = data_loss + 0.1 * physics_loss
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % verbose_every == 0 or epoch == n_epochs - 1:
            print(f"  epoch {epoch:5d}  data_loss={data_loss.item():.6e}  "
                  f"physics_loss={physics_loss.item():.6e}  total={loss.item():.6e}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

    return model, normalizer, trajectories


# =====================================================================
# VALIDATION — mean relative error against held-out ground truth
# =====================================================================

def evaluate_mean_relative_error(model, normalizer, held_out_trajectories):
    model.eval()
    errors = []
    with torch.no_grad():
        for traj in held_out_trajectories:
            r_arr, t_arr, C_true_grid = traj["r_cm"], traj["t_h"], traj["C_mg_mL"]
            n_r, n_t = C_true_grid.shape
            for ti in range(0, n_t, max(1, n_t // 50)):  # subsample for speed
                for ri in range(n_r):
                    r_n = torch.tensor([[normalizer.norm_r(r_arr[ri])]], dtype=torch.float32, device=DEVICE)
                    t_n = torch.tensor([[normalizer.norm_t(t_arr[ti])]], dtype=torch.float32, device=DEVICE)
                    D_n = torch.tensor([[normalizer.norm_D(traj["D"])]], dtype=torch.float32, device=DEVICE)
                    P_n = torch.tensor([[normalizer.norm_peff(traj["peff"])]], dtype=torch.float32, device=DEVICE)
                    c_in = torch.tensor([[C_true_grid[0, ti]]], dtype=torch.float32, device=DEVICE)
                    c_out = torch.tensor([[C_true_grid[-1, ti]]], dtype=torch.float32, device=DEVICE)

                    pred = hard_constrained_output(model, r_n, t_n, D_n, P_n, c_in, c_out).item()
                    true = C_true_grid[ri, ti]
                    if abs(true) > 1e-8:
                        errors.append(abs(pred - true) / abs(true))
    model.train()
    return 100.0 * float(np.mean(errors)) if errors else float("nan")


# =====================================================================
# MODEL PERSISTENCE — app.py must NOT retrain live (a ~3min training
# run inside an interactive session is unusable). Train once here,
# save weights + normalizer bounds, and have app.py load them.
# =====================================================================

def save_model(model, normalizer: Normalizer, path: str = "pinn_weights.pt"):
    torch.save({
        "state_dict": model.state_dict(),
        "r_inner": normalizer.r_inner, "r_outer": normalizer.r_outer,
        "t_end_h": normalizer.t_end_h,
        "D_lo": normalizer.D_lo, "D_hi": normalizer.D_hi,
        "peff_lo": normalizer.peff_lo, "peff_hi": normalizer.peff_hi,
    }, path)


def load_model(path: str = "pinn_weights.pt"):
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    model = PINN().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    normalizer = Normalizer(
        r_inner=checkpoint["r_inner"], r_outer=checkpoint["r_outer"],
        t_end_h=checkpoint["t_end_h"],
        D_range=(checkpoint["D_lo"], checkpoint["D_hi"]),
        peff_range=(checkpoint["peff_lo"], checkpoint["peff_hi"]),
    )
    return model, normalizer


# =====================================================================
# WIRING
# =====================================================================

if __name__ == "__main__":
    import sys
    quick = "--quick" in sys.argv

    print("CDDSM pinn_model.py -- training on a parametric D/Peff sweep\n")

    if quick:
        model, normalizer, train_trajectories = train_pinn(
            n_epochs=300, n_drugs=6, t_end_h=8.0, verbose_every=50,
        )
    else:
        model, normalizer, train_trajectories = train_pinn(
            n_epochs=600, n_drugs=25, t_end_h=8.0, n_r=18, n_points=55, verbose_every=100,
        )

    print("\nGenerating held-out drugs (different seed) for validation...")
    held_out, _ = generate_training_trajectories(n_drugs=4, t_end_h=8.0, seed=99)

    mre = evaluate_mean_relative_error(model, normalizer, held_out)
    print(f"\nMean relative error on held-out synthetic drugs: {mre:.3f}%")

    print("\nSanity check against Paracetamol's own (D, Peff):")
    t1 = Tier1AssayInputs(
        smiles="CC(=O)NC1=CC=C(O)C=C1", dose_mg=500.0, drug_status="known",
        dissolution_rate_mg_min_cm2=2.5, papp_cm_s=2.0e-3, fu=0.9,
        logp=0.46, clint_ul_min_mg=8.0, pka=9.4,
    )
    t3_para = Tier3PhysiologicalConstants()
    para_field = get_spatial_concentration_field(t1, t3_para, t_end_h=8.0)
    para_traj = [{"D": t3_para.diffusion_coefficient_cm2_s, "peff": 2.0e-3, **para_field}]
    para_mre = evaluate_mean_relative_error(model, normalizer, para_traj)
    print(f"Mean relative error on Paracetamol: {para_mre:.3f}%")

    save_model(model, normalizer, path="pinn_weights.pt")
    print("\nSaved trained weights to pinn_weights.pt (app.py loads this, never trains live).")
