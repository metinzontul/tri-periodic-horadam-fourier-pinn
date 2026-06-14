# -*- coding: utf-8 -*-
"""
Created on Sat Jun 13 00:12:28 2026

@author: Metin Zontul
"""

# -*- coding: utf-8 -*-
"""
Tri-Periodic Horadam Sequence Parameter Estimation
Fourier-Embedded Recurrence-Informed PINN - NOISE ROBUSTNESS ANALYSIS

Updated: 
- Fixed Scenario: Safe Zone (Low Asymmetry: p=1.2, q=0.9, s=1.1)
- Varying Parameter: Noise Level (3%, 5%, 7%, 10%)
- Extrapolation Horizon N=60 maintained.
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from dataclasses import dataclass, asdict

torch.set_default_dtype(torch.float64)

# ============================================================
# 1. CONFIGURATION CLASSES
# ============================================================
@dataclass
class ExperimentConfig:
    p_true: float = 1.2; q_true: float = 0.9; s_true: float = 1.1 # Fixed to Safe Zone
    r_true: float = 0.5; a_value: float = 2.0; b_value: float = 1.0
    n_points: int = 60; train_end: int = 35; noise_std: float = 0.05; 
    seed: int = 123 

@dataclass
class TrainConfig:
    epochs_adam: int = 6000; lr_net: float = 1e-3; lr_params: float = 1e-2
    phys_weight: float = 50.0; loss_name: str = "logcosh"; use_fourier: bool = True
    teacher_forcing: bool = True; use_lbfgs: bool = True; lbfgs_lr: float = 0.1
    lbfgs_max_iter: int = 1000; grad_clip: float = 10.0; hidden_dim: int = 64
    n_hidden_layers: int = 2; verbose: bool = False

# ============================================================
# 2. DATA GENERATION & PINN ARCHITECTURE
# ============================================================
def generate_horadam_by_recurrence(p, q, s, r, a, b, n_points):
    H = np.zeros(n_points, dtype=np.float64)
    H[0], H[1] = a, b
    for n in range(2, n_points):
        coeff = p if n % 3 == 0 else q if n % 3 == 1 else s
        H[n] = coeff * H[n - 1] + r * H[n - 2]
    return H

def generate_dataset(p, q, s, r, a, b, n_points, noise_std, seed=42):
    clean = generate_horadam_by_recurrence(p, q, s, r, a, b, n_points)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, noise_std * np.abs(clean), size=n_points)
    return clean, clean + noise

class HoradamPINN(nn.Module):
    def __init__(self, cfg: TrainConfig, r_value=0.5):
        super().__init__()
        input_dim = 3 if cfg.use_fourier else 1
        layers = [nn.Linear(input_dim, cfg.hidden_dim), nn.Tanh()]
        for _ in range(cfg.n_hidden_layers - 1):
            layers += [nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(cfg.hidden_dim, 1))
        self.net = nn.Sequential(*layers)
        self.p_log = nn.Parameter(torch.log(torch.tensor([1.1])))
        self.q_log = nn.Parameter(torch.log(torch.tensor([1.1])))
        self.s_log = nn.Parameter(torch.log(torch.tensor([1.1])))
        self.register_buffer("r_fixed", torch.tensor([r_value]))

    def forward(self, n_raw, n_max, use_fourier=True):
        n_norm = n_raw / (n_max + 1e-14)
        if not use_fourier: return self.net(n_norm)
        sin_emb = torch.sin(2.0 * torch.pi * n_raw / 3.0)
        cos_emb = torch.cos(2.0 * torch.pi * n_raw / 3.0)
        return self.net(torch.cat([n_norm, sin_emb, cos_emb], dim=1))

    @property
    def p(self): return torch.exp(self.p_log)
    @property
    def q(self): return torch.exp(self.q_log)
    @property
    def s(self): return torch.exp(self.s_log)
    @property
    def r(self): return self.r_fixed

# ============================================================
# 3. TRAINING ENGINE
# ============================================================
def train_pinn(noisy, train_idx, r_value, cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    h_scale = float(max(np.max(np.abs(noisy[train_idx])), 1.0))
    model = HoradamPINN(cfg, r_value=r_value).to(device)
    n_train = torch.tensor(train_idx.reshape(-1, 1), dtype=torch.float64, device=device)
    n_max_val = float(np.max(train_idx))
    H_train = torch.tensor((noisy[train_idx] / h_scale).reshape(-1, 1), dtype=torch.float64, device=device)
    
    H_prev1 = torch.tensor((noisy[train_idx - 1] / h_scale).reshape(-1, 1), dtype=torch.float64, device=device)
    H_prev2 = torch.tensor((noisy[train_idx - 2] / h_scale).reshape(-1, 1), dtype=torch.float64, device=device)
    phase = torch.tensor(train_idx % 3, dtype=torch.long, device=device)
    
    opt = optim.Adam([{"params": model.net.parameters(), "lr": cfg.lr_net}, {"params": [model.p_log, model.q_log, model.s_log], "lr": cfg.lr_params}])
    
    def loss_fn():
        H_pred = model(n_train, torch.tensor([n_max_val]), use_fourier=cfg.use_fourier)
        d_loss = torch.mean(torch.log(torch.cosh(H_pred - H_train) + 1e-8)) if cfg.loss_name == "logcosh" else torch.mean((H_pred - H_train)**2)
        p_loss = 0
        for i, m in enumerate([phase==0, phase==1, phase==2]):
            if torch.any(m):
                coeff = [model.p, model.q, model.s][i]
                if cfg.teacher_forcing:
                    res = H_pred[m] - (coeff * H_prev1[m] + model.r * H_prev2[m])
                else:
                    n_prev1_tensor = torch.tensor(((train_idx[m.cpu()] - 1).reshape(-1, 1)), dtype=torch.float64, device=device)
                    n_prev2_tensor = torch.tensor(((train_idx[m.cpu()] - 2).reshape(-1, 1)), dtype=torch.float64, device=device)
                    pred_prev1 = model(n_prev1_tensor, torch.tensor([n_max_val]), use_fourier=cfg.use_fourier)
                    pred_prev2 = model(n_prev2_tensor, torch.tensor([n_max_val]), use_fourier=cfg.use_fourier)
                    res = H_pred[m] - (coeff * pred_prev1 + model.r * pred_prev2)
                p_loss += torch.mean(torch.log(torch.cosh(res) + 1e-8)) if cfg.loss_name == "logcosh" else torch.mean(res**2)
        return d_loss + cfg.phys_weight * p_loss

    for epoch in range(cfg.epochs_adam):
        opt.zero_grad()
        loss_fn().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
    
    if cfg.use_lbfgs:
        lbfgs = optim.LBFGS(model.parameters(), lr=cfg.lbfgs_lr, max_iter=cfg.lbfgs_max_iter, line_search_fn="strong_wolfe")
        def closure():
            lbfgs.zero_grad()
            loss = loss_fn()
            loss.backward()
            return loss
        lbfgs.step(closure)
        
    return model, h_scale, n_max_val

# ============================================================
# 4. EXPERIMENT RUNNER & INDIVIDUAL PLOTTING
# ============================================================
def run_experiment(exp: ExperimentConfig, cfg: TrainConfig, plot: bool = False, title: str = "") -> Dict:
    clean, noisy = generate_dataset(exp.p_true, exp.q_true, exp.s_true, exp.r_true, exp.a_value, exp.b_value, exp.n_points, exp.noise_std, exp.seed)
    train_idx = np.arange(2, exp.train_end)
    test_idx = np.arange(exp.train_end, exp.n_points)
    
    model, _, _ = train_pinn(noisy, train_idx, exp.r_true, cfg)
    p_est, q_est, s_est = model.p.item(), model.q.item(), model.s.item()
    recon = generate_horadam_by_recurrence(p_est, q_est, s_est, exp.r_true, exp.a_value, exp.b_value, exp.n_points)
    
    mae = (abs(exp.p_true - p_est) + abs(exp.q_true - q_est) + abs(exp.s_true - s_est)) / 3.0
    log_rmse = np.sqrt(np.mean((np.log1p(np.abs(clean[test_idx])) - np.log1p(np.abs(recon[test_idx])))**2))
    mse = np.mean((clean[test_idx] - recon[test_idx])**2)
    
    if plot:
        fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
        ax.plot(clean, 'g-', label="Clean Sequence (Ground Truth)", alpha=0.8, linewidth=2)
        ax.scatter(np.arange(exp.n_points), noisy, c='red', s=25, alpha=0.6, label=f"Noisy Obs. ({int(exp.noise_std*100)}%)")
        ax.plot(recon, 'b--', label="PINN Reconstruction", linewidth=2.5)
        
        ax.fill_between(np.arange(exp.n_points), clean, recon, color='red', alpha=0.15, label="Error Area")
        ax.axvline(exp.train_end - 1, color='black', linestyle=':', linewidth=2, label="Train/Test Boundary")
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("n (Sequence Step)", fontsize=11)
        ax.set_ylabel("Amplitude (symlog scale)", fontsize=11)
        ax.set_yscale('symlog')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=10)

        # Inset Zoom (Adjusted for n=60)
        axins = ax.inset_axes([0.05, 0.45, 0.40, 0.35]) 
        axins.plot(clean, 'g-', linewidth=2)
        axins.plot(recon, 'b--', linewidth=2)
        axins.fill_between(np.arange(exp.n_points), clean, recon, color='red', alpha=0.2)
        axins.set_xlim(exp.train_end - 2, exp.n_points - 1)
        
        y_min_inset = min(np.min(clean[exp.train_end-2:]), np.min(recon[exp.train_end-2:]))
        y_max_inset = max(np.max(clean[exp.train_end-2:]), np.max(recon[exp.train_end-2:]))
        axins.set_ylim(y_min_inset * 0.8, y_max_inset * 1.2)
        axins.set_yscale('log')
        axins.set_title("Zoom: Extrapolation Region Divergence", fontsize=9)
        axins.grid(True, alpha=0.2)
        ax.indicate_inset_zoom(axins, edgecolor="black")

        # Academic styling
        textstr = '\n'.join((
            r'$\mathbf{True\ Params:}$', f'p = {exp.p_true:.3f}, q = {exp.q_true:.3f}, s = {exp.s_true:.3f}',
            r'$\mathbf{PINN\ Estimates:}$', f'p = {p_est:.3f}, q = {q_est:.3f}, s = {s_est:.3f}',
            r'$\mathbf{Success\ Metrics:}$', f'Param MAE = {mae:.3f}', f'Test Log-RMSE = {log_rmse:.2f}'
        ))
        props = dict(boxstyle='square,pad=0.4', facecolor='white', edgecolor='black', alpha=0.9)
        ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=10, verticalalignment='bottom', horizontalalignment='right', bbox=props)
        
        safe_title = title.replace(" ", "_").replace(":", "").replace("(", "").replace(")", "").replace(",", "").replace("%", "")
        plt.tight_layout()
        plt.savefig(f"Figure_{safe_title}.png", bbox_inches='tight')
        plt.close()

    return {"mae": mae, "log_rmse": log_rmse, "mse": mse}

# ============================================================
# 6. MAIN EXECUTION
# ============================================================
if __name__ == "__main__":
    train_cfg = TrainConfig(verbose=False)
    
    # FIXED ASYMMETRY (SAFE ZONE), VARYING NOISE
    scenarios = [
        {"title": "(a) Safe Zone (3% Noise)", "p": 1.2, "q": 0.9, "s": 1.1, "noise": 0.03},
        {"title": "(b) Safe Zone (5% Noise)", "p": 1.2, "q": 0.9, "s": 1.1, "noise": 0.05},
        {"title": "(c) Safe Zone (7% Noise)", "p": 1.2, "q": 0.9, "s": 1.1, "noise": 0.07},
        {"title": "(d) Safe Zone (10% Noise)", "p": 1.2, "q": 0.9, "s": 1.1, "noise": 0.10}
    ]

    print("Generating 4 Separate High-Resolution Manuscript Figures (Varying Noise, N=60)...")
    all_res = []
    
    for cfg in scenarios:
        exp_cfg = ExperimentConfig(p_true=cfg['p'], q_true=cfg['q'], s_true=cfg['s'], noise_std=cfg['noise'], seed=123, n_points=60)
        res = run_experiment(exp_cfg, train_cfg, plot=True, title=cfg['title'])
        res['title'] = cfg['title']
        all_res.append(res)
        print(f"Saved: Figure_{cfg['title'].replace(' ', '_').replace(':', '').replace('.', '').replace('%', '')}.png")

    # Print Summary Table
    print("\n" + "="*85)
    print(f"{'SCENARIO (SAFE ZONE, VARYING NOISE, N=60)':<42} | {'PARAM MAE':<12} | {'TEST LOG-RMSE':<15}")
    print("-" * 85)
    for r in all_res:
        print(f"{r['title']:<42} | {r['mae']:<12.5f} | {r['log_rmse']:<15.5f}")
    print("="*85)

    print("\nTasks completed. The plots demonstrating Noise Robustness in the Safe Zone are ready.")