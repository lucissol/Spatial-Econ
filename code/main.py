# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 14:34:54 2026

@author: lseibert
"""
import sys
import os

script_dir = os.path.dirname(os.path.abspath("helper.py"))
if script_dir not in sys.path:
    sys.path.append(script_dir)
    
import config as config
import optimizer
import stoch_tools
import macro_var
import helper
import matplotlib.pyplot as plt
import numpy as np

# idiosyncratic AR(1) shock
# a' = (1+r)a + w * z - c
# where log(z_t) = theta log(z_t-1) + iid shock
# Highly persistent thus discretize shock with Rouwenhorst

# ==============================
# CONFIG
# ==============================
beta = 0.96
r_f = (1/beta) - 1
r = r_f * 0.95
print(f"Complete Market Interest Rate:{r_f}")
print(f"Utilizing r of {r}")

w_core = 1
w_periphery = 0.8
sigma = 1
theta = 0.6
mu_z = 0
sigma_eps = 0.2

eps_process = stoch_tools.discretize_AR1(theta, mu_z, sigma_eps)
eps_grid_log, eps_Q = eps_process.Tauchen(5)
eps_grid = np.exp(eps_grid_log)


economy_vals = optimizer.Economy(beta, sigma, eps_grid, eps_Q)


## grid
b = 0
na = 1000
households_core = optimizer.Household(economy_vals, na, b)
households_periphery = optimizer.Household(economy_vals, na, b)
# solve for the optimal policy
households_core._solve_EGM(r, w_core)
households_periphery._solve_EGM(r, w_periphery)

#%%

# scenario analysis comparing positive and negative shocks to periphery and core
if __name__ == '__main__': 
    print("Running first model")
    model1 = helper.run_scenario(households_core, households_periphery, 0.1, "core")
    print("Running second model")
    model2 =  helper.run_scenario(households_core, households_periphery, 0.1, "periphery")
    print("Running third model")
    model3 = helper.run_scenario(households_core, households_periphery, -0.1, "core")
    print("Running fourth model")
    model4 = helper.run_scenario(households_core, households_periphery, -0.1, "periphery")
    
    # Organize data
    scenarios = [
        # (Data Core, Data Periph, Title)
        (model1[0], model1[1], "Scenario 1: +10% Shock to Core"),
        (model2[0], model2[1], "Scenario 2: +10% Shock to Periphery"),
        (model3[0], model3[1], "Scenario 3: -10% Shock to Core"),
        (model4[0], model4[1], "Scenario 4: -10% Shock to Periphery")
    ]
    # Create 2x2 Grid
    plt.style.use('seaborn-v0_8-paper')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=100, sharex=True, sharey=True)
    axes = axes.flatten() # Flattens the 2x2 grid into a list [0, 1, 2, 3] for easy looping
    # Time axis (assumin shock path is equal across models)
    t = np.arange(len(model1[0]))
    
    for i, ax in enumerate(axes):
        core_data, periph_data, title = scenarios[i]
        
        # Plot Lines
        ax.plot(t, core_data, color='tab:blue', linewidth=2, label='Core Gini Impact')
        ax.plot(t, periph_data, color='tab:red', linewidth=2, label='Periphery Gini Impact')
        
        # Zero Line
        ax.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.6)
        
        # Shading (Red if Periphery > 0, Blue if Periphery < 0)
        ax.fill_between(t, periph_data, 0, where=(periph_data > 0), color='tab:red', alpha=0.1)
        ax.fill_between(t, periph_data, 0, where=(periph_data < 0), color='tab:blue', alpha=0.1)
    
        # Styling
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Only add labels to outer edges to reduce clutter
        if i in [0, 2]: # Left column
            ax.set_ylabel("Change in Gini Points", fontsize=10)
        if i in [2, 3]: # Bottom column
            ax.set_xlabel("Time (Periods)", fontsize=10)
    
    # Single Legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.02), ncol=2, frameon=False, fontsize=12)
    
    plt.tight_layout()
    plt.show()
    
    # Calculate the Differences across regions
    # Positive Shocks (+10%)
    diff_pos_core   = model1[0] - model2[0]
    diff_pos_periph = model1[1] - model2[1]
    
    # Negative Shocks (-10%)
    diff_neg_core   = model3[0] - model4[0]
    diff_neg_periph = model3[1] - model4[1]
    
    # Setup Plot
    plt.style.use('seaborn-v0_8-paper')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=100, sharey=True)
    t = np.arange(len(diff_pos_core))
    
    # --- Plot 1: Difference in Positive Shocks (+10%) ---
    ax1.plot(t, diff_pos_core, color='tab:blue', label='Difference in Core Impact')
    ax1.plot(t, diff_pos_periph, color='tab:red', linestyle='--', label='Difference in Periphery Impact')
    
    ax1.set_title("Diff: (Shock Core) - (Shock Periphery) [+10% Case]", fontsize=12, fontweight='bold')
    ax1.set_ylabel("Difference in Gini Points (Residuals)")
    ax1.set_xlabel("Time (Periods)")
    ax1.axhline(0, color='black', alpha=0.8, linewidth=1) # Zero line
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # --- Plot 2: Difference in Negative Shocks (-10%) ---
    ax2.plot(t, diff_neg_core, color='tab:blue', label='Difference in Core Impact')
    ax2.plot(t, diff_neg_periph, color='tab:red', linestyle='--', label='Difference in Periphery Impact')
    
    ax2.set_title("Diff: (Shock Core) - (Shock Periphery) [-10% Case]", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Time (Periods)")
    ax2.axhline(0, color='black', alpha=0.8, linewidth=1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle("Does Location Matter? (Residuals between Left and Right plots)", fontsize=16, y=1.05)
    plt.tight_layout()
    plt.show()
    
    # Print Max Differences to check for floating point errors
    print(f"Max absolute difference in Core Response (+10% case): {np.max(np.abs(diff_pos_core)):.6f}")
    print(f"Max absolute difference in Periphery Response (+10% case): {np.max(np.abs(diff_pos_periph)):.6f}")
    
 