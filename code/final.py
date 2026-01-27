# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 14:34:54 2026

@author: lseibert
"""

import optimizer
import stoch_tools
import macro_var
import helper as tools
from joblib import Parallel, delayed

import matplotlib.pyplot as plt
import numpy as np
import random

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
#%%
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
# ========================
# Macro Layer
# ========================

T = 100
rho_core = 0.7
rho_periphery = 0.6
rho_c_p = 0.5
rho_p_c = 0.2

macro_series = macro_var.SpatialMacroEconomy(rho_core=rho_core,
                                             rho_periph=rho_periphery,
                                             spill_c_to_p=rho_c_p,
                                             spill_p_to_c=rho_p_c
                                             )

# steady state paths
w_core_path_SS = np.ones(T) * w_core
w_periphery_path_SS = np.ones(T) * w_periphery

# shock config
shock_size = 0.1 # in % deviations
shock_region = "core" # "core" or "periphery"

deviations = macro_series.simulate_irf(T=T, shock_region=shock_region, shock_size=shock_size)


# shock at t = 1
w_core_path = w_core * (1 + deviations[0])
w_periphery_path = w_periphery * (1 + deviations[1])



# plot the time series using python style sheets
plt.style.use('seaborn-v0_8-paper') 
fig, ax = plt.subplots(figsize=(10, 5), dpi=100)

# Plotting the Path Series
ax.plot(w_core_path, label='Core Wage Path (Direct Shock)', 
        color='#1f77b4', linewidth=2, linestyle='-')
ax.plot(w_periphery_path, label='Periphery Wage Path (Spatial Spillover)', 
        color='#d62728', linewidth=2, linestyle='--')

# Styling the Chart
ax.set_title(f"Spatial Wage Propagation: {shock_size*100}% Shock to {shock_region.capitalize()}", 
             fontsize=14, fontweight='bold')
ax.set_xlabel("Time Periods (T)", fontsize=12)
ax.set_ylabel("Wage Level ($w_t$)", fontsize=12)

# Adding a horizontal line for the Core Steady State
ax.axhline(1.0, color='black', alpha=0.3, linestyle=':', label='Core Steady State')

ax.legend(frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()


#%%
# =======================
# Invariant Distribution
# =======================
H_sim =10_000
# start from the stationary distribution of the aiyagari economy
T_sim = 20000
burn_in = 5000

a_core, _ = households_core._simulate_stationary_distribution(r=r,
                                                              w=w_core,
                                                              T=T_sim,
                                                              H=H_sim,
                                                              burn_in=burn_in
                                                              )


a_periphery, _ = households_periphery._simulate_stationary_distribution(r=r,
                                                                        w=w_periphery,
                                                                        T=T_sim,
                                                                        H=H_sim,
                                                                        burn_in=burn_in
                                                                        )
# plot
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 6), dpi=100)
# We cap the range at a reasonable multiple of the mean to see the tail clearly
n_bins = 100
plot_range = (-households_core.b, np.percentile(a_core, 99)) 

# Plot Core
ax.hist(a_core, bins=n_bins, range=plot_range, density=True, 
        alpha=0.5, color='steelblue', label='Core Region ($w=1.0$)', 
        edgecolor='white', linewidth=0.5)

# Plot Periphery
ax.hist(a_periphery, bins=n_bins, range=plot_range, density=True, 
        alpha=0.5, color='indianred', label='Periphery Region ($w=0.7$)', 
        edgecolor='white', linewidth=0.5)

# Styling
ax.set_title("Stationary Wealth Distributions: Core vs. Periphery", fontsize=15, fontweight='bold')
ax.set_xlabel("Assets ($a$)", fontsize=12)
ax.set_ylabel("Probability Density", fontsize=12)

# Mark  Borrowing Limit
ax.axvline(-households_core.b, color='black', linestyle='--', alpha=0.7, label=f'Borrowing Limit ($-b$)')
ax.legend(loc='upper right', frameon=True, facecolor='white')
ax.grid(axis='y', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#%%
# plot Lorenz curve
tools.plot_spatial_lorenz(a_core, a_periphery, 0)
# since only the level of w is different, relative ineuqality is equal
# I expect ths to change when stress testing the economy generating a bad state, then, more of the poor households will be constraint in the periphery


#%%
# =====================
# FACTUAL ANALYSIS / NO SHOCK SCENARIO / STEADY STATE ECONOMY
# =====================
H_na = 10_000 # number of simulated households

gini_core_base = households_core.simulate_gini_path(r, w_core_path_SS, initial_dist=a_core, H=H_na, T=T)
gini_core_shocked = households_core.simulate_gini_path(r, w_core_path, initial_dist=a_core, H=H_na, T=T)

#%%
gini_periphery_base = households_periphery.simulate_gini_path(r, w_periphery_path_SS, initial_dist=a_periphery, H=H_na, T=T)
gini_periphery_shocked = households_periphery.simulate_gini_path(r, w_periphery_path, initial_dist=a_periphery, H=H_na, T=T)  
#%%

plt.style.use('seaborn-v0_8-whitegrid')

# IRFs
irf_core = gini_core_shocked - gini_core_base
irf_periphery = gini_periphery_shocked - gini_periphery_base

# figures
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=100)
t_axis = np.arange(len(gini_core_base))

# --- Top Plot: Level Paths ---
ax1.plot(t_axis, gini_core_base, color='tab:blue', linestyle='--', alpha=0.5, label='Core (Steady State)')
ax1.plot(t_axis, gini_core_shocked, color='tab:blue', linewidth=2, label='Core (Shocked Path)')

ax1.plot(t_axis, gini_periphery_base, color='tab:red', linestyle='--', alpha=0.5, label='Periphery (Steady State)')
ax1.plot(t_axis, gini_periphery_shocked, color='tab:red', linewidth=2, label='Periphery (Shocked Path)')

ax1.set_title(r"Comparison of Wealth Inequality Paths ($Gini$)", fontsize=14)
ax1.set_ylabel("Gini Coefficient")
ax1.legend(loc='upper right', frameon=True)

# --- Bottom Plot: IRF ---
ax2.plot(t_axis, irf_core, color='tab:blue', linewidth=2.5, label=r'$\Delta$ Gini Core')
ax2.plot(t_axis, irf_periphery, color='tab:red', linewidth=2.5, label=r'$\Delta$ Gini Periphery')

# Add a baseline at zero
ax2.axhline(0, color='black', linewidth=1, linestyle='-')

ax2.set_title(r"Impulse Response Function: $Gini_{shocked} - Gini_{base}$", fontsize=14)
ax2.set_ylabel("Change in Gini")
ax2.set_xlabel("Time (Periods)")
ax2.legend(loc='upper right', frameon=True)

plt.tight_layout()
plt.show()
 #%%

# =========================
# SHOCKING PERIPHERY
# =========================
deviations2 = macro_series.simulate_irf(T=T, shock_region="periphery", shock_size=shock_size)
w_core_path2 = w_core * (1 + deviations2[0])
w_periphery_path2 = w_periphery * (1 + deviations2[1])

gini_core_shocked2 = households_core.simulate_gini_path(r, w_core_path2, initial_dist=a_core, H=H_na, T=T)

gini_periphery_shocked2 = households_periphery.simulate_gini_path(r, w_periphery_path2, initial_dist=a_periphery, H=H_na, T=T) 

#%%
MC_core_base, _  = tools.monte_carlo_gini(households_core, r, w_path=w_core_path_SS, H=H_na, T=T, S=150)
MC_core_2, _ = tools.monte_carlo_gini(households_core, r, w_path=w_core_path2, H=H_na, T=T, S=150)
#%%

# plot shock 2 minus shock 1 for both core and periphery
diff_gini_core = MC_core_2 - MC_core_base
diff_gini_periphery = gini_periphery_shocked2 - gini_periphery_shocked

# 2. Setup Figure
plt.style.use('seaborn-v0_8-muted')
fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
t_axis = np.arange(len(diff_gini_core))

# 3. Plot the differences
ax.plot(t_axis, diff_gini_core, color='tab:blue', linewidth=2.5, 
        label='Core Gini: (Shock Periphery - Shock Core)')

ax.plot(t_axis, diff_gini_periphery, color='tab:red', linewidth=2.5, 
        label='Periphery Gini: (Shock Periphery - Shock Core)')

# 4. Formatting for clarity
ax.axhline(0, color='black', linewidth=1.2, linestyle='--') # Zero line is crucial here
ax.set_title("Relative Impact of Shock Location on Inequality", fontsize=14, pad=15)
ax.set_ylabel("Difference in Gini Points", fontsize=12)
ax.set_xlabel("Time (Periods)", fontsize=12)

# Adding a shaded area to highlight which shock is more 'inequality-inducing'
ax.fill_between(t_axis, diff_gini_periphery, 0, where=(diff_gini_periphery > 0), 
                color='tab:red', alpha=0.1)
ax.fill_between(t_axis, diff_gini_periphery, 0, where=(diff_gini_periphery < 0), 
                color='tab:blue', alpha=0.1)

ax.legend(loc='best', frameon=True, fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()