# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 11:45:46 2026

@author: lseibert
"""

# Storing old functions and code for future implementation into the main file
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
shock_size = -0.1 # in % deviations
shock_region = "core" # "core" or "periphery"

deviations = macro_series.simulate_irf(T=T, shock_region=shock_region, shock_size=shock_size)


# shock at t = 1
w_core_path = w_core * (1 + deviations[0])
w_periphery_path = w_periphery * (1 + deviations[1])


#%%
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
