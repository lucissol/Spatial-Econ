# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:07:53 2026

@author: lseibert
"""

import numpy as np
from joblib import Parallel, delayed
import matplotlib.pyplot as plt

from config import config
import optimizer
import stoch_tools
import macro_var
import helper as tools
import worker_tools

def run_scenario(model_core, model_periphery, shock_size, shock_region, config=None, plots=True):
    """ 
    Compare IRF of specified shock. Generates micro response on a macro wage shock for a shocked and non-shocked path. 
    Returns shock path and Gini over time for both core and periphery.
    
    model: Class
        Class from optimizer households. Contains the policy of the Aiyagari Economy
    """
    # ========================
    # Macro Layer
    # ========================
    
    # getattr(object, "attribute_name", default_value)
    T             = getattr(config, 'T', 100)
    rho_core      = getattr(config, 'rho_core', 0.7)
    rho_periphery = getattr(config, 'rho_periphery', 0.6)
    rho_c_p       = getattr(config, 'rho_c_p', 0.5)
    rho_p_c       = getattr(config, 'rho_p_c', 0.2)
    w_core        = getattr(config, 'w_core', 1)
    w_periphery   = getattr(config, 'w_periphery', 0.8)
    H_sim         = getattr(config, 'H_sim', 10_000)
    T_sim         = getattr(config, 'T_sim', 500)
    burn_in       = getattr(config, 'burnin_sim', 150)
    H_na          = getattr(config, 'H_na', 1_500)
    S_na          = getattr(config, 'S_na', 150)
    beta          = getattr(config, 'S_na', 0.96)
    r             = getattr(config, 'r', ((1/beta) - 1) * 0.95)
    figs = {}
    
    
    macro_series = macro_var.SpatialMacroEconomy(rho_core=rho_core,
                                                 rho_periph=rho_periphery,
                                                 spill_c_to_p=rho_c_p,
                                                 spill_p_to_c=rho_p_c
                                                 )

    # steady state paths
    w_core_path_SS = np.ones(T) * w_core
    w_periphery_path_SS = np.ones(T) * w_periphery

    deviations = macro_series.simulate_irf(T=T, shock_region=shock_region, shock_size=shock_size)

    # shock at t = 1
    w_core_path = w_core * (1 + deviations[0])
    w_periphery_path = w_periphery * (1 + deviations[1])


    if plots == True:
        # plot the time series using python style sheets
        plt.style.use('seaborn-v0_8-paper') 
        fig, ax = plt.subplots(figsize=(10, 5), dpi=100)
    
        # Plotting the Path Series
        ax.plot(w_core_path, label='Shocked wage path in the core', 
                color='#1f77b4', linewidth=2, linestyle='-')
        ax.plot(w_periphery_path, label='Shocked wage path in the periphery', 
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
        
    # =======================
    # Invariant Distribution of baseline
    # =======================
    a_core, _ = model_core._simulate_stationary_distribution(r=r,
                                                                  w=w_core,
                                                                  T=T_sim,
                                                                  H=H_sim,
                                                                  burn_in=burn_in
                                                                  )


    a_periphery, _ = model_periphery._simulate_stationary_distribution(r=r,
                                                                            w=w_periphery,
                                                                            T=T_sim,
                                                                            H=H_sim,
                                                                            burn_in=burn_in
                                                                            )
    # distribution plots  
    if plots:
        plt.style.use('ggplot')
        fig1, ax = plt.subplots(figsize=(12, 6), dpi=100)
        # We cap the range at a reasonable multiple of the mean to see the tail clearly
        n_bins = 100
        plot_range = (-model_core.b, np.percentile(a_core, 99)) 
   
        # Plot Core
        ax.hist(a_core, bins=n_bins, range=plot_range, density=True, 
                alpha=0.5, color='steelblue', label='Core Region ($w=1.0$)', 
                edgecolor='white', linewidth=0.5)
   
        # Plot Periphery
        ax.hist(a_periphery, bins=n_bins, range=plot_range, density=True, 
                alpha=0.5, color='indianred', label='Periphery Region ($w=0.8$)', 
                edgecolor='white', linewidth=0.5)
   
        # Styling
        ax.set_title("Stationary Wealth Distributions in SS: Core vs. Periphery", fontsize=15, fontweight='bold')
        ax.set_xlabel("Assets ($a$)", fontsize=12)
        ax.set_ylabel("Probability Density", fontsize=12)
   
        # Mark  Borrowing Limit
        ax.axvline(-model_core.b, color='black', linestyle='--', alpha=0.7, label=f'Borrowing Limit ($-b$)')
        ax.legend(loc='upper right', frameon=True, facecolor='white')
        ax.grid(axis='y', linestyle='--', alpha=0.7)
   
        plt.tight_layout()
        figs['distribution'] = fig1
        plt.show()

    # =====================
    # Scenario Analysis
    # =====================
    
    
    import concurrent.futures
    print("Running 4 processes in parallel - this may take some time...")
    # 1. Define your tasks in a list of tuples (function_args)
    tasks = [
        (model_core, r, w_core_path_SS, H_na, T, S_na, a_core),
        (model_core, r, w_core_path, H_na, T, S_na, a_core),
        (model_periphery, r, w_periphery_path_SS, H_na, T, S_na, a_periphery),
        (model_periphery, r, w_periphery_path, H_na, T, S_na, a_periphery)
    ]
    
    # 2. Use a ProcessPoolExecutor to run them in parallel
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        # We use a list comprehension to submit the tasks
        # Each 'star' unpacks the tuple into the function arguments
        futures = [executor.submit(worker_tools.monte_carlo_worker, *task) for task in tasks]
        
        # Collect results as they finish
        results = [f.result() for f in futures]

    # 3. Assign back to your variables
    MC_core_base, _      = results[0]
    MC_core_shock, _     = results[1]
    MC_periphery_base, _  = results[2]
    MC_periphery_shock, _ = results[3]
    
    print("Succesful run of Bootstrapp procedure")
    
    #MC_core_base, _  = tools.monte_carlo_gini(model_core, r, w_path=w_core_path_SS, H=H_na, T=T, S=150)
    #MC_core_shock, _ = tools.monte_carlo_gini(model_core, r, w_path=w_core_path, H=H_na, T=T, S=150)
    #MC_periphery_base, _  = tools.monte_carlo_gini(model_periphery, r, w_path=w_periphery_path_SS, H=H_na, T=T, S=150)
    #MC_periphery_shock, _ = tools.monte_carlo_gini(model_periphery, r, w_path=w_periphery_path, H=H_na, T=T, S=150)
    
    # ==============================
    # Results
    # ==============================
    
    if plots:
        # plot shock minus shock 1 for both core and periphery
        diff_gini_core = MC_core_shock - MC_core_base
        diff_gini_periphery = MC_periphery_shock - MC_periphery_base
    
        # 2. Setup Figure
        plt.style.use('seaborn-v0_8-muted')
        fig2, ax = plt.subplots(figsize=(10, 6), dpi=100)
        t_axis = np.arange(len(diff_gini_core))
    
        # 3. Plot the differences
        ax.plot(t_axis, diff_gini_core, color='tab:blue', linewidth=2.5, 
                label=f'Core Gini: (Periphery - shocked region: {shock_region})')
    
        ax.plot(t_axis, diff_gini_periphery, color='tab:red', linewidth=2.5, 
                label=f'Periphery Gini: (Periphery - shocked region: {shock_region})')
    
        # 4. Formatting for clarity
        ax.axhline(0, color='black', linewidth=1.2, linestyle='--') # Zero line is crucial here
        ax.set_title(f"Relative Impact of Shock Location on Inequality - shock size: {shock_size}", fontsize=14, pad=15)
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
        figs['IRF'] = fig2

        plt.show()
        
        
        #plot 2
        plt.style.use('seaborn-v0_8-whitegrid')

        # figures
        fig3, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, dpi=100)
        t_axis = np.arange(len(MC_core_base))

        # --- Top Plot: Level Paths ---
        ax1.plot(t_axis, MC_core_base, color='tab:blue', linestyle='--', alpha=0.5, label='Core (Steady State)')
        ax1.plot(t_axis, MC_core_shock, color='tab:blue', linewidth=2, label='Core (Shocked Path)')

        ax1.plot(t_axis, MC_periphery_base, color='tab:red', linestyle='--', alpha=0.5, label='Periphery (Steady State)')
        ax1.plot(t_axis, MC_periphery_shock, color='tab:red', linewidth=2, label='Periphery (Shocked Path)')

        ax1.set_title(f"Comparison of Wealth Inequality Paths ($Gini$) - shocked region: {shock_region}", fontsize=14)
        ax1.set_ylabel("Gini Coefficient")
        ax1.legend(loc='upper right', frameon=True)

        # --- Bottom Plot: IRF ---
        ax2.plot(t_axis, diff_gini_core, color='tab:blue', linewidth=2.5, label=r'$\Delta$ Gini Core')
        ax2.plot(t_axis, diff_gini_periphery, color='tab:red', linewidth=2.5, label=r'$\Delta$ Gini Periphery')

        # Add a baseline at zero
        ax2.axhline(0, color='black', linewidth=1, linestyle='-')

        ax2.set_title(r"Impulse Response Function: $Gini_{shocked} - Gini_{base}$", fontsize=14)
        ax2.set_ylabel("Change in Gini")
        ax2.set_xlabel("Time (Periods)")
        ax2.legend(loc='upper right', frameon=True)

        plt.tight_layout()
        figs['IRF2'] = fig3

        plt.show()

    
    return diff_gini_core, diff_gini_periphery, figs


def monte_carlo_gini(model, r, w_path, H, T, S=50, initial_dist=None):
    """
    S: Number of simulations (trials)
    """
    all_paths = []
    
    for s in range(S):
        # We need a new seed or just let it use the global random state
        path = model.simulate_gini_path(r, w_path, H=H, T=T, initial_dist=initial_dist)
        all_paths.append(path)
        
    all_paths = np.array(all_paths)
    mean_path = np.mean(all_paths, axis=0)
    std_path = np.std(all_paths, axis=0)
    
    return mean_path, std_path


def plot_spatial_lorenz(a_core, a_periphery, b):
    plt.style.use('ggplot')
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    
    def get_lorenz_coords(assets):
        # 1. Sort and shift to wealth space (non-negative)
        y = np.sort(assets + b)
        n = len(y)
        
        # 2. Cumulative shares
        cum_wealth = np.cumsum(y) / np.sum(y)
        cum_pop = np.arange(1, n + 1) / n
        
        # 3. Force (0,0) starting point
        return np.append([0], cum_pop), np.append([0], cum_wealth)

    # Calculate coordinates
    x_core, y_core = get_lorenz_coords(a_core)
    x_peri, y_peri = get_lorenz_coords(a_periphery)
    
    # Plotting
    ax.plot(x_core, y_core, label='Core Region', color='steelblue', lw=2.5)
    ax.plot(x_peri, y_peri, label='Periphery Region', color='indianred', lw=2.5)
    
    # 45-degree line (Perfect Equality)
    ax.plot([0, 1], [0, 1], color='black', linestyle='--', alpha=0.6, label='Perfect Equality')
    
    # Formatting
    ax.set_title("Spatial Inequality: Lorenz Curves", fontsize=15, fontweight='bold')
    ax.set_xlabel("Cumulative Share of Population", fontsize=12)
    ax.set_ylabel("Cumulative Share of Wealth", fontsize=12)
    ax.legend(loc='upper left', frameon=True, facecolor='white')
    ax.set_aspect('equal') # Keep it square
    
    plt.tight_layout()
    plt.show()