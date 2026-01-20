# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 08:39:38 2026

@author: lseibert
"""
import numpy as np
import pandas as pd
# ==========================================
# MACRO LAYER
# ==========================================

# rederive Spatial Macro Model of NiReMs
# config 
T = 50
# Note that there exist two regions A and B

# AR(1) national output factor model
# log normal process
phi_F = 0.9 # High persistence business cycle
eta_mu = 0 # Mean of the shock
eta_sigma = 0.2 # Volatility of the national shock

# --- Regional Parameters ---
# Two regions: A (Index 0) and B (Index 1)
phi_regions = np.array([0.5, 0.3])  # Auto-regression for A and B
lambda_regions = np.array([1.2, 0.2])

# Noise parameters
eps_mu = np.array([0, 0])
eps_sigma = np.array([0.05, 0.05])

# --- Spatial Configuration ---
rho = 0.4  # General strength of spillover

# Asymmetric Matrix:
# Row 0 (Reg A): affected by B (0.2) -> Weak link
# Row 1 (Reg B): affected by A (0.9) -> Strong link
W = np.array([
    [0.0, 0.1],  
    [0.9, 0.0]
])

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class SpatialMacroModel:
    def __init__(self, T=60, phi_F=0.9, phi_regions=[0.7, 0.3], 
                 lambda_regions=[1.2, 0.1], rho=0.5, W=None):
        """
        Initialize the Spatial Factor Model.
        
        Parameters:
        -----------
        T : int
            Time horizon for simulation.
        phi_F : float
            Persistence of the National Factor AR(1).
        phi_regions : list or np.array
            Persistence (AR) for each region [Reg A, Reg B].
        lambda_regions : list or np.array
            Sensitivity of each region to National Factor [Reg A, Reg B].
        rho : float
            Spatial autoregressive parameter (strength of spillover).
        W : np.array
            Spatial weight matrix (N x N). If None, defaults to asymmetric 2-region.
        """
        self.T = T
        self.phi_F = phi_F
        self.phi_regions = np.array(phi_regions)
        self.lambda_regions = np.array(lambda_regions)
        self.rho = rho
        
        # Default W: Asymmetric "Engine-Satellite" structure
        if W is None:
            self.W = np.array([[0.0, 0.05], 
                               [0.95, 0.0]])
        else:
            self.W = np.array(W)
            
        self.N = len(self.phi_regions)
        
        # Pre-compute the System Matrix A for the solver: (I - rho*W)
        self.I = np.eye(self.N)
        self.A = self.I - self.rho * self.W
        
        # Internal storage for results
        self.results = None
        self.last_run_type = None  # simulated 'path' or 'irf'

    def _solve_spatial_step(self, y_prev, f_curr):
        """
        Solves the spatial equilibrium for a single time step t.
        Equation: (I - rho*W) * y_t = phi*y_{t-1} + lambda*f_t
        """
        # b = Vector of exogenous forces hitting the regions
        b = (self.phi_regions * y_prev) + (self.lambda_regions * f_curr)
        
        # Solve the linear system A * y = b for y
        # depending on the spatial weight matrix this step can be very inefficient. Potentially change to sparse matrix implementation!
        y_curr = np.linalg.solve(self.A, b)
        
        return y_curr

    def simulate_path(self, base_level = 100,seed=123, sigma_F=0.1, sigma_R=0.05):
        """
        Runs a STOCHASTIC simulation with random shocks.
        Returns a DataFrame of Levels (assuming Log-Normal process).
        """
        np.random.seed(seed)
        
        f_log = np.zeros(self.T)
        y_log = np.zeros((self.T, self.N))
        
        # Loop
        for t in range(1, self.T):
            # 1. National Factor (AR + Noise)
            shock_F = np.random.normal(0, sigma_F)
            f_log[t] = self.phi_F * f_log[t-1] + shock_F
            
            # 2. Regional Solution (Deterministic part + Add Noise after solve)
            y_log[t] = self._solve_spatial_step(y_log[t-1], f_log[t]) + np.random.normal(0, sigma_R, self.N)
            
        # Store and Return
        self.results = pd.DataFrame({
            'Time': np.arange(self.T),
            'National_Factor': base_level * np.exp(f_log),  # Convert log -> level
            'Region_A': base_level * np.exp(y_log[:, 0]),
            'Region_B': base_level * np.exp(y_log[:, 1])
        })
        self.last_run_type = 'path'
        return self.results

    def simulate_irf(self, shock_magnitude=1.0, shock_target='national', shock_time=1):
        """
        Runs a DETERMINISTIC simulation (Impulse Response Function).
        No random noise.
        
        Parameters:
        -----------
        shock_magnitude : float
            Size of the shock (default 1.0).
        shock_target : str
            'national', 'region_A', 'region_B'
        """
        f_log = np.zeros(self.T)
        y_log = np.zeros((self.T, self.N))
        
        # Create Shock Vector
        shock_F_vec = np.zeros(self.T)
        shock_R_vec = np.zeros((self.T, self.N))
        
        if shock_target == 'national':
            shock_F_vec[shock_time] = shock_magnitude
        elif shock_target == 'region_A':
            # Add a direct shock to Region A's equation
            # (Requires modifying the solve step logic slightly to handle regional shocks)
            # For simplicity, we add it to the 'b' vector logic inside the loop
            pass 
        
        # Simulation Loop
        for t in range(1, self.T):
            # 1. National Factor (Decay + Manual Shock)
            f_log[t] = self.phi_F * f_log[t-1] + shock_F_vec[t]
            
            # 2. Regional Solution
            # We calculate equilibrium, then ADD specific regional shock if needed
            base_sol = self._solve_spatial_step(y_log[t-1], f_log[t])
            
            # Add specific regional shock if requested
            if t == shock_time:
                if shock_target == 'region_A':
                    # We inject the shock into the system. 
                    # Note: A shock to A transmits instantly to B via the solver!
                    # So we add it to the 'b' vector effectively. 
                    # b_new = b_old + shock
                    # y = A^-1 * (b + shock) = y_old + A^-1 * shock
                    shock_vec = np.array([shock_magnitude, 0])
                    spatial_impact = np.linalg.solve(self.A, shock_vec)
                    base_sol += spatial_impact
                elif shock_target == 'region_B':
                    shock_vec = np.array([0, shock_magnitude])
                    spatial_impact = np.linalg.solve(self.A, shock_vec)
                    base_sol += spatial_impact

            y_log[t] = base_sol

        # Store and Return (Keep as Log Deviations for IRF)
        self.results = pd.DataFrame({
            'Time': np.arange(self.T),
            'National_Factor': f_log,
            'Region_A': y_log[:, 0],
            'Region_B': y_log[:, 1]
        })
        self.last_run_type = 'irf'
        return self.results

    def plot(self):
        """
        Plots the stored results. Auto-detects if it's a Path or IRF.
        """
        if self.results is None:
            print("No data to plot. Run simulate_path() or simulate_irf() first.")
            return
        
        df = self.results
        plt.figure(figsize=(10, 6))
        
        if self.last_run_type == 'path':
            # Plot Levels (Business Cycle)
            plt.plot(df['Time'], df['National_Factor'], 'k--', alpha=0.6, label='National Factor')
            plt.plot(df['Time'], df['Region_A'], lw=2, label='Region A (Engine)')
            plt.plot(df['Time'], df['Region_B'], lw=2, label='Region B (Satellite)')
            plt.title("Stochastic Simulation: Output Levels")
            plt.ylabel("Output Level")
            
        elif self.last_run_type == 'irf':
            # Plot Deviations (IRF)
            plt.plot(df['Time'], df['National_Factor'], 'k:', label='Shock Input')
            plt.plot(df['Time'], df['Region_A'], lw=3, label='Region A Response')
            plt.plot(df['Time'], df['Region_B'], lw=3, ls='--', label='Region B Response')
            plt.axhline(0, color='black', lw=0.5)
            plt.title("Impulse Response Function (Deviations)")
            plt.ylabel("% Deviation")
            
        plt.xlabel("Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

# ==========================================
# USAGE EXAMPLES
# ==========================================

# 1. Instantiate the Model
model = SpatialMacroModel(
    T=50, 
    phi_regions=[0.7, 0.3],     # A persistent, B not
    lambda_regions=[1.2, 0.1],  # A follows nation, B doesn't
    rho=0.5                     # Strong spillover parameter
)

# 2. Run an IRF (National Shock) and Plot
print("--- Plotting National Shock IRF ---")
model.simulate_irf(shock_magnitude=1.0, shock_target='national')
model.plot()

# 3. Run a Stochastic Path (Business Cycle) and Plot
print("--- Plotting Business Cycle Path ---")
path_data = model.simulate_path(seed=42)
model.plot()

# 4. Run a Regional Shock IRF (Shock Region A only)
print("--- Plotting Regional Shock (A) IRF ---")
model.simulate_irf(shock_magnitude=1.0, shock_target='region_A')
model.plot()

# 5. Retrieve Data
print("\n--- First 5 rows of data ---")
print(model.results.head())