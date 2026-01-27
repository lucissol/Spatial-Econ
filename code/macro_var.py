# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 12:56:02 2026

@author: lseibert
"""

import numpy as np
import matplotlib.pyplot as plt

class SpatialMacroEconomy:
    def __init__(self, rho_core=0.8, rho_periph=0.7, spill_c_to_p=0.15, spill_p_to_c=0.05):
        """
        Initializes the 2-Region Spatial VAR.
        
        Parameters:
        rho_core (float): Persistence of Core economy (0 to 1)
        rho_periph (float): Persistence of Periphery economy
        spill_c_to_p (float): How much Core affects Periphery (Spatial Lag)
        spill_p_to_c (float): How much Periphery affects Core
        """
        # The Transition Matrix (Phi)
        # [[Effect of C on C,  Effect of P on C],
        #  [Effect of C on P,  Effect of P on P]]
        self.Phi = np.array([
            [rho_core,      spill_p_to_c],
            [spill_c_to_p,  rho_periph]
        ])
        
        # Check stability (Eigenvalues must be < 1)
        eigvals = np.linalg.eigvals(self.Phi)
        if np.max(np.abs(eigvals)) >= 1.0:
            print("WARNING: System is explosive! Reduce persistence or spillovers.")
            
    def simulate_irf(self, T=50, shock_region='Core', shock_size=-0.1):
        """
        Generates an Impulse Response Function (IRF).
        """
        # State Vector: 2 regions x T time steps
        # Row 0 = Core, Row 1 = Periphery
        X = np.zeros((2, T))
        
        # Apply Shock at t=1 (t=0 is pre-shock steady state)
        # 0 = Core, 1 = Periphery
        idx = 0 if shock_region == 'Core' else 1
        X[idx, 1] = shock_size
        
        # Iterate forward
        for t in range(2, T):
            # X_t = Phi * X_{t-1}
            X[:, t] = self.Phi @ X[:, t-1]
            
        return X

# --- TEST THE ENGINE ---
macro = SpatialMacroEconomy()

# Generate a "Recession in Periphery" Scenario
# We shock Periphery (Index 1) by -10% (log deviation -0.10)
irf_paths = macro.simulate_irf(T=50, shock_region='Core', shock_size=-0.050)

# Visualization
time = np.arange(50)
plt.figure(figsize=(10, 5))
plt.plot(time, irf_paths[0, :], label='Core (Spillover Effect)', linestyle='--')
plt.plot(time, irf_paths[1, :], label='Periphery (Direct Shock)', linewidth=2)
plt.axhline(0, color='black', linewidth=0.5)
plt.title("Macro Dynamics: -10% Shock to Periphery")
plt.ylabel("Log Deviation from Steady State")
plt.xlabel("Time (Quarters/Years)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
