# -*- coding: utf-8 -*-
"""
Created on Tue Jan 20 09:49:28 2026

@author: lseibert
"""

import numpy as np
import time

class ConsumptionSavingsModel:
    '''
    Solves the RANK consumption model for the optimal micro responses. 
    Note that this version implies no informations about the macro economy!
    '''
    
    def __init__(self, beta, r, y, state_grid, choice_grid=None):
        '''
        Parameters
        ----------
        beta : float
            Discount factor.
        r : float
            Interest rate (e.g., 0.04).
        y : float
            Constant labor income.
        state_grid : Vector
            Grid for Cash-on-Hand (x).
        choice_grid : Vector, optional
            Grid for Savings (a'). Defaults to state_grid if None.
        '''
        self.beta = beta
        self.r = r
        self.y = y
        self.x_grid = state_grid # Cash on Hand (x)
        self.a_grid = choice_grid if choice_grid is not None else state_grid # Savings (a)
        
        # Pre-compute the utility matrix for efficiency
        self.u_matrix = self._compute_utility_matrix()
        
    @staticmethod
    def _log_util(c):
        '''Log utility with safe lower bound'''
        return np.where(c > 1e-12, np.log(c), -1e10)
    
    def _compute_utility_matrix(self):
        '''
        Computes u(c) for all combinations of state x and choice a.
        Constraint from image (4): c = x - a
        '''
        # c[i, j] = x_grid[i] - a_grid[j]
        c = self.x_grid[:, None] - self.a_grid[None, :]

        # Apply constraint c > 0 (utility is -inf otherwise)
        utility = np.full_like(c, -1e10, dtype=float)
        mask = c > 0
        utility[mask] = self._log_util(c[mask])
        
        return utility

    def solve(self, Vguess=None, tol=1e-6, max_iter=1000, verbose=True):
        '''
        Solves the Bellman equation using Value Function Iteration with Interpolation.
        '''
        start = time.perf_counter()
        
        # 1. Initialize Guess
        if Vguess is None:
             Vguess = np.zeros(len(self.x_grid))
        V = Vguess.copy()
        
        # 2. Pre-calculate the Future State (x') for every choice (a')
        # Equation from Image: x' = (1+r)*a + y
        # We compute this once for the grid of choices
        x_next_values = (1 + self.r) * self.a_grid + self.y
        
        if verbose:
            print(f"Solving with r={self.r}, y={self.y}...")

        for it in range(max_iter):
            V_old = V.copy()
            
            # --- INTERPOLATION STEP ---
            # We need V(x') for every choice a'. 
            # Since x' falls between grid points, we interpolate V_old onto x_next_values.
            # np.interp(values_to_query, x_coordinates, y_values)
            V_next_interpolated = np.interp(x_next_values, self.x_grid, V_old)
            
            # --- BELLMAN MAXIMIZATION ---
            # RHS = u(c) + beta * V(x')
            # We broadcast V_next_interpolated to shape (1, n_choices) to add to u_matrix (n_states, n_choices)
            rhs = self.u_matrix + self.beta * V_next_interpolated[None, :]
            
            # Maximize over choices (columns)
            V = np.max(rhs, axis=1)
            policy_indices = np.argmax(rhs, axis=1)
            
            # --- CHECK CONVERGENCE ---
            # Using Sup-norm (max absolute difference)
            dist = np.max(np.abs(V - V_old))
            if dist < tol:
                end = time.perf_counter()
                
                # Retrieve optimal savings
                opt_savings = self.a_grid[policy_indices]
                
                # Calculate implied consumption
                opt_consumption = self.x_grid - opt_savings
                
                if verbose:
                    print(f"Converged in {it} iterations! Time: {end-start:.4f}s")
                return V, opt_savings, opt_consumption

        print("Warning: Max iterations reached.")
        opt_savings = self.a_grid[policy_indices]
        opt_consumption = self.x_grid - opt_savings
        return V, opt_savings, opt_consumption
    
#%%
import numpy as np
import matplotlib.pyplot as plt

# --- 1. SETUP GRIDS CORRECTLY ---
# Savings Grid (a): MUST start at 0 so poor agents can choose to save nothing.
a_grid = np.linspace(0.0, 10.0, 200)

# State Grid (x): Cash-on-Hand
# (Ideally covers the range [y, max_possible_wealth])
x_grid = np.linspace(0.1, 10.0, 200)

# Parameters
beta = 0.95
r = 0.04
y = 1.0

# --- 2. SOLVE WITH SEPARATE GRIDS ---
# We explicitly pass the choice_grid (a_grid)
model = ConsumptionSavingsModel(beta=beta, r=r, y=y, 
                                state_grid=x_grid, 
                                choice_grid=a_grid)

V, opt_savings, opt_consumption = model.solve(verbose=True)

# --- 3. VERIFY THE FIX ---
print("\n--- Policy Check (First 5 points) ---")
print(f"{'Cash-on-Hand (x)':<20} | {'Savings (a)':<15} | {'Consumption (c)':<15}")
print("-" * 56)
for i in range(5):
    print(f"{x_grid[i]:<20.4f} | {opt_savings[i]:<15.4f} | {opt_consumption[i]:<15.4f}")

# --- 4. PLOT ---
plt.figure(figsize=(8, 6))
plt.plot(x_grid, opt_savings, label="Savings Policy $a'(x)$", linewidth=2)
plt.plot(x_grid, x_grid, '--', color='gray', alpha=0.5, label="45-degree line (Save Everything)")
plt.title("Savings Policy Function")
plt.xlabel("Cash on Hand ($x$)")
plt.ylabel("Savings ($a'$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
#%%
na = 2000
s = np.linspace(0, 1, na)
zeta = 0.3
agrid_pow = 0 + (25 - 0) * s**(1/zeta)
x_pow = .1 + (25 - 1) * s**(1/zeta)

model = ConsumptionSavingsModel(beta=beta, r=r, y=y, 
                                state_grid=x_grid, 
                                choice_grid=agrid_pow)

V, opt_savings, opt_consumption = model.solve(verbose=True)

# --- 4. PLOT ---
plt.figure(figsize=(8, 6))
plt.plot(x_grid, opt_savings, label="Savings Policy $a'(x)$", linewidth=2)
plt.plot(x_grid, x_grid, '--', color='gray', alpha=0.5, label="45-degree line")
plt.title("Savings Policy Function")
plt.xlabel("Cash on Hand ($x$)")
plt.ylabel("Savings ($a'$)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()