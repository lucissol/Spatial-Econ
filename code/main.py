# -*- coding: utf-8 -*-
"""
Created on Sat Jan 17 13:24:23 2026

@author: lseibert
"""
from optimizer import ConsumptionSavingsModel
import numpy as np
from dy_simulation import Region, calculate_gini
from macro_var import SpatialMacroModel
import matplotlib.pyplot as plt

# ==========================================
# CONFIG
# ==========================================

############### MACRO LAYER
T = 15
phi_regions = [0.5, 0.5]
lambda_regions = [1.2, 0.1]
rho = 0.5
W = np.array([[0, 0.1],
              [0.9, 0]])
 
############### AGENT OPTIMIZATION
# Power-spaced grid
na = 1000
s = np.linspace(0, 1, na)
zeta = 0.3
agrid_pow = 0 + (25 - 0) * s**(1/zeta)
x_pow = 1 + (25 - 1) * s**(1/zeta)

# parameter
beta = 0.96
r = 0.0
y = 1.0

# ==========================================
# MAIN CODE
# ==========================================

# define macro environment
model = SpatialMacroModel(
    T=T, 
    phi_regions=phi_regions,     # A persistent, B not
    lambda_regions=lambda_regions,  # A follows nation, B doesn't
    rho=rho                   #  spillover parameter
)

Y = model.simulate_path()
Y_dev = model.simulate_irf()
# micro optimization
model = ConsumptionSavingsModel(beta=beta, r=r, y=y, 
                                state_grid=x_pow, 
                                choice_grid=agrid_pow)

_, opt_savings, opt_consumption = model.solve(verbose=False)

a_policy = lambda x: np.interp(x, x_pow, opt_savings)

############### SIMULATIONS
reg_A = Region(name = 'A',
               macro_series = Y['Region_A'],
               policy = a_policy,
               initial_dist_type='pareto')

reg_B = Region(name = 'B',
               macro_series = Y['Region_B'],
               policy = a_policy,
               initial_dist_type='uniform')

reg_A.init_population(100)
reg_B.init_population(100)

# execute dynamic simulation
for t in range(T-1):
    reg_A.step()
    reg_B.step()
    
    
##########################################
# COUNTERFACTUAL
##########################################

############### SIMULATIONS
reg_ACF = Region(name = 'A',
               macro_series = Y['Region_A'] * np.exp(Y_dev['Region_A']),
               policy = a_policy,
               initial_dist_type='pareto')

reg_BCF = Region(name = 'B',
               macro_series = Y['Region_B'] * np.exp(Y_dev['Region_B']),
               policy = a_policy,
               initial_dist_type='uniform')

reg_ACF.init_population(100)
reg_BCF.init_population(100)

# execute dynamic simulation
for t in range(T-1):
    reg_ACF.step()
    reg_BCF.step()

#%%

# Loop through all agents and grab the first item in their history
initial_wealth_values = [agent.history[0] for agent in reg_B.population]

# 1. Setup the figure
plt.figure(figsize=(10, 6))

# 2. Create the Histogram
# bins: How many 'buckets' to divide the data into
# alpha: Transparency
# density: If True, area under curve sums to 1 (good for comparing different regions)
plt.hist(initial_wealth_values, bins=1, color='skyblue', edgecolor='black', alpha=0.7, density=False)

# 3. Add a Mean Line for context
mean_wealth = np.mean(initial_wealth_values)
plt.axvline(mean_wealth, color='red', linestyle='dashed', linewidth=2, label=f'Mean Wealth: {mean_wealth:.2f}')

# 4. Labels and Styling
plt.title(f'Initial Wealth Distribution (Time 0) - {reg_A.name}')
plt.xlabel('Initial Wealth ($x_0$)')
plt.ylabel('Density (Frequency)')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 5. Show
plt.show()


#%%
gini_A = [calculate_gini([a.history[t] for a in reg_A.population]) for t in range(T)]
gini_B = [calculate_gini([a.history[t] for a in reg_B.population]) for t in range(T)]
gini_ACF = [calculate_gini([a.history[t] for a in reg_ACF.population]) for t in range(T)]
gini_BCF = [calculate_gini([a.history[t] for a in reg_BCF.population]) for t in range(T)]


# plot differences
g_A = np.array(gini_A)
g_B = np.array(gini_B)
g_ACF = np.array(gini_ACF)
g_BCF = np.array(gini_BCF)

# 2. Calculate the Difference (Shocked - Counterfactual)
# Positive value = Shock increased inequality
# Negative value = Shock decreased inequality
diff_A = g_ACF - g_A
diff_B = g_BCF - g_B

# 3. Plot
plt.figure(figsize=(10, 6))

plt.plot(diff_A, label='Region A (Difference)', color='blue', linewidth=2)
plt.plot(diff_B, label='Region B (Difference)', color='red', linewidth=2, linestyle='--')

# Zero line (No difference reference)
plt.axhline(0, color='black', linewidth=1, linestyle=':', alpha=0.8)

# Styling
plt.title('Net Impact of Shock on Inequality\n(Shocked Gini - Counterfactual Gini)')
plt.ylabel('Difference in Gini Points')
plt.xlabel('Time (Periods)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.show()

#%%

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Wealth Distribution (Histogram at T=0 vs T=End)
final_wealth_pareto = [a.wealth for a in reg_ACF.population]
ax1.hist(final_wealth_pareto, bins=20, alpha=0.5, label='Pareto End State', density=True)
ax1.set_title("Final Wealth Distribution (Pareto Region)")
ax1.set_xlabel("Wealth")
ax1.legend()

# Plot 2: Inequality Evolution (Gini)
ax2.plot(gini_ACF, label='Pareto Initial', linewidth=2)
ax2.plot(gini_BCF, label='Uniform Initial', linewidth=2, linestyle='--')
ax2.set_title("Evolution of Inequality (Gini Coefficient)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Gini (0=Equal, 1=Unequal)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#%%
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Wealth Distribution (Histogram at T=0 vs T=End)
final_wealth_pareto = [a.wealth for a in reg_A.population]
ax1.hist(final_wealth_pareto, bins=20, alpha=0.5, label='Pareto End State', density=True)
ax1.set_title("Final Wealth Distribution (Pareto Region)")
ax1.set_xlabel("Wealth")
ax1.legend()

# Plot 2: Inequality Evolution (Gini)
ax2.plot(gini_A, label='Pareto Initial', linewidth=2)
ax2.plot(gini_B, label='Uniform Initial', linewidth=2, linestyle='--')
ax2.set_title("Evolution of Inequality (Gini Coefficient)")
ax2.set_xlabel("Time")
ax2.set_ylabel("Gini (0=Equal, 1=Unequal)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
#%%
# Loop through all agents and grab the first item in their history
initial_wealth_values = [agent.history[-1] for agent in reg_A.population]

# 1. Setup the figure
plt.figure(figsize=(10, 6))

# 2. Create the Histogram
# bins: How many 'buckets' to divide the data into
# alpha: Transparency
# density: If True, area under curve sums to 1 (good for comparing different regions)
plt.hist(initial_wealth_values, bins=30, color='skyblue', edgecolor='black', alpha=0.7, density=True)

# 3. Add a Mean Line for context
mean_wealth = np.mean(initial_wealth_values)
plt.axvline(mean_wealth, color='red', linestyle='dashed', linewidth=2, label=f'Mean Wealth: {mean_wealth:.2f}')

# 4. Labels and Styling
plt.title(f'Initial Wealth Distribution (Time 0) - {reg_A.name}')
plt.xlabel('Initial Wealth ($x_0$)')
plt.ylabel('Density (Frequency)')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 5. Show
plt.show()
