# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 12:45:52 2026

@author: lseibert
"""

import numpy as np
import matplotlib.pyplot as plt


class Economy:
    def __init__(self, r=0.04, beta=0.95, policy=None):
        self.r = r
        self.beta = beta
        self.policy = policy

    def get_desired_savings(self, cash_on_hand):
        return self.policy(cash_on_hand)

# --- 2. THE CONTAINER (Region with Macro Constraint) ---
class Region:
    def __init__(self, name, macro_series, policy, initial_dist_type='uniform'):
        self.name = name
        self.macro_series = macro_series # List of total regional wealth values from macro layer
        self.dist_type = initial_dist_type
        
        self.economy = Economy(policy = policy)
        
        self.population = []
        self.time_step = 0

    def init_population(self, n_agents, wealth_scale=1.0):
        '''
        Distributes INITIAL wealth according to Pareto or Uniform rules.
        '''
        if self.dist_type == 'pareto':
            # Pareto: start with a skewed distribution (not extreme though)
            a, m = 3, 1.5 
            initial_wealths = (np.random.pareto(a, n_agents) + 1) * m
        else:
            # Uniform: Everyone is equal
            initial_wealths = np.random.uniform(0.5, 5.0, n_agents)
            
        # Normalize initial wealth so it matches the start of the Macro Series
        current_total = np.sum(initial_wealths)
        target_total = self.macro_series[0]
        scaling_factor = target_total / current_total
        
        normalized_wealths = initial_wealths * scaling_factor
        
        # Create Agents
        for i, w in enumerate(normalized_wealths):
            self.population.append(Agent(i, w))

    def step(self):
        '''
        Top-Down Consistency Step
        '''
        if self.time_step >= len(self.macro_series) - 1:
            return # End of simulation
            
        current_macro_target = self.macro_series[self.time_step + 1]
        
        # 1. Agents formulate DESIRED savings (Micro Level)
        # Note: They assume the economy's r holds true.
        total_desired_savings = 0
        desired_savings_vector = []
        
        for agent in self.population:
            # Calculate CoH based on previous wealth
            coh = (1 + self.economy.r) * agent.wealth
            
            # Agent decides how much to save
            a_star = self.economy.get_desired_savings(coh)
            
            desired_savings_vector.append(a_star)
            total_desired_savings += a_star
            
        # adjust the interest rate ex-post
        normalization_factor = current_macro_target / total_desired_savings
        
        # update Agents with REALIZED wealth
        for i, agent in enumerate(self.population):
            a_star = desired_savings_vector[i]

            realized_wealth = a_star * normalization_factor
            agent.update(realized_wealth)
            
        self.time_step += 1
        return normalization_factor

# --- 3. THE AGENT ---
class Agent:
    def __init__(self, uid, wealth):
        self.id = uid
        self.wealth = wealth
        self.history = [wealth]
        
    def update(self, new_wealth):
        self.wealth = new_wealth
        self.history.append(new_wealth)

def calculate_gini(wealths):
    sorted_w = np.sort(wealths)
    n = len(wealths)
    cum_w = np.cumsum(sorted_w)
    return (n + 1 - 2 * np.sum(cum_w) / cum_w[-1]) / n
