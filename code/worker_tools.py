# -*- coding: utf-8 -*-
"""
Created on Thu Jan 29 11:38:30 2026

@author: lseibert
"""
import numpy as np

def monte_carlo_worker(model, r, w_path, H, T, S, initial_dist):
    """
    This is the only thing the parallel workers will run.
    It is a standalone function at the top level of its own file.
    """
    all_paths = []
    # Loop for S trials
    for s in range(S):
        # We pass a unique seed to each trial for randomness
        path = model.simulate_gini_path(r, w_path, H=H, T=T, 
                                        initial_dist=initial_dist, seed=s)
        all_paths.append(path)
    
    all_paths = np.array(all_paths)
    return np.mean(all_paths, axis=0), np.std(all_paths, axis=0)