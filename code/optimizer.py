# -*- coding: utf-8 -*-
"""
Created on Fri Jan 23 14:15:01 2026

@author: lseibert
"""

import numpy as np
from functools import wraps
from scipy.interpolate import interp1d
import time

def time_method(func):
    """Decorator to time class methods."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        start = time.perf_counter()
        result = func(self, *args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f" {func.__name__} took {elapsed:.3f} seconds")
        return result
    return wrapper


class Economy:
    '''
    Class to store key values of th economy
    '''
    def __init__(self, beta, sigma, states, Q):
        self.beta = beta
        self.sigma = sigma
        self.states = states
        self.Q = Q
        
class Household:
    '''
    Class to store key values of the household
    '''
    def __init__(self, Economy, na, b):
        '''
        Initialize hosuehold with values of the economy and savings setting

        Parameters
        ----------
        Economy : class
            Economic variables from class Economy.
        sigma : float
            Coefficient of relative risk aversion.
        na : int
            Number of grid points.
        b : float
            Allowed number of debt for the houesholds.
            Budget constraint of next period's assets in form a'>= -b.

        Returns
        -------
        None.

        '''
        self.econ = Economy
        self.b = b      
        self.agrid = construct_grid(
                        b=b,
                        max_state=np.max(self.econ.states),
                        na=na
                        )
        print(f"Defined asset grid on {np.min(self.agrid):.2f} - {np.max(self.agrid):.2f}")
        print(f"For more info, access it class {type(self).__name__} stored youre_class_name.agrid")
        
    @time_method
    def _solve_EGM(self, r, w, max_iter=1000, tol=1e-8, lamb = 0.7, verbose=False):
        '''
        Solve Aiyagari household problem with EGM method.

        Parameters
        ----------
        r : float
            Exogeneous interest rate.
        w : float
            Exogeneous wage.
        max_iter : int, optional
            Maxixum numbers of iteration in the EGM solver. The default is 1000.
        tol : float, optional
            Tolerance for policy to converge. The default is 1e-8.
        lamb : float, optional
            Dampening factor. The default is 0.7.
        verbose : boolean, optional
            Print statements. The default is False.

        Returns
        -------
        Standard return is saving optimal cash at hand -> assets next periods mapping to class OptimalPolicy.
        Policy funciton is saved as self.policy(xgrid, agrid) to get. For more detials inspect class OptimalPolicy!
        '''
        na = len(self.agrid)
        nz = len(self.econ.states)
        k_policy = np.zeros((nz, na))
        k_policy[:] = self.agrid * 0.5
        k_policy_new = np.zeros((nz, na))
        xgrid = comp_ressources(self.agrid, r, w, self.econ.states)

        for i in range(max_iter):
            c_next = xgrid - k_policy
            # compute implied consumption today
            # compute C_t**-sigma = E(c-t+1**-sigma) * (R+beta)
            marginal_utility = Emu(c_next, r, self.econ.sigma, self.econ.beta, self.econ.Q)
            # invert to get c_today!
            c_today = (marginal_utility) ** (-1/self.econ.sigma)
            # compute implied cash at hand
            cash_imp = c_today + self.agrid
            for z_i in range(nz):

                # Interpolate: resources_today -> capital_tomorrow
                policy_interp = interp1d(cash_imp[z_i, :], self.agrid, kind='linear',
                       bounds_error=False, fill_value="extrapolate") 
                k_policy_new[z_i, :] = policy_interp(xgrid[z_i, :])
                binding_indices = xgrid[z_i, :] < cash_imp[z_i, 0]
                k_policy_new[z_i, binding_indices] = -self.b           
            # lowest endogenous resource point (cash_imp[z_i, 0]) means 
            # the constraint is binding. cash_imp is unconstraiend since agrid[0] == -b

            # Check convergence
            diff = np.linalg.norm(k_policy_new - k_policy)
            if diff < (np.linalg.norm(k_policy))*tol:
                # store class object in self.policy
                self.policy = OptimalPolicy(cash_imp, self.agrid, self.b)
                if verbose:
                    print(f"Converged in {i} iterations")
                    print("Policy saved under self.policy!")
                return None
            k_policy = k_policy_new * lamb + k_policy * (1 - lamb)
        print("Max iterations reached")
        print(f"required value: {(1+np.linalg.norm(k_policy))*tol}")
        raise ValueError("EGM did not converge!")

        
    def _simulate_stationary_distribution(
            self,
            r,
            w,
            verbose=False,
            H=100, 
            T=500, 
            burn_in=200
        ):
        """
        Monte Carlo simulation of invariant distribution in Aiyagari model.
        To be exchanged with histogram iteration. WIP
        """
        nz = len(self.econ.states)
        
        # Initialize assets & shocks
        a = np.random.choice(self.agrid, size=H) 
        z_idx = np.random.choice(nz, size=H)
        
        # Pre-calculate Cumulative Probability for Markov transition
        Q_cumsum = np.cumsum(self.econ.Q, axis=1)
        K_series = []
        if verbose:
            print(f"Simulating {H} agents for {T} periods...")
    
        # Simulation Loop
        for t in range(T):
            
            # Vectorized Markov Transition
            r_draws = np.random.rand(H)
            
            # for every agent i, find where r_draws[i] fits in Q_cumsum[z_idx[i]]
            # Look up the specific CDF row for each agent
            current_cdfs = Q_cumsum[z_idx] # shape (H, nz)
            
            # argmax returns the first True index because True > False
            z_next = (current_cdfs > r_draws[:, None]).argmax(axis=1)
            z_idx = z_next
        
            # policy maps cash -> assets
            cash = (1+r)*a + w*self.econ.states[z_idx]
            a_prime = self.policy.simulate_policy(z_idx, cash) # or use self.policy.__call__() -> USP of __call__
    
            a = a_prime
        
            if t > burn_in:
                K_series.append(np.mean(a))
        
        # Return the final distribution AND the smoothed aggregate capital
        mean_K = np.mean(K_series) 
        return a, mean_K
    
    
    def simulate_gini_path(self, r, w_series, initial_dist = None, seed=123, H=1000, T=500):
        """
        w_series: a numpy array or list of length T
        r: can be a constant or a vector of length T
        """
        
        # random number generator for consistency across simulations
        rng = np.random.RandomState(seed)
        nz = len(self.econ.states)
        
        # 2. Initialize Assets
        if initial_dist is not None:
            # Bootstrap: Draw H agents from the provided stationary distribution
            a = rng.choice(initial_dist, size=H, replace=True)
        else:
            # Fallback if no distribution provided
            a = rng.choice(self.agrid, size=H)
            
        z_idx = rng.choice(nz, size=H)
        Q_cumsum = np.cumsum(self.econ.Q, axis=1)
        
        gini_series = []
        
        # Ensure r is treatable as an array to handle both constant and fluctuating cases
        r_series = np.full(T, r) if np.isscalar(r) else r
    
        for t in range(T):
            # Update individual shocks
            r_draws = rng.rand(H)
            z_idx = (Q_cumsum[z_idx] > r_draws[:, None]).argmax(axis=1)
        
            # Extract aggregate values for this period
            current_w = w_series[t]
            current_r = r_series[t]
    
            # Calculate Policy update
            # Income = (1+r)*assets + wage * individual_productivity
            cash = (1 + current_r) * a + current_w * self.econ.states[z_idx]
            a = self.policy.simulate_policy(z_idx, cash)
    
            # Record Inequality
            gini_series.append(calculate_gini(a))
            # potentially store also net savings in periods to look at deviations there
            # 3d plot of assets over time over density
        return np.array(gini_series)
    

def Emu(c, r, sigma, beta, Q):
    # CRITICAL: Consumption cannot be zero or negative
    c_safe = np.maximum(c, 1e-10) 
    mu = c_safe**(-sigma)
    return Q @ (mu * (1 + r) * beta)
 
    
def comp_ressources(agrid, r, w, states):
    '''
    Compute cash at hand in the aiyagari model

    Parameters
    ----------
    agrid : 1d-array
        asset grid.
    r : float
        Interest rate.
    w : float
        Wage.
    states : 1d-array
        Array with state values of discretized markov chain.

    Returns
    -------
    2d object
        Cash at hand in shape (nz, na).

    '''
    return (1+r) * agrid[None, :] + w * states[:, None]

def calculate_gini(x):
    # Mean absolute difference
    n = len(x)
    x = np.sort(x)
    index = np.arange(1, n + 1)
    return ((2 * index - n - 1) * x).sum() / (n * x.sum())


def construct_grid(b, max_state, na, w_ref=5, zeta=0.15):
    '''
    Return a power grid that approximately matches the target asset possibilites of the agents.
    Upper bound determined by 10 * maximum savings

    Parameters
    ----------
    b : TYPE
        DESCRIPTION.
    w_ref : TYPE
        Reference value for the wage. Default is 2
    max_state : float
        Highest value of discretized markov shock.
    na : Int
        Number of grid points.
    zeta : TYPE, optional
        DESCRIPTION. The default is 0.15.

    Returns
    -------
    TYPE
        1d-array with asset grid.

    '''
    a_max = 1* w_ref * max_state 
    s = np.linspace(0, 1, na)
    zeta = 0.15
    return -b + (a_max + b) * s**(1/zeta)


class OptimalPolicy:
    def __init__(self, cash, a_prime, b_constraint):
        """
        cash: The endogenous resource grid computed by EGM (nz, na)
        a_prime: The fixed grid of assets chosen for tomorrow (1D array)
        b_constraint: The borrowing limit (positive number, e.g., 0.0 or 2.0)
                      The constraint is a' >= -b_constraint
        """
        self.a_prime = a_prime
        self.nz = cash.shape[0]
        self.na = cash.shape[1]
        self.b_constraint = b_constraint
        self.interpolators = []
        if a_prime.ndim != 1:
            raise ValueError("a_prime (Y-axis) must be a 1D array (e.g., agrid).")
        for z_i in range(self.nz):
            # We map Resources (x) -> Assets (y)
            f = interp1d(
                cash[z_i, :],  # x: endogenous resources
                a_prime,       # y: chosen assets
                kind='linear',
                bounds_error=False,
                fill_value="extrapolate"
            )
            self.interpolators.append(f) # check interpolation constraitn when applying interpolation

    def __call__(self, xgrid):
        """
        The standard call method: Looks up the policy for every point in a 2D Cash-on-Hand grid.
        This is primarily for TESTING, VISUALIZATION, and GRID LOOKUP.
        
        Parameters:
            xgrid: The (nz, na) Cash-on-Hand matrix.
                   
        Returns:
            a_prime_matrix: The resulting (nz, na) matrix of chosen assets (a').
        """
        if xgrid.shape != (self.nz, self.na):
            raise ValueError(f"Input xgrid must be of shape ({self.nz}, {self.na}).")
            
        # Flattening to 1d inputs
        cash_flat = xgrid.flatten()
        z_idx_flat = np.repeat(np.arange(self.nz), self.na)

        # internally call optimizer with grid values and corresponding states
        a_prime_flat = self.simulate_policy(z_idx_flat, cash_flat)

        # reshape into Shape (nz, na)
        a_prime_matrix = a_prime_flat.reshape(self.nz, self.na)
        
        return a_prime_matrix

    def simulate_policy(self, z_idx, current_cash):
        """
        Optimized method for SIMULATION.
        Takes 1D vectors of agent states and cash-on-hand.
        
        Parameters:
            z_idx: 1D array of agent state indices (0, 1, 2, ...)
            current_cash: 1D array of agent cash-on-hand values.
            
        Returns:
            final_a_prime: 1D array of chosen assets (a').
        """
        # Ensure inputs are arrays, even if scalar
        current_cash = np.atleast_1d(current_cash)
        z_idx = np.atleast_1d(z_idx)
        a_prime = np.zeros_like(current_cash, dtype=np.float64)

        # Vectorized Interpolation
        for z in range(self.nz):
            mask = (z_idx == z)
            
            if np.any(mask):
                # Interpolate cash for agents in state z
                a_prime[mask] = self.interpolators[z](current_cash[mask])

        # Enforce Constraint!
        final_a_prime = np.maximum(a_prime, -self.b_constraint)
        
        # Return item if original input was scalar, else return array
        return final_a_prime.item() if final_a_prime.size == 1 and z_idx.size == 1 else final_a_prime

        