"""
Utilities for measuring Signal-to-Noise Ratio (SNR) in optimizer states.
Specifically designed for analyzing Adam/AdamW optimizer's second moments.
"""
import os
import torch
import numpy as np
import pandas as pd
import pickle
from typing import Dict, Tuple, List, Union, Any
from itertools import combinations

def get_all_combinations(input_tuple: Tuple[int, ...]) -> List[Tuple[int, ...]]:
    """
    Get all possible combinations of dimensions for a given tensor shape.
    
    Args:
        input_tuple: Tuple of dimension indices
        
    Returns:
        List of all possible combinations of dimensions
    """
    all_combos = []
    for r in range(1, len(input_tuple) + 1):
        combos = [tuple(int(i) for i in c) for c in combinations(input_tuple, r)]
        all_combos.extend(combos)
    return all_combos


def estimate_snr(tensor: torch.Tensor) -> Tuple[Dict[Tuple, float], Dict[Tuple, float], Dict[Tuple, float]]:
    """Estimate SNR for all possible dimension combinations in a tensor."""
    snr_results = {}
    mean_sqr_results = {}
    vars_results = {}
    
    tensor_np = tensor.detach().cpu().float().numpy()
    ndim = tensor_np.ndim
    dims = np.arange(ndim, dtype = int)
    all_combos = get_all_combinations(dims)
    
    for axes in all_combos:
        mean_sqr = np.mean(tensor_np, axis = axes)**2
        variance = np.var(tensor_np, axis = axes)
        snr = np.divide(mean_sqr, variance) # mean_square / variance     

        # Always save mean_sqr and variance
        mean_sqr_results[axes] = float(mean_sqr.mean())
        vars_results[axes] = float(variance.mean())   
      
        # For SNR, use mean only where values are finite
        mask = np.isfinite(snr)
        if mask.any():
            snr_results[axes] = float(snr[mask].mean())
        else:
            snr_results[axes] = np.nan
    
    return snr_results, mean_sqr_results, vars_results


def compute_snr_metrics_slim(optimizer, model: torch.nn.Module, step: int) -> pd.DataFrame:
    """Compute SNR metrics for all parameters in optimizer state."""
    snr_results = []
    
    for group in optimizer.param_groups:
        param = group['params'][0]
        name = group['name']
        
        if param in optimizer.state:
            state = optimizer.state[param]
            if 'nu' in state:
                second_moment = state['nu']
                
                # Compute SNR for all possible compression dimensions
                snr_dict, mean_sqr_dict, vars_dict = estimate_snr(second_moment)
                
                # Create entries for all possible compression dimensions
                for axes in snr_dict.keys():
                    df = pd.DataFrame({
                        'step': [step],
                        'key': [name],
                        'shape': [str(tuple(second_moment.shape))],
                        'compress_dims': [str(tuple(sorted(axes)))],
                        'snr': [snr_dict[axes]],
                        'mean_sqr': [mean_sqr_dict[axes]],
                        'var': [vars_dict[axes]]
                    })
                    snr_results.append(df)

    return pd.concat(snr_results, ignore_index=True) if snr_results else pd.DataFrame()


class SNRTracker:
    """ Handles SNR measurement"""
    
    def __init__(self, start_step: int = 1, early_freq: int = 100, late_freq: int = 1000, early_step_threshold: int = 1000, save_dir: str = 'snr_data', save_format: str = 'csv'):
        
        """Initialize SNR tracker."""
        self.start_step = start_step
        self.early_freq = early_freq
        self.late_freq = late_freq
        self.early_step_threshold = early_step_threshold
        self.save_dir = save_dir
        self.save_format = save_format
        
        os.makedirs(self.save_dir, exist_ok = True)
        self.snr_results = []
        
    def should_measure(self, step: int) -> bool:
        """Determine if SNR should be measured at current step."""
        if step == self.start_step:
            return True
        if step < self.early_step_threshold:
            return step % self.early_freq == 0
        return step % self.late_freq == 0
    
    def measure_and_save(self, optimizer: torch.optim.Optimizer, model: torch.nn.Module, step: int) -> None:
        """ Measure SNR metrics."""
        snr_df = compute_snr_metrics_slim(optimizer, model, step = step)
        self.snr_results.append(snr_df)
    
    def save_all_results(self, filename: str) -> None:
        """Save all collected SNR results at the end of training."""
        if not self.snr_results:
            return
            
        combined_df = pd.concat(self.snr_results, ignore_index = True)
        
        save_path = os.path.join(self.save_dir, filename)
        combined_df.to_csv(save_path, index=False)