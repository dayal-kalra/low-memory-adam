import math
from typing import Iterable, Tuple, Union, Optional
import json

import torch
import torch.nn as nn
import torch.distributed as dist
# dist.init_process_group(backend='nccl')  # or 'gloo' for CPU
from torch.distributed._tensor import Replicate
device = 'cuda' if torch.cuda.is_available() else 'cpu'


import math
from typing import Iterable, Tuple, Union, Optional

DEFAULT_COMPRESSION_RULES = {
    # Attention layers
    "attention_query": (1,),  # fan_in 
    "attention_key": (1,),    # fan_in
    "attention_value": (0,),  # fan_out
    "attention_output": (0,), # fan_out
    
    # MLP layers
    "mlp_up": (0,),    # fan_out (marked with * in paper - inconsistent)
    "mlp_gate": (0,),  # fan_out (marked with * in paper - inconsistent)
    "mlp_down": (0,),  # fan_out
    
    # Special layers
    "token_embedding": (1,),       # fan_out
    "position_embedding": (1,),    # fan_out 
    "lm_head": (1,),               # fan_in
    "vision_first": (1,),          # fan_in
    "vision_classification": (1,), # fan_in (marked with * in paper - inconsistent)
    
    # No compression for normalization layers
    "norm": None
}

def get_rule_for_parameter(param_name, layer_map):
    """
    Determine compression rule for a single parameter based on layer type.
    
    Args:
        param_name (str): Name of the parameter
        layer_map (dict): Mapping of layer types to name patterns
        
    Returns:
        tuple or None: Compression dimensions or None if no compression
    """
    # Find layer type based on parameter name
    layer_type = None
    for lt, patterns in layer_map.items():
        if any(pattern in param_name for pattern in patterns):
            layer_type = lt
            break
            
    # Apply default compression rule
    if layer_type and layer_type in DEFAULT_COMPRESSION_RULES:
        return DEFAULT_COMPRESSION_RULES[layer_type]
    
    return None

class SlimAdamW(torch.optim.Optimizer):
    """
    Generalized SlimAdam which also takes into account KQV compressions
    * SlimAdam takes in a dictionary `axes` corresponding to each block
    """
    def __init__(
            self,
            named_parameters,
            *,
            lr: Union[float, torch.Tensor],
            betas: Tuple[float, float],
            eps: float,
            weight_decay: float,
            rules_json_path: Optional[str] = None,
            layer_map_path: Optional[str] = 'layer_map.json',
            verbose = True,
    ):
        # named model parameters 
        self.named_parameters = named_parameters

        self.compression_rules = {} 

        # Load any pre-determined compression rules
        compression_rules = {}
    
        # ---------- LOAD COMPRESSION CONFIGURATION ----------
        if rules_json_path:
            print("\n" + "="*100)
            print(f"Using explicit compression rules from: {rules_json_path}")
            print("="*100)

            with open(rules_json_path, 'r') as fi:
                self.compression_rules = json.load(fi)

            # convert compression rules from list to tuple
            self.compression_rules = {k: tuple(v) if v is not None else None for k, v in self.compression_rules.items()}

            # Remove unwanted prefix if present (for nanoGPT compatibility)
            unwanted_prefix = '_orig_mod.'
            for key, value in list(self.compression_rules.items()):
                if key.startswith(unwanted_prefix):
                    self.compression_rules[key[len(unwanted_prefix):]] = self.compression_rules.pop(key)

        elif layer_map_path:
            # default compression rules from Table 1 of the paper
            print("\n" + "="*100)
            print(f"Compression rules not provided, defaulting to recommended SlimAdam rules using: {layer_map_path}")
            print("="*100)
            
            # Load layer map for determining compression rules
            with open(layer_map_path, 'r') as f:    
                self.layer_map = json.load(f)
        else:
            # No compression
            print("\n" + "="*100)
            print(f"Compression rules and layer map both were not provided, defaulting to full Adam")
            print("="*100)


        ## Create optimizer groups  ###
        optim_groups = []
    
        # create parameter groups
        total_params = 0
        total_slim_params = 0

        for param_name, param in named_parameters:

            if not param.requires_grad:
                continue

            total_params += param.numel()

            group_config = {} # dictionary for each parameter
            group_config['name'] = param_name
            group_config['params'] = param

            # assign compression axes
            if rules_json_path:
                # Use preloaded rules from JSON
                group_config['compress_dims'] = self.compression_rules.get(param_name, None)
            elif layer_map_path:
                # Calculate rule based on layer type
                group_config['compress_dims'] = get_rule_for_parameter(param_name, self.layer_map)
            else:
                # No compression
                group_config['compress_dims'] = None
            

            compressed_size = param.numel()
            compressed_shape = param.shape
            if group_config['compress_dims'] is not None:
                compressed_shape = list(param.shape)
                for dim in group_config['compress_dims']:
                    compressed_shape[dim] = 1
                compressed_size = math.prod(compressed_shape)
            
            if verbose:
                print(f'Found param block: {param_name} with shape: {param.shape}')
                print(f'Compressing along', group_config['compress_dims'])
                print(f'Resultant shape', tuple(compressed_shape))
            
            total_slim_params += compressed_size
            
            # No weight decay for normed params
            if param.dim() >= 2:
                group_config['weight_decay'] = weight_decay
            else:
                group_config['weight_decay'] = 0.0

            optim_groups.append(group_config)
        
        # Print compression savings
        savings_pct = (1 - total_slim_params / total_params) * 100
        print("\n" + "="*100)
        print(f"SlimAdam is saving {savings_pct:.2f}% of second moments")
        print("="*100)
        
        # default parameters for params
        defaults = dict(lr = lr, betas = betas, eps = eps) 
        super().__init__(optim_groups, defaults)
    
    @torch.no_grad()
    def compress_grad_squared(self, grad: torch.Tensor, dims: Tuple[int, ...] = None) -> torch.Tensor:
        """ squares the gradient and the compresses along dim if provided; if dims = None, no compression is performed """
        if dims is None:
            return grad**2
        else:
            return torch.mean(grad * grad, dim = dims, keepdim = True)

    @torch.no_grad()
    def step(self):
        """ Optimizer step """
                
        for group_config in self.param_groups:
            # group hparams
            beta1, beta2 = group_config['betas']
            lr, eps = group_config['lr'], group_config['eps']
            name = group_config['name']
            weight_decay = group_config['weight_decay']

            # compression dims for the second moment
            compress_dims = group_config['compress_dims']
            
            for param in group_config['params']:
                if param.grad is None:
                    continue
                # for every parameter, state stores the first and second moment
                state = self.state[param] # get the state corresponding to the param
            
                # Optimizer state initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['mu'] = torch.zeros_like(param, memory_format = torch.preserve_format)
                    state['nu'] = torch.zeros_like(self.compress_grad_squared(param, compress_dims), memory_format = torch.preserve_format)

                # Optimizer state update
                # update first moment using p.grad
                state['mu'].mul_(beta1).add_(param.grad, alpha = 1-beta1)
                # compute gradient squared                
                grad_squared = self.compress_grad_squared(param.grad, compress_dims)
                # update the second moment using squared gradients
                state['nu'].mul_(beta2).add_(grad_squared, alpha = 1-beta2)
                
                # Add weight decay
                if weight_decay > 0.0:
                    param.mul_(1 - lr*weight_decay)
                    
                # Compute update
                state['step'] += 1

                # bias correction
                mu_hat = state['mu'] / (1 - beta1 ** state['step']) 
                nu_hat = state['nu'] / (1 - beta2 ** state['step'])

                # Optimizer parameter update
                update = lr * (mu_hat / (nu_hat.sqrt() + eps))
                # Apply update
                param.add_(-update)
        return

