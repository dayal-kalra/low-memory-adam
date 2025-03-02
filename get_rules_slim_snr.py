import pandas as pd
import argparse
import os
import re
import json

# Function to format tick labels
def format_tick(x, pos):
    if x == 0:
        return '0'
    elif x >= 1000:
        return fr'${int(x/1000)}$k'
    else:
        return str(int(x))

def get_rows_where_col_equals(df, col, value):
    return df.loc[df[col] == value].copy()

def get_rows_where_col_in(df, col, values):
    return df.loc[df[col].isin(values)].copy()

def get_rows_where_col_greater(df, col, value):
    return df.loc[df[col] > value].copy()

def get_rows_where_col_less(df, col, value):
    return df.loc[df[col] < value].copy()

def find_max_col_rows(df, col_name):
    return df[df[col_name] == df[col_name].max()]
    
def extract_rules(grouped_stats, cutoff):

    """
    Extracts compression rules for neural network layers by analyzing signal-to-noise ratios (SNR).

    Args:
        grouped_stats (pd.DataFrame): DataFrame containing layer statistics with columns:
            - key: unique layer identifier
            - compress_dims: tuple of dimensions to compress (as string)
            - snr: signal-to-noise ratio
            - shape: layer shape (as string tuple)
        cutoff (float): SNR threshold for compression selection

    Returns:
        tuple:
            - slim_rules_cutoff (dict): Maps layer key to compress_dims that maximize
              compression among dimensions with SNR > cutoff. None if no dimensions
              exceed cutoff.
            - slim_rules (dict): Maps layer key to compress_dims with highest SNR,
              regardless of cutoff.
            - adam_rules (dict): Currently returns None for all layers.
            - adalayer_rules (dict): Maps layer key to tuple of all dimensions for
              non-embedding layers, None for embedding layers.

    Example:
        For layer shape (512, 1024) with compress_dims options [(0,), (1,), (0,1)]:
        - If all SNRs > cutoff, slim_rules_cutoff picks (0,1) for max compression
        - slim_rules picks the option with highest SNR
    """

    slim_trim = {}
    adam_rules = {}
    adalayer_rules = {}

    unique_layers = df['key'].unique()

    for layer in unique_layers:
        
        # Get data for this layer
        df_layer = get_rows_where_col_equals(grouped_stats, 'key', layer)
    
        ## Adam rules ##
        adam_rules[layer] = None

        ## Adalayer rules ##
        shape = eval(df_layer['shape'].unique()[0])
        
        if 'tok_embed' in layer or 'head' in layer or 'wte.weight' in layer:
            adalayer_rules[layer] = None
        elif 'ln_1.weight' in layer or 'ln_2.weight' in layer or 'ln_f.weight' in layer:
            adalayer_rules[layer] = None
        else:
            adalayer_rules[layer] = tuple(range(0, len(shape)))


        # 3. For slim_trim
        # First filter rows above cutoff
        df_above_cutoff = get_rows_where_col_greater(df_layer, 'snr', cutoff)
        if not df_above_cutoff.empty:
            max_snr_row = df_above_cutoff.loc[df_above_cutoff['snr'].idxmax()]
            compress_dims = eval(max_snr_row['compress_dims'])
        else:
            compress_dims = None  # or pd.Series() or whatever default value makes sense for your use case

        if 'ln_1.weight' in layer or 'ln_2.weight' in layer or 'ln_f.weight' in layer:
            slim_trim[layer] = None
        else:
            slim_trim[layer] = compress_dims
    return slim_trim, adam_rules, adalayer_rules

#####################################################################

parser = argparse.ArgumentParser(description = 'Hyperparameters')
parser.add_argument('--snr_filename', type = str, default = 'snr_data')
parser.add_argument('--snr_cutoff', type = float, default = 1.0)


cfg = parser.parse_args()

df = pd.read_csv(cfg.snr_filename)

unique_layers = df['key'].unique()

dfs = list()
# create a new dataframe with layer name and number for plotting and further analysis
for layer in unique_layers:
    df_layer = get_rows_where_col_equals(df, 'key', layer)
    shape = df_layer['shape'].unique()[0]
    dfs.append(df_layer)

dfs = pd.concat(dfs, axis = 0, ignore_index = True)
    
grouped_stats = dfs.groupby(['key', 'compress_dims']).agg({'snr': 'mean', 'step': 'first', 'shape': 'first'}).reset_index()
columns = ['step', 'key', 'shape', 'compress_dims', 'snr']
grouped_stats = grouped_stats[columns]

slim_trim_rules, adam_rules, adalayer_rules = extract_rules(grouped_stats, cfg.snr_cutoff)

# Create rules directory if it doesn't exist
os.makedirs('rules', exist_ok = True)

with open(f'rules/slimadam_C{cfg.snr_cutoff}_rules.json', 'w') as f:
    json.dump(slim_trim_rules, f, indent = 4)

with open(f'rules/adam_rules.json', 'w') as f:
    json.dump(adam_rules, f, indent = 4)

with open(f'rules/adalayer_rules.json', 'w') as f:
    json.dump(adalayer_rules, f, indent = 4)
