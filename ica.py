from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA
from scipy.stats import zscore
import pandas as pd
from itertools import combinations
from features import get_features
from scipy.stats import pearsonr


# Specify base directory
base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data')
results_dir = join(base_dir, 'results')


# Specify number of matchups, repeats, and players
n_matchups = 4
n_repeats = 8
n_players = 4


# Load in PCA-reduced data
k = 100

lstm_pca_reduce = np.load(join(results_dir, f'lstms_tanh-z_pca-k{k}.npy'))


# Stack PCA-reduced data an run ICA
k = 100

lstm_ica = []
for m in np.arange(n_matchups):
    
    stack_lstm = []
    for r in np.arange(n_repeats):
        for p  in np.arange(n_players):
            stack_lstm.append(lstm_pca_reduce[m, r, p])
            
    stack_lstm = np.vstack(stack_lstm)
    print(f"Stacked matchup {m} data and starting ICA")
    ica = FastICA(n_components=k, whiten=True)
    transformed = ica.fit_transform(stack_lstm)
    
    # Un-stack ICA-transformed arrays for repeats, players
    unstack_lstm = np.stack(np.split(np.stack(
        np.split(transformed, 8), axis=0), 4, axis=1), axis=1)
    
    lstm_ica.append(unstack_lstm)
    print(f"Finished running stacked ICA for matchup {m}")
    
lstm_ica = np.stack(lstm_ica, axis=0)

np.save(f'results/lstms_tanh-z_pca-k{k}_ica.npy', lstm_ica)


# Load helper function(s) for interacting with CTF dataset
from ctf_dataset.load import create_wrapped_dataset

base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data')

# Create wrapped CTF dataset
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

n_lstms = 512
n_repeats = 8
n_players = 4
map_id = 0

# Get matchups with all same agents (e.g. AA vs AA)
agent_ids = wrap_f['map/matchup/repeat/player/agent_id'][0, :, :, :, 0]
matchup_ids = np.all(agent_ids[:, 0, :] == agent_ids[:, 0, 0][:, np.newaxis], axis=1)
n_matchups = np.sum(matchup_ids) # 0, 34, 49, 54

# Load pre-saved PCA's 
k = 100
lstm_ica = np.load(f'results/lstms_tanh-z_pca-k{k}_ica.npy')
lstm_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')


# Exclude degenerate features from analysis 
feature_set = ['position', 'health', 'events']
all_features, labels = get_features(wrap_f, feature_set=feature_set, map_id=map_id, matchup_id=matchup_ids, player_id=slice(None), repeat_id=slice(None))


features_exclude = []
for label in labels: 
    features = all_features[..., np.array(labels) == label]
    n_nonzeros = np.sum(np.nonzero(features))
    print(f'checking {label} for all nonzeros; found {n_nonzeros} nonzeros')
    if n_nonzeros == 0:
        features_exclude.append(label)
        print(f'excluding {label}')

#labels = [l for l in labels if l not in features_exclude]  
        
# Define a single variable to pull stats for (this may be redundant, review later)

ica_corrs = {}
for game_var in labels:
    features = all_features[..., np.array(labels) == game_var]
    # code is breaking above because new labels code that removes degenerative features does not match dimensions of 
    feature_shape = features.shape[:-2]
    ica_corrs[game_var] = np.full(feature_shape + (k,), np.nan)
 
    for matchup_id in np.arange(n_matchups):
        for repeat_id in np.arange(n_repeats):   
            for player_id in np.arange(n_players): 
                for ic_id in np.arange(k):
                    ic_corr = pearsonr(features[matchup_id, repeat_id, player_id, :, 0], 
                                       lstm_ica[matchup_id, repeat_id, player_id, :, ic_id])[0]
                    ica_corrs[game_var][matchup_id, repeat_id, player_id, ic_id] = ic_corr
  

    print(f"finished ica correlations w/ {game_var}")

# Save dictionary 
np.save(f'results/lstm_ica-k{k}_feature_correlations.npy', ica_corrs)

## Plot

ica_corrs = np.load('results/lstm_ica-k100_feature_correlations.npy', allow_pickle=True)

# Summarize PCA Corrs across players and repeats
ica_corr_means = []

for game_var in ica_corrs:
    ica_corr_means.append(np.nanmean(ica_corrs[game_var], axis=(1, 2)))

ica_corr_means = np.stack(ica_corr_means, 1)

assert ica_corr_means.shape[1] == len(labels)

ic_id = 2

for ic_id in np.arange(1,10):
    plt.matshow(ica_corr_means[..., ic_id], cmap='RdBu_r')
    plt.yticks([0, 1, 2, 3], ['A','B','C','D'])
    plt.xticks(np.arange(ica_corr_means.shape[1]), labels, rotation=90);
    plt.title(f'ICA Feature Correlations for IC{ic_id}')
    plt.colorbar()
