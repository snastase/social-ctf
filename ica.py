from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import FastICA


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
