from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
import pandas as pd
from itertools import combinations

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
matchup_ids = np.all(agent_ids[:, 0, :] == 
                     agent_ids[:, 0, 0][:, np.newaxis], axis=1)
n_matchups = np.sum(matchup_ids) # 0, 34, 49, 54


# Extract LSTMs for one map and matchup
lstms_matched = np.tanh(wrap_f['map/matchup/repeat/player/time/lstm'][
    map_id, matchup_ids, ...].astype(np.float32))
print("Loaded LSTMs for within-population matchups")


# Loop through matchups, repeats, and players to compute PCA
k = n_lstms

lstm_pca = {}
for m in np.arange(n_matchups):
    lstm_pca[m] = {}
    for r in np.arange(n_repeats):
        lstm_pca[m][r] = {}
        for p in np.arange(n_players):
            lstm_pca[m][r][p] = {}
            pca = PCA(n_components=k)
            transformed = pca.fit_transform(
                #zscore(lstms_matched[m, r, p], axis=0))
                #np.tanh(lstms_matched[m, r, p]))
                zscore(lstms_matched[m, r, p], axis=0))
            lstm_pca[m][r][p]['transformed'] = transformed
            lstm_pca[m][r][p]['pca'] = pca
            print(f"Finished running PCA for matchup {m}, "
                  f"repeat {r}, player {p}")

np.save('results/pca_lstm_tanh-z_results.npy', lstm_pca)

# Convert PCA outputs to long dictionary for plotting
lstm_pca_long = {'population': [], 'repeat': [], 'player': [],
                 'variance explained': [], 'dimension': []}

pops = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

for m in np.arange(n_matchups):
    for r in np.arange(n_repeats):
        for p in np.arange(n_players):
            for k, v in enumerate(lstm_pca[m][r][p][
                'pca'].explained_variance_ratio_):
                lstm_pca_long['population'].append(pops[m])
                lstm_pca_long['repeat'].append(r)
                lstm_pca_long['player'].append(p)
                lstm_pca_long['variance explained'].append(v)
                lstm_pca_long['dimension'].append(k + 1)

lstm_pca_long = pd.DataFrame(lstm_pca_long)

max_k = 30
lstm_pca_trunc = lstm_pca_long[lstm_pca_long['dimension'] <= max_k]
                
sns.set(font_scale=1.2, style='white')
sns.relplot(data=lstm_pca_trunc, x='dimension',
            y='variance explained', hue='repeat',
            col='population', col_wrap=2,
            kind='line')

# Compute number of components required for percentage variance
percents = [.5, .75, .9, .95, .99]

percents_vaf = np.zeros((n_matchups, n_repeats, n_players, len(percents)))
for m in np.arange(n_matchups):
    for r in np.arange(n_repeats):
        for p in np.arange(n_players):
            for i, perc in enumerate(percents):
                k = np.sum(np.cumsum(
                    lstm_pca[m][r][p][
                        'pca'].explained_variance_ratio_) <= perc) + 1     
                percents_vaf[m, r, p, i] = k

for m in np.arange(n_matchups):
    for i, perc in enumerate(percents):
        median = int(np.median(percents_vaf[m, ..., i]))
        min = int(np.amin(percents_vaf[m, ..., i]))
        max = int(np.amax(percents_vaf[m, ..., i]))
        print(f"Population {pops[m]}: {median} dimensions "
              f"for {perc} variance (range: {min}-{max})")
    print('\n')


# Stack pairs of players and compute joint PCA
pairs = list(combinations(np.arange(n_players), 2))
n_pairs = len(pairs)

k = n_lstms * 2

coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

lstm_pair_pca = {}
for m in np.arange(n_matchups):
    lstm_pair_pca[m] = {}
    for r in np.arange(n_repeats):
        lstm_pair_pca[m][r] = {}
        for p, pair in enumerate(pairs):
            lstm_pair_pca[m][r][p] = {}
            stack_lstm = np.hstack((lstms_matched[m, r, pair[0]],
                                    lstms_matched[m, r, pair[1]]))
            pca = PCA(n_components=k)
            transformed = pca.fit_transform(
                zscore(stack_lstm, axis=0))
            lstm_pair_pca[m][r][p]['transformed'] = transformed
            lstm_pair_pca[m][r][p]['pca'] = pca
            print(f"Finished running PCA for matchup {m}, "
                  f"repeat {r}, pair {pair}")
            
np.save('results/pair-pca_lstm_tanh-z_results.npy', lstm_pair_pca)

# Convert PCA outputs to long dictionary for plotting
lstm_pair_pca_long = {'population': [], 'repeat': [], 'pair': [],
                      'variance explained': [], 'dimension': [],
                      'type': []}

pops = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
pair_type = {c:('cooperative' if c in coop_ids else 'competitive')
             for c in np.arange(n_pairs)}

for m in np.arange(n_matchups):
    for r in np.arange(n_repeats):
        for p in np.arange(n_pairs):
            for k, v in enumerate(lstm_pair_pca[m][r][p][
                'pca'].explained_variance_ratio_):
                lstm_pair_pca_long['population'].append(pops[m])
                lstm_pair_pca_long['repeat'].append(r)
                lstm_pair_pca_long['pair'].append(p)
                lstm_pair_pca_long['variance explained'].append(v)
                lstm_pair_pca_long['dimension'].append(k + 1)
                lstm_pair_pca_long['type'].append(pair_type[p])

lstm_pair_pca_long = pd.DataFrame(lstm_pair_pca_long)

max_k = 10
lstm_pair_pca_trunc = lstm_pair_pca_long[
    lstm_pair_pca_long['dimension'] <= max_k]
                
sns.set(font_scale=1.2, style='white')
sns.relplot(data=lstm_pair_pca_trunc, x='dimension',
            y='variance explained', hue='type',
            col='population', col_wrap=2, linewidth=3,
            kind='line')

# Compute number of components required for percentage variance
percents = [.5, .75, .9, .95, .99]

percents_vaf = np.zeros((n_matchups, n_repeats, n_pairs, len(percents)))
for m in np.arange(n_matchups):
    for r in np.arange(n_repeats):
        for p in np.arange(n_pairs):
            for i, perc in enumerate(percents):
                k = np.sum(np.cumsum(
                    lstm_pair_pca[m][r][p][
                        'pca'].explained_variance_ratio_) <= perc) + 1     
                percents_vaf[m, r, p, i] = k

for m in np.arange(n_matchups):
    for type, c in zip(['cooperative', 'competitive'],
                       [coop_ids, comp_ids]):
        for i, perc in enumerate(percents):
            median = int(np.median(percents_vaf[m, :, c, i]))
            min = int(np.amin(percents_vaf[m, :, c, i]))
            max = int(np.amax(percents_vaf[m, :, c, i]))
            print(f"Population {pops[m]} {type}: {median} dimensions "
                  f"for {perc} variance (range: {min}-{max})")
        print('\n')


# Stack across all repeats and run PCA
k = n_lstms

lstm_stack_pca = {}
for m in np.arange(n_matchups):
    lstm_stack_pca[m] = {}
    
    stack_lstm = []
    for r in np.arange(n_repeats):
        for p  in np.arange(n_players):
            stack_lstm.append(zscore(lstms_matched[m, r, p],
                                     axis=0))
            
    stack_lstm = np.vstack(stack_lstm)
    pca = PCA(n_components=k)
    transformed = pca.fit_transform(stack_lstm)

    lstm_stack_pca[m]['transformed'] = transformed
    lstm_stack_pca[m]['pca'] = pca
    print(f"Finished running stacked PCA for matchup {m}")
            
np.save('results/stack-pca_lstm_tanh-z_results.npy', lstm_stack_pca)

# Convert PCA outputs to long dictionary for plotting
lstm_stack_pca_long = {'population': [], 'variance explained': [],
                       'dimension': []}

pops = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

for m in np.arange(n_matchups):
    for k, v in enumerate(lstm_stack_pca[m][
        'pca'].explained_variance_ratio_):
        lstm_stack_pca_long['population'].append(pops[m])
        lstm_stack_pca_long['variance explained'].append(v)
        lstm_stack_pca_long['dimension'].append(k + 1)

lstm_stack_pca_long = pd.DataFrame(lstm_stack_pca_long)

max_k = 8
lstm_stack_pca_trunc = lstm_stack_pca_long[
    lstm_stack_pca_long['dimension'] <= max_k]
                
sns.set(font_scale=1.2, style='white')
sns.lineplot(data=lstm_stack_pca_trunc, x='dimension',
             y='variance explained', hue='population',
             linewidth=3)

# Compute number of components required for percentage variance
percents = [.5, .75, .9, .95, .99]

percents_vaf = np.zeros((n_matchups, len(percents)))
for m in np.arange(n_matchups):
    for i, perc in enumerate(percents):
        k = np.sum(np.cumsum(
            lstm_stack_pca[m][
                'pca'].explained_variance_ratio_) <= perc) + 1     
        percents_vaf[m, i] = k

for m in np.arange(n_matchups):
    for i, perc in enumerate(percents):
        median = int(np.median(percents_vaf[m, i]))
        print(f"Population {pops[m]}: {median} dimensions "
              f"for {perc} variance")
    print('\n')


# Create reduced-dimension version of data (e.g. k = 100)
k = 100

lstm_pca_reduce = []
for m in np.arange(n_matchups):
    
    stack_lstm = []
    for r in np.arange(n_repeats):
        for p  in np.arange(n_players):
            stack_lstm.append(zscore(lstms_matched[m, r, p],
                                     axis=0))
            
    stack_lstm = np.vstack(stack_lstm)
    pca = PCA(n_components=k)
    transformed = pca.fit_transform(stack_lstm)
    
    percent_vaf = np.sum(pca.explained_variance_ratio_)
    
    # Un-stack PCA-transformed arrays for repeats, players
    unstack_lstm = np.stack(np.split(np.stack(
        np.split(transformed, 8), axis=0), 4, axis=1), axis=1)
    
    lstm_pca_reduce.append(unstack_lstm)
    print(f"Finished running stacked PCA for matchup {m}")
    print(f"Proportion variance at for matchup {m} at k = {k}: "
          f"{percent_vaf:.3f}")
    
lstm_pca_reduce = np.stack(lstm_pca_reduce, axis=0)

np.save(f'results/lstms_tanh-z_pca-k{k}.npy', lstm_pca_reduce)
