from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from sklearn.decomposition import PCA
import pandas as pd
from itertools import combinations
from statistical_tests import bootstrap_test, fisher_mean
from statsmodels.stats.multitest import multipletests

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
lstm = 'lstm'

lstms_matched = wrap_f[f'map/matchup/repeat/player/time/{lstm}'][
    map_id, matchup_ids, ...].astype(np.float32)
print("Loaded LSTMs for within-population matchups")

# Apply tanh to LSTMs
if lstm == 'lstm':
    lstms_matched = np.tanh(lstms_matched)


# Matchup-, repeat-, player-specific PCA
k = n_lstms

lstm_within_pca = {}
for m in np.arange(n_matchups):
    lstm_within_pca[m] = {}
    for r in np.arange(n_repeats):
        lstm_within_pca[m][r] = {}
        for p in np.arange(n_players):
            lstm_within_pca[m][r][p] = {}
            pca = PCA(n_components=k)
            transformed = pca.fit_transform(
                zscore(lstms_matched[m, r, p], axis=0))
            lstm_within_pca[m][r][p]['transformed'] = transformed
            lstm_within_pca[m][r][p]['pca'] = pca
            print(f"Finished running PCA for matchup {m}, "
                  f"repeat {r}, player {p}")

np.save('results/within-pca_tanh-z_results.npy', lstm_within_pca)

# Convert PCA outputs to long dictionary for plotting
lstm_within_pca_long = {'population': [], 'repeat': [], 'player': [],
                        'variance explained': [], 'dimension': []}

pops = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

for m in np.arange(n_matchups):
    for r in np.arange(n_repeats):
        for p in np.arange(n_players):
            for k, v in enumerate(lstm_within_pca[m][r][p][
                'pca'].explained_variance_ratio_):
                lstm_within_pca_long['population'].append(pops[m])
                lstm_within_pca_long['repeat'].append(r)
                lstm_within_pca_long['player'].append(p)
                lstm_within_pca_long['variance explained'].append(v)
                lstm_within_pca_long['dimension'].append(k + 1)

lstm_within_pca_long = pd.DataFrame(lstm_within_pca_long)

max_k = 30
lstm_within_pca_trunc = lstm_within_pca_long[
    lstm_within_pca_long['dimension'] <= max_k]
                
sns.set(font_scale=1.2, style='white')
sns.relplot(data=lstm_within_pca_trunc, x='dimension',
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
                    lstm_within_pca[m][r][p][
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


# Horizontally stack pairs of players and compute joint PCA
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


# Plot scree plot of variance accounted
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

matchup = 0
pca_k = 100
evr = lstm_stack_pca[matchup]['pca'].explained_variance_ratio_
evr_cum = np.cumsum(evr)
dimensions = np.arange(1, 513)

percents = [.9, .95, .99]
percents_vaf = {}
for i, perc in enumerate(percents):
    k = np.sum(np.cumsum(
        lstm_stack_pca[m][
            'pca'].explained_variance_ratio_) <= perc) + 1     
    percents_vaf[k] = perc

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(dimensions, evr, color='.5')
ax.scatter(dimensions[:pca_k], evr[:pca_k], color='tab:red')
ax.set_xlabel('dimensions')
ax.set_ylabel('proportion of\nvariance explained')
for k, perc in percents_vaf.items():
    ax.axvline(k, 0, .35, color='.5', zorder=-1)
    ax.annotate(f'{perc:.0%}', xy=(k + 10, .37), ha='center',
                xycoords=('data', 'axes fraction'))
axins = inset_axes(ax, width=1.2, height=1)
axins.scatter(dimensions, evr_cum, color='.5')
axins.scatter(dimensions[:pca_k], evr_cum[:pca_k], color='tab:red')
axins.xaxis.set_ticks([])
axins.yaxis.set_ticks([])
axins.set_xlabel('dimensions', size=12)
axins.set_ylabel('cumulative\nvariance', size=12)
plt.savefig(f'figures/scree_tanh-z_pca-k{pca_k}_m{matchup}.png', dpi=300,
            bbox_inches='tight')
    
    
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


# Plot some example PC time series
k = 100
lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')


### Get rid of everything south of heres

# Compute correlations for PC in comparison to game variable 
from features import get_features
from scipy.stats import pearsonr

# Load pre-saved PCA's 
k = 100
lstm_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')


# Exclude degenerate features from analysis 
feature_set = ['position', 'health', 'events']
all_features, labels = get_features(wrap_f, feature_set=feature_set,
                                    map_id=map_id, matchup_id=matchup_ids,
                                    player_id=slice(None),
                                    repeat_id=slice(None))

features_exclude = []
for label in labels: 
    features = all_features[..., np.array(labels) == label]
    n_nonzeros = np.sum(np.nonzero(features))
    print(f'checking {label} for all nonzeros; "found {n_nonzeros} nonzeros')
    if n_nonzeros == 0:
        features_exclude.append(label)
        print(f'excluding {label}')
        
labels = [l for l in labels if l not in features_exclude]  
        
# Define a single variable to pull stats from
# (this may be redundant, review later)
pca_corrs = {}
for game_var in labels:
    features = all_features[..., np.array(labels) == game_var]
    # code is breaking above because new labels code that
    # removes degenerative features does not match dimensions of 
    feature_shape = features.shape[:-2]
    pca_corrs[game_var] = np.full(feature_shape + (k,), np.nan)
 
    for matchup_id in np.arange(n_matchups):
        for repeat_id in np.arange(n_repeats):   
            for player_id in np.arange(n_players): 
                for pc_id in np.arange(k):
                    pc_corr = pearsonr(features[matchup_id, repeat_id, player_id, :, 0],
                                       lstm_pca[matchup_id, repeat_id, player_id,
                                                :, pc_id])[0]
                    pca_corrs[game_var][matchup_id, repeat_id, player_id, pc_id] = pc_corr
  

    print(f"finished pca correlations w/ {game_var}")

# Save dictionary 
np.save(f'results/lstm_pca-k{k}_feature_correlations.npy', pca_corrs)


# Summarize PCA correlations across players and repeats
pca_corrs = np.load('results/lstm_pca-k100_feature_correlations.npy',
                    allow_pickle=True)
pca_corr_means = []

for game_var in pca_corrs:
    pca_corr_means.append(np.nanmean(pca_corrs[game_var], axis=(1, 2)))

pca_corr_means = np.stack(pca_corr_means, 1)

assert pca_corr_means.shape[1] == len(labels)

pc_id = 2
for pc_id in np.arange(1,10):
    plt.matshow(pca_corr_means[..., pc_id], cmap='RdBu_r')
    plt.yticks([0, 1, 2, 3], ['A','B','C','D'])
    plt.xticks(np.arange(pca_corr_means.shape[1]), labels, rotation=90);
    plt.title(f'PCA Feature Correlations for PC{pc_id}')
    plt.colorbar()


# Look at some properties of the PCA-reduced LSTMs
k = 100
#lstm_pca_reduce = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')
lstm_pca_reduce = np.load(f'results/lstms_tanh-z_pca-k{k}_reg-pre.npy')

# Look at ISC of PCs for individual games
matchup, repeat = 0, 0

n_pcs = 10
fig, axs = plt.subplots(2, 5, figsize=(25, 8))
for pc, ax in zip(np.arange(n_pcs), axs.ravel()):
    corr = np.corrcoef(lstm_pca_reduce[matchup, repeat, ..., pc])
    sns.heatmap(corr, square=True, annot=True, vmin=-1, vmax=1,
                cmap='RdBu_r', xticklabels=False, yticklabels=False,
                fmt='.2f', ax=ax)
    ax.set_title(f'PC{pc + 1}')


# Look at ISC of PCs averaged across games
pca_k = 100
matchup = 0
n_repeats = 8

n_pcs = 10
fig, axs = plt.subplots(2, 5, figsize=(25, 8))
for pc, ax in zip(np.arange(n_pcs), axs.ravel()):
    corr = fisher_mean([np.corrcoef(lstm_pca_reduce[matchup, r, ..., pc])
                        for r in np.arange(n_repeats)], axis=0)
    sns.heatmap(corr, square=True, annot=True, vmin=-1, vmax=1,
                cmap='RdBu_r', xticklabels=False, yticklabels=False,
                fmt='.2f', ax=ax)
    ax.set_title(f'PC{pc + 1}')
#plt.savefig(f'figures/isc_coop-comp_tanh-z_pca-k{pca_k}_m{matchup}.png',
#            dpi=300, bbox_inches='tight')
plt.savefig(f'figures/isc_coop-comp_tanh-z_pca-k{pca_k}_reg-pre_m{matchup}.png',
            dpi=300, bbox_inches='tight')
    

# Specific subset of PCs averaged across games
matchup = 0
n_repeats = 8

pcs = [2, 5, 9]
fig, axs = plt.subplots(1, len(pcs), figsize=(12, 2.8))
for pc, ax in zip(pcs, axs.ravel()):
    corr = np.mean([np.corrcoef(lstm_pca_reduce[matchup, r, ..., pc])
                    for r in np.arange(n_repeats)], axis=0)
    sns.heatmap(corr, square=True, annot=True, vmin=-1, vmax=1,
                cmap='RdBu_r', xticklabels=False, yticklabels=False,
                fmt='.2f', ax=ax)
    ax.set_title(f'PC{pc + 1}')

    
# Difference in cooperative/competitive ISC across PCs
matchup = 0
n_repeats = 8
n_pcs = 100

isc_diffs = []
isc_diffs_df = {'difference': [], 'PC': [], 'repeat': []}
for pc in np.arange(n_pcs):
    corrs = [np.corrcoef(lstm_pca_reduce[matchup, r, ..., pc])
             for r in np.arange(n_repeats)]
    diffs = [np.mean(c[[0, 3], [1, 2]]) - np.mean(c[0:2, 2:4])
             for c in corrs]
    isc_pc_diffs = []
    for r, diff in enumerate(diffs):
        isc_diffs_df['difference'].append(diff)
        isc_diffs_df['PC'].append(pc + 1)
        isc_diffs_df['repeat'].append(r)
        isc_pc_diffs.append(diff)
    isc_diffs.append(isc_pc_diffs)
isc_diffs_df = pd.DataFrame(isc_diffs_df)
isc_diffs = np.array(isc_diffs).T

observed, ci, p, distribution = bootstrap_test(isc_diffs,
                                               bootstrap_axis=0,
                                               n_bootstraps=1000,
                                               estimator=fisher_mean,
                                               ci_percentile=95,
                                               side='two-sided')

_, fdr_p, _, _ = multipletests(p, method='fdr_bh')

sig_pos = ((fdr_p < .05) & (observed > 0)).nonzero()[0]
sig_neg = ((fdr_p < .05) & (observed < 0)).nonzero()[0]

fig, ax = plt.subplots(figsize=(16, 4))
sns.barplot(x='PC', y='difference', data=isc_diffs_df, ax=ax, color='.6',
            estimator=fisher_mean)
# ax.set_ylim(-.375, .325) # for matchup = 3 (sig y = -.01)
ax.set_ylim(-.3, 1) # for matchup = 0
ax.set_xticks([0, 19, 39, 59, 79, 99])
for sig_pc in sig_pos:
    ax.annotate('.', (sig_pc, -.02), color='tab:red', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
for sig_pc in sig_neg:
    ax.annotate('.', (sig_pc, -.02), color='tab:blue', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
ax.set_ylabel('cooperative – competitive ISC')
ax.set_title(f'difference in cooperative vs. competitive ISC for 100 PCs');
plt.savefig(f'figures/isc_diff-bars_tanh-z_pca-k{pca_k}_m{matchup}.png',
            dpi=300, bbox_inches='tight')



# Examine time-resolved intersubject synchrony of PCs
from brainiak.isc import isc
from detectors import get_following

matchup = 0
pops = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}
repeat = 1
pc = 10 - 1 

following = get_following(wrap_f, matchup_id=matchup, repeat_id=repeat)

lstm_pc = lstm_pca_reduce[matchup, repeat, ..., pc]
lstm_pc_isc = isc(lstm_pc.T, pairwise=True)[:, 0]

fig, axs = plt.subplots(10, 1, figsize=(12, 14))
axs[0].plot(lstm_pc[0], c='darkred', alpha=.7)
axs[0].plot(lstm_pc[1], c='coral', alpha=.7)
axs[0].set_xticks([])
axs[0].set_ylabel('activation')
axs[0].set_title(f'PC{pc + 1} (population {pops[matchup]}, '
                 f'repeat {repeat})')
axs[0].annotate(f'ISC: {lstm_pc_isc[0]:.3f}', (.95, .95),
                ha='right', xycoords='axes fraction')
axs[1].plot(lstm_pc[2], c='darkblue', alpha=.7)
axs[1].plot(lstm_pc[3], c='lightseagreen', alpha=.7)
axs[1].set_xticks([])
axs[1].set_ylabel('activation')
axs[1].annotate(f'ISC: {lstm_pc_isc[5]:.3f}', (.95, .95),
                ha='right', xycoords='axes fraction')

lstm_pc_isps = isps(lstm_pc.T)
axs[2].plot(lstm_pc_isps[0], c='darkred', alpha=.7)
axs[2].set_xticks([])
axs[2].set_ylabel('phase\nnsynchrony')
axs[3].plot(lstm_pc_isps[5], c='darkblue', alpha=.7)
axs[3].set_xticks([])
axs[3].set_ylabel('phase\nsynchrony')

lstm_pc_iscf = iscf(lstm_pc.T)
axs[4].plot(lstm_pc_iscf[0], c='darkred', alpha=.7)
axs[4].set_xticks([])
axs[4].set_ylabel('co-fluctuation')
axs[5].plot(lstm_pc_iscf[5], c='darkblue', alpha=.7)
axs[5].set_xticks([])
axs[5].set_ylabel('co-fluctuation')

lstm_pc_win = window_isc(lstm_pc.T)
axs[6].plot(resample_windows(lstm_pc_win.T[0], width=150,
                             collapse=np.nanmean),
            c='darkred', alpha=.7)
axs[6].set_xticks([])
axs[6].set_ylabel('window ISC\n(width=150)')
axs[7].plot(resample_windows(lstm_pc_win.T[1], width=150,
                             collapse=np.nanmean), 
            c='darkblue', alpha=.7)
axs[7].set_ylabel('window ISC\n(width=150)')
axs[7].set_xticks([])

axs[8].plot(following[0], c='darkred', alpha=.7)
axs[8].set_ylabel('following')
axs[8].set_xticks([])
axs[9].plot(following[1], c='darkblue', alpha=.7)
axs[9].set_ylabel('following')
plt.xlabel('time')
sns.despine()

plt.savefig(f'PC{pc + 1}_coop_m{matchup}_r{repeat}.png',
            dpi=300, transparent=False,
            bbox_inches='tight')


# Correlation matrix of coupling metrics
matchup = 0
n_repeats = 8
pc = 10 - 1

corrs = []
for repeat in np.arange(n_repeats):
    lstm_pc = lstm_pca_reduce[matchup, repeat, ..., pc]
    lstm_pc_isps = isps(lstm_pc.T)
    lstm_pc_iscf = iscf(lstm_pc.T)
    lstm_pc_wins = window_isc(lstm_pc.T)
    following = get_following(wrap_f, repeat_id=repeat, pair_ids=np.arange(6))
    
    pair_corrs = []
    for pair in [0, 5]:
        lstm_pc_win = np.nan_to_num(resample_windows(lstm_pc_wins.T[pair],
                                                     width=150,
                                                     collapse=np.nanmean))
        
        corr = np.corrcoef(np.vstack((lstm_pc_isps[pair],
                                      lstm_pc_iscf[pair],
                                      lstm_pc_win,
                                      following[pair])))
        corrs.append(corr)

corrs_mean = np.mean(corrs, axis=0)

labels = ['phase synchrony', 'co-fluctuation', 'windowed ISC', 'following']
fig, ax = plt.subplots(figsize=(5, 4))
m = sns.heatmap(corrs_mean, vmin=0, vmax=1, cmap='inferno', annot=True,
                cbar=True, square=True, xticklabels=labels, yticklabels=labels)
m.tick_params(left=False, bottom=False)
m.set_title(f'PC{pc + 1} coupling correlation')
plt.savefig(f'PC{pc + 1}_coop_corrs.png', dpi=300, bbox_inches='tight')




