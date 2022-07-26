#!/usr/bin/env python3

from itertools import product
from sys import argv
from os.path import exists
import numpy as np
from pyspi.calculator import Calculator
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import pandas as pd
import json
from scipy.stats import pearsonr, ttest_ind, zscore
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import PredefinedSplit, KFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score


n_maps = 32
n_pcs = 142
matchup_id = 0
repeat_id = 0


# Load in SPI labels
example = pd.read_csv(table_f, header=[0, 1], index_col=[0])
spi_labels = list(example.columns.levels[0])


# Reorganize SPIs for classification
spi_f = f'results/spis-fast_matchup-{matchup_id}.npz'
if not exists(spi_f):
    spi_dict = dict.fromkeys(spi_labels)
    np.savez(spi_f, **spi_dict, allow_pickle=True)

spi_npz = np.load(spi_f, allow_pickle=True)
spi_dict = dict(spi_npz)

for spi_label in spi_labels:
    spi_stack = []
    for map_id in np.arange(n_maps):
        map_stack = []
        for pc_id in np.arange(n_pcs):
            table_f = (f'results/spis-fast_pc-{pc_id}_matchup-{matchup_id}_'
                       f'map-{map_id}_repeat-{repeat_id}.csv')
            table = pd.read_csv(table_f, header=[0, 1],
                                index_col=[0])[spi_label].to_numpy()
            map_stack.append(table)
            print(f"Loaded map {map_id}, PC{pc_id + 1} ({spi_label})")
        print(f"Loaded all PCs for map {map_id} ({spi_label})")
        spi_stack.append(map_stack)

    spi_stack = np.array(spi_stack)

    if not spi_dict[spi_label]:
        spi_dict[spi_label] = spi_stack
        np.savez(spi_f, **spi_dict)


# Manually add in Pearson correlation for convenience
repeat_id = 0
pearson_labels = ['pearsonr', 'pearsonr-sq']
spi_labels += pearson_labels

for spi_label in pearson_labels:

    spi_stack = []
    for map_id in np.arange(n_maps):

        lstms = np.load(f'results/lstms-pca_matchup-{matchup_id}_'
                        f'map-{map_id}_repeat-{repeat_id}.npy')    

        map_stack = []
        for pc_id in np.arange(n_pcs):
            table = np.corrcoef(lstms[..., pc_id])
            
            if spi_label == 'pearson-sq':
                table = np.square(table)
            
            map_stack.append(table)
        print(f"Loaded all PCs for map {map_id} ({spi_label})")
        spi_stack.append(map_stack)

    spi_stack = np.array(spi_stack)
    spi_dict[spi_label] = spi_stack
    np.savez(spi_f, **spi_dict)


# Create a dictionary of selected SPIs with keywords
fav_spis = {'pearsonr':
            {'name': 'Pearson correlation',
             'keywords': ['undirected', 'linear', 'signed',
                          'bivariate', 'contemporaneous', 'basic']},
            'pearsonr-sq':
            {'name': 'Pearson correlation (squared)',
             'keywords': ['undirected', 'linear', 'unsigned',
                          'bivariate', 'contemporaneous', 'basic']},
            'cov_EmpiricalCovariance':
            {'name': 'covariance',
             'keywords': ['undirected', 'linear', 'signed',
                          'multivariate', 'contemporaneous', 'basic']},
            'prec-sq_EmpiricalCovariance': 
            {'name': 'precision',
             'keywords': ['undirected', 'linear', 'signed',
                          'multivariate', 'contemporaneous', 'basic']},
            'spearmanr': 
            {'name': "Spearman's correlation",
             'keywords': ['undirected', 'nonlinear', 'signed',
                          'bivariate', 'contemporaneous', 'basic']},
            'kendalltau':
            {'name': "Kendall's tau",
             'keywords': ['undirected', 'nonlinear', 'signed',
                          'bivariate', 'contemporaneous', 'basic']},
            'xcorr_max_sig-True':
            {'name': 'cross-correlation',
             'keywords': ['undirected', 'linear', 'signed/unsigned',
                          'bivariate', 'time-dependent', 'basic']},
            'dcorr':
            {'name': 'distance correlation',
             'keywords': ['undirected', 'nonlinear', 'unsigned',
                          'bivariate', 'contemporaneous', 'distance']},
            'hsic':
            {'name': 'Hilbert-Schmidt independence criterion',
             'keywords': ['undirected', 'nonlinear', 'unsigned',
                          'bivariate', 'contemporaneous', 'distance']},
            'anm':
            {'name': 'additive noise model',
             'keywords': ['directed', 'nonlinear', 'unsigned',
                          'bivariate', 'contemporaneous', 'causal',]},
            'igci': 
            {'name': 'information-geometric causal inference',
             'keywords': ['directed', 'nonlinear', 'unsigned',
                          'bivariate', 'contemporaneous', 'causal']},
            'je_gaussian': 
            {'name': 'joint entropy',
             'keywords': ['undirected', 'nonlinear', 'unsigned',
                          'bivariate', 'contemporaneous', 'information']},
            'ce_gaussian':
            {'name': 'conditional entropy',
             'keywords': ['undirected', 'nonlinear', 'unsigned',
                          'bivariate', 'contemporaneous', 'information']},
            'mi_gaussian':
            {'name': 'mutual information',
             'keywords': ['undirected', 'nonlinear', 'unsigned',
                          'bivariate', 'contemporaneous', 'information']},
            'tlmi_gaussian':
            {'name': 'time-lagged mutual information',
             'keywords': ['undirected', 'nonlinear', 'unsigned',
                          'bivariate', 'time-dependent', 'information']},
            'te_kraskov_NN-4_DCE_k-1_kt-1_l-1_lt-1':
            {'name': 'transfer entropy',
             'keywords': ['directed', 'nonlinear', 'unsigned',
                          'bivariate', 'time-dependent', 'information']},
            'gc_gaussian_k-1_kt-1_l-1_lt-1':
            {'name': 'Granger causality',
             'keywords': ['directed', 'linear', 'unsigned',
                          'bivariate', 'time-dependent', 'information']},
            'si_gaussian_k-1':
            {'name': 'stochastic interaction',
             'keywords': ['undirected', 'nonlinear', 'unsigned',
                          'bivariate', 'time-dependent', 'information']},
            'cohmag_multitaper_mean_fs-1_fmin-0_fmax-0-5':
            {'name': 'coherence magnitude',
             'keywords': ['undirected', 'linear', 'unsigned',
                          'bivariate', 'frequency-dependent', 'spectral']},
            'phase_multitaper_mean_fs-1_fmin-0_fmax-0-5':
            {'name': 'coherence phase',
             'keywords': ['undirected', 'linear', 'unsigned',
                          'bivariate', 'frequency-dependent', 'spectral']},
            'psi_wavelet_mean_fs-1_fmin-0_fmax-0-5_mean':
            {'name': 'phase slope index',
             'keywords': ['directed', 'linear', 'nonlinear',
                          'unsigned', 'bivariate', 'frequency-dependent',
                          'time-frequency dependent', 'spectral']},
            'ppc_multitaper_mean_fs-1_fmin-0_fmax-0-5':
            {'name': 'pairwise phase consistency',
             'keywords': ['undirected', 'linear', 'unsigned',
                          'bivariate', 'frequency-dependent', 'spectral']},
            'pli_multitaper_mean_fs-1_fmin-0_fmax-0-5':
            {'name': 'phase lag index',
             'keywords': ['undirected', 'linear', 'unsigned',
                          'bivariate', 'frequency-dependent', 'spectral']},
            'dtf_multitaper_mean_fs-1_fmin-0_fmax-0-5': 
            {'name': 'directed transfer function',
             'keywords': ['directed', 'linear', 'unsigned',
                          'bivariate', 'frequency-dependent', 'spectral']},
            'dcoh_multitaper_mean_fs-1_fmin-0_fmax-0-5':
            {'name': 'directed coherence',
             'keywords': ['directed', 'linear', 'unsigned',
                          'bivariate', 'frequency-dependent', 'spectral']},
            'sgc_nonparametric_mean_fs-1_fmin-0_fmax-0-5':
            {'name': 'spectral Granger causality',
             'keywords': ['directed', 'linear', 'unsigned',
                          'bivariate', 'frequency-dependent', 'spectral']},
            'coint_johansen_max_eig_stat_order-0_ardiff-1':
            {'name': 'cointegration',
             'keywords': ['undirected', 'linear', 'unsigned',
                          'bivariate', 'time-dependent', 'miscellaneous']},
            'pec':
            {'name': 'power envelope correlation',
             'keywords': ['undirected', 'linear', 'unsigned',
                          'bivariate', 'time-dependent']},
            'pdist_euclidean':
            {'name': 'Euclidean distance',
             'keywords': ['pairwise distance']},
            'pdist_cosine':
            {'name': 'cosine distance',
             'keywords': ['pairwise distance']}}

with open('results/fav_spi_keywords.json', 'w') as f:
    json.dump(fav_spis, f, sort_keys=True, indent=2)
    
with open('results/fav_spi_keywords.json', 'r') as f:
    fav_spis = json.load(f)


# Load in pre-exiting SPIs
spi_npz = np.load(spi_f, allow_pickle=True)

# Loop through SPIs for cooperative/competitive classification
def pair_classification(spi_npz, pc_id=None):
    results = dict.fromkeys(spi_labels + ['pearsonr'])
    for spi_label in results.keys():
        print(f'Running classification on "{spi_label}"')

        spi_data = spi_npz[spi_label]
        if pc_id is not None:
            spi_data = spi_data[:, pc_id]
            spi_data = spi_data[:, np.newaxis, ...]

        if spi_data.dtype != float:
            print(f'Skipping "{spi_label}" with {spi_data.dtype}')
            continue

        # Extract off-diagonal triangle(s) from SPI tables 
        spi_triu = spi_data[..., np.triu_indices(4, k=1)[0],
                                 np.triu_indices(4, k=1)[1]]
        spi_tril = spi_data[..., np.tril_indices(4, k=-1)[0],
                              np.tril_indices(4, k=-1)[1]][
            ..., np.array([0, 1, 3, 2, 4, 5])]

        if np.allclose(np.nan_to_num(spi_triu), np.nan_to_num(spi_tril)):
            symmetric = True
            spi_tri = spi_triu
            print("Found symmetric SPI table; using upper triangle")
        else:
            symmetric = False
            spi_tri = np.concatenate((spi_triu, spi_tril), axis=-1)
            print("Found asymmetric SPI table; concatenating triangles")

        # Classify cooperative vs competitive coupling
        n_players = 4
        n_pairs = n_players * (n_players - 1) // 2
        coop_ids, comp_ids = (0, 5), (1, 2, 3, 4)
        
        if not symmetric:
            n_pairs = n_players * (n_players - 1) // 2 * 2
            coop_ids = np.concatenate((coop_ids, np.array(coop_ids) + 6))
            comp_ids = np.concatenate((comp_ids, np.array(comp_ids) + 6))

        spi_long, pair_ids, map_ids = [], [], []
        for map_id in np.arange(n_maps):
            for pair_id in np.arange(n_pairs):
                spi_long.append(spi_tri[map_id, :, pair_id])
                pair_ids.append(1 if pair_id in coop_ids else 0)
                map_ids.append(map_id)

        assert len(spi_long) == len(pair_ids) == len(map_ids)
        spi_long = np.nan_to_num(spi_long)
        pair_ids, map_ids = np.array(pair_ids), np.array(map_ids)

        # Initialize simple classifier
        clf = LogisticRegression(class_weight='balanced')

        # Set up leave-one-map-out cross-validation
        cv = PredefinedSplit(map_ids)

        scores = []
        for f, (train, test) in enumerate(cv.split()):

            # Z-score training SPIs and test SPIs (based on training set)
            scaler = StandardScaler()
            spi_train = scaler.fit_transform(spi_long[train])
            spi_test = scaler.transform(spi_long[test])

            # Train competitive/cooperative classifier based on train SPIs
            clf.fit(spi_train, pair_ids[train])

            # Use fitted classifier to predict test cooperative/competitive 
            pred = clf.predict(spi_test)

            # Evaluate classifier accuracy
            score = balanced_accuracy_score(pair_ids[test], pred)
            scores.append(score)
            #print(f"Fold {f} accuracy: {score:.3f} ({spi_label})")

        results[spi_label] = np.array(scores)
        print(f'Mean "{spi_label}" accuracy: {np.mean(score):.3f}\n')
        
    return results

pair_results = pair_classification(spi_npz)

np.save(f'results/clf-pair_X-all_spi-fast_matchup-{matchup_id}.npy', pair_results)

pair_results = np.load(f'results/clf-pair_X-all_spi-fast_'
                       f'matchup-{matchup_id}.npy', allow_pickle=True).item()

# Compute mean accuracy for each SPI
pair_means = {}
for l, s in pair_results.items():
    if s is not None:
        pair_means[l] = np.mean(s)
        
pair_ranks = {k: pair_means[k] for k in 
              np.array(list(pair_means.keys()))[
                  np.argsort(list(pair_means.values()))[::-1]]}

for spi_label in fav_spis:
    print(f"Accuracy for {fav_spis[spi_label]['name']}: "
          f"{pair_means[spi_label]:.3f}\n\t"
          f"{fav_spis[spi_label]['keywords']}")

# Plot histogram of competitive/cooperative classification performance
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(pair_means.values(), bins=14, binrange=(.4, 1.0), 
             stat='probability', palette='Greys', legend=False, ax=ax)
ax.set(xlabel='balanced accuracy', ylabel='proportion of SPIs',
       title='cooperative vs. competitive pair classification',
       xlim=(.4, 1.))
ax.vlines(.5, ymin=0, ymax=.5, color='goldenrod', ls='--', lw=1, zorder=0)
ax.annotate('chance = .50', xy=(.51, .49), xycoords='data', va='top')
sns.despine(right=False, trim=True)
plt.savefig(f'figures/clf-pair_histogram_matchup-{matchup_id}.png',
            dpi=300, bbox_inches='tight')


# Run cooperative/competitive classification per PC
pair_pc_results = {}
for pc_id in np.arange(n_pcs):
    pair_pc_result = pair_classification(spi_npz, pc_id=pc_id)
    pair_pc_results[pc_id] = pair_pc_result
    print(f"Finished classification for PC{pc_id + 1}")

np.save(f'results/clf-pair_X-pc_spi-fast_matchup-{matchup_id}.npy',
        pair_pc_results)

pair_pc_results = np.load(f'results/clf-pair_X-pc_spi-fast_'
                          f'matchup-{matchup_id}.npy',
                          allow_pickle=True).item()

pair_pcs = {}
for spi_label in pair_results:
    pc_stack = []
    for pc_id in np.arange(n_pcs):
        pc_stack.append(pair_pc_results[pc_id][spi_label])
    pair_pcs[spi_label] = np.array(pc_stack).T

# Plot barplots of PC-wise cooperative/competitive decoding accuracy
for spi_label in fav_spis:
    spi_name = fav_spis[spi_label]['name']
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.bar(np.arange(142) + 1,
           np.nanmean(pair_pcs[spi_label], axis=0) - .5,
           color='.5')
    ax.set(xlabel='PC', ylabel='balanced accuracy', xlim=(0, 143),
           title=(f'cooperative vs. competitive pair classification'
                  f'\nSPI = {spi_name}'))
    yticks = np.array((.25, .5, .75, 1.))
    ax.set_ylim((yticks[0] - .5, yticks[-1] - .5))
    ax.set_yticks(yticks - .5, yticks)
    ax.axhline(0, c='goldenrod', ls='--', lw=1)
    handle = [Line2D([0], [0], color='goldenrod', lw=1,
                     ls='--', label='chance = .50')]
    ax.legend(handles=handle, loc='upper right', frameon=False)
    sns.despine()
    plt.savefig(f'figures/clf-pair_bar-{spi_label}_matchup-{matchup_id}.png',
                dpi=300, bbox_inches='tight')
            

# Compute standardized difference between cooperative and competitive
spi_npz = np.load(spi_f, allow_pickle=True)

spi_stacks = {}
ttest_results = {}
for spi_label in spi_labels:

    ttest_results[spi_label] = {}
    spi_data = spi_npz[spi_label]
    
    if spi_data.dtype != float:
        print(f'Skipping "{spi_label}" with {spi_data.dtype}')
        continue

    # Extract off-diagonal triangle(s) from SPI tables 
    spi_triu = spi_data[..., np.triu_indices(4, k=1)[0],
                             np.triu_indices(4, k=1)[1]]
    spi_tril = spi_data[..., np.tril_indices(4, k=-1)[0],
                          np.tril_indices(4, k=-1)[1]][
        ..., np.array([0, 1, 3, 2, 4, 5])]

    if np.allclose(np.nan_to_num(spi_triu), np.nan_to_num(spi_tril)):
        symmetric = True
        spi_tri = spi_triu
        print("Found symmetric SPI table; using upper triangle")
    else:
        symmetric = False
        spi_tri = np.concatenate((spi_triu, spi_tril), axis=-1)
        print("Found asymmetric SPI table; concatenating triangles")

    # Z-score data across all pairs and maps
    spi_stack = np.squeeze(np.vstack(np.split(
        spi_tri, spi_tri.shape[-1], axis=-1)))
    spi_stack = zscore(spi_stack, axis=0)
    spi_data = np.stack(np.split(
        spi_stack, spi_tri.shape[-1], axis=0), axis=-1)
    
    n_players = 4
    n_pairs = n_players * (n_players - 1) // 2
    coop_ids, comp_ids = (0, 5), (1, 2, 3, 4)
    
    if symmetric:
        coop_spis = spi_data[..., coop_ids]
        comp_spis = spi_data[..., comp_ids]
    else:
        coop_spis = spi_data[..., np.concatenate(
            (coop_ids, np.array(coop_ids) + n_pairs))]
        comp_spis = spi_data[..., np.concatenate(
            (comp_ids, np.array(comp_ids) + n_pairs))]

    coop_spis = np.squeeze(np.vstack(np.split(
        coop_spis, coop_spis.shape[-1], axis=-1)))
    comp_spis = np.squeeze(np.vstack(np.split(
        comp_spis, comp_spis.shape[-1], axis=-1)))

    t = ttest_ind(coop_spis, comp_spis, equal_var=False)
    ttest_results[spi_label]['t-values'] = t[0]
    ttest_results[spi_label]['p-values'] = t[1]
    print(f'Finished t-test for "{spi_label}"')

# Plot t-values for difference
for spi_label in fav_spis:
    spi_name = fav_spis[spi_label]['name']
    fig, ax = plt.subplots(figsize=(10, 2.5))
    t = ttest_results[spi_label]['t-values']
    ax.bar(np.arange(n_pcs) + 1, t, color='.5')
    ax.set(xlabel='PC', ylabel='t-value', xlim=(0, 143), ylim=(-60, 60),
           title=(f'standardized difference '
                  f'(cooperative \N{MINUS SIGN} competitive)'
                  f'\nSPI = {spi_name}'))
    sns.despine()
    plt.savefig(f'figures/ttest-pair_bar-{spi_label}_matchup-{matchup_id}.png',
                dpi=300, bbox_inches='tight')
            

# Loop through cooperative SPIs for win/loss classification
def win_classification(spi_npz, pc_id=None):
    
    wins = np.load(f'results/wins_matchup-{matchup_id}.npy')[:, repeat_id]
    
    results = dict.fromkeys(spi_labels + ['pearsonr'])
    for spi_label in results.keys():
        print(f'Running classification on "{spi_label}"')

        spi_data = spi_npz[spi_label]
        if pc_id is not None:
            spi_data = spi_data[:, pc_id]
            spi_data = spi_data[:, np.newaxis, ...]

        if spi_data.dtype != float:
            print(f'Skipping "{spi_label}" with {spi_data.dtype}')
            continue

        # Extract off-diagonal triangle(s) from SPI tables 
        spi_triu = spi_data[..., np.triu_indices(4, k=1)[0],
                                 np.triu_indices(4, k=1)[1]]
        spi_tril = spi_data[..., np.tril_indices(4, k=-1)[0],
                              np.tril_indices(4, k=-1)[1]][
            ..., np.array([0, 1, 3, 2, 4, 5])]

        if np.allclose(np.nan_to_num(spi_triu), np.nan_to_num(spi_tril)):
            symmetric = True
            spi_tri = spi_triu
            print("Found symmetric SPI table; using upper triangle")
        else:
            symmetric = False
            spi_tri = np.concatenate((spi_triu, spi_tril), axis=-1)
            print("Found asymmetric SPI table; concatenating triangles")

        # Classify cooperative vs competitive coupling
        n_players = 4
        n_pairs = n_players * (n_players - 1) // 2
        coop_ids = (0, 5)
        
        if not symmetric:
            n_pairs = n_players * (n_players - 1) // 2 * 2
            coop_ids = np.concatenate((coop_ids, np.array(coop_ids) + 6))

        spi_long, win_ids, map_ids = [], [], []
        for map_id in np.arange(n_maps):
            for p, pair_id in enumerate(np.arange(len(coop_ids))):
                spi_long.append(spi_tri[map_id, :, pair_id])
                map_ids.append(map_id)
                                     
                if not symmetric and p >= 2:
                    win_ids.append(wins[map_id, pair_id - 2])
                else:
                    win_ids.append(wins[map_id, pair_id])

        assert len(spi_long) == len(win_ids) == len(map_ids)
        spi_long = np.nan_to_num(spi_long)
        win_ids, map_ids = np.array(win_ids), np.array(map_ids)

        # Initialize simple classifier
        clf = LogisticRegression(class_weight='balanced')

        # Set up leave-one-map-out cross-validation
        cv = PredefinedSplit(map_ids)

        scores = []
        for f, (train, test) in enumerate(cv.split()):

            # Z-score training SPIs and test SPIs (based on training set)
            scaler = StandardScaler()
            spi_train = scaler.fit_transform(spi_long[train])
            spi_test = scaler.transform(spi_long[test])

            # Train competitive/cooperative classifier based on train SPIs
            clf.fit(spi_train, win_ids[train])

            # Use fitted classifier to predict test cooperative/competitive 
            pred = clf.predict(spi_test)

            # Evaluate classifier accuracy
            score = balanced_accuracy_score(win_ids[test], pred)
            scores.append(score)
            #print(f"Fold {f} accuracy: {score:.3f} ({spi_label})")

        results[spi_label] = np.array(scores)
        print(f'Mean "{spi_label}" accuracy: {np.mean(score):.3f}\n')
        
    return results

win_results = win_classification(spi_npz)

np.save(f'results/clf-win_X-all_spi-fast_matchup-{matchup_id}.npy',
        win_results)

win_results = np.load(f'results/clf-win_X-all_spi-fast_'
                      f'matchup-{matchup_id}.npy', allow_pickle=True).item()

# Compute mean accuracy for each SPI
win_means = {}
for l, s in win_results.items():
    if s is not None:
        win_means[l] = np.mean(s)
        
win_ranks = {k: win_means[k] for k in
             np.array(list(win_means.keys()))[
                 np.argsort(list(win_means.values()))[::-1]]}

for spi_label in fav_spis:
    print(f"Accuracy for {fav_spis[spi_label]['name']}: "
          f"{win_means[spi_label]:.3f}\n\t"
          f"{fav_spis[spi_label]['keywords']}")    

# Plot histogram of win classification performance
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(win_means.values(), bins=15, binrange=(.25, 1.0), 
             stat='probability', palette='Greys', legend=False, ax=ax)
ax.set(xlabel='balanced accuracy', ylabel='proportion of SPIs',
       title='win classification',
       xlim=(.25, 1.), ylim=(0, .5))
ax.vlines(.5, ymin=0, ymax=.5, color='goldenrod', ls='--', lw=1, zorder=0)
ax.annotate('chance = .50', xy=(.515, .42), xycoords='data', va='top')
sns.despine()
plt.savefig(f'figures/clf-win_histogram_matchup-{matchup_id}.png',
            dpi=300, bbox_inches='tight')

# Run win classification per PC
win_pc_results = {}
for pc_id in np.arange(n_pcs):
    win_pc_result = win_classification(spi_npz, pc_id=pc_id)
    win_pc_results[pc_id] = win_pc_result
    print(f"Finished classification for PC{pc_id + 1}")

np.save(f'results/clf-win_X-pc_spi-fast_matchup-{matchup_id}.npy',
        win_pc_results)

win_pc_results = np.load(f'results/clf-win_X-pc_spi-fast_'
                         f'matchup-{matchup_id}.npy', allow_pickle=True).item()

win_pcs = {}
for spi_label in win_results:
    pc_stack = []
    for pc_id in np.arange(n_pcs):
        pc_stack.append(win_pc_results[pc_id][spi_label])
    win_pcs[spi_label] = np.array(pc_stack).T

# Plot barplots of PC-wise win decoding accuracy
for spi_label in fav_spis:
    spi_name = fav_spis[spi_label]['name']
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.bar(np.arange(142) + 1,
           np.nanmean(win_pcs[spi_label], axis=0) - .5,
           color='.5')
    ax.set(xlabel='PC', ylabel='balanced accuracy', xlim=(0, 143),
           title=(f'win classification'
                  f'\nSPI = {spi_name}'))
    yticks = np.array((.1, .2, .3, .4, .5, .6, .7, .8, .9))
    ax.set_ylim((yticks[0] - .5, yticks[-1] - .5))
    ax.set_yticks(yticks - .5, yticks)
    ax.axhline(0, c='goldenrod', ls='--', lw=1)
    handle = [Line2D([0], [0], color='goldenrod', lw=1,
                     ls='--', label='chance = .50')]
    ax.legend(handles=handle, loc='upper right', frameon=False)
    sns.despine()
    plt.savefig(f'figures/clf-win_bar-{spi_label}_matchup-{matchup_id}.png',
                dpi=300, bbox_inches='tight')
    
    
# Loop through cooperative SPIs for score prediction
def score_regression(spi_npz, pc_id=None, n_splits=4):
    
    scores = np.load(f'results/scores_matchup-{matchup_id}.npy')[:, repeat_id]
    
    results = dict.fromkeys(spi_labels + ['pearsonr'])
    for spi_label in results.keys():
        print(f'Running regression on "{spi_label}"')

        spi_data = spi_npz[spi_label]
        if pc_id is not None:
            spi_data = spi_data[:, pc_id]
            spi_data = spi_data[:, np.newaxis, ...]

        if spi_data.dtype != float:
            print(f'Skipping "{spi_label}" with {spi_data.dtype}')
            continue

        # Extract off-diagonal triangle(s) from SPI tables 
        spi_triu = spi_data[..., np.triu_indices(4, k=1)[0],
                                 np.triu_indices(4, k=1)[1]]
        spi_tril = spi_data[..., np.tril_indices(4, k=-1)[0],
                              np.tril_indices(4, k=-1)[1]][
            ..., np.array([0, 1, 3, 2, 4, 5])]

        if np.allclose(np.nan_to_num(spi_triu), np.nan_to_num(spi_tril)):
            symmetric = True
            spi_tri = spi_triu
            print("Found symmetric SPI table; using upper triangle")
        else:
            symmetric = False
            spi_tri = np.concatenate((spi_triu, spi_tril), axis=-1)
            print("Found asymmetric SPI table; concatenating triangles")

        # Classify cooperative vs competitive coupling
        n_players = 4
        n_pairs = n_players * (n_players - 1) // 2
        coop_ids = (0, 5)
        
        if not symmetric:
            n_pairs = n_players * (n_players - 1) // 2 * 2
            coop_ids = np.concatenate((coop_ids, np.array(coop_ids) + 6))

        spi_long, score_ids, map_ids = [], [], []
        for map_id in np.arange(n_maps):
            for p, pair_id in enumerate(np.arange(len(coop_ids))):
                spi_long.append(spi_tri[map_id, :, pair_id])
                map_ids.append(map_id)
                                     
                if not symmetric and p >= 2:
                    score_ids.append(scores[map_id, pair_id - 2])
                else:
                    score_ids.append(scores[map_id, pair_id])

        assert len(spi_long) == len(score_ids) == len(map_ids)
        spi_long = np.nan_to_num(spi_long)
        score_ids, map_ids = np.array(score_ids), np.array(map_ids)

        # Initialize simple classifier
        reg = LinearRegression()

        # Set up leave-one-map-out cross-validation
        cv = KFold(n_splits=n_splits)

        reg_scores = []
        for f, (train, test) in enumerate(cv.split(spi_long)):

            # Z-score training SPIs and test SPIs (based on training set)
            scaler = StandardScaler()
            spi_train = scaler.fit_transform(spi_long[train])
            spi_test = scaler.transform(spi_long[test])

            # Train competitive/cooperative classifier based on train SPIs
            reg.fit(spi_train, score_ids[train])

            # Use fitted classifier to predict test cooperative/competitive 
            pred = reg.predict(spi_test)

            # Evaluate classifier accuracy
            reg_score = spearmanr(score_ids[test], pred)[0]
            reg_scores.append(reg_score)
            #print(f"Fold {f} accuracy: {score:.3f} ({spi_label})")

        results[spi_label] = np.array(reg_scores)
        print(f'Mean "{spi_label}" accuracy: {np.mean(reg_score):.3f}\n')
        
    return results

n_splits = 4
score_results = score_regression(spi_npz, n_splits=n_splits)

np.save(f'results/reg-score_X-all_spi-fast_matchup-{matchup_id}.npy',
        score_results)

score_results = np.load(f'results/reg-score_X-all_spi-fast_'
                        f'matchup-{matchup_id}.npy', allow_pickle=True).item()

# Compute mean performance for each SPI
score_means = {}
for l, s in score_results.items():
    if s is not None:
        score_means[l] = np.mean(s)
        
score_ranks = {k: score_means[k] for k in 
               np.array(list(score_means.keys()))[
                   np.argsort(list(score_means.values()))[::-1]]}

for spi_label in fav_spis:
    print(f"Correlation for {fav_spis[spi_label]['name']}: "
          f"{win_means[spi_label]:.3f}\n\t"
          f"{fav_spis[spi_label]['keywords']}")  

# Plot histogram of win classification performance
fig, ax = plt.subplots(figsize=(6, 4))
sns.histplot(score_means.values(), bins=16, binrange=(-.4, .4), 
             stat='probability', palette='Greys', legend=False, ax=ax)
ax.set(xlabel='Spearman correlation\n(predicted vs. actual)',
       ylabel='proportion of SPIs', ylim=(0, .16),
       title=f'score prediction ({n_splits}-fold OLS regression)')
sns.despine()
plt.savefig(f'figures/clf-score_histogram_matchup-{matchup_id}.png',
            dpi=300, bbox_inches='tight')

# Run score regression per PC
score_pc_results = {}
for pc_id in np.arange(n_pcs):
    score_pc_result = score_regression(spi_npz, pc_id=pc_id, n_splits=n_splits)
    score_pc_results[pc_id] = score_pc_result
    print(f"Finished classification for PC{pc_id + 1}")

np.save(f'results/reg-score_X-pc_spi-fast_matchup-{matchup_id}.npy',
        score_pc_results)

score_pc_results = np.load(f'results/reg-score_X-pc_spi-fast_'
                           f'matchup-{matchup_id}.npy',
                           allow_pickle=True).item()

score_pcs = {}
for spi_label in score_results:
    pc_stack = []
    for pc_id in np.arange(n_pcs):
        pc_stack.append(score_pc_results[pc_id][spi_label])
    score_pcs[spi_label] = np.array(pc_stack).T

# Plot barplots of PC-wise score decoding accuracy
for spi_label in fav_spis:
    spi_name = fav_spis[spi_label]['name']
    fig, ax = plt.subplots(figsize=(10, 2.5))
    ax.bar(np.arange(142) + 1,
           np.nanmean(score_pcs[spi_label], axis=0),
           color='.5')
    ax.set(xlabel='PC', xlim=(0, 143), ylim=(-.4, .4),
           ylabel='Spearman correlation\n(predicted vs. actual)',
           title=(f'score prediction ({n_splits}-fold OLS regression)'
                  f'\nSPI = {spi_name}'))
    sns.despine()
    plt.savefig(f'figures/reg-score_bar-{spi_label}_matchup-{matchup_id}.png',
                dpi=300, bbox_inches='tight')
