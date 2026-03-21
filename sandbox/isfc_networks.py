from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
import pandas as pd
from brainiak.isc import isfc
from statistical_tests import fisher_mean, bootstrap_test, squareform_fdr
from matplotlib.animation import ArtistAnimation

# Load helper function(s) for interacting with CTF dataset
from ctf_dataset.load import create_wrapped_dataset

base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data')


# Create wrapped CTF dataset
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

n_lstms = 512
n_repeats = 8
n_players = 4
n_pairs = n_players * (n_players - 1) // 2
n_samples = 4501

map_id = 0 # 0
matchup_id = 0 # 0-54
repeat_id = 0 # 0-7
player_id = 0 # 0-3


# Load pre-saved PCA's
n_pairs = n_players * (n_players - 1) // 2
n_matchups = 4
k = 100
lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')


# Loop through matchups and repeats and compute ISFC
isfc_results = np.zeros((n_matchups, n_repeats, n_pairs,
                         lstms_pca.shape[-1],
                         lstms_pca.shape[-1]))
for matchup in np.arange(n_matchups):
    for repeat in np.arange(n_repeats):
    
        lstms = lstms_pca[matchup, repeat, ...]
        lstms = np.rollaxis(lstms, 0, 3)

        # Compute ISCs between each pair for 4 agents
        isfcs = isfc(lstms, pairwise=True, vectorize_isfcs=False)
        isfc_results[matchup, repeat, ...] = isfcs
        
        print(f"Computed ISFC for matchup {matchup} (repeat {repeat})")

np.save(f'results/isfc_lstm_tanh-z_pca-k{k}.npy', isfc_results)


# Compute ISFC based on confound regression PCs
n_pairs = n_players * (n_players - 1) // 2
n_matchups = 4
k = 100
reg = 'com'
lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}_reg-{reg}.npy')

# Loop through matchups and repeats and compute ISFC
isfc_results = np.zeros((n_matchups, n_repeats, n_pairs,
                         lstms_pca.shape[-1],
                         lstms_pca.shape[-1]))
for matchup in np.arange(n_matchups):
    for repeat in np.arange(n_repeats):
    
        lstms = lstms_pca[matchup, repeat, ...]
        lstms = np.rollaxis(lstms, 0, 3)

        # Compute ISCs between each pair for 4 agents
        isfcs = isfc(lstms, pairwise=True, vectorize_isfcs=False)
        isfc_results[matchup, repeat, ...] = isfcs
        
        print(f"Computed ISFC for matchup {matchup} (repeat {repeat})")

np.save(f'results/isfc_lstm_tanh-z_pca-k{k}_reg-{reg}.npy',
        isfc_results)


# Load and plot ISFC results
reg = 'com'
if reg:
    isfcs = np.load(f'results/isfc_lstm_tanh-z_pca-k{k}_reg-{reg}.npy')
else:
    isfcs = np.load(f'results/isfc_lstm_tanh-z_pca-k{k}.npy')

coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

# Average across repeats and pairs
isfcs_coop = np.mean(np.mean(isfcs[:, :, coop_ids, ...],
                             axis=2),
                     axis=1)
isfcs_comp = np.mean(np.mean(isfcs[:, :, comp_ids, ...],
                             axis=2),
                     axis=1)

# Simple ISFC matrix plot
matchup = 0
fig, axs = plt.subplots(1, 2, figsize=(10, 8))
ax0 = axs[0].matshow(isfcs_coop[matchup], vmin=-.2, vmax=.2, cmap='RdBu_r')
axs[0].xaxis.set_ticks([])
axs[0].yaxis.set_ticks([])
axs[0].set_xlabel('LSTM PCs')
axs[0].xaxis.set_label_position('top') 
axs[0].set_ylabel('LSTM PCs')
axs[0].set_title('cooperative ISFC', y=0, pad=-20)
ax1 = axs[1].matshow(isfcs_comp[matchup], vmin=-.2, vmax=.2, cmap='RdBu_r')
axs[1].xaxis.set_ticks([])
axs[1].yaxis.set_ticks([])
axs[1].set_xlabel('LSTM PCs')
axs[1].xaxis.set_label_position('top') 
axs[1].set_ylabel('LSTM PCs')
axs[1].set_title('competitive ISFC', y=0, pad=-20)
cbar = fig.colorbar(ax1, ax=axs.ravel().tolist(),
                    fraction=0.0215, pad=0.04)
cbar.set_label('correlation', rotation=270, verticalalignment='bottom')
plt.savefig('figures/isfc_coop-comp_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')


# Run bootstrap hypothesis test to assess significance
matchup = 0
isfcs_test = (np.mean(isfcs[matchup, :, coop_ids], axis=0) -
              np.mean(isfcs[matchup, :, comp_ids], axis=0))

observed, ci, p, distribution = bootstrap_test(isfcs_test,
                                               bootstrap_axis=0,
                                               n_bootstraps=1000,
                                               estimator=fisher_mean,
                                               ci_percentile=95,
                                               side='two-sided')

# FDR correction on p-values
fdr_p = squareform_fdr(p)


# Difference between cooperative and competitive ISFC matrices
matchup = 0
isfcs_diff = isfcs_coop[matchup] - isfcs_comp[matchup]
isfcs_mask = np.full(fdr_p.shape, np.nan)
isfcs_mask[fdr_p > .05] = 1

fig, axs = plt.subplots(1, 2, figsize=(10, 8))
ax0 = axs[0].matshow(isfcs_diff, vmin=-.2, vmax=.2, cmap='RdBu_r')
axs[0].set_xticks([])
axs[0].set_yticks([])
axs[0].set_xlabel('LSTM PCs')
axs[0].xaxis.set_label_position('top') 
axs[0].set_ylabel('LSTM PCs')
axs[0].set_title('cooperative – competitive ISFC', y=0, pad=-15,
                 verticalalignment='top')
ax1 = axs[1].matshow(isfcs_diff, vmin=-.2, vmax=.2, cmap='RdBu_r')
axs[1].matshow(isfcs_mask, vmin=0, vmax=1, cmap='binary_r')
axs[1].set_xticks([])
axs[1].set_yticks([])
axs[1].set_xlabel('LSTM PCs')
axs[1].xaxis.set_label_position('top') 
axs[1].set_ylabel('LSTM PCs')
axs[1].set_title('cooperative – competitive ISFC\n(p < .05, FDR corrected)',
                 y=0, pad=-15, verticalalignment='top')
cbar = fig.colorbar(ax1, ax=axs.ravel().tolist(),
                    fraction=0.0215, pad=0.04)
cbar.set_label('difference in correlation',
               rotation=270, verticalalignment='bottom')
plt.savefig('figures/isfc_diff-fdr_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')

    
# First pass visualize ISFC networks with scipy/seaborn
from scipy.cluster.hierarchy import fcluster, linkage
matchup = 3

isfcs_dist_coop = isfcs_coop[matchup].copy()
isfcs_dist_coop = 1 - isfcs_dist_coop 
Z = linkage(squareform(isfcs_dist_coop, checks=False),
            method='average')
g = sns.clustermap(isfcs_dist_coop, cmap='RdBu',
                   row_linkage=Z, col_linkage=Z,
                   vmin=.5, vmax=1.5)
g.ax_heatmap.tick_params(right=False, bottom=False, labelsize=False)


# First let's convert the ISFCs to a distance matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn import metrics

n_matchups = 4
n_units = 100
isfcs_coop_dist = 1 - isfcs_coop

cluster_ks = np.arange(2, n_units + 1)
cluster_labels = {}

# Hierarchical agglomerative clustering
linkage = 'average'
cluster_labels = {}

for m in np.arange(n_matchups):
    cluster_labels[m] = {}
    for k in cluster_ks:
        model = AgglomerativeClustering(n_clusters=k,
                                        affinity='precomputed',
                                        linkage=linkage)
        labels = model.fit(isfcs_coop_dist[m]).labels_
        cluster_labels[m][k] = labels
        print(f"Finishered clustering matchup {m} with {k} clusters")

# Let's try some no-ground-truth metric silhouette score
label_scores = {}

scorers = {'silhouette': metrics.silhouette_score}

scorer = 'silhouette'

if scorer not in label_scores:
    label_scores[scorer] = {}
    for m in np.arange(n_matchups):

        label_scores[scorer][m] = {}
        isfcs_coop_diag = isfcs_coop_dist[m].copy()
        np.fill_diagonal(isfcs_coop_diag, 0)

        for k in list(cluster_labels[m].keys())[:-1]:
            score = scorers[scorer](isfcs_coop_diag,
                                    cluster_labels[m][k],
                                    metric='precomputed')
            label_scores[scorer][m][k] = score
            print(f'Matchup {m} at {k} clusters {scorer}: {score}')

# Plot silhouette score
label_scores_long = {'score': [], 'metric': [], 'population': [],
                     'k': []}

pops = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

for scorer in label_scores:
    for m in label_scores[scorer]:
        for k in label_scores[scorer][m]:
            label_scores_long['score'].append(
                label_scores[scorer][m][k])
            label_scores_long['metric'].append(scorer)
            label_scores_long['population'].append(pops[m])
            label_scores_long['k'].append(k)
label_scores_long = pd.DataFrame(label_scores_long)

sns.set(font_scale=1.2, style='white')
sns.relplot(data=label_scores_long, x='k', y='score',
            col='population', col_wrap=2,
            kind='line')
        
# Hierarchical clustering with leave-one-repeat-out
isfcs_coop_reps = np.mean(isfcs[:, :, coop_ids, ...],
                          axis=2)

n_matchups = 4
n_reps = 8

cluster_labels_cv = {}
for m in np.arange(n_matchups):
    cluster_labels_cv[m] = {}
    for test_rep in np.arange(n_reps):
        train_reps = [r for r in np.arange(n_reps) if r != test_rep]
        isfcs_coop_train = 1 - np.mean(isfcs_coop_reps[m, train_reps, ...],
                                       axis=0)

        cluster_labels_cv[m][test_rep] = {}
        for k in cluster_ks:
            model = AgglomerativeClustering(n_clusters=k,
                                            affinity='precomputed',
                                            linkage=linkage)
            labels = model.fit(isfcs_coop_train).labels_
            cluster_labels_cv[m][test_rep][k] = labels
            print(f"Finishered clustering matchup {m} test "
                  f"repeat {test_rep} at {k} clusters")

np.save('results/isfc_coop_cluster_labels_cv_lstm_tanh.npy', cluster_labels_cv)

cluster_labels_cv = np.load('results/isfc_coop_cluster_labels_cv_lstm_tanh.npy',
                            allow_pickle=True).item()

# Evaluate training cluster labels against test repeats
label_scores_cv = {}

scorers_cv = {'Rand': metrics.adjusted_rand_score,
              'AMI': metrics.adjusted_mutual_info_score}

scorer = 'Rand'

if scorer not in label_scores_cv:
    label_scores_cv[scorer] = {}
    for m in np.arange(n_matchups):
        label_scores_cv[scorer][m] = {}
        for test_rep in np.arange(n_reps):
                isfcs_coop_test = 1 - isfcs_coop_reps[m, test_rep, ...]
                label_scores_cv[scorer][m][test_rep] = {}
                for k in cluster_ks:
                    model = AgglomerativeClustering(n_clusters=k,
                                                    affinity='precomputed',
                                                    linkage=linkage)
                    test_labels = model.fit(isfcs_coop_test).labels_

                    score = scorers_cv[scorer](
                        test_labels,
                        cluster_labels_cv[m][test_rep][k])
                    
                    label_scores_cv[scorer][m][test_rep][k] = score
                    print(f"Finished matchup {m} for test "
                          f"repeat {test_rep} at N = {k}")

np.save('results/isfc_coop_cluster_scores_cv_lstm_tanh.npy', label_scores_cv)

label_scores_cv = np.load('results/isfc_coop_cluster_scores_cv_lstm.npy',
                          allow_pickle=True).item()

# Plot evaluation metric across number of clusters
label_scores_long_cv = {'score': [], 'metric': [], 'population': [],
                        'repeat': [], 'k': []}

pops = {0: 'A', 1: 'B', 2: 'C', 3: 'D'}

for scorer in label_scores_cv:
    for m in label_scores_cv[scorer]:
        for test_rep in label_scores_cv[scorer][m]:
            for k in label_scores_cv[scorer][m][test_rep]:
                label_scores_long_cv['score'].append(
                    label_scores_cv[scorer][m][test_rep][k])
                label_scores_long_cv['metric'].append(scorer)
                label_scores_long_cv['population'].append(pops[m])
                label_scores_long_cv['repeat'].append(test_rep)
                label_scores_long_cv['k'].append(k)
label_scores_long_cv = pd.DataFrame(label_scores_long_cv)

scores_cv_plot = label_scores_long_cv[
    label_scores_long_cv['metric'] == 'Rand']

sns.set(font_scale=1.2, style='white')
sns.relplot(data=scores_cv_plot, x='k', y='score', hue='repeat',
            col='population', col_wrap=2,
            kind='line', palette='Blues')

sns.set(font_scale=1.2, style='white')
sns.relplot(data=label_scores_long_cv, x='k', y='score', hue='metric',
            col='population', col_wrap=2,
            kind='line')


# Load in instantaneous ISFC cofluctuations
matchup = 0
k = 100
n_repeats = 8
iscfs = []
for r in np.arange(n_repeats):
    iscf = np.load(f'results/iscf_lstm_tanh-z_pca-k{k}_m{matchup}_r{r}.npy')
    iscfs.append(iscf)
    print(f"Loaded intersubject cofluctuation for repeat {r}")
iscfs = np.array(iscfs)


# Plot animated example of windowed ISFCs
from matplotlib.animation import ArtistAnimation

repeat = 0
pair = 0
iscf_pair = iscfs[repeat, pair, ...]

fig, ax = plt.subplots()

iscf_anim = []
for t in np.arange(iscf_pair.shape[0]):
    mat = ax.imshow(iscf_pair[t], cmap='RdYlBu_r', vmin=-5, vmax=5)
    if t == 0:
        ax.imshow(iscf_pair[t], cmap='RdYlBu_r', vmin=-5, vmax=5)
        ax.axis('off')
        fig.colorbar(mat, ax=ax)
    iscf_anim.append([mat])

ani = ArtistAnimation(fig, iscf_anim, interval=67, blit=True)
ani.save(f'results/iscf_lstm_tanh-z_pca-k{k}_m{matchup}_r{r}_p{pair}_ani.mp4')
plt.show()


matchup, repeat = 0, 0
n_players = 4
n_units = 512
n_pairs = n_players * (n_players - 1) // 2
n_samples = 4501
width = 150
onsets = np.arange(n_samples - width)
n_windows = len(onsets)

win_isfc_results = np.load((f'results/isfc_win-{width}_lstm_pca-k{k}_ica_'
                            f'matchup-{matchup}_repeat-{repeat}_results.npy'))





# Vectorize square ISFC matrices first
#n_sq = n_units * (n_units - 1) // 2
n_sq = k * (k - 1) // 2
win_isfc_vec = np.zeros((n_windows, n_pairs, n_sq))
for w in np.arange(n_windows):
    for p in np.arange(n_pairs):
        win_isfc_vec[w, p] = squareform(win_isfc_results[w, p, ...],
                                         checks=False)
    if w > 0 and w % 100 == 0:
        print(f"Finished squareforming ISFCs at window {w}")
        
# Compute window-by-window correlation matrix
from scipy.spatial.distance import pdist
pair = 4
win_isfc_corr = pdist(win_isfc_vec[:, pair, :], metric='correlation')
win_isfc_corr = 1 - squareform(win_isfc_corr)

fig, ax = plt.subplots()
mat = ax.matshow(win_isfc_corr, cmap='inferno', vmin=0, vmax=.9)
ax.set_xticks([])
ax.set_xlabel('windows')
ax.xaxis.set_label_position('top') 
ax.set_ylabel('windows')
fig.colorbar(mat, ax=ax, label='correlation')


# Run PCA on vectorized windowed ISFCs
from sklearn.decomposition import PCA
from time import time

n_components = 100

win_isfcs_pca = {}
for p in np.arange(n_pairs):
    win_isfcs_pca[p] = {}
    win_isfcs_sq[:, p, :]
    pca = PCA(n_components=n_components)
    start = time()
    win_isfcs_pca[p]['pca'] = pca.fit_transform(zscore(win_isfcs_sq[:, p, :],
                                                       axis=0))
    win_isfcs_pca[p]['transformed'] = pca
    finish = time() - start
    print(f"Finished PCA on windowed ISFCs for "
          f"pair {p} ({finish:.3f} s elapsed)")

np.save(f'results/isfc_win-{width}_lstm_matchup-{matchup}_'
        f'repeat-{repeat}_pca.npy', win_isfcs_pca)

win_isfcs_pca = np.load(f'results/isfc_win-{width}_post-lstm_'
                        f'matchup-{matchup}_repeat-{repeat}_pca.npy',
                        allow_pickle=True).item()

# Scree plots for windowed ISFC w/ PCA
win_isfcs_pca_long = {'pair': [], 'k': [], 'type': [], 
                      'variance explained': []}
for p in win_isfcs_pca:
    
    if p in coop_ids:
        type = 'cooperative'
    elif p in comp_ids:
        type = 'competitive'
        
    for k, v in enumerate(
        win_isfcs_pca[p]['transformed'].explained_variance_ratio_):
        
        win_isfcs_pca_long['type'].append(type)
        win_isfcs_pca_long['pair'].append(p)
        win_isfcs_pca_long['k'].append(k + 1)
        win_isfcs_pca_long['variance explained'].append(v)
        
win_isfcs_pca_long = pd.DataFrame(win_isfcs_pca_long)

max_k = 30
win_isfcs_pca_trunc = win_isfcs_pca_long[
    win_isfcs_pca_long['k'] <= max_k]

sns.lineplot(data=win_isfcs_pca_trunc, x='k',
             y='variance explained', hue='type')

sns.lineplot(data=win_isfcs_pca_trunc[win_isfcs_pca_trunc['type'] ==
                                      'cooperative'], x='k',
             y='variance explained', hue='pair',
             palette=sns.color_palette('tab20c')[:2])
sns.lineplot(data=win_isfcs_pca_trunc[win_isfcs_pca_trunc['type'] ==
                                      'competitive'], x='k',
             y='variance explained', hue='pair',
             palette=sns.color_palette('tab20c')[4:8])


win_isfcs_pca_stack = np.vstack([win_isfcs_pca[p][
        'transformed'].explained_variance_ratio_
                                 for p in win_isfcs_pca])

sns.lineplot(data=win_isfcs_pca_stack[coop_ids, :max_k].T, palette='Blues')
sns.lineplot(data=win_isfcs_pca_stack[coop_ids, :max_k].T, palette='Oranges')


for p in coop_ids:
    plt.plot(win_isfcs_pca[p][
        'transformed'].explained_variance_ratio_[:max_k])
    
for p in comp_ids:
    plt.plot(win_isfcs_pca[p][
        'transformed'].explained_variance_ratio_[:max_k])
    
    
# Check diagonal of sliding-window ISFCs on PCA
win_isfc_results = np.load((f'results/isfc_win-{width}_lstm_pca-k{k}_ica_'
                            f'matchup-{matchup}_repeat-{repeat}_results.npy'))
fig, ax = plt.subplots(figsize=(6, 6))
mat = ax.matshow(np.mean(win_isfc_results[:, pair, ...], axis=0),
                  cmap="RdYlBu_r", vmin=-.5, vmax=.5)
ax.tick_params(axis='both', which='both', length=0)
plt.colorbar(mat, fraction=0.046, pad=0.04)

pair = 0
diags = np.diagonal(win_isfc_results[:, pair, ...], axis1=1, axis2=2)
plt.plot(diags[:, :])
plt.title('all ICs')
plt.xlabel('windows')
plt.ylabel('ISC (correlation)')
