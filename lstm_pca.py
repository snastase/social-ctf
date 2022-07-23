from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.linalg import eigh
import pandas as pd
from itertools import combinations
from statistical_tests import bootstrap_test, fisher_mean
from statsmodels.stats.multitest import multipletests

# Load helper function(s) for interacting with CTF dataset
from ctf_dataset.load import create_wrapped_dataset

base_dir = '/jukebox/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data_v1')

# Create wrapped CTF dataset
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

# Hard-code some dataset variables
matchup_id = 0
n_maps = 32
n_repeats = 32
n_players = 4
n_lstms = 512
n_samples = 4501

# Load in LSTMs by game, apply tanh, z-score, and save for memmapping
lstm = 'lstm'
n_total = n_maps * n_repeats * n_players * n_samples

lstms_stack = np.memmap(f'results/lstms-stack_tanh-z_matchup-{matchup_id}.npy',
                        dtype='float64', mode='w+', shape=(n_total, n_lstms))
print(f"Populating stacked LSTMs (shape: {lstms_stack.shape})")

game_counter = 0
for map_id in np.arange(n_maps):    
    for repeat_id in np.arange(n_repeats):
        for player_id in np.arange(n_players):
            lstms = wrap_f[f'map/matchup/repeat/player/time/{lstm}'][
                map_id, matchup_id, repeat_id,
                player_id, ...].astype(np.float64)
            lstms = zscore(np.tanh(lstms), axis=0)
            start = game_counter * n_samples
            end = start + n_samples
            lstms_stack[start:end] = lstms
            lstms_stack.flush()
            game_counter += 1
            print(f"Finished stacking map {map_id}, "
                  f"repeat {repeat_id}, player {player_id}")

# Function for reduced-memory incremental eigendecomposition
def incremental_pca(data, n_components=None, n_samples=None, filename=None):

    # If number of samples isn't specified, infer from data
    n_features = data.shape[1]
    if not n_samples:
        n_samples = data.shape[0]
    
    # Incrementally populate covariance matrix
    cov = np.zeros((n_features, n_features))
    game_count = 0
    for i, row in zip(np.arange(n_samples), data):
        outer = np.outer(row, row)
        cov += outer / (n_samples - 1)
        if (i + 1) % (4 * 4501) == 0:
            print(f"Finished computing game {game_count} covariance")
            game_count += 1

    # Recover eignvalues and eigenvectors
    vals, vecs = eigh(cov)

    # Reorder values, vectors and extract column eigenvectors
    vals, vecs = vals[::-1], vecs[:, ::-1]
    if n_components:
        vecs = vecs[:, :n_components]

    # Project data onto the eigenvectors
    if filename:
        proj = np.memmap(filename, dtype='float64',
                         mode='w+', shape=(n_total, n_lstms))

    proj[:] = data @ vecs
    proj.flush()
    print("Finished running incremental PCA")

    return proj, vals, vecs

pca_filename = f'results/lstms-stack_tanh-z_pca-proj_matchup-{matchup_id}.npy'
proj, vals, vecs = incremental_pca(lstms_stack, n_components=n_lstms,
                                   n_samples=n_total,
                                   filename=pca_filename)

np.save(f'results/lstms-stack_tanh-z_pca-vals_matchup-{matchup_id}.npy', vals)
np.save(f'results/lstms-stack_tanh-z_pca-vecs_matchup-{matchup_id}.npy', vecs)


# Compute proportion variance explained from eigenvalues
vaf = vals / np.sum(vals)

# Compute number of components required for percentage variance
percents = [.5, .75, .9, .95, .99]
for i, perc in enumerate(percents):
    k = np.sum(np.cumsum(vaf) <= perc) + 1
    print(f"{perc:.0%} variance: {k} PCs")

print(f"100 PCs: {np.cumsum(vaf)[100]:.0%}")


# Un-stack PCA-transformed data for maps, repeats, players
game_counter = 0
for map_id in np.arange(n_maps):    
    for repeat_id in np.arange(n_repeats):
        start = game_counter * n_players * n_samples
        end = start + n_players * n_samples
        proj_game = proj[start:end]
        proj_game = np.stack(np.split(proj_game, 4), axis=0)   
        game_counter += 1
        np.save((f'results/lstms-pca_matchup-{matchup_id}_map-{map_id}_'
                 f'repeat-{repeat_id}.npy'), proj_game)
        print(f"Extracted map {map_id}, repeat {repeat_id} "
              f"(stacked rows {start} to {end})")

    
# Plot scree plot of variance accounted
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

pca_k = 142
vaf_cum = np.cumsum(vaf)
dimensions = np.arange(1, 513)

percents = [.9, .95, .99]
percents_vaf = {}
for i, perc in enumerate(percents):
    k = np.sum(vaf_cum <= perc) + 1     
    percents_vaf[k] = perc

fig, ax = plt.subplots(figsize=(5.5, 4.5))
ax.scatter(dimensions, vaf, color='.5')
ax.scatter(dimensions[:pca_k], vaf[:pca_k], color='tab:red')
ax.set_xlabel('dimensions', size=12)
ax.set_ylabel('proportion of\nvariance explained', size=12)
for k, perc in percents_vaf.items():
    ax.axvline(k, 0, .35, color='.5', zorder=-1)
    ax.annotate(f'{perc:.0%}', xy=(k + 10, .37), ha='center',
                xycoords=('data', 'axes fraction'))
axins = inset_axes(ax, width=1.2, height=1)
axins.scatter(dimensions, vaf_cum, color='.5')
axins.scatter(dimensions[:pca_k], vaf_cum[:pca_k], color='tab:red')
axins.xaxis.set_ticks([])
axins.yaxis.set_ticks([])
axins.set_xlabel('dimensions', size=12)
axins.set_ylabel('cumulative\nvariance', size=12)
plt.savefig(f'figures/scree_pca-k{pca_k}_matchup-{matchup_id}.png', dpi=300,
            bbox_inches='tight')


# Plot PC time series for individual games (with simple ISC)
from scipy.stats import pearsonr

matchup_id = 0
map_id = 0
repeat_id = 0

for pc_id in np.arange(10):
    lstms_pc = np.load(f'results/lstms-pca_matchup-{matchup_id}_'
                       f'map-{map_id}_repeat-{repeat_id}.npy')[..., pc_id]

    fig, axs = plt.subplots(2, 1, figsize=(12, 3))
    axs[0].plot(lstms_pc[0], c='darkred', alpha=.7)
    axs[0].plot(lstms_pc[1], c='coral', alpha=.7)
    axs[0].set(xticks=[], ylabel='activation', xlim=(0, 4500))
    axs[0].set_title(f'PC{pc_id + 1} (map {map_id}, repeat {repeat_id})')
    axs[0].annotate(f'ISC: {pearsonr(lstms_pc[0], lstms_pc[1])[0]:.3f}', (.99, 1),
                    ha='right', va='bottom', xycoords='axes fraction')
    axs[1].plot(lstms_pc[2], c='darkblue', alpha=.7)
    axs[1].plot(lstms_pc[3], c='lightseagreen', alpha=.7)
    axs[1].set(xticks=[], ylabel='activation', xlim=(0, 4500))
    axs[1].annotate(f'ISC: {pearsonr(lstms_pc[2], lstms_pc[3])[0]:.3f}', (.99, 1),
                    ha='right', va='bottom', xycoords='axes fraction')
    sns.despine()
    plt.savefig((f'figures/pca-ts_pc-{pc_id + 1}_matchup-{matchup_id}_'
                 f'map-{map_id}_repeat-{repeat_id}.png'),
                dpi=300, bbox_inches='tight')


# Horizontally stack cooperative pairs of players and compute joint PCA
lstm = 'lstm'
n_total = n_maps * n_repeats * n_players // 2 * n_samples

lstms_coop_stack = np.memmap(f'results/lstms-coop-stack_tanh-z_'
                             f'matchup-{matchup_id}.npy',
                             dtype='float64', mode='w+',
                             shape=(n_total, n_lstms * 2))
print("Populating cooperatively stacked LSTMs" 
      f"(shape: {lstms_coop_stack.shape})")

game_counter = 0
for map_id in np.arange(n_maps):    
    for repeat_id in np.arange(n_repeats):
        lstms = wrap_f[f'map/matchup/repeat/player/time/{lstm}'][
            map_id, matchup_id, repeat_id, ...].astype(np.float64)
        lstms = zscore(np.tanh(lstms), axis=1)
        lstms_coop = np.vstack((np.hstack((lstms[0], lstms[1])),
                                np.hstack((lstms[2], lstms[3]))))
        start = game_counter * n_samples * 2
        end = start + n_samples * 2
        lstms_coop_stack[start:end] = lstms_coop
        lstms_coop_stack.flush()
        game_counter += 1
        print(f"Finished stacking map {map_id}, "
              f"repeat {repeat_id}")

pca_filename = (f'results/lstms-coop-stack_tanh-z_'
                f'pca-proj_matchup-{matchup_id}.npy')
proj, vals, vecs = incremental_pca(lstms_coop_stack,
                                   n_components=n_lstms * 2,
                                   n_samples=n_total,
                                   filename=pca_filename)

np.save((f'results/lstms-coop-stack_tanh-z_'
         f'pca-vals_matchup-{matchup_id}.npy'), vals)
np.save((f'results/lstms-coop-stack_tanh-z_'
         f'pca-vecs_matchup-{matchup_id}.npy'), vecs)


# Compute proportion variance explained from eigenvalues
vaf = vals / np.sum(vals)

# Compute number of components required for percentage variance
percents = [.5, .75, .9, .95, .99]
for i, perc in enumerate(percents):
    k = np.sum(np.cumsum(vaf) <= perc) + 1
    print(f"{perc:.0%} variance: {k} PCs")

print(f"100 PCs: {np.cumsum(vaf)[100]:.0%}")
        

# Horizontally stack competitive pairs of players and compute joint PCA
lstm = 'lstm'
n_total = n_maps * n_repeats * n_players // 2 * n_samples

pairs = [((0, 2), (1, 3)), ((0, 3), (1, 2))]
for p, pair in enumerate(pairs):
    lstms_comp_stack = np.memmap(f'results/lstms-comp{p + 1}-stack_'
                                 f'tanh-z_matchup-{matchup_id}.npy',
                                 dtype='float64', mode='w+',
                                 shape=(n_total, n_lstms * 2))
    print(f"Populating competitively stacked LSTMs (pairing {p + 1}; " 
          f"shape: {lstms_comp_stack.shape})")

    game_counter = 0
    for map_id in np.arange(n_maps):    
        for repeat_id in np.arange(n_repeats):
            lstms = wrap_f[f'map/matchup/repeat/player/time/{lstm}'][
                map_id, matchup_id, repeat_id, ...].astype(np.float64)
            lstms = zscore(np.tanh(lstms), axis=1)
            lstms_comp = np.vstack((np.hstack((lstms[pair[0][0]],
                                               lstms[pair[0][1]])),
                                    np.hstack((lstms[pair[1][0]],
                                               lstms[pair[1][1]]))))
            start = game_counter * n_samples * 2
            end = start + n_samples * 2
            lstms_comp_stack[start:end] = lstms_comp
            lstms_comp_stack.flush()
            game_counter += 1
            print(f"Finished stacking map {map_id}, "
                  f"repeat {repeat_id}, pairing {p + 1}")

    pca_filename = (f'results/lstms-comp{p + 1}-stack_tanh-z_'
                    f'pca-proj_matchup-{matchup_id}.npy'
    proj, vals, vecs = incremental_pca(lstms_comp_stack,
                                       n_components=n_lstms * 2,
                                       n_samples=n_total,
                                       filename=pca_filename)

    np.save(f'results/lstms-comp{p + 1}-stack_tanh-z_'
            f'pca-vals_matchup-{matchup_id}.npy', vals)
    np.save(f'results/lstms-comp{p + 1}-stack_tanh-z_'
            f'pca-vecs_matchup-{matchup_id}.npy', vecs)


    # Compute proportion variance explained from eigenvalues
    vaf = vals / np.sum(vals)

    # Compute number of components required for percentage variance
    percents = [.5, .75, .9, .95, .99]
    for i, perc in enumerate(percents):
        k = np.sum(np.cumsum(vaf) <= perc) + 1
        print(f"{perc:.0%} variance: {k} PCs")

    print(f"100 PCs: {np.cumsum(vaf)[100]:.0%}")
