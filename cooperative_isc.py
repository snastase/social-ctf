import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from brainiak.isc import isc
from statsmodels.stats.multitest import multipletests
from statistical_tests import bootstrap_test, fisher_mean
from coupling_metrics import lagged_isc


# Load in PCA-reduced LSTMS
k = 100
lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')


# Compute simple ISC and save
n_matchups = 4
n_repeats = 8
n_players = 4
n_pairs = n_players * (n_players - 1) // 2
iscs = np.full((n_matchups, n_repeats, n_pairs, k), np.nan)
for matchup in np.arange(n_matchups):
    for repeat in np.arange(n_repeats):
        lstms_rep = np.moveaxis(lstms_pca[matchup, repeat], 0, 2)
        iscs[matchup, repeat] = isc(lstms_rep, pairwise=True)
        print("Finished computing ISC for"
              f"matchup {matchup} repeat {repeat}")
np.save(f'results/iscs_tanh-z_pca-k{k}.npy', iscs)


# Plot cooperative/competitive ISC for top 10 PCs
matchup = 0
n_repeats = 8
pcs = np.arange(10)
sns.set_context('notebook', font_scale=1.2)
fig, axs = plt.subplots(2, 5, figsize=(25, 8))
for pc, ax in zip(pcs, axs.ravel()):
    corr = fisher_mean([np.corrcoef(lstms_pca[matchup, r, ..., pc])
                        for r in np.arange(n_repeats)], axis=0)
    sns.heatmap(corr, square=True, annot=True, vmin=-1, vmax=1,
                cmap='RdBu_r', xticklabels=False, yticklabels=False,
                fmt='.2f', ax=ax)
    ax.set_title(f'PC{pc + 1}')
plt.savefig(f'figures/isc_coop-comp_tanh-z_pca-k{k}_m{matchup}.png',
            dpi=300, bbox_inches='tight')


# Difference in cooperative/competitive ISC across PCs
matchup = 0
n_repeats = 8
n_pcs = 100

# Compute differences between cooperative and competitive ISCs
isc_diffs = []
isc_diffs_df = {'difference': [], 'PC': [], 'repeat': []}
for pc in np.arange(n_pcs):
    corrs = [np.corrcoef(lstms_pca[matchup, r, ..., pc])
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

# Bootstrap test for significance of difference
observed, ci, p, distribution = bootstrap_test(isc_diffs,
                                               bootstrap_axis=0,
                                               n_bootstraps=1000,
                                               estimator=fisher_mean,
                                               ci_percentile=95,
                                               side='two-sided')

# FDR correction of p-values
_, fdr_p, _, _ = multipletests(p, method='fdr_bh')

# Plot ISCs for 100 PCs with significance markers
sig_pos = ((fdr_p < .05) & (observed > 0)).nonzero()[0]
sig_neg = ((fdr_p < .05) & (observed < 0)).nonzero()[0]

sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 4))
sns.barplot(x='PC', y='difference', data=isc_diffs_df, ax=ax, color='.6',
            estimator=fisher_mean)
#ax.set_ylim(-.375, .325) # for matchup = 3 (sig y = -.01)
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
sns.despine()
plt.savefig(f'figures/isc_diff-bars_tanh-z_pca-k{k}_m{matchup}.png',
            dpi=300, bbox_inches='tight')


# Load in PCA-reduced LSTMs with confounds regressed out
reg = 'com' # 'pre', 'hud', 'act', or 'com'
lstms_pca_reg = np.load(f'results/lstms_tanh-z_pca-k{k}_reg-{reg}.npy')


# Compute simple ISC and save
n_matchups = 4
n_repeats = 8
n_players = 4
n_pairs = n_players * (n_players - 1) // 2
iscs = np.full((n_matchups, n_repeats, n_pairs, k), np.nan)
for matchup in np.arange(n_matchups):
    for repeat in np.arange(n_repeats):
        lstms_rep = np.moveaxis(lstms_pca_reg[matchup, repeat], 0, 2)
        iscs[matchup, repeat] = isc(lstms_rep, pairwise=True)
        print("Finished computing ISC for"
              f"matchup {matchup} repeat {repeat}")
np.save(f'results/iscs_tanh-z_pca-k{k}_reg-{reg}.npy', iscs)


# Plot cooperative/competitive ISC for top 10 PCs
matchup = 0
n_repeats = 8
pcs = np.arange(10)
sns.set_context('notebook', font_scale=1.2)
fig, axs = plt.subplots(2, 5, figsize=(25, 8))
for pc, ax in zip(pcs, axs.ravel()):
    corr = fisher_mean([np.corrcoef(lstms_pca_reg[matchup, r, ..., pc])
                        for r in np.arange(n_repeats)], axis=0)
    sns.heatmap(corr, square=True, annot=True, vmin=-1, vmax=1,
                cmap='RdBu_r', xticklabels=False, yticklabels=False,
                fmt='.2f', ax=ax)
    ax.set_title(f'PC{pc + 1}')
plt.savefig(f'figures/isc_coop-comp_tanh-z_pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')


# Compute differences between cooperative and competitive ISCs
matchup = 0
n_repeats = 8
n_pcs = 100
isc_diffs, isc_coops = [], []
isc_diffs_df = {'difference': [], 'PC': [], 'repeat': []}
for pc in np.arange(n_pcs):
    corrs = [np.corrcoef(lstms_pca_reg[matchup, r, ..., pc])
             for r in np.arange(n_repeats)]
    coops = [np.mean(c[[0, 3], [1, 2]]) for c in corrs]
    diffs = [np.mean(c[[0, 3], [1, 2]]) - np.mean(c[0:2, 2:4])
             for c in corrs]
    isc_coops.append(coops)
    isc_diffs.append(diffs)
    isc_pc_diffs = []
    for r, diff in enumerate(diffs):
        isc_diffs_df['difference'].append(diff)
        isc_diffs_df['PC'].append(pc + 1)
        isc_diffs_df['repeat'].append(r)
        isc_pc_diffs.append(diff)
isc_diffs_df = pd.DataFrame(isc_diffs_df)
isc_coops = np.array(isc_coops).T
isc_diffs = np.array(isc_diffs).T

# Get PCs with largest difference between cooperative/competitive
n_top = 10
isc_diff_means = fisher_mean(isc_diffs, axis=0)
top_diffs = np.argpartition(isc_diff_means, -n_top)[-n_top:]
top_diffs = top_diffs[np.argsort(isc_diff_means[top_diffs])[::-1]]

# Get PCs with largest cooperative ISC (irrespective of competitive ISC)
n_top = 10
isc_coop_means = fisher_mean(isc_coops, axis=0)
top_coops = np.argpartition(isc_coop_means, -n_top)[-n_top:]
top_coops = top_coops[np.argsort(isc_coop_means[top_coops])[::-1]]

# Find overlap between top PCs
top_both = list(set(top_diffs) & set(top_coops))
# For matchup 0: [2, 7, 9, 23, 24]

# Bootstrap test for significance of difference
observed, ci, p, distribution = bootstrap_test(isc_diffs,
                                               bootstrap_axis=0,
                                               n_bootstraps=1000,
                                               estimator=fisher_mean,
                                               ci_percentile=95,
                                               side='two-sided')

# FDR correction of p-values
_, fdr_p, _, _ = multipletests(p, method='fdr_bh')

# Plot ISCs for 100 PCs with significance markers
sig_pos = ((fdr_p < .05) & (observed > 0)).nonzero()[0]
sig_neg = ((fdr_p < .05) & (observed < 0)).nonzero()[0]

sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 4))
sns.barplot(x='PC', y='difference', data=isc_diffs_df, ax=ax, color='.6',
            estimator=fisher_mean)
#ax.set_ylim(-.375, .325) # for matchup = 3
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
sns.despine()
plt.savefig('figures/isc_diff-bars_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')


# Zoom in and replot to highlight top PCs
from matplotlib.patches import Patch

colors = np.array(['.7'] * k, dtype='object')
colors[top_coops] = 'tab:red'
colors[top_diffs] = 'tab:blue'
colors[top_both] = 'tab:purple'
np.save('figures/colors_top-bars_tanh-z_'
         f'pca-k{k}_reg-{reg}_m{matchup}.npy', colors)

sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 4))
sns.barplot(x='PC', y='difference', data=isc_diffs_df, ax=ax, color='.6',
            estimator=fisher_mean, palette=colors)
#ax.set_ylim(-.375, .325) # for matchup = 3
ax.set_ylim(-.05, .4) # for matchup = 0
ax.set_xticks([0, 19, 39, 59, 79, 99])
ax.set_ylabel('cooperative – competitive ISC')
ax.set_title(f'difference in cooperative vs. competitive ISC for 100 PCs')
sns.despine()
legend_elements = [Patch(facecolor='tab:red'),
                   Patch(facecolor='tab:blue'),
                   Patch(facecolor='tab:purple'),
                   Patch(facecolor='tab:purple')]
ax.legend(handles=legend_elements, loc='upper right',
          labels=['', '', 'top 10 cooperative PCs',
                  'top 10 difference PCs'],
          ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5)
plt.savefig('figures/isc_top-bars_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')

# Plot cooperative/competitive ISC for top 10 PCs
matchup = 0
n_repeats = 8
pcs = top_both
fig, axs = plt.subplots(1, 5, figsize=(18, 8))
for pc, ax in zip(pcs, axs.ravel()):
    corr = fisher_mean([np.corrcoef(lstms_pca_reg[matchup, r, ..., pc])
                        for r in np.arange(n_repeats)], axis=0)
    sns.heatmap(corr, square=True, annot=True, vmin=-1, vmax=1,
                cmap='RdBu_r', xticklabels=False, yticklabels=False,
                fmt='.2f', ax=ax, cbar_kws={'shrink': .32})
    ax.set_title(f'PC{pc + 1}')
plt.savefig(f'figures/isc_top-coop_tanh-z_pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')


# Lagged ISC for selected PCs
matchup = 0
n_repeats = 8
n_lags = 900
pc_ids = np.arange(10)

# Compute lagged ISC for each repeat
lagged_iscs = []
for repeat in np.arange(n_repeats):
    
    # Slicing with array seems to shift axis?
    lstms_rep = np.moveaxis(lstms_pca[matchup, repeat, ..., pc_ids], 2, 0)
    lagged_rep, lags = lagged_isc(lstms_rep, n_lags=n_lags, circular=True)
    lagged_iscs.append(lagged_rep)
    print(f"Finished computing lagged ISC for repeat {repeat}")

lagged_iscs = np.stack(lagged_iscs, axis=0)

# Get lagged ISCs for cooperative pairs
coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]
lagged_coop = np.mean(lagged_iscs[:, coop_ids, ...], axis=1)
lagged_comp = np.mean(lagged_iscs[:, coop_ids, ...], axis=1)

# Bootstrap test to assess significance
observed, ci, ps, distribution = bootstrap_test(lagged_coop,
                                                bootstrap_axis=0,
                                                n_bootstraps=1000,
                                                estimator=fisher_mean,
                                                ci_percentile=95,
                                                side='right')

# FDR correction across lags
fdr_ps = []
for p in ps:
    _, fdr_p, _, _ = multipletests(p, method='fdr_bh')
    fdr_ps.append(fdr_p)

fdr_ps = np.array(fdr_ps)

# Plot lagged ISC with significance indicator
n_rows, n_cols = 5, 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(9, 8))
pc_ids = np.arange(10)
threshold = .02
sig_ids = (fdr_ps[pc_id] <= threshold).nonzero()[0]

for i, (pc_id, ax) in enumerate(zip(pc_ids, axs.ravel())):
    ax.plot(lags, np.concatenate(lagged_iscs[:, coop_ids, pc_id]).T,
             color='.8', alpha=.5, zorder=1);
    ax.plot(lags, np.mean(lagged_coop[:, pc_id, :], axis=0),
             color='.4', zorder=2);
    if i not in [8, 9]:
        ax.xaxis.set_ticks([])
    else:
        ax.set_xticks(lags[::15 * 10])
        ax.set_xticklabels(np.unique(lags // 15)[::10])
        ax.set_xlabel('lag (seconds)')
    if i % 2 != 0:
        ax.yaxis.set_ticks([])
    ax.set_ylim(-.3, .7)
    ax.set_xlim(-n_lags, n_lags)
    ax.set_title(f'PC{pc_id + 1} cooperative ISC',
                 loc='left', va='top', x=.02, y=.95)
sns.despine()
plt.tight_layout()
plt.savefig('figures/isc_lag-60s_tanh-z_'
            f'pca-k{k}_m{matchup}.png',
            dpi=300, bbox_inches='tight')
#plt.scatter(lags[sig_ids], np.mean(lagged_coop[:, pc_id], axis=0)[sig_ids],
#            color='tab:red', marker='.', zorder=3)


# Load in PCA-reduced LSTMs with confounds regressed out
reg = 'com' # 'pre', 'hud', 'act', or 'com'
lstms_pca_reg = np.load(f'results/lstms_tanh-z_pca-k{k}_reg-{reg}.npy')

# Lagged ISC for selected PCs
matchup = 0
n_repeats = 8
n_lags = 900
pc_ids = np.arange(10)

# Compute lagged ISC for each repeat
lagged_iscs = []
for repeat in np.arange(n_repeats):
    
    # Slicing with array seems to shift axis?
    lstms_rep = np.moveaxis(lstms_pca_reg[matchup, repeat, ..., pc_ids], 2, 0)
    lagged_rep, lags = lagged_isc(lstms_rep, n_lags=n_lags, circular=True)
    lagged_iscs.append(lagged_rep)
    print(f"Finished computing lagged ISC for repeat {repeat}")

lagged_iscs = np.stack(lagged_iscs, axis=0)

# Get lagged ISCs for cooperative pairs
coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]
lagged_coop = np.mean(lagged_iscs[:, coop_ids, ...], axis=1)
lagged_comp = np.mean(lagged_iscs[:, coop_ids, ...], axis=1)

# Bootstrap test to assess significance
observed, ci, ps, distribution = bootstrap_test(lagged_coop,
                                                bootstrap_axis=0,
                                                n_bootstraps=1000,
                                                estimator=fisher_mean,
                                                ci_percentile=95,
                                                side='right')

# FDR correction across lags
fdr_ps = []
for p in ps:
    _, fdr_p, _, _ = multipletests(p, method='fdr_bh')
    fdr_ps.append(fdr_p)

fdr_ps = np.array(fdr_ps)

# Plot lagged ISC with significance indicator
n_rows, n_cols = 5, 2
fig, axs = plt.subplots(n_rows, n_cols, figsize=(9, 8))
pc_ids = np.arange(10)
threshold = .02
sig_ids = (fdr_ps[pc_id] <= threshold).nonzero()[0]

for i, (pc_id, ax) in enumerate(zip(pc_ids, axs.ravel())):
    ax.plot(lags, np.concatenate(lagged_iscs[:, coop_ids, pc_id]).T,
             color='.8', alpha=.5, zorder=1);
    ax.plot(lags, np.mean(lagged_coop[:, pc_id, :], axis=0),
             color='.4', zorder=2);
    if i not in [8, 9]:
        ax.xaxis.set_ticks([])
    else:
        ax.set_xticks(lags[::15 * 10])
        ax.set_xticklabels(np.unique(lags // 15)[::10])
        ax.set_xlabel('lag (seconds)')
    if i % 2 != 0:
        ax.yaxis.set_ticks([])
    ax.set_ylim(-.3, .7)
    ax.set_xlim(-n_lags, n_lags)
    ax.set_title(f'PC{pc_id + 1} cooperative ISC',
                 loc='left', va='top', x=.02, y=.95)
sns.despine()
plt.tight_layout()
plt.savefig('figures/isc_lag-60s_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')
#plt.scatter(lags[sig_ids], np.mean(lagged_coop[:, pc_id], axis=0)[sig_ids],
#            color='tab:red', marker='.', zorder=3)
