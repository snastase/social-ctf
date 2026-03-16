import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import spearmanr, ttest_1samp
from scipy.spatial.distance import squareform
from brainiak.isc import isc, isfc
from statsmodels.stats.multitest import multipletests
from statistical_tests import bootstrap_test, fisher_mean
from coupling_metrics import lagged_isc


# Load in PCA-reduced LSTMS
k = 142

# Compute simple ISC and save
matchup_id = 0
n_maps = 32
n_repeats = 32
n_players = 4
n_pairs = n_players * (n_players - 1) // 2

iscs = np.full((n_maps, n_repeats, n_pairs, k), np.nan)
for map_id in np.arange(n_maps):
    for repeat_id in np.arange(n_repeats):
        #lstms_pca = np.load(f'results/lstms-pca_matchup-{matchup_id}_'
        #                    f'map-{map_id}_repeat-{repeat_id}.npy')
        lstms_pca = np.load(f'results/lstms-pca_reg-all_matchup-{matchup_id}_'
                            f'map-{map_id}_repeat-{repeat_id}.npy')
        lstms_pca = lstms_pca[..., :k]
        lstms_pca = np.moveaxis(lstms_pca, 0, 2)
        iscs[map_id, repeat_id] = isc(lstms_pca, pairwise=True)
        print("Finished computing ISC for "
              f"map {map_id} repeat {repeat_id}")
np.save(f'results/iscs_reg-all_matchup-{matchup_id}_data-v1.npy', iscs)


isfcs = np.full((n_maps, n_repeats, n_pairs, k, k), np.nan)
for map_id in np.arange(n_maps):
    for repeat_id in np.arange(n_repeats):
        #lstms_pca = np.load(f'results/lstms-pca_matchup-{matchup_id}_'
        #                    f'map-{map_id}_repeat-{repeat_id}.npy')
        lstms_pca = np.load(f'results/lstms-pca_reg-pre_matchup-{matchup_id}_'
                            f'map-{map_id}_repeat-{repeat_id}.npy')
        lstms_pca = lstms_pca[..., :k]
        lstms_pca = np.moveaxis(lstms_pca, 0, 2)
        isfcs[map_id, repeat_id] = isfc(lstms_pca, pairwise=True,
                                        vectorize_isfcs=False)
        print("Finished computing ISFC for "
              f"map {map_id} repeat {repeat_id}")
np.save(f'results/isfcs_reg-pre_matchup-{matchup_id}_data-v1.npy', isfcs)


# Plot cooperative/competitive ISC for top 10 PCs
matchup_id = 0
iscs = np.load(f'results/iscs_matchup-{matchup_id}_data-v1.npy')

pcs = np.arange(10)
sns.set(style='ticks', font_scale=1.1)
fig, axs = plt.subplots(2, 5, figsize=(10, 4.5))
cbar_ax = fig.add_axes([1, .3, .013, .4])
for pc, ax in zip(pcs, axs.ravel()):
    iscs_tri = fisher_mean(iscs[..., pc], axis=(0, 1))
    corrs = squareform(iscs_tri, checks=False)
    np.fill_diagonal(corrs, 1)
    sns.heatmap(corrs, square=True, annot=True, vmin=-1, vmax=1,
                cmap='RdBu_r', xticklabels=False, yticklabels=False,
                fmt='.2f', ax=ax, annot_kws={'size': 10},
                cbar=pc == 0, cbar_ax=None if pc else cbar_ax,
                cbar_kws={'ticks': [-1, 1]})
    ax.set_title(f'PC{pc + 1}')
cbar_ax.tick_params(size=0)
fig.text(1.016, 0.5, f'correlation', ha='left',
         va='center', rotation=270, size=13)
plt.tight_layout()
plt.savefig(f'figures/isc_coop-comp_matchup-{matchup_id}_data-v1.svg',
            dpi=300, bbox_inches='tight', transparent=True)

# Plot cooperative/competitive ISC for select set of PCs
matchup_id = 0
iscs = np.load(f'results/iscs_matchup-{matchup_id}_data-v1.npy')

pcs = [0, 4, 6, 8, 10, 14]
sns.set(style='ticks', font_scale=1.5)
fig, axs = plt.subplots(2, 3, figsize=(13.5, 8), sharex=True, sharey=True)
for i, (pc, ax) in enumerate(zip(pcs, axs.ravel())):
    iscs_tri = fisher_mean(iscs[..., pc], axis=(0, 1))
    corrs = squareform(iscs_tri, checks=False)
    np.fill_diagonal(corrs, 1)
    sns.heatmap(corrs, square=True, annot=True, vmin=-1, vmax=1,
                cmap='RdBu_r', xticklabels=False, yticklabels=False,
                fmt='.2f', ax=ax, cbar=False if i not in [2, 5] else True)
    ax.set_title(f'PC{pc + 1}')
plt.tight_layout()
plt.savefig(f'figures/isc_coop-comp__select-pcs_matchup-{matchup_id}_data-v1.png',
            dpi=300, bbox_inches='tight')


# Difference in cooperative/competitive ISC across PCs
matchup_id = 0
iscs = np.load(f'results/iscs_reg-pre_matchup-{matchup_id}_data-v1.npy')
#iscs = np.load(f'results/iscs_matchup-{matchup_id}_data-v1.npy')

n_maps = 32
n_repeats = 32
n_pcs = 142
coop_pairs, comp_pairs = (0, 5), (1, 2, 3, 4)

# Compute differences between cooperative and competitive ISCs
isc_diffs = []
isc_diffs_df = {'difference': [], 'PC': [], 'repeat': [], 'map': []}
for pc in np.arange(n_pcs):
    diffs = (np.mean(iscs[:, :, coop_pairs, pc], axis=2) -
             np.mean(iscs[:, :, comp_pairs, pc], axis=2))
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):
            isc_diffs_df['difference'].append(diffs[map_id, repeat_id])
            isc_diffs_df['PC'].append(pc + 1)
            isc_diffs_df['repeat'].append(repeat_id)
            isc_diffs_df['map'].append(map_id)
    isc_diffs.append(diffs.flatten())
isc_diffs_df = pd.DataFrame(isc_diffs_df)
isc_diffs = np.array(isc_diffs).T

#np.save(f'results/isc-diffs_matchup-{matchup_id}_data-v1.npy', isc_diffs)

# Bootstrap test for significance of difference
observed, ci, p, distribution = bootstrap_test(isc_diffs,
                                               bootstrap_axis=0,
                                               n_bootstraps=10000,
                                               estimator=fisher_mean,
                                               ci_percentile=95,
                                               side='two-sided')

# FDR correction of p-values
_, fdr_p, _, _ = multipletests(p, method='fdr_bh')

# Plot ISCs for 100 PCs with significance markers
sig_pos = ((fdr_p < .05) & (observed > 0)).nonzero()[0]
sig_neg = ((fdr_p < .05) & (observed < 0)).nonzero()[0]

#sns.set_context('notebook', font_scale=1.2)
sns.set(style='ticks', font_scale=1.1)
bar_width = 1
fig, ax = plt.subplots(figsize=(12, 3))
sns.barplot(x='PC', y='difference', data=isc_diffs_df, ax=ax, color='.9',
            estimator=fisher_mean, ci=None, zorder=0)
for patch in ax.patches:
    curr_width = patch.get_width()
    diff_width = curr_width - bar_width
    patch.set_width(bar_width)
    patch.set_x(patch.get_x() + diff_width * .5)
sns.pointplot(x='PC', y='difference', errwidth=1,
              data=isc_diffs_df, join=False, ax=ax,
              scale=.6, color='.6')
ax.set_xticks(np.insert(np.arange(20, n_pcs, 20) - 1, 0, 0))
ax.set_ylim(-.06, .217)
for sig_pc in sig_pos:
    #ax.annotate('.', (sig_pc, -.04), color='tab:red', size=40,
    #            xycoords=('data', 'axes fraction'),
    #            ha='center', va='bottom')
    ax.annotate('.', (sig_pc, .93), color='darkgoldenrod', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
for sig_pc in sig_neg:
    ax.annotate('.', (sig_pc, -.047), color='darkgoldenrod', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
ax.set_ylabel('difference in correlation')
ax.set_xlabel('LSTM PC')
#ax.text(141, -.075, 'difference in cooperative vs. competitive ISC',
#        ha='right', va='top');
ax.text(141, .19, 'difference: cooperative – competitive ISC',
        ha='right', va='top');
#ax.set_yticks([-.1, 0, 0.1, 0.2])
#ax.set_yticklabels(["\N{MINUS SIGN}0.1", 0, 0.1, 0.2])
sns.despine()
plt.savefig(f'figures/isc-diffs_reg-pre_matchup-{matchup_id}_data-v1_bar.svg',
            dpi=300, bbox_inches='tight', transparent=True)
#plt.savefig(f'figures/isc-diffs_matchup-{matchup_id}_data-v1_bar.png', dpi=300, bbox_inches='tight', transparent=True)


#sns.set_context('notebook', font_scale=1.2)
sns.set(style='ticks', font_scale=1.1)
bar_width = 1
fig, ax = plt.subplots(figsize=(12, 3))
sns.barplot(x='PC', y='difference', data=isc_diffs_df, ax=ax, color='.9',
            estimator=fisher_mean, ci=None, zorder=0)
for patch in ax.patches:
    curr_width = patch.get_width()
    diff_width = curr_width - bar_width
    patch.set_width(bar_width)
    patch.set_x(patch.get_x() + diff_width * .5)
sns.pointplot(x='PC', y='difference', errwidth=1,
              data=isc_diffs_df, join=False, ax=ax,
              scale=.6, color='.6')
ax.set_xticks(np.insert(np.arange(20, n_pcs, 20) - 1, 0, 0))
ax.set_ylim(-.68, .43)
for sig_pc in sig_pos:
    #ax.annotate('.', (sig_pc, -.04), color='tab:red', size=40,
    #            xycoords=('data', 'axes fraction'),
    #            ha='center', va='bottom')
    ax.annotate('.', (sig_pc, .93), color='darkgoldenrod', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
for sig_pc in sig_neg:
    ax.annotate('.', (sig_pc, -.05), color='darkgoldenrod', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
ax.set_ylabel('difference in correlation', y=.52)
ax.set_xlabel('LSTM PC')
ax.text(141, -.1, 'difference: cooperative – competitive ISC',
        ha='right', va='top');
#ax.set_yticks([-.1, 0, 0.1, 0.2])
#ax.set_yticklabels(["\N{MINUS SIGN}0.1", 0, 0.1, 0.2])
sns.despine()
plt.savefig(f'figures/isc-diffs_matchup-{matchup_id}_data-v1_bar.svg',
            dpi=300, bbox_inches='tight', transparent=True)



# Split ISCs according to win/loss
matchup_id = 0
n_pcs = 142
coop_ids = (0, 5)

iscs = np.load(f'results/iscs_matchup-{matchup_id}_data-v1.npy')
scores = np.load(f'results/scores_matchup-{matchup_id}.npy')

coop_split = {'map': [], 'repeat': [], 'outcome': [], 'ISC': [], 'PC': []}
for map_id in np.arange(n_maps):
    for repeat_id in np.arange(n_repeats):
        for pc_id in np.arange(n_pcs):
            coop_iscs = iscs[map_id, repeat_id, coop_ids, pc_id]
            coop_scores = scores[map_id, repeat_id]
            if coop_scores[0] > coop_scores[1]:
                coop_split['outcome'].extend(['win', 'loss'])
            elif coop_scores[0] < coop_scores[1]:
                coop_split['outcome'].extend(['loss', 'win'])
            else:
                continue
            coop_split['ISC'].extend(coop_iscs)
            coop_split['map'].extend([map_id, map_id])
            coop_split['repeat'].extend([repeat_id, repeat_id])
            coop_split['PC'].extend([pc_id, pc_id])

coop_split = pd.DataFrame(coop_split)

pc_id = 5
sns.stripplot(x='outcome', y='ISC', color='.7', alpha=.3,
              zorder=0,
              data=coop_split[coop_split['PC'] == pc_id])
sns.pointplot(x='outcome', y='ISC', color='darkgoldenrod',
              markers='', lw=1,
              data=coop_split[coop_split['PC'] == pc_id])


# T-test for difference in ISC based on win/loss
matchup_id = 0
n_pcs = 142
coop_ids = (0, 5)

iscs = np.load(f'results/iscs_matchup-{matchup_id}_data-v1.npy')
scores = np.load(f'results/scores_matchup-{matchup_id}.npy')

coop_ttest = {'map': [], 'repeat': [], 'difference in ISC': [], 'PC': []}
for map_id in np.arange(n_maps):
    for repeat_id in np.arange(n_repeats):
        for pc_id in np.arange(n_pcs):
            coop_iscs = iscs[map_id, repeat_id, coop_ids, pc_id]
            coop_scores = scores[map_id, repeat_id]
            if coop_scores[0] > coop_scores[1]:
                coop_diff = coop_iscs[0] - coop_iscs[1]
            elif coop_scores[1] > coop_scores[0]:
               coop_diff = coop_iscs[1] - coop_iscs[0]
            else:
                continue
            coop_ttest['difference in ISC'].append(coop_diff)
            coop_ttest['map'].append(map_id)
            coop_ttest['repeat'].append(repeat_id)
            coop_ttest['PC'].append(pc_id + 1)

coop_ttest = pd.DataFrame(coop_ttest)

win_ts, win_ps = [], []
for pc_id in np.arange(n_pcs):
    t, p = ttest_1samp(coop_ttest[coop_ttest['PC'] == pc_id + 1][
        'difference in ISC'], popmean=0)
    win_ts.append(t)
    win_ps.append(p)
    
# FDR correction of p-values
_, win_qs, _, _ = multipletests(win_ps, method='fdr_bh')

win_ts, win_qs = np.array(win_ts), np.array(win_qs)

print(f"{np.sum(win_qs < .05)} "
      "significant PCs (FDR corrected)")
for pc_id in np.arange(n_pcs):
   print(f'PC{pc_id + 1}: t = {win_ts[pc_id]:.5f} (q= {win_qs[pc_id]:.5f}) '
         f'{"*" if win_qs[pc_id] < .05 else ""}')

# Plot mean difference between win/loss and t-test for significance
sig_pos = ((win_qs < .05) & (win_ts > 0)).nonzero()[0]
sig_neg = ((win_qs < .05) & (win_ts < 0)).nonzero()[0]

sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 5))
sns.barplot(x='PC', y='difference in ISC',
            data=coop_ttest, color='.6',
            errwidth=1, ax=ax)
ax.set(xticks=np.insert(np.arange(20, n_pcs, 20) - 1, 0, 0),
       xlim=(-1, n_pcs + 1),
       ylabel='difference in ISC (win \N{MINUS SIGN} loss)',
       title='t-test for difference between winning vs. losing ISC')
for sig_pc in sig_pos:
    ax.annotate('.', (sig_pc, -.02), color='tab:red', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
for sig_pc in sig_neg:
    ax.annotate('.', (sig_pc, -.02), color='tab:blue', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
sns.despine()
plt.savefig(f'figures/isc-ttest_matchup-{matchup_id}_data-v1_bar.png',
            dpi=300, bbox_inches='tight')


# ISC correlation with own score
matchup_id = 0
n_pcs = 142
coop_ids = (0, 5)

iscs = np.load(f'results/iscs_matchup-{matchup_id}_data-v1.npy')
scores = np.load(f'results/scores_matchup-{matchup_id}.npy').astype('float64')

coop_iscs = iscs[:, :, coop_ids, :]
coop_iscs = coop_iscs.reshape(n_maps * n_repeats * len(coop_ids), n_pcs)
coop_scores = scores.reshape(n_maps * n_repeats * len(coop_ids))

score_rs, score_ps = [], []
for pc_id in np.arange(n_pcs):
    r, p = spearmanr(coop_iscs[:, pc_id], coop_scores)
    score_rs.append(r)
    score_ps.append(p)

# FDR correction of p-values
_, score_qs, _, _ = multipletests(score_ps, method='fdr_bh')

score_rs, score_qs = np.array(score_rs), np.array(score_qs)
    
print(f"{np.sum(score_qs < .05)} "
      "significant PCs (FDR corrected)")
for pc_id in np.arange(n_pcs):
   print(f'PC{pc_id + 1}: t = {score_rs[pc_id]:.5f} '
         f'(q = {score_qs[pc_id]:.5f}) '
         f'{"*" if score_qs[pc_id] < .05 else ""}')

# Plot correlations for 142 PCs with significance markers
sig_pos = ((score_qs < .05) & (score_rs > 0)).nonzero()[0]
sig_neg = ((score_qs < .05) & (score_rs < 0)).nonzero()[0]

sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(np.arange(n_pcs) + 1, score_rs, color='.6')
ax.set(xticks=np.insert(np.arange(20, n_pcs, 20), 0, 1),
       xlim=(0, n_pcs + 1), ylabel='correlation', xlabel='PC',
       title='correlation between ISC and own team score')
for sig_pc in sig_pos:
    ax.annotate('.', (sig_pc + 1, -.02), color='tab:red', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
for sig_pc in sig_neg:
    ax.annotate('.', (sig_pc + 1, -.02), color='tab:blue', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
sns.despine()
plt.savefig(f'figures/isc-score_matchup-{matchup_id}_data-v1_bar.png',
            dpi=300, bbox_inches='tight')


# Flip score assignment for "defensive coupling"
scores_flip = scores[..., [1, 0]]
coop_scores_flip = scores_flip.reshape(n_maps * n_repeats * len(coop_ids))

flip_rs, flip_ps = [], []
for pc_id in np.arange(n_pcs):
    r, p = spearmanr(coop_iscs[:, pc_id], coop_scores_flip)
    flip_rs.append(r)
    flip_ps.append(p)

# FDR correction of p-values
_, flip_qs, _, _ = multipletests(flip_ps, method='fdr_bh')

flip_rs, flip_qs = np.array(flip_rs), np.array(flip_qs)
    
print(f"{np.sum(flip_qs < .05)} "
      "significant PCs (FDR corrected)")
for pc_id in np.arange(n_pcs):
   print(f'PC{pc_id + 1}: t = {flip_rs[pc_id]:.5f} '
         f'(q = {flip_qs[pc_id]:.5f}) '
         f'{"*" if flip_qs[pc_id] < .05 else ""}')

# Plot correlations for 142 PCs with significance markers
sig_pos = ((flip_qs < .05) & (flip_rs > 0)).nonzero()[0]
sig_neg = ((flip_qs < .05) & (flip_rs < 0)).nonzero()[0]

sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(np.arange(n_pcs) + 1, flip_rs, color='.6')
ax.set(xticks=np.insert(np.arange(20, n_pcs, 20), 0, 1),
       xlim=(0, n_pcs + 1), ylabel='correlation', xlabel='PC',
       title='correlation between ISC and opponent team score')
for sig_pc in sig_pos:
    ax.annotate('.', (sig_pc + 1, -.02), color='tab:red', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
for sig_pc in sig_neg:
    ax.annotate('.', (sig_pc + 1, -.02), color='tab:blue', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
sns.despine()
plt.savefig(f'figures/isc-flip_matchup-{matchup_id}_data-v1_bar.png',
            dpi=300, bbox_inches='tight')


# Difference ISC correlation with difference in score
matchup_id = 0
n_pcs = 142
coop_ids = (0, 5)

iscs = np.load(f'results/iscs_matchup-{matchup_id}_data-v1.npy')
scores = np.load(f'results/scores_matchup-{matchup_id}.npy').astype('float64')

coop_diffs = np.full((n_maps, n_repeats, n_pcs), np.nan)
score_diffs = np.full((n_maps, n_repeats), np.nan)
for map_id in np.arange(n_maps):
    for repeat_id in np.arange(n_repeats):
        coop_scores = scores[map_id, repeat_id]
        score_diffs[map_id, repeat_id] = (coop_scores[0] -
                                          coop_scores[1])
        
        for pc_id in np.arange(n_pcs):
            coop_iscs = iscs[map_id, repeat_id, coop_ids, pc_id]
            
            coop_diffs[map_id, repeat_id, pc_id] = (coop_iscs[0] -
                                                    coop_iscs[1])

coop_diffs = np.vstack(np.split(coop_diffs, n_repeats, axis=1)).squeeze()
score_diffs = np.vstack(np.split(score_diffs, n_repeats, axis=1)).squeeze()

diff_rs, diff_ps = [], []
for pc_id in np.arange(n_pcs):
    r, p = spearmanr(coop_diffs[:, pc_id], score_diffs)
    diff_rs.append(r)
    diff_ps.append(p)

# FDR correction of p-values
_, diff_qs, _, _ = multipletests(diff_ps, method='fdr_bh')

diff_rs, diff_qs = np.array(diff_rs), np.array(diff_qs)
    
print(f"{np.sum(diff_qs < .05)} "
      "significant PCs (FDR corrected)")
for pc_id in np.arange(n_pcs):
   print(f'PC{pc_id + 1}: t = {diff_rs[pc_id]:.5f} '
         f'(q = {diff_qs[pc_id]:.5f}) '
         f'{"*" if diff_qs[pc_id] < .05 else ""}')

# Plot correlations for 142 PCs with significance markers
sig_pos = ((diff_qs < .05) & (diff_rs > 0)).nonzero()[0]
sig_neg = ((diff_qs < .05) & (diff_rs < 0)).nonzero()[0]

sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 5))
ax.bar(np.arange(n_pcs) + 1, diff_rs, color='.6')
ax.set(xticks=np.insert(np.arange(20, n_pcs, 20), 0, 1),
       xlim=(0, n_pcs + 1), ylabel='correlation', xlabel='PC',
       title=('correlation between difference in '
              'cooperative ISC and difference in score'))
for sig_pc in sig_pos:
    ax.annotate('.', (sig_pc + 1, -.02), color='tab:red', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
for sig_pc in sig_neg:
    ax.annotate('.', (sig_pc + 1, -.02), color='tab:blue', size=40,
                xycoords=('data', 'axes fraction'),
                ha='center', va='bottom')
sns.despine()
plt.savefig(f'figures/isc-diff_matchup-{matchup_id}_data-v1_bar.png',
            dpi=300, bbox_inches='tight')

    
# Plot scatter plot for correlation between difference in score and ISC
pc_id = 0
pc_df = pd.DataFrame({'difference in score': score_diffs,
                      'difference in cooperative ISC': coop_diffs[:, pc_id]})
fig, ax = plt.subplots(figsize=(6, 5))
sns.regplot(x='difference in score',
            y='difference in cooperative ISC',
            data=pc_df, ax=ax, line_kws={'color': 'darkgoldenrod'},
            scatter_kws={'color': '.6', 'alpha': .6})
ax.set(xlim=(-12, 13), title=(f'PC{pc_id + 1}'))
ax.annotate('r = ' + f'{diff_rs[pc_id]:.3f}'.lstrip('0') + 
            ('\nq = ' + f'{diff_qs[pc_id]:.3f}'.lstrip('0') if
             diff_qs[pc_id] >= .001 else '\nq < .001'),
            xy=(0.02, 1), xycoords='axes fraction', va='top', size=12)
sns.despine()
plt.savefig(f'figures/isc-score_pc-{pc_id + 1}_'
            f'matchup-{matchup_id}_data-v1_scatter.png',
            dpi=300, bbox_inches='tight')


###################################
    
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

# Compare PC1 to health
from os.path import join
from scipy.stats import pearsonr
from ctf_dataset.load import create_wrapped_dataset

base_dir = '/jukebox/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data_v1')
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual_v2.hdf5")

pc_id = 0
stat_label = 'health'
#stat_label = 'is_alive'
corrs = []
for map_id in np.arange(n_maps):
    #for repeat_id in np.arange(n_repeats):
    for repeat_id in [0]:
        lstm_pc = np.load(
            f'results/lstms-pca_matchup-{matchup_id}_'
            f'map-{map_id}_repeat-{repeat_id}.npy')[..., pc_id]
        for player_id in np.arange(n_players):
            stat = wrap_f[f'map/matchup/repeat/player/time/{stat_label}'][
                map_id, matchup_id, repeat_id, player_id][..., 0]
            corrs.append(pearsonr(stat, lstm_pc[player_id])[0])
    print(f"Finished running correlations for map {map_id}")
            
