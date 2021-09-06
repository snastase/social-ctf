from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr, zscore
from statistical_tests import (fisher_mean, block_iscf, 
                               block_randomization)
from ctf_dataset.load import create_wrapped_dataset

# Import behavior heuristics
from ctf_dataset.behaviours.heuristic import (escort,
                                              teammate_has_flag,
                                              near_teammate,
                                              camp_own_base,
                                              camp_opponent_base,
                                              approaching_teammate)
                                          
# Set base directory 
base_dir = '/jukebox/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data')

# Create wrapped CTF dataset
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

map_id = 0 # 0
matchup_id = 0 # 0-54 (0, 34, 49, 54)
repeat_id = slice(None) # 0-7
player_id = slice(None) # 0-3


# Compute whether escoring teammate with flag
teammate_flags = teammate_has_flag(wrap_f, map_id=map_id,
                                   matchup_id=matchup_id,
                                   repeat_id=slice(None),
                                   player_id=slice(None)).squeeze()


# Compute whether escoring teammate with flag
escorts = escort(wrap_f, map_id=map_id, matchup_id=matchup_id,
                        repeat_id=slice(None), player_id=slice(None),
                        min_behaviour_length=15, teammate_radius=3).squeeze()


# Compute near teammate behavior time series (symmetric)
nears = near_teammate(wrap_f, map_id=map_id, matchup_id=matchup_id,
                      repeat_id=slice(None), player_id=slice(None),
                      min_behaviour_length=15, teammate_radius=3).squeeze()

# Compute base-camping behavior time series (symmetric)
approachs = approaching_teammate(wrap_f, map_id=map_id,
                                 matchup_id=matchup_id,
                                 repeat_id=slice(None),
                                 player_id=slice(None),
                                 min_behaviour_length=15,
                                 min_approach=.5).squeeze()

# Compute base-camping behavior time series (asymmetric)
basecamps = camp_own_base(wrap_f, map_id=map_id, matchup_id=matchup_id,
                          repeat_id=slice(None), player_id=slice(None),
                          min_behaviour_length=15,
                          base_radius=3).squeeze()

# Compute base-camping behavior time series (asymmetric)
spawncamps = camp_opponent_base(wrap_f, map_id=map_id,
                                matchup_id=matchup_id,
                                repeat_id=slice(None),
                                player_id=slice(None),
                                min_behaviour_length=15,
                                base_radius=3).squeeze()


# Create combined vector for all "cooperative" behaviors
symmetric = ['near teammate', 'approaching teammate']
behaviors = {'teammate has flag': teammate_flags,
             'near teammate': nears,
             'escort': escorts,
             #'approaching teammate': approachs,
             'basecamping': basecamps,
             'spawncamping': spawncamps}

n_repeats = 8
n_players = 4
n_samples = 4501
#combined = np.full((n_repeats, n_players, n_samples), False).astype(bool)
#for repeat in np.arange(n_repeats):
#    for player in np.arange(n_players):
#        combined[repeat, player] = np.sum([behaviors[b][repeat, player]
#                                           for b in behaviors],
#                                          axis=0).astype(bool)

#behaviors = {'near teammate': nears,
#             'approaching teammate': approachs,
#             'basecamping': basecamps,
#             'spawncamping': spawncamps,
#             'combined': combined}


# Load an example actions output matrix
map_id, matchup, repeat, player = 0, 0, 0, 0
actions = wrap_f['map/matchup/repeat/player/time/action'][
    map_id, matchup, repeat, player, ...]
actions = actions / np.max(np.abs(actions), axis=0)


# Load some environment variables
feature_labels = ['health', 'position', 'velocity',
                  'player_from_own_base_xy_distance',
                  'player_from_teammate_xy_distance']
features = {}
for label in feature_labels:
    features[label] = wrap_f[f'map/matchup/repeat/player/time/{label}'][
        map_id, matchup, repeat, player, ...]
features['distance from base'] = features.pop(
    'player_from_own_base_xy_distance')
features['distance from teammate'] = features.pop(
    'player_from_teammate_xy_distance')

#dictionary[new_key] = dictionary.pop(old_key)

#"map/matchup/repeat/player/time/player_from_own_base_xy_distance"

# Plot behavior time series
k = 100
matchup = 0
repeat = 0
player = 0

sns.set_context('notebook', font_scale=1)
fig, axs = plt.subplots(len(behaviors) + len(features) + 1, 1,
                        figsize=(10, 6), sharex=True)
axs[0].matshow(np.repeat(actions, 100, axis=1).T,
               cmap='gray', vmin=0, vmax=1)
axs[0].set_xlim(0, 4500)
axs[0].set_aspect('auto')
axs[0].yaxis.set_ticks([])
axs[0].set_ylabel('actions', rotation=0, ha='right', va='center')
axs[0].tick_params(axis='x', length = 0)

for i, (ax, feature) in enumerate(zip(axs[1:len(features) + 1],
                                       features)):
    ax.plot(features[feature][:, :2], c='.5')
    ax.yaxis.set_ticks([])
    ax.set_xlim(0, 4500)
    ax.set_xlabel('time')
    ax.set_ylabel(feature, rotation=0, ha='right', va='center')
    ax.tick_params(axis='x', length = 0)

for i, (ax, behavior) in enumerate(zip(axs[len(features) + 1:],
                                       behaviors)):
    ax.plot(behaviors[behavior][repeat, player], c='.5')
    ax.yaxis.set_ticks([])
    ax.set_xlim(0, 4500)
    ax.set_xlabel('time')
    ax.set_ylabel(behavior, rotation=0, ha='right', va='center')
    ax.tick_params(axis='x', length = 0)
    #if i != len(behaviors):
    #    ax.xaxis.set_ticks([])
sns.despine()
plt.savefig(f'figures/behavior_ts-10_'
            f'm{matchup}_r{repeat}_p{player}_new.png',
            dpi=300, bbox_inches='tight')


# Plot behavior alongside PC time series
k = 100
map_id = 0
matchup = 0
repeat = 0
lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')[matchup]
pc_id = 2
behavior = 'near teammate'
teams = [[0, 1], [2, 3]]

if wrap_f['map/matchup/repeat/player/color_id'][
    map_id, matchup, repeat, 0][0] == 0:
    team_colors = ['darkred', 'darkblue']
    player_colors = ['darkred', 'coral', 'darkblue', 'lightseagreen']
elif wrap_f['map/matchup/repeat/player/color_id'][
    map_id, matchup_id, repeat_id, 0][0] == 1:
    team_colors = ['darkblue', 'darkred']
    player_colors = ['darkblue', 'lightseagreen', 'darkred', 'coral']

sns.set_context('notebook', font_scale=1.2)
fig, axs = plt.subplots(4, 1, figsize=(10, 4), sharex=True,
                        gridspec_kw={'height_ratios':
                                     [2.5, 2.5, 1, 1]})
for team, ax in zip(teams, axs[:2]):
    ax.plot(lstms_pca[repeat, team[0], :, pc_id],
            c=player_colors[team[0]], alpha=.7, lw=.8)
    ax.plot(lstms_pca[repeat, team[1], :, pc_id],
            c=player_colors[team[1]], alpha=.7, lw=.8)
    ax.set_yticks([])
    ax.tick_params(axis='x', length = 0)
    ax.set_ylabel(f'PC{pc_id + 1}', rotation=0,
                  ha='right', va='center')
    ax.set_xlim([0, 4500])
    #ax.annotate(f'ISC: {lstm_pc_isc[0]:.3f}', (.995, .83),
    #                  ha='right', xycoords='axes fraction')
for team, ax in zip(teams, axs[2:]):
    ax.plot(behaviors[behavior][repeat, team[0]],
            c=player_colors[team[0]])
    ax.yaxis.set_ticks([])
    ax.set_xlim(0, 4500)
    ax.set_ylabel(behavior.replace(' ', '\n'),
                  rotation=0, ha='right', va='center')
    ax.tick_params(axis='x', length = 0)
ax.set_xlabel('time')
sns.despine()
plt.savefig(f'figures/behavior_ts-pc{pc_id + 1}_'
            f'm{matchup}_r{repeat}.png',
            dpi=300, bbox_inches='tight')


# Plot behavior alongside PC and cofluctuation time series
k = 100
map_id = 0
matchup = 0
repeat = 0
reg = 'com'
pc_id = 2

lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}_reg-{reg}.npy')[matchup]
lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')[matchup]

iscfs = np.load(f'results/iscf_lstm_tanh-z_pca-k{k}_reg-{reg}_'
                f'm{matchup}_r{repeat}.npy')[..., pc_id, pc_id]
iscfs = np.load(f'results/iscf_lstm_tanh-z_pca-k{k}_'
                f'm{matchup}_r{repeat}.npy')[..., pc_id, pc_id]
behavior = 'near teammate'
teams = [[0, 1], [2, 3]]
coop_ids = [0, 5]

if wrap_f['map/matchup/repeat/player/color_id'][
    map_id, matchup, repeat, 0][0] == 0:
    team_colors = ['darkred', 'darkblue']
    player_colors = ['darkred', 'coral', 'darkblue', 'lightseagreen']
elif wrap_f['map/matchup/repeat/player/color_id'][
    map_id, matchup_id, repeat_id, 0][0] == 1:
    team_colors = ['darkblue', 'darkred']
    player_colors = ['darkblue', 'lightseagreen', 'darkred', 'coral']

sns.set_context('notebook', font_scale=1.2)
fig, axs = plt.subplots(6, 1, figsize=(10, 6), sharex=True,
                        gridspec_kw={'height_ratios':
                                     [2.5, 2.5, 2.5, 2.5, 1, 1]})
for team, ax in zip(teams, axs[:2]):
    ax.plot(lstms_pca[repeat, team[0], :, pc_id],
            c=player_colors[team[0]], alpha=.7, lw=.8)
    ax.plot(lstms_pca[repeat, team[1], :, pc_id],
            c=player_colors[team[1]], alpha=.7, lw=.8)
    ax.set_yticks([])
    ax.tick_params(axis='x', length = 0)
    ax.set_ylabel(f'PC{pc_id + 1}', rotation=0,
                  ha='right', va='center')
    ax.set_xlim([0, 4500])
    #ax.annotate(f'ISC: {lstm_pc_isc[0]:.3f}', (.995, .83),
    #            ha='right', xycoords='axes fraction')
for coop_id, team, ax in zip(coop_ids, teams, axs[2:]):
    ax.plot(iscfs[coop_id], lw=.8,
            c=player_colors[team[0]])
    ax.yaxis.set_ticks([])
    ax.set_xlim(0, 4500)
    ax.set_ylabel(f'PC{pc_id + 1} CF',
                 rotation=0, ha='right', va='center')
    ax.tick_params(axis='x', length = 0)
for team, ax in zip(teams, axs[4:]):
    ax.plot(behaviors[behavior][repeat, team[0]],
            c=player_colors[team[0]])
    ax.yaxis.set_ticks([])
    ax.set_xlim(0, 4500)
    ax.set_ylabel(behavior.replace(' ', '\n'),
                  rotation=0, ha='right', va='center')
    ax.tick_params(axis='x', length = 0)
ax.set_xlabel('time')
sns.despine()
plt.savefig(f'figures/behavior_ts-cf{pc_id + 1}-trans_'
            f'm{matchup}_r{repeat}.png', transparent=True,
            dpi=300, bbox_inches='tight')


# Plot behavior alongside PC and cofluctuation time series
k = 100
map_id = 0
matchup = 0
repeat = 0
reg = 'com'
pc_id = 2

lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}_reg-{reg}.npy')[matchup]
lstms_pca = np.load(f'results/lstms_tanh-z_pca-k{k}.npy')[matchup]

iscfs = np.load(f'results/iscf_lstm_tanh-z_pca-k{k}_reg-{reg}_'
                f'm{matchup}_r{repeat}.npy')[..., pc_id, pc_id]
iscfs = np.load(f'results/iscf_lstm_tanh-z_pca-k{k}_'
                f'm{matchup}_r{repeat}.npy')[..., pc_id, pc_id]
behavior = 'near teammate'
teams = [[0, 1], [2, 3]]
coop_ids = [0, 5]

if wrap_f['map/matchup/repeat/player/color_id'][
    map_id, matchup, repeat, 0][0] == 0:
    team_colors = ['darkred', 'darkblue']
    player_colors = ['darkred', 'coral', 'darkblue', 'lightseagreen']
elif wrap_f['map/matchup/repeat/player/color_id'][
    map_id, matchup_id, repeat_id, 0][0] == 1:
    team_colors = ['darkblue', 'darkred']
    player_colors = ['darkblue', 'lightseagreen', 'darkred', 'coral']


# Prevalence (i.e. proportion) of behavior over time
n_samples = 4501
behavior_prevs = {}
for behavior in behaviors:
    behavior_prevs[behavior] = (np.sum(behaviors[behavior], axis=2)
                                / n_samples)

# Load in ISC values
k = 100
reg = 'com'
iscs = np.load(f'results/iscs_tanh-z_pca-k{k}_reg-{reg}.npy')[matchup]
iscs = np.load(f'results/iscs_tanh-z_pca-k{k}.npy')[matchup]

coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]
teams = [[0, 1], [2, 3]]
prev_rs = {'correlation': [], 'PC': [], 'behavior': [], 'p': []}
for pc_id in np.arange(k):
    for behavior in behaviors:
        prevs = np.concatenate([np.mean(behavior_prevs[behavior][:, teams[0]],
                                axis=1),
                                np.mean(behavior_prevs[behavior][:, teams[1]],
                                axis=1)])
        r, p = pearsonr(prevs, np.concatenate([iscs[:, coop_ids[0], pc_id],
                                               iscs[:, coop_ids[0], pc_id]]))
        prev_rs['correlation'].append(r)
        prev_rs['p'].append(p)
        prev_rs['PC'].append(pc_id + 1)
        prev_rs['behavior'].append(behavior)

prev_rs = pd.DataFrame(prev_rs)
    
# Plot correlation with behavior prevalence
from matplotlib.patches import Patch

colors = np.load('figures/colors_top-bars_tanh-z_'
                 f'pca-k{k}_reg-{reg}_m{matchup}.npy',
                 allow_pickle=True)

sns.set_context('notebook', font_scale=3)
sns.catplot(x='PC', y='correlation', row='behavior',
            data=prev_rs, color='.6', kind='bar',
            estimator=fisher_mean, palette=colors, height=6, aspect=7)
plt.xticks([0, 19, 39, 59, 79, 99])
legend_elements = [Patch(facecolor='tab:red'),
                   Patch(facecolor='tab:blue'),
                   Patch(facecolor='tab:purple'),
                   Patch(facecolor='tab:purple')]
plt.legend(handles=legend_elements, loc='upper right',
           labels=['', '', 'top 10 cooperative PCs',
                   'top 10 difference PCs'],
           ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5)


# Compare ISC values to score/winning
map = 0
matchup = 0
n_repeat = 0

# Load in ISC values
k = 100
reg = 'com'
iscs = np.load(f'results/iscs_tanh-z_pca-k{k}_reg-{reg}.npy')[matchup]
#iscs = np.load(f'results/iscs_tanh-z_pca-k{k}.npy')[matchup]

coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]
teams = [[0, 1], [2, 3]]
score_rs = {'correlation': [], 'PC': [], 'p': []}
diff_rs = {'correlation': [], 'PC': [], 'p': []}
for pc_id in np.arange(k):
    
    # Reorganize color-labeled scores
    scores = []
    for repeat in np.arange(n_repeats):
        if wrap_f['map/matchup/repeat/player/color_id'][
            map, matchup, repeat, 0, 0] == 0:
            team_colors = ['red', 'blue']
        elif wrap_f['map/matchup/repeat/player/color_id'][
            map, matchup, repeat, 0, 0] == 1:
            team_colors = ['blue', 'red']
        score0 = wrap_f[f'map/matchup/repeat/{team_colors[0]}_team_score'][
            map, matchup, repeat, 0].astype(float)
        score1 = wrap_f[f'map/matchup/repeat/{team_colors[1]}_team_score'][
            map, matchup, repeat, 0].astype(float)
        scores.append([score0, score1])
        
    scores = np.array(scores)

    # Correlate ISC with score across both teams
    iscs_cat = np.concatenate([iscs[:, coop_ids[0], pc_id],
                              iscs[:, coop_ids[1], pc_id]])
    scores_cat = np.concatenate([scores[:, 0], scores[:, 1]])
    score_r, score_p = pearsonr(iscs_cat, scores_cat)
    score_rs['correlation'].append(score_r)
    score_rs['p'].append(score_p)
    score_rs['PC'].append(pc_id + 1)
    
    # Correlate difference in ISC with difference in scores
    iscs_diff = (iscs[:, coop_ids[0], pc_id] -
                 iscs[:, coop_ids[1], pc_id])
    scores_diff = scores[:, 0] - scores[:, 1]
    diff_r, diff_p = pearsonr(iscs_diff, scores_diff)
    diff_rs['correlation'].append(diff_r)
    diff_rs['p'].append(diff_p)
    diff_rs['PC'].append(pc_id + 1)

score_rs = pd.DataFrame(score_rs)
diff_rs = pd.DataFrame(diff_rs)

# Bar plot of scores
sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 4))
sns.barplot(x='PC', y='correlation', data=score_rs, ax=ax, color='.6',
            estimator=fisher_mean, palette=colors)
#ax.set_ylim(-.375, .325) # for matchup = 3
#ax.set_ylim(-.05, .4) # for matchup = 0
ax.set_xticks([0, 19, 39, 59, 79, 99])
#ax.set_ylabel('cooperative – competitive ISC')
#ax.set_title(f'difference in cooperative vs. competitive ISC for 100 PCs')
sns.despine()
legend_elements = [Patch(facecolor='tab:red'),
                   Patch(facecolor='tab:blue'),
                   Patch(facecolor='tab:purple'),
                   Patch(facecolor='tab:purple')]
ax.legend(handles=legend_elements, loc='upper right',
          labels=['', '', 'top 10 cooperative PCs',
                  'top 10 difference PCs'],
          ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5)

# Bar plot of correlation between ISC and scores
sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 4))
sns.barplot(x='PC', y='correlation', data=diff_rs, ax=ax, color='.6',
            estimator=fisher_mean, palette=colors)
#ax.set_ylim(-.375, .325) # for matchup = 3
#ax.set_ylim(-.05, .4) # for matchup = 0
ax.set_xticks([0, 19, 39, 59, 79, 99])
#ax.set_ylabel('cooperative – competitive ISC')
#ax.set_title(f'difference in cooperative vs. competitive ISC for 100 PCs')
sns.despine()
legend_elements = [Patch(facecolor='tab:red'),
                   Patch(facecolor='tab:blue'),
                   Patch(facecolor='tab:purple'),
                   Patch(facecolor='tab:purple')]
ax.legend(handles=legend_elements, loc='upper right',
          labels=['', '', 'top 10 cooperative PCs',
                  'top 10 difference PCs'],
          ncol=2, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5)


# Load in ISCFs
reg = 'com'
k = 100
matchup = 0
repeat = 0
n_repeats = 8
top_pcs = {'PC3 CF': 2, 'PC8 CF': 7, 'PC10 CF': 9,
           'PC24 CF': 23, 'PC25 CF': 24}

coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]
teams = [[0, 1], [2, 3]]

iscfs = []
for repeat in np.arange(n_repeats):
    iscfs_rep = np.load(f'results/iscf_lstm_tanh-z_pca-k{k}_'
                        f'reg-{reg}_m{matchup}_r{repeat}.npy')
    iscfs_diag = np.diagonal(iscfs_rep, axis1=-2, axis2=-1)
    iscfs.append(iscfs_diag)
    print(f"Loaded ISCFs for matchup {matchup} repeat {repeat}")
iscfs = np.stack(iscfs, axis=0)
#iscfs_top = np.moveaxis(iscfs[matchup, ..., top_pcs, top_pcs], 0, 3)


# Compute mean ISCF (i.e. ISC) within indicator blocks for all PCs
block_iscfs = {'block CF': [], 'PC': [], 'behavior': [], 'repeat': []}
behaviors = behaviors.pop('teammate has flag')
for behavior in behaviors:
    for repeat in np.arange(n_repeats):
        for pc in np.arange(k):
            coop_rs = []
            for coop_id, players in zip(coop_ids, teams):
                team_rs = []
                for player in players:
                    r = block_iscf(iscfs[repeat, coop_id, :, pc],
                                   behaviors[behavior][repeat, player])
                    team_rs.append(r)
                
                if behavior in symmetric:
                    if not np.sum(np.isnan(team_rs)) == 2:
                        assert team_rs[0] == team_rs[1]
                    coop_r = team_rs[0]
                else:
                    assert team_rs[0] != team_rs[1]
                    coop_r = fisher_mean(team_rs)
                    
                coop_rs.append(coop_r)
                
            block_iscfs['block CF'].append(fisher_mean(coop_rs))
            block_iscfs['PC'].append(pc + 1)
            block_iscfs['behavior'].append(behavior)
            block_iscfs['repeat'].append(repeat)

block_iscfs = pd.DataFrame(block_iscfs)

colors = np.load('figures/colors_top-bars_tanh-z_'
                 f'pca-k{k}_reg-{reg}_m{matchup}.npy',
                 allow_pickle=True)
sns.set_context('notebook', font_scale=1.2)
fig, ax = plt.subplots(figsize=(16, 4))
sns.barplot(x='PC', y='block CF', color='.6',
            data=block_iscfs[block_iscfs['behavior'] == 'spawncamping'],
            estimator=fisher_mean, palette=colors)
ax.set_ylabel('within-behavior cofluctuation')
ax.set_xticks([0, 19, 39, 59, 79, 99])
sns.despine()


# Focus in on top PCs
# Compute mean ISCF (i.e. ISC) within indicator blocks for all PCs
behaviors.pop('teammate has flag')
top_pcs = {'PC3': 2, 'PC8': 7, 'PC10': 9,
           'PC24': 23, 'PC25': 24}
block_iscfs = {'block CF': [], 'PC': [], 'behavior': [], 'repeat': [],
               'team': []}
for behavior in behaviors:
    for repeat in np.arange(n_repeats):
        for pc in top_pcs:
            coop_rs = []
            for coop_id, players in zip(coop_ids, teams):
                team_rs = []
                for player in players:
                    r = block_iscf(iscfs[repeat, coop_id, :, top_pcs[pc]],
                                   behaviors[behavior][repeat, player],
                                   precede=30)
                    team_rs.append(r)
                
                if behavior in symmetric:
                    if not np.sum(np.isnan(team_rs)) == 2:
                        assert team_rs[0] == team_rs[1]
                    coop_r = team_rs[0]
                else:
                    assert team_rs[0] != team_rs[1]
                    coop_r = fisher_mean(team_rs)
                    
                coop_rs.append(coop_r)
                
            block_iscfs['block CF'].append(fisher_mean(coop_rs))
            block_iscfs['PC'].append(pc)
            block_iscfs['behavior'].append(behavior)
            block_iscfs['repeat'].append(repeat)
            block_iscfs['team'].append('matching')

            coop_rs = []
            for coop_id, players in zip(coop_ids, teams[::-1]):
                team_rs = []
                for player in players:
                    r = block_iscf(iscfs[repeat, coop_id, :, top_pcs[pc]],
                                   behaviors[behavior][repeat, player])
                    team_rs.append(r)
                
                if behavior in symmetric:
                    if not np.sum(np.isnan(team_rs)) == 2:
                        assert team_rs[0] == team_rs[1]
                    coop_r = team_rs[0]
                else:
                    assert team_rs[0] != team_rs[1]
                    coop_r = fisher_mean(team_rs)
                    
                coop_rs.append(coop_r)
                
            block_iscfs['block CF'].append(fisher_mean(coop_rs))
            block_iscfs['PC'].append(pc)
            block_iscfs['behavior'].append(behavior)
            block_iscfs['repeat'].append(repeat)
            block_iscfs['team'].append('mismatching')
            
block_iscfs = pd.DataFrame(block_iscfs)

sns.set_context("notebook", font_scale=1.05)
c = sns.catplot(x='block CF', y='PC', row='behavior',
                color='tab:red', legend=False,
                data=block_iscfs[block_iscfs['team'] == 'matching'],
                kind='point', join=False, seed=1,
                estimator=fisher_mean, height=1.75, aspect=2)
c.set_titles(row_template='{row_name}')
plt.xlabel('within-behavior cofluctuation')
axs = plt.gca()
for ax in c.axes:
    ax[0].tick_params(axis='y', length = 0)
    ax[0].axvline(0, ls='--', c='.7', zorder=-1)
    ax[0].set_xlim(-.05, .52)
handles, labels = c.axes[0, 0].get_legend_handles_labels()
plt.savefig('figures/iscf_block-iscf_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')

sns.set_context("notebook", font_scale=1.05)
c = sns.catplot(x='block CF', y='PC', row='behavior', hue='team',
                hue_order=['mismatching', 'matching'], seed=1,
                palette=['.7', 'tab:red'], legend=False,
                data=block_iscfs, kind='point', join=False,
                estimator=fisher_mean, height=1.75, aspect=2)
c.set_titles(row_template='{row_name}')
plt.xlabel('within-behavior cofluctuation')
axs = plt.gca()
for ax in c.axes:
    ax[0].tick_params(axis='y', length = 0)
    ax[0].axvline(0, ls='--', c='.7', zorder=-1)
    ax[0].set_xlim(-.05, .52)
#handles, labels = c.axes[0, 0].get_legend_handles_labels()
#l = c.axes[0, 0].legend(handles[::-1], labels[::-1],
#                        loc='upper right', bbox_to_anchor=(1.7, 1.2),
#                         title='team', handletextpad=0, borderpad=.5)
#l._legend_box.align = 'left'
plt.savefig('figures/iscf_block-iscf-team_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')

#sns.set_context("notebook", font_scale=1.9)
#c = sns.catplot(x='behavior', y='block CF', col='PC',
#                data=block_iscfs, kind='bar', color='.5',
#                estimator=fisher_mean, height=6, aspect=.8);
#c.set_titles(col_template='{col_name}')
#c.set(xlabel=None)
#c.set_xticklabels(rotation=90, ha='center')
#plt.savefig('figures/iscf_block-iscf_tanh-z_'
#            f'pca-k{k}_reg-{reg}_m{matchup}.png',
#            dpi=300, bbox_inches='tight')




# Try this with block randomization
block_iscfs = {'block CF (z)': [], 'PC': [], 'behavior': [], 'repeat': []}
for behavior in behaviors:
    for repeat in np.arange(n_repeats):
        for pc in top_pcs:
            coop_rs = []
            for coop_id, players in zip(coop_ids, teams):
                team_rs = []
                for player in players:
                    _, z, _, _ = block_randomization(
                        iscfs[repeat, coop_id, :, top_pcs[pc]],
                        behaviors[behavior][repeat, player])
                    team_rs.append(z)
                
                coop_r = np.mean(team_rs)
                coop_rs.append(coop_r)
            
            print("Finished block randomization for "
                  f"matchup {matchup} repeat {repeat} {pc}")
            block_iscfs['block CF (z)'].append(np.mean(coop_rs))
            block_iscfs['PC'].append(pc)
            block_iscfs['behavior'].append(behavior)
            block_iscfs['repeat'].append(repeat)

block_iscfs = pd.DataFrame(block_iscfs)

sns.set_context("notebook", font_scale=1.9)
c = sns.catplot(x='behavior', y='block CF (z)', col='PC',
                data=block_iscfs, kind='bar', color='.5',
                height=6, aspect=.8);
c.set(xlabel=None)
c.set_xticklabels(rotation=90, ha='center')
c.set_titles(col_template='{col_name}')
plt.savefig('figures/iscf_block-iscf_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')


# Correlate ISCF time series with behavioral time series
iscf_rs = {'correlation': [], 'PC': [], 'behavior': [], 'repeat': []}
for repeat in np.arange(n_repeats):
    iscfs = np.load(f'results/iscf_lstm_tanh-z_pca-k{k}_'
                    f'reg-{reg}_m{matchup}_r{repeat}.npy')
    print(f"Loaded ISCFs for repeat {repeat}")
    
    for pc in top_pcs:
        for behavior in behaviors:
            coop_rs = []
            for coop_id, players in zip(coop_ids, teams):
                team_rs = []
                for player in players:
                    r = pearsonr(iscfs[coop_id, :, top_pcs[pc], top_pcs[pc]],
                                 behaviors[behavior][repeat, player])[0]
                    team_rs.append(r)

                if behavior in symmetric:
                    assert team_rs[0] == team_rs[1]
                    coop_r = team_rs[0]
                else:
                    assert team_rs[0] != team_rs[1]
                    coop_r = fisher_mean(team_rs)

                coop_rs.append(coop_r)

            iscf_rs['correlation'].append(fisher_mean(coop_rs))
            iscf_rs['PC'].append(pc)
            iscf_rs['behavior'].append(behavior)
            iscf_rs['repeat'].append(repeat)
        
iscf_rs = pd.DataFrame(iscf_rs)

sns.set_context("notebook", font_scale=1.05)
c = sns.catplot(x='correlation', y='PC', row='behavior',
                data=iscf_rs, kind='point', color='.5', join=False,
                estimator=fisher_mean, height=1.75, aspect=2)
c.set_titles(row_template='{row_name}')
plt.xlabel('correlation with behavior')
axs = plt.gca()
for ax in c.axes:
    ax[0].tick_params(axis='y', length = 0)
    ax[0].axvline(0, ls='--', c='.7', zorder=-1)
    #ax[0].set_xlim(0, .5)
plt.savefig('figures/iscf_r-behavior_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')
#sns.set_context("notebook", font_scale=1.9)
#c = sns.catplot(x='behavior', y='correlation', col='PC',
#                data=iscf_rs, kind='bar', color='.5',
#                estimator=fisher_mean, height=6, aspect=.8);
#c.set(xlabel=None)
#c.set_xticklabels(rotation=90, ha='center')
#c.set_titles(col_template='{col_name}')
#plt.savefig('figures/iscf_r-behavior_tanh-z_'
#            f'pca-k{k}_reg-{reg}_m{matchup}.png',
#            dpi=300, bbox_inches='tight')


# Correlate ISCF time series with behavioral time series
top_pcs = {'PC3': 2, 'PC8': 7, 'PC10': 9,
           'PC24': 23, 'PC25': 24}
iscf_rs = {'correlation': [], 'PC': [], 'behavior': [], 'repeat': [],
           'team': []}
for repeat in np.arange(n_repeats):
    iscfs = np.load(f'results/iscf_lstm_tanh-z_pca-k{k}_'
                    f'reg-{reg}_m{matchup}_r{repeat}.npy')
    print(f"Loaded ISCFs for repeat {repeat}")
    
    for pc in top_pcs:
        for behavior in behaviors:
            coop_rs = []
            for coop_id, players in zip(coop_ids, teams):
                team_rs = []
                for player in players:
                    r = pearsonr(iscfs[coop_id, :, top_pcs[pc], top_pcs[pc]],
                                 behaviors[behavior][repeat, player])[0]
                    team_rs.append(r)

                if behavior in symmetric:
                    assert team_rs[0] == team_rs[1]
                    coop_r = team_rs[0]
                else:
                    assert team_rs[0] != team_rs[1]
                    coop_r = fisher_mean(team_rs)

                coop_rs.append(coop_r)

            iscf_rs['correlation'].append(fisher_mean(coop_rs))
            iscf_rs['PC'].append(pc)
            iscf_rs['behavior'].append(behavior)
            iscf_rs['repeat'].append(repeat)
            iscf_rs['team'].append('matching')
            
            coop_rs = []
            for coop_id, players in zip(coop_ids, teams[::-1]):
                team_rs = []
                for player in players:
                    r = pearsonr(iscfs[coop_id, :, top_pcs[pc], top_pcs[pc]],
                                 behaviors[behavior][repeat, player])[0]
                    team_rs.append(r)

                if behavior in symmetric:
                    assert team_rs[0] == team_rs[1]
                    coop_r = team_rs[0]
                else:
                    assert team_rs[0] != team_rs[1]
                    coop_r = fisher_mean(team_rs)

                coop_rs.append(coop_r)

            iscf_rs['correlation'].append(fisher_mean(coop_rs))
            iscf_rs['PC'].append(pc)
            iscf_rs['behavior'].append(behavior)
            iscf_rs['repeat'].append(repeat)
            iscf_rs['team'].append('mismatching')
        
iscf_rs = pd.DataFrame(iscf_rs)

sns.set_context("notebook", font_scale=1.05)
c = sns.catplot(x='correlation', y='PC', row='behavior',
                color='tab:red', legend=False,
                data=iscf_rs[iscf_rs['team'] == 'matching'],
                kind='point', join=False, seed=1,
                estimator=fisher_mean, height=1.75, aspect=2)
c.set_titles(row_template='{row_name}')
plt.xlabel('correlation with behavior')
axs = plt.gca()
for ax in c.axes:
    ax[0].tick_params(axis='y', length = 0)
    ax[0].axvline(0, ls='--', c='.7', zorder=-1)
    ax[0].set_xlim(-.06, .1)
handles, labels = c.axes[0, 0].get_legend_handles_labels()
plt.savefig('figures/iscf_r-behavior_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')

sns.set_context("notebook", font_scale=1.05)
c = sns.catplot(x='correlation', y='PC', row='behavior', hue='team',
                hue_order=['mismatching', 'matching'], seed=1,
                palette=['.7', 'tab:red'], legend=False,
                data=iscf_rs, kind='point', join=False,
                estimator=fisher_mean, height=1.75, aspect=2)
c.set_titles(row_template='{row_name}')
plt.xlabel('correlation with behavior')
axs = plt.gca()
for ax in c.axes:
    ax[0].tick_params(axis='y', length = 0)
    ax[0].axvline(0, ls='--', c='.7', zorder=-1)
    ax[0].set_xlim(-.06, .1)
#handles, labels = c.axes[0, 0].get_legend_handles_labels()
#l = c.axes[0, 0].legend(handles[::-1], labels[::-1],
#                        loc='upper right', bbox_to_anchor=(1.7, 1.2),
#                         title='team', handletextpad=0, borderpad=.5)
#l._legend_box.align = 'left'
plt.savefig('figures/iscf_r-behavior-team_tanh-z_'
            f'pca-k{k}_reg-{reg}_m{matchup}.png',
            dpi=300, bbox_inches='tight')

###

from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ctf_dataset.load import create_wrapped_dataset
from features import get_features

#import luke's heuristics functions 
from ctf_dataset.behaviours.heuristic import near_teammate
from ctf_dataset.behaviours.heuristic import camp_own_base, camp_opponent_base, running_forwards, running_backwards, approaching_own_base, approaching_opponent_base, approaching_own_flag, approaching_opponent_flag, approaching_teammate


#import sam's heuristics functions 
from detectors_sam import get_position, get_proximity, get_following

# Set base directory 
base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data')

# Create wrapped CTF dataset
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

map_id = 0 # 0
matchup_id = 54 # 0-54 (0, 34, 49, 54)
repeat_id = slice(None) # 0-7
player_id = slice(None) # 0-3

# Test functions by viewing output for one matchup one repeat 

#testing sam's following function 
following_sam = get_following(wrap_f, matchup_id=0, repeat_id=0)

#testing luke's following function 
following_luke = near_teammate(wrap_f, map_id=0, matchup_id=0, repeat_id=0,
                         min_behaviour_length=30, teammate_radius=2)

## Plotting to verify 

plt.plot(following_sam[0].T, label='sam p1')
plt.plot(following_sam[1].T, label='sam p2')

# Plotting Luke's 
fig, axs = plt.subplots(6,1)
axs[0].plot(following_luke[0,:,0])
axs[1].plot(following_luke[1,:,0])
axs[2].plot(following_luke[2,:,0])
axs[3].plot(following_luke[3,:,0])
axs[4].plot(following_sam[0,:])
axs[5].plot(following_sam[1,:])

# Compare Sam and Luke's directly 
# It appears that Luke's function takes account of isalive and this limits the time courses for following (as it should be)

fig, axs = plt.subplots(2,1)
axs[0].plot(following_luke[0,:500,0], label='luke')
axs[1].plot(following_sam[0,:500], label='sam')
plt.legend()

from animations import time_series_animation
from IPython.display import Video

## Testing basecamping functions 

camp_own = camp_own_base(wrap_f, map_id=0, matchup_id=0, repeat_id=0,
                         min_behaviour_length=15, base_radius=1)

# Plot 

fig, axs = plt.subplots(4,1)
axs[0].plot(camp_own[0,:,0], label='p1')
axs[1].plot(camp_own[1,:,0], label='p2')
axs[2].plot(camp_own[2,:,0], label='p3')
axs[3].plot(camp_own[3,:,0], label='p4')
fig.legend()

## Animation 
map_id = 0
matchup_id = 0
repeat_id = 0
anim = time_series_animation(camp_own[[0,2],:,0], wrap_f, map_id=map_id, matchup_id=matchup_id, repeat_id=repeat_id, label='basecamping')


anim.save(f'figures/time_series_animation_basecamping_min-br_m{matchup_id}_'
          f'r{repeat_id}.mp4', dpi=90)

Video(f'figures/time_series_animation_basecamping_min-br_m{matchup_id}_' f'r{repeat_id}.mp4')

## Testing basecamping functions 

camp_opp = camp_opponent_base(wrap_f, map_id=0, matchup_id=0, repeat_id=0,
                         min_behaviour_length=15, base_radius=1)

# Plot 

fig, axs = plt.subplots(4,1)
axs[0].plot(camp_opp[0,:,0], label='p1')
axs[1].plot(camp_opp[1,:,0], label='p2')
axs[2].plot(camp_opp[2,:,0], label='p3')
axs[3].plot(camp_opp[3,:,0], label='p4')
fig.legend()

## Animation 
map_id = 0
matchup_id = 0
repeat_id = 0
anim = time_series_animation(camp_opp[[0,2],:,0], wrap_f, map_id=map_id, matchup_id=matchup_id, repeat_id=repeat_id, label='spawncamping')


anim.save(f'figures/time_series_animation_spawncamping_min-br_m{matchup_id}_' f'r{repeat_id}.mp4', dpi=90)

Video(f'figures/time_series_animation_spawncamping_min-br_m{matchup_id}_' f'r{repeat_id}.mp4')

### It appears that what we're categorizing as 'spawncamping' might actually be being 'trapped' at opponent's plate after flag capture but before being killed. A kill follows almost every instance of 'spawncamping'


