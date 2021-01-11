from os.path import join
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ctf_dataset.load import create_wrapped_dataset
from features import get_features

base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data')

# Create wrapped CTF dataset
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

map_id = 0 # 0
matchup_id = 54 # 0-54 (0, 34, 49, 54)
repeat_id = slice(None) # 0-7
player_id = slice(None) # 0-3

# Let's start with the simple actions
feature_set = ['actions']

actions, action_labels = get_features(wrap_f,
                                      feature_set=feature_set,
                                      map_id=map_id,
                                      matchup_id=matchup_id,
                                      repeat_id=repeat_id,
                                      player_id=player_id)

# Concatenate across repeats and players
actions = np.concatenate(actions, axis=1)
actions = np.concatenate(actions, axis=0)
n_repeats = 8
n_samples = 4501
n_players = 4
assert actions.shape[0] == n_repeats * n_samples * n_players

# Convert this to a longform dictionary for some visualization
actions_long = {'actions': [], 'channel': []}
for channel, label in zip(actions.T, action_labels):
    for action in channel:
        actions_long['actions'].append(action)
        actions_long['channel'].append(label)
actions_long = pd.DataFrame(actions_long)

# Plot histogram of actions
max_count = n_repeats * n_samples * n_players
sns.set(style='white', font_scale=1.65)
g = sns.catplot(x='actions', data=actions_long, hue='channel',
                dodge=False, col='channel', col_wrap=3, kind='count')
(g.set_xticklabels(['0', '1', '2', '3', '4'])
  .set_titles('{col_name}')
  .set(ylim=(0, max_count))
  .set(yticks=[0, max_count // 3, max_count // 3 * 2, max_count])
  .set_yticklabels(labels=[0, n_samples // 3, n_samples // 3 * 2, n_samples - 1]))
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(f'matchup {matchup_id}');


# Let's look at the frequency of these actions
from scipy.fft import rfft, rfftfreq
from scipy.stats import zscore

map_id = 0 # 0
matchup_id = 54 # 0-54 (0, 34, 49, 54)
repeat_id = slice(None) # 0-7
player_id = slice(None) # 0-3

actions, action_labels = get_features(wrap_f,
                                      feature_set=feature_set,
                                      map_id=map_id,
                                      matchup_id=matchup_id,
                                      repeat_id=repeat_id,
                                      player_id=player_id)

# Compute fast fourier transform on each action
yfs = {f: [] for f in action_labels}
for repeat in actions:
    for player in repeat:
        for action, label in zip(player.T, action_labels):
            yfs[label].append(np.abs(rfft(zscore(action))))

# Average fourier output across players and repeats
yfs = {a: np.mean(yfs[a], axis=0) for a in yfs}

# Plot fourier for different actions                  
xf = rfftfreq(actions.shape[2], 1/15)

# Convert fft output into longform dictionary for plotting
xyf_long = {'frequency': [], 'power': [], 'actions': []}
for action in yfs:
    for x, y in zip(xf, yfs[action]):
        xyf_long['actions'].append(action)
        xyf_long['power'].append(y)
        xyf_long['frequency'].append(x)
xyf_long = pd.DataFrame(xyf_long)

sns.set(style='white', font_scale=1.65)
g = sns.relplot(x='frequency', y='power', hue='actions', data=xyf_long,
                col='actions', col_wrap=3, kind='line', legend=False)
#g.set(xscale='log')
g.set_titles('{col_name}')
g.fig.subplots_adjust(top=0.9)
g.fig.suptitle(f'matchup {matchup_id}');
for ax, action in zip(g.axes, yfs):
    peak = f'peak = {xf[np.argmax(yfs[action])]:.3f} Hz'
    ax.text(1, 0.05, peak, ha='right', transform = ax.transAxes)


# Expand action space into binary dummy variables
def expand_actions(actions, action_labels, stack=True):
    
    subactions = {}
    for action, label in zip(actions.T, action_labels):
        
        # Expand look left/right into five separate actions
        n_subactions = len(np.unique(action))
        subactions[label] = np.zeros((actions.shape[0],
                                      n_subactions))
        for subaction in np.arange(n_subactions):
            subactions[label][:, subaction][
                action == subaction] = 1
        
        # Check that each subaction occurs uniquely in time
        assert np.all(np.sum(subactions[label], axis=1) == 1)
        
    if stack:
        subactions = np.concatenate([subactions[a] for a in subactions],
                                    axis=1)
            
    return subactions

# Plot expanded binary subactions
subactions = expand_actions(actions[0, 0, ...], action_labels)
plt.matshow(subactions[:150, :].T)
plt.xlabel('time')
plt.ylabel('subactions')
plt.yticks([0, 10, 20], [0, 10, 20])
plt.tick_params(axis='x', length=0)
plt.title("subactions for 10 seconds of gameplay")


# Extract position to look at proximity, approach, avoidance
from itertools import combinations

map_id = 0 # 0
matchup_id = 0 # 0-54 (0, 34, 49, 54)
repeat_id = 0 # 0-7
player_id = slice(None) # 0-3

feature_set = ['position']

position, position_labels = get_features(wrap_f,
                                         feature_set=feature_set,
                                         map_id=map_id,
                                         matchup_id=matchup_id,
                                         repeat_id=repeat_id,
                                         player_id=player_id)

# Ignore z-position for now
position = position[..., :2]

# Compute Euclidean distance over time for all payers of players
n_players = 4
proximities = []
for pair in combinations(np.arange(n_players), 2):
    proximities.append(np.sqrt(np.sum((position[pair[0], ...] -
                                       position[pair[1], ...]) ** 2,
                                      axis=1)))
proximities = np.array(proximities).T

# Get proximities for cooperating and competing agents
coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

prox_coop = proximities[:, coop_ids]
prox_comp = proximities[:, comp_ids]

# Load in spawns as well since they rapidly change proximity
feature_set = ['events']
events, event_labels = get_features(wrap_f,
                                    feature_set=feature_set,
                                    map_id=map_id,
                                    matchup_id=matchup_id,
                                    repeat_id=repeat_id,
                                    player_id=player_id)

spawns = events[..., event_labels.index('spawn player player')]

# Plot proximity with spawns
dur = 900
sns.set(style='white', font_scale=1.1)
fig, ax = plt.subplots(figsize=(10, 3))
sns.despine(fig=fig)
ax.plot(prox_coop[:dur, 0], color='darkred', alpha=.7)
ax.plot(prox_coop[:dur, 1], color='darkblue', alpha=.7)
ax.hlines(0, -10, dur+10, linestyle='--', color='.5')
ax.vlines(np.where(np.sum(spawns[:2, :dur], axis=0) > 0)[0],
          0, -2, color='darkred')
ax.vlines(np.where(np.sum(spawns[2:4, :dur], axis=0) > 0)[0],
          0, -2, color='darkblue')
ax.set_ylabel('Euclidean distance')
ax.set_xlabel('time')
plt.title('Proximity between cooperating agents on both teams\n'
          f'(matchup {matchup_id}; 1 minute of gameplay; '
          'spawns for reference)')

# Remove spawns from proximity derivative
spawn_mask = []
for pair in combinations(np.arange(n_players), 2):
    spawn_mask.append(np.sum(spawns[pair, :], axis=0) > 0)
spawn_mask = np.column_stack(spawn_mask)

prox_nospawn = np.where(~spawn_mask, proximities, np.nan)

# Compute first difference of proximity
prox_deriv = np.diff(prox_nospawn, axis=0)
deriv_coop = prox_deriv[:, coop_ids]
deriv_comp = prox_deriv[:, comp_ids]

# Plot derivative of proximity with spawns
dur = 900
sns.set(style='white', font_scale=1.1)
fig, ax = plt.subplots(figsize=(10, 3))
sns.despine(fig=fig)
ax.plot(deriv_coop[:dur, 0], color='darkred', alpha=.7)
ax.plot(deriv_coop[:dur, 1], color='darkblue', alpha=.7)
ax.hlines(0, -10, dur+10, linestyle='--', color='.5')
ax.vlines(np.where(np.sum(spawns[:2, :dur], axis=0) > 0)[0],
          -.6, -.7, color='darkred')
ax.vlines(np.where(np.sum(spawns[2:4, :dur], axis=0) > 0)[0],
          -.6, -.7, color='darkblue')
ax.set_ylabel('derivative of distance\n(approach  —  avoid)')
ax.set_xlabel('time')
ax.set_ylim(-.7, .7)
plt.title('Change in distance between cooperating agents on both teams\n'
          f'(matchup {matchup_id}; 1 minute of gameplay; '
          'spawns for reference)')

# Get proximities across all repeats
map_id = 0 # 0
matchup_id = 54 # 0-54 (0, 34, 49, 54)
repeat_id = slice(None) # 0-7
player_id = slice(None) # 0-3

feature_set = ['position']

position, position_labels = get_features(wrap_f,
                                         feature_set=feature_set,
                                         map_id=map_id,
                                         matchup_id=matchup_id,
                                         repeat_id=repeat_id,
                                         player_id=player_id)

# Ignore z-position for now
position = position[..., :2]

# Compute Euclidean distance over time for all payers of players
n_repeats = 8
n_players = 4
proximities = []
for repeat in np.arange(n_repeats):
    rep_prox = []
    for pair in combinations(np.arange(n_players), 2):
        rep_prox.append(np.sqrt(np.sum((position[repeat, pair[0], ...] -
                                        position[repeat, pair[1], ...]) ** 2,
                                       axis=1)))
    proximities.append(rep_prox)
proximities = np.array(proximities)


# DFT on proximities 
yfs = []
for repeat in proximities:
    rep_yfs = []
    for pair in repeat:
        rep_yfs.append(np.abs(rfft(zscore(pair))))
    yfs.append(rep_yfs)
yfs = np.mean(yfs, axis=0)

# Plot fourier for different actions                  
xf = rfftfreq(actions.shape[2], 1/15)

# Convert fft output into longform dictionary for plotting
xyf_long = {'frequency': [], 'power': [], 'pairs': []}

yfs_coop = np.mean(yfs[coop_ids], axis=0)
yfs_comp = np.mean(yfs[comp_ids], axis=0)

for yf, pair in zip([yfs_coop, yfs_comp], ['cooperation', 'competition']):
    for x, y in zip(xf, yf):
        xyf_long['power'].append(y)
        xyf_long['frequency'].append(x)
        xyf_long['pairs'].append(pair)

xyf_long = pd.DataFrame(xyf_long)

sns.set(style='white', font_scale=1.2)
g = sns.relplot(x='frequency', y='power', hue='pairs', data=xyf_long,
                col='pairs', kind='line', legend=False)
g.set_titles('{col_name}')
g.fig.subplots_adjust(top=0.85)
g.fig.suptitle(f'Frequency of proximity fluctuations (matchup {matchup_id})');
g.axes[0][0].text(1, 0.15, f'peak = {xf[np.argmax(yfs_coop)]:.3f} Hz',
                  ha='right', transform = g.axes[0][0].transAxes)
g.axes[0][1].text(1, 0.15, f'peak = {xf[np.argmax(yfs_comp)]:.3f} Hz',
                  ha='right', transform = g.axes[0][1].transAxes)

