from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.linalg import eigh

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
n_samples = 4501
n_actions = 6

action_labels = {'look left/right': [0., 1., 2., 3., 4.],
                 'look up/down': [0., 1., 2.],
                 'strafe left/right': [0., 1., 2.],
                 'move backward/forward': [0., 1., 2.],
                 'fire or switch': [0., 1., 2., 3.],
                 'jump': [0., 1.]}

n_subactions = len(np.concatenate(list(action_labels.values())))
assert n_subactions == 20

n_total = n_maps * n_repeats * n_players * n_samples

# Function for expanding 6 action channels into 20 subactions
def expand_actions(actions, action_labels):
    
    subactions = {}
    for action, label in zip(actions.T, action_labels):
        
        # Expand look left/right into five separate actions
        subactions[label] = np.zeros((actions.shape[0],
                                      len(action_labels[label])))
        for subaction in action_labels[label]:
            subactions[label][:, int(subaction)][
                action == subaction] = 1
        
        # Check that each subaction occurs uniquely in time
        #assert np.all(np.sum(subactions[label], axis=1) == 1)
        
    subactions = np.concatenate([subactions[a] for a in subactions],
                                axis=1)
    #assert np.array_equal(np.unique(b), np.array([0, 1]))
            
    return subactions

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

    else:
        proj = data @ vecs
        
    print("Finished running incremental PCA")

    return proj, vals, vecs


# Z-score and stack original actions across all maps, repeats, players
actions_stack = np.memmap(f'results/actions-stack_tanh-z_'
                          f'matchup-{matchup_id}.npy',
                          dtype='float64', mode='w+',
                          shape=(n_total, n_actions))
print(f"Populating stacked actions (shape: {actions_stack.shape})")

game_counter = 0
for map_id in np.arange(n_maps):    
    for repeat_id in np.arange(n_repeats):
        for player_id in np.arange(n_players):
            actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                map_id, matchup_id, repeat_id,
                player_id, ...].astype(np.float64)
            actions = zscore(actions, axis=0)
            start = game_counter * n_samples
            end = start + n_samples
            actions_stack[start:end] = actions
            actions_stack.flush()
            game_counter += 1
            print(f"Finished stacking map {map_id}, "
                  f"repeat {repeat_id}, player {player_id}")

# Stack expanded (sub)actions across all maps, repeats, players
actions_stack = np.memmap(f'results/actions-stack_tanh-z_'
                          f'matchup-{matchup_id}.npy',
                          dtype='float64', mode='w+',
                          shape=(n_total, n_subactions))
print(f"Populating stacked actions (shape: {actions_stack.shape})")

game_counter = 0
for map_id in np.arange(n_maps):    
    for repeat_id in np.arange(n_repeats):
        for player_id in np.arange(n_players):
            actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                map_id, matchup_id, repeat_id,
                player_id, ...].astype(np.float64)
            actions = expand_actions(actions, action_labels)
            start = game_counter * n_samples
            end = start + n_samples
            actions_stack[start:end] = actions
            actions_stack.flush()
            game_counter += 1
            print(f"Finished stacking map {map_id}, "
                  f"repeat {repeat_id}, player {player_id}")

proj, vals, vecs = incremental_pca(actions_stack, n_components=n_actions,
                                   n_samples=n_total)

# Compute proportion variance explained from eigenvalues
vaf = vals / np.sum(vals)

# Compute number of components required for percentage variance
percents = [.5, .75, .9, .95, .99]
for i, perc in enumerate(percents):
    k = np.sum(np.cumsum(vaf) <= perc) + 1
    print(f"{perc:.0%} variance: {k} PCs")


# Stack expanded joint cooperative (sub)actions
n_total = n_maps * n_repeats * n_players // 2 * n_samples

actions_stack = np.memmap(f'results/actions-coop-stack_tanh-z_'
                          f'matchup-{matchup_id}.npy',
                          dtype='float64', mode='w+',
                          shape=(n_total, n_subactions * 2))
print(f"Populating stacked actions (shape: {actions_stack.shape})")

game_counter = 0
for map_id in np.arange(n_maps):
    for repeat_id in np.arange(n_repeats):
        coop_stack = []
        for player_id in np.arange(n_players):
            actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                map_id, matchup_id, repeat_id,
                player_id, ...].astype(np.float64)
            actions = expand_actions(actions, action_labels)
            coop_stack.append(zscore(actions, axis=0))
        coop_stack = np.vstack((np.hstack((coop_stack[0], coop_stack[1])),
                                np.hstack((coop_stack[2], coop_stack[3]))))
        start = game_counter * n_samples * 2
        end = start + n_samples * 2
        actions_stack[start:end] = coop_stack
        actions_stack.flush()
        game_counter += 1
        print(f"Finished stacking map {map_id}, "
              f"repeat {repeat_id}")

proj, vals, vecs = incremental_pca(actions_stack, n_components=n_actions,
                                   n_samples=n_total)

np.save((f'results/actions-coop-stack_tanh-z_'
         f'pca-vals_matchup-{matchup_id}.npy'), vals)

# Compute proportion variance explained from eigenvalues
vaf = vals / np.sum(vals)

# Compute number of components required for percentage variance
percents = [.5, .75, .9, .95, .99]
for i, perc in enumerate(percents):
    k = np.sum(np.cumsum(vaf) <= perc) + 1
    print(f"{perc:.0%} variance: {k} PCs")


# Stack expanded joint competitive (sub)actions
n_total = n_maps * n_repeats * n_players // 2 * n_samples

pairs = [((0, 2), (1, 3)), ((0, 3), (1, 2))]
for p, pair in enumerate(pairs):
    actions_stack = np.memmap(f'results/actions-comp{p + 1}-stack_tanh-z_'
                              f'matchup-{matchup_id}.npy',
                              dtype='float64', mode='w+',
                              shape=(n_total, n_subactions * 2))
    print(f"Populating stacked actions (shape: {actions_stack.shape})")

    game_counter = 0
    for map_id in np.arange(n_maps):    
        for repeat_id in np.arange(n_repeats):
            comp_stack = []
            for player_id in np.arange(n_players):
                actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                    map_id, matchup_id, repeat_id,
                    player_id, ...].astype(np.float64)
                actions = expand_actions(actions, action_labels)
                comp_stack.append(zscore(actions, axis=0))
            comp_stack = np.vstack((np.hstack((comp_stack[pair[0][0]],
                                               comp_stack[pair[0][1]])),
                                    np.hstack((comp_stack[pair[1][0]],
                                               comp_stack[pair[1][1]]))))
            start = game_counter * n_samples * 2
            end = start + n_samples * 2
            actions_stack[start:end] = comp_stack
            actions_stack.flush()
            game_counter += 1
            print(f"Finished stacking map {map_id}, "
                  f"repeat {repeat_id}")

    proj, vals, vecs = incremental_pca(actions_stack, n_components=n_actions,
                                       n_samples=n_total)

    np.save((f'results/actions-comp{p + 1}-stack_tanh-z_'
             f'pca-vals_matchup-{matchup_id}.npy'), vals)

    # Compute proportion variance explained from eigenvalues
    vaf = vals / np.sum(vals)

    # Compute number of components required for percentage variance
    percents = [.5, .75, .9, .95, .99]
    for i, perc in enumerate(percents):
        k = np.sum(np.cumsum(vaf) <= perc) + 1
        print(f"{perc:.0%} variance: {k} PCs")


# Compute similarity of discrete action variables
from itertools import combinations
from scipy.stats import pearsonr, kendalltau
from sklearn.metrics import matthews_corrcoef, mutual_info_score

coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]
repeat_id = 0

#Compute sample-by-sample Kendall's tau across action variables
coop_sims, comp_sims = [], []
for map_id in np.arange(n_maps):
    actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                     map_id, matchup_id, repeat_id, ...].astype(np.float64)
    for p, pair in enumerate(combinations(np.arange(n_players), 2)):
        for t in np.arange(n_samples):
            sim = kendalltau(actions[pair[0], t], actions[pair[1], t])[0]
            if p in coop_ids:
                coop_sims.apend(sim)
            else: comp_sims.append(sim)
    print(f"Finished computing Kendall's tau for map {map_id}")

# Compute Kendall's tau over samples for each action variable
coop_sims = {a: [] for a in action_labels}
comp_sims = {a: [] for a in action_labels}
for map_id in np.arange(n_maps):
    actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                     map_id, matchup_id, repeat_id, ...].astype(np.float64)
    for p, pair in enumerate(combinations(np.arange(n_players), 2)):
        for a, l in enumerate(action_labels):
            sim = kendalltau(actions[pair[0], :, a],
                             actions[pair[1], :, a])[0]
            if p in coop_ids:
                coop_sims[l].append(sim)
            else: comp_sims[l].append(sim)
    print(f"Finished computing Kendall's tau for map {map_id}")

# Compute Pearson r over samples for each action variable
coop_sims = {a: [] for a in action_labels}
comp_sims = {a: [] for a in action_labels}
for map_id in np.arange(n_maps):
    actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                     map_id, matchup_id, repeat_id, ...].astype(np.float64)
    for p, pair in enumerate(combinations(np.arange(n_players), 2)):
        for a, l in enumerate(action_labels):
            sim = pearsonr(actions[pair[0], :, a],
                           actions[pair[1], :, a])[0]
            if p in coop_ids:
                coop_sims[l].append(sim)
            else: comp_sims[l].append(sim)
    print(f"Finished computing Pearson r for map {map_id}")

# Compute mutual information over samples for each action variable
coop_sims = {a: [] for a in action_labels}
comp_sims = {a: [] for a in action_labels}
for map_id in np.arange(n_maps):
    actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                     map_id, matchup_id, repeat_id, ...].astype(np.float64)
    for p, pair in enumerate(combinations(np.arange(n_players), 2)):
        for a, l in enumerate(action_labels):
            sim = mutual_info_score(actions[pair[0], :, a],
                                    actions[pair[1], :, a])
            if p in coop_ids:
                coop_sims[l].append(sim)
            else: comp_sims[l].append(sim)
    print(f"Finished computing Pearson r for map {map_id}")

# Compute Matthews correlation over samples for each action variable
coop_sims = {a: [] for a in action_labels}
comp_sims = {a: [] for a in action_labels}
for map_id in np.arange(n_maps):
    actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                     map_id, matchup_id, repeat_id, ...].astype(np.float64)
    for p, pair in enumerate(combinations(np.arange(n_players), 2)):
        for a, l in enumerate(action_labels):
            sim = matthews_corrcoef(actions[pair[0], :, a],
                                    actions[pair[1], :, a])
            if p in coop_ids:
                coop_sims[l].append(sim)
            else: comp_sims[l].append(sim)
    print(f"Finished computing Pearson r for map {map_id}")
    
for label in action_labels:
    print(f'Coooperative {label} mean = {np.nanmean(coop_sims[label]):.4f} '
          f'(SD = {np.std(coop_sims[label]):.4f})')
    print(f'Coooperative {label} mean = {np.nanmean(comp_sims[label]):.4f} '
          f'(SD = {np.std(comp_sims[label]):.4f})')


# Stack original actions across all maps, repeats, players
n_total = n_maps * n_repeats * n_players * n_samples

actions_stack = np.memmap(f'results/actions-stack_tanh-z_'
                          f'matchup-{matchup_id}.npy',
                          dtype='float64', mode='w+',
                          shape=(n_total, n_actions))
print(f"Populating stacked actions (shape: {actions_stack.shape})")

game_counter = 0
for map_id in np.arange(n_maps):    
    for repeat_id in np.arange(n_repeats):
        for player_id in np.arange(n_players):
            actions = wrap_f[f'map/matchup/repeat/player/time/action'][
                map_id, matchup_id, repeat_id,
                player_id, ...].astype(np.float64)
            start = game_counter * n_samples
            end = start + n_samples
            actions_stack[start:end] = actions
            actions_stack.flush()
            game_counter += 1
            print(f"Finished stacking map {map_id}, "
                  f"repeat {repeat_id}, player {player_id}")

# Convert this to a longform dictionary for some visualization
actions_long = {'actions': [], 'channel': []}
for channel, label in zip(actions_stack.T, action_labels):
    for action in channel:
        actions_long['actions'].append(action)
        actions_long['channel'].append(label)
actions_long = pd.DataFrame(actions_long)

# Plot histogram of actions
max_count = n_maps * n_repeats * n_samples * n_players
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
plt.savefig(f'figures/actions_histogram_matchup-{matchup_id}.png', dpi=300,
            bbox_inches='tight')