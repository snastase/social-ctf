from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.spatial.distance import squareform
from itertools import combinations
from brainiak.isc import isc, isfc
from features import get_features
from features import get_events

# Load helper function(s) for interacting with CTF dataset
from ctf_dataset.load import create_wrapped_dataset


base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data')


# Helper function for Fisher-transformed average
def fisher_mean(correlations, axis=None):
    return np.tanh(np.mean(np.arctanh(correlations), axis=axis))


# Create wrapped CTF dataset
# This picks out the matchups (self vs. self) in which the behavior is easiest to identify
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

map_id = 0 # 0
matchup_id = 0 # 0-54
repeat_id = 0 # 0-7
player_id = 0 # 0-3

n_lstms = 512
n_repeats = 8
n_players = 4
n_pairs = n_players * (n_players - 1) // 2

# Print out some structure of the HDF5 dataset for convenience

# to look up what you're pulling from within the hdf5, index through to the desired level and then add .keys 
#something like wrap_f['map/matchup/repeat/player/time'].keys() or agent_ids.shape will reveal more about the internal structure
combined_key = ''
for key in ['map', 'matchup', 'repeat', 'player', 'time']:
    combined_key += key + '/'
    print(f"{combined_key}: \n\t{list(wrap_f[combined_key].keys())}\n")
    
    
# Get matchups with all same agents (e.g. AA vs AA)
agent_ids = wrap_f['map/matchup/repeat/player/agent_id'][0, :, :, :, 0]
matchup_ids = np.all(agent_ids[:, 0, :] == 
                     agent_ids[:, 0, 0][:, np.newaxis], axis=1)
n_matchups = np.sum(matchup_ids) # 0, 34, 49, 54


# Extract LSTMs for one map and matchup
# this is where we make the active choice to use post_lstm vs lstm
# we're tanh'ing the data as it's loaded in to truncate the values to a measurable range 
lstms_matched = np.tanh(wrap_f['map/matchup/repeat/player/time/lstm'][
    map_id, matchup_ids, ...].astype(np.float32)) 
print("Loaded LSTMs for within-population matchups")


# Loop through matchups and repeats
# The next 10 lines are the ones that are actually GENERATING the data (up until squareform)
isc_results = np.zeros((n_matchups, n_repeats, n_pairs,
                        lstms_matched.shape[-1]))
for matchup in np.arange(n_matchups):
    for repeat in np.arange(n_repeats):
    
        lstms = lstms_matched[matchup, repeat, ...]
        lstms = np.rollaxis(lstms, 0, 3)

        # Compute ISCs between each pair for 4 agents
        iscs = isc(lstms, pairwise=True)
        isc_results[matchup, repeat, ...] = iscs

        # Squareform ISCs for matrix visualization
        iscs_sq = []
        for u in iscs.T:
            u = squareform(u, checks=False)
            np.fill_diagonal(u, 1)
            iscs_sq.append(u)
        iscs_sq = np.dstack(iscs_sq)

        # Cooperative and competitive pairs from 4 x 4 (6) pairs
        coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

        iscs_coop = iscs[coop_ids, :]
        iscs_comp = iscs[comp_ids, :]

        plt.plot(fisher_mean(iscs_coop, axis=0))
        plt.plot(fisher_mean(iscs_comp, axis=0))
        plt.show()

        iscs_diff = np.tanh(np.arctanh(fisher_mean(iscs_coop, axis=0)) -
                            np.arctanh(fisher_mean(iscs_comp, axis=0)))
        plt.plot(iscs_diff)
        plt.show()
        print(f"Mean cooperative ISC (matchup {matchup}, repeat {repeat}): "
              f"{fisher_mean(iscs_coop):.3f}\n"
              f"Mean competitive ISC (matchup {matchup}, repeat {repeat}): "
              f"{fisher_mean(iscs_comp):.3f}\n"
              "Difference between coperative vs competitive ISC: "
              f"{fisher_mean(iscs_coop) - fisher_mean(iscs_comp):.3f}")
        print("Proportion of units with cooperative > competitive ISC: "
              f"{np.sum(iscs_diff > 0) / n_lstms:.3f}")

np.save('results/isc_post-lstm_results.npy', isc_results)
# commenting out the above line for now because it's breaking le code 


# Compare full results array
coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

# isc results index is all repeats, all matchups, coop/comp pairs, each LSTM
iscs_coop = np.mean(np.mean(isc_results[:, :, coop_ids, :], axis=2), axis=1)
iscs_comp = np.mean(np.mean(isc_results[:, :, comp_ids, :], axis=2), axis=1)

isc_diffs = []
for coop, comp in zip(iscs_coop, iscs_comp):
    isc_diffs.append(coop - comp)
    
# Convenience function for plotting grid of LSTM values
def plot_lstm_grid(lstms, n_rows=16, n_cols=32, title=None, **kwargs):

    lstm_grid = lstms.reshape(n_rows, n_cols)
    ratio = lstm_grid.shape[0] / lstm_grid.shape[1]

    fig, ax = plt.subplots(figsize=(8, 6))

    m = ax.matshow(lstm_grid, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('LSTM units', loc='left')
    ax.set_title(title)
    fig.colorbar(m, ax=ax, fraction=0.047 * ratio, pad=0.04)
    plt.show()

plot_lstm_grid(iscs_comp[3], title='competition ISC (matchup 54)',
               vmin=0, vmax=1)
    
plot_lstm_grid(isc_diffs[3],
               title='cooperation – competition ISC (matchup 54)',
               cmap='RdBu_r', vmin=-1, vmax=1)


# Relate ISCs to game score
# Grab scores for team one and two for within-population matchups
scores = wrap_f['map/matchup/repeat/player/my_team_score'][
            0, :, :, [0, 2], :][matchup_ids][..., 0]
scores_diff = scores[..., 0] - scores[..., 1]

iscs_coop = isc_results[:, :, coop_ids, :]
iscs_diff = isc_results[..., 0, :] - isc_results[..., 1, :]

# Loop through matchups and units and compute cor
diff_corrs, diff_ps = [], []
for m in np.arange(iscs_diff.shape[0]):
    matchup_corrs, matchup_ps = [], []
    
    for u in np.arange(iscs_diff.shape[2]):
        r, p = pearsonr(iscs_diff[m, :, u], scores_diff[m])
        matchup_corrs.append(r)
        matchup_ps.append(p)
        
    diff_corrs.append(matchup_corrs)
    diff_ps.append(matchup_ps)

diff_corrs = np.array(diff_corrs)
diff_ps = np.array(diff_ps)

m = 3
plot_lstm_grid(diff_corrs[m],
               title=('correlation between difference in ISC and\n'
                      'difference in score across repeats (matchup 49)'),
               cmap='RdBu_r', vmin=-1, vmax=1)

plot_lstm_grid(np.where(diff_ps < .05, diff_corrs, np.nan)[m],
               title=('correlation between difference in ISC and\n'
                      'difference in score across repeats '
                      '(p < .05; matchup 49)'),
               cmap='RdBu_r', vmin=-1, vmax=1)


# IMPORTANT SECTION
# Compute spatial ISC (ISPC) per time point
n_samples = 4501
map_id = 0
matchup_ids = [0, 34, 49, 54]
n_matchups = 4
n_repeats = 8
n_pairs = 6

# Extract LSTMs for one map and matchup (switch out post_lstm and lstm depending on analysis)
# we're tanh'ing the data as it's loaded in to truncate the values to a measurable range
lstms_matched = np.tanh(wrap_f['map/matchup/repeat/player/time/lstm'][
    map_id, matchup_ids, ...].astype(np.float32)) # swtich with post lstm when needed
print("Loaded LSTMs for within-population matchups")


# Loop through matchups and repeats
ispc_results = np.zeros((n_matchups, n_repeats, n_pairs, n_samples, n_samples))
for matchup in np.arange(n_matchups):
    for repeat in np.arange(n_repeats):
    
        lstms = lstms_matched[matchup, repeat, ...]
        lstms = np.rollaxis(lstms, 0, 3)
        lstms = np.rollaxis(lstms, 0, 2)

        # Compute ISCs between each pair for 4 agents
        ispcs = isfc(lstms, pairwise=True, vectorize_isfcs=False)
        ispc_results[matchup, repeat, ...] = ispcs
        print(f"finished ispc for matchup {matchup} repeat {repeat}")

#np.save('results/ispc_post-lstm_results.npy', ispc_results) 
np.save('results/ispc_lstm_tanh_results.npy', ispc_results)

# Compare full results array
coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

ispcs_coop = np.mean(ispc_results[:, :, coop_ids, ...], axis=2)
ispcs_comp = np.mean(ispc_results[:, :, comp_ids, ...], axis=2)

# Specify an example repeat
rep = 3
pop_labels = ['AA vs. AA', 'BB vs. BB', 'CC vs. CC', 'DD vs. DD']
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.suptitle(f'cooperation – competition spatial ISC (example repeat {rep})',
             y=.96)
for ax, m, p in zip(axs.ravel(), np.arange(n_matchups), pop_labels):
    mat = ax.matshow(ispcs_coop[m, rep] - ispcs_comp[m, rep], cmap='RdYlBu_r',
                     vmin=-.2, vmax=.2)
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)
    ax.set_title(p)
    if m > 1:
        ax.set_xlabel('time points')
fig.colorbar(mat, ax=axs.ravel().tolist(), fraction=0.025, pad=0.04);

# Get the average ISPC across episodes
pop_labels = ['AA vs. AA', 'BB vs. BB', 'CC vs. CC', 'DD vs. DD']
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
plt.suptitle(f'cooperation – competition spatial ISC (example repeat {rep})',
             y=.96)
# SADE: replaced e w/ rep to see if that clears the error 
for ax, m, p in zip(axs.ravel(), np.arange(n_matchups), pop_labels):
    mat = ax.matshow((np.mean(ispcs_coop[m], axis=0) -
                      np.mean(ispcs_comp[m], axis=0)),
                     cmap='RdYlBu_r', vmin=-.13, vmax=.13)
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)
    ax.set_title(p)
    if m > 1:
        ax.set_xlabel('time points')
fig.colorbar(mat, ax=axs.ravel().tolist(), fraction=0.025, pad=0.04)


# Get diagonal ISPC values i.e. time-point-by-time-point
ispcs_diag = np.diagonal(ispc_results, axis1=-2, axis2=-1)[..., coop_ids, :]
captures = wrap_f['map/matchup/repeat/player/time/events'][0, matchup_ids, ..., 7]

matchup = 0
repeat = 0
#plt.plot(captures[matchup, repeat, 0], color='maroon')
#plt.plot(captures[matchup, repeat, 2], color='darkblue')
#plt.plot(np.zeros(4501), color='gray')
plt.plot(ispcs_diag[matchup, repeat, 0], color='tomato', alpha=.5)
plt.plot(ispcs_diag[matchup, repeat, 1], color='deepskyblue', alpha=.5)
#plt.plot(ispcs_diag[matchup, repeat, 1], color='deepskyblue', alpha=.5)
plt.plot((np.cumsum(captures[matchup, repeat, 0]) /
          np.amax([np.amax(np.cumsum(captures[matchup, repeat, 0])),
                   np.amax(np.cumsum(captures[matchup, repeat, 2]))])), color='maroon')
plt.plot((np.cumsum(captures[matchup, repeat, 2]) /
          np.amax([np.amax(np.cumsum(captures[matchup, repeat, 0])),
                   np.amax(np.cumsum(captures[matchup, repeat, 2]))])), color='darkblue')
print(f'{pearsonr(captures[matchup, repeat, 0], ispcs_diag[matchup, repeat, 0])[0]:.3f}')
print(f'{pearsonr(captures[matchup, repeat, 2], ispcs_diag[matchup, repeat, 1])[0]:.3f}')
#plt.plot(captures[matchup, repeat, 2], color='darkblue')
plt.plot(np.zeros(4501), color='gray')


scores = wrap_f['map/matchup/repeat/player/my_team_score'][
            0, :, :, [0, 2], :][matchup_ids][..., 0]
coop_ids
reward = wrap_f['map/matchup/repeat/player/time/reward'][0, matchup_ids, :, :, :, 0]
values = wrap_f['map/matchup/repeat/player/time/reward'][0, matchup_ids, :, :, :, 0]


### 

# SADE: Test running one match/one repeat ISC and correlating against game variable 

# Load in existing ISPC results (in place of code chunk above)
ispc_results = np.load('results/ispc_lstm_tanh_results.npy')


# Exclude degenerate features from analysis 
feature_set = ['position', 'health', 'events']
all_features, labels = get_features(wrap_f, feature_set=feature_set, map_id=map_id,
                                    matchup_id=matchup_ids, player_id=slice(None),
                                    repeat_id=slice(None))

features_exclude = []
for label in labels: 
    features = all_features[..., np.array(labels) == label]
    n_nonzeros = np.sum(np.nonzero(features))
    print(f'checking {label} for all nonzeros; found {n_nonzeros} nonzeros')
    if n_nonzeros == 0:
        features_exclude.append(label)
        print(f'excluding {label}')
        
# Get proximities
def get_proximity(position):
    
    # Ignore z-position for now
    position = position[..., :2]

    # Compute Euclidean distance over time for all pay ers of players
    proximity = np.full(tuple(np.array(position.shape)[[0, 1, 3]]) + (6,), np.nan)
    for matchup in np.arange(position.shape[0]):
        for repeat in np.arange(position.shape[1]):
            for p, pair in enumerate(combinations(np.arange(position.shape[2]), 2)):
                proximity[matchup, repeat, :, p] = np.sqrt(np.sum((position[matchup, repeat, pair[0], ...] -
                                                 position[matchup, repeat, pair[1], ...]) ** 2,
                                                            axis=1))
    return proximity

position, _ = get_features(wrap_f, feature_set=['position'], map_id=map_id,
                           matchup_id=matchup_ids, player_id=slice(None),
                           repeat_id=slice(None))

proximity = get_proximity(position)

# Compare full results array
coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

# Compute correlation between game variable and ISPC 

#variables = wrap_f['map/matchup/repeat/player/time'].keys()

# Initialize dict to sort between pairs and individual players 
pair_player = {pair: player for pair, player in zip(np.arange(6), combinations(np.arange(4), 2))}

ispc_corrs = {} # big ultimate result dict 

#game_stats =  wrap_f['map/matchup/repeat/player/time'][game_var] # pull variable info
#matchup_stat = game_stats[map_id, matchup_id, rep_id, ...]
map_id = 0 
matchup_ids = [0, 34, 49, 54]
n_repeats = 8


# define a single variable to pull the stats for (this may be redundant)

for game_var in labels:
    # skips degenerate game variables 
    if game_var in features_exclude: 
        continue 
    features = all_features[..., np.array(labels) == game_var]
    #ispc_feature_corr[game_var] = {} # internal dict for game_var
    feature_shape = features.shape[:-2]
    if len(feature_shape) == 3: 
        ispc_corrs[game_var] = np.full(feature_shape[:-1] + (6, 2), np.nan)
    elif len(feature_shape) == 2:
        ispc_corrs[game_var] = np.full(feature_shape + (6,), np.nan)

    #ispc_corrs[game_var] = np.full((len(matchup_ids),n_repeats,6,2), np.nan)
    #n_nans = 0
    for m, matchup_id in enumerate(matchup_ids):
        for repeat_id in np.arange(n_repeats):   
            # loop through and extract player ids from each pairing
            for pair_id in np.arange(6): 
                #isolate the ispc for each player within the pair
                #if np.sum(features[m, repeat_id, pair_player[pair_id][0], :, 0] == 0) == n_samples: 
                #    n_nans += 1
                #    print(f"WARNING: {game_var} is all zeros for matchup {matchup_id} and repeat {repeat_id}")
                ispcs = np.diagonal(ispc_results[map_id, m, pair_id, ...])
                #print("ispc_coop for", pair_id, "is", ispc_coop.shape)
                if len(feature_shape) == 3: 
                    pl1_corr = pearsonr(features[m, repeat_id, pair_player[pair_id][0], :, 0], ispcs)[0]
                    pl2_corr = pearsonr(features[m, repeat_id, pair_player[pair_id][1], :, 0], ispcs)[0]                    
                    ispc_corrs[game_var][m, repeat_id, pair_id, 0] = pl1_corr
                    ispc_corrs[game_var][m, repeat_id, pair_id, 1] = pl2_corr
                elif len(feature_shape) == 2:
                    team_corr = pearsonr(features[m, repeat_id, :, 0], ispcs)[0]                    
                    ispc_corrs[game_var][m, repeat_id, pair_id] = team_corr

    #assert np.sum(np.isnan(ispc_corrs[game_var])) > n_nans # check to make sure array is filled w/ real values 
    print(f"finished ispc correlations w/ {game_var}")

# Save dictionary 
np.save('results/lstm_ispc_tanh_feature_correlations.npy', ispc_corrs) #switch with post when appropriate 

# To load dictionary
ispc_corrs = np.load('results/lstm_ispc_tanh_feature_correlations.npy', allow_pickle=True).item() #switch with post when appropriate 

# Compare ispc across repeats and matchups
coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]


# Coop Plot 

# Summarize ISPC Corrs across pairs and repeats
ispc_corr_coop_means = []

for game_var in ispc_corrs:
    if ispc_corrs[game_var].ndim == 4:
        ispc_corr_coop = ispc_corrs[game_var][..., coop_ids, :]
        ispc_corr_coop_means.append(np.nanmean(ispc_corr_coop, axis=(1, 2, 3)))
    elif ispc_corrs[game_var].ndim == 3:
        ispc_corr_coop = ispc_corrs[game_var][..., coop_ids]
        ispc_corr_coop_means.append(np.nanmean(ispc_corr_coop, axis=(1, 2)))
        
ispc_corr_coop_means = np.column_stack(ispc_corr_coop_means)


# Summarize ISPC Corrs across pairs and repeats
ispc_corr_comp_means = []

for game_var in ispc_corrs:
    if ispc_corrs[game_var].ndim == 4:
        ispc_corr_comp = ispc_corrs[game_var][..., comp_ids, :]
        ispc_corr_comp_means.append(np.nanmean(ispc_corr_comp, axis=(1, 2, 3)))
    elif ispc_corrs[game_var].ndim == 3:
        ispc_corr_comp = ispc_corrs[game_var][..., comp_ids]
        ispc_corr_comp_means.append(np.nanmean(ispc_corr_comp, axis=(1, 2)))

#eventually try to figure out how to get proximities and game status (winning vs. losing) 
ispc_corr_comp_means = np.column_stack(ispc_corr_comp_means)


# sorts between populations blah blah blah 
sorter = np.argsort(np.nanmean(np.vstack([ispc_corr_coop_means, 
                                          ispc_corr_comp_means]), axis=0))[::-1]

ispc_corr_coop_sorted = ispc_corr_coop_means[:, sorter]
ispc_corr_comp_sorted = ispc_corr_comp_means[:, sorter]
xlabels = np.array(list(ispc_corrs.keys()))[sorter]

#store variables to be manually exluded 
win_vars = ['player draw player', 'player loss player', 'player win player']

# Attempt to plot!!

#Coop 

plt.matshow(ispc_corr_coop_means[:, sorter], vmin=-.05, vmax=.05, cmap='RdBu_r')
plt.yticks([0, 1, 2, 3], ['A','B','C','D'])
plt.xticks(np.arange(ispc_corr_coop_means.shape[1]), xlabels, rotation=90);
plt.title("Cooperative")

#Comp 

plt.matshow(ispc_corr_comp_means[:, sorter], vmin=-.05, vmax=.05, cmap='RdBu_r')
plt.yticks([0, 1, 2, 3], ['A','B','C','D'])
plt.xticks(np.arange(ispc_corr_comp_means.shape[1]), xlabels, rotation=90);
plt.title("Competitive")

