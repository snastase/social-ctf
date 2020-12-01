from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from brainiak.isc import isc

# Load helper function(s) for interacting with CTF dataset
from ctf_dataset.load import create_wrapped_dataset


base_dir = '/mnt/bucket/labs/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data')


# Helper function for Fisher-transformed average
def fisher_mean(correlations, axis=None):
    return np.tanh(np.mean(np.arctanh(correlations), axis=axis))


# Create wrapped CTF dataset
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
lstms_matched = wrap_f['map/matchup/repeat/player/time/post_lstm'][
    map_id, matchup_ids, ...].astype(np.float32)
print("Loaded post-LSTMs for within-population matchups")

# Loop through matchups and repeats
isc_results = np.zeros((n_matchups, n_repeats, n_pairs, n_lstms))
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

np.save('results/isc_lstm_results.npy', isc_results)


# Compare full results array
coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

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
    
plot_lstm_grid(isc_diffs[2],
               title='cooperation – competition ISC (matchup 34)',
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
