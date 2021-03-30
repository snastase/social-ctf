from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from brainiak.isc import isc, isfc

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
lstms_matched = np.tanh(wrap_f['map/matchup/repeat/player/time/lstm'][
    map_id, matchup_ids, ...].astype(np.float32))
print("Loaded LSTMs for within-population matchups")


# Compute spatial ISC (ISPC) per time point
n_samples = 4501

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

np.save('results/ispc_lstm_tanh_results.npy', ispc_results)

ispc_results = np.load('results/ispc_lstm_tanh_results.npy')

# Compare full results array
coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

ispcs_coop = ispc_results[:, :, coop_ids, ...]
ispcs_comp = ispc_results[:, :, comp_ids, ...]

# Plot raw cooperative ISPCs
matchup = 3
repeat = 2
fig, axs = plt.subplots(1, 2, figsize=(8, 8))
for ax, p in zip(axs, np.arange(len(coop_ids))):
    mat = ax.matshow(ispcs_coop[matchup, repeat, p, ...],
                     cmap='RdYlBu_r', vmin=0, vmax=.75)
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)
fig.colorbar(mat, ax=axs.ravel().tolist(), fraction=0.021, pad=0.04);

# Plot raw competitive ISPCs
fig, axs = plt.subplots(2, 2, figsize=(8, 8))
for ax, p in zip(axs.ravel(), np.arange(len(comp_ids))):
    mat = ax.matshow(ispcs_comp[matchup, repeat, p, ...],
                     cmap='RdYlBu_r', vmin=0, vmax=.75)
    ax.xaxis.set_tick_params(which='both', length=0)
    ax.yaxis.set_tick_params(which='both', length=0)
fig.colorbar(mat, ax=axs.ravel().tolist(), fraction=0.021, pad=0.04);


# Difference between cooperative and competitive ISPCs for a given repeat
rep = 0
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
plt.suptitle(f'cooperation – competition spatial ISC (example repeat {e})',
             y=.96)
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

