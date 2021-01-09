from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import squareform
from brainiak.isc import isfc

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
print("Loaded LSTMs for within-population matchups")

# Loop through matchups and repeats
isfc_results = np.zeros((n_matchups, n_repeats, n_pairs,
                         lstms_matched.shape[-1],
                         lstms_matched.shape[-1]))
for matchup in np.arange(n_matchups):
    for repeat in np.arange(n_repeats):
    
        lstms = lstms_matched[matchup, repeat, ...]
        lstms = np.rollaxis(lstms, 0, 3)

        # Compute ISCs between each pair for 4 agents
        isfcs = isfc(lstms, pairwise=True, vectorize_isfcs=False)
        isfc_results[matchup, repeat, ...] = isfcs
        
        print(f"Computed ISFC for matchup {matchup} (repeat {repeat})")

np.save('results/isfc_post-lstm_results.npy', isfc_results)


# Compute ISFC using sliding window
from time import time
n_samples = 4501
width = 150
onsets = np.arange(n_samples - width)
n_windows = len(onsets)

# Loop through matchups and repeats
for matchup in np.arange(n_matchups):
    for repeat in np.arange(n_repeats):
        
        win_isfc_results = np.zeros((n_windows, n_pairs,
                                     lstms_matched.shape[-1],
                                     lstms_matched.shape[-1]))

        lstms = lstms_matched[matchup, repeat, ...]
        lstms = np.rollaxis(lstms, 0, 3)

        start = time()
        for onset in onsets:
            window_lstm = lstms[onset:onset + width, ...]
            window_isfc = isfc(window_lstm, pairwise=True, vectorize_isfcs=False)
            win_isfc_results[onset, ...] = window_isfc
            if onset > 0 and onset % 500 == 0:
                print(f"Finished computing ISC for {onset} windows "
                      f"({time() - start:.1f} s elapsed)")

        np.save((f'results/isfc_win-{width}_post-lstm_matchup-{matchup}_'
                 f'repeat-{repeat}_results.npy'),
                win_isfc_results)
        print(f"Finished ISFC (window = {width}) for matchup {matchup} "
              f"(repeat {repeat})")


# Load and plot ISFC results
isfc_results = np.load('results/isfc_post-lstm_results.npy')

coop_ids, comp_ids = [0, 5], [1, 2, 3, 4]

isfcs_coop = np.mean(np.mean(isfc_results[:, :, coop_ids, ...],
                             axis=2),
                     axis=1)
isfcs_comp = np.mean(np.mean(isfc_results[:, :, comp_ids, ...],
                             axis=2),
                     axis=1)

matchup = 3
plt.matshow(isfcs_coop[matchup], vmin=-.2, vmax=.2, cmap='RdYlBu_r')
plt.matshow(isfcs_comp[matchup], vmin=-.2, vmax=.2, cmap='RdYlBu_r')
plt.matshow(isfcs_coop[matchup] - isfcs_comp[matchup],
            vmin=-.2, vmax=.2, cmap='RdBu_r')


# Load in windowed ISFCs
matchup, repeat = 0, 0
width = 150

win_isfc_results = np.load((f'results/isfc_win-{width}_post-lstm_'
                            f'matchup-{matchup}_repeat-{repeat}_results.npy'))