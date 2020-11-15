from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.spatial.distance import squareform
from brainiak.isc import isc

# Load helper functions for interacting with CTF dataset
from ctf_dataset.load import create_wrapped_dataset
from ctf_dataset import info
from ctf_dataset.info import constants
from ctf_dataset.info import events
from ctf_dataset.load import virtual
from ctf_dataset.load import utils as load_utils


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

# Print out some structure of the HDF5 dataset for convenience
combined_key = ''
for key in ['map', 'matchup', 'repeat', 'player', 'time']:
    combined_key += key + '/'
    print(f"{combined_key}: \n\t{list(wrap_f[combined_key].keys())}\n")


# Extract LSTMs for one map and matchup
lstms_rep = wrap_f['map/matchup/repeat/player/time/lstm'][
    map_id, matchup_id, ...].astype(np.float32)

# Loop through repeats
for lstms in lstms_rep:
    lstms = np.rollaxis(np.tanh(lstms), 0, 3)

    # Compute ISCs between each pair for 4 agents
    iscs = isc(lstms, pairwise=True)

    # Squareform ISCs for matrix visualization
    iscs_sq = []
    for u in iscs.T:
        u = squareform(u, checks=False)
        np.fill_diagonal(u, 1)
        iscs_sq.append(u)
    iscs_sq = np.dstack(iscs_sq)

    # Cooperative and competitive pairs from 4 x 4 (6) pairs
    coop, comp = [0, 5], [1, 2, 3, 4]

    iscs_coop = iscs[coop, :]
    iscs_comp = iscs[comp, :]

    plt.plot(fisher_mean(iscs_coop, axis=0))
    plt.plot(fisher_mean(iscs_comp, axis=0))
    plt.show()

    iscs_diff = np.tanh(np.arctanh(fisher_mean(iscs_coop, axis=0)) -
                        np.arctanh(fisher_mean(iscs_comp, axis=0)))
    plt.plot(iscs_diff)
    plt.show()
    print(f"Mean cooperative ISC across units: {fisher_mean(iscs_coop):.3f}\n"
          f"Mean competitive ISC across units: {fisher_mean(iscs_comp):.3f}\n"
          "Difference between coperative vs competitive ISC: "
          f"{fisher_mean(iscs_coop) - fisher_mean(iscs_comp):.3f}")
    print("Proportion of units with cooperative > competitive ISC: "
          f"{np.sum(iscs_diff > 0) / n_lstms:.3f}")
