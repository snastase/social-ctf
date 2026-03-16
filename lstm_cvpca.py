from os.path import join
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore
from scipy.linalg import eigh
from sklearn.model_selection import PredefinedSplit


# Load helper function(s) for interacting with CTF dataset
from ctf_dataset.load import create_wrapped_dataset

base_dir = '/jukebox/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data_v1')

# Create wrapped CTF dataset
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

# Get map ID from job scheduler
test_map = int(argv[1]) - 1

# Hard-code some dataset variables
matchup_id = 0
n_maps = 32
n_repeats = 32
n_players = 4
n_lstms = 512
n_samples = 4501

# Load in LSTMs by game, apply tanh, z-score, and save for memmapping
lstm = 'lstm'
n_total = n_maps * n_repeats * n_players * n_samples

lstms_stack = np.memmap(f'results/lstms-stack_tanh-z_matchup-{matchup_id}.npy',
                        dtype='float64', mode='w+', shape=(n_total, n_lstms))
print(f"Populating stacked LSTMs (shape: {lstms_stack.shape})")

game_counter = 0
for map_id in np.arange(n_maps):    
    for repeat_id in np.arange(n_repeats):
        for player_id in np.arange(n_players):
            lstms = wrap_f[f'map/matchup/repeat/player/time/{lstm}'][
                map_id, matchup_id, repeat_id,
                player_id, ...].astype(np.float64)
            lstms = zscore(np.tanh(lstms), axis=0)
            start = game_counter * n_samples
            end = start + n_samples
            lstms_stack[start:end] = lstms
            lstms_stack.flush()
            game_counter += 1
            print(f"Finished stacking map {map_id}, "
                  f"repeat {repeat_id}, player {player_id}")


# Function for reduced-memory incremental eigendecomposition
def incremental_cvpca(train_data, test_data, n_components=None,
                      train_filename=None, test_filename=None):

    # If number of samples isn't specified, infer from data
    n_features = train_data.shape[1]
    n_total = train_data.shape[0]
    
    # Incrementally populate covariance matrix
    cov = np.zeros((n_features, n_features))
    game_count = 0
    for i, row in zip(np.arange(n_total), train_data):
        outer = np.outer(row, row)
        cov += outer / (n_total - 1)
        if (i + 1) % (4 * 4501) == 0:
            print(f"Finished computing game {game_count} covariance")
            game_count += 1

    # Recover eignvalues and eigenvectors
    vals, vecs = eigh(cov)

    # Reorder values, vectors and extract column eigenvectors
    vals, vecs = vals[::-1], vecs[:, ::-1]
    if n_components:
        vecs = vecs[:, :n_components]

    # Project training data onto the eigenvectors
    train_proj = np.memmap(train_filename, dtype='float64',
                           mode='w+', shape=(train_data.shape[0],
                                             n_components))
    test_proj = np.memmap(test_filename, dtype='float64',
                          mode='w+', shape=(test_data.shape[0],
                                            n_components))

    train_proj[:] = train_data @ vecs
    train_proj.flush()
    
    test_proj[:] = test_data @ vecs
    test_proj.flush()
    raise

    return vals, vecs


map_split = np.repeat(np.arange(n_maps), n_repeats * n_players * n_samples)
assert map_split.shape[0] == lstms_stack.shape[0]
cv = PredefinedSplit(map_split)


train_filename = (f'results/TEST_lstms-stack_tanh-z_cvpca-train_'
                  f'matchup-{matchup_id}_map-{test_map}.npy')
test_filename = (f'results/TEST_lstms-stack_tanh-z_cvpca-test_'
                 f'matchup-{matchup_id}_map-{test_map}.npy')

cvs = list(cv.split())
train, test = cvs[test_map]
vals, vecs = incremental_cvpca(
    lstms_stack[train], lstms_stack[test], n_components=n_lstms,
    train_filename=train_filename, test_filename=test_filename)

np.save(f'results/lstms-stack_tanh-z_cvpca-vals_'
        f'matchup-{matchup_id}_map-{test_map}.npy', vals)
np.save(f'results/lstms-stack_tanh-z_cvpca-vecs_'
        f'matchup-{matchup_id}_map-{test_map}.npy', vecs)
print(f"Finished running incremental cvPCA for map {test_map}")
