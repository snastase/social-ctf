import sys
from os.path import join
import numpy as np
from itertools import product
from ctf_dataset.load import create_wrapped_dataset
from behavior_heuristics import (following_teammate, escort, cycling,
                                 assist, spawn_camping, mobbing)


base_dir = '/jukebox/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data_v1')
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

matchup_id = 0
n_maps = 32
n_repeats = 32
n_pcs = 142
n_isfcs = n_pcs * (n_pcs - 1) // 2 + n_pcs
team_pairs = {0: {'coop': 0, 'comp': [1, 2]},
              1: {'coop': 0, 'comp': [3, 4]},
              2: {'coop': 5, 'comp': [1, 3]},
              3: {'coop': 5, 'comp': [2, 4]}}
coop_pairs = [0, 0, 5, 5]
player_id = slice(None)


map_combs = list(product(np.arange(n_maps), np.arange(n_repeats)))
map_id, repeat_id = map_combs[int(sys.argv[1]) - 1]
#map_id, repeat_id = map_combs[0]


behavior_names = {'following': {'f': following_teammate, 'n': 'follow'},
                  'escort': {'f': escort, 'n': 'escort'},
                  'cycling': {'f': cycling, 'n': 'cycle'},
                  'spawncamping': {'f': spawn_camping, 'n': 'camp'},
                  'mobbing': {'f': mobbing, 'n': 'mob'},
                  'assist': {'f': assist, 'n': 'assist'}}


iscfs = np.load(f'results/iscfs_matchup-{matchup_id}_'
                f'map-{map_id}_repeat-{repeat_id}.npy',
                allow_pickle=True)
iscfs_tri = iscfs[..., np.triu_indices(n_pcs)[0],
                  np.triu_indices(n_pcs)[1]]
iscfs_coop = iscfs_tri[coop_pairs]
print(f"Loaded cooperative ISFCs for map {map_id}, repeat {repeat_id}")

iscf_behav = []
for behavior_name in behavior_names.items():
    behavior = behavior_name[1]['f'](
        wrap_f, map_id=map_id, matchup_id=matchup_id,
        repeat_id=repeat_id, player_id=player_id)

    assert iscfs_coop.shape[:-1] == behavior.shape[:-1]
    assert n_isfcs == iscfs_coop.shape[-1]

    iscf_pcs = []
    for pc_id in np.arange(n_isfcs):
        iscf_pcs.append(iscfs_coop[..., pc_id][behavior[..., 0]])
    iscf_pcs = np.column_stack(iscf_pcs)
    iscf_behav.append(iscf_pcs)
    print(f"Extracted ISFCs for {behavior_name[0]} "
          f"(map {map_id} repeat {repeat_id})", flush=True)

iscf_behav = np.vstack(iscf_behav)
    
    
# Infer number of samples from data
n_features = iscf_behav.shape[1]
n_total = iscf_behav.shape[0]
    
# Incrementally populate covariance matrix
cov = np.zeros((n_features, n_features))
for i, row in zip(np.arange(n_total), iscf_behav):
    outer = np.outer(row, row)
    cov += outer / (n_total - 1)
    if i > 0 and i % 10 == 0:
        print(f"Finished computing {i}th covariance", flush=True)

np.save(f'results/decode-behav_isfcs-cov_matchup-{matchup_id}_'
        f'map-{map_id}_repeat-{repeat_id}.npy', cov)
print(f"Finished constructing covariance for map {map_id} repeat {repeat_id}",
      flush=True)

# Check for jobs that didn't run/finish
#from glob import glob
#rerun_ids = []
#for f in glob('logs/isfc_behav-cov_475133_*.log'):
#    with open(f) as h:
#        s = h.read()
#    if 'DUE TO TIME LIMIT' in s:
#        rerun_ids.append(int(f.split('_')[-1].split('.')[0]))
#print(','.join(map(str, sorted(rerun_ids))))