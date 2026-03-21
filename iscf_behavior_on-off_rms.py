from os.path import exists, join
from os import makedirs
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ctf_dataset.load import create_wrapped_dataset


base_dir = '/jukebox/hasson/snastase/social-ctf'
data_dir = join(base_dir, 'data_v1')
wrap_f = create_wrapped_dataset(data_dir, output_dataset_name="virtual.hdf5")

matchup_id = 0
n_maps = 32
n_repeats = 32


# Convenience function for computing root mean square (RMS)
def rms(a, axis=None):
    return np.sqrt(np.nanmean(a ** 2, axis=axis))


# Load in escort behavior
behavior_name = sys.argv[1]

if behavior_name == 'following':
    from behavior_heuristics import following_teammate
    map_id, repeat_id = slice(None), slice(None) # 0, 0
    matchup_id, player_id = 0, slice(None)

    behavior = following_teammate(wrap_f, map_id=map_id, matchup_id=matchup_id,
                      repeat_id=repeat_id, player_id=player_id)

if behavior_name == 'escort':
    from behavior_heuristics import escort
    map_id, repeat_id = slice(None), slice(None) # 0, 0
    matchup_id, player_id = 0, slice(None)

    behavior = escort(wrap_f, map_id=map_id, matchup_id=matchup_id,
                      repeat_id=repeat_id, player_id=player_id)

if behavior_name == 'spawncamping':
    from behavior_heuristics import spawn_camping
    map_id, repeat_id = slice(None), slice(None) # 0, 0
    matchup_id, player_id = 0, slice(None)

    behavior = spawn_camping(wrap_f, map_id=map_id, matchup_id=matchup_id,
                      repeat_id=repeat_id, player_id=player_id)
    
if behavior_name == 'cycling':
    from behavior_heuristics import cycling
    map_id, repeat_id = slice(None), slice(None) # 0, 0
    matchup_id, player_id = 0, slice(None)

    behavior = cycling(wrap_f, map_id=map_id, matchup_id=matchup_id,
                      repeat_id=repeat_id, player_id=player_id)
    
if behavior_name == 'mobbing':
    from behavior_heuristics import mobbing
    map_id, repeat_id = slice(None), slice(None) # 0, 0
    matchup_id, player_id = 0, slice(None)

    behavior = mobbing(wrap_f, map_id=map_id, matchup_id=matchup_id,
                      repeat_id=repeat_id, player_id=player_id)
    
if behavior_name == 'assist':
    from behavior_heuristics import assist
    map_id, repeat_id = slice(None), slice(None) # 0, 0
    matchup_id, player_id = 0, slice(None)
    
    behavior = assist(wrap_f, map_id=map_id, matchup_id=matchup_id,
                      repeat_id=repeat_id, player_id=player_id)

print(f"{behavior_name} frequency: {np.sum(behavior) / behavior.size:.3f}")


# Compute cofluctuation RMS for on and off behaviors
n_pcs = 142
team_pairs = {0: {'coop': 0, 'comp': [1, 2]},
              1: {'coop': 0, 'comp': [3, 4]},
              2: {'coop': 5, 'comp': [1, 3]},
              3: {'coop': 5, 'comp': [2, 4]}}

if not exists(f'results/{behavior_name}-on_iscfs'):
    makedirs(f'results/{behavior_name}-on_iscfs')
    makedirs(f'results/{behavior_name}-across_iscfs')
    makedirs(f'results/{behavior_name}-off_iscfs')


# Only need to compute the "on" (and "across") ISCFs once
if sys.argv[2] == 'on':
    on_iscfs, across_iscfs = [], []
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):
            iscfs = np.load(f'results/iscfs_matchup-{matchup_id}_'
                            f'map-{map_id}_repeat-{repeat_id}.npy',
                            allow_pickle=True)

            on_behavior = behavior[map_id, repeat_id]
            off_behavior = np.load(
                f'results/{behavior_name}-offov_matchup-{matchup_id}_'
                f'map-{map_id}_repeat-{repeat_id}_bool.npy')

            on_stack, off_stack, across_stack = [], [], []
            for player_id in team_pairs:
                coop_iscf = iscfs[team_pairs[player_id]['coop']]
                on_iscf = coop_iscf[on_behavior[player_id, :, 0]]
                #np.save(join(f'results/{behavior_name}-on_iscfs',
                #             f'{behavior_name}-on_matchup-{matchup_id}_'
                #             f'map-{map_id}_repeat-{repeat_id}_'
                #             f'player-{player_id}_iscfs.npy'), on_iscf)
                on_iscf = rms(on_iscf)
                
                comp_iscf = iscfs[team_pairs[player_id]['comp']]
                across_iscf = np.stack(
                    [comp_iscf[0, on_behavior[player_id, :, 0]],
                     comp_iscf[1, on_behavior[player_id, :, 0]]],
                                       axis=0)
                #np.save(join(f'results/{behavior_name}-across_iscfs',
                #             f'{behavior_name}-across_matchup-{matchup_id}_'
                #             f'map-{map_id}_repeat-{repeat_id}_'
                #             f'player-{player_id}_iscfs.npy'), across_iscf)
                across_iscf = np.nanmean([rms(across_iscf[0]),
                                          rms(across_iscf[1])])

                on_stack.append(on_iscf)
                across_stack.append(across_iscf)

            on_iscfs.append(np.nanmean(on_stack))
            across_iscfs.append(np.nanmean(across_stack))
            print(f"Finished {behavior_name} on/across "
                  f"ISCFs for map {map_id} repeat {repeat_id}",
                  flush=True)
    
    np.save(f'results/iscf_{behavior_name}-on_'
            f'matchup-{matchup_id}_results.npy',
            on_iscfs)
    
    np.save(f'results/iscf_{behavior_name}-across_'
            f'matchup-{matchup_id}_results.npy',
            across_iscfs)

else:

    resample_init = int(sys.argv[2]) - 1
    #resample_ids = np.arange(resample_init * 100,
    #                         resample_init * 100 + 100)
    resample_ids = np.arange(resample_init * 10,
                             resample_init * 10 + 10)
    print(f"Running resample chunk {resample_init}:")
    print(resample_ids)

    off_iscfs = []
    for map_id in np.arange(n_maps):
        for repeat_id in np.arange(n_repeats):
            iscfs = np.load(f'results/iscfs_matchup-{matchup_id}_'
                            f'map-{map_id}_repeat-{repeat_id}.npy',
                            allow_pickle=True)

            on_behavior = behavior[map_id, repeat_id]
            off_behavior = np.load(
                f'results/{behavior_name}-offov_matchup-{matchup_id}_'
                f'map-{map_id}_repeat-{repeat_id}_bool.npy')

            off_stack = []
            for player_id in team_pairs:
                coop_iscf = iscfs[team_pairs[player_id]['coop']]
                off_iscf = []
                for resample_id in resample_ids:
                    off_r = coop_iscf[off_behavior[player_id, :, resample_id]]
                    #np.save(join(f'results/{behavior_name}-off_iscfs',
                    #         f'{behavior_name}-off_matchup-{matchup_id}_'
                    #         f'map-{map_id}_repeat-{repeat_id}_'
                    #         f'player-{player_id}_resample-{resample_id}_'
                    #         'iscfs.npy'), off_r)
                    off_iscf.append(rms(off_r))

                off_stack.append(off_iscf)

            off_iscfs.append(np.nanmean(off_stack, axis=0))   

            print(f"Finished summarizing {behavior_name} ISCF "
                  f"for map {map_id} repeat {repeat_id}",
                  flush=True)

    off_iscfs = np.stack(off_iscfs, axis=0)

    np.save(f'results/iscf_{behavior_name}-offov_'
            f'matchup-{matchup_id}_resample-{resample_init}_results.npy',
            off_iscfs)