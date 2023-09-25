# Copyright 2020 The CTF Dataset Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Heuristic behaviors library"""

import itertools

import numpy as np

DEFAULT_MIN_BEHAVIOR_LENGTH = 15


def local_behavior_length(inp, time_axis):
  """Returns the local length of behavior."""

  n_times = inp.shape[time_axis]
  index_prefix = (slice(None),) * (
      time_axis if time_axis >= 0 else (len(inp.shape) + time_axis))
  continuous_time = np.zeros(inp.shape, dtype=np.int32)
  continuous_time[index_prefix + (0,)] = inp[
      index_prefix + (0,)].astype(np.int32)

  for t in range(1, n_times):
    is_true = inp[index_prefix + (t,)]
    update = continuous_time[index_prefix + (t-1,)] + 1
    continuous_time[index_prefix + (t,)] = np.where(is_true, update, 0)

  for t in range(n_times - 2, -1, -1):
    continuous_time[index_prefix + (t,)] = np.where(
        continuous_time[index_prefix + (t,)] > 0,
        np.maximum(continuous_time[index_prefix + (t + 1,)],
                   continuous_time[index_prefix + (t,)]),
        0)

  return continuous_time


def pad_behavior(inp, time_axis, before_steps=15, after_steps=15):
  """Returns padded behavior."""

  n_times = inp.shape[time_axis]
  index_prefix = (slice(None),) * (
      time_axis if time_axis >= 0 else (len(inp.shape) + time_axis))
  padded = inp.copy()

  for t in range(0, n_times-before_steps):
    is_true = np.any(
        inp[index_prefix + (slice(t, t+before_steps),)], axis=time_axis)
    padded[index_prefix + (t,)] |= is_true

  for t in range(after_steps, n_times):
    is_true = np.any(
        inp[index_prefix + (slice(t-after_steps, t),)], axis=time_axis)
    padded[index_prefix + (t,)] |= is_true

  return padded


def behavior_start_and_end(behavior, time_index, max_per_epsiode=16):
  """Returns start_time, end_times, and number of behaviors found."""
  n_times = behavior.shape[time_index]
  found = np.zeros(behavior.shape[:-2], dtype=np.int32)
  start_times = -np.ones(
      behavior.shape[:-2] + (max_per_epsiode,), dtype=np.int32)
  end_times = -np.ones(
      behavior.shape[:-2] + (max_per_epsiode,), dtype=np.int32)
  for t in range(n_times):
    for ind in itertools.product(*[range(s) for s in behavior.shape[:-2]]):
      # Get the index.
      found_ind = found[ind]
      if found_ind >= max_per_epsiode:
        continue
      ind_ = ind + (found_ind,)
      behaviors_ind = ind + (t, 0)

      # Start of a new behavior.
      start = (start_times[ind_] == -1) & behavior[behaviors_ind]
      if start:
        start_times[ind_] = t

      # End of a behavior:
      end = ((start_times[ind_] > 0) &
             (~behavior[behaviors_ind] | (t == (n_times - 1))))
      if end:
        end_times[ind_] = t
        found[ind] += 1

  return start_times, end_times, found


# Proximity behaviors
def camp_own_base(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    base_radius=1):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Near own base
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or ids
    min_behavior_length: Minimum behavior length (integer)
    base_radius: Maximum distance from base

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  player_from_own_base_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_own_base_xy_distance"][
          map_id, matchup_id, repeat_id, player_id]

  is_camping_own_base = is_alive.copy()
  is_camping_own_base &= player_from_own_base_xy_distance <= base_radius
  is_camping_own_base &= (
      local_behavior_length(is_camping_own_base, -2) >= min_behavior_length)

  return is_camping_own_base


def camp_opponent_base(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    base_radius=1):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Near opponent base
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    base_radius: Maximum distance from base

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  player_from_opponent_base_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_opponent_base_xy_distance"][
          map_id, matchup_id, repeat_id, player_id]

  is_camping_opponent_base = is_alive.copy()
  is_camping_opponent_base &= player_from_opponent_base_xy_distance <= base_radius
  is_camping_opponent_base &= (
      local_behavior_length(is_camping_opponent_base, -2)
      >= min_behavior_length)

  return is_camping_opponent_base


def spawn_camping(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    base_radius=1, flag_base_radius=2):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Near opponent base
  * Does not have opponent flag
  * Opponent flag is outside radius of opponent base
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    base_radius: Maximum distance from base

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  player_from_opponent_base_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_opponent_base_xy_distance"][
          map_id, matchup_id, repeat_id, player_id]
  opponent_flag_from_opponent_base_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/opponent_flag_from_opponent_base_xy_distance"][
          map_id, matchup_id, repeat_id, player_id]

  is_spawn_camping = is_alive.copy()
  is_spawn_camping &= player_from_opponent_base_xy_distance <= base_radius
  is_spawn_camping &= opponent_flag_from_opponent_base_xy_distance >= flag_base_radius
  is_spawn_camping &= (
      local_behavior_length(is_spawn_camping, -2)
      >= min_behavior_length)

  return is_spawn_camping


def cycling(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    teammate_angle=90, base_angle=90):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Teammate alive
  * Moving in opposite direction of teammate
  * Moving toward own or opponent base while
  * Teammate moving toward opposite base
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    base_radius: Maximum distance from base

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_is_alive = wrap_file[
      "map/matchup/repeat/player/time/teammate_is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  own_velocity = wrap_file[
      "map/matchup/repeat/player/time/velocity"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_velocity = wrap_file[
      "map/matchup/repeat/player/time/teammate_velocity"][
          map_id, matchup_id, repeat_id, player_id]
  own_base_position = wrap_file[
      "map/matchup/repeat/player/own_base_position"][
          map_id, matchup_id, repeat_id, player_id]
  opponent_base_position = wrap_file[
      "map/matchup/repeat/player/opponent_base_position"][
          map_id, matchup_id, repeat_id, player_id]

  # Angle via dot product between velocities
  velocity_angle = np.degrees(np.arccos(np.sum(own_velocity[..., :2] *
                                             teammate_velocity[..., :2], axis=-1) /
                                        (np.linalg.norm(own_velocity[..., :2], axis=-1) *
                                         np.linalg.norm(teammate_velocity[..., :2], axis=-1))))[..., np.newaxis]

  opponent_base_angle = np.degrees(np.arccos(np.sum(own_velocity[..., :2] *
                                                    opponent_base_position[..., np.newaxis, :2], axis=-1) /
                                             (np.linalg.norm(own_velocity[..., :2], axis=-1) *
                                              np.linalg.norm(opponent_base_position[..., np.newaxis, :2], axis=-1))))[..., np.newaxis]

  own_base_angle = np.degrees(np.arccos(np.sum(own_velocity[..., :2] *
                                               own_base_position[..., np.newaxis, :2], axis=-1) /
                                        (np.linalg.norm(own_velocity[..., :2], axis=-1) *
                                         np.linalg.norm(own_base_position[..., np.newaxis, :2], axis=-1))))[..., np.newaxis]

  teammate_opponent_base_angle = np.degrees(np.arccos(np.sum(teammate_velocity[..., :2] *
                                                    opponent_base_position[..., np.newaxis, :2], axis=-1) /
                                             (np.linalg.norm(teammate_velocity[..., :2], axis=-1) *
                                              np.linalg.norm(opponent_base_position[..., np.newaxis, :2], axis=-1))))[..., np.newaxis]

  teammate_own_base_angle = np.degrees(np.arccos(np.sum(teammate_velocity[..., :2] *
                                                        own_base_position[..., np.newaxis, :2], axis=-1) /
                                                 (np.linalg.norm(teammate_velocity[..., :2], axis=-1) *
                                                  np.linalg.norm(own_base_position[..., np.newaxis, :2], axis=-1))))[..., np.newaxis]

  # Approaching opponent base while teammate is approaching own base or vice versa
  approaching_opposite_bases = (((opponent_base_angle <= base_angle) & (teammate_own_base_angle <= base_angle)) |
                                ((own_base_angle <= base_angle) & (teammate_opponent_base_angle <= base_angle)))

  is_cycling = is_alive.copy()
  is_cycling &= teammate_is_alive
  is_cycling &= velocity_angle >= teammate_angle
  is_cycling &= approaching_opposite_bases
  is_cycling &= (
      local_behavior_length(is_cycling, -2)
      >= min_behavior_length)

  return is_cycling


def near_teammate(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    teammate_radius=1):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Teammate alive
  * Near teammate
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player Id or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    teammate_radius: Maximum distance from teammate

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_is_alive = wrap_file[
      "map/matchup/repeat/player/time/teammate_is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  player_from_teammate_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_teammate_xy_distance"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  is_near_teammate = is_alive.copy()
  is_near_teammate &= teammate_is_alive
  is_near_teammate &= player_from_teammate_xy_distance <= teammate_radius
  is_near_teammate &= (
      local_behavior_length(is_near_teammate, -2)
      >= min_behavior_length)

  return is_near_teammate


def following_teammate(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    teammate_radius=1, teammate_lag=15, following_angle=90, leading_angle=90):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Teammate alive
  * Near teammate
  * Velocity toward teammate within angle
  * Teammate velocity away exceeding angle
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player Id or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    teammate_radius: Maximum distance from teammate
    following_angle: Maximum angle for follower moving toward leader (degrees)
    leading_angle: Minimum angle for leader moving way from follower (degrees)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_is_alive = wrap_file[
      "map/matchup/repeat/player/time/teammate_is_alive"][
          map_id, matchup_id, repeat_id, player_id]

  player_from_teammate_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_teammate_xy_distance"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  player_position = wrap_file['map/matchup/repeat/player/time/position'][
      map_id, matchup_id, repeat_id, player_id].astype(np.float32)
    
  teammate_position = wrap_file['map/matchup/repeat/player/time/teammate_position'][
      map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  player_velocity = wrap_file['map/matchup/repeat/player/time/velocity'][
      map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  teammate_velocity = wrap_file['map/matchup/repeat/player/time/teammate_velocity'][
      map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  # Difference between player and teammate position
  position_difference = teammate_position[..., :2] - player_position[..., :2]

  # Angle via dot product between velocity and difference in position
  player_angle = np.degrees(np.arccos(np.sum(position_difference *
                                             player_velocity[..., :2], axis=-1) /
                                      (np.linalg.norm(position_difference, axis=-1) *
                                       np.linalg.norm(player_velocity[..., :2], axis=-1))))[..., np.newaxis]

  # Angle via dot product between velocity and difference in position
  teammate_angle = np.degrees(np.arccos(np.sum(-position_difference *
                                               teammate_velocity[..., :2], axis=-1) /
                                      (np.linalg.norm(-position_difference, axis=-1) *
                                       np.linalg.norm(teammate_velocity[..., :2], axis=-1))))[..., np.newaxis]

  is_following_teammate = is_alive.copy()
  is_following_teammate &= teammate_is_alive
  is_following_teammate &= player_from_teammate_xy_distance <= teammate_radius
  is_following_teammate &= player_angle < following_angle
  is_following_teammate &= teammate_angle > leading_angle
  is_following_teammate &= (
      local_behavior_length(is_following_teammate, -2)
      >= min_behavior_length)

  return is_following_teammate


def escort(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    teammate_radius=1):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Teammate alive
  * Near teammate
  * Teammate has flag
  * For a sustained period of time
  
  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    teammate_radius: Maximum distance from teammate

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_is_alive = wrap_file[
      "map/matchup/repeat/player/time/teammate_is_alive"][
          map_id, matchup_id, repeat_id, player_id]

  own_position = wrap_file[
      "map/matchup/repeat/player/time/position"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)
  teammate_position = wrap_file[
      "map/matchup/repeat/player/time/teammate_position"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)
  opponent_flag_position = wrap_file[
      "map/matchup/repeat/player/time/opponent_flag_position"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)
  opponent_flag_status = wrap_file[
      "map/matchup/repeat/player/time/opponent_flag_status"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  teammate_has_opponent_flag = (np.sum((own_position - opponent_flag_position) ** 2, axis=-1) >
                                np.sum((teammate_position - opponent_flag_position) ** 2, axis=-1))[..., np.newaxis]
  
  teammate_has_opponent_flag &= (opponent_flag_status == 1)

  player_from_teammate_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_teammate_xy_distance"][
          map_id, matchup_id, repeat_id][
      ..., player_id, :, :].astype(np.float32)

  is_escort = is_alive.copy()
  is_escort &= teammate_is_alive
  is_escort &= player_from_teammate_xy_distance <= teammate_radius
  is_escort &= teammate_has_opponent_flag
  is_escort &= (
      local_behavior_length(is_escort, -2)
      >= min_behavior_length)

  return is_escort


def assist(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    min_approach=0.0, own_flag_radius=4):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Teammate alive
  * Teammate has opponent flag
  * Own flag is stray or opponent has flag
  * Approaching own flag
  * Within radius of own flag
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    min_approach: Minimum approach speed (positive float)
    own_flag_radius: Maximum distance from own flag (positive float)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_is_alive = wrap_file[
      "map/matchup/repeat/player/time/teammate_is_alive"][
          map_id, matchup_id, repeat_id, player_id]

  own_position = wrap_file[
      "map/matchup/repeat/player/time/position"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)
  teammate_position = wrap_file[
      "map/matchup/repeat/player/time/teammate_position"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)
  opponent_flag_position = wrap_file[
      "map/matchup/repeat/player/time/opponent_flag_position"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)
  opponent_flag_status = wrap_file[
      "map/matchup/repeat/player/time/opponent_flag_status"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  teammate_has_opponent_flag = (np.sum((own_position - opponent_flag_position) ** 2, axis=-1) >
                                np.sum((teammate_position - opponent_flag_position) ** 2, axis=-1))[..., np.newaxis]
  
  teammate_has_opponent_flag &= (opponent_flag_status == 1)
    
  own_flag_is_away = wrap_file[
      "map/matchup/repeat/player/time/own_flag_status"][
          map_id, matchup_id, repeat_id][..., player_id, :, :] > 0
  approach_own_flag = wrap_file[
      "map/matchup/repeat/player/time/player_from_own_flag_xy_approach"][
          map_id, matchup_id, repeat_id][
      ..., player_id, :, :].astype(np.float32)
    
  player_from_own_flag_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_own_flag_xy_distance"][
          map_id, matchup_id, repeat_id][
      ..., player_id, :, :].astype(np.float32)

  is_assist = is_alive.copy()
  is_assist &= teammate_is_alive
  is_assist &= teammate_has_opponent_flag
  is_assist &= own_flag_is_away
  is_assist &= approach_own_flag <= min_approach
  is_assist &= player_from_own_flag_xy_distance <= own_flag_radius
  is_assist &= (
      local_behavior_length(is_assist, -2)
      >= min_behavior_length)

  return is_assist


def hold_fort(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    base_radius=1):
    
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Teammate alive
  * Near own base
  * Teammate near opponent base
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or ids
    min_behavior_length: Minimum behavior length (integer)
    base_radius: Maximum distance from base

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  player_from_own_base_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_own_base_xy_distance"][
          map_id, matchup_id, repeat_id, player_id]

  # Not sure how to assess teammate distance from without loop
  # BREAKS IF WE SLICE MATCHUPS!!! :(
  teammate_ids = {0: 1, 1: 0, 2: 3, 3: 2}
  if type(player_id) == slice:
      assert player_from_own_base_xy_distance.shape[-3] == 4
      teammate_from_opponent_base_xy_distance = np.zeros(
          player_from_own_base_xy_distance.shape)
      for own_id in np.arange(4):
          teammate_from_opponent_base_xy_distance[..., own_id, :, :] = \
          wrap_file['map/matchup/repeat/player/time/'
                    'player_from_opponent_base_xy_distance'][
              map_id, matchup_id, repeat_id][
              ..., teammate_ids[own_id], :, :]
  elif type(player_id) == int:
      teammate_from_opponent_base_xy_distance = wrap_file[
          'map/matchup/repeat/player/time/player_from_opponent_base_xy_distance'][
          map_id, matchup_id, repeat_id][
          ..., teammate_ids[player_id], :, :]
  else:
      raise ValueError("Unrecognized player_id")

  is_holding_fort = is_alive.copy()
  is_holding_fort &= player_from_own_base_xy_distance <= base_radius
  is_holding_fort &= teammate_from_opponent_base_xy_distance <= base_radius
  is_holding_fort &= (
      local_behavior_length(is_holding_fort, -2) >= min_behavior_length)

  return is_holding_fort


def siege(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    base_radius=1):
    
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Teammate alive
  * Near opponent base
  * Teammate near opponent base
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or ids
    min_behavior_length: Minimum behavior length (integer)
    base_radius: Maximum distance from base

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  player_from_opponent_base_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_opponent_base_xy_distance"][
          map_id, matchup_id, repeat_id, player_id]

  # Not sure how to assess teammate distance from without loop
  # BREAKS IF WE SLICE MATCHUPS!!! :(
  teammate_ids = {0: 1, 1: 0, 2: 3, 3: 2}
  if type(player_id) == slice:
      assert player_from_opponent_base_xy_distance.shape[-3] == 4
      teammate_from_opponent_base_xy_distance = np.zeros(
          player_from_opponent_base_xy_distance.shape)
      for own_id in np.arange(4):
          teammate_from_opponent_base_xy_distance[..., own_id, :, :] = \
          wrap_file['map/matchup/repeat/player/time/'
                    'player_from_opponent_base_xy_distance'][
              map_id, matchup_id, repeat_id][
              ..., teammate_ids[own_id], :, :]
  elif type(player_id) == int:
      teammate_from_opponent_base_xy_distance = wrap_file[
          'map/matchup/repeat/player/time/player_from_opponent_base_xy_distance'][
          map_id, matchup_id, repeat_id][
          ..., teammate_ids[player_id], :, :]
  else:
      raise ValueError("Unrecognized player_id")

  is_siege = is_alive.copy()
  is_siege &= player_from_opponent_base_xy_distance <= base_radius
  is_siege &= teammate_from_opponent_base_xy_distance <= base_radius
  is_siege &= (
      local_behavior_length(is_siege, -2) >= min_behavior_length)

  return is_siege


def redoubt(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    base_radius=1):
    
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Teammate alive
  * Near own base
  * Teammate near own base
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or ids
    min_behavior_length: Minimum behavior length (integer)
    base_radius: Maximum distance from base

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  player_from_own_base_xy_distance = wrap_file[
      "map/matchup/repeat/player/time/player_from_own_base_xy_distance"][
          map_id, matchup_id, repeat_id, player_id]

  # Not sure how to assess teammate distance from without loop
  # BREAKS IF WE SLICE MATCHUPS!!! :(
  teammate_ids = {0: 1, 1: 0, 2: 3, 3: 2}
  if type(player_id) == slice:
      assert player_from_own_base_xy_distance.shape[-3] == 4
      teammate_from_own_base_xy_distance = np.zeros(
          player_from_own_base_xy_distance.shape)
      for own_id in np.arange(4):
          teammate_from_own_base_xy_distance[..., own_id, :, :] = \
          wrap_file['map/matchup/repeat/player/time/'
                    'player_from_own_base_xy_distance'][
              map_id, matchup_id, repeat_id][
              ..., teammate_ids[own_id], :, :]
  elif type(player_id) == int:
      teammate_from_own_base_xy_distance = wrap_file[
          'map/matchup/repeat/player/time/player_from_own_base_xy_distance'][
          map_id, matchup_id, repeat_id][
          ..., teammate_ids[player_id], :, :]
  else:
      raise ValueError("Unrecognized player_id")

  is_redoubt = is_alive.copy()
  is_redoubt &= player_from_own_base_xy_distance <= base_radius
  is_redoubt &= teammate_from_own_base_xy_distance <= base_radius
  is_redoubt &= (
      local_behavior_length(is_redoubt, -2) >= min_behavior_length)

  return is_redoubt


# Visibility-based behaviors
### Field of view:
#map/matchup/repeat/player/time/player_visible
#map/matchup/repeat/player/time/entity_visible
# four players, four entities (red flag, blue flag, red base, blue base)
# if zero (not visible), if between zero and one (it's occluded),
# if it's greater than one (it's in field of view)


def watching_teammate(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    visibility_threshold=.5):
  """Returns boolean where behavior is occurring.
  
  This behavior requires:
  
  * Alive
  * Teammate alive
  * Teammate is visible (above visibility threshold)
  * For a sustained period of time
  
  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    visibility_threshold: Minimum proportion of occlusion (float)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_is_alive = wrap_file[
      "map/matchup/repeat/player/time/teammate_is_alive"][
          map_id, matchup_id, repeat_id, player_id]

  is_watching_teammate = is_alive.copy()
  is_watching_teammate &= teammate_is_alive
  ####
  is_watching_teammate &= (
      local_behavior_length(is_watching_teammate, -2)
      >= min_behavior_length)

  return is_watching_teammate


# Sustained action based behaviors
def running_forwards(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Running forwards
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  move_action = wrap_file[  # 0 is forward, 1 is still, 2 is backward
      "map/matchup/repeat/player/time/action"][
          map_id, matchup_id, repeat_id, player_id, ..., 3:4]

  is_moving_forward = is_alive.copy()
  is_moving_forward &= move_action == 0
  is_moving_forward &= (
      local_behavior_length(is_moving_forward, -2)
      >= min_behavior_length)

  return is_moving_forward


def running_backwards(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Running backwards
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  move_action = wrap_file[  # 0 is forward, 1 is still, 2 is backward
      "map/matchup/repeat/player/time/action"][
          map_id, matchup_id, repeat_id, player_id, ..., 3:4]

  is_moving_backward = is_alive.copy()
  is_moving_backward &= move_action == 2
  is_moving_backward &= (
      local_behavior_length(is_moving_backward, -2)
      >= min_behavior_length)

  return is_moving_backward


def turning_left(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Turning left
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  move_action = wrap_file[
      "map/matchup/repeat/player/time/action"][
          map_id, matchup_id, repeat_id, player_id, ..., 0:1]

  is_turning_left = is_alive.copy()
  is_turning_left &= move_action <= 1  # 0 or 1
  is_turning_left &= (
      local_behavior_length(is_turning_left, -2)
      >= min_behavior_length)

  return is_turning_left


def turning_right(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Turning right
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  move_action = wrap_file[
      "map/matchup/repeat/player/time/action"][
          map_id, matchup_id, repeat_id, player_id, ..., 0:1]

  is_turning_right = is_alive.copy()
  is_turning_right &= move_action >= 3  # 3 or 4
  is_turning_right &= (
      local_behavior_length(is_turning_right, -2)
      >= min_behavior_length)

  return is_turning_right


def strafing_left(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Strafing left
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  move_action = wrap_file[
      "map/matchup/repeat/player/time/action"][
          map_id, matchup_id, repeat_id, player_id, ..., 2:3].astype(np.float32)

  is_strafing_left = is_alive.copy()
  is_strafing_left &= move_action == 0
  is_strafing_left &= (
      local_behavior_length(is_strafing_left, -2)
      >= min_behavior_length)

  return is_strafing_left


def strafing_right(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Strafing right
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  move_action = wrap_file[  # 0 is backward, 1 is still, 2 is forward
      "map/matchup/repeat/player/time/action"][
          map_id, matchup_id, repeat_id, player_id, ..., 2:3].astype(np.float32)

  is_strafing_right = is_alive.copy()
  is_strafing_right &= move_action == 2
  is_strafing_right &= (
      local_behavior_length(is_strafing_right, -2)
      >= min_behavior_length)

  return is_strafing_right


def strafing(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Strafing left or right
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  move_action = wrap_file[
      "map/matchup/repeat/player/time/action"][
          map_id, matchup_id, repeat_id, player_id, ..., 2:3].astype(np.float32)

  is_strafing = is_alive.copy()
  is_strafing &= move_action != 1
  is_strafing &= (
      local_behavior_length(is_strafing, -2)
      >= min_behavior_length)

  return is_strafing


# Approach based behaviors
def approaching_own_base(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    min_approach=2.0):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Approaching own base
  * For a sustained period of time

  Use this distribution to help determine threshold:
  ```
  xy_approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_own_base_xy_approach"][...]
  plot.hist(xy_approach, bins=50)
  ```

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    min_approach: Minimum approach speed (positive float)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_own_base_xy_approach"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  is_approaching = is_alive.copy()
  is_approaching &= approach >= min_approach
  is_approaching &= (
      local_behavior_length(is_approaching, -2)
      >= min_behavior_length)

  return is_approaching


def approaching_opponent_base(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    min_approach=2.0):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Approaching opponent base
  * For a sustained period of time

  Use this distribution to help determine threshold:
  ```
  xy_approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_opponent_base_xy_approach"][...]
  plot.hist(xy_approach, bins=50)
  ```

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    min_approach: Minimum approach speed (positive float)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_opponent_base_xy_approach"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  is_approaching = is_alive.copy()
  is_approaching &= approach >= min_approach
  is_approaching &= (
      local_behavior_length(is_approaching, -2)
      >= min_behavior_length)

  return is_approaching


def approaching_own_flag(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    min_approach=2.0):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Approaching own flag
  * For a sustained period of time

  Use this distribution to help determine threshold:
  ```
  xy_approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_own_flag_xy_approach"][...]
  plot.hist(xy_approach, bins=50)
  ```

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    min_approach: Minimum approach speed (positive float)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_own_flag_xy_approach"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  is_approaching = is_alive.copy()
  is_approaching &= approach >= min_approach
  is_approaching &= (
      local_behavior_length(is_approaching, -2)
      >= min_behavior_length)

  return is_approaching


def approaching_opponent_flag(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    min_approach=2.0):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Approaching opponent flag
  * For a sustained period of time

  Use this distribution to help determine threshold:
  ```
  xy_approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_opponent_flag_xy_approach"][...]
  plot.hist(xy_approach, bins=50)
  ```

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    min_approach: Minimum approach speed (positive float)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_opponent_flag_xy_approach"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  is_approaching = is_alive.copy()
  is_approaching &= approach >= min_approach
  is_approaching &= (
      local_behavior_length(is_approaching, -2)
      >= min_behavior_length)

  return is_approaching


def approaching_teammate(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None), min_behavior_length=DEFAULT_MIN_BEHAVIOR_LENGTH,
    min_approach=2.0):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * Teammate alive
  * Approaching teammate
  * For a sustained period of time

  Use this distribution to help determine threshold:
  ```
  xy_approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_teammate_xy_approach"][...]
  plot.hist(xy_approach, bins=50)
  ```

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs
    min_behavior_length: Minimum behavior length (integer)
    min_approach: Minimum approach speed (positive float)

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_is_alive = wrap_file[
      "map/matchup/repeat/player/time/teammate_is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  approach = wrap_file[
      "map/matchup/repeat/player/time/player_from_teammate_xy_approach"][
          map_id, matchup_id, repeat_id, player_id].astype(np.float32)

  is_approaching = is_alive.copy()
  is_approaching &= teammate_is_alive
  is_approaching &= approach >= min_approach
  is_approaching &= (
      local_behavior_length(is_approaching, -2)
      >= min_behavior_length)

  return is_approaching


# Boosting behaviors
def boosted(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None)):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * High speed (only possible from boosting)
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  speed = wrap_file[
      "map/matchup/repeat/player/time/speed"][
          map_id, matchup_id, repeat_id, player_id]

  is_boosted = is_alive.copy()
  is_boosted &= speed >= 6.0
  is_boosted &= (
      local_behavior_length(is_boosted, -2) >= 2)
  is_boosted = pad_behavior(is_boosted, -2)

  return is_boosted


def boosted_teammate(
    wrap_file,
    map_id=slice(None), matchup_id=slice(None), repeat_id=slice(None),
    player_id=slice(None)):
  """Returns boolean where behavior is occurring.

  This behavior requires:

  * Alive
  * High speed (only possible from boosting)
  * For a sustained period of time

  Args:
    wrap_file: The wrapped dataset file
    map_id: The map ID or slice of IDs
    matchup_id: The matchup ID or slice of IDs
    repeat_id: The repeat ID or slice of IDs
    player_id: The player ID or slice or IDs

  Returns:
    Boolean array.
  """
  is_alive = wrap_file[
      "map/matchup/repeat/player/time/is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_is_alive = wrap_file[
      "map/matchup/repeat/player/time/teammate_is_alive"][
          map_id, matchup_id, repeat_id, player_id]
  teammate_speed = wrap_file[
      "map/matchup/repeat/player/time/teammate_speed"][
          map_id, matchup_id, repeat_id, player_id]

  is_boosted = is_alive.copy()
  is_boosted &= teammate_is_alive
  is_boosted &= teammate_speed >= 6.0
  is_boosted &= (
      local_behavior_length(is_boosted, -2) >= 2)
  is_boosted = pad_behavior(is_boosted, -2)

  # TODO(marris): Need some way of determining that the boost was freindly.

  return is_boosted




