import numpy as np
import time

############################################
############################################


def sample_trajectory(env, policy, max_path_length, render=False, render_mode=('rgb_array',)):

    # initialize env for the beginning of a new rollout
    observations = [env.reset().astype(np.float32)] # HINT: should be the output of resetting the env

    # init vars
    actions, rewards, terminals, image_obs = [], [], [], []
    rollout_done = 0
    while rollout_done == 0:
        # render image of the simulated env
        if render:
            if 'rgb_array' in render_mode:
                if hasattr(env, 'sim'):
                    image_obs.append(env.sim.render(camera_name='track', height=500, width=500)[::-1])
                else:
                    image_obs.append(env.render(mode=render_mode))
            if 'human' in render_mode:
                env.render(mode=render_mode)
                time.sleep(env.model.opt.timestep)

        # use the most recent ob to decide what to do
        observation = observations[-1]
        action = policy.get_action(observation).numpy()[0] # HINT: query the policy's get_action function
        actions.append(action)

        # take that action and record results
        observation, reward, env_done, _ = env.step(action)
        observation = observation.astype(np.float32)

        # record result of taking that action
        observations.append(observation)
        rewards.append(reward)

        # TODO end the rollout if the rollout ended 
        # HINT: rollout can end due to done, or due to max_path_length
        rollout_done = int(env_done or (len(observations) > max_path_length)) # HINT: this is either 0 or 1
        terminals.append(rollout_done)

    return Path(observations[:-1], image_obs, actions, rewards, observations[1:], terminals)


def sample_trajectories(
        env, policy, min_timesteps_per_batch, max_path_length, render=False, render_mode=('rgb_array',)):
    """
        Collect rollouts until we have collected min_timesteps_per_batch steps.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
        Hint2: use get_pathlength to count the timesteps collected in each path
    """
    timesteps_this_batch = 0
    paths = []
    path = None
    while timesteps_this_batch < min_timesteps_per_batch:
        path = sample_trajectory(env, policy, max_path_length, render=render, render_mode=render_mode)
        paths.append(path)
        timesteps_this_batch += get_pathlength(path)
    return paths, timesteps_this_batch


def sample_n_trajectories(env, policy, ntraj, max_path_length, render=False, render_mode=('rgb_array',)):
    """
        Collect ntraj rollouts.

        TODO implement this function
        Hint1: use sample_trajectory to get each path (i.e. rollout) that goes into paths
    """
    return [sample_trajectory(env, policy, max_path_length, render=render, render_mode=render_mode)
            for _ in range(ntraj)]

############################################
############################################


def Path(obs, image_obs, acs, rewards, next_obs, terminals):
    """
        Take info (separate arrays) from a single rollout
        and return it in a single dictionary
    """
    if image_obs != []:
        image_obs = np.stack(image_obs, axis=0)
    return {"observation" : np.array(obs, dtype=np.float32),
            "image_obs" : np.array(image_obs, dtype=np.uint8),
            "reward" : np.array(rewards, dtype=np.float32),
            "action" : np.array(acs, dtype=np.float32),
            "next_observation": np.array(next_obs, dtype=np.float32),
            "terminal": np.array(terminals, dtype=np.float32)}


def convert_listofrollouts(paths):
    """
        Take a list of rollout dictionaries
        and return separate arrays,
        where each array is a concatenation of that array from across the rollouts
    """
    observations = np.concatenate([path["observation"] for path in paths])
    actions = np.concatenate([path["action"] for path in paths])
    next_observations = np.concatenate([path["next_observation"] for path in paths])
    terminals = np.concatenate([path["terminal"] for path in paths])
    concatenated_rewards = np.concatenate([path["reward"] for path in paths])
    unconcatenated_rewards = [path["reward"] for path in paths]
    return observations, actions, next_observations, terminals, concatenated_rewards, unconcatenated_rewards

############################################
############################################


def get_pathlength(path):
    return len(path["reward"])
