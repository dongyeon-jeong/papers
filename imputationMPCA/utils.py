import os
import numpy as np
import copy


def missing_generation_gamma(data, missing_rate, shape, scale):

    occur_num = int(len(data) * (missing_rate + 0.1))
    existing_missing_rate = np.mean(np.isnan(data))

    shape_obs = (shape) * (1 / missing_rate + existing_missing_rate - 1)

    obs = np.random.gamma(shape_obs, scale, occur_num).astype(int)
    mis = np.random.gamma(shape, scale, occur_num).astype(int)

    idx = 0
    obs_idx = np.zeros(len(data), dtype=bool)
    for o, m in zip(obs, mis):
        idx_obs = idx + o
        obs_idx[idx:idx_obs] = True
        idx = idx_obs + m
    data_obs = copy.deepcopy(data)
    data_obs[np.invert(obs_idx)] = np.nan
    data_miss = copy.deepcopy(data)
    data_miss[obs_idx] = np.nan

    miss_idx = np.isnan(data_obs)
    # print(np.mean(np.isnan(data)), np.mean(miss_idx), np.mean(np.invert(miss_idx)))

    return data_obs, np.invert(miss_idx)

def make_dir(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

def make_validation(filename, data, val_rate = 0.1):

    if '2h' in filename:
        shape = 4 * 2
        scale = 1 / 2
        missing_idx = np.isnan(data)
        val_data, obs_idx = missing_generation_gamma(data, missing_rate=val_rate, shape=shape, scale=scale)
        val_idx = np.logical_xor(np.invert(obs_idx), missing_idx)
        return val_data, val_idx

    elif '8h' in filename:
        shape = 4 * 8 * 2
        scale = 1 / 2
        missing_idx = np.isnan(data)
        val_data, obs_idx = missing_generation_gamma(data, missing_rate=val_rate, shape=shape, scale=scale)
        val_idx = np.logical_xor(np.invert(obs_idx), missing_idx)
        return val_data, val_idx

    elif '1d' in filename:
        shape = 4 * 24 * 3
        scale = 1 / 3
        missing_idx = np.isnan(data)
        val_data, obs_idx = missing_generation_gamma(data, missing_rate=val_rate, shape=shape, scale=scale)
        val_idx = np.logical_xor(np.invert(obs_idx), missing_idx)
        return val_data, val_idx

    else:
        val_data = copy.deepcopy(data)
        val_idx = np.random.permutation(np.where(~np.isnan(data))[0])[:int(len(data) / 10)]
        val_data[val_idx] = np.nan
        return val_data, val_idx
