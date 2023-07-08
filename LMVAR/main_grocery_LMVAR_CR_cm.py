import os

import LMVAR_dataprocessing as DP
import matplotlib.pyplot as plt
import LMVAR as LMAR_model
import pandas as pd
import numpy as np
import preprocess_MIMO as PR
import pickle
import utils
import pdb

dp = DP.preprocessing()

path_dir = './data/'
result_path = './result_missing_grocery/'
missing_rate = 50
missing_dir = 'missing' + str(missing_rate) + '/'
file_dir = 'missing_warped_' + str(missing_rate) + '/'

day_measurement = 96 * 3
num_class = 3
sequence_length1 = 1
sequence_length2 = 1

utils.make_dir(result_path)
utils.make_dir(result_path + missing_dir)

filelist = os.listdir(path_dir + missing_dir)

wd = path_dir + '492.csv'
df = pd.read_csv(wd).iloc[286:, :]
gt = df['value'].to_numpy()

day_gt = np.reshape(gt, [-1, day_measurement])
print(11)
for ith, fn in enumerate(filelist):
    num_pickle = fn.split('.')[0]
    file_wdata_train = [path_dir + file_dir + 'train/' + num_pickle + '_' + str(_) + 'train_warped.csv' for _ in range(1, 5) ]
    file_wdata_test = [path_dir + file_dir + 'test/' + num_pickle + '_' + str(_) + 'test_warped.csv' for _ in range(1, 5) ]

    load_warped_train = []
    load_warped_test = []
    for fw_train, fw_test in zip(file_wdata_train, file_wdata_test):
        load_warped_train.append(dp.read_original_data_v2(fw_train))
        load_warped_test.append(dp.read_original_data_v2(fw_test))


    file_name = path_dir + missing_dir + fn
    file_result = result_path + missing_dir + fn
    with open(file_name, 'rb') as f:
        data = pickle.load(f)

    # load = data['imputed_data']
    load = data['load']
    # print(data['ground_truth_idx']) # True = nan
    ground = ~data['ground_truth_idx']

    day_load = np.reshape(load, [-1, day_measurement])
    day_ground = np.reshape(ground, [-1, day_measurement])

    total_day = len(load) // day_measurement
    fold_size = total_day // 8

    N_com = [2, 3, 4, 5, 6]
    N_com_mix = [2, 3]
    result = {'real': [],
              'pred': [],
              'ground_truth_idx': [],
              'n_com': [],
              'n_com_mix': []}

    nested_train_len = [int(len(day_load) / 8) * _ for _ in range(4, 8)]
    nested_test_len = [int(len(day_load) / 8) * _ for _ in range(5, 9)]

    for i, nth_test in enumerate([0, 1, 2, 3]):

        train_load, train_target = day_load[:(nested_train_len[nth_test] - 1)], \
                                   day_load[1:nested_train_len[nth_test]]
        train_load_warped, train_warped_target = load_warped_train[i][:-1], load_warped_train[i][1:]
        test_load, test_target = day_gt[nested_train_len[nth_test]:(nested_test_len[nth_test] - 1)], \
                                 day_gt[(nested_train_len[nth_test] + 1):nested_test_len[nth_test]]
        test_load_warped = load_warped_test[i][:fold_size-1]
        train_load_idx, train_target_idx = day_ground[:(nested_train_len[nth_test] - 1)], \
                                           day_ground[1:nested_train_len[nth_test]]
        test_load_idx, test_target_idx = day_ground[nested_train_len[nth_test]:(nested_test_len[nth_test] - 1)], \
                                         day_ground[(nested_train_len[nth_test] + 1):nested_test_len[nth_test]]
        train_idx = train_load_idx.all(axis=1) & train_target_idx.all(axis=1)
        test_idx = test_load_idx.all(axis=1) & test_target_idx.all(axis=1)

        train_load, train_target = train_load[train_idx], train_target[train_idx]
        train_load_warped = train_load_warped[train_idx]
        # test_load, test_target = test_load[test_idx], test_target[test_idx]
        # test_load_warped = test_load_warped[test_idx]
        # print(train_load.shape, train_target.shape, train_load_warped.shape)

        validation_loss = []
        parameter = []
        for n_com in N_com:
            for n_com_mix in N_com_mix:
                pca_data, pca = dp.make_pca_data(np.r_[train_load, np.expand_dims(train_target[-1], 0)], n_com)
                pca_wdata, pca_w = dp.make_pca_data(train_load_warped, n_com_mix)

                if n_com < n_com_mix:
                    continue

                # train dataset
                train_ar_pca = np.expand_dims(pca_data[:-1], 1)
                train_mix_pca = np.expand_dims(pca_wdata, 1)
                train_target_ar_pca = pca_data[1:]
                train_target_mix_pca = pca_wdata
                train_target_raw = train_target

                train_len = int(len(train_ar_pca) * 0.8)
                train_ar_pca, val_ar_pca = train_ar_pca[:train_len], train_ar_pca[train_len:]
                train_target_ar_pca, val_target_ar_pca = train_target_ar_pca[:train_len], train_target_ar_pca[train_len:]
                train_target_mix_pca, val_target_mix_pca = train_target_mix_pca[:train_len], train_target_mix_pca[train_len:]
                train_target_raw, val_target_raw = train_target_raw[:train_len], train_target_raw[train_len:]
                train_mix_pca, val_mix_pca = train_mix_pca[:train_len], train_mix_pca[train_len:]

                # model = LMAR_model.LMAR(num_class, n_com, n_com_mix, [], sequence_length1, sequence_length2, pca, [])
                model = LMAR_model.LMAR(num_class, n_com, n_com_mix, sequence_length1, sequence_length2, pca, cluster = "AgglomerativeClustering")
                model.fit(train_ar_pca, train_mix_pca, train_target_ar_pca, train_target_mix_pca, iteration=40)
                predY, alpha = model.demand_pca_predict(val_ar_pca, val_mix_pca)
                predY = model.to_warpeddata(predY)
                val_loss = model.RMSE(val_target_raw, predY)

                validation_loss.append(val_loss)
                parameter.append([n_com, n_com_mix])
                print(nth_test, n_com, n_com_mix, val_loss)

        idx_model = np.argmin(validation_loss)
        n_com = parameter[idx_model][0]
        n_com_mix = parameter[idx_model][1]
        pca_data, pca = dp.make_pca_data(np.r_[train_load, np.expand_dims(train_target[-1], 0)], n_com)
        pca_wdata, pca_w = dp.make_pca_data(train_load_warped, n_com_mix)

        # train dataset
        train_ar_pca = np.expand_dims(pca_data[:-1], 1)
        train_mix_pca = np.expand_dims(pca_wdata, 1)
        train_target_ar_pca = pca_data[1:]
        train_target_mix_pca = pca_wdata
        train_target_raw = train_target

        # test dataset
        test_pca_data = pca.transform(test_load)
        test_pca_wdata = pca_w.transform(test_load_warped)
        test_ar_pca = np.expand_dims(test_pca_data, 1)
        test_mix_pca = np.expand_dims(test_pca_wdata, 1)
        test_target_raw = test_target

        # model = LMAR_model.LMAR(num_class, n_com, n_com_mix, [], sequence_length1, sequence_length2, pca, [])
        model = LMAR_model.LMAR(num_class, n_com, n_com_mix, sequence_length1, sequence_length2, pca, cluster = "AgglomerativeClustering")
        model.fit(train_ar_pca, train_mix_pca, train_target_ar_pca, train_target_mix_pca, iteration=40)
        predY, alpha = model.demand_pca_predict(test_ar_pca, test_mix_pca)
        predY = model.to_warpeddata(predY)
        print(predY.shape, test_target_raw.shape)
        result['pred'].append(predY)
        result['real'].append(test_target_raw)
        result['n_com'].append(n_com)
        result['n_com_mix'].append(n_com_mix)
    #
    #     test_target_ground = np.reshape(test_target_ground, [-1])
    #     predY = np.reshape(predY, [-1])
    #     test_target_raw = np.reshape(test_target_raw, [-1])
    #     _loss = model.RMSE(test_target_raw[test_target_ground], predY[test_target_ground])
        _loss = model.RMSE(test_target_raw, predY)

        print("Final test:", nth_test, "n_com:", n_com, "n_com_mix:", n_com_mix, "rmse:", _loss)
    #
    with open(file_result, 'wb') as w:
        pickle.dump(result, w, protocol=3)