import numpy as np

def make_trainset(data, nth_fold, number_of_fold = 8):
    fold_size = len(data) // number_of_fold
    train_load = []
    target = []
    for i in range(fold_size * nth_fold):
        end = i + 96
        train_load.append(data[i:end])
        target.append(data[end])

    train_load = np.array(train_load)
    train_target = np.array(target)

    # missing_train_load = np.isnan(train_load).any(axis=1)
    # missing_train_target = np.isnan(train_target)
    #
    # idx = np.invert(missing_train_target | missing_train_load)
    # # print(idx.shape, np.sum(idx), np.sum(missing_train_target), np.sum(missing_train_target))
    #
    # train_load = train_load[idx]
    # train_target = train_target[idx]

    return train_load, train_target

def make_trainset_MIMO(data, nth_fold, day_measurement = 96, number_of_fold = 8):
    fold_size = len(data) // number_of_fold // day_measurement
    train_load = []
    train_target = []
    for i in range(fold_size * nth_fold):
        st = i * day_measurement
        ed = (i + 1) * day_measurement
        st_t = (i + 1) * day_measurement
        ed_t = (i + 2) * day_measurement
        train_load.append(data[st:ed])
        train_target.append(data[st_t:ed_t])
    train_load = np.array(train_load)
    train_target = np.array(train_target)

    # missing_train_load = np.isnan(train_load).any(axis=1)
    # missing_train_target = np.isnan(train_target).any(axis=1)
    #
    # idx = np.invert(missing_train_target | missing_train_load)
    #
    # train_load = train_load[idx]
    # train_target = train_target[idx]

    return train_load, train_target

def make_testset(data, nth_fold, day_measurement = 96, number_of_fold = 8):
    fold_size = len(data) // number_of_fold // day_measurement
    test_load = []
    test_target = []
    for i in range(fold_size * nth_fold, fold_size * (nth_fold + 1)):
        st = i * day_measurement
        ed = (i + 1) * day_measurement
        st_t = (i + 1) * day_measurement
        ed_t = (i + 2) * day_measurement
        test_load.append(data[st:ed])
        test_target.append(data[st_t:ed_t])
    test_load = np.array(test_load)
    test_target = np.array(test_target)

    # missing_train_load = np.isnan(test_load).any(axis=1)
    # missing_train_target = np.isnan(test_target).any(axis=1)

    # idx = np.invert(missing_train_target | missing_train_load)

    # test_load = test_load[idx]
    # test_target = test_target[idx]

    return test_load, test_target

