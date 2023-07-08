import numpy as np
import csv
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class preprocessing:

    def __init__(self):
        a = 0

    def read_original_data(self, filename):
        power_data = []
        f = open(filename)
        lines = csv.reader(f)
        for line in lines:
            power_data.append([float(_) for _ in line])
        f.close()
        power_data = np.array(power_data)[:, 1:].transpose()
        # print(power_data, power_data.shape)
        return power_data

    def read_original_data_v2(self, filename):
        power_data = []
        f = open(filename)
        lines = csv.reader(f)
        for line in lines:
            power_data.append([float(_) for _ in line])
        f.close()
        power_data = np.array(power_data)
        # print(power_data, power_data.shape)
        return power_data

    def read_warped_data(self, filename):

        power_data = []
        for f_name in filename:
            f = open(f_name)
            lines = csv.reader(f)
            for line in lines:
                power_data.append([float(_) for _ in line])
            f.close()
        power_data = np.array(power_data)
        # print(power_data, power_data.shape)
        return power_data

    def read_label(self, filename):
        label = []
        for f_name in filename:
            f = open(f_name)
            lines = csv.reader(f)
            for line in lines:
                label.append(float(line[0]))
            f.close()
        label = np.array(label)
        return label

    def read_day(self, filename):
        days = []
        f = open(filename)
        lines = csv.reader(f)
        for line in lines:
            days.append(float(line[0]))
        f.close()

        days = np.array(days)

        cum_days = days
        flag = 0
        for i, day in enumerate(days):
            if flag == 0:
                if day == 365:
                    flag= 1
            else:
                cum_days[i] = cum_days[i] + 365

        return days, cum_days

    def read_wday(self, filename):
        days = []
        f = open(filename)
        lines = csv.reader(f)
        for line in lines:
            days.append(float(line[0]))
        f.close()
        days = np.array(days)

        return days

    def read_warping_fcn_data(self, filename):

        power_wfn = []
        for f_name in filename:
            f = open(f_name)
            lines = csv.reader(f)
            for line in lines:
                power_wfn.append([float(_) for _ in line])
            f.close()
        power_wfn = np.array(power_wfn)
        return power_wfn

    def make_pca_data(self, wdata, n_com):
        pca = PCA(n_components=n_com)
        pca_data = pca.fit_transform(wdata)

        # plt.scatter(pca_data[:, 0], pca_data[:, 1])
        # plt.show()
        return pca_data, pca

    def make_pca_wdata(self, wdata, n_com):
        from sklearn.decomposition import KernelPCA
        pca = KernelPCA(n_components=n_com, kernel="poly", degree=2, fit_inverse_transform=True)
        pca_data = pca.fit_transform(wdata)

        return pca_data, pca

    def make_sequence_data(self, data, cum_days, sequence_length):
        seq_data = []
        pre_day = []
        for i in range(len(data) - sequence_length):
            end = i + sequence_length
            if (cum_days[end - 1] - cum_days[i]) == (sequence_length - 1):
                # print(data[i:end])
                seq_data.append(data[i:end])
                pre_day.append(cum_days[end])

        seq_data = np.array(seq_data)
        pre_day = np.array(pre_day)
        return seq_data, pre_day

    def make_target_data(self, data, cum_days, sequence_length):
        target_data = []
        for i in range(len(data) - sequence_length):
            end = i + sequence_length
            if (cum_days[end - 1] - cum_days[i]) == (sequence_length - 1):
                target_data.append(data[end])
        return np.array(target_data)

    def make_target_wday(self, wdays, cum_days, sequence_length):
        target_wdays = []
        for i in range(len(wdays) - sequence_length):
            end = i + sequence_length
            if (cum_days[end - 1] - cum_days[i]) == (sequence_length - 1):
                target_wdays.append(wdays[end])
        return np.array(target_wdays)



