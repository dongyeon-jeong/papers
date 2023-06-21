from imputation.FA import FA
import copy
import numpy as np
import matplotlib.pyplot as plt

class imputation(FA):

    def __init__(self, n_component, day_measurement):
        super(imputation, self).__init__(n_component)
        self.day_measurement = day_measurement

    def imputation(self, data):

        # 1. Process data.
        day_curve = []
        self.len_data = len(data)
        for i in range(self.len_data // self.day_measurement):
            idx1 = i * self.day_measurement
            idx2 = (i + 1) * self.day_measurement
            day_curve.append(data[idx1:idx2])
        day_curve = np.array(day_curve)
        self.num_curve_total = len(day_curve)

        # 2. Fit PPCA
        self.fit(day_curve)

        # 3. Impute data
        imputed_data = self.inference_curve(day_curve)
        imputed_data = np.reshape(imputed_data, [-1])
        return imputed_data

    def inference_curve(self, y_o):

        y_hat = np.matmul(self.x, self.W.transpose()) + self.mu
        check_nan = np.isnan(y_o)
        check_nonnan = np.invert(check_nan)

        for i, (ck, ckn) in enumerate(zip(check_nan, check_nonnan)):
            if ckn.any():
                y_hat[i, ckn] = y_o[i, ckn]
        return y_hat

























