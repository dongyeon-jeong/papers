from imputation.MPCA import MPCA
import numpy as np
import copy


class imputation(MPCA):

    def __init__(self, n_component, n_component_pca, day_measurement, PCA = None):
        '''
        :param n_component:
        :param n_component_pca:
        '''
        super(imputation, self).__init__(n_component, n_component_pca)
        self.n_component = n_component
        self.n_component_pca = n_component_pca
        self.day_measurement = day_measurement
        self.pca = PCA

    def imputation(self, data):
        '''
        - Main
        1. Process data.
        2. Find cluster with curveclustering module.
        3. Impute the missing parts of the curves.
        :param Y:
        :return:
        '''

        # 1. Process data.
        day_curve = []
        self.len_data = len(data)
        for i in range(self.len_data // self.day_measurement):
            idx1 = i * self.day_measurement
            idx2 = (i + 1) * self.day_measurement
            day_curve.append(data[idx1:idx2])
        day_curve = np.array(day_curve)

        # 2. Find cluster.
        Y_hat = self.fit_cluster(day_curve)

        # 3. Impute the missing parts of the curves.
        imputed_data = self.inference_curve(day_curve, Y_hat)
        imputed_data = np.reshape(imputed_data, [-1])

        return imputed_data


    def fit_cluster(self, Y):
        if self.pca is None:
            self.fit(Y)
            return self.predict_self(Y)
        else:
            self.pca.fit(Y)
            return self.pca.predict_self(Y)

    def inference_curve(self, y_o, y_hat):

        check_nan = np.isnan(y_o)
        check_nonnan = np.invert(check_nan)
        for i, (ck, ckn) in enumerate(zip(check_nan, check_nonnan)):
            if ckn.any():
                y_hat[i, ckn] = y_o[i, ckn]
        return y_hat













