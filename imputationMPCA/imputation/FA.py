import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import copy

class FA:

    def __init__(self, n_component):

        self.n_com = n_component

    def initialize(self, Y):
        self.length, self.dim = np.shape(Y)
        self.x = np.zeros([len(Y), self.n_com])
        self.xx = np.zeros([len(Y), self.n_com, self.n_com])
        self.mu = np.nanmean(Y, axis=0)
        self.sigma2 = np.diag(np.nanvar(Y, axis=0))

        missing_idx = np.isnan(Y)
        Y_temp_imputed = copy.deepcopy(Y)

        for i, idx_m in enumerate(missing_idx):
            Y_temp_imputed[i, idx_m] = self.mu[idx_m]
        pca = PCA(self.n_com)
        pc_score = pca.fit_transform(Y_temp_imputed)
        self.W = pca.components_.transpose()


    def fit(self, Y, epochs = 50):

        self.initialize(Y)

        pre_lik = -1e+8
        flag = False
        for epoch in range(epochs):
            self.Estep(Y)
            self.Mstep(Y)
            lik = self.loglik(Y)
            if lik > pre_lik:
                pre_lik = lik
                self.x_ = copy.deepcopy(self.x)
                self.xx_ = copy.deepcopy(self.xx)
                self.mu_ = copy.deepcopy(self.mu)
                self.sigma2_ = copy.deepcopy(self.sigma2)
                self.W_ = copy.deepcopy(self.W)
                flag = True
            else:
                if flag:
                    self.x = copy.deepcopy(self.x_)
                    self.xx = copy.deepcopy(self.xx_)
                    self.mu = copy.deepcopy(self.mu_)
                    self.sigma2 = copy.deepcopy(self.sigma2_)
                    self.W = copy.deepcopy(self.W_)
                    self.aic = self.AIC(pre_lik)
                    self.bic = self.BIC(pre_lik)
                break

    def loglik(self, Y):
        loglik = 0
        for h in range(self.dim):
            I_h = np.invert(np.isnan(Y[:, h]))
            wx = np.matmul(self.x[I_h], self.W[h, :])
            loglik += - np.sum( np.square(Y[I_h, h] - wx - self.mu[h])) / (2 * self.sigma2[h, h]) - np.log(self.sigma2[h, h]) / 2

        loglik += - np.square(np.sum(self.x)) / 2
        return loglik

    def BIC(self, loglik):
        dm, n_com = self.W.shape
        num_para = dm * n_com + 1
        return num_para * np.log(self.length) - 2 * loglik

    def AIC(self, loglik):
        dm, n_com = self.W.shape
        num_para = dm * n_com + n_com
        return 2 * num_para - 2 * loglik

    def Estep(self, Y):
        '''

        :param Y:
        :return:
        '''
        missing_idx = np.isnan(Y)
        nomissing_idx = np.invert(missing_idx).all(axis=1)
        Y_complete = Y[nomissing_idx, :]

        # nomissing data update
        Y_mu = Y_complete - self.mu

        C = self.sigma2 + np.matmul(self.W, self.W.transpose())
        try:
            C = self.sigma2 + np.matmul(self.W, self.W.transpose())
            C_inv = np.linalg.inv(C)
        except:
            diag_sigma = np.array(np.diag(self.sigma2))
            zero_idx = diag_sigma == 0
            if zero_idx.any():
                sigma_mean = np.mean(diag_sigma)
                diag_sigma[zero_idx] = sigma_mean
                self.sigma2 = np.diag(diag_sigma)
            C = self.sigma2 + np.matmul(self.W, self.W.transpose())
            C_inv = np.linalg.pinv(C)
        beta_t = np.matmul(C_inv, self.W)

        self.x[nomissing_idx] = np.matmul(Y_mu, beta_t)
        Val_x = np.eye(self.n_com) - np.matmul(self.W.transpose(), beta_t)

        self.xx[nomissing_idx] = np.expand_dims(Val_x, axis=0) \
                                 + np.matmul(np.expand_dims(self.x[nomissing_idx], axis=2), np.expand_dims(self.x[nomissing_idx], axis=1))

        # missing data update
        for i, idx_mis in enumerate(missing_idx):
            if idx_mis.any():
                idx_obs = np.invert(idx_mis)
                Y_mu = Y[i, idx_obs] - self.mu[idx_obs]

                C_ = C[:, idx_obs][idx_obs, :]
                try:
                    C_inv_ = np.linalg.inv(C_)
                except:
                    C_inv_ = np.linalg.pinv(C_)
                    print(C_)

                beta_t_ = np.matmul(C_inv_, self.W[idx_obs])
                Val_x = np.eye(self.n_com) - np.matmul(self.W[idx_obs].transpose(), beta_t_)

                self.x[i] = np.matmul(Y_mu, beta_t_)
                self.xx[i] = np.expand_dims(Val_x, axis=0) \
                             + np.matmul(np.expand_dims(self.x[i], axis=1), np.expand_dims(self.x[i], axis=0))

    def Mstep(self, Y):
        '''

        :param Y:
        :return:
        '''
        sigma_sum = np.zeros([self.dim])
        sigma_din = np.zeros([self.dim])
        for h in range(self.dim):
            I_h = np.invert(np.isnan(Y[:, h]))
            y_h = Y[I_h, h]

            # mu update
            wx = np.matmul(self.x[I_h], self.W[h, :])
            y_wx = y_h - wx
            mu_h = np.mean(y_wx, axis=0)
            self.mu[h] = mu_h

            # w_h update
            y_mu = y_h - self.mu[h]
            y_mu_x = np.matmul(y_mu, self.x[I_h])
            xx_inv = np.linalg.inv(np.nansum(self.xx[I_h], axis=0))
            w_h = np.matmul(y_mu_x, xx_inv)
            self.W[h, :] = w_h

            # sigma
            sigma_sum[h] += np.sum(np.square(y_h - wx - mu_h))
            sigma_din[h] += np.sum(I_h)

        sigma = sigma_sum / sigma_din
        self.sigma2 = np.diag(sigma)
        diag_sigma = np.array(np.diag(self.sigma2))
        zero_idx = diag_sigma == 0
        if zero_idx.any():
            sigma_mean = np.mean(diag_sigma)
            diag_sigma[zero_idx] = sigma_mean
            self.sigma2 = np.diag(diag_sigma)

    def predict_self(self):
        y_pred = np.matmul(self.x, self.W.transpose()) + self.mu
        return y_pred











