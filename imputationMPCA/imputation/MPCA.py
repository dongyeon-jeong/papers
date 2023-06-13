import numpy as np
import copy
from imputation.PPCA import PPCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import BayesianGaussianMixture

class MPCA:

    def __init__(self, n_component, n_component_pca):

        self.n_component = n_component
        self.n_component_pca = n_component_pca

    def initialize(self, Y):
        self.length, self.dim = np.shape(Y)

        missing_idx = np.isnan(Y)
        Y_temp_imputed = copy.deepcopy(Y)
        mu = np.nanmean(Y_temp_imputed, axis=0)

        for i, idx_m in enumerate(missing_idx):
            Y_temp_imputed[i, idx_m] = mu[idx_m]
        self.Y_temp_imputed = Y_temp_imputed

        pca = PPCA(self.n_component_pca)
        pca.fit(Y_temp_imputed)
        x = pca.x

        if self.n_component > 1:
            cm = KMeans(self.n_component, random_state=0)
            label = cm.fit_predict(x)
            Z = np.eye(self.n_component)[label]
            self.Z = Z

            self.W = np.zeros([self.n_component, self.dim, self.n_component_pca])
            self.mu = np.zeros([self.n_component, self.dim])
            self.pi = np.nanmean(self.Z, axis=0)
            self.sigma2 = np.zeros([self.n_component])
            self.x = np.zeros([self.n_component, self.length, self.n_component_pca])
            self.xx = np.zeros([self.n_component, self.length, self.n_component_pca, self.n_component_pca])
            # WWT = np.matmul(self.W.transpose(), self.W)
            for k in range(self.n_component):
                idx = label == k
                Y2 = copy.deepcopy(Y[idx])
                pca_ = PPCA(self.n_component_pca)
                pca_.fit(Y2)

                self.W[k] =pca_.W
                self.mu[k] = pca_.mu
                self.sigma2[k] = pca_.sigma2
                WWT = np.matmul(self.W[k].transpose(), self.W[k])

                M = self.sigma2[k] * np.eye(self.n_component_pca) + WWT
                M_inv = np.linalg.inv(M)
                self.x[k] = np.matmul(np.matmul(Y_temp_imputed - self.mu[k], self.W[k]), M_inv)
                xx = self.sigma2[k] * np.expand_dims(M_inv, axis=0) + np.matmul(np.expand_dims(self.x[k], axis=2), np.expand_dims(self.x[k], axis=1))
                self.xx[k] = xx
        else:
            label = np.zeros([len(x)])
            self.Z = np.ones([len(x), 1])
            self.length, self.dim = np.shape(Y)
            pca = PPCA(self.n_component_pca)
            pca.fit(Y)
            self.Z = np.ones([len(Y), 1])
            self.mu = np.zeros([self.n_component, self.dim])
            self.pi = np.nanmean(self.Z, axis=0)
            self.sigma2 = np.zeros([self.n_component])
            self.x = np.zeros([self.n_component, self.length, self.n_component_pca])
            self.xx = np.zeros([self.n_component, self.length, self.n_component_pca, self.n_component_pca])
            self.W = np.ones([self.n_component, self.dim, self.n_component_pca])
            for k in range(self.n_component):
                self.W[k] = pca.W
                self.mu[k] = pca.mu
                self.sigma2[k] = pca.sigma2
                self.x[k] = pca.x
                xx = pca.xx
                self.xx[k] = xx

                missing_idx = np.isnan(Y)
                Y_temp_imputed = copy.deepcopy(Y)
                mu = np.nanmean(Y_temp_imputed, axis=0)

                for i, idx_m in enumerate(missing_idx):
                    Y_temp_imputed[i, idx_m] = mu[idx_m]
                self.Y_temp_imputed = Y_temp_imputed

    def fit(self, Y, epochs = 100, preTrain = False):

        if ~preTrain:
            self.initialize(Y)

        if self.n_component > 1:
            tol = 1e-3
            val_tol = 3
            val = 0
            pre_lik = -1e+8
            min_epoch = 0
            for epoch in range(epochs):
                self.Estep(Y)
                self.Mstep(Y)
                lik = self.loglik(Y)
                rel_error = (lik - pre_lik) / np.abs(pre_lik)
                # print(epoch, lik, rel_error, val)
                min_epoch += 1
                if rel_error > tol and ~np.isnan(rel_error):
                    val = 0
                    pre_lik = lik
                    self.Z_ = copy.deepcopy(self.Z)
                    self.pi_ = copy.deepcopy(self.pi)
                    self.x_ = copy.deepcopy(self.x)
                    self.xx_ = copy.deepcopy(self.xx)
                    self.mu_ = copy.deepcopy(self.mu)
                    self.sigma2_ = copy.deepcopy(self.sigma2)
                    self.W_ = copy.deepcopy(self.W)
                    self.bic = self.BIC(lik)
                    self.aic = self.AIC(lik)
                else:
                    # val += 1
                    # if val >= val_tol:
                    self.Z = copy.deepcopy(self.Z_)
                    self.pi = copy.deepcopy(self.pi_)
                    self.x = copy.deepcopy(self.x_)
                    self.xx = copy.deepcopy(self.xx_)
                    self.mu = copy.deepcopy(self.mu_)
                    self.sigma2 = copy.deepcopy(self.sigma2_)
                    self.W = copy.deepcopy(self.W_)
                    self.bic = self.BIC(pre_lik)
                    self.aic = self.AIC(pre_lik)
                    break
                
        else:
            lik = self.loglik(Y)
            self.bic = self.BIC(lik)
            self.aic = self.AIC(lik)

    def loglik(self, Y):
        loglik = 0
        for h in range(self.dim):
            I_h = np.invert(np.isnan(Y[:, h]))
            for k in range(self.n_component):
                Z_I = self.Z[:, k] * I_h

                wx = np.matmul(self.x[k], self.W[k, h, :])
                y_mu = self.Y_temp_imputed[:, h] - self.mu[k, h]

                wtw = np.matmul(np.expand_dims(self.W[k, h, :], axis=1), np.expand_dims(self.W[k, h, :], axis=0))
                wwxx = np.matmul(self.xx[k], np.expand_dims(wtw, axis=0))
                tr_wwxx = np.trace(wwxx, axis1=1, axis2=2)

                loglik += np.matmul( Z_I, (- np.square(y_mu) / (2 * self.sigma2[k]) + wx * y_mu / self.sigma2[k] - tr_wwxx / (2 * self.sigma2[k]) - np.log(self.sigma2[k]) / 2))
                loglik += - np.matmul(self.Z[:, k], np.squeeze(np.trace(self.xx[k], axis1=1, axis2=2))) / 2

        return loglik

    def AIC(self, loglik):
        k, dm, n_com = self.W.shape
        num_para = (n_com + 1) * dm * self.n_component + self.n_component
        return 2 * num_para - 2 * loglik

    def BIC(self, loglik):
        k, dm, n_com = self.W.shape
        num_para = (n_com + 1) * dm * self.n_component + self.n_component
        return num_para * np.log(self.length) - 2 * loglik

    def Estep(self, Y):
        '''

        :param Y:
        :return:
        '''
        if self.n_component > 1:
            missing_idx = np.isnan(Y)
            nomissing_idx = np.invert(missing_idx).all(axis=1)
            Y_complete = Y[nomissing_idx, :]

            logit = np.zeros([self.length, self.n_component])
            M_inv_k = np.zeros([self.n_component, self.n_component_pca, self.n_component_pca])
            C = np.zeros([self.n_component, self.dim, self.dim])
            C_inv_k = np.zeros([self.n_component, self.dim, self.dim])
            logdet_C_inv = np.zeros([self.n_component])
            # nomissing data update
            for k in range(self.n_component):
                M = np.matmul(self.W[k].transpose(), self.W[k]) + self.sigma2[k] * np.eye(self.n_component_pca)
                M_inv = np.linalg.inv(M)
                M_inv_k[k] = M_inv
                Y_mu = Y_complete - self.mu[k]
                Y_wx_mu = Y_mu - np.matmul(self.x[k, nomissing_idx], self.W[k].transpose())

                C[k] = np.matmul(self.W[k], self.W[k].transpose()) + self.sigma2[k] * np.eye(self.dim)
                C_inv_k[k] = np.linalg.inv(C[k])
                logdet_C_inv[k] = np.linalg.slogdet(C_inv_k[k])[1]

                logit[nomissing_idx, k] = np.sum(np.matmul(Y_mu, C_inv_k[k]) * Y_mu, axis=1)
                logit[nomissing_idx, k] = - logit[nomissing_idx, k] / 2 + np.log(self.pi[k]) + logdet_C_inv[k] / 2

                self.x[k, nomissing_idx] = np.matmul(np.matmul(Y_mu, self.W[k]), M_inv)
                self.xx[k, nomissing_idx] = np.matmul(np.expand_dims(self.x[k, nomissing_idx], axis=2), np.expand_dims(self.x[k, nomissing_idx], axis=1)) + self.sigma2[k] * np.expand_dims(M_inv, axis=0)

            # missing data update
            for i, idx_mis in enumerate(missing_idx):
                if idx_mis.any():
                    idx_obs = np.invert(idx_mis)
                    for k in range(self.n_component):
                        W_obs = self.W[k, idx_obs, :]
                        M_inv = M_inv_k[k]
                        Y_mu = Y[i, idx_obs] - self.mu[k, idx_obs]
                        Y_wx_mu = Y_mu - np.matmul(W_obs, self.x[k, i])

                        try:
                            C_inv = np.linalg.inv(C[k][:, idx_obs][idx_obs, :])
                        except:
                            C_inv = np.linalg.pinv(C[k][:, idx_obs][idx_obs, :])

                        logit[i, k] = np.sum(np.inner(Y_mu, C_inv) * Y_mu)
                        logit[i, k] = - logit[i, k] / 2 + np.log(self.pi[k]) + logdet_C_inv[k] / 2
                        self.x[k, i] = np.matmul(np.matmul(Y_mu, W_obs), M_inv)
                        self.xx[k, i] = np.matmul(np.expand_dims(self.x[k, i], axis=1), np.expand_dims(self.x[k, i], axis=0)) + self.sigma2[k] * M_inv
            logit = logit - np.max(logit, axis=1, keepdims=True)
            logit = np.exp(logit)
            Z = logit / np.sum(logit, axis=1, keepdims=True)
            self.Z = Z

        else:
            self.Z = np.ones([len(Y), 1])

    def Mstep(self, Y):
        '''

        :param Y:
        :return:
        '''
        self.pi = np.nanmean(self.Z, axis=0)
        sigma_sum = np.zeros([self.n_component])
        sigma_din = np.zeros([self.n_component])
        for h in range(self.dim):
            I_h = np.invert(np.isnan(Y[:, h]))
            y_h = copy.deepcopy(Y[:, h])
            y_h[np.invert(I_h)] = 0
            for k in range(self.n_component):
                Z_I = self.Z[:, k] * I_h

                # w_h update
                ZI_y_mu_x = np.matmul(Z_I * (y_h - self.mu[k, h]), self.x[k])
                ZI_xx_inv = np.linalg.inv( np.sum(self.xx[k] * np.reshape(Z_I, [-1, 1, 1]), axis=0) )
                self.W[k, h, :] = np.matmul(ZI_y_mu_x, ZI_xx_inv)

                # mu update
                wx = np.matmul(self.x[k], self.W[k, h, :])
                y_wx = y_h - wx
                self.mu[k, h] = np.average(y_wx, axis=0, weights=Z_I)

                # sigma update
                wtw = np.matmul(np.expand_dims(self.W[k, h, :], axis=1), np.expand_dims(self.W[k, h, :], axis=0))
                wwxx = np.matmul(self.xx[k], np.expand_dims(wtw, axis=0))
                tr_wwxx = np.trace(wwxx, axis1=1, axis2=2)

                sigma_sum[k] += np.matmul(Z_I, np.square(y_h - self.mu[k, h]) - 2 * wx * (y_h - self.mu[k, h]) + tr_wwxx)
                sigma_din[k] += np.sum(Z_I * I_h)

        sigma = sigma_sum / sigma_din
        self.sigma2 = sigma

    def predict_self(self, Y):
        yhat = np.zeros([self.length, self.n_component, self.dim])
        for k in range(self.n_component):
            yhat[:, k, :] = np.matmul(self.x[k], self.W[k].transpose()) + self.mu[k]

        Z = np.expand_dims(self.Z, axis=1)
        y_pred = np.squeeze(np.matmul(Z, yhat))

        return y_pred

    def post_dist_z(self, Y, Y_hat):

        nomissing_idx = np.invert(np.isnan(Y)).all(axis=1)
        Y_temp = Y[nomissing_idx]
        Y_hat_temp = Y_hat[nomissing_idx]
        logit = np.zeros([len(Y_temp), self.n_component])
        for k in range(self.n_component):
            Y_wx_mu = Y_temp - Y_hat_temp[:, k, :]
            logit[:, k] = - np.sum(np.square(Y_wx_mu), axis=1) / 2 + np.log(self.pi[k])
        logit = logit - np.max(logit, axis=1, keepdims=True)
        logit = np.exp(logit)
        Z = logit / np.sum(logit, axis=1, keepdims=True)

        return Z, nomissing_idx


    def calculate_logit(self, Y):

        missing_idx = np.isnan(Y)
        nomissing_idx = np.invert(missing_idx).all(axis=1)
        Y_complete = Y[nomissing_idx, :]

        logit = np.zeros([self.length, self.n_component])
        C = np.zeros([self.n_component, self.dim, self.dim])
        C_inv_k = np.zeros([self.n_component, self.dim, self.dim])
        logdet_C_inv = np.zeros([self.n_component])
        # nomissing data update
        for k in range(self.n_component):
            Y_mu = Y_complete - self.mu[k]

            C[k] = np.matmul(self.W[k], self.W[k].transpose()) + self.sigma2[k] * np.eye(self.dim)
            C_inv_k[k] = np.linalg.inv(C[k])
            logdet_C_inv[k] = np.linalg.slogdet(C_inv_k[k])[1]

            logit[nomissing_idx, k] = np.sum(np.matmul(Y_mu, C_inv_k[k]) * Y_mu, axis=1)
            logit[nomissing_idx, k] = - logit[nomissing_idx, k] / 2 + np.log(self.pi[k]) + logdet_C_inv[k] / 2


        # missing data update
        for i, idx_mis in enumerate(missing_idx):
            if idx_mis.any():
                idx_obs = np.invert(idx_mis)
                for k in range(self.n_component):
                    Y_mu = Y[i, idx_obs] - self.mu[k, idx_obs]

                    C_inv = np.linalg.inv(C[k][:, idx_obs][idx_obs, :])

                    logit[i, k] = np.sum(np.inner(Y_mu, C_inv) * Y_mu)
                    logit[i, k] = - logit[i, k] / 2 + np.log(self.pi[k]) + logdet_C_inv[k] / 2

        return logit


    def calculate_y_k(self):

        yhat = np.zeros([self.length, self.n_component, self.dim])
        for k in range(self.n_component):
            yhat[:, k, :] = np.matmul(self.x[k], self.W[k].transpose()) + self.mu[k]

        return yhat

















