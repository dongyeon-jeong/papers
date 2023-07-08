import numpy as np
from scipy.stats import multivariate_normal
from sklearn import mixture
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering

class LMAR:

    def __init__(self, num_class, n_com, n_com_mix, seq_length1, seq_length2, pca, include_mean = False, cluster = "BGMM"):
        self.num_class = num_class
        self.n_com = n_com
        self.n_com_mix = n_com_mix
        self.seq_length1 = seq_length1
        self.seq_length2 = seq_length2
        self.pca = pca
        self.beta = []
        for i in range(num_class):
            self.beta.append(np.zeros([n_com_mix * seq_length2 + 1]))

        self.beta = np.array(self.beta).transpose()

        self.Sigma = []
        for i in range(num_class):
            self.Sigma.append(30 * np.identity(n_com))

        self.include_mean = include_mean

        if cluster == "BGMM":
            self.cluster = mixture.BayesianGaussianMixture(n_components=self.num_class, covariance_type='tied')
        elif cluster == "AgglomerativeClustering":
            self.cluster = AgglomerativeClustering(n_clusters=self.num_class)

        self.alpha = 1
        self.lambda_ = 0.001

    def loglik(self, pca_data, wpca_data, pca_target):

        alpha = self.calculate_alpha(wpca_data)
        zlog_alpha = np.multiply(self.Z, np.log(alpha + 1e-100))

        det = np.array([np.linalg.det(s) for s in self.Sigma])
        logdet = np.multiply(self.Z, np.log(det + 1e-100))

        inv = []
        for s in self.Sigma:
            try:
                inv.append(np.linalg.inv(s))
            except:
                inv.append(10 * np.identity(self.n_com))

        mu = self.calculate_mu(pca_data)
        error = np.zeros_like(self.Z)
        for i in range(self.num_class):
            err = pca_target - mu[i]
            # np.multiply(np.matmul(err, inv[i]), err)
            err = np.sum(np.multiply(np.matmul(err, inv[i]), err), axis=1)
            error[:, i] += err

        z_error = np.multiply(self.Z, error)

        lik = np.sum(zlog_alpha) - np.sum(logdet) / 2 - np.sum(z_error) / 2
        return lik

    def calculate_alpha(self, pca_data):
        # alpha = self.gmm.predict_proba(pca_data[:, 0, :])
        pca_data2 = np.reshape(pca_data[:, -self.seq_length2:, :self.n_com_mix],
                               [-1, self.n_com_mix * self.seq_length2])
        x1 = np.c_[pca_data2, np.ones([len(pca_data2)])]
        logit = np.matmul(x1, self.beta)
        logit = logit - np.max(logit, axis=1, keepdims=True)
        logit = np.exp(logit)
        logit_sum = np.sum(logit, axis=1, keepdims=True)
        alpha = logit / logit_sum

        return alpha


    def initialize_Z(self, pca_data, pca_target):

        self.gmm = self.cluster
        Z = self.gmm.fit_predict(pca_target)
        self.Zff = Z
        self.Z_wf = Z

        if self.include_mean:
            for i in range(self.num_class):
                idx = Z == i
                self.phi0[i] = np.mean(pca_target, axis=0)

        X = np.reshape(pca_data[:, :, :self.n_com_mix], [-1, self.seq_length1 * self.n_com_mix])
        X = np.c_[X, np.ones([len(X)])]
        lrm = LogisticRegression(fit_intercept=False)
        lrm.fit(X, Z)
        score = lrm.score(X, Z)

        # print("score", score)
        # print(lrm.coef_)
        for i, coef in enumerate(lrm.coef_):
            self.beta[:, i] = coef

        self.Z = np.eye(self.num_class)[Z]

    def fit(self, pca_data, wpca_data, pca_target, wpca_target, iteration = 5):
        if self.include_mean:
            self.phi = [[] for _ in range(self.num_class)]
            for i in range(self.num_class):
                for j in range(self.seq_length1):
                    self.phi[i].append(np.zeros([self.n_com, self.n_com]))

            self.phi0 = []
            for i in range(self.num_class):
                self.phi0.append(np.zeros([self.n_com]))
            self.phi0 = np.array(self.phi0)

        else:
            self.phi = [[] for _ in range(self.num_class)]
            for i in range(self.num_class):
                for j in range(self.seq_length1):
                    # self.phi[i].append(temp_phi[j])
                    # self.phi[i].append(np.zeros([self.n_com, self.n_com]))
                    if j == 0:
                        self.phi[i].append(np.eye(self.n_com))
                    else:
                        self.phi[i].append(np.zeros([self.n_com, self.n_com]))

        self.phi = np.array(self.phi)

        self.initialize_Z(wpca_data, wpca_target)

        # self.Z = np.random.randint(0, self.num_class, [pca_data.shape[0]])
        # self.Z = np.eye(self.num_class)[self.Z]
        pre_log = -1e+10
        for epoch in range(iteration):
            # self.Estep(pca_data, wpca_data, pca_target)
            # self.Mstep(pca_data, wpca_data, pca_target)
            if epoch > 0:
                self.Estep(pca_data, wpca_data, pca_target)
                self.Mstep(pca_data, wpca_data, pca_target)
            else:
                self.Mstep(pca_data, wpca_data, pca_target)
            cur_log = self.loglik(pca_data, wpca_data, pca_target)
            # print(epoch, cur_log)
            if (cur_log < pre_log):
                break
            pre_log = cur_log
            # print(epoch, cur_log)

    def calculate_mu(self, pca_data):

        mu = []
        for i in range(self.num_class):
            mu.append(np.zeros([len(pca_data), self.n_com]))
            for j in range(self.seq_length1):
                mu[i] += np.matmul(pca_data[:, j, :], self.phi[i, j])

        if self.include_mean:
            for i in range(self.num_class):
                mu.append(np.zeros([len(pca_data), self.n_com]))
                for j in range(self.seq_length1):
                    mu[i] += self.phi0[i]

        return np.array(mu)

    def Estep(self, pca_data, wpca_data, pca_target):
        # alpha = self.calculate_alpha(wpca_data)
        # mu = self.calculate_mu(pca_data)
        # psi = self.normal(pca_target, mu)
        # alpha_psi = np.multiply(alpha, psi)
        # self.Z = alpha_psi / (np.sum(alpha_psi, axis=1, keepdims=True) + 1e-200)

        alpha = self.calculate_alpha(wpca_data)

        mu = self.calculate_mu(pca_data)
        psi = - np.sum(np.square(np.expand_dims(pca_target, axis=0) - mu), axis=2) / (2 * np.reshape(np.array(self.Sigma)[:, 0, 0], [-1, 1]))
        psi = psi.transpose()
        psi = psi - np.max(psi, axis=1, keepdims=True)
        # print(alpha.shape, psi.shape)
        psi = np.exp(psi) / np.reshape((np.array(self.Sigma)[:, 0, 0] ** (pca_data.shape[-1] / 2)), [-1, 1]).transpose()
        # print("logit", logit)

        logit = np.multiply(alpha, psi)
        logit_sum = np.sum(logit, axis=1, keepdims=True)
        alpha_psi = logit / logit_sum
        # print("alpha_psi", alpha_psi)

        self.Z = alpha_psi / (np.sum(alpha_psi, axis=1, keepdims=True))
        # print("Z_sum", np.sum(self.Z, axis=0))
        # self.Z = np.around(self.Z)

    def normal(self, pca_target, mu):
        psi = np.zeros_like(self.Z)
        for t in range(len(pca_target)):
            for i in range(self.num_class):
                # print("why", i, self.Sigma[i])
                psi[t, i] = multivariate_normal.pdf(pca_target[t], mu[i, t], self.Sigma[i])
        return psi

    def Mstep(self, pca_data, wpca_data, pca_target):

        mu = self.calculate_mu(pca_data)

        # Mstep - beta update
        X = np.reshape(wpca_data[:, :, :self.n_com_mix], [-1, self.seq_length1 * self.n_com_mix])
        X = np.c_[X, np.ones([len(X)])]

        alpha = self.calculate_alpha(wpca_data)
        grad = []
        grad_scale = []
        for i in range(self.num_class):
            yx = np.sum(np.multiply(X, np.expand_dims(self.Z[:, i], axis=1)), axis=0)
            x_alpha = np.multiply(X, np.expand_dims(alpha[:, i], axis=1))
            x_alpha = np.sum(x_alpha, axis=0)
            dldb = yx - x_alpha + self.lambda_ * self.beta[:, i]
            grad.append(dldb)
            grad_scale.append(np.sum(np.abs(dldb)))

        if np.max(grad_scale) > 1e-4:

            XX = np.matmul(np.expand_dims(X, axis=2), np.expand_dims(X, axis=1))
            for i, dldb in enumerate(grad):
                exp_alpha = np.reshape(alpha[:, i], [-1, 1, 1])
                xx_alpha = np.multiply(XX, exp_alpha * (1 - exp_alpha))
                xx_alpha = np.sum(xx_alpha, axis=0)
                hess = xx_alpha + self.lambda_ * np.eye(xx_alpha.shape[0])
                hess_inv = np.linalg.inv(hess)

                self.beta[:, i] = self.beta[:, i] - np.matmul(hess_inv, dldb)


        # Mstep - Sigma update
        for i in range(self.num_class):
            err = pca_target - mu[i]
            temp = np.matmul(np.expand_dims(err, axis=1), np.expand_dims(err, axis=2))
            temp = self.Z[:, i] * np.squeeze(temp)
            # print(self.Z.shape)
            self.Sigma[i] = np.identity(self.n_com) * ( np.sum(temp) / (self.n_com * np.sum(self.Z[:, i])))
            # print(self.Sigma[i])

            # self.Sigma[i] = 2 * np.identity(self.n_com) * ( np.sum(temp) / (self.n_com * np.sum(self.Z[:, i])))

        inv = []
        for i, s in enumerate(self.Sigma):
            try:
                inv.append(np.linalg.inv(s))
            except:
                self.Sigma[i] = np.mean(self.Sigma, axis=0)
                inv.append(np.linalg.inv(np.mean(self.Sigma, axis=0)))

        # Mstep - phi update
        idx = [True for _ in range(len(pca_data))]
        for i in range(self.num_class):
            temp_mux = mu[i] - pca_target
            temp_mux_exp = np.expand_dims(temp_mux[idx], axis=1)

            for j in range(self.seq_length1):
                dldp = np.matmul(np.expand_dims(pca_data[idx, j, :], axis=2), temp_mux_exp)
                dldp = np.matmul(dldp, inv[i])
                dldp = np.multiply(dldp, np.reshape(self.Z[idx, i], [-1, 1, 1]))
                dldp = np.sum(dldp, axis=0)

                dldpp = np.matmul(np.expand_dims(pca_data[idx, j, :], axis=2),
                                  np.expand_dims(pca_data[idx, j, :], axis=1))
                dldpp = np.multiply(dldpp, np.reshape(self.Z[idx, i], [-1, 1, 1]))

                dldpp = np.sum(dldpp, axis=0)
                self.phi[i, j] = self.phi[i, j] - np.matmul(np.linalg.inv(dldpp), dldp)

    def demand_pca_predict(self, pca_data, wpca_data):

        alpha = self.calculate_alpha(wpca_data)
        mu = self.calculate_mu(pca_data)
        # print(np.multiply(alpha, mu).shape, alpha.shape, mu.shape)
        pred = np.zeros_like(mu[0])
        for i in range(self.num_class):
            pred += np.multiply(mu[i], np.expand_dims(alpha[:, i], axis=1))

        # self.cov = np.zeros([self.num_class, self.pca.n_components_, self.pca.n_components_])
        # WWT = np.matmul(self.pca.components_.T, self.pca.components_)
        # # print(np.mean(alpha, axis=0))
        # # print(WWT.shape, self.Sigma)
        # for k in range(self.num_class):
        #     self.cov[i] = WWT / self.Sigma[k][0, 0]
        # np.matmul(self.pca.component_

        return pred, alpha

    def to_warpeddata(self, pred):
        wpred = self.pca.inverse_transform(pred)
        return wpred

    def RMSE(self, real, wpred):
        loss = real - wpred
        loss = np.sqrt(np.mean(np.square(loss)))
        return loss

