import os
import numpy as np
import pickle

from utils import *

from imputation.imputation_PPCA import imputation as im_PCA
from imputation.imputation_MPCA import imputation as im_MPCA
from imputation.imputation_FA import imputation as im_FA
from imputation.imputation_MFA import imputation as im_MFA

import matplotlib.pyplot as plt

############################# Read data ###################################

path_dir = 'grocery_miss_8h/'

missing_state = 'missing_20'
missing_dir = path_dir + missing_state + '/'
day_measurement = 96 * 1

miss_list = os.listdir(missing_dir)

file_miss = missing_dir + miss_list[0]
with open(file_miss, 'rb') as f:
    data = pickle.load(f)
load_data = data['load']
n_component = 2
n_component_pca = 2

# FA
im_fa = im_FA(n_component=n_component_pca, day_measurement=day_measurement)
imputed_data_fa = im_fa.imputation(load_data)

# MFA
im_mfa = im_MFA(n_component=n_component, n_component_pca=n_component_pca, day_measurement=day_measurement)
imputed_data_mfa = im_mfa.imputation(load_data)

# PPCA
im_pca = im_PCA(n_component=n_component_pca, day_measurement=day_measurement)
imputed_data_pca = im_pca.imputation(load_data)

# MPCA
im_mpca = im_MPCA(n_component=n_component, n_component_pca=n_component_pca, day_measurement=day_measurement)
imputed_data_mpca = im_mpca.imputation(load_data)

plt.plot(imputed_data_fa[:day_measurement * 7])
plt.plot(imputed_data_mfa[:day_measurement * 7])
plt.plot(imputed_data_pca[:day_measurement * 7])
plt.plot(imputed_data_mpca[:day_measurement * 7])
plt.show()
















