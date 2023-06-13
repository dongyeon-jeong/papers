import os
import numpy as np
import pickle

from utils import *

from imputation.imputation_MPCA import imputation
from imputation.MPCA import MPCA

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
im = imputation(n_component=n_component, n_component_pca=n_component_pca, day_measurement=day_measurement)
imputed_data = im.imputation(load_data)
print(imputed_data)

plt.plot(imputed_data[:day_measurement * 7])
plt.show()


















