# -*- coding: 1251 -*-


import numpy as np
import sys
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error as mse


if __name__ == '__main__':
    train_data = np.load('train/embeddings_128.npy')#[:30000, :]
    val_data = np.load('val/embeddings_128.npy')#[:5000, :]
    test_data = np.load('test/embeddings_128.npy')#[:5000, :]

    n_components_list = [
        100,
        80,
        60,
        40,
        16,
    ]

    print(f'train_shape {train_data.shape},\nval_shape {val_data.shape}\ntest_shape {test_data.shape}\n\n', flush=True)

    for n_components in n_components_list:
        # logging
        print('logging params:')
        print(f'n_components {n_components}')

        pca = PCA(n_components)
        transformed_train = pca.fit_transform(train_data)
        reconstructed_train = pca.inverse_transform(transformed_train)
        transformed_val = pca.fit_transform(val_data)
        reconstructed_val= pca.inverse_transform(transformed_val)
        transformed_test = pca.fit_transform(test_data)
        reconstructed_test= pca.inverse_transform(transformed_test)

        print()
        print(f'mse for train_data: {mse(train_data, reconstructed_train)}')
        print(f'mse for val_data: {mse(val_data, reconstructed_val)}')
        print(f'mse for test_data: {mse(test_data, reconstructed_test)}')
        print('\n')
