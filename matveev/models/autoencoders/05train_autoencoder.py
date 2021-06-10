# -*- coding: 1251 -*-

"""
В этом файле классы автоенкодера и процесс поиска лучшей его конфигурации
"""


import numpy as np
import sys
from keras.callbacks import History
from keras.layers import InputLayer, Dense
from keras.models import Model, Sequential
from sklearn.metrics import mean_squared_error as mse
from typing import List, Optional, Sequence


class Autoencoder(Model):
    """
    Холдер для одного автоенкодера
    """
    def __init__(self, input_dim: int, output_dim: int, activation: Optional[str] = None) -> None:
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activation = activation
        self.build_autoencoder()

    def call(self, inputs):
        return self.autoencoder(inputs)

    def build_autoencoder(self) -> None:
        self.encoder = Sequential([
            InputLayer(input_shape=(self.input_dim,)),
            # Dense(90, activation='tanh'),
            # Dense(32, activation='relu'),
            Dense(self.output_dim, activation=self.activation)
        ])
        self.decoder = Sequential([
            InputLayer(input_shape=(self.output_dim,)),
            # Dense(32, activation='relu'),
            # Dense(90, activation='tanh'),
            Dense(self.input_dim, activation=None)
        ])
        self.autoencoder = Model(inputs=self.encoder.input, outputs=self.decoder(self.encoder.output))
        self.autoencoder.compile(loss='mse', optimizer='adam')


class StackedAutoencoder:
    """
    Ступенчатый автоенкодер: отдельно обучается каждый слой-encoder на основе результатов предыдущего слоя
    """
    def __init__(self, dims: Sequence[int], activation: Optional[str] = 'tanh') -> None:
        """:param dims: все размерности шагов (от input_dim до output_dim)"""
        self.dims = dims
        self.input_dim = dims[0]
        self.output_dim = dims[-1]
        self.autoencoders = [Autoencoder(dims[i - 1], dims[i], activation=activation) for i in range(1, len(dims))]

    def fit(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None, verbose: bool = False, epochs: int = 10) -> List[History]:
        if verbose:
            print(f'\nStackAutoencoder: fitting {len(self.autoencoders)} autoencoders with dimensions {self.dims}', flush=True)

        train_data_reduced = train_data
        val_data_reduced = val_data
        self.model_histories = []
        for i, autoencoder in enumerate(self.autoencoders):
            autoencoder.compile(loss='mse', optimizer='adam')
            if verbose:
                print(f'\tfitting autoencoder number {i} with dimensions {autoencoder.input_dim, autoencoder.output_dim}', file=sys.stderr)
            model_history = autoencoder.fit(
                x=train_data_reduced,
                y=train_data_reduced,
                epochs=epochs,
                batch_size=64,
                validation_data=(val_data_reduced, val_data_reduced),
                shuffle=True,
                verbose=False,
                # callbacks=[tensorboard]
            )
            train_data_reduced = autoencoder.encoder(train_data_reduced)
            val_data_reduced = autoencoder.encoder(val_data_reduced)
            self.model_histories.append(model_history)
            if verbose:
                print(f'StackedAutoencoder: autoencoder {i} - model history\n{model_history.history}', flush=True)

        self.fine_tune(train_data, val_data, verbose, epochs)
        return self.model_histories

    def fine_tune(self, train_data: np.ndarray, val_data: Optional[np.ndarray] = None, verbose: bool = False, epochs: int = 10) -> None:
        """Дообучает всю цепь сразу"""
        if verbose:
            print(f'StackedAutoencoder: fine-tuning...')

        self.autoencoder_sequence = Sequential()
        for autoencoder in self.autoencoders:
            self.autoencoder_sequence.add(autoencoder.encoder)
        for autoencoder in self.autoencoders[::-1]:
            self.autoencoder_sequence.add(autoencoder.decoder)

        self.autoencoder_sequence.compile(loss='mse', optimizer='adam')
        self.finetune_history = self.autoencoder_sequence.fit(
            x=train_data,
            y=train_data,
            epochs=epochs,
            batch_size=64,
            validation_data=(val_data, val_data),
            shuffle=True,
            verbose=False
        )
        if verbose:
            print(f'StackedAutoencoder: fine-tuning history model history\n{self.finetune_history.history}')

    def encode(self, data: np.ndarray) -> np.ndarray:
        for autoencoder in self.autoencoders:
            data = autoencoder.encoder(data)
        return data

    def decode(self, data: np.ndarray) -> np.ndarray:
        for autoencoder in self.autoencoders[::-1]:
            data = autoencoder.decoder(data)
        return data


if __name__ == '__main__':
    train_data = np.load('train/embeddings_128.npy')#[:30000, :]
    val_data = np.load('val/embeddings_128.npy')#[:5000, :]
    test_data = np.load('test/embeddings_128.npy')#[:5000, :]

    params_list = [
        [[128, 80], None, 40],
        [[128, 80], 'sigmoid', 40],
        [[128, 80], 'tanh', 40],
        [[128, 80], 'LeakyReLU', 40],

        [[128, 100, 80], None, 40],
        [[128, 100, 80], 'sigmoid', 40],
        [[128, 100, 80], 'tanh', 40],
        [[128, 100, 80], 'LeakyReLU', 40],
    ]

    for params in params_list:
        dims, activation, epochs = params
        # logging
        print('logging params:')
        print(f'dims {dims}')
        print(f'activation {activation}')
        print(f'epochs {epochs}')
        print(f'train_shape {train_data.shape},\nval_shape {val_data.shape}\ntest_shape {test_data.shape}', flush=True)

        sae = StackedAutoencoder(dims=dims, activation=activation)
        sae.fit(train_data, val_data, verbose=True, epochs=40)

        print()
        print(f'mse for train_data: {mse(train_data, sae.decode(sae.encode(train_data)))}')
        print(f'mse for val_data: {mse(val_data, sae.decode(sae.encode(val_data)))}')
        print(f'mse for test_data: {mse(test_data, sae.decode(sae.encode(test_data)))}')
        print('\n')
