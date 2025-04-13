import sys

import torch.nn as nn
from autoencoder import _Autoencoder
from ae1 import AE1
from ae2 import AE2
from data import Data
from model import Model


class AE3(_Autoencoder):

    def __init__(self, path):
        super().__init__(path)

        n_kernels = 64

        self.encoder = Model(
            input_shape=(self.BATCH_SIZE, 64, 16, 16),
            layers=[
                nn.Conv2d(n_kernels, n_kernels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
            ]
        )

        self.decoder = Model(
            input_shape=(self.BATCH_SIZE, 64, 8, 8),
            layers=[
                nn.ConvTranspose2d(n_kernels, n_kernels, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.ReLU(),
            ]
        )

        self.model = Model(
            input_shape=self.encoder.input_shape,
            layers=[
                self.encoder,
                self.decoder
            ]
        )


if __name__ == '__main__':

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    ae1 = AE1('models/ae1.pt')
    ae2 = AE2('models/ae2.pt')

    data = Data.load('data', image_size=64)
    data.shuffle()
    data = ae1.encode(data)
    data = ae2.encode(data)

    ae = AE3('models/ae3.pt')
    ae.print()

    if not epochs:
        print(f'\nLoading {ae.path}...')
        ae.load()
    else:
        print(f'\nTraining...')
        ae.train(epochs, data)
        print(f'\nSaving {ae.path}...')
        ae.save()

    print(f'\nGenerating samples...')
    samples = ae.generate(data)
    data.display(32)
    samples.display(32)
