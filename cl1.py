import sys

import torch.nn as nn
from classifier import _Classifier
from ae1 import AE1
from ae2 import AE2
from ae3 import AE3
from data import Data
from model import Model


class Cl1(_Classifier):

    def __init__(self, path, encoder1, encoder2, encoder3):
        super().__init__(path)

        n_kernels = 64

        self.model = Model(

            input_shape=(self.BATCH_SIZE, 3, 64, 64),

            layers=[
                encoder1,
                encoder2,
                encoder3,
                nn.Flatten(),
                nn.Dropout(p=0.1),
                nn.Linear(n_kernels * 8 * 8, 256),
                nn.ReLU(),
                nn.Dropout(p=0.1),
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 4),
            ]
        )


if __name__ == '__main__':

    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else None

    ae1 = AE1('models/ae1.pt')
    ae2 = AE2('models/ae2.pt')
    ae3 = AE3('models/ae3.pt')

    data = Data.load('data', image_size=64)
    data.shuffle()

    cl = Cl1('models/cl1.pt', ae1.load().encoder, ae2.load().encoder, ae3.load().encoder)
    cl.print()

    if not epochs:
        print(f'\nLoading {cl.path}...')
        cl.load()
    else:
        train_data, test_data = data.split(.8)
        print(f'\nTraining...')
        cl.train(epochs, train_data, test_data)
        print(f'\nSaving {cl.path}...')
        cl.save()

    results = cl.classify(data)
    print(f'\nAccuracy: {results.accuracy(data):.1f}%')
    print(f'\nConfusion Matrix:\n\n{results.confusion_matrix(data)}')
    results.display(32, data)
