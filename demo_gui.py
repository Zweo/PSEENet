import os
import torch
from torch.utils.data import DataLoader, Dataset
from sklearn.manifold import TSNE
from net import PSENet as MODEL
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from scipy.io import loadmat
from gooey import Gooey, GooeyParser
from glob import glob
import random

device = torch.device("cpu")


class MData(Dataset):

    def __init__(self, path, mode, seed):
        super().__init__()
        '''
            Adjusted to the dataset in use.
        '''
        self.data = None
        self.label = None
        data = self.get_dataset(seed, path)
        for file in data[mode]:
            xx, yy = self.load_file(file)
            if self.data is None:
                self.data = xx
                self.label = yy
            else:
                self.data = np.append(self.data, xx, axis=0)
                self.label = np.append(self.label, yy, axis=0)

    def get_dataset(self, seed, path='data'):
        files = glob(f'{path}/*.mat')
        k = len(files) // 10
        sidx = (seed % 10) * k
        # op
        #
        # random.shuffle(files)
        test_files = files[sidx:sidx + k]
        train_files = list(set(files) - set(test_files))
        random.seed(seed)
        valid_files = random.sample(train_files, k=k)
        train_files = list(set(train_files) - set(valid_files))
        dataset_split = {
            'all': files,
            'train': train_files,
            'valid': valid_files,
            'test': test_files
        }
        return dataset_split

    def load_file(self, file):
        mat = loadmat(file)
        label = mat['y'].reshape(-1)
        signal = np.zeros((label.shape[0], 2, 3000))
        # raw data
        # signal[:, 0, :] = mat['eeg1fpz']  # EEG
        # signal[:, 1, :] = mat['eog']  # EOG
        # self-defined data
        sig = mat['x'][:].transpose(0, 2, 1)
        signal[:, 0, :] = sig[:, 1, :]  # EEG
        signal[:, 1, :] = sig[:, 0, :]  # EOG
        return signal, label

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        signal = self.data[idx]
        label = self.label[idx]
        return signal, label


def get_tSNE(path, datapath, random_state=2024):
    X = []
    y = []
    seed = 21
    test_dataset = MData(datapath, 'all', seed)
    test_loader = DataLoader(test_dataset, batch_size=64)
    model = MODEL()
    model.load_state_dict(torch.load(path, map_location='cpu')['model'])
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for data, label in test_loader:
            data_eeg = data[:, 0, :].float().to(device).unsqueeze(1)
            data_eog = data[:, 1, :].float().to(device).unsqueeze(1)
            _, _, out1, out2 = model(data_eog, data_eeg)
            X.extend(list(out1.cpu().numpy()))
            X.extend(list(out2.cpu().numpy()))
            lls = list(label.cpu().numpy())
            y.extend(lls)
            y.extend([10 + item for item in lls])
    X = np.array(X)
    ty = np.array(y)
    X_embedded = TSNE(n_components=2,
                      learning_rate='auto',
                      init='random',
                      random_state=random_state).fit_transform(X)
    return X_embedded, ty


def plot_1(X_embedded, ty):
    # 0-4 EOG
    # 10-14 EEG
    idx0 = np.where(ty < 10)
    idx1 = np.where(ty >= 10)
    plt.figure()
    my_colors = cm.YlOrRd(np.arange(10) / 5)
    idx0 = np.where(ty < 10)
    idx1 = np.where(ty >= 10)
    plt.scatter(X_embedded[idx0, 0], X_embedded[idx0, 1], color='blue')
    plt.scatter(X_embedded[idx1, 0], X_embedded[idx1, 1], color='red')
    # plt.subplot(3, 6, num_p + 1)
    for i in range(5):
        idx0 = np.where((ty % 10) == i)
        plt.scatter(X_embedded[idx0, 0] + 120,
                    X_embedded[idx0, 1],
                    color=my_colors[i])
    plt.savefig(f'tsne.png', dpi=200)


@Gooey(
    program_name='PSENet',
    program_description=
    'A simple demo of PSENet. Draw t-SNE plot of EEG and EOG signals.',
    required_cols=1,
)
def main():
    parser = GooeyParser()
    parser.add_argument(
        'model_path',
        default='models/model.pt',
        help='The path (File) where the model weights are located',
        widget='FileChooser')
    parser.add_argument('data_path',
                        default='data',
                        help='The path (Dir) where the data is located',
                        widget='DirChooser')
    parser.add_argument('random_state',
                        default=2024,
                        help='random state of t-SNE',
                        type=int)
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        raise FileNotFoundError("Must train first!")
    data = get_tSNE(args.model_path, args.data_path, args.random_state)
    plot_1(*data)


if __name__ == '__main__':
    main()
