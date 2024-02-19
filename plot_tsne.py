import os
from glob import glob
import torch
from torch.utils.data import DataLoader
from train import MData
from sklearn.manifold import TSNE
from net import PSENet as MODEL
import pickle
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cpu")


def load_model(path):
    model = MODEL()
    files = glob('{}/*.pt'.format(path))
    if files:
        files = sorted(files,
                       key=lambda x: float(
                           os.path.basename(x).split('.pt')[0].split('-')[1]))
        file = files[-1]
        model.load_state_dict(torch.load(file, map_location='cpu')['model'])
    model = model.to(device)
    return model


def get_tSNE(path, datapath):
    X = []
    y = []
    file = glob(path + '/*')[0]
    print(file)
    seed = 21
    test_dataset = MData(datapath, 'all', seed)
    test_loader = DataLoader(test_dataset, batch_size=64)
    model = load_model(file)
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
                      random_state=2022).fit_transform(X)
    return X_embedded, ty


def plot_1(X_embedded, ty):
    # 0-4 EOG
    # 10-14 EEG
    idx0 = np.where(ty < 10)
    idx1 = np.where(ty >= 10)
    plt.figure()
    from matplotlib import cm
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



def main(model_path='EE-cos/Sleep-EDFX/21', data_path='data'):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Must train first!")
    data= get_tSNE(model_path, data_path')
    plot_1(*data)


def demo():
    # demo
    # draw tSNE of X1,X2 with labels on one figure.
    # 
    X1 = np.random.randn(25, 256)
    X2 = np.random.randn(25, 256)
    X = np.vstack([X1, X2])
    y1 = np.random.random_integers(0,4, 25)
    ty = np.vstack([y1, y1])
    X_embedded = TSNE(n_components=2,
                      learning_rate='auto',
                      init='random',
                      random_state=2022).fit_transform(X)
    plot_1(X_embedded, ty)

if __name__ == '__main__':
    try:
        main()
    except:
        demo()