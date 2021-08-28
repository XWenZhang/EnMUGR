import pandas as pd
from scipy.io import loadmat
from enmugr import *
from MeanACC import meanacc

class Config(object):
    def __init__(self):
        self.embDim = 30
        self.k = 6
        self.alpha = 0.1
        self.reg = 40

opt = Config()

def main():
    data = loadmat('../data/simulation_data.mat')
    network, label = data['S'], data['label']

    represent_feature = enmugr(network, opt, opt.embDim)
    mean_retrv_acc = meanacc(represent_feature, label)

    df = pd.DataFrame(represent_feature)
    df.to_csv("simulation_feature.csv", header=None, index=None)

    print(mean_retrv_acc)

if __name__=='__main__':
    main()