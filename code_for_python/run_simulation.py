import pandas as pd
from scipy.io import loadmat
from enmugr import *


class Config(object):
    def __init__(self):
        self.embDim = 20
        self.alpha = 0.1
        self.k = 6
        self.reg = 30

opt = Config()

def main():
    data = loadmat('data/simulation_data.mat')
    network = data['S']

    represent_feature = enmugr(network, opt, opt.embDim)
    df = pd.DataFrame(represent_feature)

    df.to_csv("simulation_feature.csv", header=None, index=None)

if __name__=='__main__':
    main()