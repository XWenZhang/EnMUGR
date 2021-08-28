import pandas as pd
from scipy.io import loadmat
from enmugr import *


class Config(object):
    def __init__(self):
        self.embDrug = 100
        self.embProtein = 400
        self.k = 15
        self.alpha = 0.5
        self.reg = 0.06

opt = Config()

def main():
    drug_data = loadmat('../data/drug_network.mat')
    drug_network = drug_data['drug_network']
    prot_data = loadmat('../data/protein_network.mat')
    prot_network = prot_data['protein_network']

    represent_drug = enmugr(drug_network, opt, opt.embDrug)
    represent_prot = enmugr(prot_network, opt, opt.embProtein)

    df = pd.DataFrame(represent_drug)
    pf = pd.DataFrame(represent_prot)
    df.to_csv("drug_feature.csv", header=None, index=None)
    pf.to_csv("protein_feature.csv", header=None, index=None)

if __name__=='__main__':
    main()