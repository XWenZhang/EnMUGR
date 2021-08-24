addpath(genpath('./core'))

load 'drug_network.mat'
load 'protein_network.mat'

sample_drug = 708;
sample_prot = 1512;
dim_drug = 100;  % dimension of embedding space for drug network
dim_prot = 400;  % dimension of embedding space for protein network

k = 15;          % parameter of KNN in Network Enhancement  
alpha = 0.5;     % parameter of random walk in Network Enhancement 
reg = 0.06;      % regulation coefficient

% feature learning for drug network
X = enmugr_for_DTI(drug_network, dim_drug, k, alpha, reg);
X = X';

feature learning for protein network
Y = enmugr_for_DTI(protein_network, dim_prot, k, alpha, reg);
Y = Y';

save('feature.mat','X','Y');
