clc;clear;close all;
addpath(genpath('./matlab'))

sample_drug = 708;
sample_prot = 1512;
dim_drug = 100;  % dimension of embedding space for drug network
dim_prot = 400;  % dimension of embedding space for protein network

k = 12;          % parameter of KNN in Network Enhancement  
alpha = 0.5;     % parameter of random walk in Network Enhancement 
reg = 0.07;      % regulation coefficient

% assign drug and protein networks
drugnets = {'Sim_mat_drug_drug', 'Sim_mat_drug_disease', ...
    'Sim_mat_drug_se','Sim_mat_Drugs'};
protnets = {'Sim_mat_protein_protein', ...
   'Sim_mat_protein_disease','Sim_mat_Proteins'};

% load drug networks
network_dr = cell(1, length(drugnets));
for i = 1 : length(drugnets)
    netID = char(strcat('./data/',drugnets(i),'.txt'));
    network_dr{i} = load(netID);
end
% feature learning for drug network    
X = enmugr_for_DTI(network_dr, dim_drug, k, alpha, reg);
X = X';

% load protein networks
 network_pr = cell(1, length(protnets));
 for j = 1 : length(protnets)
     netID = char(strcat('./data/',protnets(j),'.txt'));
     network_pr{j} = load(netID);
 end 
% feature learning for protein network
 Y = enmugr_for_DTI(network_pr, dim_prot, k, alpha, reg);
 Y = Y';

save('feature.mat','X','Y');
