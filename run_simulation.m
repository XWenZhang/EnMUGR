addpath(genpath('./core'))

load 'simulation_data.mat'

%compute representation featrues via EnMUGR
method = 'enmugr';
noiseLevel = 'highSNR';
seqID = 1:1:30;

ndim = 30;
knn = 6;
alpha = 0.1;
regular = 40;

acc_vec = zeros(length(seqID), 1);
for n = 1:length(seqID)
    
    S = network{n};
    
    W  = enmugr_for_DTI(S, ndim, knn, alpha, regular);
    Similarity = compute_similarity_matrix(W', 0.5);
    
    [~,acc]  = CallACC(Similarity, label);
    acc_vec(n) = acc;
end
            
mean_acc = mean(acc_vec);

%%
function [W_normal, W] = compute_similarity_matrix(org_matrix, var)

n = size(org_matrix,1);

% computing similarities
W = zeros(n,n);
for p = 1:n
    for q = 1:n
        y1 = org_matrix(p,:);
        y2 = org_matrix(q,:);   
 
        w_pq = exp(-(norm(y1-y2,2)/var)^2);
        
        W(p,q) = w_pq;
    end
end

W = W - diag(diag(W));
sumW = sum(W,2);
W0 = W./repmat(sumW, 1, size(W,1));
W_normal = (W0+W0')/2;
end
%% refer to retrieval accuracy in Network Enhancement
function [ retrievalacc, meanacc] = CallACC( W, labels )
W = W - diag(diag(W));
U = unique(labels);

if size(labels,1)<size(labels,2)
    labels = labels';
end


for i = 1 : length(U)
    indexx{i} = find(labels==U(i));
    leni(i) = length(indexx{i})-1;
end

[~, SortedIndex] = sort(W,2,'descend');

 for i = 1 : length(labels)
     li = labels(i);
     li = find(U==li);
     retrievalacc(i) = length(intersect(indexx{li},SortedIndex(i,1:leni(li))))/leni(li);
 end
     meanacc = mean(retrievalacc);
end