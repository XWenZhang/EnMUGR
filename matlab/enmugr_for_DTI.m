function x = enmugr_for_DTI(networks, ndim, k, alpha, reg)  
    % running diffusion with Network Enhancement
    Q_concat = [];
    L_concat = zeros;   
    for i = 1:length(networks)
      S = networks{i};
      Q = NE_for_DTI(S,20,k,alpha);
      
      D = diag(sum(Q));
      L = D - Q;
      Q_concat = [Q_concat; Q];
      L_concat = L_concat + L;
    end
    
    clear Q A
    Q_concat = Q_concat / length(networks);

    % learning vector embedding via iterative optimization with lbfgsb
    x = ve_for_DTI(Q_concat, L_concat, ndim, reg);

end
