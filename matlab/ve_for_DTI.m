%%% refer to vector_embedding in Mashup
function [x, w, P, fval, info] = ve_for_DTI(Q, L, ndim, reg)
rng(100)
    %% Optimize
    [nnode ncontext] = size(Q);
    nparam = (nnode + ncontext) * ndim;    

    opts = struct('factr', 1e4, 'pgtol', 0, 'm', 5, 'printEvery', 10, 'maxIts', 1000);
    
    while true
      % initialize vectors
      wx = rand(ndim, nnode + ncontext) / 10 - .05;
      opts.x0 = wx(:);
      
      % learning vector embedding via iterative optimization with lbfgsb
      [xopt, fval, info] = lbfgsb(@optim_fn, -inf(nparam,1), inf(nparam,1), opts);
     
      fval_vec = info.err(:,1);
      niterate = length(fval_vec);
      
      figure()
      plot(1:niterate, fval_vec);
      xlabel('iteration')
      ylabel('objective function')
      
      if info.iterations > 10
        break
      end
      fprintf('Premature termination (took %d iter to converge); trying again.\n', info.iterations);
    end
    wx = reshape(xopt, ndim, nnode + ncontext);

    % summarize output
    x = wx(:,1:ncontext);
    w = wx(:,ncontext+1:end);
    P = P_fn(x,w);
    
    %% Difine functions
    function [fval, grad] = optim_fn(wx)
        wx = reshape(wx, ndim, nnode + ncontext);

        P = P_fn(wx(:,1:ncontext), wx(:,ncontext+1:end));
        
        fval = obj_fn(P) + reg * trace(wx(:,1:ncontext) * L * wx(:,1:ncontext)');

        xgrad = wx(:,ncontext+1:end) * (P-Q) + 2 * reg * wx(:,1:ncontext) * L;
        wgrad = wx(:,1:ncontext) * (P-Q)';
        
        grad  = [xgrad, wgrad];
        grad = grad(:);
    end

    function P = P_fn(x, w)
        P = exp(w' * x);
        P = bsxfun(@rdivide, P, sum(P));
    end

    function res = obj_fn(P)
        v = zeros(ncontext,1);
        for j = 1:ncontext
            v(j) = kldiv(Q(:,j),P(:,j));
        end
        res = sum(v);
    end
   
    function res = kldiv(p,q)
        filt = p > 0;
        res = sum(p(filt) .* log(p(filt) ./ q(filt)));
    end
end
