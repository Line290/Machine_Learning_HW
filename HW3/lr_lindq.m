function [W, glist] = lr_lindq(X, y, type)
% input:
%      x_hom:    data matrix with homogeneous form
%          y:    label, a vector
%       type:    3 methods, 0, 1 and 2 respectively. 
%
% output:
%      weight:  parameters in logistic regression weight = [b, w]
%      glist:   record the norm of gradient in iteration, a vector
[N, D] = size(X);
%C = y.max;
W = ones(D, 1);

if type == 0
    stop = 0.00001;
    % learning rate
    lr = 0.1;
    
    count = 0;
    while(count == 0|| glist(count) > stop)
        y_p = 1./(1 + exp(-X*W));
        
        dW = X' * (y_p - y);
        dW = dW/N;
        
        count = count + 1;
        glist(count) = norm(dW);
        % update weight
        W = W - lr * dW;
        glist(count);
    end
end

if type == 1
    stop = 0.00001;
    count = 0;
    
    while (count == 0 || glist(count) > stop)
        y_p = 1./(1 + exp(-X*W));
        
        dW = X'*(y_p - y);
        dW = dW / N;
        
        count = count + 1;
        glist(count) = norm(dW);
        
        d = y_p .* (1 - y_p);
        D = diag(d);
        
        H = X' * D * X
        W = W - inv(H) * dW
        
        glist(count)
        
    end
    
end


if type == 2 
    % normalize the X
%     X_col_min = min(X, [], 1)
    X_col_max = max(X, [], 1)
    X = X ./ X_col_max
    X = (X - 0.5)*2
    stop = 0.001;
    count = 0;
    I = diag(ones(D,1));

    while (count == 0 || glist(count) > stop)
%         W = (W - min(W)) ./ (max(W) - min(W));
        score = X*W;
%         score = (score - min(score)) ./ (max(score) - min(score))
        y_p = 1./(1 + exp(-score));
        
        dW = X'*(y_p - y);
        dW = dW / N;
        
        d = y_p .* (1 - y_p);
        if count == 0
            D = diag(d);
            H = X' * D * X;
            H = inv(H);
            
            W_prior = W;
            W = W - H * dW;
            dW_prior = dW;
        else
            s = W - W_prior;
            ddW = dW - dW_prior;
            
            dv = dot(ddW, s);
            H = (I - (s*ddW')/dv) * H * (I - (ddW*s')/dv) + (s*s')/dv;
            
            W_prior = W;
            W = W - H * dW;
            dW_prior = dW;
            
        end
        count = count + 1;
        glist(count) = norm(dW);
    end
end
    
if type == 3 
    % normalize the X
    X_col_max = max(X, [], 1)
    X = X ./ X_col_max
    X = (X - 0.5)*2
    
    idx = find(y==1);
    
    stop = 0.001;
    count = 0;
    I = diag(ones(D,1));

    while (count == 0 || glist(count) > stop)    
        y_p = 1./(1 + exp(-X*W));
        
        dW = X'*(y_p - y);
        dW = dW / N;
        
        d = y_p .* (1 - y_p);
        % deal with class-imbalance
%         idx = find(y==1);
        d(idx) = d(idx) * 20000
        
        if count == 0
            D = diag(d);
            H = X' * D * X;
            H = inv(H);
            
            W_prior = W;
            W = W - H * dW;
            dW_prior = dW;
        else
            s = W - W_prior;
            ddW = dW - dW_prior;
            
            dv = dot(ddW, s);
            H = (I - (s*ddW')/dv) * H * (I - (ddW*s')/dv) + (s*s')/dv;
            
            W_prior = W;
            W = W - H * dW;
            dW_prior = dW;
            
        end
        
        count = count + 1;
        glist(count) = norm(dW);
    end  
end

end
