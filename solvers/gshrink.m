function Y = gshrink(X, tau, p, q, groups)
[P,N] = size(X);

if nargin<5
    groups = 1:P;
end

Y = zeros(P,N);

if p == 2 && q == 1 % group penalty
    Y = prox_group_lasso(X,groups,tau);
    
elseif p == 1 && q == 2 % elitist
    Y = prox_elitist_group(X,groups,tau);
    
elseif p == 1 && q <= 1 && q > 0 % fractional
    Y = approx_prox_fractional(X,tau,q);
    
end

function Y = prox_elitist_group(X,groups,tau)
[P,N] = size(X);
Y = zeros(P,N);
for idx = 1:nbg
    %     for k = 1:N
    %    tau_p = tau/(1+tau) * sum(abs(X(groups == p,k)));
    %    Y(groups == p,k) = soft(X(groups == p,k),tau_p);
    %     end
    curr_X = X(groups == idx,:);
    tau_p = tau/(1+tau) * sum(abs(curr_X));
    Y(groups == idx,:) = max(abs(curr_X)-repmat(tau_p,sum(groups == idx),1),0).*sign(curr_X );
end


function Y = approx_prox_fractional(X,tau,q)
Y = max(abs(X)-tau^(2-q)*abs(X).^(q-1),0).*sign(X);


function Y = prox_group_lasso(X,groups,tau)
[P,N] = size(X);
Y = zeros(P,N);
nbg = P;%nbg = max(groups);
for idx = 1:nbg
    Y(groups == idx,:) = vector_soft_col(X(groups == idx,:),tau);
end


function Y = vector_soft_col(X,tau)
%  computes the vector soft columnwise
NU = sqrt(sum(X.^2));
A = max(0, NU-tau);
Y = repmat((A./(A+tau)),size(X,1),1).* X;


