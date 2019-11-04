function [count, A, S]= glnmf(X, P, A_init, S_init, W, d, para)
D = sparse(diag(d));
[~, N] = size(X);

delta = 15;

A = A_init;
S = S_init;
Xbar = [X; delta*ones(1,N)];

count = 0;
while (norm(X-A*S, 'fro')/norm(X, 'fro') > para.tol) && (count < para.itermax)
    
    A = A.*(X*S'./(A*(S*S')));

    Abar = [A; delta*ones(1,P)];
    
    S = max(10^(-8), S);
    mask = S>10^(-4);

    S = S.*(Abar'*Xbar+para.mu*S*W)./(Abar'*Abar*S+0.5*para.lambda*(S.^(-0.5)).*mask+para.mu*S*D);
    count = count +1;
end
end