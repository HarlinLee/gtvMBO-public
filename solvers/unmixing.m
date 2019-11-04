function [S,A,it] = unmixing_v3r(X,S0,A0,para)
% Solve the following problem via ADMM
% min_{S,A>=0} .5*|X-SA|^2+lambda1*R(A)
%   where R(A) is the regularization of A
% rewritten as
% min_{S,A,B,C} .5*|X-CA|^2+lambda*R(B) 
%        s.t. A=B, S=C, S>=0, B\in\Pi
% Inputs:
% X: data matrix of size Lx(mn) with L bands and (mxn) spatial points
% S0: initial S of size LxK
% A0: initial A of size KxN
% para.maxiter: maximum number of iterations
% para.lambda: regularization parameter (smoothness/sparsity)
% para.rho: penalty parameter for A=B
% para.gamma: penalty parameter for S=C
% para.itermax: maximum number of iterations
% para.method: 'graphL', '3dTV', 'MBO'
% para.V: eigenvectors of graph Laplacian
% para.S: eigenvalues of graph Laplacian
% 
% 
% Outputs:
% S: mixing matrix
% A: abundance map
% it: number of iterations executed
% obj: objective function values

% Jing, Updated on 8-13-2019

K = size(S0,2);
N = size(X,2);
% N: the number of total spatial pixels

% Initialization
S = S0;
A = A0;
B = A;
Btilde = zeros(size(A));
Ctilde = zeros(size(S));

%obj = zeros(para.itermax,1);
print_flag = 0;
% Main loop
if print_flag
    fprintf('Method: %s\n',para.method)
end
for i = 1:para.itermax
    if rem(i,50) == 0 && print_flag
        fprintf('iter %d\n',i)
        disp(cond(A*A'))
        disp(cond(S'*S)) % tune parameter by checking ill-cond.
    end
    
    Apre = A;
    Spre = S;
    
    % C-subproblem:
    C = (X*A'+para.gamma*(S+Ctilde))/(A*A'+para.gamma*eye(K));
    
    % S-subproblem
    S = max(0,C-Ctilde);
        
    % A-subproblem:
    LHS = S'*S+para.rho*eye(K);
    RHS = S'*X+para.rho*(B-Btilde);
    A = LHS\RHS;
    
    if any(any(isnan(A))) >0 || any(any(isnan(S))) >0
        it = i;
        return 
    end
    
    if max(max(A)) > 10^3 || max(max(S)) > 10^3
        it = i;
        return
    end
    
    % projection onto the probability simplex
    A = SimplexProj(A); 

    % B-subproblem:
    tmp = A + Btilde;
    mu = para.rho/para.lambda;
    switch para.method
        case '3dTV'
            t = zeros(para.m,para.n,K);
            for j = 1:K
                t(:,:,j) = reshape(tmp(j,:),para.m,para.n);
            end
            B2 = ITV_ROF_3D(t,mu,para.tv_mu,1e2);
            B = zeros(K,N);
            for j = 1:K
                B(j,:) = reshape(B2(:,:,j),para.m*para.n,1);
            end
            
        case 'graphL'            
            V = para.V;
            Sigma = para.S;
            B = mu*tmp*V*diag(1./(Sigma+mu))*V';
                       
        case 'gtvMBO'
            B = gtvMBO(para.V, para.S, tmp, mu, para.tol, para.dt);                     
    end


    
    % update of auxiliary variables
    Btilde = Btilde + A-B;
    Ctilde = Ctilde + S-C;
    
    % save the objective function values at all iterations
    %obj(i) = 0.5*norm(X-S*A,'fro')+para.lambda*R(A)
    
    % stopping criteria
    if norm(Spre-S,'fro')<para.tol*norm(Spre) ...
            && norm(Apre-A,'fro')<para.tol*norm(Apre)
        break
    end
end
it = i;
