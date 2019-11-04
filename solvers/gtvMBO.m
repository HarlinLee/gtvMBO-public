function sol = gtvMBO(V, Lambda, Y, mu, tol, dt)

    % Inputs:
    % V = eigenvectors of approximate Ls
    % Lambda = eigenvalues of approximate Ls
    % Y: a matrix of P x N (N-number of points, P-number of
    % features)
    % Outputs:
    % sol: solution to min_u |u|_{gTV}+|u_t|+mu|u-y|^2

    % Setup other input parameters
    Y = Y'; % size N x P
    [N, P] = size(Y);
    K = length(Lambda); % number of singular values
    U0 = zeros(N,P);
    
    
    %dt = 1e-2; % <---------------------- TUNE THIS
    maxiters = 5;
    num_bits = 8;
    denom = 1-dt*Lambda;
    
    % Decompose Y into 8 channels
    Ychannel = zeros(N, P, num_bits);
    for a = 1:P                                         
        t = min(max(ceil(Y(:,a)*255),0),255);
        M1 = de2bi(t,num_bits);
        Ychannel(:,a,:) = M1;
    end
      
    
    % Run iters for each channel
    Uchannels = zeros(N,P,num_bits);
    for e = 1:num_bits
        
        U = U0; % Initialize U for each channel
        
        % Set up iterations for each channel
        converged = 0;
        iter = 1;
       
        a = V'*U; % K x P
        d = zeros(K,P);
        Yslice = squeeze(Ychannel(:,:,e));
        
        while (iter <= maxiters && converged == 0)
            
            % Get U update                                    
            a = diag(denom)*a-dt*d;
            U = V*a;            
            d = mu*V'*(U-Yslice); % K x P
            
            % Threshold
            Unext = zeros(size(U));
            Unext(U >= 0.5) = 1;
                      
            % Check relative diff in update
            iter = iter + 1;
            err = norm(U - Unext, 'fro')/norm(U, 'fro');
            if (err <= tol)
                converged = 1;
            end
            U = Unext;
            
        end
        Uchannels(:,:,e) = U;
    end
    
    % Recombine Ui
    Uout = zeros(N,P);
    
    for h = 1:P
        Uout(:,h) = bi2de(squeeze(Uchannels(:,h,:)));
    end
    
    
    sol = (Uout/255)';
    sol = min(1,max(0,sol));
end