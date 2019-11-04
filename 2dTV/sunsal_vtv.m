function U = sunsal_tv(A,Y,varargin)

%% [U,res,rmse] = sunsal_vtv(M,y,varargin)
%
%  SUNSAL_VTV ->  sparse unmixing with vector TV via variable splitting and 
%                 augmented
%  Lagragian methods  intoduced in
%
%  J. Bioucas-Dias, "???", 2013
%
%
%% --------------- Description ---------------------------------------------
%
%  SUNSAL_VTV solves the following l_2 + l_{1,1} + VTV optimization problem:
%
%     Definitions
%
%      A  -> L * n; Mixing matrix (Library)
%      X  -> n * N; collection of N fractional vectors; each column  of X
%                   contains the fraction of a correspondent  pixel. Each
%                   row of X  corresponds to a band with [nlins, ncols] 
%      
%
%      Optimization problem
%
%    min  (1/2) ||A X-Y||^2_F  + lambda_1  ||X||_{1,1}
%     X                         + lambda_vtv VTV(X);
%
%
%    where
%
%        (1/2) ||A X-Y||^2_F is a quadractic data misfit term
%
%        ||X||_{1,1} = sum_i ||X(:,i)||_1, for i = 1,...,N.
%                      is the standard l1 regularizer
%
%        VTV(X)  is the VTV (non-isotropic or isotropic regularizer)
%
%        a) NON-ISOTROPIC:  
%    
%               VTV(X) = sum_i ||X(:,i) - X(:,h(i))||_2 + ||X(:,i) - X(:,v(i))||_2
%               
%        where h(i) are the  is the right and  bottom  neighbors of i, respectively.
%        We are assuming cyclic boundaries
%
%
%        a) NON-ISOTROPIC:  
% 
%               VTV(X) = sum_i sqrt(   ||X(:,i) - X(:,h(i))||_2^2 
%                                    + ||X(:,i) -  X(:,v(i))||_2^2   )
%               
%
% -------------------------------------------------------------------------
%
%
%
%    CONSTRAINTS ACCEPTED:
%
%    1) Positivity X(:,i) >= 0, for i=1,...,N
%    2) Sum-To-One sum( X(:,i)) = 1, for for i=1,...,N
%
%
%    NOTES:
%
%       1) If X is a matrix and lambda_TV = 0, SUNSAL_TV solves
%           columnwise independent optimizations.
%
%       2) If both the Positivity and Sum-To-One constraints are active,
%          then we have ||X||_{1,1} = n and therefore this regularizer
%          is doing nothing.
%
%
%% -------------------- Line of Attack  -----------------------------------
%
%  SUNSAL_VTV solves the above optimization problem by introducing a variable
%  splitting and then solving the resulting constrained optimization with
%  the augmented Lagrangian method.
%
%
%   The initial problem is converted into
%
%    min  (1/2) ||A X-Y||^2_F  + i_R_+(X)
%     X                        + i_S(X)
%                              + lambda_1  ||X||_{1,1}
%                              + lambda_tv psi(LhX,LvX);
%
%
%   where i_R_+ and i_S are the indicator functions of the set R_+ and
%   the probability simplex, respecively, applied to the columns ox X.
%   the fuction phi is 
%
%        a) NON-ISOTROPIC:  
%    
%               psi(Uh,Uv) = sum_i ||Uh(:,i)||_2 + ||Uv(:,i)||_2
%               
%        where h(i) are the  is the right and  bottom  neighbors of i, respectively.
%        We are assuming cyclic boundaries
%
%
%        a) NON-ISOTROPIC:  
% 
%               psi(Uh,Uv) = sum_i sqrt{ ||Uh(:,i)||_2 + ||Uv(:,i)||_2 }
%
%
%  Then, we apply the following variable splitting
%
%  IF 'POSITIVITY' = 'no'   &&  'ADDONE' = 'no'
%
%    min  (1/2) ||AV1-Y||^2    + lambda_1  ||V2||_{1,1}
%   U,V1,V2,V3                  + lambda_vtv psi(V3,V4);
%
%     subject to:  
%                  U    = V1
%                  U    = V2
%                  LhU  = V3
%                  LvU  = V4
%
%
%  IF 'POSITIVITY' = 'no'   &&  'ADDONE' = 'yes'
%
%    min  (1/2) ||AV1-Y||^2    + lambda_1  ||V2||_{1,1}
%   U,V1,V2,V3                 + lambda_vtv psi(V3,V4);
%                              + i_S(V5)
%
%     subject to:  U  = V1
%                  U  = V2
%                  LhU  = V3
%                  LvU  = V4
%                  U    = V5
%
%
%   IF 'POSITIVITY' = 'yes'  replace V2 with V2 >= 0 above
%
%  For details see
%
%
%  J. Bioucas-Dias and M. Figueiredo, �Alternating direction algorithms for
%  constrained sparse regression: Application to hyperspectral unmixing�,
%  in  2nd  IEEE GRSS Workshop on Hyperspectral Image and Signal Processing-WHISPERS'2010,
%  Raykjavik, Iceland, 2010.
%
%
%  M.-D. Iordache, J. Bioucas-Dias, and A. Plaza, "Total variation spatial
%  regularization for sparse hyperspectral unmixing", IEEE Transactions on
%  Geoscience and Remote Sensing, vol. PP, no. 99, pp. 1-19, 2012.
%
%  M.-D. Iordache, J. Bioucas-Dias and A. Plaza, "Sparse unmixing
%  of hyperspectral data", IEEE Transactions on Geoscience and Remote Sensing,
%  vol. 49, no. 6, pp. 2014-2039, 2011.
%
%  M. V. Afonso, J. Bioucas-Dias, and M. Figueiredo, �An Augmented
%  Lagrangian Approach to the Constrained Optimization Formulation of
%  Imaging Inverse Problems�, IEEE Transactions on Image Processing,
%  vol. 20, no. 3, pp. 681-695, 2011.
%
%
%
%
% ------------------------------------------------------------------------
%%  ===== Required inputs =============
%
%  M - [L(observations) * n (variables)] system matrix (usually a library)
%
%  Y - matrix with  L(observation) x N(pixels).
%
%
%%  ====================== Optional inputs =============================
%
%
%  'LAMBDA_1' - regularization parameter for l11 norm.
%               Default: 0;
%
%  'LAMBDA_VTV' - regularization parameter for TV norm.
%                Default: 0;
%
%  'TV_TYPE'   - {'iso','niso'} type of total variation:  'iso' ==
%                isotropic; 'n-iso' == non-isotropic; Default: 'niso'
%
%  'IM_SIZE'   - [nlins, ncols]   number of lines and rows of the
%                spectral cube. These parameters are mandatory when
%                'LAMBDA_TV' is  passed.
%                Note:  n_lin*n_col = N
%
%
%  'AL_ITERS' - (double):   Minimum number of augmented Lagrangian iterations
%                           Default 100;
%
%
%  'MU' - (double):   augmented Lagrangian weight
%                           Default 0.001;
%
%
%
%  'POSITIVITY'  = {'yes', 'no'}; Default 'no'
%                  Enforces the positivity constraint: x >= 0
%
%  'ADDONE'  = {'yes', 'no'}; Default 'no'
%               Enforces the positivity constraint: x >= 0
%
%  'TRUE_X'  - [n (variables), N (pixels)] original data in matrix format.
%              If  the XT (the TRUE X) is inputted, then the RMSE is
%              ||X-XT||computed along the iterations
%
%
%  'VERBOSE'   = {'yes', 'no'}; Default 'no'
%
%                 'no' - work silently
%                 'yes' - display warnings
%
%%  =========================== Outputs ==================================
%
% U  =  [nxN] estimated  X matrix
%
%

%%
% ------------------------------------------------------------------
% Author: Jose Bioucas-Dias, July, 2013.
%
%%
%
%% -------------------------------------------------------------------------
%
% Copyright (January, 2013):        Jos� Bioucas-Dias (bioucas@lx.it.pt)
%
% SUNSAL_VTV is distributed under the terms of
% the GNU General Public License 2.0.
%
% Permission to use, copy, modify, and distribute this software for
% any purpose without fee is hereby granted, provided that this entire
% notice is included in all copies of any software which is or includes
% a copy or modification of this software and in all copies of the
% supporting documentation for such software.
% This software is being provided "as is", without any express or
% implied warranty.  In particular, the authors do not make any
% representation or warranty of any kind concerning the merchantability
% of this software or its fitness for any particular purpose."
% ----------------------------------------------------------------------

%%
%--------------------------------------------------------------
% test for number of required parametres
%--------------------------------------------------------------
if (nargin-length(varargin)) ~= 2
    error('Wrong number of required parameters');
end
% mixing matrix size
[LM,n] = size(A);
% data set size
[L,N] = size(Y);
if (LM ~= L)
    error('mixing matrix M and data set y are inconsistent');
end




%%
%--------------------------------------------------------------
% Set the defaults for the optional parameters
%--------------------------------------------------------------
%


% 'LAMBDA_1'
lambda_l1 = 0;

% 'LAMBDA_VTV'
%  TV regularization
lambda_VTV = 0;
im_size = []; % image size
tv_type = 'iso'; % isotropic TV

% 'AL:ITERS'
% maximum number of AL iteration
AL_iters = 1000;

% 'MU'
% AL weight
mu = 0.001;

% 'VERBOSE'
% display only sunsal warnings
verbose = 'no';

% 'POSITIVITY'
% Positivity constraint
positivity = 'no';
reg_pos = 0; % absent

% 'ADDONE'
%  Sum-to-one constraint
addone = 'no';
reg_add = 0; % absent

%

% initialization
U0 = 0;

% true X
true_x = 0;
rmse = 0;

%%
%--------------------------------------------------------------
% Local variables
%--------------------------------------------------------------


%--------------------------------------------------------------
% Read the optional parameters
%--------------------------------------------------------------
if (rem(length(varargin),2)==1)
    error('Optional parameters should always go by pairs');
else
    for i=1:2:(length(varargin)-1)
        switch upper(varargin{i})
            case 'LAMBDA_1'
                lambda_l1 = varargin{i+1};
                if lambda_l1 < 0
                    error('lambda must be positive');
                elseif lambda_l1 > 0
                    reg_l1 = 1;
                end
            case 'LAMBDA_VTV'
                lambda_VTV = varargin{i+1};
                if lambda_VTV < 0
                    error('lambda must be non-negative');
                elseif lambda_VTV > 0
                    reg_TV = 1;
                end
            case 'TV_TYPE'
                tv_type = varargin{i+1};
                if ~(strcmp(tv_type,'iso') | strcmp(tv_type,'niso'))
                    error('wrong TV_TYPE');
                end
            case 'IM_SIZE'
                im_size = varargin{i+1};
            case 'AL_ITERS'
                AL_iters = round(varargin{i+1});
                if (AL_iters <= 0 )
                    error('AL_iters must a positive integer');
                end
            case 'POSITIVITY'
                positivity = varargin{i+1};
                if strcmp(positivity,'yes')
                    reg_pos = 1;
                end
            case 'ADDONE'
                addone = varargin{i+1};
                if strcmp(addone,'yes')
                    reg_add = 1;
                end
            case 'MU'
                mu = varargin{i+1};
                if mu <= 0
                    error('mu must be positive');
                end
            case 'VERBOSE'
                verbose = varargin{i+1};
            case 'X0'
                U0 = varargin{i+1};
            case 'TRUE_X'
                XT = varargin{i+1};
                true_x = 1;
            otherwise
                % Hmmm, something wrong with the parameter string
                error(['Unrecognized option: ''' varargin{i} '''']);
        end;
    end;
end

% test for true data size correctness
if true_x
    [nr nc] = size(XT);
    if (nr ~= n) | (nc ~= N)
        error('wrong image size')
    end
end


% test for image size correctness
if N ~= prod(im_size)
    error('wrong image size')
end
n_lin = im_size(1);
n_col = im_size(2);

% build handlers and necessary stuff
% horizontal difference operators
FDh = zeros(im_size);
FDh(1,1) = -1;
FDh(1,end) = 1;
FDh = fft2(FDh);
FDhC = conj(FDh);

% vertical difference operator
FDv = zeros(im_size);
FDv(1,1) = -1;
FDv(end,1) = 1;
FDv = fft2(FDv);
FDvC = conj(FDv);

II = 1     ./(abs(FDh).^2+ abs(FDv).^2 + 2);
IDh= FDhC ./(abs(FDh).^2+ abs(FDv).^2 + 2);
IDv = FDvC ./(abs(FDh).^2+ abs(FDv).^2 + 2);


    
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% define a circular convolution (the same for all bands) accepting a 
% matrix  and returnig a matrix 

ConvC = @(X,FK)  reshape(real(ifft2(fft2(reshape(X', n_lin,n_col,n)).*repmat(FK,[1,1,n]))), n_lin*n_col,n)';

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% convert matrix to image
conv2im  = @(X)  reshape(X',n_lin,n_col,n);
% convert image to matrix
conv2mat = @(X)  reshape(X,n_lin,n_col,n)';

% simplex projection

proj_simplex_array = @(y) max(bsxfun(@minus,y,max(bsxfun(@rdivide,cumsum(sort(y,1,'descend'),1)-1,(1:size(y,1))'),[],1)),0); % projection on simplex


%%
%---------------------------------------------
% just least squares
%---------------------------------------------
if ~reg_TV && ~reg_l1 && ~reg_pos && ~reg_add
    U = pinv(M)*Y;
    res = norm(M*X-Y,'fro');
    return
end
%---------------------------------------------
% just ADDONE constrained (sum(x) = 1)
%---------------------------------------------
SMALL = 1e-12;
B = ones(1,n);
a = ones(1,N);

if  ~reg_TV && ~reg_l1 && ~reg_pos && reg_add
    F = M'*M;
    % test if F is invertible
    if rcond(F) > SMALL
        % compute the solution explicitly
        IF = inv(F);
        U = IF*M'*Y-IF*B'*inv(B*IF*B')*(B*IF*M'*Y-a);
        res = norm(M*U-Y,'fro');
        return
    end
    % if M'*M is singular, let sunsal_tv run
end



%%
%---------------------------------------------
%  Initializations and  constants
%---------------------------------------------

ATY = A'*Y;
IA = inv(A'*A + mu*eye(n));

% define and initialize variables
% All variables (U,V1,V2,V3,D1,D2,D3) will be represented as matrices as matrices

% no intial solution supplied
if U0 == 0
    U = IA*A'*Y;
else
    U = U0;
end
V1 = U;
D1 = U;
V2 = U;
D2 = U;
V3 = U;
D3 = U;
V4 = U;
D4 = U;



for i=1:AL_iters
%   min     ||U - V1 - D1||_F^2  
%    U    + ||U - V1 - D1||_F^2   
%         + ||UDh - V3 - D3||_F^2  
%         + ||UDv - V4 - D4||_F^2
%    
U =    ConvC(V1+D1, II) + ConvC(V2+D2, II) + ...  
       ConvC(V3+D3, IDh) +  ConvC(V4+D4, IDv); 


%  max (1/2)||AV1-Y|_F^2 + (mu/2)||U - V1 - D1||_F^2
%   V1
NU1 =  U - D1;
V1 = IA*(ATY + mu*NU1);


%  max lambda_1||X|_{1,1} + (mu/2)||U - V2 - D2||_F^2
%   V2
NU2 =  U - D2;
V2 = soft(NU2,lambda_l1/mu);
% posityvity
if reg_pos
%    V2 = max(0,V2); 
V2 = proj_simplex_array(V2);
end

% min lambda_vtv + (mu/2)||UDh - V3 - D4||_F^2 + (mu/2)||UDv - V4 - D4||_F^2
% V3,V4
NU3 =  ConvC(U,FDh) - D3;
NU4 =  ConvC(U,FDv) - D4;
if strcmp(tv_type,'iso')
   [V3,V4] = vector_soft_col_iso(NU3,NU4,lambda_VTV/mu);
else
   V3 = vector_soft_col(NU3,lambda_VTV/mu);
   V4 = vector_soft_col(NU4,lambda_VTV/mu);
end


if strcmp(verbose,'yes')
fprintf('iter = %d, ||U-V1|| = %2.2f, ||U-V2|| = %2.2f, ||UDh-V3|| = %2.2f, ||UDv-V4|| = %2.2f\n', ...
        i,  norm(NU1+D1-V1, 'fro'), norm(NU2+D2-V2, 'fro'), norm(NU3+D3-V3, 'fro'), norm(NU4+D4-V4, 'fro') )
end
    
 % update Lagrange multipliers
 D1 = -NU1 + V1;    % D1 -(UB-V1)
 D2 = -NU2 + V2;    % D2 -(UDH-V2)
 D3 = -NU3 + V3;    % D3 -(UDV-V3)
 D4 = -NU4 + V4;    % D3 -(UDV-V3)
 
end

if reg_pos
   U = max(0,U); 
end









% % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % % %
