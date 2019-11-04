% =========================================================================
% Title: Code for running synthetic data experiments for journal version
%       Ummixing version
% Author: JTC (Adapted from Harlin's code for Urban experiments)
% =========================================================================

%%  Prep workspacce

close all;
clear;
clc;

seed = 1;
rng(seed);

addpath(genpath('./'));
resultsFolder = 'results';
fname = fullfile(resultsFolder,'synthetic_output.mat')

%% Load synthetic grayscale grid data

% Parameters for making data using Linda's code
m = 50;
n = 50;
N = m * n;
k = 144; % spectral dimension
nEndmembers = 4;
load('SyntheticData_grayscalegrids.mat');

%% Experiments

% Rename things to match Harlin's conventions
X = Xdata;
A_true = Atruth;
S_true = Struth;

% Set parameters
P = nEndmembers; 
tol = 1e-6;

% Initialize results container
errs = zeros(13, 7);

%% FCLSU
c = 1;

% Get S using VCA
seed = 12345;
bundle_nbr = 10; % number of VCA runs
percent = 10; % percentage of pixels considered in each run
[groups, bundle] = batchvca(X, P, bundle_nbr, percent, seed); % extract endmembers and cluster them into groups

tic
disp('FCLSU bundle')
A_FCLSU = FCLSU(X, bundle)'; % abundance is initialized as A_FCLSU_bundle for all other algorithms
t = toc;

[A_init,~] = bundle2global(A_FCLSU,bundle,groups);
S_init = [];
for idx = 1:P
    s = mean(bundle(:, groups == idx),2);
    S_init = [S_init, s];
end
S_init = max(S_init,0);
[A_init, S_init, nmse] = find_perm(A_true, A_init, S_init);

% Save results
save(fname, 'A_FCLSU', 'A_init', 'S_init');

errs(1, c) = RMSE(X, S_init*A_init);
errs(2, c) = nMSE(X, S_init*A_init);
errs(3, c) = RMSE(S_true, S_init);
errs(4, c) = nMSE(S_true, S_init);
errs(5, c) = SAM(S_true, S_init);
errs(6, c) = RMSE(A_true, A_init);
errs(7, c) = nMSE(A_true, A_init);
errs(8, c) = Inf;
errs(9, c) = t;
errs(10, c) = Inf;
errs(11, c) = Inf;
errs(12, c) = Inf;
errs(13, c) = Inf;

%% Fractional
c = c + 1;
disp('fractional')

para_fr.method = 'fractional';
para_fr.fraction = 1/10;
para_fr.itermax = 300;
para_fr.plot_flag = 0;
para_fr.lambda = 10^(-5.5);
para_fr.rho = 10;
para_fr.tol = tol;

tic;
[A_frac, optim_struct] = social_unmixing(X,bundle,groups,A_FCLSU, ...
        para_fr.lambda,para_fr.rho,para_fr.itermax,...
        para_fr.method,para_fr.fraction,para_fr.tol,0);
t_frac = toc;

para_fr.iter = optim_struct.iter;

A_frac = get_A(A_frac, groups, P);
    
[A_frac, ~] = find_perm2(A_true, A_frac);
    
% (Note: No S obtained via this method)
errs(1, c) = RMSE(X, S_init*A_frac);
errs(2, c) = nMSE(X, S_init*A_frac);
errs(3, c) = Inf;
errs(4, c) = Inf;
errs(5, c) = Inf;
errs(6, c) = RMSE(A_true, A_frac);
errs(7, c) = nMSE(A_true, A_frac);
errs(8, c) = Inf;
errs(9, c) = t_frac;
errs(10, c) = para_fr.lambda; 
errs(11, c) = para_fr.rho; 
errs(12, c) = Inf;
errs(13, c) = para_fr.iter;

save(fname, '-append', 'A_frac', 'para_fr', 'errs');

%% 2dTV
c = c+1;
disp('2dTV');
para_tv.maxiter = 1000;
para_tv.m = m;
para_tv.n = n;
para_tv.lambda = 0.1;

tic;
A_TV = sunsal_vtv(S_init,X,'LAMBDA_1',0, 'LAMBDA_VTV', para_tv.lambda, 'X0', A_init, 'IM_SIZE', [para_tv.m,para_tv.n],  'POSITIVITY', 'yes', 'ADDONE', 'no', 'VERBOSE', 'no','AL_ITERS', para_tv.maxiter); % do not pay attention to the "'ADDONE', 'no'" bit, the sum to one constraint is actually enforced.
t_2d = toc;

para_tv.iter = para_tv.maxiter;

[A_TV, ~] = find_perm2(A_true, A_TV);

errs(1, c) = RMSE(X, S_init*A_TV);
errs(2, c) = nMSE(X, S_init*A_TV);
errs(3, c) = Inf;
errs(4, c) = Inf;
errs(5, c) = Inf;
errs(6, c) = RMSE(A_true, A_TV);
errs(7, c) = nMSE(A_true, A_TV);
errs(8, c) = Inf;
errs(9, c) = t_2d;
errs(10, c) = para_tv.lambda; 
errs(11, c) = Inf; 
errs(12, c) = Inf;
errs(13, c) = para_tv.iter;

save(fname, '-append', 'A_TV', 'para_tv');

%% GLNMF

c = c+1;
k = P;
sigma = 5;

disp('calculating adj matrix');
tic % urban images are too large to make an NxN arary.
iis = zeros(k*N,1);
jjs = zeros(k*N,1);
vvs = zeros(k*N,1);
for col = 1:N
    w = pdist2(X(:, col)', X','squaredeuclidean');
    [~, I] = sort(w,'ascend'); 
    iis(k*(col-1)+1:k*col) = I(2:k+1); % no self loops 
    jjs(k*(col-1)+1:k*col) = col*ones(k,1);
    vvs(k*(col-1)+1:k*col) = exp(-w(I(2:k+1))/sigma);
end
para_nmf.W = sparse(iis, jjs, vvs, N, N);
para_nmf.d = sum(para_nmf.W,2);
t_graph = toc;

disp('glnmf');

para_nmf.method = 'glnmf';
para_nmf.itermax = 1000;
para_nmf.plot_flag = 0;
para_nmf.tol = 10^(-1);
para_nmf.itermax = 1000;
para_nmf.lambda = 10^(-5.5);
para_nmf.mu = 10^(-5.5);

tic;
[iter, S_nmf, A_nmf]= glnmf(X, P, S_init, A_init, para_nmf.W, para_nmf.d, para_nmf);
t_nmf = toc;
para_nmf.iter = iter;

[A_nmf, S_nmf, err] = find_perm(A_true, A_nmf, S_nmf);

errs(1, c) = RMSE(X, S_nmf*A_nmf);
errs(2, c) = nMSE(X, S_nmf*A_nmf);
errs(3, c) = RMSE(S_true, S_nmf);
errs(4, c) = nMSE(S_true, S_nmf);
errs(5, c) = SAM(S_true, S_nmf);
errs(6, c) = RMSE(A_true, A_nmf);
errs(7, c) = nMSE(A_true, A_nmf);
errs(8, c) = t_graph;
errs(9, c) = t_nmf;
errs(10, c) = para_nmf.lambda;
errs(11, c) = Inf;
errs(12, c) = para_nmf.mu; %gamma
errs(13, c) = para_nmf.iter;

save(fname, '-append', 'A_nmf', 'S_nmf', 'para_nmf');

%% Graph Laplacian
c = c + 1;

% Nystrom extension for graphL
disp('Nystrom extension');
sample_size = floor(0.01*N);
sigma2 = 1; 
seed = 12345;

tic
[V_gL, Sigma_gL] = laplacian_nystrom(X', 3, sample_size, sigma2, seed);
t_nystromGL = toc;

save(fname, '-append', 'V_gL', 'Sigma_gL', 'sigma');

para_gL.method = 'graphL';
para_gL.tol = tol;
para_gL.m = m;
para_gL.n = n;
para_gL.V = V_gL;
para_gL.S = Sigma_gL;
para_gL.itermax = 10;
para_gL.lambda = 0.001;
para_gL.rho = 10^(-4.5);
para_gL.gamma = 100;

disp('graph Laplacian');
tic;
[S_graphL, A_graphL, iter] = unmixing(X, S_init, A_init, para_gL);
t_gL = toc;
para_gL.iter = iter;

[A_graphL, S_graphL, ~] = find_perm(A_true, A_graphL, S_graphL);

errs(1, c) = RMSE(X, S_graphL*A_graphL);
errs(2, c) = nMSE(X, S_graphL*A_graphL);
errs(3, c) = RMSE(S_true, S_graphL);
errs(4, c) = nMSE(S_true, S_graphL);
errs(5, c) = SAM(S_true, S_graphL);
errs(6, c) = RMSE(A_true, A_graphL);
errs(7, c) = nMSE(A_true, A_graphL);
errs(8, c) = t_nystromGL;
errs(9, c) = t_gL;
errs(10, c) = para_gL.lambda;
errs(11, c) = para_gL.rho;
errs(12, c) = para_gL.gamma;
errs(13, c) = para_gL.iter;

save(fname, '-append', 'A_graphL', 'S_graphL', 'errs', 'para_gL');

%% MBO
c = c + 1;

% Nystrom extension for MBO
disp('Nystrom extension');
sample_size = floor(0.01*N);
sigma2 = 5; 
seed = 12345;

tic
[V, Sigma] = laplacian_nystrom(X', 3, sample_size, sigma2, seed);
t_nystrom = toc;

save(fname, '-append', 'V', 'Sigma', 'sigma2');

para_mbo.method = 'gtvMBO';
para_mbo.tol = tol;
para_mbo.m = m;
para_mbo.n = n;
para_mbo.V = V;
para_mbo.S = Sigma;
para_mbo.itermax = 10;
para_mbo.dt = 0.01;
para_mbo.lambda = 10^(-6.5);
para_mbo.rho = 10^(-4.25);
para_mbo.gamma = 10^(1.5);

disp('gtvMBO');

tic;
[S_MBO, A_MBO, iter] = unmixing(X, S_init, A_init, para_mbo);
t_MBO = toc;
para_mbo.iter = iter;

[A_MBO, S_MBO, ~] = find_perm(A_true, A_MBO, S_MBO);

errs(1, c) = RMSE(X, S_MBO*A_MBO);
errs(2, c) = nMSE(X, S_MBO*A_MBO);
errs(3, c) = RMSE(S_true, S_MBO);
errs(4, c) = nMSE(S_true, S_MBO);
errs(5, c) = SAM(S_true, S_MBO);
errs(6, c) = RMSE(A_true, A_MBO);
errs(7, c) = nMSE(A_true, A_MBO);
errs(8, c) = t_nystrom;
errs(9, c) = t_MBO;
errs(10, c) = para_mbo.lambda; 
errs(11, c) = para_mbo.rho; 
errs(12, c) = para_mbo.gamma;
errs(13, c) = para_mbo.iter;

clear para_mbo.m para_mbo.n para_mbo.V para_mbo.S t_MBO_fixed
save(fname, '-append', 'A_MBO', 'S_MBO', 'errs','para_mbo');

%% MBO fixed ratio
c = c+1;

para_mbo_fixed.method = 'gtvMBO';
para_mbo_fixed.tol = tol;

para_mbo_fixed.m = m;
para_mbo_fixed.n = n;

para_mbo_fixed.V = V;
para_mbo_fixed.S = Sigma;

para_mbo_fixed.dt = 0.01;
para_mbo_fixed.itermax = 10;
para_mbo_fixed.lambda = 10^(-5.5);
para_mbo_fixed.rho = para_mbo_fixed.lambda;
para_mbo_fixed.gamma = para_mbo_fixed.lambda*10^7;

disp('gtvMBO fixed ratio');

tic;
[S_MBO_fixed, A_MBO_fixed, iter] = unmixing(X, S_init, A_init, para_mbo_fixed);
t_MBO_fixed = toc;
para_mbo_fixed.iter = iter;

[A_MBO_fixed, S_MBO_fixed, ~] = find_perm(A_true, A_MBO_fixed, S_MBO_fixed);

errs(1, c) = RMSE(X, S_MBO_fixed*A_MBO_fixed);
errs(2, c) = nMSE(X, S_MBO_fixed*A_MBO_fixed);
errs(3, c) = RMSE(S_true, S_MBO_fixed);
errs(4, c) = nMSE(S_true, S_MBO_fixed);
errs(5, c) = SAM(S_true, S_MBO_fixed);
errs(6, c) = RMSE(A_true, A_MBO_fixed);
errs(7, c) = nMSE(A_true, A_MBO_fixed);
errs(8, c) = t_nystrom;
errs(9, c) = t_MBO_fixed;
errs(10, c) = para_mbo_fixed.lambda; 
errs(11, c) = para_mbo_fixed.rho; 
errs(12, c) = para_mbo_fixed.gamma;
errs(13, c) = para_mbo_fixed.iter;

clear para_mbo_fixed.m para_mbo_fixed.n para_mbo_fixed.V para_mbo_fixed.S t_MBO_fixed
save(fname, '-append', 'A_MBO_fixed', 'S_MBO_fixed', 'errs','para_mbo_fixed');

%% MBO post-processed using 2dTV
c = c+1;

para_MBO2.method = '2dTV';
para_MBO2.itermax = 10;
para_MBO2.m = m;
para_MBO2.n = n;
para_MBO2.lambda = 10^(-2.75);
para_MBO2.iter = para_MBO2.itermax;

tic;
A_MBO2 = sunsal_vtv(S_MBO,X,'LAMBDA_1',0, 'LAMBDA_VTV', para_MBO2.lambda, 'X0', A_MBO, 'IM_SIZE', [para_MBO2.m,para_MBO2.n],  'POSITIVITY', 'yes', 'ADDONE', 'no', 'VERBOSE', 'no','AL_ITERS', para_MBO2.itermax); % do not pay attention to the "'ADDONE', 'no'" bit, the sum to one constraint is actually enforced.
t_MBO2 = toc;

[A_MBO2, ~] = find_perm2(A_true, A_MBO2);


errs(1, c) = RMSE(X, S_MBO*A_MBO2);
errs(2, c) = nMSE(X, S_MBO*A_MBO2);
errs(3, c) = RMSE(S_true, S_MBO);
errs(4, c) = nMSE(S_true, S_MBO);
errs(5, c) = SAM(S_true, S_MBO);
errs(6, c) = RMSE(A_true, A_MBO2);
errs(7, c) = nMSE(A_true, A_MBO2);
errs(8, c) = t_nystrom;
errs(9, c) = t_MBO2;
errs(10, c) = para_MBO2.lambda; 
errs(11, c) = Inf; 
errs(12, c) = Inf;
errs(13, c) = para_MBO2.iter;

save(fname, '-append', 'A_MBO2', 'para_MBO2', 'errs');


%% Plot results

% Look at error table
format longG;
errs

% Plot results for A
fig = figure;
[ha, ~] = tight_subplot(9, P, [.03 .03], [.03 .03], [0 0]);
for i = 1:P
    axes(ha(i));
    imshow(reshape(A_true(i,:), m,n), []);axis off;colormap gray
    title('Truth')
end
for i = 1:P
    axes(ha(i+P));
    imshow(reshape(A_init(i,:), m,n), []); axis off;
    title('FCLSU')
end
for i = 1:P
    axes(ha(i+2*P));
    imshow(reshape(A_frac(i,:),m,n),[]); axis off;
    title('fractional')
end
for i = 1:P
    axes(ha(i+3*P));
    imshow(reshape(A_TV(i,:),m,n),[]); axis off;
    title('2dTV')
end
for i = 1:P
    axes(ha(i+4*P));
    imshow(reshape(A_nmf(i,:),m,n),[]); axis off;
    title('GLNMF')
end
for i = 1:P
    axes(ha(i+5*P));
    imshow(reshape(A_graphL(i,:),m,n),[]); axis off;
    title('graphL')
end
for i = 1:P
    axes(ha(i+6*P));
    imshow(reshape(A_MBO(i,:),m,n),[]);axis off;
    title('gtvMBO')
end
for i = 1:P
    axes(ha(i+7*P));
    imshow(reshape(A_MBO(i,:),m,n),[]);axis off;
    title('gtvMBO fixed ratio')
end
for i = 1:P
    axes(ha(i+8*P));
    imshow(reshape(A_MBO2(i,:),m,n),[]);axis off;
    title('gtvMBO + 2dTV')
end
saveas(fig, fullfile(resultsFolder,'syntheticResultsA.jpg'));

% Save A plots for journal version (by row)
for i = 1:P
    f = figure('visible','off'); imshow(reshape(A_true(i,:), m,n), []);
    pngFileName = sprintf('truth%d.png', i);
    fullFileName = fullfile(resultsFolder, pngFileName);
    export_fig(fullFileName);
    
	f = figure('visible','off'); imshow(reshape(A_init(i,:), m,n), []); 
	pngFileName = sprintf('fclsu%d.png', i);
    fullFileName = fullfile(resultsFolder, pngFileName);
    export_fig(fullFileName);   
    
    f = figure('visible','off'); imshow(reshape(A_frac(i,:), m,n), []); 
    pngFileName = sprintf('frac%d.png', i);
    fullFileName = fullfile(resultsFolder, pngFileName);
    export_fig(fullFileName);
     
    f = figure('visible','off'); imshow(reshape(A_TV(i,:), m,n), []); 
    pngFileName = sprintf('tv%d.png', i);
    fullFileName = fullfile(resultsFolder, pngFileName);
    export_fig(fullFileName);
    
    f = figure('visible','off'); imshow(reshape(A_nmf(i,:), m,n), []); 
    pngFileName = sprintf('glnmf%d.png', i);
    fullFileName = fullfile(resultsFolder, pngFileName);
    export_fig(fullFileName);
     
    f = figure('visible','off'); imshow(reshape(A_graphL(i,:), m,n), []);
    pngFileName = sprintf('graphl%d.png', i);
    fullFileName = fullfile(resultsFolder, pngFileName);
    export_fig(fullFileName);
    
    f = figure('visible','off'); imshow(reshape(A_MBO(i,:), m,n), []); 
    pngFileName = sprintf('mbo%d.png', i);
    fullFileName = fullfile(resultsFolder, pngFileName);
    export_fig(fullFileName);
    
    f = figure('visible','off'); imshow(reshape(A_MBO_fixed(i,:), m,n), []); 
    pngFileName = sprintf('mbo_fixed%d.png', i);
    fullFileName = fullfile(resultsFolder, pngFileName);
    export_fig(fullFileName);
    
    f = figure('visible','off'); imshow(reshape(A_MBO2(i,:), m,n), []); 
    pngFileName = sprintf('mbotv%d.png', i);
    fullFileName = fullfile(resultsFolder, pngFileName);
    export_fig(fullFileName);
end

% Plot results for S (FCLSU and fractional don't produce S)
figout3 = figure;
subplot(2,3,1); plot(S_true); title('Truth');
subplot(2,3,2); plot(S_init); title('VCA');
subplot(2,3,3); plot(S_nmf); title('GLNMF');
subplot(2,3,4); plot(S_graphL); title('GraphL');
subplot(2,3,5); plot(S_MBO); title('gtvMBO');
subplot(2,3,6); plot(S_MBO_fixed); title('gtvMBO fixed ratio');

export_fig(fullfile(resultsFolder,'syntheticResultsS.png'));
export_fig(fullfile(resultsFolder,'syntheticResultsS_transparent.png'), '-transparent');

% Save S plots for journal version

f = figure('visible','off'); plot(S_true, 'LineWidth', 2); 
axis square;
axis([0 150 0 0.45])
set(gca,'FontSize',24)
pngFileName = 'S_true.png';
fullFileName = fullfile(resultsFolder, pngFileName);
export_fig(fullFileName, '-transparent');

f = figure('visible','off'); plot(S_init, 'LineWidth', 2); 
axis square;
axis([0 150 0 0.45])
set(gca,'FontSize',24)
pngFileName = 'S_vca.png';
fullFileName = fullfile(resultsFolder, pngFileName);
export_fig(fullFileName, '-transparent');

f = figure('visible','off'); plot(S_nmf, 'LineWidth', 2); 
axis square;
axis([0 150 0 0.45])
set(gca,'FontSize',24)
pngFileName = 'S_nmf.png';
fullFileName = fullfile(resultsFolder, pngFileName);
export_fig(fullFileName, '-transparent');

f = figure('visible','off'); plot(S_graphL, 'LineWidth', 2); 
axis square;
axis([0 150 0 0.45])
set(gca,'FontSize',24)
pngFileName = 'S_graphL.png';
fullFileName = fullfile(resultsFolder, pngFileName);
export_fig(fullFileName, '-transparent');

f = figure('visible','off'); plot(S_MBO, 'LineWidth', 2); 
lgd = legend('Parking','Roofs','Chairs','Vegetation','Location', 'eastoutside');
lgd.FontSize = 36;
legend('boxoff')     
axis square;
axis([0 150 0 0.45])
set(gca,'FontSize',20)
pngFileName = 'S_mbo.png';
fullFileName = fullfile(resultsFolder, pngFileName);
export_fig(fullFileName, '-transparent');

% Save laTeX table
rowNames = {'RMSE$(X, \hat{S}\hat{A})$','nMSE$(X, \hat{S}\hat{A})$','RMSE$(S, \hat{S})$','nMSE$(S, \hat{S})$','SAM$(S, \hat{S})$','RMSE$(A, \hat{A})$','nMSE$(A, \hat{A})$', 'Graph time (sec)', 'time (sec)','$\lambda$','$\rho$','$\gamma$', 'Iterations'};
colNames = {'FCLSU','FRAC','TV','GLNMF','GraphL','gtvMBO','gtvMBOfixed','gtvMBO2dTV'};
errsTable = array2table(errs(1:13,:),'RowNames',rowNames,'VariableNames',colNames)
table2latexfancy(errsTable, fullfile(resultsFolder,'errsTable.tex'));

