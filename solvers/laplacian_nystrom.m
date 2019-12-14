function [V, L] = laplacian_nystrom(Xt, metric, num_samples, sigma2, seed)
% -------------------------------------------------------------------------
% 
% Input: 
% Xt: matrix transpose (#pixels x #wavelengths)
% metric: Distance metric to use
%   1 for Euclidean squared distance
%   2 for acos similarity
%   3 for cos similarity
% num_samples: number of eigenvalues/vectors, also the number of random
% samples
% sigma2: variance constant used in computing similarity
% Output: 
% V: eigenvectors of size #pixels x #samples
% L: eigenvalues of Laplacian #samples x #samples
%
% edited by Jing for the WiSDM group, 7-29-2019
% -------------------------------------------------------------------------
if nargin >4
    rng(seed); % for reproducibility
end

% randomly select samples
num_rows = size(Xt, 1);
permed_index = randperm(num_rows);
sample_data = Xt(permed_index(1:num_samples), :);
other_data = Xt(permed_index(num_samples+1:num_rows), :);
clear Xt;

% plot the sampled pixels
% s = permed_index(1:num_samples);
% mask = ismember(1:num_rows, s);
%figure; imshow(reshape(mask, 307,307), [],'border','tight');

% Calculate the distance between samples themselves
A = zeros(num_samples, num_samples);
for i = 1:num_samples
    for j = i:num_samples
        
        x = sample_data(i,:);
        y = sample_data(j,:);
        
        if (metric == 1)
            d = sum((x-y).^2);
        elseif (metric == 2)
            d = acos((x * y')/(norm(x)*norm(y)));
        elseif (metric == 3)
            d = 1 - (x * y')/(norm(x)*norm(y));
        end
        
        A(i,j) = exp(-d/sigma2);
        A(j,i) = A(i,j);
        
    end
end
%A = single(A);

% calculate the euclidean distance between samples and other points
other_points = num_rows - num_samples;
B = zeros(num_samples, other_points);
for i = 1:num_samples
    for j = 1:other_points
        
        x = sample_data(i,:);
        y = other_data(j,:);
        
        if (metric == 1)
            d = sum((x-y).^2);
        elseif (metric == 2)
            d = acos((x * y')/(norm(x)*norm(y)));
        elseif (metric == 3)
            d = 1 - (x * y')/(norm(x)*norm(y));
        end
        
        B(i,j) = exp(-d/sigma2);
    end
end
%B = single(B);
clear sample_data other_data;


% Normalize A and B using row sums of W, where W = [A B; B' B'*A^-1*B].
% Let d1 = [A B]*1, d2 = [B' B'*A^-1*B]*1, dhat = sqrt(1./[d1; d2]).
B_T = B';
d1 = sum(A, 2) + sum(B, 2);
d2 = sum(B_T, 2) + B_T*(pinv(A)*sum(B, 2));
dhat = sqrt(1./[d1; d2]);
A = A .* (dhat(1:num_samples)*dhat(1:num_samples)');
B1 = dhat(1:num_samples)*dhat(num_samples+(1:other_points))';
B = B .* B1;
clear d1 d2 B1 dhat;

% Do orthogalization and eigendecomposition
Asi = sqrtm(pinv(A));
B_T = B';
BBT = B*B_T;
%W = single(zeros(size(A, 1)+size(B_T, 1), size(A, 2)));
W = zeros(size(A, 1)+size(B_T, 1), size(A, 2));
W(1:size(A, 1), :) = A;
W(size(A, 1)+1:size(W, 1), :) = B_T;
clear B B_T;
% Calculate R = A + A^-1/2*B*B'*A^-1/2
R = A + Asi*BBT*Asi;
R = (R + R')/2; % symmetrize R
[U, L] = eig(R);
[~, ind] = sort(diag(L), 'descend');
U = U(:, ind); % in decreasing order
L = L(ind, ind); % in decreasing order
clear A R BBT;
W = W*Asi;
V = W*U(:, 1:num_samples)*pinv(sqrt(L(1:num_samples, 1:num_samples)));

V(permed_index,:) = V;
V = real(V);
L = 1-diag(L);
