function [A_hat, nmse] = find_perm2(A_true, A_hat)
P = size(A_true,1);
ords = perms(1:P);
n = size(ords,1);
errs = 100*ones(n,1);

for idx = 1:n
    A = A_hat(ords(idx,:),:);
    errs(idx) = nMSE(A_true, A);
end

[nmse, I] = min(errs);
A_hat = A_hat(ords(I,:), :);
end