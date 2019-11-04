
function err = RMSE(x, xhat)
[m, ~] = size(x);
err = mean(sqrt(1/m*sum((x-xhat).^2,1)));
end