function err = nMSE(x, xhat)
err = norm(x-xhat, 'fro')/norm(x, 'fro');
end