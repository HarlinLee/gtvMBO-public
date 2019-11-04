function err = SAM(X, Xhat)
[L, N] = size(X);
errs = zeros(N,1);

for idx = 1:N
    a1 = X(:,idx);
    a2 = Xhat(:,idx);
    errs(idx) = getRadErr(a1,a2);
end

err = mean(mean(errs(~isnan(errs))));

    function raderr = getRadErr(x,s)
%         if norm(x) < 10^(-6) || norm(s) < 10^(-6)
%             x = x + 10^(-3);
%         end
%         
        raderr = 180/pi*real(acos((x'*s)/(norm(x)*norm(s))));
    end
end