function [A1] = get_A(A, groups, P)
A1 = [];
for idx = 1:P
    a = sum(A(groups == idx,:), 1);
    A1 = [A1; a];
end
end
