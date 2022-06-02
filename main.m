A = randi([0, 10], [1000, 1000], 'double');
B = randi([0, 10], [1000, 1000], 'double');
b = randi([0, 10], [1000, 1], 'double');

% f = @() mat_mul(A,B); 
% timeit(f)

tic
for i = 1:100
    mat_mul(A,B); 
end
toc

tic
for i = 1:10000
    mat_mul(A,b); 
end
toc

% f = @() dot_product(A,B); 
% timeit(f)

tic
for i = 1:100000
    dot_product(A,B); 
end
toc

% f = @() element_wise_ops(A,B); 
% timeit(f)

tic
for i = 1:1000
    element_wise_ops(A,B); 
end
toc

% f = @() LU_decomp(A); 
% timeit(f)

tic
for i = 1:10
    LU_decomp(A); 
end
toc

% f = @() QR_decomp(A); 
% timeit(f)

tic
for i = 1:10
    QR_decomp(A); 
end
toc

% f = @() eig_decomp(A);
% timeit(f)

tic
for i = 1:10
    eig_decomp(A);
end
toc

function C = mat_mul(A,B)
    C = A * B;
end

function C = dot_product(A,B)
    C = dot(A,B);
end

function C = element_wise_ops(A,B)
    C = A .* B;
end

function [L, U] = LU_decomp(A)
    [L, U] = lu(A);
end

function [Q, R] = QR_decomp(A)
    [Q, R] = qr(A);
end

function [V,D] = eig_decomp(A)
    [V,D] = eig(A);
end