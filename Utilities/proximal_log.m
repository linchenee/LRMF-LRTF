function [X] = proximal_log(A,alpha,epsilon)
% The proximal operator of the matrix logarithmic norm, solving the following optimization problem:
% min_{X}  0.5*||X-A||_F^2 + alpha*||X||_L^1
% ---------------------------------------------
% Input:
%   A:        m1*m2 matrix
%   alpha:    penalty parameter (alpha>0)
%   epsilon:  constant in the definition of matrix logarithmic norm (epsilon>0)
% ---------------------------------------------
% Output:
%   X:        m1*m2 matrix
% ---------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%
  
[u,s1,v] = svd(A,'econ');
s2 = diag(s1);
s = zeros(size(s2));

temp = s2 > (2 * sqrt(alpha) - epsilon);  % judge whether $\Delta$>0
s3 = s2(temp);
root = 0.5 .* (s3 - epsilon + sqrt((s3 + epsilon).^2 - 4 * alpha)); % the larger root of the quadratic equation
s(temp) = (root > 0 & root .* (0.5 * root - s3) + alpha * log(1 + root / epsilon)  <0) .* root; % determine non-zero singular values

X = u * diag(s) * v';

end