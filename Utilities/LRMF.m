function [X] = LRMF(Y,W,opts)
% LRMF with n=2 for matrix completion, solving the following optimization problem:
% min_{X1,X2}  lambda*(||X1||_L^1 + ||X2||_L^1) + ||W.*(Y-X1*X2)||_F^2
% ---------------------------------------------
% Input:
%   Y:             m1*m2 matrix
%   W:             binary mask to indicate the locations of available entries.
%   opts.lambda:   regularization parameter (lambda>0)
%   opts.rank:     rank parameter of matrices X1,X2 and X (1<=rank<=min{m1,m2})
%   opts.epsilon:  constant in the definition of matrix logarithmic norm (epsilon>0)
%   opts.eta:      reduction scale of the Lipschitz constant (0<eta<=1, and should be large enough to make the algorithm stable.)
%   opts.max_iter: maximum number of iterations
%   opts.tol:      termination tolerance
% ---------------------------------------------
% Output:
%   X:             m1*m2 matrix, i.e., X=X1*X2
% ---------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%

if isfield(opts,'lambda');    lambda = opts.lambda;       else lambda = 3e6;    end
if isfield(opts,'rank');      rank = opts.rank;           else rank = 25;       end 
if isfield(opts,'epsilon');   epsilon = opts.epsilon;     else epsilon = 1e3;   end  
if isfield(opts,'eta');       eta = opts.eta;             else eta = 1;         end 
if isfield(opts,'max_iter');  max_iter = opts.max_iter;   else max_iter = 500;  end   
if isfield(opts,'tol');       tol = opts.tol;             else tol = 1e-3;      end  

[m1,m2] = size(Y); 

% Initialization
X1 = randn(m1,rank);  X1_old = X1;
X2 = randn(rank,m2);  X2_old = X2; 
mu = ones(1,2);       mu_old = mu; 
mu_min = 1e-3;
t_old = 1;

% Iteration
for i = 1 : max_iter  
  if mod(i,3) == 1                                  
    t = (1 + sqrt(1 + 4 * t_old^2)) / 2;
    temp = (t_old - 1)/t;   
    t_old = t; 
  end
   
  % Update X1
  w = min(temp,sqrt(mu_old(1) / mu(1)));
  X1_extrapol = X1 + w * (X1 - X1_old);  
  X1_old = X1;
  mu_old(1) = mu(1);
  mu(1) = eta * (max(norm(X2,2),mu_min))^2;
  A1 = X1_extrapol - W .* (X1_extrapol * X2 - Y) * X2' / mu(1);
  X1 = proximal_log(A1,lambda / mu(1),epsilon);
  tols(1) =  norm(X1 - X1_old,'fro') / norm(X1_old,'fro');           
   
  % Update X2
  w = min(temp,sqrt(mu_old(2) / mu(2)));
  X2_extrapol = X2 + w * (X2 - X2_old);    
  X2_old = X2;
  mu_old(2) = mu(2); 
  mu(2) = eta * (max(norm(X1,2),mu_min))^2;
  A2 = X2_extrapol - X1' * (W .* (X1 * X2_extrapol - Y)) / mu(2);
  X2 = proximal_log(A2,lambda / mu(2),epsilon);
  tols(2) = norm(X2 - X2_old,'fro') / norm(X2_old,'fro');   
    
  % Stop if the termination condition is met
  if mean(tols) < tol
    fprintf('Iteration number of LRMF is %d\n',i);
    break;
  end
end

X = X1 * X2;

end
    
