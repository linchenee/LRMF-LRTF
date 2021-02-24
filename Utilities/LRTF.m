function [X] = LRTF(Y,W,opts)
% LRTF with n=2 for tensor completion, solving the following optimization problem:
% min_{X1,X2}  lambda*(||X1||_L^1 + ||X2||_L^1) + ||W.*(Y-X1*X2)||_F^2
% ---------------------------------------------
% Input:
%   Y:             m1*m2*m3 tensor
%   W:             binary tensor to indicate the locations of available entries.
%   opts.lambda:   regularization parameter (lambda>0)
%   opts.rank:     tubal rank of tensors X1,X2 and X (1<=rank<=min{m1,m2})
%   opts.epsilon:  constant in the definition of tensor logarithmic norm (epsilon>0)
%   opts.eta:      reduction scale of the Lipschitz constant (0<eta<=1, and should be large enough to make the algorithm stable.)
%   opts.max_iter: maximum number of iterations
%   opts.tol:      termination tolerance
%   opts.trunc:    truncation scale of the rank of the tensor's frontal slice in initialization (0<trunc<=1)
% ---------------------------------------------
% Output:
%   X:             m1*m2*m3 tensor, i.e., X=X1*X2
% ---------------------------------------------
% Written by Lin Chen (linchenee@sjtu.edu.cn)
%

if isfield(opts,'lambda');    lambda = opts.lambda;      else lambda = 3e6;   end
if isfield(opts,'rank');      rank = opts.rank;          else rank = 60;      end
if isfield(opts,'epsilon');   epsilon = opts.epsilon;    else epsilon = 1e3;  end 
if isfield(opts,'eta');       eta = opts.eta;            else eta = 1;        end 
if isfield(opts,'max_iter');  max_iter = opts.max_iter;  else max_iter = 500; end   
if isfield(opts,'tol');       tol = opts.tol;            else tol = 1e-3;     end      
if isfield(opts,'trunc');     trunc = opts.trunc;        else trunc = 1;      end

[m1,m2,m3] = size(Y);
half = round(m3/2);
even = (mod(m3,2) == 0);

%% Initialization 
X1 = randn(m1,rank,m3);  X1f = fft(X1,[],3);  
X2 = randn(rank,m2,m3);  X2f = fft(X2,[],3);
if trunc ~= 1
  %% Truncating the ranks of frontal slices (2~m3), but don't change the value of the tensor's tubal rank
  for i = 2 : m3
    [u1,s1,v1] = svds(X1f(:,:,i),floor(trunc * rank));
    X1f(:,:,i) = u1 * s1 * v1';
    [u2,s2,v2] = svds(X2f(:,:,i),floor(trunc * rank));
    X2f(:,:,i) = u2 * s2 * v2';
  end
end

X1f_old = X1f;
X2f_old = X2f;
mu = ones(1,2); 
mu_old = mu; 
mu_min = 1e-2;
t_old = 1;
   
%% Iteration
for i = 1 : max_iter  
%  if mod(i,10) == 1     
  t = (1 + sqrt(1 + 4 * t_old^2)) / 2;
  temp = (t_old - 1) / t;   
  t_old = t; 
%  end
        
  %% Update X1f
  w = min(temp,sqrt(mu_old(1) / mu(1)));
  X1f_extrapol = X1f + w * (X1f - X1f_old); % extrapolated point
  X1f_old = X1f;  
  mu_old(1) = mu(1);
        
  Temp1(:,:,1) = X1f_extrapol(:,:,1) * X2f(:,:,1);
  Norm1(1) = norm(X2f(:,:,1),2);
  for j = 2 : half
    Temp1(:,:,j) = X1f_extrapol(:,:,j) * X2f(:,:,j); 
    Temp1(:,:,m3-j+2) = conj(Temp1(:,:,j));
    Norm1(j) = norm(X2f(:,:,j),2);
  end
  if even
    Temp1(:,:,half+1) = X1f_extrapol(:,:,half+1) * X2f(:,:,half+1);
    Norm1(half+1) = norm(X2f(:,:,half+1),2);
  end
        
  mu(1) = eta * (max(max(Norm1),mu_min))^2;
  Temp3 = fft(W .* (ifft(Temp1,[],3) - Y),[],3);
        
  A1 = X1f_extrapol(:,:,1) - Temp3(:,:,1) * (X2f(:,:,1))' ./ mu(1);
  X1f(:,:,1) = proximal_log(A1,lambda / mu(1),epsilon);
  for j = 2 : half  
    A1 = X1f_extrapol(:,:,j) - Temp3(:,:,j) * (X2f(:,:,j))' ./ mu(1);
    X1f(:,:,j) = proximal_log(A1,lambda / mu(1),epsilon);
    X1f(:,:,m3-j+2) = conj(X1f(:,:,j));
  end
  if even
    A1 = X1f_extrapol(:,:,half+1) - Temp3(:,:,half+1) * (X2f(:,:,half+1))' ./ mu(1);
    X1f(:,:,half+1) = proximal_log(A1,lambda / mu(1),epsilon);
  end
        
  tols(1) =  norm(X1f(:) - X1f_old(:),'fro') / norm(X1f_old(:),'fro');
        
  %% Update X2f
  w = min(temp,sqrt(mu_old(2) / mu(2)));
  X2f_extrapol = X2f + w * (X2f - X2f_old);  % extrapolated point
  X2f_old = X2f;  
  mu_old(2) = mu(2);
          
  Temp2(:,:,1) = X1f(:,:,1) * X2f_extrapol(:,:,1);
  Norm2(1) = norm(X1f(:,:,1),2);
  for j = 2 : half
    Temp2(:,:,j) = X1f(:,:,j) * X2f_extrapol(:,:,j);
    Temp2(:,:,m3-j+2) = conj(Temp2(:,:,j));
    Norm2(j) = norm(X1f(:,:,j),2);
  end
  if even
    Temp2(:,:,half+1) = X1f(:,:,half+1) * X2f_extrapol(:,:,half+1);
    Norm2(half+1) = norm(X1f(:,:,half+1),2);
  end
        
  mu(2) = eta * (max(max(Norm2),mu_min))^2;
  Temp4 = fft(W .* (ifft(Temp2,[],3) - Y),[],3);
        
  A2 = X2f_extrapol(:,:,1) - (X1f(:,:,1))' * Temp4(:,:,1) ./ mu(2);
  X2f(:,:,1) = proximal_log(A2,lambda / mu(2),epsilon);
  for j = 2 : half  
    A2 = X2f_extrapol(:,:,j) - (X1f(:,:,j))' * Temp4(:,:,j) ./ mu(2);
    X2f(:,:,j) = proximal_log(A2,lambda / mu(2),epsilon);
    X2f(:,:,m3-j+2) = conj(X2f(:,:,j));
  end
  if even
    A2 = X2f_extrapol(:,:,half+1) - (X1f(:,:,half+1))' * Temp4(:,:,half+1) ./ mu(2);
    X2f(:,:,half+1) = proximal_log(A2,lambda / mu(2),epsilon);
  end
       
  tols(2) = norm(X2f(:) - X2f_old(:),'fro') / norm(X2f_old(:),'fro');

  %% Stop if the termination condition is met
  if (mean(tols)) < tol
     fprintf('Iteration number of LRTF is %d\n',i);
     break;
  end
end
     
Xf(:,:,1) = X1f(:,:,1) * X2f(:,:,1);
for i = 2 : half
  Xf(:,:,i) = X1f(:,:,i) * X2f(:,:,i);
  Xf(:,:,m3-i+2) = conj(Xf(:,:,i));
end
if even
  Xf(:,:,half+1) = X1f(:,:,half+1) * X2f(:,:,half+1);
end

X = ifft(Xf,[],3);

end
