clc;
clear;
close all;
addpath(genpath('test_image'));
addpath(genpath('Utilities'));

I=double(imread('1.jpg'));   

%% Scenario generation 
[m1,m2,m3] = size(I); 
sampling_ratio = 0.15;
omega = find(rand(m1 * m2 * m3,1) < sampling_ratio); % locations of the available entries.
W = zeros(m1,m2,m3);                                 % mask
Y = zeros(m1,m2,m3);                                 % incomplete image
W(omega) = 1;
Y(omega) = I(omega);

%% Appropriate parameter adjustment can yield a better PSNR result.  
opts.lambda = 1e6;
opts.epsilon = 1e3;
opts.eta = 0.2;      
opts.max_iter = 500;            
opts.tol = 1e-3;
opts.rank = 20;

%% Image inpainting
for i = 1 : m3
  X1(:,:,i) = LRMF(Y(:,:,i), W(:,:,i), opts);
end
psnr1= PSNR(I,X1,double(~W));
fprintf('PSNR achieved by LRMF is %d dB\n',psnr1);
figure(1);
imshow(uint8(X1),[]);

%% Appropriate parameter adjustment can yield a better PSNR result.
opts.lambda = 5e6;
opts.rank = 25;   
opts.epsilon = 3e3;
opts.eta = 0.18;                 
opts.max_iter = 500;               
opts.tol = 1e-3; 
opts.trunc = 0.3;

%% Image inpainting
X2 = LRTF(Y,W,opts);
psnr2 = PSNR(I,X2,double(~W));
fprintf('PSNR achieved by LRTF is %d dB\n',psnr2);
figure(2);
imshow(uint8(X2),[]);
