clc;
clear;
close all;
addpath(genpath('test_video'));
addpath(genpath('Utilities'));

load('claire_qcif.mat');

% Scenario generation
[m1,m2,m3] = size(I);
sampling_ratio = 0.1;
omega = find(rand(m1 * m2 * m3,1) < sampling_ratio); % locations of the available entries.
W = zeros(m1,m2,m3);                                 % mask
Y = zeros(m1,m2,m3);                                 % incomplete image
W(omega) = 1;
Y(omega) = I(omega);

% Appropriate parameter adjustment can yield a better PSNR result.
opts.lambda = 3.5e6;  
opts.rank = 90;
opts.epsilon = 1e3;
opts.eta = 0.2; 
opts.max_iter = 400;            
opts.tol = 1e-3; 
opts.trunc = 0.3;

% Video inpainting
tic
X = LRTF(Y,W,opts);
toc
psnr = PSNR(I,X,double(~W));
fprintf('PSNR achieved by LRTF is %d dB\n',psnr);
figure(1);
imshow(uint8(X(:,:,1)'),[]);
