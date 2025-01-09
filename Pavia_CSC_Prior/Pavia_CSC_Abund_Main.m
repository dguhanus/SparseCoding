 % =========================================================================
% NSSR for Hyperspectral image super-resolution, Version 1.0
% Copyright(c) 2016 Weisheng Dong
% All Rights Reserved.
%
% ----------------------------------------------------------------------
% Permission to use, copy, or modify this software and its documentation
% for educational and research purposes only and without fee is here
% granted, provided that this copyright notice and the original authors'
% names appear on all copies and supporting documentation. This program
% shall not be used, rewritten, or adapted as the basis of a commercial
% software or hardware product without first obtaining permission of the
% authors. The authors make no representations about the suitability of
% this software for any purpose. It is provided "as is" without express
% or implied warranty.
%----------------------------------------------------------------------
%
% This is an implementation of the algorithm for Hyperspectral image super-
% resolution from a pair of low-resolution hyperspectral image and a high-
% resolution RGB image.
% 
% Please cite the following paper if you use this code:
%
% Weisheng Dong, Fazuo Fu, et. al.,"Hyperspectral image super-resolution via 
% non-negative structured sparse representation", IEEE Trans. On Image Processing, 
% vol. 25, no. 5, pp. 2337-2352, May 2015.
% 
%--------------------------------------------------------------------------

clc;
clear;
Current_Folder = pwd;
addpath(genpath('Utilities'));
addpath(genpath('Data'));
Dir             =    './Data/CAVE';
Result_dir      =    './Results/CAVE_Results/';
Test_file       =    {'oil_painting_ms', 'cloth_ms', 'fake_and_real_peppers_ms'};
%Test_file       =    { 'fake_and_real_peppers_ms'};
kernel_type     =    {'uniform_blur', 'Gaussian_blur'};
pre             =   'NSSR_';
sf              =    8;
Out_dir         =    fullfile(Result_dir, sprintf('sf_%d',sf));

%[D, B] = Dict_Abundance_from_X(Dir, Test_file{2}, sf, kernel_type{1});
[Z_res, RMSE, PSNR, sz]     =    NSSR_HSI_SR( Dir, Test_file{2}, sf, kernel_type{2} );


disp( sprintf('Scaling factor = %d,  %s,  RMSE = %3.3f, PSNR = %2.3f \n', sf, Test_file{2}, RMSE, PSNR));

