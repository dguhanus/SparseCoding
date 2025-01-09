% =========================================================================
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

[Z_res, RMSE, PSNR, sz]     =    NSSR_HSI_SR( Dir, Test_file{2}, sf, kernel_type{2} );


disp( sprintf('Scaling factor = %d,  %s,  RMSE = %3.3f, PSNR = %2.3f \n', sf, Test_file{2}, RMSE, PSNR));

