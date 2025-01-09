% =========================================================================
%--------------------------------------------------------------------------

function    [HSI_res, RMSE, PSNR, sz]    =    NSSR_HSI_SR( Dir, image, sf, kernel_type )
rand('seed',0);
time0           =    clock;
[Z_ori,sz]      =    load_HSI( Dir, image, sf );
par             =    Parameters_setting( sf, kernel_type, sz );
X               =    par.H(Z_ori);
par.P           =    create_P();
Y               =    par.P*Z_ori;

%save('SavedX.mat', 'X');
%save('SavedY.mat', 'Y');
abundance_csc = load('Abundance_CSC_code.mat');
dict_dong = load('Dict_D_from_Dong.mat');
D = dict_dong.D;

D0              =    par.P*D;
U               =    abundance_csc.abund_sparse_code;

N               =    Comp_NLM_Matrix( Y, sz );   

%HSI_res         =    Nonnegative_SSR( D, D0, X, Y, N, par, Z_ori, sf, sz );
HSI_res         =    Nonnegative_SSR( D, D0, X, Y, N, U, par, Z_ori, sf, sz );


MSE             =    mean( mean( (Z_ori-HSI_res).^2 ) );
PSNR            =    10*log10(1/MSE);       
RMSE            =    sqrt(MSE)*255;
disp( sprintf('The final RMSE = %3.3f, PSNR = %3.2f', RMSE, PSNR) );   
disp(sprintf('Total elapsed time = %f min\n', (etime(clock,time0)/60) ));


