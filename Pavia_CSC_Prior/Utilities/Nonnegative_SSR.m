% =========================================================================
%--------------------------------------------------------------------------

function  [Z, fun]    =   Nonnegative_SSR( D, D0, X, Y, N, Abun_CSC, par, Z_ori, sf, sz  )
A         =   zeros( par.K, par.h*par.w );
S         =   zeros( par.K, par.h*par.w );
Z         =   zeros( size(X,1), par.h*par.w );

eta1      =   par.eta1;
mu        =   par.mu;
ro        =   par.ro;
eta2      =   par.eta2; 
eta3      =   par.eta3; 

XHT       =   par.HT(X);
D02       =   D0'*D0;
D2        =   D'*D;
D0TY      =   D0'*Y;
DTU       =   zeros(par.K, par.h*par.w);
Ek        =   eye(par.K);

V1        =   zeros( size(Z) );
V2        =   zeros( size(A) );
fun       =   zeros(par.Iter, 1);
%U = U*N;




for  i  =  1 : par.Iter
    
    A     =    inv( D02 + (eta1 + mu)*D2 + (mu+eta3)*Ek ) * ( D0TY + eta1*DTU + eta3*Abun_CSC + D'*(mu*Z-V1/2) + (mu*S+V2/2) );
    DA    =    D*A;
    
    B     =    (XHT + mu*(DA + V1/(2*mu)))';    
    for  j  =  1 : size( X, 1 )
        [z,flag]     =    pcg( @(x)A_x(x, mu, par.fft_B, par.fft_BT, sf, sz), B(:,j), 1E-3, 350, [], [], Z(j, :)' );
        Z(j, :)      =    z';
    end
    
    
    S       =    max( soft(A-V2/(2*mu), eta2/(2*mu)), 0);
    V2      =    V2 + mu*( S - A );
    
    V1      =    V1 + mu*( DA-Z );
    U       =    Z*N;
    DTU     =    D'*U;
    %    
    mu      =    mu*ro;
    
    fun(i)      =   sum(sum( (X-par.H(DA)).^2 )) + sum( sum( (Y-par.P*DA).^2 ) ) + eta1*sum( sum( (DA-U).^2 ) ) + eta3*sum( sum( (A-Abun_CSC).^2 ) ) + eta2*sum( sum( abs(A) ) );
    %fun(i)      =   sum(sum( (X-par.H(DA)).^2 )) + sum( sum( (Y-par.P*DA).^2 ) ) + eta1*sum( sum( (A-U).^2 ) ) + eta2*sum( sum( abs(A) ) );
    if mod(i,5)==0
        MSE       =   mean( mean( (Z_ori-Z).^2 ) );
        PSNR      =   10*log10(1/MSE);                
        disp( sprintf('Iter %d, RMSE = %3.3f, PSNR = %3.2f', i, sqrt(MSE)*255, PSNR) );                        
    end   
end 
Z         =    alternating_back_projection( Z,X,Y,par.P,create_H(sz, sf), par );
