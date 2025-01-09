function [Z,sz,mv]    =  load_HSI( Dir, image, sf )
abc_pavia = load('PaviaU.mat');
Orig_Data = abc_pavia.paviaU;

Data_3D = Orig_Data(1:320,1:320,10:40);
s_Z = reshape(Data_3D, [size(Data_3D,3), size(Data_3D,1)*size(Data_3D,2)]);

mv     =  max(s_Z(:));
%Z = s_Z;
Z = s_Z/mv;
sz = [size(Data_3D,1), size(Data_3D,2)]  ;


end
