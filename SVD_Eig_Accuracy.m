A0 = gallery(3);
A1 = A0 + eps*randn(3,3).*A0;  % eps = 2.2204e-16

SVD_A0 = svd(A0);
SVD_A1 = svd(A1);
SVD_Err = (SVD_A0.*SVD_A0 - SVD_A1.*SVD_A1) ./ (SVD_A0.*SVD_A0);

Eig_A0 = eig(A0*A0');
Eig_A1 = eig(A1*A1');
Eig_Err = (Eig_A0 - Eig_A1) ./ Eig_A0;
