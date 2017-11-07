clear all
clc

% Create random low-rank matrix with m>n
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Specifications
m = 10;
n = 5;
rank_X = 7;

% Construction
aux = randn(m,n);
[U,S,V] = svd(aux);

dS = diag(aux);
dS(rank_X+1:end) = 0;
S(1:length(dS),1:length(dS)) = diag(dS);
clear aux dS

% real-valued low-rank matrix of size (m,n) and rank rank_X
X_real = U*S*V';

% Construction
aux = randn(m,n) + sqrt(-1)*randn(m,n);
[U,S,V] = svd(aux);

dS = diag(aux);
dS(rank_X+1:end) = 0;
S(1:length(dS),1:length(dS)) = diag(dS);
clear aux dS

% complex-valued low-rank matrix of size (m,n) and rank rank_X
X_complex = U*S*V';
clear U S V

% Compute entries of the Jacobian 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% real-valued matrix
J_real_num = svd_jacobian_real_num(X_real, 1, 1);
J_real_alg = svd_jacobian_real(X_real, 10, 1);
J_real_alg_full = svd_jacobian_real(X_real);

% complex-valued matrix
J_complex_num = svd_jacobian_complex_num(X_complex, 8, 1);
J_complex_ana = svd_jacobian_complex(X_complex, 8, 1);
J_complex_ana_full = svd_jacobian_real(X_real);

% Check some results: compare analytic calculation with numerical
% calculation of the sensitivities via their ratio
round_prec = 10^4;  % round to specified precision

round(round_prec*J_complex_ana.dV./J_complex_num.dV)/round_prec
round(round_prec*J_complex_ana.dV_c./J_complex_num.dV_c)/round_prec

round(round_prec*J_complex_ana.dU./J_complex_num.dU)/round_prec
round(round_prec*J_complex_ana.dU_c./J_complex_num.dU_c)/round_prec

round(round_prec*J_complex_ana.dSigma_c./J_complex_num.dSigma_c)/round_prec
round(round_prec*J_complex_ana.dSigma./J_complex_num.dSigma)/round_prec
