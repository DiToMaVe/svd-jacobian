function Jac = svd_jacobian_complex_num(X, row_index, column_index)

[U, Sigma, V] = svd(X, 'econ');     % reduced svd
eps = 1.e-8;

m = size(X,1);
n = size(X,2);

% Choose the phase phi_n such that the entry of V(:,r) with the largest
% magnitude is real and positive
[~, idx_max_abs] = max(abs(V));
lin_idx = idx_max_abs + ([1:n]-1)*n;
phi_n = -angle(V(lin_idx));
phase_mat = diag(exp(sqrt(-1)*phi_n));

V = V*phase_mat;
V(lin_idx) = real(V(lin_idx));  % Remove spurious imaginary parts
U = U*phase_mat;

% Delta
X_perturb = X;
% i*Delta
X_perturb_c = X;

% Delta
X_perturb(row_index, column_index) = X_perturb(row_index, column_index) + eps;
% i*Delta
X_perturb_c(row_index, column_index) = X_perturb_c(row_index, column_index) + sqrt(-1)*eps;

% Delta
[dU, dSigma, dV] = svd(X_perturb, 'econ');  % reduced svd
% i*Delta
[dU_c, dSigma_c, dV_c] = svd(X_perturb_c, 'econ');    % reduced svd

% Again, choose phi_n such that the entry in dV(:,r) corresponding to the
% entry in V(:,r) with the largest magnitude is real and positive

% Delta
phi_n = -angle(dV(lin_idx));
phase_mat = diag(exp(sqrt(-1)*phi_n));

dV = dV*phase_mat;
dV(lin_idx) = real(dV(lin_idx));  % Remove spurious imaginary parts
dU = dU*phase_mat;

% i*Delta
phi_n = -angle(dV_c(lin_idx));
phase_mat = diag(exp(sqrt(-1)*phi_n));

dV_c = dV_c*phase_mat;
dV_c(lin_idx) = real(dV_c(lin_idx));  % Remove spurious imaginary parts
dU_c = dU_c*phase_mat;

% Delta
Jac.dU = (dU-U)/eps;
Jac.dSigma = (dSigma-Sigma)/eps;
Jac.dV = (dV-V)/eps;

% i*Delta
Jac.dU_c = (dU_c-U)/eps;
Jac.dSigma_c = (dSigma_c-Sigma)/eps;
Jac.dV_c = (dV_c-V)/eps;
