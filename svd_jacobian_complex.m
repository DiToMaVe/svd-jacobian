function Jac = svd_jacobian_complex(X, row_index, column_index)

    %   SVD_JACOBIAN_COMPLEX:   Computation of  the Jacobian of the 
    %                           Singular Value Decomposition (SVD) for a 
    %                           complex-valued matrix X.
    %                         
    %   USAGE:
    %       Jac = SVD_JACOBIAN_COMPLEX(X)
    %       Jac = SVD_JACOBIAN_COMPLEX(X, row_index, column_index)
    %
    %   DESCRIPTION:
    %       X           : Matrix of size (m,n) with m>n
    %       row_index   : vector of row indices
    %       column_index: vector of column indices
    %       If row_index and column_index are given, only the partial
    %       derivatives w.r.t. the specified matrix entries are calculated.
    %       Otherwise the full Jacobian is computed.
    %
    %   REFERENCES:
    %
    %       ``Unbiased Risk Estimates for Singular Value Thresholding''
    %           E.J.Candes, C.A.Sing-Long, and J.D. Trzasko
    %       ``Estimating the Jacobian of the Singular Value Decomposition: 
    %           Theory and Application''
    %           T. Papadopoulo, M.I. A. Lourakis
    %       ``Uncertainty calculation in (operational) modal analysis''
    %           R. Pintelon, P. Guillaume, J. Schoukens
    %
    %   Author: Dieter Verbeke (DV)
    %   V1.0:   February 2017.
    
m = size(X,1);
n = size(X,2);

switch nargin
    case 3
        N_var = 1;          % number of dependent variables to be considered in Jacobian
        Jac = struct( 'dU', zeros(m,n,N_var), 'dSigma', zeros(n,n,N_var), 'dV', zeros(n,n,N_var) );
    case 1
        N_var = m*n;        % number of dependent variables to be considered in Jacobian
        Jac = struct( 'dU', zeros(m,n,N_var), 'dSigma', zeros(n,n,N_var), 'dV', zeros(n,n,N_var) );
        
end % switch

% Singular Value Decomposition
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[U, Sigma, V] = svd(X, 'econ');     % reduced svd
[U_tilde, Sigma_tilde, V] = svd(X);           % full svd

% Choose the phase phi_n such that the entry of V(:,r) with the largest
% magnitude is real and positive
[~, idx_max_abs] = max(abs(V));
lin_idx = idx_max_abs + ([1:n]-1)*n;
phi_n = -angle(V(lin_idx));
phase_mat = diag(exp(sqrt(-1)*phi_n));

V = V*phase_mat;
V(lin_idx) = real(V(lin_idx));  % Remove spurious imaginary parts
U = U*phase_mat;
if m>n
    U_tilde = U_tilde*[phase_mat, zeros(n,m-n); zeros(m-n,n), eye(m-n,m-n)];
else
    U_tilde = U_tilde*phase_mat;
end % if

% Compute Jacobian
%%%%%%%%%%%%%%%%%%

for var_index = 1:N_var
    
    Omega_U = zeros(m,n);
    Omega_V = zeros(n,n);
    Omega_U_c = zeros(m,n);
    Omega_V_c = zeros(n,n);
    Delta = zeros(m,n);
    dSigma = zeros(n);
    dSigma_c = zeros(n);
    
    switch nargin
        case 3
            kk = row_index(var_index);
            ll = column_index(var_index);
        case 1
            kk = mod(var_index,m);
            if kk==0
                kk = m;
                ll = var_index/m;
            else
                ll = floor(var_index/m)+1;
            end
    end % switch
    
    Delta(kk,ll) = 1;   % J(k,l)   
    UDV = U_tilde'*Delta*V;
    UDVc = sqrt(-1)*UDV; % i*Delta instead of Delta
    
    for jj=1:n
        for ii=1:n
            if ii~=jj
                C = -1/(Sigma(ii,ii)^2-Sigma(jj,jj)^2)*[Sigma(jj,jj), Sigma(ii,ii); -Sigma(ii,ii), -Sigma(jj,jj)];  % Eq. 23
                % Delta
                Omega_ij = C*[UDV(ii,jj); conj(UDV(jj,ii))];
                Omega_U(ii,jj) = Omega_ij(1);
                Omega_V(ii,jj) = Omega_ij(2);
                % i*Delta
                Omega_c_ij = C*[UDVc(ii,jj); conj(UDVc(jj,ii))];
                Omega_U_c(ii,jj) = Omega_c_ij(1);
                Omega_V_c(ii,jj) = Omega_c_ij(2);                
            end % if ii~=jj
        end % for ii
        for ii=n+1:m
            % Delta
            Omega_U(ii,jj) = UDV(ii,jj)/Sigma(jj,jj);    % Eq. 30 with U^H instead of U^T 
            % i*Delta
            Omega_U_c(ii,jj) = UDVc(ii,jj)/Sigma(jj,jj);     
        end % for ii
    end % for jj
        
    for ii=1:n
        UDV(ii,ii);
        % Delta
        dSigma(ii,ii) = real(UDV(ii,ii));
        % i*Delta
        dSigma_c(ii,ii) = real(UDVc(ii,ii));
        % Choose the degree of freedom in Eq. 2 such that dV = V*Omega_V*
        % is equal to zero for the entries of the right singular values
        % which were previously made real and positive by an appropriate
        % phase phi_n.
        
        % Delta
        Omega_V(ii,ii) = -sqrt(-1)*imag(V(idx_max_abs(ii),:)*Omega_V(:,ii))/real(V(idx_max_abs(ii),ii)); % dV = 
        Omega_U(ii,ii) = (sqrt(-1)*imag(UDV(ii,ii))-Omega_V(ii,ii)*Sigma(ii,ii))/Sigma(ii,ii); % Eq. 28
        
        % i*Delta
        Omega_V_c(ii,ii) = -sqrt(-1)*imag(V(idx_max_abs(ii),:)*Omega_V_c(:,ii))/real(V(idx_max_abs(ii),ii)); % dV = 
        Omega_U_c(ii,ii) = (sqrt(-1)*imag(UDVc(ii,ii))-Omega_V_c(ii,ii)*Sigma(ii,ii))/Sigma(ii,ii); % Eq. 28
        
    end % for ii
    
    % Delta
    dU = U_tilde*Omega_U;
    dV = V*(Omega_V');
    
    % i*Delta
    dU_c = U_tilde*Omega_U_c;
    dV_c = V*(Omega_V_c');
    
    % Delta
    Jac.dU(:, :, var_index) = dU;
    Jac.dSigma(:, :, var_index) = dSigma;
    Jac.dV(:, :, var_index) = dV;
    
    % i*Delta
    Jac.dU_c(:, :, var_index) = dU_c;
    Jac.dSigma_c(:, :, var_index) = dSigma_c;
    Jac.dV_c(:, :, var_index) = dV_c;
    
    if nargin==3
        % Delta
        Jac.dU = squeeze(Jac.dU);
        Jac.dSigma = squeeze(Jac.dSigma);
        Jac.dV = squeeze(Jac.dV);
        
        % i*Delta
        Jac.dU_c = squeeze(Jac.dU_c);
        Jac.dSigma_c = squeeze(Jac.dSigma_c);
        Jac.dV_c = squeeze(Jac.dV_c);
        
    end % if nargin==3
    
end % for var_index
