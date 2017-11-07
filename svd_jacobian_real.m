function Jac = svd_jacobian_real(X, row_index, column_index)

    %   SVD_JACOBIAN_REAL:    Computation of  the Jacobian of the Singular 
    %                         Value Decomposition (SVD) for a real_valued 
    %                         matrix X.
    %   USAGE:
    %       Jac = SVD_JACOBIAN_REAL(X)
    %       Jac = SVD_JACOBIAN_REAL(X, row_index, column_index)
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
    %           E.J.Candes, C.A.Sing-Long, and J.D.Trzasko
    %       ``Estimating the Jacobian of the Singular Value Decomposition: 
    %           Theory and Application''
    %           T. Papadopoulo, M.I. A. Lourakis
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
[U_tilde, ~, V] = svd(X);           % full svd

% Compute Jacobian
%%%%%%%%%%%%%%%%%%

for var_index = 1:N_var
    
    Omega_U = zeros(m,n);
    Omega_V = zeros(n,n);
    Delta = zeros(m,n);
    dSigma = zeros(n);
    
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
    
    for ii=1:n
        dSigma(ii,ii) = UDV(ii,ii);     
    end % for ii
    
    for jj=1:n
        for ii=1:n
            if ii~=jj
                C = -1/(Sigma(ii,ii)^2-Sigma(jj,jj)^2)*[Sigma(jj,jj), Sigma(ii,ii); -Sigma(ii,ii), -Sigma(jj,jj)];  % Eq. 23
                Omega_ij = C*[UDV(ii,jj);UDV(jj,ii)];
                Omega_U(ii,jj) = Omega_ij(1);
                Omega_V(ii,jj) = Omega_ij(2);
            end % if ii~=jj
        end % for ii
        for ii=n+1:m
            Omega_U(ii,jj) = UDV(ii,jj)/Sigma(jj,jj);  
        end % for ii
    end % for jj
    
    dU = U_tilde*Omega_U;
    dV = V*(Omega_V');
    
    Jac.dU(:, :, var_index) = dU;
    Jac.dSigma(:, :, var_index) = dSigma;
    Jac.dV(:, :, var_index) = dV;
    
    if nargin==3
        Jac.dU = squeeze(Jac.dU);
        Jac.dSigma = squeeze(Jac.dSigma);
        Jac.dV = squeeze(Jac.dV);
    end % if nargin==3
    
end % for var_index
