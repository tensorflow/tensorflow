function L_bar = chol_rev(L, L_bar)
%CHOL_REV push derivatives back through a Cholesky decomposition
%
%     A_bar = chol_rev(L, L_bar)
%
% Back-propagate derivatives of some function f, through the Cholesky
% decomposition:
% A_bar = tril(df/dA), when L = chol(A, 'lower') and L_bar = df/dL.
%
% Inputs:
%          L NxN lower-triangular matrix, resulting from chol(A, 'lower'),
%                where A is a symmetric +ve definite matrix
%      L_bar NxN df/dL for some scalar function f
%
% Outputs:
%      A_bar NxN tril(df/dA)
%
% This code resulted from backpropagating derivatives (as in reverse-mode
% automatic differentiation) through the Cholesky decomposition algorithm in the
% LAPACK routine DPOTRF.

% Iain Murray, January 2016

persistent HAVE_WARNED
if isempty(HAVE_WARNED)
    HAVE_WARNED = 1;
    warning('chol_rev.m is slow. Please compile the mex version.')
end

N = size(L, 2);
assert(isequal(size(L), [N N]));
assert(isequal(size(L_bar), [N N]));

% Blocksize. Should be chosen more actively, as by LAPACK's ILAENV routine tuned
% for the current machine.
NB = 32;

for Ji = (N-NB+1):-NB:(1-NB+1)
    J = max(1, Ji);
    JB = NB - (J - Ji); % corrected block-size
    % At each stage we consider a lower triangular matrix "L_T" and 3 rectangular
    % matrices "L_B", "L_C", and "L_D", which are blocks of the large L matrix:
    %   L_T = L(J:J+JB-1, J:J+JB-1)
    %   L_B = L(J+JB:N, J:J+JB-1)
    %   L_C = L(J:J+JB-1, 1:J-1)
    %   L_D = L(J+JB:N, 1:J-1)
    % Similarly, Lb_T, Lb_B, Lb_C, and Lb_D are the corresponding blocks in L_bar.

    % Would be neater in numpy with views...

    % Lb_B = Lb_B/L_T
    L_bar(J+JB:N, J:J+JB-1) = L_bar(J+JB:N, J:J+JB-1)/L(J:J+JB-1, J:J+JB-1);
    % Lb_T -= tril(Lb_B'*L_B)
    L_bar(J:J+JB-1, J:J+JB-1) = L_bar(J:J+JB-1, J:J+JB-1) - tril(L_bar(J+JB:N, J:J+JB-1)'*L(J+JB:N, J:J+JB-1));
    % Lb_D -= Lb_B*L_C
    L_bar(J+JB:N, 1:J-1) = L_bar(J+JB:N, 1:J-1) - L_bar(J+JB:N, J:J+JB-1)*L(J:J+JB-1, 1:J-1);
    % Lb_C -= Lb_B'*L_D
    L_bar(J:J+JB-1, 1:J-1) = L_bar(J:J+JB-1, 1:J-1) - L_bar(J+JB:N, J:J+JB-1)'*L(J+JB:N, 1:J-1);
    % Lb_T = dpotf2_rev(L_T, Lb_T)
    L_bar(J:J+JB-1, J:J+JB-1) = chol_rev_unblocked(L(J:J+JB-1, J:J+JB-1), L_bar(J:J+JB-1, J:J+JB-1));
    % Lb_C -= (Lb_T + Lb_T')*C
    L_bar(J:J+JB-1, 1:J-1) = L_bar(J:J+JB-1, 1:J-1) - (L_bar(J:J+JB-1, J:J+JB-1) + L_bar(J:J+JB-1, J:J+JB-1)')*L(J:J+JB-1, 1:J-1);
end

