function L_bar = chol_rev_unblocked(L, L_bar)
%CHOL_REV_UNBLOCKED push derivatives back through a Cholesky decomposition
%
%     A_bar = chol_rev_unblocked(L, L_bar)
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
% unblocked LAPACK routine DPOTF2. This function shouldn't be used directly. The
% blocked routine CHOL_REV (corresponding to DPOTRF), which uses the routine in
% this file as a helper, is normally faster.

% Iain Murray, January 2016

N = size(L, 2);
assert(isequal(size(L), [N N]));
assert(isequal(size(L_bar), [N N]));

for J = N:-1:1
    L_bar(J,J) = L_bar(J,J) - L(J+1:N,J)'*L_bar(J+1:N,J) / L(J,J);
    L_bar(J:N,J) = L_bar(J:N,J) / L(J,J);
    L_bar(J,1:J-1) = L_bar(J,1:J-1) - L_bar(J:N,J)'*L(J:N,1:J-1);
    L_bar(J+1:N,1:J-1) = L_bar(J+1:N,1:J-1) - L_bar(J+1:N,J)*L(J,1:J-1);
    %L_bar(J,J) = 0.5 * L_bar(J,J); % can take out of loop if like.
end
L_bar(1:(N+1):end) = 0.5*L_bar(1:(N+1):end); % can put back in loop if like.

