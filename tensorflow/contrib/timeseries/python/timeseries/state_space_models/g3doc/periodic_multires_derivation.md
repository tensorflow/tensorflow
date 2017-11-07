# Derivations for multi-resolution cycle transition matrix powers

This document contains derivations for the special-cased matrix-to-powers and
power sums used in `state_space_models/periodic.py`'s `ResolutionCycleModel` as
part of TensorFlow Time Series (TFTS).

## Setting and notation

Let $$M$$ be the number of latent values being cycled through
(`num_latent_values` in the code).

The `ResolutionCycleModel` transition matrix is based on roots of a matrix which
cycles through $$M$$ (odd) values and constrains their sum to be
zero (which when included in a state space model means that the expected sum
over a complete period is zero). Call this $$M - 1$$ x
$$M - 1$$ matrix $$C$$ (`cycle_matrix`):

$$ {\boldsymbol C}_{i, j} = \begin{cases} -1 & i = 0\\ 1 & j = i - 1\\ 0 &
\text{otherwise}\end{cases} $$

`ResolutionCycleModel` takes roots of this matrix using the following
parameterization:

$$ {\boldsymbol C}^p = \text{cycle_eigenvectors} *
\text{diag}(\text{cycle_eigenvalues})^{p} * \text{cycle_eigenvectors}^{-1} $$

Where:

$$\text{cycle_eigenvectors}_{i, j} = w_{\lfloor j / 2 \rfloor + 1}^{i (-1)^{j +
1}} - w_{\lfloor j / 2 \rfloor + 1}^{(i + 1) (-1)^{j + 1}}$$

$$(\text{cycle_eigenvectors}^{-1})_{i, j} = \frac{1}{M}
\sum_{k=0}^j w_{\lfloor i / 2 \rfloor + 1}^{k (-1)^i}$$

$$\text{cycle_eigenvalues}_{j} = w_{\lfloor j / 2 \rfloor + 1}^{(-1)^j}$$

Where $$w_j$$ is a root of unity:

$$w_j = e^{\frac{2 \pi j \sqrt{-1}}{M}}$$

In Sympy (useful for checking expressions when $$M$$ is small),
this looks like:

```python
import sympy
def root_of_unity(nth, number, to_power=1):
    return sympy.exp(2 * sympy.pi * number * sympy.I * to_power / nth)
matsize = 4
def eigvec_mat_fn(i, j):
    number = j // 2 + 1
    powersign = (j % 2) * 2 - 1
    return (root_of_unity(matsize + 1, number=number,
                          to_power=powersign * i)
            - root_of_unity(matsize + 1, number=number,
                            to_power=powersign * (i + 1)))
def eigvec_inverse_mat_fn(row, column):
    number = row // 2 + 1
    powersign = ((row + 1) % 2) * 2 - 1
    runningsum = 0
    for j in range(column + 1):
        runningsum += root_of_unity(
          matsize + 1, number, to_power=j * powersign) / (matsize + 1)
    return runningsum
def make_eigval_mat_fn(to_power=1):
    def eigval_mat_fn(i, j):
        if i == j:
            number = j // 2 + 1
            powersign = ((j + 1) % 2) * 2 - 1
            return root_of_unity(matsize + 1, number=number, 
                                 to_power=powersign*to_power)
        else:
            return 0
    return eigval_mat_fn
eigval_power = sympy.Rational(1, 1)
eigvecs = sympy.Matrix(matsize, matsize, eigvec_mat_fn)
eigvals = sympy.Matrix(matsize, matsize, make_eigval_mat_fn(eigval_power))
eigvecs_inv = sympy.Matrix(matsize, matsize, eigvec_inverse_mat_fn)
print (eigvecs * eigvals * eigvecs_inv).evalf()
```

## Proof that these are eigenvectors/eigenvalues of `cycle_matrix`

We want to show that:

$${\boldsymbol C} * \text{cycle_eigenvectors}_{\bullet, j} =
\text{cycle_eigenvalues}_j * \text{cycle_eigenvectors}_{\bullet, j} $$

Where $$\text{cycle_eigenvectors}_{\bullet, j}$$ is a column vector containing
the $$j^\text{th}$$ eigenvector.

We have telescoping sum in the first entry:

$$({\boldsymbol C} * \text{cycle_eigenvectors}_{\bullet, j})_i =
\begin{cases} -\sum_{k=0}^{M - 2}
\text{cycle_eigenvectors}_{k, j} & i = 0\\ \text{cycle_eigenvectors}_{i - 1, j}
& \text{otherwise} \end{cases}$$

$$ = \begin{cases} w_{\lfloor j / 2 \rfloor + 1}^{(M -
1)(-1)^{j + 1}} - w_{\lfloor j / 2 \rfloor + 1}^{0(-1)^{j + 1}} & i = 0\\
\text{cycle_eigenvectors}_{i - 1, j} & \text{otherwise} \end{cases}$$

$$ = \begin{cases} w_{\lfloor j / 2 \rfloor + 1}^{(-1)^{j}} \left (w_{\lfloor j
/ 2 \rfloor + 1}^{M(-1)^{j + 1}} - w_{\lfloor j / 2
\rfloor + 1}^{(-1)^{j + 1}} \right ) & i = 0\\ \text{cycle_eigenvectors}_{i - 1,
j} & \text{otherwise} \end{cases}$$

$$ = \begin{cases} w_{\lfloor j / 2 \rfloor + 1}^{(-1)^{j}} \left (w_{\lfloor j
/ 2 \rfloor + 1}^{0(-1)^{j + 1}} - w_{\lfloor j / 2 \rfloor + 1}^{(-1)^{j + 1}}
\right ) & i = 0\\ \text{cycle_eigenvectors}_{i - 1, j} & \text{otherwise}
\end{cases}$$

$$ = \begin{cases} w_{\lfloor j / 2 \rfloor + 1}^{(-1)^{j}}
\text{cycle_eigenvectors}_{0, j} ) & i = 0\\ \text{cycle_eigenvectors}_{i - 1,
j} & \text{otherwise} \end{cases}$$

The remaining cases follow from the fact that:

$$w_{\lfloor j / 2 \rfloor + 1}^{(-1)^{j}} \text{cycle_eigenvectors}_{i, j} =
\text{cycle_eigenvectors}_{i - 1, j}$$

$$w_{\lfloor j / 2 \rfloor + 1}^{(-1)^{j}} \left( w_{\lfloor j / 2 \rfloor +
1}^{i (-1)^{j + 1}} - w_{\lfloor j / 2 \rfloor + 1}^{(i + 1) (-1)^{j + 1}}
\right) = w_{\lfloor j / 2 \rfloor + 1}^{(i - 1) (-1)^{j + 1}} - w_{\lfloor j /
2 \rfloor + 1}^{i (-1)^{j + 1}}$$

## Proof of eigenvector inverse matrix

We want to show that (for the expressions above):

$$ I = \text{cycle_eigenvectors} * \text{cycle_eigenvectors}^{-1} $$

Multiplying it out, we have:

$$(\text{cycle_eigenvectors} * \text{cycle_eigenvectors}^{-1})_{i, j} =
\sum_{k=0}^{M - 2} \text{cycle_eigenvectors}_{i, k}
(\text{cycle_eigenvectors}^{-1})_{k, j} $$

$$ = \frac{1}{M} \sum_{k=0}^{M -
2} \left[ \left( w_{\lfloor k / 2 \rfloor + 1}^{i (-1)^{k + 1}} - w_{\lfloor k /
2 \rfloor + 1}^{(i + 1) (-1)^{k + 1}} \right) \sum_{l=0}^j w_{\lfloor k / 2
\rfloor + 1}^{l (-1)^{k}} \right]$$

$$ = \frac{1}{M} \sum_{k=0}^{M -
2} \sum_{l=0}^j \left[ \left( w_{\lfloor k / 2 \rfloor + 1}^{i (-1)^{k + 1}} -
w_{\lfloor k / 2 \rfloor + 1}^{(i + 1) (-1)^{k + 1}} \right) w_{\lfloor k / 2
\rfloor + 1}^{l (-1)^{k}} \right]$$

$$ = \frac{1}{M} \sum_{k=0}^{M -
2} \sum_{l=0}^j \left[ w_{\lfloor k / 2 \rfloor + 1}^{(i - l) (-1)^{k + 1}} -
w_{\lfloor k / 2 \rfloor + 1}^{(i - l + 1) (-1)^{k + 1}} \right]$$

Using telescoping:

$$ = \frac{1}{M} \sum_{k=0}^{M -
2} \left[ w_{\lfloor k / 2 \rfloor + 1}^{(i - j) (-1)^{k + 1}} - w_{\lfloor k /
2 \rfloor + 1}^{(i + 1) (-1)^{k + 1}} \right]$$

Since $$e^{-ix} = \text{conj}(e^{ix})$$, the imaginary components cancel out of
the sum:

$$ = \frac{2}{M} \sum_{k=0}^{(M -
1) / 2 - 1} \text{Real}\left[ w_{k + 1}^{i - j} - w_{k + 1}^{i + 1} \right]$$

$$ = \frac{2}{M} \sum_{k=0}^{(M -
1) / 2 - 1} \left[ \text{cos}\left(\frac{2 \pi (i - j) (k +
1)}{M}\right) - \text{cos}\left(\frac{2 \pi (i + 1) (k +
1)}{M}\right) \right]$$

Using Lagrange's identity $$\sum_{n=1}^N \text{cos}(n \theta) = -\frac{1}{2} +
\frac{\text{sin}\left(\left(N + \frac{1}{2}\right) \theta\right)}{2
\text{sin}\left(\frac{\theta}{2}\right)}$$:

$$ = \frac{2}{M} \left(
\frac{\text{sin}\left(\left(\left(\frac{M -
1}{2}\right) + \frac{1}{2}\right) \frac{2 \pi (i -
j)}{M}\right)}{2 \text{sin}\left(\frac{\pi (i -
j)}{M}\right)} \\-
\frac{\text{sin}\left(\left(\left(\frac{M -
1}{2}\right) + \frac{1}{2}\right) \frac{2 \pi (i +
1)}{M}\right)}{2 \text{sin}\left(\frac{\pi (i +
1)}{M}\right)} \right)$$

$$ = \frac{1}{M} \left(\frac{\text{sin}(\pi (i -
j))}{\text{sin}\left(\frac{\pi (i - j)}{M}\right)} -
\frac{\text{sin}(\pi (i + 1))}{\text{sin}\left(\frac{\pi (i +
1)}{M}\right)} \right)$$

The second term will always be zero, since $$i + 1$$ is at most
$$M - 1$$ ($$i$$ ranges from $$0$$ to
$$M - 2$$). The first term is also zero unless $$i = j$$
($$\text{sin}(x)$$ is zero at integer multiples of $$\pi$$). Taking a limit when
$$i = j$$, the expression evaluates to $$1$$ (L'Hospital's rule gives a ratio of
cosines, both one, along with a ratio of the arguments from the chain rule).

## Simplification of expression for matrix to a power

Having established that the eigenvalues, eigenvectors, and the inverse
eigenvector matrix are all correct, we can now use them to derive an expression
for the matrix raised to a power:

$$ {\boldsymbol C}^p = \text{cycle_eigenvectors} *
\text{diag}(\text{cycle_eigenvalues})^p * \text{cycle_eigenvectors}^{-1} $$

$$({\boldsymbol C}^p)_{i, j} = \sum_{k=0}^{M - 2}
\text{cycle_eigenvectors}_{i, k} * \text{cycle_eigenvalues}_k^p *
(\text{cycle_eigenvectors}^{-1})_{k, j}$$

$$ = \frac{1}{M} \sum_{k=0}^{M -
2} \left[\left( w_{\lfloor k / 2 \rfloor + 1}^{(p - i) (-1)^{k + 1}} -
w_{\lfloor k / 2 \rfloor + 1}^{(p - i - 1) (-1)^{k + 1}} \right) \sum_{l=0}^j
w_{\lfloor k / 2 \rfloor + 1}^{l (-1)^{k}} \right]$$

Following the same logic as for the inverse matrix proof, this leads to
($$cos(x)$$ being an even function):

$$ = \frac{2}{M} \sum_{k=0}^{(M -
1) / 2 - 1} \left[ \text{cos}\left(\frac{2 \pi (p - i + j)
k}{M}\right) - \text{cos}\left(\frac{2 \pi (p - i - 1)
k)}{M}\right) \right]$$

Applying Lagrange's identity and simplifying, we get:

$$ = \frac{1}{M} \left(\frac{\text{sin}(\pi (p - i +
j))}{\text{sin}\left(\frac{\pi (p - i + j)}{M}\right)} -
\frac{\text{sin}(\pi (p - i - 1))}{\text{sin}\left(\frac{\pi (p - i -
1)}{M}\right)} \right)$$

As a special/limiting case, we get the integer powers of the original matrix
($$I(b)$$ is 1 if $$b$$ is true and zero otherwise):

$$ ({\boldsymbol C}^p)_{i, j} = \frac{1}{M} \left(
M * I(p - i + j \equiv 0\pmod {M})
\\- M * I(p - i - 1 \equiv 0\pmod
{M}) \right)$$

## Simplification of expression for pre- and post-multiplication of noise matrix to a power

Raising the transition matrix to a power allows us to transform the inferred
mean state across time (imputing), but performing imputation on the covariance
of the state estimate requires adding the noise covariance at each step (and
then transforming by pre- and post-multiplying by the transition
matrix). However, this noise covariance has a very special form, having only one
non-zero element in the upper left hand corner.

$$\text{noise_covariance}_{i, j} = \begin{cases} \text{noise_scalar} & i = j =
0\\ 0 & \text{otherwise}\end{cases} $$

This makes it easy to compute an expression for $${\boldsymbol C}^p *
\text{noise_covariance} * ({\boldsymbol C}^T)^p$$:

$$\left ({\boldsymbol C}^p * \text{noise_covariance} \right)_{i, j}
= ({\boldsymbol C}^p)_{i, 0} * \text{noise_scalar} * I(j = 0)$$

$$\left ({\boldsymbol C}^p * \text{noise_covariance} *
({\boldsymbol C}^T \right)^p)_{i, j} = ({\boldsymbol C}^p)_{i, 0} *
\text{noise_scalar} * ({\boldsymbol C}^p)_{j, 0}$$

$$ = \frac{1}{M^2} \left(\frac{\text{sin}(\pi (p -
i))}{\text{sin}\left(\frac{\pi (p - i)}{M}\right)} -
\frac{\text{sin}(\pi (p - i - 1))}{\text{sin}\left(\frac{\pi (p - i -
1)}{M}\right)} \right) \\ \left(\frac{\text{sin}(\pi (p -
j))}{\text{sin}\left(\frac{\pi (p - j)}{M}\right)} -
\frac{\text{sin}(\pi (p - j - 1))}{\text{sin}\left(\frac{\pi (p - j -
1)}{M}\right)} \right)$$

This (and the transition-matrix-to-a-power expression above) can be simplified
slightly using the fact that $$\text{sin}((x + i)\pi) = (-1)^i \text{sin}(x
\pi)$$ for integers $$i$$.

## Open questions

- It would be nice to have an expression for the elements of $$f(\lambda, N) =
  \sum_{k=0}^N {\boldsymbol C}^{\lambda k} * \text{noise_covariance} *
  ({\boldsymbol C}^T)^{\lambda k}$$, especially for $$0 < \lambda < 1$$.
- It seems that when $$\lambda =
  \frac{M}{\text{periodicity}}$$, then $$f(\lambda,
  \text{periodicity}) = \frac{\text{periodicity}}{M} f(1,
  M)$$, but I'm not exactly sure why (i.e. have not
  proven it).
