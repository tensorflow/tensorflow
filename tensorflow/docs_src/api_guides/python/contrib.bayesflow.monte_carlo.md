# BayesFlow Monte Carlo (contrib)
[TOC]

Monte Carlo integration and helpers.

## Background

Monte Carlo integration refers to the practice of estimating an expectation with
a sample mean.  For example, given random variable Z in \\(R^k\\) with density `p`,
the expectation of function `f` can be approximated like:

$$E_p[f(Z)] = \int f(z) p(z) dz$$
$$          ~ S_n
          := n^{-1} \sum_{i=1}^n f(z_i),  z_i\ iid\ samples\ from\ p.$$

If \\(E_p[|f(Z)|] < infinity\\), then \\(S_n\\) --> \\(E_p[f(Z)]\\) by the strong law of large
numbers.  If \\(E_p[f(Z)^2] < infinity\\), then \\(S_n\\) is asymptotically normal with
variance \\(Var[f(Z)] / n\\).

Practitioners of Bayesian statistics often find themselves wanting to estimate
\\(E_p[f(Z)]\\) when the distribution `p` is known only up to a constant.  For
example, the joint distribution `p(z, x)` may be known, but the evidence
\\(p(x) = \int p(z, x) dz\\) may be intractable.  In that case, a parameterized
distribution family \\(q_\lambda(z)\\) may be chosen, and the optimal \\(\lambda\\) is the
one minimizing the KL divergence between \\(q_\lambda(z)\\) and
\\(p(z | x)\\).  We only know `p(z, x)`, but that is sufficient to find \\(\lambda\\).


## Log-space evaluation and subtracting the maximum

Care must be taken when the random variable lives in a high dimensional space.
For example, the naive importance sample estimate \\(E_q[f(Z) p(Z) / q(Z)]\\)
involves the ratio of two terms \\(p(Z) / q(Z)\\), each of which must have tails
dropping off faster than \\(O(|z|^{-(k + 1)})\\) in order to have finite integral.
This ratio would often be zero or infinity up to numerical precision.

For that reason, we write

$$Log E_q[ f(Z) p(Z) / q(Z) ]$$
$$   = Log E_q[ \exp\{Log[f(Z)] + Log[p(Z)] - Log[q(Z)] - C\} ] + C,$$  where
$$C := Max[ Log[f(Z)] + Log[p(Z)] - Log[q(Z)] ].$$

The maximum value of the exponentiated term will be 0.0, and the expectation
can be evaluated in a stable manner.

## Ops

*   @{tf.contrib.bayesflow.monte_carlo.expectation}
*   @{tf.contrib.bayesflow.monte_carlo.expectation_importance_sampler}
*   @{tf.contrib.bayesflow.monte_carlo.expectation_importance_sampler_logspace}
