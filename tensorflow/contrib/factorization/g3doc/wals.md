# WALS Factorization

WALS (Weighed Alternating Least Squares) is an algorithm for factorizing a
sparse matrix $$A$$ into low rank factors, $$U$$ and $$V$$, such that the
product of these factors is a "good" approximation of the full matrix.


![wals](wals.png)

Typically, it involves minimizing the following loss function:
$$ min_{U,V} (||\sqrt{W} \odot (A- UV^T)||_F^2 + \lambda (||U||_F^2 + ||V||_F^2)) $$,
where $$\lambda$$ is a regularization parameter, and $$\odot$$ represents a
component-wise product. Assuming $$W$$ is of the form 
$$W_{i, j} = w_0 + 1_{A_{ij} \neq 0}R_i C_j$$,
where $$w_0$$ is the weight of unobserved entries, and $$R$$ and $$C$$ are
the row and column weights respectively, lends this equation to an efficient
implementation.

The algorithm proceeds in phases, or "sweeps", where each sweep involves
fixing $$U$$ and solving for $$V$$, and then fixing $$V$$ and solving for $$U$$.
Convergence is typically pretty fast (10-20 sweeps).
