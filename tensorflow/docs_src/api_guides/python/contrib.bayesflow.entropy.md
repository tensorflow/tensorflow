# BayesFlow Entropy (contrib)
[TOC]

Entropy Ops.

## Background

Common Shannon entropy, the Evidence Lower BOund (ELBO), KL divergence, and more
all have information theoretic use and interpretations.  They are also often
used in variational inference.  This library brings together `Ops` for
estimating them, e.g. using Monte Carlo expectations.

## Examples

Example of fitting a variational posterior with the ELBO.

```python
# We start by assuming knowledge of the log of a joint density p(z, x) over
# latent variable z and fixed measurement x.  Since x is fixed, the Python
# function does not take x as an argument.
def log_joint(z):
  theta = tf.Variable(0.)  # Trainable variable that helps define log_joint.
  ...

# Next, define a Normal distribution with trainable parameters.
q = distributions.Normal(mu=tf.Variable(0.), sigma=tf.Variable(1.))

# Now, define a loss function (negative ELBO) that, when minimized, will adjust
# mu, sigma, and theta, increasing the ELBO, which we hope will both reduce the
# KL divergence between q(z) and p(z | x), and increase p(x).  Note that we
# cannot guarantee both, but in general we expect both to happen.
elbo = entropy.elbo_ratio(log_p, q, n=10)
loss = -elbo

# Minimize the loss
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
tf.global_variables_initializer().run()
for step in range(100):
  train_op.run()
```

## Ops

*   @{tf.contrib.bayesflow.entropy.elbo_ratio}
*   @{tf.contrib.bayesflow.entropy.entropy_shannon}
*   @{tf.contrib.bayesflow.entropy.renyi_ratio}
*   @{tf.contrib.bayesflow.entropy.renyi_alpha}
