# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""L2HMC compatible with TensorFlow's eager execution.

Reference [Generalizing Hamiltonian Monte Carlo with Neural
Networks](https://arxiv.org/pdf/1711.09268.pdf)

Code adapted from the released TensorFlow graph implementation by original
authors https://github.com/brain-research/l2hmc.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import numpy.random as npr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.eager.python.examples.l2hmc import neural_nets


class Dynamics(tf.keras.Model):
  """Dynamics engine of naive L2HMC sampler.

  Args:
    x_dim: dimensionality of observed data
    loglikelihood_fn: log-likelihood function of conditional probability
    n_steps: number of leapfrog steps within each transition
    eps: initial value learnable scale of step size
  """

  def __init__(self, x_dim, loglikelihood_fn, n_steps=25, eps=.1):
    super(Dynamics, self).__init__()

    self.x_dim = x_dim
    self.potential = loglikelihood_fn
    self.n_steps = n_steps

    self._construct_time()
    self._construct_masks()

    self.position_fn = neural_nets.GenericNet(x_dim, factor=2.)
    self.momentum_fn = neural_nets.GenericNet(x_dim, factor=1.)

    self.eps = tfe.Variable(
        initial_value=eps, name="eps", dtype=tf.float32, trainable=True)

    # TODO(lxuechen): Remove this after model.add_weight is in place
    self.vars_not_in_layers = [self.eps]
    self.vars_not_in_layers += self.position_fn.vars_not_in_layers
    self.vars_not_in_layers += self.momentum_fn.vars_not_in_layers

  def apply_transition(self, position):
    """Propose a new state and perform the accept or reject step."""

    # Simulate dynamics both forward and backward;
    # Use sampled Bernoulli masks to compute the actual solutions
    position_f, momentum_f, accept_prob_f = self.transition_kernel(
        position, forward=True)
    position_b, momentum_b, accept_prob_b = self.transition_kernel(
        position, forward=False)

    # Decide direction uniformly
    forward_mask = tf.cast(
        tf.random_uniform(shape=[tf.shape(position)[0]]) > .5, tf.float32)
    backward_mask = 1. - forward_mask

    # Obtain proposed states
    position_post = (
        forward_mask[:, None] * position_f +
        backward_mask[:, None] * position_b)
    momentum_post = (
        forward_mask[:, None] * momentum_f +
        backward_mask[:, None] * momentum_b)

    # Probability of accepting the proposed states
    accept_prob = forward_mask * accept_prob_f + backward_mask * accept_prob_b

    # Accept or reject step
    accept_mask = tf.cast(
        accept_prob > tf.random_uniform(tf.shape(accept_prob)), tf.float32)
    reject_mask = 1. - accept_mask

    # Samples after accept/reject step
    position_out = (
        accept_mask[:, None] * position_post + reject_mask[:, None] * position)

    return position_post, momentum_post, accept_prob, position_out

  def transition_kernel(self, position, forward=True):
    """Transition kernel of augmented leapfrog integrator."""

    lf_fn = self._forward_lf if forward else self._backward_lf

    # Resample momentum
    momentum = tf.random_normal(tf.shape(position))
    position_post, momentum_post = position, momentum
    sumlogdet = 0.
    # Apply augmented leapfrog steps
    for i in range(self.n_steps):
      position_post, momentum_post, logdet = lf_fn(position_post, momentum_post,
                                                   i)
      sumlogdet += logdet

    accept_prob = self._compute_accept_prob(position, momentum, position_post,
                                            momentum_post, sumlogdet)

    return position_post, momentum_post, accept_prob

  def _forward_lf(self, position, momentum, i):
    """One forward augmented leapfrog step. See eq (5-6) in paper."""

    t = self._get_time(i)
    mask, mask_inv = self._get_mask(i)
    sumlogdet = 0.

    momentum, logdet = self._update_momentum_forward(position, momentum, t)
    sumlogdet += logdet

    position, logdet = self._update_position_forward(position, momentum, t,
                                                     mask)
    sumlogdet += logdet

    position, logdet = self._update_position_forward(position, momentum, t,
                                                     mask_inv)
    sumlogdet += logdet

    momentum, logdet = self._update_momentum_forward(position, momentum, t)
    sumlogdet += logdet

    return position, momentum, tf.reduce_sum(sumlogdet, axis=1)

  def _backward_lf(self, position, momentum, i):
    """One backward augmented leapfrog step. See Appendix A in paper."""

    # Reversed index/sinusoidal time
    t = self._get_time(self.n_steps - i - 1)
    mask, mask_inv = self._get_mask(self.n_steps - i - 1)
    sumlogdet = 0.

    momentum, logdet = self._update_momentum_backward(position, momentum, t)
    sumlogdet += logdet

    position, logdet = self._update_position_backward(position, momentum, t,
                                                      mask)
    sumlogdet += logdet

    position, logdet = self._update_position_backward(position, momentum, t,
                                                      mask_inv)
    sumlogdet += logdet

    momentum, logdet = self._update_momentum_backward(position, momentum, t)
    sumlogdet += logdet

    return position, momentum, tf.reduce_sum(sumlogdet, axis=1)

  def _update_momentum_forward(self, position, momentum, t):
    """Update v in the forward leapfrog step."""

    grad = self.grad_potential(position)
    scale, translation, transformed = self.momentum_fn([position, grad, t])
    scale *= .5 * self.eps
    transformed *= self.eps
    momentum = (
        momentum * tf.exp(scale) -
        .5 * self.eps * (tf.exp(transformed) * grad - translation))

    return momentum, scale

  def _update_position_forward(self, position, momentum, t, mask):
    """Update x in the forward leapfrog step."""

    mask_inv = 1. - mask
    scale, translation, transformed = self.position_fn(
        [momentum, mask * position, t])
    scale *= self.eps
    transformed *= self.eps
    position = (
        mask * position +
        mask_inv * (position * tf.exp(scale) + self.eps *
                    (tf.exp(transformed) * momentum + translation)))

    return position, mask_inv * scale

  def _update_momentum_backward(self, position, momentum, t):
    """Update v in the backward leapfrog step. Inverting the forward update."""

    grad = self.grad_potential(position)
    scale, translation, transformed = self.momentum_fn([position, grad, t])
    scale *= -.5 * self.eps
    transformed *= self.eps
    momentum = (
        tf.exp(scale) * (momentum + .5 * self.eps *
                         (tf.exp(transformed) * grad - translation)))

    return momentum, scale

  def _update_position_backward(self, position, momentum, t, mask):
    """Update x in the backward leapfrog step. Inverting the forward update."""

    mask_inv = 1. - mask
    scale, translation, transformed = self.position_fn(
        [momentum, mask_inv * position, t])
    scale *= -self.eps
    transformed *= self.eps
    position = (
        mask_inv * position + mask * tf.exp(scale) *
        (position - self.eps * tf.exp(transformed) * momentum + translation))

    return position, mask * scale

  def _compute_accept_prob(self, position, momentum, position_post,
                           momentum_post, sumlogdet):
    """Compute the prob of accepting the proposed state given old state."""

    old_hamil = self.hamiltonian(position, momentum)
    new_hamil = self.hamiltonian(position_post, momentum_post)

    return tf.exp(tf.minimum(old_hamil - new_hamil + sumlogdet, 0.))

  def _construct_time(self):
    """Convert leapfrog step index into sinusoidal time."""

    self.ts = []
    for i in range(self.n_steps):
      t = tf.constant(
          [
              np.cos(2 * np.pi * i / self.n_steps),
              np.sin(2 * np.pi * i / self.n_steps)
          ],
          dtype=tf.float32)
      self.ts.append(t[None, :])

  def _get_time(self, i):
    """Get sinusoidal time for i-th augmented leapfrog step."""

    return self.ts[i]

  def _construct_masks(self):
    """Construct different binary masks for different time steps."""

    self.masks = []
    for _ in range(self.n_steps):
      idx = npr.permutation(np.arange(self.x_dim))[:self.x_dim // 2]
      mask = np.zeros((self.x_dim,))
      mask[idx] = 1.
      mask = tf.constant(mask, dtype=tf.float32)
      self.masks.append(mask[None, :])

  def _get_mask(self, i):
    """Get binary masks for i-th augmented leapfrog step."""

    m = self.masks[i]
    return m, 1. - m

  def kinetic(self, v):
    """Compute the kinetic energy."""

    return .5 * tf.reduce_sum(v**2, axis=1)

  def hamiltonian(self, position, momentum):
    """Compute the overall Hamiltonian."""

    return self.potential(position) + self.kinetic(momentum)

  def grad_potential(self, position, check_numerics=True):
    """Get gradient of potential function at current location."""

    if not tf.executing_eagerly():
      # TODO(lxuechen): Change this to tfe.gradients_function when it works
      grad = tf.gradients(self.potential(position), position)[0]
    else:
      grad = tfe.gradients_function(self.potential)(position)[0]

    if check_numerics:
      return tf.check_numerics(grad, message="gradient of potential")

    return grad


# Defining loss and grads for training
def compute_loss(x, dynamics, scale=.1, eps=1e-4):
  """Compute loss defined in equation (8)."""

  z = tf.random_normal(tf.shape(x))
  x_, _, x_accept_prob, x_out = dynamics.apply_transition(x)
  z_, _, z_accept_prob, _ = dynamics.apply_transition(z)

  # Add eps for numerical stability; following released impl
  x_loss = tf.reduce_sum((x - x_)**2, axis=1) * x_accept_prob + eps
  z_loss = tf.reduce_sum((z - z_)**2, axis=1) * z_accept_prob + eps

  loss = tf.reduce_mean(
      (1. / x_loss + 1. / z_loss) * scale - (x_loss + z_loss) / scale, axis=0)

  return loss, x_out


def loss_and_grads(x, dynamics):
  """Obtain loss value and gradients."""

  with tf.GradientTape() as tape:
    loss_val, x_out = compute_loss(x, dynamics)

  vars_ = dynamics.variables + dynamics.vars_not_in_layers
  grads = tape.gradient(loss_val, vars_)

  return loss_val, grads, x_out


def warmup(dynamics, optimizer, n_iters=1, n_samples=200):
  """Warmup optimization to reduce overhead."""

  samples = tf.random_normal(
      shape=[n_samples, dynamics.x_dim], dtype=tf.float32)

  for _ in range(n_iters):
    _, grads, samples = loss_and_grads(samples, dynamics)
    vars_ = dynamics.variables + dynamics.vars_not_in_layers
    optimizer.apply_gradients(zip(grads, vars_))


def fit(dynamics,
        optimizer,
        n_samples=200,
        n_iters=5000,
        verbose=True,
        logdir=None):
  """Fit L2HMC sampler with given log-likelihood function."""

  if logdir:
    summary_writer = tf.contrib.summary.create_file_writer(logdir)

  samples = tf.random_normal(
      shape=[n_samples, dynamics.x_dim], dtype=tf.float32)

  tf.train.get_or_create_global_step()
  for i in range(n_iters):
    loss, grads, samples = loss_and_grads(samples, dynamics)
    # TODO(lxuechen): Proper learning rate decay
    grads_ = [grad * .96**(i // 1000) for grad in grads]
    vars_ = dynamics.variables + dynamics.vars_not_in_layers
    optimizer.apply_gradients(
        zip(grads_, vars_), global_step=tf.train.get_global_step())

    if verbose:
      print("Iteration %d: loss %.4f" % (i, loss))

    if logdir:
      with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
          tf.contrib.summary.scalar("loss", loss)


def get_scg_energy_fn():
  """Get energy function for 2d strongly correlated Gaussian."""

  # Avoid recreating tf constants on each invocation of gradients
  mu = tf.constant([0., 0.])
  sigma = tf.constant([[50.05, -49.95], [-49.95, 50.05]])
  sigma_inv = tf.matrix_inverse(sigma)

  def energy(x):
    """Unnormalized log density/energy of 2d strongly correlated Gaussian."""

    xmmu = x - mu
    return .5 * tf.diag_part(
        tf.matmul(tf.matmul(xmmu, sigma_inv), tf.transpose(xmmu)))

  return energy
