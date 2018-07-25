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
"""L2HMC on simple Gaussian mixture model with TensorFlow eager."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

from absl import flags
import numpy as np
import tensorflow as tf
from tensorflow.contrib.eager.python.examples.l2hmc import l2hmc
try:
  import matplotlib.pyplot as plt  # pylint: disable=g-import-not-at-top
  HAS_MATPLOTLIB = True
except ImportError:
  HAS_MATPLOTLIB = False
tfe = tf.contrib.eager


def main(_):
  tf.enable_eager_execution()
  global_step = tf.train.get_or_create_global_step()
  global_step.assign(1)

  energy_fn, mean, covar = {
      "scg": l2hmc.get_scg_energy_fn(),
      "rw": l2hmc.get_rw_energy_fn()
  }[FLAGS.energy_fn]

  x_dim = 2
  train_iters = 5000
  eval_iters = 2000
  eps = 0.1
  n_steps = 10  # Chain length
  n_samples = 200
  record_loss_every = 100

  dynamics = l2hmc.Dynamics(
      x_dim=x_dim, minus_loglikelihood_fn=energy_fn, n_steps=n_steps, eps=eps)
  learning_rate = tf.train.exponential_decay(
      1e-3, global_step, 1000, 0.96, staircase=True)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  checkpointer = tf.train.Checkpoint(
      optimizer=optimizer, dynamics=dynamics, global_step=global_step)

  if FLAGS.train_dir:
    summary_writer = tf.contrib.summary.create_file_writer(FLAGS.train_dir)
    if FLAGS.restore:
      latest_path = tf.train.latest_checkpoint(FLAGS.train_dir)
      checkpointer.restore(latest_path)
      print("Restored latest checkpoint at path:\"{}\" ".format(latest_path))
      sys.stdout.flush()

  if not FLAGS.restore:
    # Training
    if FLAGS.use_defun:
      # Use `tfe.deun` to boost performance when there are lots of small ops
      loss_fn = tfe.defun(l2hmc.compute_loss)
    else:
      loss_fn = l2hmc.compute_loss

    samples = tf.random_normal(shape=[n_samples, x_dim])
    for i in range(1, train_iters + 1):
      loss, samples, accept_prob = train_one_iter(
          dynamics,
          samples,
          optimizer,
          loss_fn=loss_fn,
          global_step=global_step)

      if i % record_loss_every == 0:
        print("Iteration {}, loss {:.4f}, x_accept_prob {:.4f}".format(
            i, loss.numpy(),
            accept_prob.numpy().mean()))
        if FLAGS.train_dir:
          with summary_writer.as_default():
            with tf.contrib.summary.always_record_summaries():
              tf.contrib.summary.scalar("Training loss", loss, step=global_step)
    print("Training complete.")
    sys.stdout.flush()

    if FLAGS.train_dir:
      saved_path = checkpointer.save(
          file_prefix=os.path.join(FLAGS.train_dir, "ckpt"))
      print("Saved checkpoint at path: \"{}\" ".format(saved_path))
      sys.stdout.flush()

  # Evaluation
  if FLAGS.use_defun:
    # Use tfe.deun to boost performance when there are lots of small ops
    apply_transition = tfe.defun(dynamics.apply_transition)
  else:
    apply_transition = dynamics.apply_transition

  samples = tf.random_normal(shape=[n_samples, x_dim])
  samples_history = []
  for i in range(eval_iters):
    samples_history.append(samples.numpy())
    _, _, _, samples = apply_transition(samples)
  samples_history = np.array(samples_history)
  print("Sampling complete.")
  sys.stdout.flush()

  # Mean and covariance of target distribution
  mean = mean.numpy()
  covar = covar.numpy()
  ac_spectrum = compute_ac_spectrum(samples_history, mean, covar)
  print("First 25 entries of the auto-correlation spectrum: {}".format(
      ac_spectrum[:25]))
  ess = compute_ess(ac_spectrum)
  print("Effective sample size per Metropolis-Hastings step: {}".format(ess))
  sys.stdout.flush()

  if FLAGS.train_dir:
    # Plot autocorrelation spectrum in tensorboard
    plot_step = tfe.Variable(1, trainable=False, dtype=tf.int64)

    for ac in ac_spectrum:
      with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
          tf.contrib.summary.scalar("Autocorrelation", ac, step=plot_step)
      plot_step.assign(plot_step + n_steps)

    if HAS_MATPLOTLIB:
      # Choose a single chain and plot the trajectory
      single_chain = samples_history[:, 0, :]
      xs = single_chain[:100, 0]
      ys = single_chain[:100, 1]
      plt.figure()
      plt.plot(xs, ys, color="orange", marker="o", alpha=0.6)  # Trained chain
      plt.savefig(os.path.join(FLAGS.train_dir, "single_chain.png"))


def train_one_iter(dynamics,
                   x,
                   optimizer,
                   loss_fn=l2hmc.compute_loss,
                   global_step=None):
  """Train the sampler for one iteration."""
  loss, grads, out, accept_prob = l2hmc.loss_and_grads(
      dynamics, x, loss_fn=loss_fn)
  optimizer.apply_gradients(
      zip(grads, dynamics.trainable_variables), global_step=global_step)

  return loss, out, accept_prob


def compute_ac_spectrum(samples_history, target_mean, target_covar):
  """Compute autocorrelation spectrum.

  Follows equation 15 from the L2HMC paper.

  Args:
    samples_history: Numpy array of shape [T, B, D], where T is the total
        number of time steps, B is the batch size, and D is the dimensionality
        of sample space.
    target_mean: 1D Numpy array of the mean of target(true) distribution.
    target_covar: 2D Numpy array representing a symmetric matrix for variance.
  Returns:
    Autocorrelation spectrum, Numpy array of shape [T-1].
  """

  # Using numpy here since eager is a bit slow due to the loop
  time_steps = samples_history.shape[0]
  trace = np.trace(target_covar)

  rhos = []
  for t in range(time_steps - 1):
    rho_t = 0.
    for tau in range(time_steps - t):
      v_tau = samples_history[tau, :, :] - target_mean
      v_tau_plus_t = samples_history[tau + t, :, :] - target_mean
      # Take dot product over observation dims and take mean over batch dims
      rho_t += np.mean(np.sum(v_tau * v_tau_plus_t, axis=1))

    rho_t /= trace * (time_steps - t)
    rhos.append(rho_t)

  return np.array(rhos)


def compute_ess(ac_spectrum):
  """Compute the effective sample size based on autocorrelation spectrum.

  This follows equation 16 from the L2HMC paper.

  Args:
    ac_spectrum: Autocorrelation spectrum
  Returns:
    The effective sample size
  """
  # Cutoff from the first value less than 0.05
  cutoff = np.argmax(ac_spectrum[1:] < .05)
  if cutoff == 0:
    cutoff = len(ac_spectrum)
  ess = 1. / (1. + 2. * np.sum(ac_spectrum[1:cutoff]))
  return ess


if __name__ == "__main__":
  flags.DEFINE_string(
      "train_dir",
      default=None,
      help="[Optional] Directory to store the training information")
  flags.DEFINE_boolean(
      "restore",
      default=False,
      help="[Optional] Restore the latest checkpoint from `train_dir` if True")
  flags.DEFINE_boolean(
      "use_defun",
      default=False,
      help="[Optional] Use `tfe.defun` to boost performance")
  flags.DEFINE_string(
      "energy_fn",
      default="scg",
      help="[Optional] The energy function used for experimentation"
      "Other options include `rw`")
  FLAGS = flags.FLAGS
  tf.app.run(main)
