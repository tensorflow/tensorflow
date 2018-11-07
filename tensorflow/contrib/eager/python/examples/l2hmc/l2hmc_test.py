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
"""Tests l2hmc fit to 2D strongly correlated Gaussian executed eagerly."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

import numpy.random as npr
import tensorflow as tf
import tensorflow.contrib.eager as tfe
from tensorflow.contrib.eager.python.examples.l2hmc import l2hmc


def get_default_hparams():
  return tf.contrib.training.HParams(
      x_dim=2,
      n_samples=200,
      n_steps=10,
      eps=.1,
      n_iters=10,
      learning_rate=.0003,
      n_warmup_iters=3)


def step(dynamics, optimizer, samples):
  loss, grads, samples, _ = l2hmc.loss_and_grads(
      dynamics, samples, loss_fn=l2hmc.compute_loss)
  optimizer.apply_gradients(zip(grads, dynamics.variables))

  return loss, samples


# To be defunnable, the function cannot return an Operation, so the above
# function is used for defun or eager, and this function is used in graph to be
# able to run the gradient updates.
def graph_step(dynamics, optimizer, samples):
  loss, grads, samples, _ = l2hmc.loss_and_grads(
      dynamics, samples, loss_fn=l2hmc.compute_loss)
  train_op = optimizer.apply_gradients(zip(grads, dynamics.variables))

  return train_op, loss, samples


def warmup(dynamics,
           optimizer,
           n_iters=1,
           n_samples=200,
           step_fn=step):
  """Warmup optimization to reduce overhead."""

  samples = tf.random_normal(
      shape=[n_samples, dynamics.x_dim], dtype=tf.float32)

  for _ in range(n_iters):
    _, samples = step_fn(dynamics, optimizer, samples)


def fit(dynamics,
        samples,
        optimizer,
        step_fn=step,
        n_iters=5000,
        verbose=True,
        logdir=None):
  """Fit L2HMC sampler with given log-likelihood function."""

  if logdir:
    summary_writer = tf.contrib.summary.create_file_writer(logdir)

  for i in range(n_iters):
    loss, samples = step_fn(dynamics, optimizer, samples)
    if verbose:
      print("Iteration %d: loss %.4f" % (i, loss))

    if logdir:
      with summary_writer.as_default():
        with tf.contrib.summary.always_record_summaries():
          tf.contrib.summary.scalar("loss", loss)


class L2hmcTest(tf.test.TestCase):
  """Unit tests for l2hmc in both eager and graph mode."""

  def test_apply_transition(self):
    """Testing function `Dynamics.apply_transition` in graph and eager mode."""

    # Eager mode testing
    hparams = get_default_hparams()
    energy_fn, _, _ = l2hmc.get_scg_energy_fn()
    dynamics = l2hmc.Dynamics(
        x_dim=hparams.x_dim,
        minus_loglikelihood_fn=energy_fn,
        n_steps=hparams.n_steps,
        eps=hparams.eps)
    samples = tf.random_normal(shape=[hparams.n_samples, hparams.x_dim])
    x_, v_, x_accept_prob, x_out = dynamics.apply_transition(samples)

    self.assertEqual(x_.shape, v_.shape)
    self.assertEqual(x_out.shape, samples.shape)
    self.assertEqual(x_.shape, x_out.shape)
    self.assertEqual(x_accept_prob.shape, (hparams.n_samples,))

    # Graph mode testing
    with tf.Graph().as_default():
      energy_fn, _, _ = l2hmc.get_scg_energy_fn()
      dynamics = l2hmc.Dynamics(
          x_dim=hparams.x_dim,
          minus_loglikelihood_fn=energy_fn,
          n_steps=hparams.n_steps,
          eps=hparams.eps)
      x = tf.placeholder(tf.float32, shape=[None, hparams.x_dim])
      x_, v_, x_accept_prob, x_out = dynamics.apply_transition(x)
      samples = npr.normal(size=[hparams.n_samples, hparams.x_dim])

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        np_x_, np_v_, np_x_accept_prob, np_x_out = sess.run(
            [x_, v_, x_accept_prob, x_out], feed_dict={x: samples})

        self.assertEqual(np_x_.shape, np_v_.shape)
        self.assertEqual(samples.shape, np_x_out.shape)
        self.assertEqual(np_x_.shape, np_x_out.shape)
        self.assertEqual(np_x_accept_prob.shape, (hparams.n_samples,))


class L2hmcBenchmark(tf.test.Benchmark):
  """Eager and graph benchmarks for l2hmc."""

  def benchmark_graph(self):
    """Benchmark Graph performance."""

    hparams = get_default_hparams()
    tf.enable_resource_variables()
    for sample_size in [10, 25, 50, 100, 200]:
      hparams.n_samples = sample_size
      tf.reset_default_graph()
      with tf.Graph().as_default():
        energy_fn, _, _ = l2hmc.get_scg_energy_fn()
        x = tf.random_normal([hparams.n_samples, hparams.x_dim],
                             dtype=tf.float32)
        dynamics = l2hmc.Dynamics(
            x_dim=hparams.x_dim,
            minus_loglikelihood_fn=energy_fn,
            n_steps=hparams.n_steps,
            eps=hparams.eps)
        loss, _, _ = l2hmc.compute_loss(dynamics, x)

        optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
        train_op, loss, _ = graph_step(dynamics, optimizer, x)

        # Single thread; fairer comparison against eager
        session_conf = tf.ConfigProto(inter_op_parallelism_threads=1)

        with tf.Session(config=session_conf) as sess:
          sess.run(tf.global_variables_initializer())

          # Warmup to reduce initialization effect when timing
          for _ in range(hparams.n_warmup_iters):
            _, _ = sess.run([train_op, loss])

          # Training
          start_time = time.time()
          for i in range(hparams.n_iters):
            _, loss_np = sess.run([train_op, loss])
            print("Iteration %d: loss %.4f" % (i, loss_np))
          wall_time = (time.time() - start_time) / hparams.n_iters
          examples_per_sec = hparams.n_samples / wall_time

          self.report_benchmark(
              name="graph_train_%s_%d" %
              ("gpu" if tf.test.is_gpu_available() else "cpu", sample_size),
              iters=hparams.n_iters,
              extras={"examples_per_sec": examples_per_sec},
              wall_time=wall_time)

  def benchmark_eager(self):
    self._benchmark_eager()

  def benchmark_eager_defun(self):
    self._benchmark_eager(defun=True)

  def _benchmark_eager(self, defun=False):
    """Benchmark Eager performance."""

    hparams = get_default_hparams()
    for sample_size in [10, 25, 50, 100, 200]:
      hparams.n_samples = sample_size
      energy_fn, _, _ = l2hmc.get_scg_energy_fn()
      dynamics = l2hmc.Dynamics(
          x_dim=hparams.x_dim,
          minus_loglikelihood_fn=energy_fn,
          n_steps=hparams.n_steps,
          eps=hparams.eps)
      optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)
      step_fn = tfe.defun(step) if defun else step

      # Warmup to reduce initialization effect when timing
      warmup(
          dynamics,
          optimizer,
          n_iters=hparams.n_warmup_iters,
          n_samples=hparams.n_samples,
          step_fn=step_fn)

      # Training
      samples = tf.random_normal(
          shape=[hparams.n_samples, hparams.x_dim], dtype=tf.float32)
      start_time = time.time()
      fit(dynamics,
          samples,
          optimizer,
          step_fn=step_fn,
          n_iters=hparams.n_iters)
      wall_time = (time.time() - start_time) / hparams.n_iters
      examples_per_sec = hparams.n_samples / wall_time

      self.report_benchmark(
          name="eager_train_%s%s_%d" %
          ("gpu" if tf.test.is_gpu_available() else "cpu",
           "_defun" if defun else "", sample_size),
          iters=hparams.n_iters,
          extras={"examples_per_sec": examples_per_sec},
          wall_time=wall_time)

    del dynamics


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
