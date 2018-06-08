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
      n_iters=5,
      learning_rate=.001,
      n_warmup_iters=1)


class L2hmcTest(tf.test.TestCase):
  """Unit tests for l2hmc in both eager and graph mode."""

  def testComputeLoss(self):
    """Testing function l2hmc.compute_loss in both graph and eager mode."""

    # Eager mode testing
    hparams = get_default_hparams()
    dynamics = l2hmc.Dynamics(
        x_dim=hparams.x_dim,
        loglikelihood_fn=l2hmc.get_scg_energy_fn(),
        n_steps=hparams.n_steps,
        eps=hparams.eps)
    samples = tf.random_normal(shape=[hparams.n_samples, hparams.x_dim])
    loss, x_out = l2hmc.compute_loss(samples, dynamics)

    # Check shape and numerical stability
    self.assertEqual(x_out.shape, samples.shape)
    self.assertEqual(loss.shape, [])
    self.assertAllClose(loss.numpy(), loss.numpy(), rtol=1e-5)

    # Graph mode testing
    with tf.Graph().as_default():
      dynamics = l2hmc.Dynamics(
          x_dim=hparams.x_dim,
          loglikelihood_fn=l2hmc.get_scg_energy_fn(),
          n_steps=hparams.n_steps,
          eps=hparams.eps)
      x = tf.placeholder(tf.float32, shape=[None, hparams.x_dim])
      loss, x_out = l2hmc.compute_loss(x, dynamics)
      samples = npr.normal(size=[hparams.n_samples, hparams.x_dim])

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        loss_np, x_out_np = sess.run([loss, x_out], feed_dict={x: samples})

        # Check shape and numerical stability
        self.assertEqual(x_out_np.shape, samples.shape)
        self.assertEqual(loss_np.shape, ())
        self.assertAllClose(loss_np, loss_np, rtol=1e-5)


class L2hmcBenchmark(tf.test.Benchmark):
  """Eager and graph benchmarks for l2hmc."""

  def benchmarkEagerL2hmc(self):
    """Benchmark Eager performance."""

    hparams = get_default_hparams()
    dynamics = l2hmc.Dynamics(
        x_dim=hparams.x_dim,
        loglikelihood_fn=l2hmc.get_scg_energy_fn(),
        n_steps=hparams.n_steps,
        eps=hparams.eps)
    # TODO(lxuechen): Add learning rate decay
    optimizer = tf.train.AdamOptimizer(learning_rate=hparams.learning_rate)

    # Warmup to reduce initialization effect when timing
    l2hmc.warmup(dynamics, optimizer, n_iters=hparams.n_warmup_iters)

    # Time
    start_time = time.time()
    l2hmc.fit(
        dynamics,
        optimizer,
        n_samples=hparams.n_samples,
        n_iters=hparams.n_iters)
    wall_time = time.time() - start_time
    examples_per_sec = hparams.n_samples / wall_time

    self.report_benchmark(
        name="eager_train_%s" % ("gpu" if tfe.num_gpus() > 0 else "cpu"),
        iters=hparams.n_iters,
        extras={"examples_per_sec": examples_per_sec},
        wall_time=wall_time)

  def benchmarkGraphL2hmc(self):
    """Benchmark Graph performance."""

    hparams = get_default_hparams()
    with tf.Graph().as_default():
      dynamics = l2hmc.Dynamics(
          x_dim=hparams.x_dim,
          loglikelihood_fn=l2hmc.get_scg_energy_fn(),
          n_steps=hparams.n_steps,
          eps=hparams.eps)
      x = tf.placeholder(tf.float32, shape=[None, hparams.x_dim])
      loss, x_out = l2hmc.compute_loss(x, dynamics)

      global_step = tf.Variable(0., name="global_step", trainable=False)
      learning_rate = tf.train.exponential_decay(
          hparams.learning_rate, global_step, 1000, 0.96, staircase=True)
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
      train_op = optimizer.minimize(loss, global_step=global_step)

      with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Warmup to reduce initialization effect when timing
        samples = npr.normal(size=[hparams.n_samples, hparams.x_dim])
        for _ in range(hparams.n_warmup_iters):
          samples, _, _, _ = sess.run(
              [x_out, loss, train_op, learning_rate], feed_dict={x: samples})

        # Time
        start_time = time.time()
        for _ in range(hparams.n_iters):
          samples, _, _, _ = sess.run(
              [x_out, loss, train_op, learning_rate], feed_dict={x: samples})
        wall_time = time.time() - start_time
        examples_per_sec = hparams.n_samples / wall_time

        self.report_benchmark(
            name="graph_train_%s" % ("gpu"
                                     if tf.test.is_gpu_available() else "cpu"),
            iters=hparams.n_iters,
            extras={"examples_per_sec": examples_per_sec},
            wall_time=wall_time)


if __name__ == "__main__":
  tf.enable_eager_execution()
  tf.test.main()
