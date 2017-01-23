# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for Mixture distribution."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib

import numpy as np
from tensorflow.contrib import distributions
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import random_seed
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

distributions_py = distributions


def _swap_first_last_axes(array):
  rank = len(array.shape)
  transpose = [rank - 1] + list(range(0, rank - 1))
  return array.transpose(transpose)


@contextlib.contextmanager
def _test_capture_mvndiag_sample_outputs():
  """Use monkey-patching to capture the output of an MVNDiag _sample_n."""
  data_container = []
  true_mvndiag_sample_n = distributions_py.MultivariateNormalDiag._sample_n

  def _capturing_mvndiag_sample_n(self, n, seed=None):
    samples = true_mvndiag_sample_n(self, n=n, seed=seed)
    data_container.append(samples)
    return samples

  distributions_py.MultivariateNormalDiag._sample_n = (
      _capturing_mvndiag_sample_n)
  yield data_container
  distributions_py.MultivariateNormalDiag._sample_n = true_mvndiag_sample_n


@contextlib.contextmanager
def _test_capture_normal_sample_outputs():
  """Use monkey-patching to capture the output of an Normal _sample_n."""
  data_container = []
  true_normal_sample_n = distributions_py.Normal._sample_n

  def _capturing_normal_sample_n(self, n, seed=None):
    samples = true_normal_sample_n(self, n=n, seed=seed)
    data_container.append(samples)
    return samples

  distributions_py.Normal._sample_n = _capturing_normal_sample_n
  yield data_container
  distributions_py.Normal._sample_n = true_normal_sample_n


def make_univariate_mixture(batch_shape, num_components):
  logits = random_ops.random_uniform(
      list(batch_shape) + [num_components], -1, 1, dtype=dtypes.float32) - 50.
  components = [
      distributions_py.Normal(
          mu=np.float32(np.random.randn(*list(batch_shape))),
          sigma=np.float32(10 * np.random.rand(*list(batch_shape))))
      for _ in range(num_components)
  ]
  cat = distributions_py.Categorical(logits, dtype=dtypes.int32)
  return distributions_py.Mixture(cat, components)


def make_multivariate_mixture(batch_shape, num_components, event_shape):
  logits = random_ops.random_uniform(
      list(batch_shape) + [num_components], -1, 1, dtype=dtypes.float32) - 50.
  components = [
      distributions_py.MultivariateNormalDiag(
          mu=np.float32(np.random.randn(*list(batch_shape + event_shape))),
          diag_stdev=np.float32(10 * np.random.rand(
              *list(batch_shape + event_shape)))) for _ in range(num_components)
  ]
  cat = distributions_py.Categorical(logits, dtype=dtypes.int32)
  return distributions_py.Mixture(cat, components)


class MixtureTest(test.TestCase):

  def testShapes(self):
    with self.test_session():
      for batch_shape in ([], [1], [2, 3, 4]):
        dist = make_univariate_mixture(batch_shape, num_components=10)
        self.assertAllEqual(batch_shape, dist.get_batch_shape())
        self.assertAllEqual(batch_shape, dist.batch_shape().eval())
        self.assertAllEqual([], dist.get_event_shape())
        self.assertAllEqual([], dist.event_shape().eval())

        for event_shape in ([1], [2]):
          dist = make_multivariate_mixture(
              batch_shape, num_components=10, event_shape=event_shape)
          self.assertAllEqual(batch_shape, dist.get_batch_shape())
          self.assertAllEqual(batch_shape, dist.batch_shape().eval())
          self.assertAllEqual(event_shape, dist.get_event_shape())
          self.assertAllEqual(event_shape, dist.event_shape().eval())

  def testBrokenShapesStatic(self):
    with self.assertRaisesWithPredicateMatch(ValueError,
                                             r"cat.num_classes != len"):
      distributions_py.Mixture(
          distributions_py.Categorical([0.1, 0.5]),  # 2 classes
          [distributions_py.Normal(
              mu=1.0, sigma=2.0)])
    with self.assertRaisesWithPredicateMatch(
        ValueError, r"\(\) and \(2,\) are not compatible"):
      # The value error is raised because the batch shapes of the
      # Normals are not equal.  One is a scalar, the other is a
      # vector of size (2,).
      distributions_py.Mixture(
          distributions_py.Categorical([-0.5, 0.5]),  # scalar batch
          [
              distributions_py.Normal(
                  mu=1.0, sigma=2.0),  # scalar dist
              distributions_py.Normal(
                  mu=[1.0, 1.0], sigma=[2.0, 2.0])
          ])
    with self.assertRaisesWithPredicateMatch(ValueError, r"Could not infer"):
      cat_logits = array_ops.placeholder(shape=[1, None], dtype=dtypes.float32)
      distributions_py.Mixture(
          distributions_py.Categorical(cat_logits),
          [distributions_py.Normal(
              mu=[1.0], sigma=[2.0])])

  def testBrokenShapesDynamic(self):
    with self.test_session():
      d0_param = array_ops.placeholder(dtype=dtypes.float32)
      d1_param = array_ops.placeholder(dtype=dtypes.float32)
      d = distributions_py.Mixture(
          distributions_py.Categorical([0.1, 0.2]), [
              distributions_py.Normal(
                  mu=d0_param, sigma=d0_param), distributions_py.Normal(
                      mu=d1_param, sigma=d1_param)
          ],
          validate_args=True)
      with self.assertRaisesOpError(r"batch shape must match"):
        d.sample().eval(feed_dict={d0_param: [2.0, 3.0], d1_param: [1.0]})
      with self.assertRaisesOpError(r"batch shape must match"):
        d.sample().eval(feed_dict={d0_param: [2.0, 3.0], d1_param: 1.0})

  def testBrokenTypes(self):
    with self.assertRaisesWithPredicateMatch(TypeError, "Categorical"):
      distributions_py.Mixture(None, [])
    cat = distributions_py.Categorical([0.3, 0.2])
    # components must be a list of distributions
    with self.assertRaisesWithPredicateMatch(
        TypeError, "all .* must be Distribution instances"):
      distributions_py.Mixture(cat, [None])
    with self.assertRaisesWithPredicateMatch(TypeError, "same dtype"):
      distributions_py.Mixture(
          cat, [
              distributions_py.Normal(
                  mu=[1.0], sigma=[2.0]), distributions_py.Normal(
                      mu=[np.float16(1.0)], sigma=[np.float16(2.0)])
          ])
    with self.assertRaisesWithPredicateMatch(ValueError, "non-empty list"):
      distributions_py.Mixture(distributions_py.Categorical([0.3, 0.2]), None)
    with self.assertRaisesWithPredicateMatch(TypeError,
                                             "either be continuous or not"):
      distributions_py.Mixture(
          cat, [
              distributions_py.Normal(
                  mu=[1.0], sigma=[2.0]), distributions_py.Bernoulli(
                      dtype=dtypes.float32, logits=[1.0])
          ])

  def testMeanUnivariate(self):
    with self.test_session() as sess:
      for batch_shape in ((), (2,), (2, 3)):
        dist = make_univariate_mixture(
            batch_shape=batch_shape, num_components=2)
        mean = dist.mean()
        self.assertEqual(batch_shape, mean.get_shape())

        cat_probs = nn_ops.softmax(dist.cat.logits)
        dist_means = [d.mean() for d in dist.components]

        mean_value, cat_probs_value, dist_means_value = sess.run(
            [mean, cat_probs, dist_means])
        self.assertEqual(batch_shape, mean_value.shape)

        cat_probs_value = _swap_first_last_axes(cat_probs_value)
        true_mean = sum(
            [c_p * m for (c_p, m) in zip(cat_probs_value, dist_means_value)])

        self.assertAllClose(true_mean, mean_value)

  def testMeanMultivariate(self):
    with self.test_session() as sess:
      for batch_shape in ((), (2,), (2, 3)):
        dist = make_multivariate_mixture(
            batch_shape=batch_shape, num_components=2, event_shape=(4,))
        mean = dist.mean()
        self.assertEqual(batch_shape + (4,), mean.get_shape())

        cat_probs = nn_ops.softmax(dist.cat.logits)
        dist_means = [d.mean() for d in dist.components]

        mean_value, cat_probs_value, dist_means_value = sess.run(
            [mean, cat_probs, dist_means])
        self.assertEqual(batch_shape + (4,), mean_value.shape)

        cat_probs_value = _swap_first_last_axes(cat_probs_value)

        # Add a new innermost dimension for broadcasting to mvn vector shape
        cat_probs_value = [np.expand_dims(c_p, -1) for c_p in cat_probs_value]

        true_mean = sum(
            [c_p * m for (c_p, m) in zip(cat_probs_value, dist_means_value)])

        self.assertAllClose(true_mean, mean_value)

  def testProbScalarUnivariate(self):
    with self.test_session() as sess:
      dist = make_univariate_mixture(batch_shape=[], num_components=2)
      for x in [
          np.array(
              [1.0, 2.0], dtype=np.float32), np.array(
                  1.0, dtype=np.float32),
          np.random.randn(3, 4).astype(np.float32)
      ]:
        p_x = dist.prob(x)

        self.assertEqual(x.shape, p_x.get_shape())
        cat_probs = nn_ops.softmax([dist.cat.logits])[0]
        dist_probs = [d.prob(x) for d in dist.components]

        p_x_value, cat_probs_value, dist_probs_value = sess.run(
            [p_x, cat_probs, dist_probs])
        self.assertEqual(x.shape, p_x_value.shape)

        total_prob = sum(c_p_value * d_p_value
                         for (c_p_value, d_p_value
                             ) in zip(cat_probs_value, dist_probs_value))

        self.assertAllClose(total_prob, p_x_value)

  def testProbScalarMultivariate(self):
    with self.test_session() as sess:
      dist = make_multivariate_mixture(
          batch_shape=[], num_components=2, event_shape=[3])
      for x in [
          np.array(
              [[-1.0, 0.0, 1.0], [0.5, 1.0, -0.3]], dtype=np.float32), np.array(
                  [-1.0, 0.0, 1.0], dtype=np.float32),
          np.random.randn(2, 2, 3).astype(np.float32)
      ]:
        p_x = dist.prob(x)

        self.assertEqual(x.shape[:-1], p_x.get_shape())

        cat_probs = nn_ops.softmax([dist.cat.logits])[0]
        dist_probs = [d.prob(x) for d in dist.components]

        p_x_value, cat_probs_value, dist_probs_value = sess.run(
            [p_x, cat_probs, dist_probs])

        self.assertEqual(x.shape[:-1], p_x_value.shape)

        total_prob = sum(c_p_value * d_p_value
                         for (c_p_value, d_p_value
                             ) in zip(cat_probs_value, dist_probs_value))

        self.assertAllClose(total_prob, p_x_value)

  def testProbBatchUnivariate(self):
    with self.test_session() as sess:
      dist = make_univariate_mixture(batch_shape=[2, 3], num_components=2)

      for x in [
          np.random.randn(2, 3).astype(np.float32),
          np.random.randn(4, 2, 3).astype(np.float32)
      ]:
        p_x = dist.prob(x)
        self.assertEqual(x.shape, p_x.get_shape())

        cat_probs = nn_ops.softmax(dist.cat.logits)
        dist_probs = [d.prob(x) for d in dist.components]

        p_x_value, cat_probs_value, dist_probs_value = sess.run(
            [p_x, cat_probs, dist_probs])
        self.assertEqual(x.shape, p_x_value.shape)

        cat_probs_value = _swap_first_last_axes(cat_probs_value)

        total_prob = sum(c_p_value * d_p_value
                         for (c_p_value, d_p_value
                             ) in zip(cat_probs_value, dist_probs_value))

        self.assertAllClose(total_prob, p_x_value)

  def testProbBatchMultivariate(self):
    with self.test_session() as sess:
      dist = make_multivariate_mixture(
          batch_shape=[2, 3], num_components=2, event_shape=[4])

      for x in [
          np.random.randn(2, 3, 4).astype(np.float32),
          np.random.randn(4, 2, 3, 4).astype(np.float32)
      ]:
        p_x = dist.prob(x)
        self.assertEqual(x.shape[:-1], p_x.get_shape())

        cat_probs = nn_ops.softmax(dist.cat.logits)
        dist_probs = [d.prob(x) for d in dist.components]

        p_x_value, cat_probs_value, dist_probs_value = sess.run(
            [p_x, cat_probs, dist_probs])
        self.assertEqual(x.shape[:-1], p_x_value.shape)

        cat_probs_value = _swap_first_last_axes(cat_probs_value)
        total_prob = sum(c_p_value * d_p_value
                         for (c_p_value, d_p_value
                             ) in zip(cat_probs_value, dist_probs_value))

        self.assertAllClose(total_prob, p_x_value)

  def testSampleScalarBatchUnivariate(self):
    with self.test_session() as sess:
      num_components = 3
      dist = make_univariate_mixture(
          batch_shape=[], num_components=num_components)
      n = 4
      with _test_capture_normal_sample_outputs() as component_samples:
        samples = dist.sample(n, seed=123)
      self.assertEqual(samples.dtype, dtypes.float32)
      self.assertEqual((4,), samples.get_shape())
      cat_samples = dist.cat.sample(n, seed=123)
      sample_values, cat_sample_values, dist_sample_values = sess.run(
          [samples, cat_samples, component_samples])
      self.assertEqual((4,), sample_values.shape)

      for c in range(num_components):
        which_c = np.where(cat_sample_values == c)[0]
        size_c = which_c.size
        # Scalar Batch univariate case: batch_size == 1, rank 1
        which_dist_samples = dist_sample_values[c][:size_c]
        self.assertAllClose(which_dist_samples, sample_values[which_c])

  # Test that sampling with the same seed twice gives the same results.
  def testSampleMultipleTimes(self):
    # 5 component mixture.
    logits = [-10.0, -5.0, 0.0, 5.0, 10.0]
    mus = [-5.0, 0.0, 5.0, 4.0, 20.0]
    sigmas = [0.1, 5.0, 3.0, 0.2, 4.0]

    with self.test_session():
      n = 100

      random_seed.set_random_seed(654321)
      components = [
          distributions_py.Normal(
              mu=mu, sigma=sigma) for mu, sigma in zip(mus, sigmas)
      ]
      cat = distributions_py.Categorical(
          logits, dtype=dtypes.int32, name="cat1")
      dist1 = distributions_py.Mixture(cat, components, name="mixture1")
      samples1 = dist1.sample(n, seed=123456).eval()

      random_seed.set_random_seed(654321)
      components2 = [
          distributions_py.Normal(
              mu=mu, sigma=sigma) for mu, sigma in zip(mus, sigmas)
      ]
      cat2 = distributions_py.Categorical(
          logits, dtype=dtypes.int32, name="cat2")
      dist2 = distributions_py.Mixture(cat2, components2, name="mixture2")
      samples2 = dist2.sample(n, seed=123456).eval()

      self.assertAllClose(samples1, samples2)

  def testSampleScalarBatchMultivariate(self):
    with self.test_session() as sess:
      num_components = 3
      dist = make_multivariate_mixture(
          batch_shape=[], num_components=num_components, event_shape=[2])
      n = 4
      with _test_capture_mvndiag_sample_outputs() as component_samples:
        samples = dist.sample(n, seed=123)
      self.assertEqual(samples.dtype, dtypes.float32)
      self.assertEqual((4, 2), samples.get_shape())
      cat_samples = dist.cat.sample(n, seed=123)
      sample_values, cat_sample_values, dist_sample_values = sess.run(
          [samples, cat_samples, component_samples])
      self.assertEqual((4, 2), sample_values.shape)
      for c in range(num_components):
        which_c = np.where(cat_sample_values == c)[0]
        size_c = which_c.size
        # Scalar Batch multivariate case: batch_size == 1, rank 2
        which_dist_samples = dist_sample_values[c][:size_c, :]
        self.assertAllClose(which_dist_samples, sample_values[which_c, :])

  def testSampleBatchUnivariate(self):
    with self.test_session() as sess:
      num_components = 3
      dist = make_univariate_mixture(
          batch_shape=[2, 3], num_components=num_components)
      n = 4
      with _test_capture_normal_sample_outputs() as component_samples:
        samples = dist.sample(n, seed=123)
      self.assertEqual(samples.dtype, dtypes.float32)
      self.assertEqual((4, 2, 3), samples.get_shape())
      cat_samples = dist.cat.sample(n, seed=123)
      sample_values, cat_sample_values, dist_sample_values = sess.run(
          [samples, cat_samples, component_samples])
      self.assertEqual((4, 2, 3), sample_values.shape)
      for c in range(num_components):
        which_c_s, which_c_b0, which_c_b1 = np.where(cat_sample_values == c)
        size_c = which_c_s.size
        # Batch univariate case: batch_size == [2, 3], rank 3
        which_dist_samples = dist_sample_values[c][range(size_c), which_c_b0,
                                                   which_c_b1]
        self.assertAllClose(which_dist_samples,
                            sample_values[which_c_s, which_c_b0, which_c_b1])

  def testSampleBatchMultivariate(self):
    with self.test_session() as sess:
      num_components = 3
      dist = make_multivariate_mixture(
          batch_shape=[2, 3], num_components=num_components, event_shape=[4])
      n = 5
      with _test_capture_mvndiag_sample_outputs() as component_samples:
        samples = dist.sample(n, seed=123)
      self.assertEqual(samples.dtype, dtypes.float32)
      self.assertEqual((5, 2, 3, 4), samples.get_shape())
      cat_samples = dist.cat.sample(n, seed=123)
      sample_values, cat_sample_values, dist_sample_values = sess.run(
          [samples, cat_samples, component_samples])
      self.assertEqual((5, 2, 3, 4), sample_values.shape)

      for c in range(num_components):
        which_c_s, which_c_b0, which_c_b1 = np.where(cat_sample_values == c)
        size_c = which_c_s.size
        # Batch univariate case: batch_size == [2, 3], rank 4 (multivariate)
        which_dist_samples = dist_sample_values[c][range(size_c), which_c_b0,
                                                   which_c_b1, :]
        self.assertAllClose(which_dist_samples,
                            sample_values[which_c_s, which_c_b0, which_c_b1, :])

  def testEntropyLowerBoundMultivariate(self):
    with self.test_session() as sess:
      for batch_shape in ((), (2,), (2, 3)):
        dist = make_multivariate_mixture(
            batch_shape=batch_shape, num_components=2, event_shape=(4,))
        entropy_lower_bound = dist.entropy_lower_bound()
        self.assertEqual(batch_shape, entropy_lower_bound.get_shape())

        cat_probs = nn_ops.softmax(dist.cat.logits)
        dist_entropy = [d.entropy() for d in dist.components]

        entropy_lower_bound_value, cat_probs_value, dist_entropy_value = (
            sess.run([entropy_lower_bound, cat_probs, dist_entropy]))
        self.assertEqual(batch_shape, entropy_lower_bound_value.shape)

        cat_probs_value = _swap_first_last_axes(cat_probs_value)

        # entropy_lower_bound = sum_i pi_i entropy_i
        # for i in num_components, batchwise.
        true_entropy_lower_bound = sum(
            [c_p * m for (c_p, m) in zip(cat_probs_value, dist_entropy_value)])

        self.assertAllClose(true_entropy_lower_bound, entropy_lower_bound_value)


class MixtureBenchmark(test.Benchmark):

  def _runSamplingBenchmark(self, name, create_distribution, use_gpu,
                            num_components, batch_size, num_features,
                            sample_size):
    config = config_pb2.ConfigProto()
    config.allow_soft_placement = True
    np.random.seed(127)
    with session.Session(config=config, graph=ops.Graph()) as sess:
      random_seed.set_random_seed(0)
      with ops.device("/gpu:0" if use_gpu else "/cpu:0"):
        mixture = create_distribution(
            num_components=num_components,
            batch_size=batch_size,
            num_features=num_features)
        sample_op = mixture.sample(sample_size).op
        sess.run(variables.global_variables_initializer())
        reported = self.run_op_benchmark(
            sess,
            sample_op,
            min_iters=10,
            name=("%s_%s_components_%d_batch_%d_features_%d_sample_%d" %
                  (name, use_gpu, num_components, batch_size, num_features,
                   sample_size)))
        print("\t".join(["%s", "%d", "%d", "%d", "%d", "%g"]) %
              (use_gpu, num_components, batch_size, num_features, sample_size,
               reported["wall_time"]))

  def benchmarkSamplingMVNDiag(self):
    print("mvn_diag\tuse_gpu\tcomponents\tbatch\tfeatures\tsample\twall_time")

    def create_distribution(batch_size, num_components, num_features):
      cat = distributions_py.Categorical(
          logits=np.random.randn(batch_size, num_components))
      mus = [
          variables.Variable(np.random.randn(batch_size, num_features))
          for _ in range(num_components)
      ]
      sigmas = [
          variables.Variable(np.random.rand(batch_size, num_features))
          for _ in range(num_components)
      ]
      components = list(
          distributions_py.MultivariateNormalDiag(
              mu=mu, diag_stdev=sigma) for (mu, sigma) in zip(mus, sigmas))
      return distributions_py.Mixture(cat, components)

    for use_gpu in False, True:
      if use_gpu and not test.is_gpu_available():
        continue
      for num_components in 1, 8, 16:
        for batch_size in 1, 32:
          for num_features in 1, 64, 512:
            for sample_size in 1, 32, 128:
              self._runSamplingBenchmark(
                  "mvn_diag",
                  create_distribution=create_distribution,
                  use_gpu=use_gpu,
                  num_components=num_components,
                  batch_size=batch_size,
                  num_features=num_features,
                  sample_size=sample_size)

  def benchmarkSamplingMVNFull(self):
    print("mvn_full\tuse_gpu\tcomponents\tbatch\tfeatures\tsample\twall_time")

    def psd(x):
      """Construct batch-wise PSD matrices."""
      return np.stack([np.dot(np.transpose(z), z) for z in x])

    def create_distribution(batch_size, num_components, num_features):
      cat = distributions_py.Categorical(
          logits=np.random.randn(batch_size, num_components))
      mus = [
          variables.Variable(np.random.randn(batch_size, num_features))
          for _ in range(num_components)
      ]
      sigmas = [
          variables.Variable(
              psd(np.random.rand(batch_size, num_features, num_features)))
          for _ in range(num_components)
      ]
      components = list(
          distributions_py.MultivariateNormalFull(
              mu=mu, sigma=sigma) for (mu, sigma) in zip(mus, sigmas))
      return distributions_py.Mixture(cat, components)

    for use_gpu in False, True:
      if use_gpu and not test.is_gpu_available():
        continue
      for num_components in 1, 8, 16:
        for batch_size in 1, 32:
          for num_features in 1, 64, 512:
            for sample_size in 1, 32, 128:
              self._runSamplingBenchmark(
                  "mvn_full",
                  create_distribution=create_distribution,
                  use_gpu=use_gpu,
                  num_components=num_components,
                  batch_size=batch_size,
                  num_features=num_features,
                  sample_size=sample_size)


if __name__ == "__main__":
  test.main()
