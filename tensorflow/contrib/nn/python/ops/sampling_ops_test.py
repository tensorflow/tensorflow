# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for sampling_ops.py."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.nn.python.ops import sampling_ops
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test


class RankSampledSoftmaxLossTest(test.TestCase):

  def setUp(self):
    self._sampled = [3, 4, 5, 6, 7]
    self._num_sampled = len(self._sampled)
    # Because values of all matrices increase with indices, logits increase with
    # class id. So, for the above sampled classes, adaptive sampling will select
    # these resampled classes.
    self._resampled = [5, 6, 7]
    self._num_resampled = len(self._resampled)
    self._num_classes = 10
    self._num_true = 2
    self._sampled_values = (self._sampled, [[0.5], [0.5]],
                            [0.5, 0.5, 0.5, 0.5, 0.5])
    self._resampled_values = (self._resampled, [[0.5], [0.5]], [0.5, 0.5, 0.5])
    self._remove_accidental_hits = False
    self._embed_dim = 5
    self._batch_size = 2

  def _weights(self):
    return constant_op.constant([
        [0.0, 0.1, 0.2, 0.3, 0.4],
        [1.0, 1.1, 1.2, 1.3, 1.4],
        [2.0, 2.1, 2.2, 2.3, 2.4],
        [3.0, 3.1, 3.2, 3.3, 3.4],
        [4.0, 4.1, 4.2, 4.3, 4.4],
        [5.0, 5.1, 5.2, 5.3, 5.4],
        [6.0, 6.1, 6.2, 6.3, 6.4],
        [7.0, 7.1, 7.2, 7.3, 7.4],
        [8.0, 8.1, 8.2, 8.3, 8.4],
        [9.0, 9.1, 9.2, 9.3, 9.4],
    ])

  def _div_sharded_weights(self):
    return [
        constant_op.constant([
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [1.0, 1.1, 1.2, 1.3, 1.4],
        ]),
        constant_op.constant([
            [2.0, 2.1, 2.2, 2.3, 2.4],
            [3.0, 3.1, 3.2, 3.3, 3.4],
        ]),
        constant_op.constant([
            [4.0, 4.1, 4.2, 4.3, 4.4],
            [5.0, 5.1, 5.2, 5.3, 5.4],
        ]),
        constant_op.constant([
            [6.0, 6.1, 6.2, 6.3, 6.4],
            [7.0, 7.1, 7.2, 7.3, 7.4],
        ]),
        constant_op.constant([
            [8.0, 8.1, 8.2, 8.3, 8.4],
            [9.0, 9.1, 9.2, 9.3, 9.4],
        ]),
    ]

  def _mod_sharded_weights(self):
    return [
        constant_op.constant([
            [0.0, 0.1, 0.2, 0.3, 0.4],
            [5.0, 5.1, 5.2, 5.3, 5.4],
        ]),
        constant_op.constant([
            [1.0, 1.1, 1.2, 1.3, 1.4],
            [6.0, 6.1, 6.2, 6.3, 6.4],
        ]),
        constant_op.constant([
            [2.0, 2.1, 2.2, 2.3, 2.4],
            [7.0, 7.1, 7.2, 7.3, 7.4],
        ]),
        constant_op.constant([
            [3.0, 3.1, 3.2, 3.3, 3.4],
            [8.0, 8.1, 8.2, 8.3, 8.4],
        ]),
        constant_op.constant([
            [4.0, 4.1, 4.2, 4.3, 4.4],
            [9.0, 9.1, 9.2, 9.3, 9.4],
        ]),
    ]

  def _biases(self):
    return constant_op.constant(
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

  def _div_sharded_biases(self):
    return [
        constant_op.constant([0.0, 0.1]),
        constant_op.constant([0.2, 0.3]),
        constant_op.constant([0.4, 0.5]),
        constant_op.constant([0.6, 0.7]),
        constant_op.constant([0.8, 0.9]),
    ]

  def _mod_sharded_biases(self):
    return [
        constant_op.constant([0.0, 0.5]),
        constant_op.constant([0.1, 0.6]),
        constant_op.constant([0.2, 0.7]),
        constant_op.constant([0.3, 0.8]),
        constant_op.constant([0.4, 0.9]),
    ]

  def _labels(self):
    return constant_op.constant(
        [[0, 1], [1, 2]],
        shape=(self._batch_size, self._num_true),
        name='labels',
        dtype=dtypes.int64)

  def _inputs(self):
    return constant_op.constant(
        [
            [0., 1., 2., 3., 4.],
            [10., 11., 12., 13., 14.],
        ],
        shape=(self._batch_size, self._embed_dim),
        name='inputs')

  def testInvalidNumSampled0(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError,
          r'num_resampled \(3\) must be less than num_sampled \(3\)'):
        sampling_ops.rank_sampled_softmax_loss(
            weights=self._weights(),
            biases=self._biases(),
            labels=self._labels(),
            inputs=self._inputs(),
            num_sampled=3,
            num_resampled=3,
            num_classes=self._num_classes,
            num_true=self._num_true,
            sampled_values=None,
            resampling_temperature=1.,
            remove_accidental_hits=True,
            partition_strategy='div')

  def testInvalidNumSampled1(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(
          ValueError,
          r'num_resampled \(3\) must be less than num_sampled \(2\)'):
        sampling_ops.rank_sampled_softmax_loss(
            weights=self._weights(),
            biases=self._biases(),
            labels=self._labels(),
            inputs=self._inputs(),
            num_sampled=2,
            num_resampled=3,
            num_classes=self._num_classes,
            num_true=self._num_true,
            sampled_values=None,
            resampling_temperature=1.,
            remove_accidental_hits=True,
            partition_strategy='div')

  def testMissingPartitionStrategy(self):
    with ops.Graph().as_default():
      with self.assertRaisesRegexp(ValueError,
                                   r'unsupported partition_strategy \(None\)'):
        sampling_ops.rank_sampled_softmax_loss(
            weights=self._weights(),
            biases=self._biases(),
            labels=self._labels(),
            inputs=self._inputs(),
            num_sampled=2,
            num_resampled=1,
            num_classes=self._num_classes,
            num_true=self._num_true,
            sampled_values=None,
            resampling_temperature=1.,
            remove_accidental_hits=True,
            partition_strategy=None)

  def _testCompareWithNN(self, weights, biases, partition_strategy):
    with ops.Graph().as_default():
      loss = sampling_ops.rank_sampled_softmax_loss(
          weights=weights(),
          biases=biases(),
          labels=self._labels(),
          inputs=self._inputs(),
          num_sampled=self._num_sampled,
          num_resampled=self._num_resampled,
          num_classes=self._num_classes,
          num_true=self._num_true,
          sampled_values=self._sampled_values,
          resampling_temperature=1.,
          remove_accidental_hits=self._remove_accidental_hits,
          partition_strategy=partition_strategy)
      loss_nn = nn.sampled_softmax_loss(
          weights=weights(),
          biases=biases(),
          labels=self._labels(),
          inputs=self._inputs(),
          num_sampled=self._num_resampled,
          num_classes=self._num_classes,
          num_true=self._num_true,
          sampled_values=self._resampled_values,
          remove_accidental_hits=self._remove_accidental_hits,
          partition_strategy=partition_strategy)
      with self.test_session() as sess:
        loss_val = sess.run(loss)
        loss_nn_val = sess.run(loss_nn)

    self.assertAllClose(loss_val, loss_nn_val)

  def testCompareWithNNUnsharded(self):
    self._testCompareWithNN(self._weights, self._biases, 'div')

  def testCompareWithNNShardWeightsDiv(self):
    self._testCompareWithNN(self._div_sharded_weights, self._biases, 'div')

  def testCompareWithNNShardWeightsAndBiasesDiv(self):
    self._testCompareWithNN(self._div_sharded_weights, self._div_sharded_biases,
                            'div')

  def testCompareWithNNShardWeightsMod(self):
    self._testCompareWithNN(self._mod_sharded_weights, self._biases, 'mod')

  def testCompareWithNNShardWeightsAndBiasesMod(self):
    self._testCompareWithNN(self._mod_sharded_weights, self._mod_sharded_biases,
                            'mod')

  def _testCompareWithNNTemperature(self, temperature, resampled):
    weights = [[1., 2.], [3., 4.]]  # two sampled classes
    inputs = [[6., -5. / 2.], [-11., 21. / 2.]]
    # Let w0, w1 = weights of sampled classes (biases set to 0 for simplicity)
    # Let x0, x1 = inputs
    # logits:
    #   w0.x0 = 1
    #   w0.x1 = 10
    #   w1.x0 = 8
    #   w1.x1 = 9
    # Resampling 1 class with temperature = t will pick the larger of:
    #   exp(1/t) + exp(10/t)  ==> w0, for values of t < 2.12
    #   exp(8/t) + exp(9/t)   ==> w1, for values of t > 2.13
    num_sampled = 2
    num_resampled = 1
    num_classes = 2
    num_true = 1
    sampled_values = [0, 1], [[1.], [1.]], [1., 1.]
    resampled_values = [resampled], [[1.], [1.]], [1.]
    remove_accidental_hits = False
    with ops.Graph().as_default():
      weights = constant_op.constant(weights)
      biases = constant_op.constant([0., 0.])
      labels = constant_op.constant([[0], [1]], dtype=dtypes.int64)
      inputs = constant_op.constant(inputs)
      loss = sampling_ops.rank_sampled_softmax_loss(
          weights=weights,
          biases=biases,
          labels=labels,
          inputs=inputs,
          num_sampled=num_sampled,
          num_resampled=num_resampled,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=sampled_values,
          resampling_temperature=constant_op.constant(temperature),
          remove_accidental_hits=remove_accidental_hits,
          partition_strategy='div')
      loss_nn = nn.sampled_softmax_loss(
          weights=weights,
          biases=biases,
          labels=labels,
          inputs=inputs,
          num_sampled=num_resampled,
          num_classes=num_classes,
          num_true=num_true,
          sampled_values=resampled_values,
          remove_accidental_hits=remove_accidental_hits,
          partition_strategy='div')
      with self.test_session() as sess:
        loss_val = sess.run(loss)
        loss_nn_val = sess.run(loss_nn)

    self.assertAllClose(loss_val, loss_nn_val)

  def testCompareWithNNTemperatureLo1(self):
    self._testCompareWithNNTemperature(1., 0)

  def testCompareWithNNTemperatureLo2(self):
    self._testCompareWithNNTemperature(2.12, 0)

  def testCompareWithNNTemperatureHi1(self):
    self._testCompareWithNNTemperature(2.13, 1)

  def testCompareWithNNTemperatureHi2(self):
    self._testCompareWithNNTemperature(3., 1)


if __name__ == '__main__':
  test.main()
