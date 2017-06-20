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
"""Tests for layers.rnn_common."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.contrib.learn.python.learn.estimators import rnn_common
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.platform import test


class RnnCommonTest(test.TestCase):

  def testMaskActivationsAndLabels(self):
    """Test `mask_activations_and_labels`."""
    batch_size = 4
    padded_length = 6
    num_classes = 4
    np.random.seed(1234)
    sequence_length = np.random.randint(0, padded_length + 1, batch_size)
    activations = np.random.rand(batch_size, padded_length, num_classes)
    labels = np.random.randint(0, num_classes, [batch_size, padded_length])
    (activations_masked_t,
     labels_masked_t) = rnn_common.mask_activations_and_labels(
         constant_op.constant(activations, dtype=dtypes.float32),
         constant_op.constant(labels, dtype=dtypes.int32),
         constant_op.constant(sequence_length, dtype=dtypes.int32))

    with self.test_session() as sess:
      activations_masked, labels_masked = sess.run(
          [activations_masked_t, labels_masked_t])

    expected_activations_shape = [sum(sequence_length), num_classes]
    np.testing.assert_equal(
        expected_activations_shape, activations_masked.shape,
        'Wrong activations shape. Expected {}; got {}.'.format(
            expected_activations_shape, activations_masked.shape))

    expected_labels_shape = [sum(sequence_length)]
    np.testing.assert_equal(expected_labels_shape, labels_masked.shape,
                            'Wrong labels shape. Expected {}; got {}.'.format(
                                expected_labels_shape, labels_masked.shape))
    masked_index = 0
    for i in range(batch_size):
      for j in range(sequence_length[i]):
        actual_activations = activations_masked[masked_index]
        expected_activations = activations[i, j, :]
        np.testing.assert_almost_equal(
            expected_activations,
            actual_activations,
            err_msg='Unexpected logit value at index [{}, {}, :].'
            '  Expected {}; got {}.'.format(i, j, expected_activations,
                                            actual_activations))

        actual_labels = labels_masked[masked_index]
        expected_labels = labels[i, j]
        np.testing.assert_almost_equal(
            expected_labels,
            actual_labels,
            err_msg='Unexpected logit value at index [{}, {}].'
            ' Expected {}; got {}.'.format(i, j, expected_labels,
                                           actual_labels))
        masked_index += 1

  def testSelectLastActivations(self):
    """Test `select_last_activations`."""
    batch_size = 4
    padded_length = 6
    num_classes = 4
    np.random.seed(4444)
    sequence_length = np.random.randint(0, padded_length + 1, batch_size)
    activations = np.random.rand(batch_size, padded_length, num_classes)
    last_activations_t = rnn_common.select_last_activations(
        constant_op.constant(activations, dtype=dtypes.float32),
        constant_op.constant(sequence_length, dtype=dtypes.int32))

    with session.Session() as sess:
      last_activations = sess.run(last_activations_t)

    expected_activations_shape = [batch_size, num_classes]
    np.testing.assert_equal(
        expected_activations_shape, last_activations.shape,
        'Wrong activations shape. Expected {}; got {}.'.format(
            expected_activations_shape, last_activations.shape))

    for i in range(batch_size):
      actual_activations = last_activations[i, :]
      expected_activations = activations[i, sequence_length[i] - 1, :]
      np.testing.assert_almost_equal(
          expected_activations,
          actual_activations,
          err_msg='Unexpected logit value at index [{}, :].'
          '  Expected {}; got {}.'.format(i, expected_activations,
                                          actual_activations))


if __name__ == '__main__':
  test.main()
