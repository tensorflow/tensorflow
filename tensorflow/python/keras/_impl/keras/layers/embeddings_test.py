# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for embedding layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python.framework import test_util as tf_test_util
from tensorflow.python.keras._impl import keras
from tensorflow.python.keras._impl.keras import testing_utils
from tensorflow.python.platform import test


class EmbeddingTest(test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def test_embedding(self):
    testing_utils.layer_test(
        keras.layers.Embedding,
        kwargs={'output_dim': 4,
                'input_dim': 10,
                'input_length': 2},
        input_shape=(3, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

    testing_utils.layer_test(
        keras.layers.Embedding,
        kwargs={'output_dim': 4,
                'input_dim': 10,
                'mask_zero': True},
        input_shape=(3, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

    testing_utils.layer_test(
        keras.layers.Embedding,
        kwargs={'output_dim': 4,
                'input_dim': 10,
                'mask_zero': True},
        input_shape=(3, 4, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

    testing_utils.layer_test(
        keras.layers.Embedding,
        kwargs={'output_dim': 4,
                'input_dim': 10,
                'mask_zero': True,
                'input_length': (None, 2)},
        input_shape=(3, 4, 2),
        input_dtype='int32',
        expected_output_dtype='float32')

  def test_embedding_correctness(self):
    with self.test_session():
      layer = keras.layers.Embedding(output_dim=2, input_dim=2)
      layer.build((None, 2))
      matrix = np.array([[1, 1], [2, 2]])
      layer.set_weights([matrix])

      inputs = keras.backend.constant([[0, 1, 0]], dtype='int32')
      outputs = keras.backend.eval(layer(inputs))
      self.assertAllClose(outputs, [[[1, 1], [2, 2], [1, 1]]])


if __name__ == '__main__':
  test.main()
