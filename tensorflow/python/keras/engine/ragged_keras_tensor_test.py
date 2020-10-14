# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""RaggedKerasTensor tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras import layers
from tensorflow.python.keras import testing_utils
from tensorflow.python.keras.engine import training
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.platform import test


class RaggedKerasTensorTest(keras_parameterized.TestCase):

  @parameterized.parameters(
      {'batch_size': None, 'shape': (None, 5), 'ragged_rank': 1},
      {'batch_size': None, 'shape': (None, 3, 5), 'ragged_rank': 1},
      {'batch_size': None, 'shape': (5, None), 'ragged_rank': 2},
      {'batch_size': None, 'shape': (3, 5, None), 'ragged_rank': 3},
      {'batch_size': None, 'shape': (None, 3, 5, None), 'ragged_rank': 4},
      {'batch_size': None, 'shape': (2, 3, None, 4, 5, None), 'ragged_rank': 6},
      {'batch_size': 8, 'shape': (None, 5), 'ragged_rank': 1},
      {'batch_size': 9, 'shape': (None, 3, 5), 'ragged_rank': 1},
      {'batch_size': 1, 'shape': (5, None), 'ragged_rank': 2},
      {'batch_size': 4, 'shape': (3, 5, None), 'ragged_rank': 3},
      {'batch_size': 7, 'shape': (None, 3, 5, None), 'ragged_rank': 4},
      {'batch_size': 12, 'shape': (2, 3, None, 4, 5, None), 'ragged_rank': 6},
  )
  def test_to_placeholder(self, shape, batch_size, ragged_rank):
    with testing_utils.use_keras_tensors_scope(True):
      inp = layers.Input(shape=shape, batch_size=batch_size, ragged=True)
      self.assertEqual(inp.ragged_rank, ragged_rank)
      self.assertAllEqual(inp.shape, [batch_size] + list(shape))
      with func_graph.FuncGraph('test').as_default():
        placeholder = inp._to_placeholder()
        self.assertEqual(placeholder.ragged_rank, ragged_rank)
        self.assertAllEqual(placeholder.shape, [batch_size] + list(shape))

  def test_add(self):
    inp = layers.Input(shape=[None], ragged=True)
    out = inp + inp
    model = training.Model(inp, out)

    x = ragged_factory_ops.constant([[3, 4], [1, 2], [3, 5]])
    self.assertAllEqual(model(x), x + x)

  def test_mul(self):
    inp = layers.Input(shape=[None], ragged=True)
    out = inp * inp
    model = training.Model(inp, out)

    x = ragged_factory_ops.constant([[3, 4], [1, 2], [3, 5]])
    self.assertAllEqual(model(x), x * x)

  def test_sub(self):
    inp = layers.Input(shape=[None], ragged=True)
    out = inp - inp
    model = training.Model(inp, out)

    x = ragged_factory_ops.constant([[3, 4], [1, 2], [3, 5]])
    self.assertAllEqual(model(x), x - x)

  def test_div(self):
    inp = layers.Input(shape=[None], ragged=True)
    out = inp / inp
    model = training.Model(inp, out)

    x = ragged_factory_ops.constant([[3, 4], [1, 2], [3, 5]])
    self.assertAllEqual(model(x), x / x)


if __name__ == '__main__':
  ops.enable_eager_execution()
  tensor_shape.enable_v2_tensorshape()
  test.main()
