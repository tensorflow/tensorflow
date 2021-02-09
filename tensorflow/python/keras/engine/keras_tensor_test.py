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
"""InputSpec tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.keras import layers
from tensorflow.python.keras.engine import keras_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class KerasTensorTest(test.TestCase):

  def test_repr_and_string(self):
    kt = keras_tensor.KerasTensor(
        type_spec=tensor_spec.TensorSpec(shape=(1, 2, 3), dtype=dtypes.float32))
    expected_str = ("KerasTensor(type_spec=TensorSpec(shape=(1, 2, 3), "
                    "dtype=tf.float32, name=None))")
    expected_repr = "<KerasTensor: shape=(1, 2, 3) dtype=float32>"
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    kt = keras_tensor.KerasTensor(
        type_spec=tensor_spec.TensorSpec(shape=(2,), dtype=dtypes.int32),
        inferred_value=[2, 3])
    expected_str = ("KerasTensor(type_spec=TensorSpec(shape=(2,), "
                    "dtype=tf.int32, name=None), inferred_value=[2, 3])")
    expected_repr = (
        "<KerasTensor: shape=(2,) dtype=int32 inferred_value=[2, 3]>")
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    kt = keras_tensor.KerasTensor(
        type_spec=sparse_tensor.SparseTensorSpec(
            shape=(1, 2, 3), dtype=dtypes.float32))
    expected_str = ("KerasTensor(type_spec=SparseTensorSpec("
                    "TensorShape([1, 2, 3]), tf.float32))")
    expected_repr = (
        "<KerasTensor: type_spec=SparseTensorSpec("
        "TensorShape([1, 2, 3]), tf.float32)>")
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    inp = layers.Input(shape=(3, 5))
    kt = layers.Dense(10)(inp)
    expected_str = (
        "KerasTensor(type_spec=TensorSpec(shape=(None, 3, 10), "
        "dtype=tf.float32, name=None), name='dense/BiasAdd:0', "
        "description=\"created by layer 'dense'\")")
    expected_repr = (
        "<KerasTensor: shape=(None, 3, 10) dtype=float32 (created "
        "by layer 'dense')>")
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    kt = array_ops.reshape(kt, shape=(3, 5, 2))
    expected_str = (
        "KerasTensor(type_spec=TensorSpec(shape=(3, 5, 2), dtype=tf.float32, "
        "name=None), name='tf.reshape/Reshape:0', description=\"created "
        "by layer 'tf.reshape'\")")
    expected_repr = ("<KerasTensor: shape=(3, 5, 2) dtype=float32 (created "
                     "by layer 'tf.reshape')>")
    self.assertEqual(expected_str, str(kt))
    self.assertEqual(expected_repr, repr(kt))

    kts = array_ops.unstack(kt)
    for i in range(3):
      expected_str = (
          "KerasTensor(type_spec=TensorSpec(shape=(5, 2), dtype=tf.float32, "
          "name=None), name='tf.unstack/unstack:%s', description=\"created "
          "by layer 'tf.unstack'\")" % (i,))
      expected_repr = ("<KerasTensor: shape=(5, 2) dtype=float32 "
                       "(created by layer 'tf.unstack')>")
      self.assertEqual(expected_str, str(kts[i]))
      self.assertEqual(expected_repr, repr(kts[i]))

if __name__ == "__main__":
  ops.enable_eager_execution()
  tensor_shape.enable_v2_tensorshape()
  test.main()
