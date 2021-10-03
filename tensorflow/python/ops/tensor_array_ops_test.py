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
"""Tests for tensor_array_ops."""

import numpy as np

from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.platform import test


class TensorArrayOpsTest(test.TestCase):

  @test_util.run_v1_only('Testing placeholders specifically.')
  def test_concat_graph(self):
    values = tensor_array_ops.TensorArray(
        size=4, dtype=dtypes.string, element_shape=[None], infer_shape=False)
    a = array_ops.placeholder(dtypes.string, [
        None,
    ])
    b = array_ops.placeholder(dtypes.string, [
        None,
    ])
    values = (values.write(0, a).write(
        1, constant_op.constant([], dtypes.string))).write(2, b).write(
            3, constant_op.constant([], dtypes.string))

    with self.session() as s:
      result = s.run(values.concat(), {a: ['a', 'b', 'c'], b: ['c', 'd', 'e']})
    self.assertAllEqual(result, [b'a', b'b', b'c', b'c', b'd', b'e'])

  @test_util.run_v2_only
  def test_concat(self):
    values = tensor_array_ops.TensorArray(
        size=4, dtype=dtypes.string, element_shape=[None], infer_shape=False)
    a = constant_op.constant(['a', 'b', 'c'], dtypes.string)
    b = constant_op.constant(['c', 'd', 'e'], dtypes.string)
    values = (values.write(0, a).write(
        1, constant_op.constant([], dtypes.string))).write(2, b).write(
            3, constant_op.constant([], dtypes.string))
    self.assertAllEqual(values.concat(), [b'a', b'b', b'c', b'c', b'd', b'e'])

  @test_util.run_v2_only
  def test_concat_in_function(self):
    @def_function.function
    def fn(a, b):
      values = tensor_array_ops.TensorArray(
          size=4, dtype=dtypes.string, element_shape=[None], infer_shape=False)
      values = (values.write(0, a).write(
          1, constant_op.constant([], dtypes.string))).write(2, b).write(
              3, constant_op.constant([], dtypes.string))
      return values.concat()

    self.assertAllEqual(fn(['a', 'b', 'c'], ['c', 'd', 'e']),
                        [b'a', b'b', b'c', b'c', b'd', b'e'])

  def test_init_numpy_shape(self):
    @def_function.function
    def fn():
      values = tensor_array_ops.TensorArray(
          np.float32,
          size=1,
          dynamic_size=False,
          element_shape=np.array((2, 3)))
      values = values.write(0, np.ones((2, 3)))
      return values.concat()
    self.assertAllEqual(fn(), [[1., 1., 1.], [1., 1., 1.]])


if __name__ == '__main__':
  test.main()
