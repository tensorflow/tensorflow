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
"""Test cases for XLA devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.platform import test


class XlaDeviceTest(xla_test.XLATestCase):

  def testCopies(self):
    """Tests that copies onto and off XLA devices work."""
    shapes = [[0], [1], [1, 0], [1024, 0], [1024, 1], [3, 777], [777, 3],
              [16384, 1], [1, 16384], [1, 20000, 1, 1]]
    for dtype in self.numeric_types:
      for shape in shapes:
        with self.cached_session() as sess:
          with ops.device("CPU"):
            x = array_ops.placeholder(dtype, shape)
          with self.test_scope():
            y = x + x
          with ops.device("CPU"):
            z = array_ops.identity(y)

          inputs = np.random.randint(-100, 100, shape).astype(dtype)
          result = sess.run(z, {x: inputs})
        self.assertAllCloseAccordingToType(result, inputs + inputs)

  def testCopiesOfUnsupportedTypesFailGracefully(self):
    """Tests that copies of unsupported types don't crash."""
    test_types = set([
        np.uint8, np.uint16, np.uint32, np.uint64, np.int8, np.int16, np.int32,
        np.int64, np.float16, np.float32, np.float16,
        dtypes.bfloat16.as_numpy_dtype
    ])
    shape = (10, 10)
    for unsupported_dtype in test_types - self.all_types:
      with self.cached_session() as sess:
        with ops.device("CPU"):
          x = array_ops.placeholder(unsupported_dtype, shape)
        with self.test_scope():
          y, = array_ops.identity_n([x])
        with ops.device("CPU"):
          z = array_ops.identity(y)

          inputs = np.random.randint(-100, 100, shape)
          inputs = inputs.astype(unsupported_dtype)
          # Execution should either succeed or raise an InvalidArgumentError,
          # but not crash. Even "unsupported types" may succeed here since some
          # backends (e.g., the CPU backend) are happy to handle buffers of
          # unsupported types, even if they cannot compute with them.
          try:
            sess.run(z, {x: inputs})
          except errors.InvalidArgumentError:
            pass

  def testControlTrigger(self):
    with self.cached_session() as sess:
      with self.test_scope():
        x = gen_control_flow_ops.control_trigger()
      sess.run(x)


if __name__ == "__main__":
  test.main()
