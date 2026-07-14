# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for tensorflow.ops.reverse_sequence_op."""

from tensorflow.compiler.tests import xla_test
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class ReverseSequenceArgsTest(xla_test.XLATestCase):
  """Tests argument verification of array_ops.reverse_sequence."""

  def testInvalidArguments(self):
    # seq_axis negative
    with self.assertRaisesRegex(
        (errors.InvalidArgumentError, ValueError), "seq_dim must be >=0"
    ):

      @def_function.function(jit_compile=True)
      def f(x):
        return array_ops.reverse_sequence(x, [2, 2], seq_axis=-1)

      f([[1, 2], [3, 4]])

    # batch_axis negative
    with self.assertRaisesRegex(ValueError, "batch_dim must be >=0"):

      @def_function.function(jit_compile=True)
      def g(x):
        return array_ops.reverse_sequence(x, [2, 2], seq_axis=1, batch_axis=-1)

      g([[1, 2], [3, 4]])


if __name__ == "__main__":
  v2_compat.enable_v2_behavior()
  test.main()
