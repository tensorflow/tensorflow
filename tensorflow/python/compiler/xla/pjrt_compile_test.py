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
"""Tests single device compilation + execution using the Device API (aka PjRt).

This feature is still under active development and is protected behind the
`--tf_xla_use_device_api` flag in the `TF_XLA_FLAGS` environment variable.
"""

from tensorflow.python.eager import def_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class PjrtCompileTest(test.TestCase):

  # Tests compilation and execution of a jit_compiled function using PjRt.
  def test_xla_local_launch(self):
    @def_function.function(jit_compile=True)
    def foo(x):
      a = x + 1
      return a

    # Currently PjRt only supports compilation and execution for the XLA_GPU
    # device to unblock development. Support for non-XLA devices (CPU/GPU/single
    # core TPU) is going to be added soon, after which support for XLA_* devices
    # will be dropped.
    # TODO(b/255826209): Modify the test as we progress towards supporting
    # non-XLA devices.
    with ops.device("/device:XLA_GPU:0"):
      with self.assertRaises(errors.UnimplementedError):
        _ = self.evaluate(foo(1))


if __name__ == "__main__":
  test.main()
