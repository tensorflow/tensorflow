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
"""Test cases for XLA devices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.client import session as session_lib
from tensorflow.python.eager import context
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test


class XlaDeviceGpuTest(test.TestCase):

  def __init__(self, method_name="runTest"):
    super(XlaDeviceGpuTest, self).__init__(method_name)
    context.context().enable_xla_devices()

  def testCopiesToAndFromGpuWork(self):
    """Tests that copies between GPU and XLA devices work."""
    if not test.is_gpu_available():
      return

    with session_lib.Session() as sess:
      x = array_ops.placeholder(dtypes.float32, [2])
      with ops.device("GPU"):
        y = x * 2
      with ops.device("device:XLA_CPU:0"):
        z = y * y
      with ops.device("GPU"):
        w = y + z
      result = sess.run(w, {x: [1.5, 0.5]})
    self.assertAllClose(result, [12., 2.], rtol=1e-3)


if __name__ == "__main__":
  test.main()
