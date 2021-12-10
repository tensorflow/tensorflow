# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for denormal handling."""

import os
import platform

import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


@test_util.with_eager_op_as_function
class DenormalTest(test.TestCase):

  def testPythonHasDenormals(self):
    """Non-tf numpy code should treat denormals correctly."""
    for dtype in np.float32, np.float64:
      tiny = np.finfo(dtype).tiny
      self.assertEqual(tiny, tiny / 16 * 16)

  def _flushDenormalsTest(self, dtypes):
    if (platform.machine() == "ppc64le" or platform.machine() == "s390x" or
        platform.machine() == "aarch64"):
      # Disabled denormal_test on power/s390x/aarch64 platform
      # Check relevant discussion -
      # https://github.com/tensorflow/tensorflow/issues/11902
      return
    for dtype in dtypes:
      tiny = np.finfo(dtype).tiny
      # Small shape to test main thread, large shape to test thread pool
      for shape in (), (1 << 20,):
        flush = 0.1 * constant_op.constant(tiny, shape=shape)
        self.assertAllEqual(self.evaluate(flush), np.zeros(shape))
        # Make sure the flags don't leak out
        self.testPythonHasDenormals()

  @test_util.run_in_graph_and_eager_modes(use_gpu=False)
  def testFlushDenormalsCPU(self):
    # On CPUs, the processor flags flush for both single and double precision.
    self._flushDenormalsTest(dtypes=(np.float32, np.float64))

  @test_util.run_in_graph_and_eager_modes(use_gpu=True)
  def testFlushDenormalsGPU(self):
    # On GPUs, only single precision can flush to zero.
    self._flushDenormalsTest(dtypes=(np.float32,))


if __name__ == "__main__":
  # When eager_op_as_function mode is enabled xla auto-clustering kicks in.
  # By default xla does not enable flush-to-zero semantics in the GPU backend.
  # This env flag has to be set before the test is setup. Setting it using the
  # decorator does not seem to propagate the flag to all required locations.
  original_xla_flags = os.environ.get("XLA_FLAGS")
  new_xla_flags = "--xla_gpu_ftz=true"
  if original_xla_flags:
    new_xla_flags = new_xla_flags + " " + original_xla_flags
  os.environ["XLA_FLAGS"] = new_xla_flags

  test.main()
