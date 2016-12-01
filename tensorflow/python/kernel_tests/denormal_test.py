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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

from tensorflow.python.platform import control_imports


class DenormalTest(tf.test.TestCase):

  def testPythonHasDenormals(self):
    """Non-tf numpy code should treat denormals correctly."""
    for dtype in np.float32, np.float64:
      tiny = np.finfo(dtype).tiny
      self.assertEqual(tiny, tiny / 16 * 16)

  def _flushDenormalsTest(self, use_gpu, dtypes):
    if control_imports.USE_OSS:
      # TODO(irving): Fix denormal flushing for open source.
      return
    with self.test_session(use_gpu=use_gpu):
      tf.identity(7).eval()
      for dtype in dtypes:
        tiny = np.finfo(dtype).tiny
        # Small shape to test main thread, large shape to test thread pool
        for shape in (), (1<<20,):
          flush = 0.1 * tf.constant(tiny, shape=shape)
          self.assertAllEqual(flush.eval(), np.zeros(shape))
          # Make sure the flags don't leak out
          self.testPythonHasDenormals()

  def testFlushDenormalsCPU(self):
    # On CPUs, the processor flags flush for both single and double precision.
    self._flushDenormalsTest(use_gpu=False, dtypes=(np.float32, np.float64))

  def testFlushDenormalsGPU(self):
    # On GPUs, only single precision can flush to zero.
    self._flushDenormalsTest(use_gpu=True, dtypes=(np.float32,))


if __name__ == "__main__":
  tf.test.main()
