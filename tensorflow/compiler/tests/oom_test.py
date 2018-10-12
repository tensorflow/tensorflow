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
"""Functional tests for out-of-memory conditions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.platform import googletest


class OutOfMemoryTest(xla_test.XLATestCase):

  def testOutputOutOfMemory(self):
    """Allocates tensors until out of memory.

    Generates a large rank-1 tensor. The tensor is an output of an XLA
    computation, not constant.

    Check that a ResourceExhaustedError is raised and can be caught.

    We spin in a loop generating larger and larger tensors until an OOM event
    happens. We may be running sandboxed, so have a small host memory limit, so
    any hardcoded value is unlikely to land in the sweet spot between device
    memory size and host memory size with stability.
    """

    def test_loop():
      size = int(2e8)
      while True:
        with self.cached_session():
          # Force the compiled code to not be constant by feeding in a
          # parameter.
          p = array_ops.placeholder(dtypes.float32, shape=[2, 1, 1])
          with self.test_scope():
            # Create a computation that produces a large R1 tensor as an
            # intermediate result.  Reduce it down so that if this file was
            # compiled without --config=cuda, we don't force a D2H copy of a
            # large tensor and potentially OOM the host.
            #
            # This is a bit tricky because XLA:GPU doesn't currently support RNG
            # ops.  Here we rely on the fact that XLA doesn't do algebraic
            # simplifications on conv(<ones>, <filter>).
            c = math_ops.reduce_sum(
                nn_ops.convolution(
                    array_ops.ones([1, size, 1]),
                    p,
                    padding='SAME',
                    data_format='NWC'))

            c.eval(feed_dict={p: [[[1.0]], [[2.0]]]})
            size *= 2

    self.assertRaises(errors.ResourceExhaustedError, test_loop)


if __name__ == '__main__':
  googletest.main()
