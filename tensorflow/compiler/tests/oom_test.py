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
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest


class OutOfMemoryTest(xla_test.XLATestCase):

  def testOutputOutOfMemory(self):
    """Allocates tensors until out of memory.

    Generates a large rank-1 tensor. The tensor is an output of an XLA
    computation, not constant.

    Check that a ResourceExhaustedError is raised and can be caught.
    """
    size = 5e8
    with self.test_session():
      # Force the compiled code to not be constant by feeding in an addend.
      p = array_ops.placeholder(dtypes.float32, shape=[])
      with self.test_scope():
        # Create a large R1 tensor.
        c = array_ops.zeros([size]) + p

        self.assertRaises(
            errors.ResourceExhaustedError, lambda: c.eval(feed_dict={p: 1.0}))

  def testConstantOutOfMemory(self):
    """Allocates constant tensors until out of memory.

    Generates a large rank-1 tensor and a small rank-1 tensor. The tensors are
    constant outputs of an XLA computation, not variable.

    Multiple constant outputs are created, one small, one large. The small
    tensor will have already been allocated when the large tensor fails.

    Check that a ResourceExhaustedError is raised and can be caught.
    """
    size = 5e8
    with self.test_session() as sess:
      with self.test_scope():
        # Create two R1 tensors, size 5 and size n.
        b = array_ops.zeros([5])
        c = array_ops.zeros([size])
        e = control_flow_ops.tuple([b, c])
        self.assertRaises(errors.ResourceExhaustedError, lambda: sess.run(e))


if __name__ == "__main__":
  googletest.main()
