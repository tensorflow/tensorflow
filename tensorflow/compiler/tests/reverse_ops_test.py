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
"""Functional tests for XLA Reverse Ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import googletest


class ReverseOpsTest(xla_test.XLATestCase):

  def testReverseOneDim(self):
    shape = (7, 5, 9, 11)
    for revdim in range(-len(shape), len(shape)):
      self._AssertReverseEqual([revdim], shape)

  def testReverseMoreThanOneDim(self):
    shape = (7, 5, 9, 11)
    # The offset is used to test various (but not all) combinations of negative
    # and positive axis indices that are guaranteed to not collide at the same
    # index.
    for revdims in itertools.chain.from_iterable(
        itertools.combinations(range(-offset,
                                     len(shape) - offset), k)
        for k in range(2,
                       len(shape) + 1)
        for offset in range(0, len(shape))):
      self._AssertReverseEqual(revdims, shape)

  def _AssertReverseEqual(self, revdims, shape):
    np.random.seed(120)
    pval = np.random.randint(0, 100, size=shape).astype(float)
    with self.session():
      with self.test_scope():
        p = array_ops.placeholder(dtypes.int32, shape=shape)
        axis = constant_op.constant(
            np.array(revdims, dtype=np.int32),
            shape=(len(revdims),),
            dtype=dtypes.int32)
        rval = array_ops.reverse(p, axis).eval({p: pval})

        slices = [
            slice(-1, None, -1)
            if d in revdims or d - len(shape) in revdims else slice(None)
            for d in range(len(shape))
        ]
      self.assertEqual(pval[slices].flatten().tolist(), rval.flatten().tolist())


if __name__ == '__main__':
  googletest.main()
