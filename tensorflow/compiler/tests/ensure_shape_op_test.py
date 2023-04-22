# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for ensure_shape_op."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.platform import test


class EnsureShapeOpTest(xla_test.XLATestCase):

  def testEnsureShape(self):
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = check_ops.ensure_shape(p, (None, 3))
      expected_out = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
      self.assertAllEqual(expected_out,
                          sess.run(op, {p: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]}))

  def testInvalidEnsureShape(self):
    with self.session() as sess:
      p = array_ops.placeholder(dtypes.int32)
      with self.test_scope():
        op = check_ops.ensure_shape(p, (None, 3, 3))
      with self.assertRaisesRegex(errors_impl.InvalidArgumentError,
                                  "is not compatible with expected shape"):
        sess.run(op, {p: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]})


if __name__ == "__main__":
  test.main()
