# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Test cases for Switch and Merge nodes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np  # pylint: disable=unused-import

from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest


class SwitchMergeTest(xla_test.XLATestCase):

  def testSwitchMerge(self):
    with self.cached_session() as sess:
      predicate = array_ops.placeholder(dtypes.bool)
      with self.test_scope():
        false_output, true_output = control_flow_ops.switch(
            data=constant_op.constant(42.0), pred=predicate)
        with ops.control_dependencies([array_ops.identity(false_output)]):
          five = constant_op.constant(5.0)
        with ops.control_dependencies([array_ops.identity(true_output)]):
          ten = constant_op.constant(10.0)
        result = control_flow_ops.merge([five, ten])

    with_true = sess.run(result, {predicate: True})
    self.assertEquals(with_true.output, 10.0)
    self.assertEquals(with_true.value_index, 1)

    with_false = sess.run(result, {predicate: False})
    self.assertEquals(with_false.output, 5.0)
    self.assertEquals(with_false.value_index, 0)


if __name__ == "__main__":
  googletest.main()
