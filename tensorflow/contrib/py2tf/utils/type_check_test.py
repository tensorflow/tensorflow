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
"""Tests for type_check."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy

from tensorflow.contrib.py2tf.utils import type_check
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test


class TypeCheckTest(test.TestCase):

  def test_checks(self):
    self.assertTrue(type_check.is_tensor(constant_op.constant([1, 2, 3])))
    self.assertTrue(
        type_check.is_tensor(test_util.variables.Variable([1, 2, 3])))
    self.assertTrue(
        type_check.is_tensor(
            test_util.array_ops.placeholder(test_util.dtypes.float32)))
    self.assertFalse(type_check.is_tensor(3))
    self.assertFalse(type_check.is_tensor(numpy.eye(3)))


if __name__ == '__main__':
  test.main()
