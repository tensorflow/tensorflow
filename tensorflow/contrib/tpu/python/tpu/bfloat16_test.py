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
# =============================================================================

"""Tests for bfloat16 helper."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.tpu.python.tpu import bfloat16
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import variable_scope

from tensorflow.python.platform import test


class BFloat16ScopeTest(test.TestCase):

  def testScopeName(self):
    """Test if name for the variable scope is propogated correctly.
    """
    with bfloat16.bfloat16_scope() as bf:
      self.assertEqual(bf.name, "bfloat16")

  def testRequestedDType(self):
    """Test if requested dtype is honored in the getter.
    """
    with bfloat16.bfloat16_scope() as scope:
      v1 = variable_scope.get_variable("v1", [])
      self.assertEqual(v1.dtype.base_dtype, dtypes.float32)
      v2 = variable_scope.get_variable("v2", [], dtype=dtypes.bfloat16)
      self.assertEqual(v2.dtype.base_dtype, dtypes.bfloat16)
      self.assertEqual([dtypes.float32, dtypes.float32],
                       [v.dtype.base_dtype for v in scope.global_variables()])


if __name__ == "__main__":
  test.main()
