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
"""Tests for slices module."""

from tensorflow.python.autograph.converters import directives as directives_converter
from tensorflow.python.autograph.converters import slices
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.lang import directives
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class SliceTest(converter_testing.TestCase):

  def test_index_access(self):

    def f(l):
      directives.set_element_type(l, dtypes.int32)
      return l[1]

    tr = self.transform(f, (directives_converter, slices))

    tl = list_ops.tensor_list_from_tensor(
        [1, 2], element_shape=constant_op.constant([], dtype=dtypes.int32))
    y = tr(tl)
    self.assertEqual(2, self.evaluate(y))

  def test_index_access_multiple_definitions(self):

    def f(l):
      directives.set_element_type(l, dtypes.int32)
      if l:
        l = []
      return l[1]

    self.transform(f, (directives_converter, slices))


if __name__ == '__main__':
  test.main()
