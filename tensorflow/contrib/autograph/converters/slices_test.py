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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.autograph.converters import slices
from tensorflow.contrib.autograph.core import converter_testing
from tensorflow.contrib.autograph.lang import directives
from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import list_ops
from tensorflow.python.platform import test


class SliceTest(converter_testing.TestCase):

  def test_index_access(self):

    def test_fn(l):
      return l[1]

    node, ctx = self.prepare(test_fn, {})
    def_, = anno.getanno(node.body[0].args.args[0], anno.Static.DEFINITIONS)
    def_.directives[directives.set_element_type] = {
        'dtype': parser.parse_expression('tf.int32')
    }
    node = slices.transform(node, ctx)

    with self.compiled(node, {}, dtypes.int32) as result:
      with self.test_session() as sess:
        tl = list_ops.tensor_list_from_tensor(
            [1, 2], element_shape=constant_op.constant([], dtype=dtypes.int32))
        y = result.test_fn(tl)
        self.assertEqual(2, sess.run(y))

  def test_index_access_multiple_definitions(self):

    def test_fn(l):
      if l:
        l = []
      return l[1]

    node, ctx = self.prepare(test_fn, {})
    def_, = anno.getanno(node.body[0].args.args[0], anno.Static.DEFINITIONS)
    def_.directives[directives.set_element_type] = {
        'dtype': parser.parse_expression('tf.int32')
    }
    def_, = anno.getanno(node.body[0].body[0].body[0].targets[0],
                         anno.Static.DEFINITIONS)
    def_.directives[directives.set_element_type] = {
        'dtype': parser.parse_expression('tf.float32')
    }
    with self.assertRaises(transformer.AutographParseError):
      slices.transform(node, ctx)


if __name__ == '__main__':
  test.main()
