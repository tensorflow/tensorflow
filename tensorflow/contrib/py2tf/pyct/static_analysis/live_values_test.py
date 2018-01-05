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
"""Tests for live_values module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.pyct import anno
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct.static_analysis import access
from tensorflow.contrib.py2tf.pyct.static_analysis import live_values
from tensorflow.python.framework import constant_op
from tensorflow.python.platform import test


class LiveValuesResolverTest(test.TestCase):

  def test_literals(self):

    def test_fn():
      return Foo  # pylint: disable=undefined-variable

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {}, {'Foo': 'bar'})

    retval_node = node.body[0].body[0].value
    self.assertEquals('bar', anno.getanno(retval_node, 'live_val'))

  def test_namespace(self):

    def foo():
      return 'bar'

    def test_fn():
      return foo()

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'foo': foo}, {})

    func_node = node.body[0].body[0].value.func
    self.assertEquals(foo, anno.getanno(func_node, 'live_val'))
    self.assertEquals(('foo',), anno.getanno(func_node, 'fqn'))

  def test_attribute_names(self):

    def test_fn():
      return constant_op.constant(0)

    node = parser.parse_object(test_fn)
    node = access.resolve(node)
    node = live_values.resolve(node, {'constant_op': constant_op}, {})

    func_node = node.body[0].body[0].value.func
    self.assertEquals(constant_op.constant, anno.getanno(func_node, 'live_val'))
    self.assertEquals((constant_op.__name__, 'constant'),
                      anno.getanno(func_node, 'fqn'))


if __name__ == '__main__':
  test.main()
