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
"""Tests for side_effect_guards module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf import utils
from tensorflow.contrib.py2tf.converters import converter_test_base
from tensorflow.contrib.py2tf.converters import side_effect_guards
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test


class TestNamer(side_effect_guards.SymbolNamer):

  def new_symbol(self, name_root, _):
    return name_root


class SideEffectGuardsTest(converter_test_base.TestCase):

  def test_transform(self):

    def test_fn(a):
      state_ops.assign(a, a + 1)
      return a

    node = self.parse_and_analyze(test_fn, {'state_ops': state_ops})
    node = side_effect_guards.transform(node, TestNamer())
    result = compiler.ast_to_object(node)
    setattr(result, 'state_ops', state_ops)
    setattr(result, 'py2tf_utils', utils)

    # TODO(mdan): Configure the namespaces instead of doing these hacks.
    ops.identity = array_ops.identity
    setattr(result, 'tf', ops)

    with self.test_session() as sess:
      v = variables.Variable(2)
      sess.run(v.initializer)
      self.assertEqual(3, sess.run(result.test_fn(v)))


if __name__ == '__main__':
  test.main()
