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
"""Tests for xla handling of placeholder_with_default."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.compiler.tests import xla_test
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest


class PlaceholderTest(xla_test.XLATestCase):

  def test_placeholder_with_default_default(self):
    with self.session() as sess, self.test_scope():
      v = resource_variable_ops.ResourceVariable(4.0)
      ph = array_ops.placeholder_with_default(v, shape=[])
      out = ph * 2
      sess.run(variables.variables_initializer([v]))
      self.assertEqual(8.0, self.evaluate(out))

  def test_placeholder_with_default_fed(self):
    with self.session() as sess, self.test_scope():
      v = resource_variable_ops.ResourceVariable(4.0)
      ph = array_ops.placeholder_with_default(v, shape=[])
      out = ph * 2
      sess.run(variables.variables_initializer([v]))
      self.assertEqual(2.0, sess.run(out, {ph: 1.0}))


if __name__ == '__main__':
  googletest.main()
