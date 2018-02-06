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
"""Tests for context_managers module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.py2tf.utils import context_managers
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


class ContextManagersTest(test.TestCase):

  def test_control_dependency_on_returns(self):
    # Just dry run them.
    with context_managers.control_dependency_on_returns(ops, None):
      pass
    with context_managers.control_dependency_on_returns(
        ops, constant_op.constant(1)):
      pass
    with context_managers.control_dependency_on_returns(
        ops, [constant_op.constant(1),
              constant_op.constant(2)]):
      pass


if __name__ == '__main__':
  test.main()
