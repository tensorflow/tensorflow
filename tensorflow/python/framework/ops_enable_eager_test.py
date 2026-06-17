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
"""Tests enabling eager execution at process level."""

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.platform import googletest


class OpsEnableAndDisableEagerTest(googletest.TestCase):

  def setUp(self):
    super().setUp()
    # test for enable eager test
    ops.enable_eager_execution()
    self.assertTrue(context.executing_eagerly())

    # Calling enable eager execution a second time should not cause an error.
    ops.enable_eager_execution()
    self.assertTrue(context.executing_eagerly())

  def testEnableDisableEagerExecution(self):
    # The entirety of the test runs in setUp/tearDown methods
    pass

  def tearDown(self):
    super().tearDown()
    # test for disable eager test
    ops.disable_eager_execution()
    self.assertFalse(context.executing_eagerly())

    # Calling disable eager execution a second time should not cause an error.
    ops.disable_eager_execution()
    self.assertFalse(context.executing_eagerly())


if __name__ == '__main__':
  googletest.main()
