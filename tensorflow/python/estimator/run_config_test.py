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
"""RunConfig tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.estimator import run_config as run_config_lib
from tensorflow.python.platform import test

_TEST_DIR = 'test_dir'
_MASTER = 'master_'
_NOT_SUPPORTED_REPLACE_PROPERTY_MSG = 'Replacing .*is not supported'


class RunConfigTest(test.TestCase):

  def test_model_dir(self):
    empty_config = run_config_lib.RunConfig()
    self.assertIsNone(empty_config.model_dir)

    new_config = empty_config.replace(model_dir=_TEST_DIR)
    self.assertEqual(_TEST_DIR, new_config.model_dir)

  def test_replace(self):
    config = run_config_lib.RunConfig()

    with self.assertRaisesRegexp(
        ValueError, _NOT_SUPPORTED_REPLACE_PROPERTY_MSG):
      # master is not allowed to be replaced.
      config.replace(master=_MASTER)

    with self.assertRaisesRegexp(
        ValueError, _NOT_SUPPORTED_REPLACE_PROPERTY_MSG):
      config.replace(some_undefined_property=_MASTER)


if __name__ == '__main__':
  test.main()
