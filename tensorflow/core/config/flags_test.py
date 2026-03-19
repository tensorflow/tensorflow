# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for flags."""

from tensorflow.core.config import flags
from tensorflow.python.platform import test


class FlagsTest(test.TestCase):

  def test_experiment_flag(self):
    self.assertTrue(flags.config().test_only_experiment_1.value())
    self.assertFalse(flags.config().test_only_experiment_2.value())

    flags.config().test_only_experiment_1.reset(False)
    flags.config().test_only_experiment_2.reset(True)

    self.assertFalse(flags.config().test_only_experiment_1.value())
    self.assertTrue(flags.config().test_only_experiment_2.value())

  def test_flags_singleton(self):
    flags.config().test_only_experiment_1.reset(False)
    self.assertFalse(flags.config().test_only_experiment_1.value())

    # Get second reference to underlying Flags singleton.
    flag = flags.flags_pybind.Flags()
    flag.test_only_experiment_1.reset(True)

    # check that both references are correctly updated.
    self.assertTrue(flags.config().test_only_experiment_1.value())
    self.assertTrue(flag.test_only_experiment_1.value())


if __name__ == '__main__':
  test.main()
