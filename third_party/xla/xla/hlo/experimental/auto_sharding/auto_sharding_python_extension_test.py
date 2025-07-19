# Copyright 2025 The OpenXLA Authors
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
"""Tests for AutoSharding Python extension."""

from absl.testing import absltest
from xla.hlo.experimental.auto_sharding import auto_sharding_python_extension


class AutoShardingPythonExtensionTest(absltest.TestCase):

  def test_register(self):
    self.assertFalse(auto_sharding_python_extension.is_registered())
    auto_sharding_python_extension.register()
    self.assertTrue(auto_sharding_python_extension.is_registered())

  def test_clear(self):
    auto_sharding_python_extension.register()
    self.assertTrue(auto_sharding_python_extension.is_registered())
    auto_sharding_python_extension.clear()
    self.assertFalse(auto_sharding_python_extension.is_registered())


if __name__ == '__main__':
  absltest.main()
