# Copyright 2024 The OpenXLA Authors.
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

import threading

from absl.testing import absltest

from xla.python import xla_client

config = xla_client._xla.config


class ConfigTest(absltest.TestCase):

  def testBasic(self):
    c = config.Config(1)
    self.assertEqual(c.value, 1)
    self.assertEqual(c.get_global(), 1)
    self.assertEqual(c.get_local(), config.unset)

    c.set_global(2)
    self.assertEqual(c.value, 2)
    self.assertEqual(c.get_global(), 2)
    self.assertEqual(c.get_local(), config.unset)

    c.set_local(3)
    self.assertEqual(c.value, 3)
    self.assertEqual(c.get_global(), 2)
    self.assertEqual(c.get_local(), 3)

    c.set_global(4)
    self.assertEqual(c.value, 3)
    self.assertEqual(c.get_global(), 4)
    self.assertEqual(c.get_local(), 3)

    c.set_local(config.unset)
    self.assertEqual(c.value, 4)
    self.assertEqual(c.get_global(), 4)
    self.assertEqual(c.get_local(), config.unset)

  def testThreading(self):
    c = config.Config(1)

    def Body():
      for i in range(100):
        c.set_local(i)
        self.assertEqual(c.get_local(), i)
        self.assertEqual(c.get_global(), 1)
        self.assertEqual(c.value, i)

    threads = [threading.Thread(target=Body) for _ in range(4)]
    for t in threads:
      t.start()
    for t in threads:
      t.join()


if __name__ == "__main__":
  absltest.main()
