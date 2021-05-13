# Copyright 2020 The TensorFlow Authors. All Rights Reserved.
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
"""Test for utilities for collectives."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.distribute import collective_util
from tensorflow.python.eager import test


class OptionsTest(test.TestCase):

  def testCreateOptionsViaExportedAPI(self):
    options = collective_util._OptionsExported(bytes_per_pack=1)
    self.assertIsInstance(options, collective_util.Options)
    self.assertEqual(options.bytes_per_pack, 1)
    with self.assertRaises(ValueError):
      collective_util._OptionsExported(bytes_per_pack=-1)

  def testCreateOptionsViaHints(self):
    with self.assertLogs() as cm:
      options = collective_util.Hints(50, 1)
    self.assertTrue(any("is deprecated" in msg for msg in cm.output))
    self.assertIsInstance(options, collective_util.Options)
    self.assertEqual(options.bytes_per_pack, 50)
    self.assertEqual(options.timeout_seconds, 1)


if __name__ == "__main__":
  test.main()
