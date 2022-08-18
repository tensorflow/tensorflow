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
"""Tests for naming module."""

from tensorflow.python.autograph.pyct import naming
from tensorflow.python.platform import test


class NamerTest(test.TestCase):

  def test_new_symbol_tracks_names(self):
    namer = naming.Namer({})
    self.assertEqual('temp', namer.new_symbol('temp', set()))
    self.assertItemsEqual(('temp',), namer.generated_names)

  def test_new_symbol_avoids_duplicates(self):
    namer = naming.Namer({})
    self.assertEqual('temp', namer.new_symbol('temp', set()))
    self.assertEqual('temp_1', namer.new_symbol('temp', set()))
    self.assertItemsEqual(('temp', 'temp_1'), namer.generated_names)

  def test_new_symbol_avoids_conflicts(self):
    namer = naming.Namer({'temp': 1})
    # temp is reserved in the global namespace
    self.assertEqual('temp_1', namer.new_symbol('temp', set()))
    # temp_2 is reserved in the local namespace
    self.assertEqual('temp_3', namer.new_symbol('temp', set(('temp_2',))))
    self.assertItemsEqual(('temp_1', 'temp_3'), namer.generated_names)


if __name__ == '__main__':
  test.main()
