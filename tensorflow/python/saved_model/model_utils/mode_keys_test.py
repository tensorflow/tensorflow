# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""ModeKey Tests."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.platform import test
from tensorflow.python.saved_model.model_utils import mode_keys


class ModeKeyMapTest(test.TestCase):

  def test_map(self):
    mode_map = mode_keys.ModeKeyMap(**{
        mode_keys.KerasModeKeys.PREDICT: 3,
        mode_keys.KerasModeKeys.TEST: 1
    })

    # Test dictionary __getitem__
    self.assertEqual(3, mode_map[mode_keys.KerasModeKeys.PREDICT])
    self.assertEqual(3, mode_map[mode_keys.EstimatorModeKeys.PREDICT])
    self.assertEqual(1, mode_map[mode_keys.KerasModeKeys.TEST])
    self.assertEqual(1, mode_map[mode_keys.EstimatorModeKeys.EVAL])
    with self.assertRaises(KeyError):
      _ = mode_map[mode_keys.KerasModeKeys.TRAIN]
    with self.assertRaises(KeyError):
      _ = mode_map[mode_keys.EstimatorModeKeys.TRAIN]
    with self.assertRaisesRegexp(ValueError, 'Invalid mode'):
      _ = mode_map['serve']

    # Test common dictionary methods
    self.assertLen(mode_map, 2)
    self.assertEqual({1, 3}, set(mode_map.values()))
    self.assertEqual(
        {mode_keys.KerasModeKeys.TEST, mode_keys.KerasModeKeys.PREDICT},
        set(mode_map.keys()))

    # Map is immutable
    with self.assertRaises(TypeError):
      mode_map[mode_keys.KerasModeKeys.TEST] = 1

  def test_invalid_init(self):
    with self.assertRaisesRegexp(ValueError, 'Multiple keys/values found'):
      _ = mode_keys.ModeKeyMap(**{
          mode_keys.KerasModeKeys.PREDICT: 3,
          mode_keys.EstimatorModeKeys.PREDICT: 1
      })


if __name__ == '__main__':
  test.main()
