# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for io_utils."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import six

from tensorflow.python.keras import keras_parameterized
from tensorflow.python.keras.utils import io_utils
from tensorflow.python.platform import test


class TestIOUtils(keras_parameterized.TestCase):

  def test_ask_to_proceed_with_overwrite(self):
    with test.mock.patch.object(six.moves, 'input') as mock_log:
      mock_log.return_value = 'y'
      self.assertTrue(io_utils.ask_to_proceed_with_overwrite('/tmp/not_exists'))

      mock_log.return_value = 'n'
      self.assertFalse(
          io_utils.ask_to_proceed_with_overwrite('/tmp/not_exists'))

      mock_log.side_effect = ['m', 'y']
      self.assertTrue(io_utils.ask_to_proceed_with_overwrite('/tmp/not_exists'))

      mock_log.side_effect = ['m', 'n']
      self.assertFalse(
          io_utils.ask_to_proceed_with_overwrite('/tmp/not_exists'))

  def test_path_to_string(self):

    class PathLikeDummy(object):

      def __fspath__(self):
        return 'dummypath'

    dummy = object()
    if sys.version_info >= (3, 4):
      from pathlib import Path  # pylint:disable=g-import-not-at-top
      # conversion of PathLike
      self.assertEqual(io_utils.path_to_string(Path('path')), 'path')
    if sys.version_info >= (3, 6):
      self.assertEqual(io_utils.path_to_string(PathLikeDummy()), 'dummypath')

    # pass-through, works for all versions of python
    self.assertEqual(io_utils.path_to_string('path'), 'path')
    self.assertIs(io_utils.path_to_string(dummy), dummy)


if __name__ == '__main__':
  test.main()
