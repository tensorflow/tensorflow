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
"""Unit tests for compat."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
  from pathlib import Path
except ImportError:
  Path = None

from tensorflow.python.platform import test
from tensorflow.python.util import compat
from tensorflow.python.ops import array_ops

class CompatPathToStrTest(test.TestCase):
  """Tests for compat path to string utilities"""

  def test_transforms_pathlib_path(self):
    if Path is None:
      self.skipTest("Test requires Python 3.6 or greater.")

    path_input = Path('/tmp/folder')
    transformed = compat.path_to_str(path_input)

    self.assertTrue(isinstance(transformed, str))
    self.assertEqual('/tmp/folder', transformed)

  def test_transforms_iterable_pathlib_path(self):
    if Path is None:
      self.skipTest("Test requires Python 3.6 or greater.")

    path_input = [Path('/tmp/folder'), Path('/tmp/folder2')]
    transformed = compat.path_to_str(path_input)

    self.assertEqual(['/tmp/folder', '/tmp/folder2'], transformed)

  def test_returns_str_unchanged(self):
    str_input = '/tmp/folder'

    transformed = compat.path_to_str(str_input)
    self.assertEqual(str_input, transformed)

  def test_transforms_iterable_str_unchanged(self):
    str_input = ['/tmp/folder', '/tmp/folder2']
    transformed = compat.path_to_str(str_input)

    self.assertEqual(['/tmp/folder', '/tmp/folder2'], transformed)

  def test_returns_tensor_unchanged(self):
    tensor_input = array_ops.constant('/tmp/folder/')

    transformed = compat.path_to_str(tensor_input)
    self.assertEqual(tensor_input, transformed)
