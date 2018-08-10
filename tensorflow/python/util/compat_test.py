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
"""Unit tests for compat."""

# pylint: disable=unused-import
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from pathlib import Path
except ImportError:
    Path = None

try:
  from collections.abc import Iterable as _Iterable
except ImportError:
  from collections import Iterable as _Iterable

import inspect

from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.ops import array_ops
from tensorflow.python import dtypes

class CompatPathToStrTest(test.TestCase):

    def testTransformsPathlibPath(self):
        if Path is None:
            return # Skip test if pathlib not available.
        
        path_input = Path('/tmp/folder')
        transformed = compat.path_to_str(path_input)

        self.assertTrue(isinstance(transformed, str))
        self.assertEqual('/tmp/folder', transformed)

    def testTransformsIterablePathlibPath(self):
        if Path is None:
            return # Skip test if pathlib not available.
        
        path_input = [Path('/tmp/folder'), Path('/tmp/folder2')]
        transformed = compat.path_to_str(path_input)

        self.assertEqual(['/tmp/folder', '/tmp/folder2'], transformed)

    def testReturnsStrUnchanged(self):
        str_input = '/tmp/folder'

        transformed = compat.path_to_str(str_input)
        self.assertEqual(str_input, transformed)

    def testTransformsIterableStrUnchanged(self):   
        str_input = ['/tmp/folder', '/tmp/folder2']
        transformed = compat.path_to_str(str_input)

        self.assertEqual(['/tmp/folder', '/tmp/folder2'], transformed)

    def testReturnsTensorUnchanged(self):
        tensor_input = array_ops.constant('/tmp/folder/', dtype=dtypes.string)

        transformed = compat.path_to_str(tensor_input)
        self.assertEqual(tensor_input, transformed)

    