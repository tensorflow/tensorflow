# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
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
"""Tests for doc generator traversal."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tempfile

from tensorflow.python.platform import googletest
from tensorflow.tools.docs import generate


def test_function():
  """Docstring for test_function."""
  pass


class TestClass(object):
  """Docstring for TestClass itself."""

  class ChildClass(object):
    """Docstring for a child class."""

    class GrandChildClass(object):
      """Docstring for a child of a child class."""
      pass


class GenerateTest(googletest.TestCase):

  def test_extraction(self):
    try:
      generate.extract()
    except RuntimeError:
      print('*****************************************************************')
      print('If this test fails, you have most likely introduced an unsealed')
      print('module. Make sure to use remove_undocumented or similar utilities')
      print('to avoid leaking symbols. See below for more information on the')
      print('failure.')
      print('*****************************************************************')
      raise

  def test_write(self):
    module = sys.modules[__name__]

    index = {
        '': sys,  # Can be any module, this test doesn't care about content.
        'TestModule': module,
        'test_function': test_function,
        'TestModule.test_function': test_function,
        'TestModule.TestClass': TestClass,
        'TestModule.TestClass.ChildClass': TestClass.ChildClass,
        'TestModule.TestClass.ChildClass.GrandChildClass':
        TestClass.ChildClass.GrandChildClass,
    }

    tree = {
        '': ['TestModule', 'test_function'],
        'TestModule': ['test_function', 'TestClass'],
        'TestModule.TestClass': ['ChildClass'],
        'TestModule.TestClass.ChildClass': ['GrandChildClass'],
        'TestModule.TestClass.ChildClass.GrandChildClass': []
    }

    duplicate_of = {
        'TestModule.test_function': 'test_function'
    }

    duplicates = {
        'test_function': ['test_function', 'TestModule.test_function']
    }

    output_dir = tempfile.mkdtemp()
    base_dir = os.path.dirname(__file__)

    generate.write_docs(output_dir, base_dir,
                        duplicate_of, duplicates,
                        index, tree)

    # Make sure that the right files are written to disk.
    self.assertTrue(os.path.exists(os.path.join(output_dir, 'index.md')))
    self.assertTrue(os.path.exists(os.path.join(output_dir, 'full_index.md')))
    self.assertTrue(os.path.exists(os.path.join(output_dir, 'TestModule.md')))
    self.assertTrue(os.path.exists(os.path.join(
        output_dir, 'test_function.md')))
    self.assertTrue(os.path.exists(os.path.join(
        output_dir, 'TestModule/TestClass.md')))
    self.assertTrue(os.path.exists(os.path.join(
        output_dir, 'TestModule/TestClass/ChildClass.md')))
    self.assertTrue(os.path.exists(os.path.join(
        output_dir, 'TestModule/TestClass/ChildClass/GrandChildClass.md')))
    # Make sure that duplicates are not written
    self.assertFalse(os.path.exists(os.path.join(
        output_dir, 'TestModule/test_function.md')))


if __name__ == '__main__':
  googletest.main()
