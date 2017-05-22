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

import tensorflow as tf

from tensorflow.python import debug as tf_debug
from tensorflow.python.platform import googletest
from tensorflow.tools.docs import generate_lib
from tensorflow.tools.docs import parser


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


class DummyVisitor(object):

  def __init__(self, index, duplicate_of):
    self.index = index
    self.duplicate_of = duplicate_of


class GenerateTest(googletest.TestCase):

  def test_extraction(self):
    py_modules = [('tf', tf), ('tfdbg', tf_debug)]

    try:
      generate_lib.extract(
          py_modules, generate_lib._get_default_do_not_descend_map())
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
        'tf': sys,  # Can be any module, this test doesn't care about content.
        'tf.TestModule': module,
        'tf.test_function': test_function,
        'tf.TestModule.test_function': test_function,
        'tf.TestModule.TestClass': TestClass,
        'tf.TestModule.TestClass.ChildClass': TestClass.ChildClass,
        'tf.TestModule.TestClass.ChildClass.GrandChildClass':
        TestClass.ChildClass.GrandChildClass,
    }

    tree = {
        'tf': ['TestModule', 'test_function'],
        'tf.TestModule': ['test_function', 'TestClass'],
        'tf.TestModule.TestClass': ['ChildClass'],
        'tf.TestModule.TestClass.ChildClass': ['GrandChildClass'],
        'tf.TestModule.TestClass.ChildClass.GrandChildClass': []
    }

    duplicate_of = {'tf.test_function': 'tf.TestModule.test_function'}

    duplicates = {
        'tf.TestModule.test_function': [
            'tf.test_function', 'tf.TestModule.test_function'
        ]
    }

    base_dir = os.path.dirname(__file__)

    visitor = DummyVisitor(index, duplicate_of)

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    parser_config = parser.ParserConfig(
        reference_resolver=reference_resolver,
        duplicates=duplicates,
        duplicate_of=duplicate_of,
        tree=tree,
        index=index,
        reverse_index={},
        guide_index={},
        base_dir=base_dir)

    output_dir = googletest.GetTempDir()

    generate_lib.write_docs(output_dir, parser_config, yaml_toc=True)

    # Make sure that the right files are written to disk.
    self.assertTrue(os.path.exists(os.path.join(output_dir, 'index.md')))
    self.assertTrue(os.path.exists(os.path.join(output_dir, 'tf.md')))
    self.assertTrue(os.path.exists(os.path.join(output_dir, '_toc.yaml')))
    self.assertTrue(
        os.path.exists(os.path.join(output_dir, 'tf/TestModule.md')))
    self.assertFalse(
        os.path.exists(os.path.join(output_dir, 'tf/test_function.md')))
    self.assertTrue(
        os.path.exists(
            os.path.join(output_dir, 'tf/TestModule/TestClass.md')))
    self.assertTrue(
        os.path.exists(
            os.path.join(output_dir,
                         'tf/TestModule/TestClass/ChildClass.md')))
    self.assertTrue(
        os.path.exists(
            os.path.join(
                output_dir,
                'tf/TestModule/TestClass/ChildClass/GrandChildClass.md')))
    # Make sure that duplicates are not written
    self.assertTrue(
        os.path.exists(
            os.path.join(output_dir, 'tf/TestModule/test_function.md')))


if __name__ == '__main__':
  googletest.main()
