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

  def get_test_objects(self):
    # These are all mutable objects, so rebuild them for each test.
    # Don't cache the objects.
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

    return reference_resolver, parser_config

  def test_write(self):
    _, parser_config = self.get_test_objects()

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

  def test_update_id_tags_inplace(self):
    test_dir = googletest.GetTempDir()
    test_sub_dir = os.path.join(test_dir, 'a/b')
    os.makedirs(test_sub_dir)

    test_path1 = os.path.join(test_dir, 'file1.md')
    test_path2 = os.path.join(test_sub_dir, 'file2.md')
    test_path3 = os.path.join(test_sub_dir, 'file3.notmd')

    with open(test_path1, 'w') as f:
      f.write('## abc&123')

    with open(test_path2, 'w') as f:
      f.write('# A Level 1 Heading\n')
      f.write('## A Level 2 Heading')

    with open(test_path3, 'w') as f:
      f.write("## don\'t change this")

    generate_lib.update_id_tags_inplace(test_dir)

    with open(test_path1) as f:
      content = f.read()

    self.assertEqual(content, '<h2 id="abc_123">abc&123</h2>')

    with open(test_path2) as f:
      content = f.read()

    self.assertEqual(
        content, '# A Level 1 Heading\n'
        '<h2 id="A_Level_2_Heading">A Level 2 Heading</h2>')

    with open(test_path3) as f:
      content = f.read()

    self.assertEqual(content, "## don\'t change this")

  def test_replace_refes(self):
    test_dir = googletest.GetTempDir()
    test_in_dir = os.path.join(test_dir, 'in')
    test_in_dir_a = os.path.join(test_dir, 'in/a')
    test_in_dir_b = os.path.join(test_dir, 'in/b')
    os.makedirs(test_in_dir)
    os.makedirs(test_in_dir_a)
    os.makedirs(test_in_dir_b)

    test_out_dir = os.path.join(test_dir, 'out')
    os.makedirs(test_out_dir)

    test_path1 = os.path.join(test_in_dir_a, 'file1.md')
    test_path2 = os.path.join(test_in_dir_b, 'file2.md')
    test_path3 = os.path.join(test_in_dir_b, 'file3.notmd')
    test_path4 = os.path.join(test_in_dir_b, 'OWNERS')

    with open(test_path1, 'w') as f:
      f.write('Use `tf.test_function` to test things.')

    with open(test_path2, 'w') as f:
      f.write('Use @{tf.TestModule.TestClass.ChildClass} to test things.\n'
              "`tf.whatever` doesn't exist")

    with open(test_path3, 'w') as f:
      file3_content = (
          'Not a .md file. Should be copied unchanged:'
          '@{tf.TestModule.TestClass.ChildClass}, `tf.test_function`')
      f.write(file3_content)

    with open(test_path4, 'w') as f:
      f.write('')

    reference_resolver, _ = self.get_test_objects()
    generate_lib.replace_refs(test_in_dir, test_out_dir, reference_resolver,
                              '*.md')

    with open(os.path.join(test_out_dir, 'a/file1.md')) as f:
      content = f.read()
      self.assertEqual(
          content,
          'Use <a href="../api_docs/python/tf/TestModule/test_function.md">'
          '<code>tf.test_function</code></a> to test things.')

    with open(os.path.join(test_out_dir, 'b/file2.md')) as f:
      content = f.read()
      self.assertEqual(
          content,
          'Use '
          '<a href="../api_docs/python/tf/TestModule/TestClass/ChildClass.md">'
          '<code>tf.TestModule.TestClass.ChildClass</code></a> '
          'to test things.\n'
          '`tf.whatever` doesn\'t exist')

    with open(os.path.join(test_out_dir, 'b/file3.notmd')) as f:
      content = f.read()
      self.assertEqual(content, file3_content)

    with self.assertRaises(IOError):
      # This should fail. The OWNERS file should not be copied
      with open(os.path.join(test_out_dir, 'b/OWNERS')) as f:
        content = f.read()


if __name__ == '__main__':
  googletest.main()
