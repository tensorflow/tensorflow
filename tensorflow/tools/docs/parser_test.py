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
"""Tests for documentation parser."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import inspect
import os
import sys

from tensorflow.python.platform import googletest
from tensorflow.tools.docs import parser


def test_function_for_markdown_reference(unused_arg):
  """Docstring with reference to @{test_function}."""
  pass


def test_function(unused_arg, unused_kwarg='default'):
  """Docstring for test function."""
  pass


def test_function_with_args_kwargs(unused_arg, *unused_args, **unused_kwargs):
  """Docstring for second test function."""
  pass


def test_function_with_fancy_docstring(arg):
  """Function with a fancy docstring.

  Args:
    arg: An argument.

  Returns:
    arg: the input, and
    arg: the input, again.

  @compatibility(numpy)
  NumPy has nothing as awesome as this function.
  @end_compatibility

  @compatibility(theano)
  Theano has nothing as awesome as this function.

  Check it out.
  @end_compatibility

  """
  return arg, arg


class TestClass(object):
  """Docstring for TestClass itself."""

  def a_method(self, arg='default'):
    """Docstring for a method."""
    pass

  class ChildClass(object):
    """Docstring for a child class."""
    pass

  @property
  def a_property(self):
    """Docstring for a property."""
    pass

  CLASS_MEMBER = 'a class member'


class ParserTest(googletest.TestCase):

  def test_documentation_path(self):
    self.assertEqual('test.md', parser.documentation_path('test'))
    self.assertEqual('test/module.md', parser.documentation_path('test.module'))

  def test_replace_references(self):
    class HasOneMember(object):

      def foo(self):
        pass

    string = ('A @{tf.reference}, another @{tf.reference}, '
              'a member @{tf.reference.foo}, and a @{tf.third}.')
    duplicate_of = {'tf.third': 'tf.fourth'}
    index = {'tf.reference': HasOneMember,
             'tf.reference.foo': HasOneMember.foo,
             'tf.third': HasOneMember,
             'tf.fourth': HasOneMember}
    result = parser.replace_references(
        string, '../..', duplicate_of, doc_index={}, index=index)
    self.assertEqual(
        'A [`tf.reference`](../../tf/reference.md), another '
        '[`tf.reference`](../../tf/reference.md), '
        'a member [`tf.reference.foo`](../../tf/reference.md#foo), '
        'and a [`tf.third`](../../tf/fourth.md).',
        result)

  def test_doc_replace_references(self):
    string = '@{$doc1} @{$doc1#abc} @{$doc1$link} @{$doc1#def$zelda} @{$do/c2}'

    class DocInfo(object):
      pass
    doc1 = DocInfo()
    doc1.title = 'Title1'
    doc1.url = 'URL1'
    doc2 = DocInfo()
    doc2.title = 'Two words'
    doc2.url = 'somewhere/else'
    doc_index = {'doc1': doc1, 'do/c2': doc2}
    result = parser.replace_references(string, 'python', {},
                                       doc_index=doc_index, index={})
    self.assertEqual(
        '[Title1](../URL1) [Title1](../URL1#abc) [link](../URL1) '
        '[zelda](../URL1#def) [Two words](../somewhere/else)',
        result)

  def test_generate_markdown_for_class(self):

    index = {
        'TestClass': TestClass,
        'TestClass.a_method': TestClass.a_method,
        'TestClass.a_property': TestClass.a_property,
        'TestClass.ChildClass': TestClass.ChildClass,
        'TestClass.CLASS_MEMBER': TestClass.CLASS_MEMBER
    }

    tree = {
        'TestClass': ['a_method', 'a_property', 'ChildClass', 'CLASS_MEMBER']
    }

    docs = parser.generate_markdown(
        full_name='TestClass', py_object=TestClass, duplicate_of={},
        duplicates={}, index=index, tree=tree, reverse_index={}, doc_index={},
        guide_index={}, base_dir='/')

    # Make sure all required docstrings are present.
    self.assertTrue(inspect.getdoc(TestClass) in docs)
    self.assertTrue(inspect.getdoc(TestClass.a_method) in docs)
    self.assertTrue(inspect.getdoc(TestClass.a_property) in docs)

    # Make sure that the signature is extracted properly and omits self.
    self.assertTrue('a_method(arg=\'default\')' in docs)

    # Make sure there is a link to the child class and it points the right way.
    self.assertTrue('[`class ChildClass`](./TestClass/ChildClass.md)' in docs)

    # Make sure CLASS_MEMBER is mentioned.
    self.assertTrue('CLASS_MEMBER' in docs)

    # Make sure this file is contained as the definition location.
    self.assertTrue(os.path.relpath(__file__, '/') in docs)

  def test_generate_markdown_for_module(self):
    module = sys.modules[__name__]

    index = {
        'TestModule': module,
        'TestModule.test_function': test_function,
        'TestModule.test_function_with_args_kwargs':
        test_function_with_args_kwargs,
        'TestModule.TestClass': TestClass,
    }

    tree = {
        'TestModule': ['TestClass', 'test_function',
                       'test_function_with_args_kwargs']
    }

    docs = parser.generate_markdown(full_name='TestModule', py_object=module,
                                    duplicate_of={}, duplicates={},
                                    index=index, tree=tree, reverse_index={},
                                    doc_index={}, guide_index={}, base_dir='/')

    # Make sure all required docstrings are present.
    self.assertTrue(inspect.getdoc(module) in docs)

    # Make sure that links to the members are there (not asserting on exact link
    # text for functions).
    self.assertTrue('./TestModule/test_function.md' in docs)
    self.assertTrue('./TestModule/test_function_with_args_kwargs.md' in docs)

    # Make sure there is a link to the child class and it points the right way.
    self.assertTrue('[`class TestClass`](./TestModule/TestClass.md)' in docs)

    # Make sure this file is contained as the definition location.
    self.assertTrue(os.path.relpath(__file__, '/') in docs)

  def test_generate_markdown_for_function(self):
    index = {
        'test_function': test_function
    }

    tree = {
        '': ['test_function']
    }

    docs = parser.generate_markdown(full_name='test_function',
                                    py_object=test_function,
                                    duplicate_of={}, duplicates={},
                                    index=index, tree=tree, reverse_index={},
                                    doc_index={}, guide_index={}, base_dir='/')

    # Make sure docstring shows up.
    self.assertTrue(inspect.getdoc(test_function) in docs)
    # Make sure the extracted signature is good.
    self.assertTrue(
        'test_function(unused_arg, unused_kwarg=\'default\')' in docs)

    # Make sure this file is contained as the definition location.
    self.assertTrue(os.path.relpath(__file__, '/') in docs)

  def test_generate_markdown_for_function_with_kwargs(self):
    index = {
        'test_function_with_args_kwargs': test_function_with_args_kwargs
    }

    tree = {
        '': ['test_function_with_args_kwargs']
    }

    docs = parser.generate_markdown(full_name='test_function_with_args_kwargs',
                                    py_object=test_function_with_args_kwargs,
                                    duplicate_of={}, duplicates={},
                                    index=index, tree=tree, reverse_index={},
                                    doc_index={}, guide_index={}, base_dir='/')

    # Make sure docstring shows up.
    self.assertTrue(inspect.getdoc(test_function_with_args_kwargs) in docs)

    # Make sure the extracted signature is good.
    self.assertTrue(
        'test_function_with_args_kwargs(unused_arg,'
        ' *unused_args, **unused_kwargs)' in docs)

  def test_references_replaced_in_generated_markdown(self):
    index = {
        'test_function_for_markdown_reference':
        test_function_for_markdown_reference
    }

    tree = {
        '': ['test_function_for_markdown_reference']
    }

    docs = parser.generate_markdown(
        full_name='test_function_for_markdown_reference',
        py_object=test_function_for_markdown_reference, duplicate_of={},
        duplicates={}, index=index, tree=tree, reverse_index={}, doc_index={},
        guide_index={}, base_dir='/')

    # Make sure docstring shows up and is properly processed.
    expected_docs = parser.replace_references(
        inspect.getdoc(test_function_for_markdown_reference),
        relative_path_to_root='.', duplicate_of={}, doc_index={}, index={})

    self.assertTrue(expected_docs in docs)

  def test_docstring_special_section(self):
    index = {
        'test_function': test_function_with_fancy_docstring
    }

    tree = {
        '': 'test_function'
    }

    docs = parser.generate_markdown(
        full_name='test_function', py_object=test_function_with_fancy_docstring,
        duplicate_of={}, duplicates={}, index=index, tree=tree,
        reverse_index={}, doc_index={}, guide_index={}, base_dir='/')
    expected = '\n'.join([
        'Function with a fancy docstring.',
        '',
        '#### Args:',
        '',
        '* <b>`arg`</b>: An argument.',
        '',
        '',
        '#### Returns:',
        '',
        '* <b>`arg`</b>: the input, and',
        '* <b>`arg`</b>: the input, again.',
        '',
        '',
        '',
        '',
        '',
        '#### numpy compatibility',
        'NumPy has nothing as awesome as this function.',
        '',
        '',
        '',
        '#### theano compatibility',
        'Theano has nothing as awesome as this function.',
        '',
        'Check it out.',
        '',
        '',
        ''])
    self.assertTrue(expected in docs)

  def test_generate_index(self):
    module = sys.modules[__name__]

    index = {
        'TestModule': module,
        'test_function': test_function,
        'TestModule.test_function': test_function,
        'TestModule.TestClass': TestClass,
        'TestModule.TestClass.a_method': TestClass.a_method,
        'TestModule.TestClass.a_property': TestClass.a_property,
        'TestModule.TestClass.ChildClass': TestClass.ChildClass,
    }

    duplicate_of = {
        'TestModule.test_function': 'test_function'
    }

    docs = parser.generate_global_index('TestLibrary', index=index,
                                        duplicate_of=duplicate_of)

    # Make sure duplicates and non-top-level symbols are in the index, but
    # methods and properties are not.
    self.assertTrue('a_method' not in docs)
    self.assertTrue('a_property' not in docs)
    self.assertTrue('TestModule.TestClass' in docs)
    self.assertTrue('TestModule.TestClass.ChildClass' in docs)
    self.assertTrue('TestModule.test_function' in docs)
    # Leading backtick to make sure it's included top-level.
    # This depends on formatting, but should be stable.
    self.assertTrue('`test_function' in docs)

  def test_argspec_for_functoos_partial(self):

    # pylint: disable=unused-argument
    def test_function_for_partial1(arg1, arg2, kwarg1=1, kwarg2=2):
      pass

    def test_function_for_partial2(arg1, arg2, *my_args, **my_kwargs):
      pass
    # pylint: enable=unused-argument

    # pylint: disable=protected-access
    # Make sure everything works for regular functions.
    expected = inspect.ArgSpec(['arg1', 'arg2', 'kwarg1', 'kwarg2'], None, None,
                               (1, 2))
    self.assertEqual(expected, parser._get_arg_spec(test_function_for_partial1))

    # Make sure doing nothing works.
    expected = inspect.ArgSpec(['arg1', 'arg2', 'kwarg1', 'kwarg2'], None, None,
                               (1, 2))
    partial = functools.partial(test_function_for_partial1)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    # Make sure setting args from the front works.
    expected = inspect.ArgSpec(['arg2', 'kwarg1', 'kwarg2'], None, None, (1, 2))
    partial = functools.partial(test_function_for_partial1, 1)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    expected = inspect.ArgSpec(['kwarg2',], None, None, (2,))
    partial = functools.partial(test_function_for_partial1, 1, 2, 3)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    # Make sure setting kwargs works.
    expected = inspect.ArgSpec(['arg1', 'arg2', 'kwarg2'], None, None, (2,))
    partial = functools.partial(test_function_for_partial1, kwarg1=0)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    expected = inspect.ArgSpec(['arg1', 'arg2', 'kwarg1'], None, None, (1,))
    partial = functools.partial(test_function_for_partial1, kwarg2=0)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    expected = inspect.ArgSpec(['arg1'], None, None, ())
    partial = functools.partial(test_function_for_partial1,
                                arg2=0, kwarg1=0, kwarg2=0)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    # Make sure *args, *kwargs is accounted for.
    expected = inspect.ArgSpec([], 'my_args', 'my_kwargs', ())
    partial = functools.partial(test_function_for_partial2, 0, 1)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    # pylint: enable=protected-access

if __name__ == '__main__':
  googletest.main()
