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
import os
import sys

from tensorflow.python.platform import googletest
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
from tensorflow.tools.docs import parser


def test_function(unused_arg, unused_kwarg='default'):
  """Docstring for test function."""
  pass


def test_function_with_args_kwargs(unused_arg, *unused_args, **unused_kwargs):
  """Docstring for second test function."""
  pass


class ParentClass(object):

  @doc_controls.do_not_doc_inheritable
  def hidden_method(self):
    pass


class TestClass(ParentClass):
  """Docstring for TestClass itself."""

  def a_method(self, arg='default'):
    """Docstring for a method."""
    pass

  def hidden_method(self):
    pass

  @doc_controls.do_not_generate_docs
  def hidden_method2(self):
    pass

  class ChildClass(object):
    """Docstring for a child class."""
    pass

  @property
  def a_property(self):
    """Docstring for a property."""
    pass

  CLASS_MEMBER = 'a class member'


class DummyVisitor(object):

  def __init__(self, index, duplicate_of):
    self.index = index
    self.duplicate_of = duplicate_of


class ParserTest(googletest.TestCase):

  def test_documentation_path(self):
    self.assertEqual('test.md', parser.documentation_path('test'))
    self.assertEqual('test/module.md', parser.documentation_path('test.module'))

  def test_replace_references(self):
    class HasOneMember(object):

      def foo(self):
        pass

    string = (
        'A @{tf.reference}, another @{tf.reference$with\nnewline}, a member '
        '@{tf.reference.foo}, and a @{tf.third$link `text` with `code` in '
        'it}.')
    duplicate_of = {'tf.third': 'tf.fourth'}
    index = {'tf.reference': HasOneMember,
             'tf.reference.foo': HasOneMember.foo,
             'tf.third': HasOneMember,
             'tf.fourth': HasOneMember}

    visitor = DummyVisitor(index, duplicate_of)

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    result = reference_resolver.replace_references(string, '../..')
    self.assertEqual('A <a href="../../tf/reference.md">'
                     '<code>tf.reference</code></a>, '
                     'another <a href="../../tf/reference.md">'
                     'with\nnewline</a>, '
                     'a member <a href="../../tf/reference.md#foo">'
                     '<code>tf.reference.foo</code></a>, '
                     'and a <a href="../../tf/fourth.md">link '
                     '<code>text</code> with '
                     '<code>code</code> in it</a>.', result)

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

    visitor = DummyVisitor(index={}, duplicate_of={})

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index=doc_index, py_module_names=['tf'])
    result = reference_resolver.replace_references(string, 'python')
    self.assertEqual('<a href="../URL1">Title1</a> '
                     '<a href="../URL1#abc">Title1</a> '
                     '<a href="../URL1">link</a> '
                     '<a href="../URL1#def">zelda</a> '
                     '<a href="../somewhere/else">Two words</a>', result)

  def test_docs_for_class(self):

    index = {
        'TestClass': TestClass,
        'TestClass.a_method': TestClass.a_method,
        'TestClass.a_property': TestClass.a_property,
        'TestClass.ChildClass': TestClass.ChildClass,
        'TestClass.CLASS_MEMBER': TestClass.CLASS_MEMBER
    }

    visitor = DummyVisitor(index=index, duplicate_of={})

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    tree = {
        'TestClass': ['a_method', 'a_property', 'ChildClass', 'CLASS_MEMBER']
    }
    parser_config = parser.ParserConfig(
        reference_resolver=reference_resolver,
        duplicates={},
        duplicate_of={},
        tree=tree,
        index=index,
        reverse_index={},
        guide_index={},
        base_dir='/')

    page_info = parser.docs_for_object(
        full_name='TestClass', py_object=TestClass, parser_config=parser_config)

    # Make sure the brief docstring is present
    self.assertEqual(
        tf_inspect.getdoc(TestClass).split('\n')[0], page_info.doc.brief)

    # Make sure the method is present
    self.assertEqual(TestClass.a_method, page_info.methods[0].obj)

    # Make sure that the signature is extracted properly and omits self.
    self.assertEqual(["arg='default'"], page_info.methods[0].signature)

    # Make sure the property is present
    self.assertIs(TestClass.a_property, page_info.properties[0].obj)

    # Make sure there is a link to the child class and it points the right way.
    self.assertIs(TestClass.ChildClass, page_info.classes[0].obj)

    # Make sure this file is contained as the definition location.
    self.assertEqual(os.path.relpath(__file__, '/'), page_info.defined_in.path)

  def test_docs_for_class_should_skip(self):

    class Parent(object):

      @doc_controls.do_not_doc_inheritable
      def a_method(self, arg='default'):
        pass

    class Child(Parent):

      def a_method(self, arg='default'):
        pass

    index = {
        'Child': Child,
        'Child.a_method': Child.a_method,
    }

    visitor = DummyVisitor(index=index, duplicate_of={})

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    tree = {
        'Child': ['a_method'],
    }

    parser_config = parser.ParserConfig(
        reference_resolver=reference_resolver,
        duplicates={},
        duplicate_of={},
        tree=tree,
        index=index,
        reverse_index={},
        guide_index={},
        base_dir='/')

    page_info = parser.docs_for_object(
        full_name='Child', py_object=Child, parser_config=parser_config)

    # Make sure the `a_method` is not present
    self.assertEqual(0, len(page_info.methods))

  def test_docs_for_message_class(self):

    class CMessage(object):

      def hidden(self):
        pass

    class Message(object):

      def hidden2(self):
        pass

    class MessageMeta(object):

      def hidden3(self):
        pass

    class ChildMessage(CMessage, Message, MessageMeta):

      def my_method(self):
        pass

    index = {
        'ChildMessage': ChildMessage,
        'ChildMessage.hidden': ChildMessage.hidden,
        'ChildMessage.hidden2': ChildMessage.hidden2,
        'ChildMessage.hidden3': ChildMessage.hidden3,
        'ChildMessage.my_method': ChildMessage.my_method,
    }

    visitor = DummyVisitor(index=index, duplicate_of={})

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    tree = {'ChildMessage': ['hidden', 'hidden2', 'hidden3', 'my_method']}

    parser_config = parser.ParserConfig(
        reference_resolver=reference_resolver,
        duplicates={},
        duplicate_of={},
        tree=tree,
        index=index,
        reverse_index={},
        guide_index={},
        base_dir='/')

    page_info = parser.docs_for_object(
        full_name='ChildMessage',
        py_object=ChildMessage,
        parser_config=parser_config)

    self.assertEqual(1, len(page_info.methods))
    self.assertEqual('my_method', page_info.methods[0].short_name)

  def test_docs_for_module(self):
    # Get the current module.
    module = sys.modules[__name__]

    index = {
        'TestModule': module,
        'TestModule.test_function': test_function,
        'TestModule.test_function_with_args_kwargs':
        test_function_with_args_kwargs,
        'TestModule.TestClass': TestClass,
    }

    visitor = DummyVisitor(index=index, duplicate_of={})

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    tree = {
        'TestModule': ['TestClass', 'test_function',
                       'test_function_with_args_kwargs']
    }
    parser_config = parser.ParserConfig(
        reference_resolver=reference_resolver,
        duplicates={},
        duplicate_of={},
        tree=tree,
        index=index,
        reverse_index={},
        guide_index={},
        base_dir='/')

    page_info = parser.docs_for_object(
        full_name='TestModule', py_object=module, parser_config=parser_config)

    # Make sure the brief docstring is present
    self.assertEqual(tf_inspect.getdoc(module).split('\n')[0],
                     page_info.doc.brief)

    # Make sure that the members are there
    funcs = {f_info.obj for f_info in page_info.functions}
    self.assertEqual({test_function, test_function_with_args_kwargs}, funcs)

    classes = {cls_info.obj for cls_info in page_info.classes}
    self.assertEqual({TestClass}, classes)

    # Make sure this file is contained as the definition location.
    self.assertEqual(os.path.relpath(__file__, '/'), page_info.defined_in.path)

  def test_docs_for_function(self):
    index = {
        'test_function': test_function
    }

    visitor = DummyVisitor(index=index, duplicate_of={})

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    tree = {
        '': ['test_function']
    }
    parser_config = parser.ParserConfig(
        reference_resolver=reference_resolver,
        duplicates={},
        duplicate_of={},
        tree=tree,
        index=index,
        reverse_index={},
        guide_index={},
        base_dir='/')

    page_info = parser.docs_for_object(
        full_name='test_function',
        py_object=test_function,
        parser_config=parser_config)

    # Make sure the brief docstring is present
    self.assertEqual(
        tf_inspect.getdoc(test_function).split('\n')[0], page_info.doc.brief)

    # Make sure the extracted signature is good.
    self.assertEqual(['unused_arg', "unused_kwarg='default'"],
                     page_info.signature)

    # Make sure this file is contained as the definition location.
    self.assertEqual(os.path.relpath(__file__, '/'), page_info.defined_in.path)

  def test_docs_for_function_with_kwargs(self):
    index = {
        'test_function_with_args_kwargs': test_function_with_args_kwargs
    }

    visitor = DummyVisitor(index=index, duplicate_of={})

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    tree = {
        '': ['test_function_with_args_kwargs']
    }
    parser_config = parser.ParserConfig(
        reference_resolver=reference_resolver,
        duplicates={},
        duplicate_of={},
        tree=tree,
        index=index,
        reverse_index={},
        guide_index={},
        base_dir='/')

    page_info = parser.docs_for_object(
        full_name='test_function_with_args_kwargs',
        py_object=test_function_with_args_kwargs,
        parser_config=parser_config)

    # Make sure the brief docstring is present
    self.assertEqual(
        tf_inspect.getdoc(test_function_with_args_kwargs).split('\n')[0],
        page_info.doc.brief)

    # Make sure the extracted signature is good.
    self.assertEqual(['unused_arg', '*unused_args', '**unused_kwargs'],
                     page_info.signature)

  def test_parse_md_docstring(self):

    def test_function_with_fancy_docstring(arg):
      """Function with a fancy docstring.

      And a bunch of references: @{tf.reference}, another @{tf.reference},
          a member @{tf.reference.foo}, and a @{tf.third}.

      Args:
        arg: An argument.

      Raises:
        an exception

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

    class HasOneMember(object):

      def foo(self):
        pass

    duplicate_of = {'tf.third': 'tf.fourth'}
    index = {
        'tf.fancy': test_function_with_fancy_docstring,
        'tf.reference': HasOneMember,
        'tf.reference.foo': HasOneMember.foo,
        'tf.third': HasOneMember,
        'tf.fourth': HasOneMember
    }

    visitor = DummyVisitor(index=index, duplicate_of=duplicate_of)

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    doc_info = parser._parse_md_docstring(test_function_with_fancy_docstring,
                                          '../..', reference_resolver)

    self.assertNotIn('@', doc_info.docstring)
    self.assertNotIn('compatibility', doc_info.docstring)
    self.assertNotIn('Raises:', doc_info.docstring)

    self.assertEqual(len(doc_info.function_details), 3)
    self.assertEqual(set(doc_info.compatibility.keys()), {'numpy', 'theano'})

    self.assertEqual(doc_info.compatibility['numpy'],
                     'NumPy has nothing as awesome as this function.\n')

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

    visitor = DummyVisitor(index=index, duplicate_of=duplicate_of)

    reference_resolver = parser.ReferenceResolver.from_visitor(
        visitor=visitor, doc_index={}, py_module_names=['tf'])

    docs = parser.generate_global_index('TestLibrary', index=index,
                                        reference_resolver=reference_resolver)

    # Make sure duplicates and non-top-level symbols are in the index, but
    # methods and properties are not.
    self.assertNotIn('a_method', docs)
    self.assertNotIn('a_property', docs)
    self.assertIn('TestModule.TestClass', docs)
    self.assertIn('TestModule.TestClass.ChildClass', docs)
    self.assertIn('TestModule.test_function', docs)
    # Leading backtick to make sure it's included top-level.
    # This depends on formatting, but should be stable.
    self.assertIn('<code>test_function', docs)

  def test_argspec_for_functools_partial(self):
    # pylint: disable=unused-argument
    def test_function_for_partial1(arg1, arg2, kwarg1=1, kwarg2=2):
      pass

    def test_function_for_partial2(arg1, arg2, *my_args, **my_kwargs):
      pass
    # pylint: enable=unused-argument

    # pylint: disable=protected-access
    # Make sure everything works for regular functions.
    expected = tf_inspect.FullArgSpec(
        args=['arg1', 'arg2', 'kwarg1', 'kwarg2'],
        varargs=None,
        varkw=None,
        defaults=(1, 2),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    self.assertEqual(expected, parser._get_arg_spec(test_function_for_partial1))

    # Make sure doing nothing works.
    expected = tf_inspect.FullArgSpec(
        args=['arg1', 'arg2', 'kwarg1', 'kwarg2'],
        varargs=None,
        varkw=None,
        defaults=(1, 2),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    partial = functools.partial(test_function_for_partial1)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    # Make sure setting args from the front works.
    expected = tf_inspect.FullArgSpec(
        args=['arg2', 'kwarg1', 'kwarg2'],
        varargs=None,
        varkw=None,
        defaults=(1, 2),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    partial = functools.partial(test_function_for_partial1, 1)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    expected = tf_inspect.FullArgSpec(
        args=['kwarg2'],
        varargs=None,
        varkw=None,
        defaults=(2,),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    partial = functools.partial(test_function_for_partial1, 1, 2, 3)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    # Make sure setting kwargs works.
    expected = tf_inspect.FullArgSpec(
        args=['arg1', 'arg2', 'kwarg2'],
        varargs=None,
        varkw=None,
        defaults=(2,),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    partial = functools.partial(test_function_for_partial1, kwarg1=0)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    expected = tf_inspect.FullArgSpec(
        args=['arg1', 'arg2', 'kwarg1'],
        varargs=None,
        varkw=None,
        defaults=(1,),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    partial = functools.partial(test_function_for_partial1, kwarg2=0)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    expected = tf_inspect.FullArgSpec(
        args=['arg1'],
        varargs=None,
        varkw=None,
        defaults=(),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    partial = functools.partial(test_function_for_partial1,
                                arg2=0, kwarg1=0, kwarg2=0)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    # Make sure *args, *kwargs is accounted for.
    expected = tf_inspect.FullArgSpec(
        args=[],
        varargs='my_args',
        varkw='my_kwargs',
        defaults=(),
        kwonlyargs=[],
        kwonlydefaults=None,
        annotations={})
    partial = functools.partial(test_function_for_partial2, 0, 1)
    self.assertEqual(expected, parser._get_arg_spec(partial))

    # pylint: enable=protected-access

  def testSaveReferenceResolver(self):
    you_cant_serialize_this = object()

    duplicate_of = {'AClass': ['AClass2']}
    doc_index = {'doc': you_cant_serialize_this}
    is_class = {
        'tf': False,
        'tf.AClass': True,
        'tf.AClass2': True,
        'tf.function': False
    }
    is_module = {
        'tf': True,
        'tf.AClass': False,
        'tf.AClass2': False,
        'tf.function': False
    }
    py_module_names = ['tf', 'tfdbg']

    resolver = parser.ReferenceResolver(duplicate_of, doc_index, is_class,
                                        is_module, py_module_names)

    outdir = googletest.GetTempDir()

    filepath = os.path.join(outdir, 'resolver.json')

    resolver.to_json_file(filepath)
    resolver2 = parser.ReferenceResolver.from_json_file(filepath, doc_index)

    # There are no __slots__, so all fields are visible in __dict__.
    self.assertEqual(resolver.__dict__, resolver2.__dict__)

RELU_DOC = """Computes rectified linear: `max(features, 0)`

Args:
  features: A `Tensor`. Must be one of the following types: `float32`,
    `float64`, `int32`, `int64`, `uint8`, `int16`, `int8`, `uint16`,
    `half`.
  name: A name for the operation (optional)

Returns:
  A `Tensor`. Has the same type as `features`
"""


class TestParseFunctionDetails(googletest.TestCase):

  def test_parse_function_details(self):
    docstring, function_details = parser._parse_function_details(RELU_DOC)

    self.assertEqual(len(function_details), 2)
    args = function_details[0]
    self.assertEqual(args.keyword, 'Args')
    self.assertEqual(len(args.header), 0)
    self.assertEqual(len(args.items), 2)
    self.assertEqual(args.items[0][0], 'features')
    self.assertEqual(args.items[1][0], 'name')
    self.assertEqual(args.items[1][1],
                     'A name for the operation (optional)\n\n')
    returns = function_details[1]
    self.assertEqual(returns.keyword, 'Returns')

    relu_doc_lines = RELU_DOC.split('\n')
    self.assertEqual(docstring, relu_doc_lines[0] + '\n\n')
    self.assertEqual(returns.header, relu_doc_lines[-2] + '\n')

    self.assertEqual(
        RELU_DOC,
        docstring + ''.join(str(detail) for detail in function_details))


class TestGenerateSignature(googletest.TestCase):

  def test_known_object(self):
    known_object = object()
    reverse_index = {id(known_object): 'location.of.object.in.api'}

    def example_fun(arg=known_object):  # pylint: disable=unused-argument
      pass

    sig = parser._generate_signature(example_fun, reverse_index)
    self.assertEqual(sig, ['arg=location.of.object.in.api'])

  def test_literals(self):
    if sys.version_info >= (3, 0):
      print('Warning: Doc generation is not supported from python3.')
      return

    def example_fun(a=5, b=5.0, c=None, d=True, e='hello', f=(1, (2, 3))):  # pylint: disable=g-bad-name, unused-argument
      pass

    sig = parser._generate_signature(example_fun, reverse_index={})
    self.assertEqual(
        sig, ['a=5', 'b=5.0', 'c=None', 'd=True', "e='hello'", 'f=(1, (2, 3))'])

  def test_dotted_name(self):
    if sys.version_info >= (3, 0):
      print('Warning: Doc generation is not supported from python3.')
      return

    # pylint: disable=g-bad-name
    class a(object):

      class b(object):

        class c(object):

          class d(object):

            def __init__(self, *args):
              pass
    # pylint: enable=g-bad-name

    e = {'f': 1}

    def example_fun(arg1=a.b.c.d, arg2=a.b.c.d(1, 2), arg3=e['f']):  # pylint: disable=unused-argument
      pass

    sig = parser._generate_signature(example_fun, reverse_index={})
    self.assertEqual(sig, ['arg1=a.b.c.d', 'arg2=a.b.c.d(1, 2)', "arg3=e['f']"])


if __name__ == '__main__':
  googletest.main()
