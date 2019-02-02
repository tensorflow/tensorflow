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
"""Tests for lists module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import weakref

from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.platform import test


class TestConverter(converter.Base):
  pass


class ConversionOptionsTest(converter_testing.TestCase):

  def test_to_ast(self):
    opts = converter.ConversionOptions()

    namer = converter_testing.FakeNamer()
    program_ctx = converter.ProgramContext(
        options=opts,
        partial_types=None,
        autograph_module=None,
        uncompiled_modules=())
    entity_info = transformer.EntityInfo(
        source_code='',
        source_file='<fragment>',
        namespace={},
        arg_values=None,
        arg_types={},
        owner_type=None)
    ctx = converter.EntityContext(namer, entity_info, program_ctx)
    opts_ast = opts.to_ast(ctx)

    template = '''
    def test_fn():
      return opts_ast
    '''
    opts_packed = templates.replace(template, opts_ast=opts_ast)

    reparsed, _ = compiler.ast_to_object(opts_packed)
    reparsed.__dict__['ag__'] = self.make_fake_mod(
        'fake_ag', converter.ConversionOptions, converter.Feature)

    reparsed_opts = reparsed.test_fn()

    self.assertEqual(opts.recursive, reparsed_opts.recursive)
    self.assertEqual(opts.verbose, reparsed_opts.verbose)
    self.assertEqual(opts.force_conversion, reparsed_opts.force_conversion)
    self.assertEqual(
        opts.internal_convert_user_code,
        reparsed_opts.internal_convert_user_code)
    self.assertEqual(opts.optional_features, reparsed_opts.optional_features)

  def test_should_strip_weakrefs(self):
    def test_fn():
      pass

    def weak_test_fn_a():
      pass

    def weak_test_fn_b():
      pass

    def weak_test_fn_c():
      pass

    wr_a = weakref.ref(weak_test_fn_a)
    # Create an extra weakref to check whether the existence of multiple weak
    # references influences the process.
    _ = weakref.ref(weak_test_fn_b)
    wr_b = weakref.ref(weak_test_fn_b)
    _ = weakref.ref(weak_test_fn_c)

    opts = converter.ConversionOptions(strip_decorators=(test_fn, wr_a, wr_b))

    self.assertTrue(opts.should_strip(test_fn))
    self.assertTrue(opts.should_strip(weak_test_fn_a))
    self.assertTrue(opts.should_strip(weak_test_fn_b))
    self.assertFalse(opts.should_strip(weak_test_fn_c))


class ConverterBaseTest(converter_testing.TestCase):

  def test_get_definition_directive_basic(self):

    directive_key = object

    def test_fn():
      a = 1
      return a

    ns = {}
    node, ctx = self.prepare(test_fn, ns)
    symbol_a = node.body[1].value
    defs, = anno.getanno(symbol_a, anno.Static.ORIG_DEFINITIONS)
    defs.directives[directive_key] = {
        'test_arg': parser.parse_expression('foo'),
        'other_arg': parser.parse_expression('bar'),
    }
    c = TestConverter(ctx)
    value = c.get_definition_directive(symbol_a, directive_key, 'test_arg',
                                       None)
    self.assertEqual(value.id, 'foo')

  def test_get_definition_directive_default(self):

    directive_key = object

    def test_fn():
      a = 1
      return a

    ns = {}
    node, ctx = self.prepare(test_fn, ns)
    symbol_a = node.body[1].value
    c = TestConverter(ctx)
    value = c.get_definition_directive(symbol_a, directive_key, 'test_arg',
                                       parser.parse_expression('default'))
    self.assertEqual(value.id, 'default')

  def test_get_definition_directive_multiple_consistent(self):

    directive_key = object

    def test_fn():
      a = 1
      if a:
        a = 2
      return a

    ns = {}
    node, ctx = self.prepare(test_fn, ns)
    symbol_a = node.body[2].value
    defs = anno.getanno(symbol_a, anno.Static.ORIG_DEFINITIONS)
    defs[0].directives[directive_key] = {
        'test_arg': parser.parse_expression('foo'),
        'other_arg': parser.parse_expression('bar'),
    }
    defs[1].directives[directive_key] = {
        'test_arg': parser.parse_expression('foo'),
        'other_arg': parser.parse_expression('baz'),
    }
    c = TestConverter(ctx)
    value = c.get_definition_directive(symbol_a, directive_key, 'test_arg',
                                       None)
    self.assertEqual(value.id, 'foo')

  def test_get_definition_directive_multiple_inconsistent(self):

    directive_key = object

    def test_fn():
      a = 1
      if a:
        a = 2
      return a

    ns = {}
    node, ctx = self.prepare(test_fn, ns)
    symbol_a = node.body[2].value
    defs = anno.getanno(symbol_a, anno.Static.ORIG_DEFINITIONS)
    defs[0].directives[directive_key] = {
        'test_arg': parser.parse_expression('foo'),
    }
    defs[1].directives[directive_key] = {
        'test_arg': parser.parse_expression('bar'),
    }
    c = TestConverter(ctx)
    with self.assertRaises(ValueError):
      c.get_definition_directive(symbol_a, directive_key, 'test_arg', None)


if __name__ == '__main__':
  test.main()
