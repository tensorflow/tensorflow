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
"""Base class for tests in this module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import imp
import sys

import six

from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import errors
from tensorflow.python.autograph.core import function_wrapping
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.platform import test


def imported_decorator(f):
  return lambda a: f(a) + 1


# TODO(mdan): We might be able to use the real namer here.
class FakeNamer(object):
  """A fake namer that uses a global counter to generate unique names."""

  def __init__(self):
    self.i = 0

  def new_symbol(self, name_root, used):
    while True:
      self.i += 1
      name = '%s%d' % (name_root, self.i)
      if name not in used:
        return name

  def compiled_function_name(self,
                             original_fqn,
                             live_entity=None,
                             owner_type=None):
    del live_entity
    if owner_type is not None:
      return None, False
    return ('renamed_%s' % '_'.join(original_fqn)), True


class FakeNoRenameNamer(FakeNamer):

  def compiled_function_name(self, original_fqn, **_):
    return str(original_fqn), False


class TestCase(test.TestCase):
  """Base class for unit tests in this module. Contains relevant utilities."""

  @contextlib.contextmanager
  def assertPrints(self, expected_result):
    try:
      out_capturer = six.StringIO()
      sys.stdout = out_capturer
      yield
      self.assertEqual(out_capturer.getvalue(), expected_result)
    finally:
      sys.stdout = sys.__stdout__

  @contextlib.contextmanager
  def compiled(self, node, namespace, *symbols):
    source = None

    self.dynamic_calls = []
    def converted_call(*args):
      """Mock version of api.converted_call."""
      self.dynamic_calls.append(args)
      return 7

    try:
      result, source = compiler.ast_to_object(node, include_source_map=True)

      # TODO(mdan): Move this into self.prepare()
      result.tf = self.make_fake_mod('fake_tf', *symbols)
      fake_ag = self.make_fake_mod('fake_ag', converted_call,
                                   converter.ConversionOptions)
      fake_ag.__dict__.update(operators.__dict__)
      fake_ag.__dict__.update(special_functions.__dict__)
      fake_ag.__dict__['utils'] = utils
      fake_ag.__dict__['rewrite_graph_construction_error'] = (
          errors.rewrite_graph_construction_error)
      fake_ag.__dict__['function_scope'] = function_wrapping.function_scope
      result.__dict__['ag__'] = fake_ag
      for k, v in namespace.items():
        result.__dict__[k] = v
      yield result
    except Exception:  # pylint:disable=broad-except
      if source is None:
        print('Offending AST:\n%s' % pretty_printer.fmt(node, color=False))
      else:
        print('Offending compiled code:\n%s' % source)
      raise

  @contextlib.contextmanager
  def converted(self, entity, converter_module, namespace, *tf_symbols):
    node, ctx = self.prepare(entity, namespace)

    if not isinstance(converter_module, (list, tuple)):
      converter_module = (converter_module,)
    for i, m in enumerate(converter_module):
      node = converter.standard_analysis(node, ctx, is_initial=not i)
      node = m.transform(node, ctx)

    with self.compiled(node, namespace, *tf_symbols) as result:
      yield result

  def make_fake_mod(self, name, *symbols):
    fake_mod = imp.new_module(name)
    for s in symbols:
      if hasattr(s, '__name__'):
        setattr(fake_mod, s.__name__, s)
      elif hasattr(s, 'name'):
        # This is a bit of a hack, but works for things like tf.int32
        setattr(fake_mod, s.name, s)
      else:
        raise ValueError('can not attach %s - what should be its name?' % s)
    return fake_mod

  def attach_namespace(self, module, **ns):
    for k, v in ns.items():
      setattr(module, k, v)

  def prepare(self,
              test_fn,
              namespace,
              namer=None,
              arg_types=None,
              owner_type=None,
              recursive=True,
              strip_decorators=()):
    namespace['ConversionOptions'] = converter.ConversionOptions

    node, source = parser.parse_entity(test_fn)
    node = node.body[0]
    if namer is None:
      namer = FakeNamer()
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(
            recursive=recursive,
            strip_decorators=strip_decorators,
            verbose=True),
        partial_types=None,
        autograph_module=None,
        uncompiled_modules=config.DEFAULT_UNCOMPILED_MODULES)
    entity_info = transformer.EntityInfo(
        source_code=source,
        source_file='<fragment>',
        namespace=namespace,
        arg_values=None,
        arg_types=arg_types,
        owner_type=owner_type)
    ctx = converter.EntityContext(namer, entity_info, program_ctx)
    origin_info.resolve(node, source, test_fn)
    node = converter.standard_analysis(node, ctx, is_initial=True)
    return node, ctx
