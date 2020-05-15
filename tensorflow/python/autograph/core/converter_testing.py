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
import inspect
import sys

import six

from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.platform import test


def whitelist(entity):
  """Helper that marks a callable as whtelitisted."""
  if 'whitelisted_module_for_testing' not in sys.modules:
    whitelisted_mod = imp.new_module('whitelisted_module_for_testing')
    sys.modules['whitelisted_module_for_testing'] = whitelisted_mod
    config.CONVERSION_RULES = (
        (config.DoNotConvert('whitelisted_module_for_testing'),) +
        config.CONVERSION_RULES)

  entity.__module__ = 'whitelisted_module_for_testing'


def is_inside_generated_code():
  """Tests whether the caller is generated code. Implementation-specific."""
  frame = inspect.currentframe()
  try:
    frame = frame.f_back

    internal_stack_functions = ('converted_call', '_call_unconverted')
    # Walk up the stack until we're out of the internal functions.
    while (frame is not None and
           frame.f_code.co_name in internal_stack_functions):
      frame = frame.f_back
    if frame is None:
      return False

    return 'ag__' in frame.f_locals
  finally:
    del frame


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
  def compiled(self, node, namespace, symbols=()):
    source = None

    self.dynamic_calls = []
    # See api.converted_call
    def converted_call(
        f, args, kwargs, unused_opts=None, unused_function_ctx=None):
      """Mock version of api.converted_call."""
      self.dynamic_calls.append((args, kwargs))
      if kwargs is None:
        kwargs = {}
      return f(*args, **kwargs)

    def fake_autograph_artifact(f):
      setattr(f, 'fake_autograph_artifact', True)
      return f

    try:
      result, source, source_map = loader.load_ast(
          node, include_source_map=True)
      # TODO(mdan): Move the unparsing from converter into pyct and reuse here.

      # TODO(mdan): Move this into self.prepare()
      result.tf = self.make_fake_mod('fake_tf', *symbols)
      fake_ag = self.make_fake_mod('fake_ag', converted_call,
                                   converter.ConversionOptions)
      fake_ag.__dict__.update(operators.__dict__)
      fake_ag.__dict__.update(special_functions.__dict__)
      fake_ag.ConversionOptions = converter.ConversionOptions
      fake_ag.Feature = converter.Feature
      fake_ag.utils = utils
      fake_ag.FunctionScope = function_wrappers.FunctionScope
      fake_ag.autograph_artifact = fake_autograph_artifact
      result.ag__ = fake_ag
      result.ag_source_map__ = source_map
      for k, v in namespace.items():
        result.__dict__[k] = v
      yield result
    except Exception:  # pylint:disable=broad-except
      if source is None:
        print('Offending AST:\n%s' % pretty_printer.fmt(node, color=False))
      else:
        print('Offending source code:\n%s' % source)
      raise

  @contextlib.contextmanager
  def converted(self, entity, converter_module, namespace, tf_symbols=()):

    node, ctx = self.prepare(entity, namespace)

    if not isinstance(converter_module, (list, tuple)):
      converter_module = (converter_module,)
    for m in converter_module:
      node = m.transform(node, ctx)

    with self.compiled(node, namespace, tf_symbols) as result:
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

  def prepare(self, test_fn, namespace, recursive=True):
    namespace['ConversionOptions'] = converter.ConversionOptions

    future_features = ('print_function', 'division')
    node, source = parser.parse_entity(test_fn, future_features=future_features)
    namer = naming.Namer(namespace)
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(recursive=recursive),
        autograph_module=None)
    entity_info = transformer.EntityInfo(
        name=test_fn.__name__,
        source_code=source,
        source_file='<fragment>',
        future_features=future_features,
        namespace=namespace)
    ctx = transformer.Context(entity_info, namer, program_ctx)
    origin_info.resolve_entity(node, source, test_fn)

    graphs = cfg.build(node)
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx, None)
    node = reaching_definitions.resolve(node, ctx, graphs)
    anno.dup(
        node,
        {
            anno.Static.DEFINITIONS: anno.Static.ORIG_DEFINITIONS,
        },
    )

    return node, ctx
