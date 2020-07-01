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

from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import api
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.framework import ops
from tensorflow.python.platform import test


def whitelist(f):
  """Helper that marks a callable as whtelitisted."""
  if 'whitelisted_module_for_testing' not in sys.modules:
    whitelisted_mod = imp.new_module('whitelisted_module_for_testing')
    sys.modules['whitelisted_module_for_testing'] = whitelisted_mod
    config.CONVERSION_RULES = (
        (config.DoNotConvert('whitelisted_module_for_testing'),) +
        config.CONVERSION_RULES)

  f.__module__ = 'whitelisted_module_for_testing'


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


class TestingTranspiler(conversion.AutoGraphTranspiler):
  """Testing version that only applies given transformations."""

  def __init__(self, converters):
    super(TestingTranspiler, self).__init__()
    if isinstance(converters, (list, tuple)):
      self._converters = converters
    else:
      self._converters = (converters,)
    self.transformed_ast = None

  def transform_ast(self, node, ctx):
    node = self.initial_analysis(node, ctx)

    for c in self._converters:
      node = c.transform(node, ctx)

    self.transformed_ast = node
    self.transform_ctx = ctx
    return node


class TestCase(test.TestCase):
  """Base class for unit tests in this module. Contains relevant utilities."""

  def setUp(self):
    # AutoGraph tests must run in graph mode to properly test control flow.
    self.graph = ops.Graph().as_default()
    self.graph.__enter__()

  def tearDown(self):
    self.graph.__exit__(None, None, None)

  @contextlib.contextmanager
  def assertPrints(self, expected_result):
    try:
      out_capturer = six.StringIO()
      sys.stdout = out_capturer
      yield
      self.assertEqual(out_capturer.getvalue(), expected_result)
    finally:
      sys.stdout = sys.__stdout__

  def transform(
      self, f, converter_module, include_ast=False, ag_overrides=None):
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(recursive=True),
        autograph_module=api)

    conversion.create_custom_vars(program_ctx)
    custom_vars = dict(conversion.custom_vars)

    if ag_overrides:
      modified_ag = imp.new_module('fake_autograph')
      modified_ag.__dict__.update(custom_vars['ag__'].__dict__)
      modified_ag.__dict__.update(ag_overrides)
      custom_vars['ag__'] = modified_ag

    tr = TestingTranspiler(converter_module)
    transformed, _, _ = tr.transform_function(
        f, program_ctx.options, program_ctx, custom_vars)

    if include_ast:
      return transformed, tr.transformed_ast, tr.transform_ctx

    return transformed
