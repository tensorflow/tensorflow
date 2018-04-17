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

from tensorflow.contrib.autograph import operators
from tensorflow.contrib.autograph import utils
from tensorflow.contrib.autograph.pyct import compiler
from tensorflow.contrib.autograph.pyct import context
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import pretty_printer
from tensorflow.contrib.autograph.pyct import qual_names
from tensorflow.contrib.autograph.pyct.static_analysis import activity
from tensorflow.contrib.autograph.pyct.static_analysis import live_values
from tensorflow.contrib.autograph.pyct.static_analysis import type_info
from tensorflow.python.platform import test


class FakeNamer(object):

  def new_symbol(self, name_root, used):
    i = 0
    while True:
      name = '%s%d' % (name_root, i)
      if name not in used:
        return name
      i += 1

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
  def compiled(self, node, *symbols):
    source = None

    self.dynamic_calls = []
    def converted_call(*args):
      """Mock version of api.converted_call."""
      self.dynamic_calls.append(args)
      return 7

    try:
      result, source = compiler.ast_to_object(node)
      result.tf = self.make_fake_mod('fake_tf', *symbols)
      fake_ag = self.make_fake_mod('fake_ag', converted_call)
      fake_ag.__dict__.update(operators.__dict__)
      fake_ag.__dict__['utils'] = utils
      result.__dict__['ag__'] = fake_ag
      yield result
    except Exception:  # pylint:disable=broad-except
      if source is None:
        print('Offending AST:\n%s' % pretty_printer.fmt(node, color=False))
      else:
        print('Offending compiled code:\n%s' % source)
      raise

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

  def parse_and_analyze(self,
                        test_fn,
                        namespace,
                        namer=None,
                        arg_types=None,
                        include_type_analysis=True,
                        owner_type=None,
                        recursive=True):
    node, source = parser.parse_entity(test_fn)
    ctx = context.EntityContext(
        namer=namer or FakeNamer(),
        source_code=source,
        source_file=None,
        namespace=namespace,
        arg_values=None,
        arg_types=arg_types,
        owner_type=owner_type,
        recursive=recursive,
        type_annotation_func=utils.set_element_type)
    node = qual_names.resolve(node)
    node = activity.resolve(node, ctx)
    node = live_values.resolve(node, ctx, {})
    if include_type_analysis:
      node = type_info.resolve(node, ctx)
      node = live_values.resolve(node, ctx, {})
    self.ctx = ctx
    return node
