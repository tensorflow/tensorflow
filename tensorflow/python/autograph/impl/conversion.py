# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Core conversion logic, serves as main point of access."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import imp
import unittest

from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.converters import asserts
from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.converters import conditional_expressions
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.converters import directives
from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.converters import logical_expressions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.converters import slices
from tensorflow.python.autograph.converters import variables
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.autograph.core import unsupported_features_checker
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import cfg
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transpiler
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.autograph.pyct.static_analysis import reaching_definitions
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager import function
from tensorflow.python.util import tf_inspect


class AutoGraphTranspiler(transpiler.FunctionTranspiler):

  def get_transformed_name(self, node):
    return 'tf__' + super(AutoGraphTranspiler, self).get_transformed_name(node)

  def transform_ast(self, node, ctx):
    # TODO(mdan): Insert list_comprehensions somewhere.
    unsupported_features_checker.verify(node)

    # Run initial analysis.
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

    node = functions.transform(node, ctx)
    node = directives.transform(node, ctx)
    node = break_statements.transform(node, ctx)
    if ctx.user.options.uses(converter.Feature.ASSERT_STATEMENTS):
      node = asserts.transform(node, ctx)
    # Note: sequencing continue canonicalization before for loop one avoids
    # dealing with the extra loop increment operation that the for
    # canonicalization creates.
    node = continue_statements.transform(node, ctx)
    node = return_statements.transform(node, ctx)
    if ctx.user.options.uses(converter.Feature.LISTS):
      node = lists.transform(node, ctx)
      node = slices.transform(node, ctx)
    node = call_trees.transform(node, ctx)
    node = control_flow.transform(node, ctx)
    node = conditional_expressions.transform(node, ctx)
    node = logical_expressions.transform(node, ctx)
    node = variables.transform(node, ctx)
    return node


_TRANSPILER = AutoGraphTranspiler()
_WHITELIST_CACHE = cache.UnboundInstanceCache()


custom_vars = None


# TODO(mdan): Superfluous function, remove.
# TODO(mdan): Put these extra fields inside __autograph_info__.
def convert(entity, program_ctx):
  """Applies AutoGraph to entity."""

  if not hasattr(entity, '__code__'):
    raise ValueError('Cannot apply autograph to a function that doesn\'t '
                     'expose a __code__ object. If this is a @tf.function,'
                     ' try passing f.python_function instead.')

  _create_custom_vars(program_ctx)
  transformed, module, source_map = _TRANSPILER.transform_function(
      entity, program_ctx.options, program_ctx, custom_vars)

  assert not hasattr(transformed, 'ag_module')
  assert not hasattr(transformed, 'ag_source_map')
  transformed.ag_module = module
  transformed.ag_source_map = source_map
  return transformed


# TODO(mdan): allow_namedtuple_subclass should be hardcoded to True.
def is_whitelisted(
    o, check_call_override=True, allow_namedtuple_subclass=False):
  """Checks whether an entity is whitelisted for use in graph mode.

  Examples of whitelisted entities include all members of the tensorflow
  package.

  Args:
    o: A Python entity.
    check_call_override: Reserved for internal use. When set to `False`, it
      disables the rule according to which classes are whitelisted if their
      __call__ method is whitelisted.
    allow_namedtuple_subclass: Reserved for internal use. When `True`,
      namedtuple subclasses are not whitelisted.

  Returns:
    Boolean
  """
  # TODO(b/120224672): Fix this.
  if isinstance(o, functools.partial):
    # tf_inspect.getmodule(functools.partial(...)) otherwise returns None since
    # functools.partial objects do not have a __module__ attribute.
    m = functools
  else:
    m = tf_inspect.getmodule(o)

  # Examples of callables that lack a __module__ property include builtins.
  if hasattr(m, '__name__'):
    for rule in config.CONVERSION_RULES:
      action = rule.get_action(m)
      if action == config.Action.CONVERT:
        logging.log(2, 'Not whitelisted: %s: %s', o, rule)
        return False
      elif action == config.Action.DO_NOT_CONVERT:
        logging.log(2, 'Whitelisted: %s: %s', o, rule)
        return True

  # The check for __code__ below is because isgeneratorfunction crashes
  # without one.
  if hasattr(o, '__code__') and tf_inspect.isgeneratorfunction(o):
    logging.warn(
        'Entity %s appears to be a generator function. It will not be converted'
        ' by AutoGraph.', o)
    logging.log(2, 'Whitelisted: %s: generator functions are not converted', o)
    return True

  if (check_call_override and not tf_inspect.isclass(o) and
      hasattr(o, '__call__')):
    # Callable objects: whitelisted if their __call__ method is.
    # The type check avoids infinite recursion around the __call__ method
    # of function objects.
    if (type(o) != type(o.__call__)) and is_whitelisted(o.__call__):  # pylint: disable=unidiomatic-typecheck
      logging.log(2, 'Whitelisted: %s: object __call__ whitelisted', o)
      return True

  owner_class = None
  if tf_inspect.ismethod(o):
    # Methods of whitelisted classes are also whitelisted, even if they are
    # bound via user subclasses.
    #
    # For example, suppose `tf.Foo` has a method called `bar`, and `baz` is
    # defined as below. `tf.Foo` is whitelisted. Then `baz.bar` is also
    # whitelisted.
    #
    #   class Custom(tf.Foo):
    #     pass
    #
    #   baz = Custom()
    #
    # For the example above, if `Custom` did overload `bar`, then it would no
    # longer be whitelisted.

    owner_class = inspect_utils.getmethodclass(o)
    if owner_class is function.TfMethodTarget:
      owner_class = o.__self__.target_class
    if owner_class is not None:
      if issubclass(owner_class, unittest.TestCase):
        logging.log(2, 'Whitelisted: %s: method of TestCase subclass', o)
        return True

      owner_class = inspect_utils.getdefiningclass(o, owner_class)
      if is_whitelisted(
          owner_class,
          check_call_override=False,
          allow_namedtuple_subclass=True):
        logging.log(2, 'Whitelisted: %s: owner is whitelisted %s', o,
                    owner_class)
        return True

  if inspect_utils.isnamedtuple(o):
    # Due to the way they're constructed, namedtuple types cannot be converted
    # because they don't expose source code. But we assume they are safe for
    # graph mode since they are just containers.
    if allow_namedtuple_subclass:
      if not any(inspect_utils.isnamedtuple(base) for base in o.__bases__):
        logging.log(2, 'Whitelisted: %s: named tuple', o)
        return True
    else:
      logging.log(2, 'Whitelisted: %s: named tuple or subclass', o)
      return True

  logging.log(2, 'Not whitelisted: %s: default rule', o)
  return False


def is_in_whitelist_cache(entity, options):
  try:
    return _WHITELIST_CACHE.has(entity, options)
  except TypeError:
    # Catch-all for entities that are unhashable or don't allow weakrefs.
    return False


def cache_whitelisted(entity, options):
  try:
    _WHITELIST_CACHE[entity][options] = True
  except TypeError:
    # Catch-all for entities that are unhashable or don't allow weakrefs.
    pass


# TODO(mdan): Move into core or replace with an actual importable module.
def _create_custom_vars(program_ctx):
  """Adds namespace references to the module that exposes the api itself."""
  global custom_vars
  if custom_vars is None:
    # Craft a module that exposes parts of the external API as well as certain
    # internal modules.
    ag_internal = imp.new_module('autograph')
    ag_internal.__dict__.update(program_ctx.autograph_module.__dict__)
    ag_internal.ConversionOptions = converter.ConversionOptions
    ag_internal.STD = converter.STANDARD_OPTIONS
    ag_internal.Feature = converter.Feature
    ag_internal.utils = utils
    ag_internal.FunctionScope = function_wrappers.FunctionScope
    ag_internal.with_function_scope = function_wrappers.with_function_scope
    # TODO(mdan): Add safeguards against name clashes.
    # We don't want to create a submodule because we want the operators to be
    # accessible as ag__.<operator>
    ag_internal.__dict__.update(special_functions.__dict__)
    ag_internal.__dict__.update(operators.__dict__)

    custom_vars = {'ag__': ag_internal}
