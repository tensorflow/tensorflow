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

import imp

import gast

from tensorflow.contrib.autograph import operators
from tensorflow.contrib.autograph import utils
from tensorflow.contrib.autograph.converters import asserts
from tensorflow.contrib.autograph.converters import break_statements
from tensorflow.contrib.autograph.converters import builtin_functions
from tensorflow.contrib.autograph.converters import call_trees
from tensorflow.contrib.autograph.converters import conditional_expressions
from tensorflow.contrib.autograph.converters import continue_statements
from tensorflow.contrib.autograph.converters import control_flow
from tensorflow.contrib.autograph.converters import decorators
from tensorflow.contrib.autograph.converters import directives
from tensorflow.contrib.autograph.converters import error_handlers
from tensorflow.contrib.autograph.converters import lists
from tensorflow.contrib.autograph.converters import logical_expressions
from tensorflow.contrib.autograph.converters import name_scopes
from tensorflow.contrib.autograph.converters import return_statements
from tensorflow.contrib.autograph.converters import side_effect_guards
from tensorflow.contrib.autograph.converters import slices
from tensorflow.contrib.autograph.core import config
from tensorflow.contrib.autograph.core import converter
from tensorflow.contrib.autograph.core import errors
from tensorflow.contrib.autograph.pyct import ast_util
from tensorflow.contrib.autograph.pyct import inspect_utils
from tensorflow.contrib.autograph.pyct import origin_info
from tensorflow.contrib.autograph.pyct import parser
from tensorflow.contrib.autograph.pyct import qual_names
from tensorflow.contrib.autograph.pyct import transformer
from tensorflow.python.util import tf_inspect


# TODO(mdan): Might we not need any renaming at all?


def is_whitelisted_for_graph(o):
  """Check whether an entity is whitelisted for use in graph mode.

  Examples of whitelisted entities include all members of the tensorflow
  package.

  Args:
    o: A Python entity.
  Returns:
    Boolean
  """
  m = tf_inspect.getmodule(o)
  for prefix, in config.DEFAULT_UNCOMPILED_MODULES:
    if m.__name__.startswith(prefix):
      return True
  return False


def entity_to_graph(o, program_ctx, arg_values, arg_types):
  """Compile a Python entity into equivalent TensorFlow.

  The function will also recursively compile all the entities that `o`
  references, updating `dependency_cache`.

  This function is reentrant, and relies on dependency_cache to avoid
  generating duplicate code.

  Args:
    o: A Python entity.
    program_ctx: A ProgramContext object.
    arg_values: A dict containing value hints for symbols like function
        parameters.
    arg_types: A dict containing type hints for symbols like function
        parameters.

  Returns:
    A tuple (ast, new_name, namespace):
        * ast: An AST representing an entity with interface equivalent to `o`,
            but which when executed it creates TF a graph.
        * new_name: The symbol name under which the new entity can be found.
        * namespace: A dict mapping all symbols visible to the converted entity,
            keyed by their symbol name.

  Raises:
    ValueError: if the entity type is not supported.
  """
  if tf_inspect.isclass(o):
    node, name, ns = class_to_graph(o, program_ctx)
  elif tf_inspect.isfunction(o):
    # TODO(mdan): This is not a reliable mechanism.
    # The most reliable way is to check the source code, the AST will contain
    # a Lambda node instead of a FunctionDef
    if o.__name__ == '<lambda>':
      raise NotImplementedError(
          'lambda functions are not yet supported; declare the function'
          ' using def instead: %s' % o)
    else:
      node, name, ns = function_to_graph(o, program_ctx, arg_values, arg_types)
  elif tf_inspect.ismethod(o):
    node, name, ns = function_to_graph(o, program_ctx, arg_values, arg_types)
  else:
    raise ValueError(
        'Entity "%s" has unsupported type "%s". Only functions and classes are '
        'supported for now.' % (o, type(o)))

  program_ctx.add_to_cache(o, node)
  if program_ctx.recursive:
    while True:
      candidate = None
      for obj in program_ctx.name_map.keys():
        if obj not in program_ctx.dependency_cache:
          candidate = obj
          break
      if candidate is None:
        break
      if (hasattr(candidate, 'im_class') and
          getattr(candidate, 'im_class') not in program_ctx.partial_types):
        # Class members are converted with their objects, unless they're
        # only converted partially.
        continue
      entity_to_graph(candidate, program_ctx, {}, {})

  return node, name, ns


def class_to_graph(c, program_ctx):
  """Specialization of `entity_to_graph` for classes."""
  converted_members = {}
  method_filter = lambda m: tf_inspect.isfunction(m) or tf_inspect.ismethod(m)
  members = tf_inspect.getmembers(c, predicate=method_filter)
  if not members:
    raise ValueError('Cannot convert %s: it has no member methods.' % c)

  class_namespace = {}
  for _, m in members:
    # Only convert the members that are directly defined by the class.
    if inspect_utils.getdefiningclass(m, c) is not c:
      continue
    node, _, namespace = function_to_graph(
        m,
        program_ctx=program_ctx,
        arg_values={},
        arg_types={'self': (c.__name__, c)},
        owner_type=c)
    if class_namespace is None:
      class_namespace = namespace
    else:
      class_namespace.update(namespace)
    converted_members[m] = node
  namer = program_ctx.new_namer(class_namespace)
  class_name = namer.compiled_class_name(c.__name__, c)

  # TODO(mdan): This needs to be explained more thoroughly.
  # Process any base classes: if the sueprclass if of a whitelisted type, an
  # absolute import line is generated. Otherwise, it is marked for conversion
  # (as a side effect of the call to namer.compiled_class_name() followed by
  # program_ctx.update_name_map(namer)).
  output_nodes = []
  renames = {}
  bases = []
  for base in c.__bases__:
    if isinstance(object, base):
      bases.append('object')
      continue
    if is_whitelisted_for_graph(base):
      alias = namer.new_symbol(base.__name__, ())
      output_nodes.append(
          gast.ImportFrom(
              module=base.__module__,
              names=[gast.alias(name=base.__name__, asname=alias)],
              level=0))
    else:
      # This will trigger a conversion into a class with this name.
      alias = namer.compiled_class_name(base.__name__, base)
    bases.append(alias)
    renames[qual_names.QN(base.__name__)] = qual_names.QN(alias)
  program_ctx.update_name_map(namer)

  # Generate the definition of the converted class.
  output_nodes.append(
      gast.ClassDef(
          class_name,
          bases=bases,
          keywords=[],
          body=list(converted_members.values()),
          decorator_list=[]))
  node = gast.Module(output_nodes)

  # Make a final pass to replace references to the class or its base classes.
  # Most commonly, this occurs when making super().__init__() calls.
  # TODO(mdan): Making direct references to superclass' superclass will fail.
  node = qual_names.resolve(node)
  renames[qual_names.QN(c.__name__)] = qual_names.QN(class_name)
  node = ast_util.rename_symbols(node, renames)

  return node, class_name, class_namespace


def _add_reserved_symbol(namespace, name, entity):
  if name not in namespace:
    namespace[name] = entity
  elif namespace[name] != entity:
    raise ValueError('The name "%s" is reserved and may not be used.' % name)


ag_internal = None


def _add_self_references(namespace, autograph_module):
  """Adds namespace references to the module that exposes the api itself."""
  global ag_internal
  if ag_internal is None:
    # Craft a module that exposes parts of the external API as well as certain
    # internal modules.
    ag_internal = imp.new_module('autograph')
    ag_internal.converted_call = autograph_module.converted_call
    ag_internal.utils = utils
    ag_internal.rewrite_graph_construction_error = (
        errors.rewrite_graph_construction_error)
    # TODO(mdan): Add safeguards against name clashes.
    # We don't want to create a submodule because we want the operators to be
    # accessible as ag__.<operator>
    ag_internal.__dict__.update(operators.__dict__)

  _add_reserved_symbol(namespace, 'ag__', ag_internal)


def function_to_graph(f, program_ctx, arg_values, arg_types, owner_type=None):
  """Specialization of `entity_to_graph` for callable functions."""

  node, source = parser.parse_entity(f)
  node = node.body[0]
  origin_info.resolve(node, source, f)
  namespace = inspect_utils.getnamespace(f)
  _add_self_references(namespace, program_ctx.autograph_module)
  namer = program_ctx.new_namer(namespace)

  entity_info = transformer.EntityInfo(
      source_code=source,
      source_file='<fragment>',
      namespace=namespace,
      arg_values=arg_values,
      arg_types=arg_types,
      owner_type=owner_type)
  context = converter.EntityContext(namer, entity_info, program_ctx)
  node = node_to_graph(node, context)

  # TODO(mdan): This somewhat duplicates the call rename logic in call_treest.py
  new_name, did_rename = namer.compiled_function_name(f.__name__, f, owner_type)
  if not did_rename:
    new_name = f.__name__
    if node.name != f.__name__:
      raise NotImplementedError('Strange corner case. Send us offending code!')

  node.name = new_name
  program_ctx.update_name_map(namer)
  # TODO(mdan): Use this at compilation.

  return node, new_name, namespace


def node_to_graph(node, context):
  """Convert Python code to equivalent TF graph mode code.

  Args:
    node: AST, the code to convert.
    context: converter.EntityContext

  Returns:
    A tuple (node, deps):
        * node: A Python ast node, representing the converted code.
        * deps: A set of strings, the fully qualified names of entity
            dependencies that this node has.
  """
  # TODO(mdan): Insert list_comprehensions somewhere.

  node = converter.standard_analysis(node, context, is_initial=True)
  # Past this point, line numbers are no longer accurate so we ignore the
  # source.
  # TODO(mdan): Is it feasible to reconstruct intermediate source code?
  context.info.source_code = None

  node = converter.apply_(node, context, decorators)
  node = converter.apply_(node, context, directives)
  node = converter.apply_(node, context, break_statements)
  node = converter.apply_(node, context, asserts)
  # Note: sequencing continue canonicalization before for loop one avoids
  # dealing with the extra loop increment operation that the for
  # canonicalization creates.
  node = converter.apply_(node, context, continue_statements)
  context.info.namespace['len'] = len
  node = converter.apply_(node, context, return_statements)
  node = converter.apply_(node, context, lists)
  node = converter.apply_(node, context, slices)
  node = converter.apply_(node, context, builtin_functions)
  node = converter.apply_(node, context, call_trees)
  node = converter.apply_(node, context, control_flow)
  node = converter.apply_(node, context, conditional_expressions)
  node = converter.apply_(node, context, logical_expressions)
  node = converter.apply_(node, context, side_effect_guards)
  node = converter.apply_(node, context, name_scopes)
  node = converter.apply_(node, context, error_handlers)
  return node
