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
# import types
import unittest

import gast

from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.converters import arg_defaults
from tensorflow.python.autograph.converters import asserts
from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import builtin_functions
from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.converters import conditional_expressions
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.converters import directives
from tensorflow.python.autograph.converters import error_handlers
from tensorflow.python.autograph.converters import function_scopes
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.converters import logical_expressions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.converters import side_effect_guards
from tensorflow.python.autograph.converters import slices
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import errors as ag_errors
from tensorflow.python.autograph.core import function_wrapping
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.utils import ag_logging as logging
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
  # TODO(b/120224672): Fix this.
  if isinstance(o, functools.partial):
    # tf_inspect.getmodule(functools.partial(...)) otherwise returns None since
    # functools.partial objects do not have a __module__ attribute.
    m = functools
  else:
    m = tf_inspect.getmodule(o)

  if hasattr(m, '__name__'):
    # Builtins typically have unnamed modules.
    for prefix, in config.DEFAULT_UNCOMPILED_MODULES:
      if m.__name__.startswith(prefix):
        logging.log(2, '%s is whitelisted: name starts with "%s"', o, prefix)
        return True

    # Temporary -- whitelist tensorboard modules.
    # TODO(b/122731813): Remove.
    if m.__name__ == 'tensorboard' or '.tensorboard' in m.__name__:
      logging.log(2, '%s is whitelisted: name contains "tensorboard"', o)
      return True

  if hasattr(o, 'autograph_info__') or hasattr(o, '__ag_compiled'):
    logging.log(2, '%s is whitelisted: already converted', o)
    return True

  if hasattr(o, '__call__'):
    # Callable objects: whitelisted if their __call__ method is.
    # The type check avoids infinite recursion around the __call__ method
    # of function objects.
    if (type(o) != type(o.__call__)) and is_whitelisted_for_graph(o.__call__):  # pylint: disable=unidiomatic-typecheck
      logging.log(2, '%s is whitelisted: object __call__ whitelisted', o)
      return True

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
    if owner_class is not None:
      if issubclass(owner_class, unittest.TestCase):
        logging.log(2, '%s is whitelisted: method of TestCase subclass', o)
        return True

      owner_class = inspect_utils.getdefiningclass(o, owner_class)
      if is_whitelisted_for_graph(owner_class):
        logging.log(2, '%s is whitelisted: owner is whitelisted %s', o,
                    owner_class)
        return True

  if inspect_utils.isnamedtuple(o):
    # Due to the way they're constructed, namedtuple types cannot be converted
    # because they don't expose source code. But we assume they are safe for
    # graph mode since they are just containers.
    if tf_inspect.isclass(o) and len(o.__bases__) > 1:
      logging.warn(
          'Entity {} looks like a namedtuple subclass. Its constructor will'
          ' not be converted by AutoGraph, but if it has any custom methods,'
          ' those will be.'.format(o), 1)
    logging.log(2, '%s is whitelisted: named tuple', o)
    return True

  logging.log(2, '%s is NOT whitelisted', o)
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
  logging.log(1, 'Converting %s', o)

  if tf_inspect.isclass(o):
    node, name, ns = class_to_graph(o, program_ctx)
  elif tf_inspect.isfunction(o):
    node, name, ns = function_to_graph(o, program_ctx, arg_values, arg_types)
  elif tf_inspect.ismethod(o):
    node, name, ns = function_to_graph(o, program_ctx, arg_values, arg_types)
  # TODO(mdan,yashkatariya): Remove when object conversion is implemented.
  elif hasattr(o, '__class__'):
    raise NotImplementedError(
        'Object conversion is not yet supported. If you are '
        'trying to convert code that uses an existing object, '
        'try including the creation of that object in the '
        'conversion. For example, instead of converting the method '
        'of a class, try converting the entire class instead. '
        'See https://github.com/tensorflow/tensorflow/blob/master/tensorflow/'
        'contrib/autograph/README.md#using-the-functional-api '
        'for more information.')
  else:
    raise ValueError(
        'Entity "%s" has unsupported type "%s". Only functions and classes are '
        'supported for now.' % (o, type(o)))

  # TODO(mdan): This is temporary. it should be created using a converter.
  # TODO(mdan): The attribute should be added with a helper, not directly.
  # The helper can ensure there are no collisions.
  template = '''
      entity.autograph_info__ = {}
  '''
  node.extend(templates.replace(template, entity=name))

  program_ctx.add_to_cache(o, node)

  if logging.has_verbosity(2):
    logging.log(2, 'Compiled output of %s:\n\n%s\n', o,
                compiler.ast_to_source(node))
  if logging.has_verbosity(4):
    for n in node:
      logging.log(4, 'Compiled AST of %s:\n\n%s\n', o, gast.dump(n))

  if program_ctx.options.recursive:
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
    converted_members[m] = node[0]
  namer = program_ctx.new_namer(class_namespace)
  class_name = namer.compiled_class_name(c.__name__, c)

  # TODO(mdan): This needs to be explained more thoroughly.
  # Process any base classes: if the superclass if of a whitelisted type, an
  # absolute import line is generated. Otherwise, it is marked for conversion
  # (as a side effect of the call to namer.compiled_class_name() followed by
  # program_ctx.update_name_map(namer)).
  output_nodes = []
  renames = {}
  base_names = []
  for base in c.__bases__:
    if isinstance(object, base):
      base_names.append('object')
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
    base_names.append(alias)
    renames[qual_names.QN(base.__name__)] = qual_names.QN(alias)
  program_ctx.update_name_map(namer)

  # Generate the definition of the converted class.
  bases = [gast.Name(n, gast.Load(), None) for n in base_names]
  class_def = gast.ClassDef(
      class_name,
      bases=bases,
      keywords=[],
      body=list(converted_members.values()),
      decorator_list=[])
  # Make a final pass to replace references to the class or its base classes.
  # Most commonly, this occurs when making super().__init__() calls.
  # TODO(mdan): Making direct references to superclass' superclass will fail.
  class_def = qual_names.resolve(class_def)
  renames[qual_names.QN(c.__name__)] = qual_names.QN(class_name)
  class_def = ast_util.rename_symbols(class_def, renames)

  output_nodes.append(class_def)

  return output_nodes, class_name, class_namespace


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
    ag_internal.__dict__.update(autograph_module.__dict__)
    ag_internal.ConversionOptions = converter.ConversionOptions
    ag_internal.Feature = converter.Feature
    ag_internal.utils = utils
    ag_internal.function_scope = function_wrapping.function_scope
    ag_internal.rewrite_graph_construction_error = (
        ag_errors.rewrite_graph_construction_error)
    # TODO(mdan): Add safeguards against name clashes.
    # We don't want to create a submodule because we want the operators to be
    # accessible as ag__.<operator>
    ag_internal.__dict__.update(special_functions.__dict__)
    ag_internal.__dict__.update(operators.__dict__)

  _add_reserved_symbol(namespace, 'ag__', ag_internal)


def function_to_graph(f,
                      program_ctx,
                      arg_values,
                      arg_types,
                      owner_type=None):
  """Specialization of `entity_to_graph` for callable functions."""

  node, source = parser.parse_entity(f)
  logging.log(3, 'Source code of %s:\n%s', f, source)
  node = node.body[0]

  # In general, the output of inspect.getsource is inexact for lambdas because
  # it uses regex matching to adjust the exact location around the line number
  # that CPython records. Then, the entire containing line is returned, which
  # we may have trouble disambiguating. For example:
  # x, y = lambda: 1, lambda: 2
  if f.__name__ == '<lambda>':
    nodes = ast_util.find_matching_definitions(node, f)
    if len(nodes) != 1:
      raise ValueError(
          'Unable to identify source code of lambda function {}. It was'
          ' defined on this line: {}, which must contain a single lambda with'
          ' matching signature. To avoid ambiguity, define each lambda'
          ' in a separate expression.'.format(f, source))
    node, = nodes

  # TODO(znado): Place inside standard_analysis.
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
  try:
    node = node_to_graph(node, context)
  except (ValueError, AttributeError, KeyError, NotImplementedError) as e:
    logging.error(1, 'Error converting %s', f, exc_info=True)
    raise errors.InternalError('conversion', e)
    # TODO(mdan): Catch and rethrow syntax errors.

  if isinstance(node, gast.Lambda):
    new_name = namer.new_symbol('tf__lambda', ())
    node = gast.Assign(
        targets=[gast.Name(new_name, gast.Store(), None)], value=node)

  else:
    # TODO(mdan): This somewhat duplicates the renaming logic in call_trees.py
    new_name, did_rename = namer.compiled_function_name(f.__name__, f,
                                                        owner_type)
    if did_rename:
      node.name = new_name
    else:
      new_name = f.__name__
      assert node.name == new_name

  program_ctx.update_name_map(namer)
  # TODO(mdan): Use this at compilation.

  return [node], new_name, namespace


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
  node = converter.apply_(node, context, arg_defaults)
  node = converter.apply_(node, context, directives)
  node = converter.apply_(node, context, break_statements)
  if context.program.options.uses(converter.Feature.ASSERT_STATEMENTS):
    node = converter.apply_(node, context, asserts)
  # Note: sequencing continue canonicalization before for loop one avoids
  # dealing with the extra loop increment operation that the for
  # canonicalization creates.
  node = converter.apply_(node, context, continue_statements)
  node = converter.apply_(node, context, return_statements)
  if context.program.options.uses(converter.Feature.LISTS):
    node = converter.apply_(node, context, lists)
    node = converter.apply_(node, context, slices)
  if context.program.options.uses(converter.Feature.BUILTIN_FUNCTIONS):
    node = converter.apply_(node, context, builtin_functions)
  node = converter.apply_(node, context, call_trees)
  node = converter.apply_(node, context, control_flow)
  node = converter.apply_(node, context, conditional_expressions)
  if context.program.options.uses(converter.Feature.LOGICAL_EXPRESSIONS):
    node = converter.apply_(node, context, logical_expressions)
  if context.program.options.uses(converter.Feature.AUTO_CONTROL_DEPS):
    node = converter.apply_(node, context, side_effect_guards)
  # TODO(mdan): If function scopes ever does more, the toggle will need moving.
  if context.program.options.uses(converter.Feature.NAME_SCOPES):
    node = converter.apply_(node, context, function_scopes)
  if context.program.options.uses(converter.Feature.ERROR_REWRITING):
    node = converter.apply_(node, context, error_handlers)
  return node
