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

import collections
import functools
import imp
import inspect
import sys
import threading
import types
import unittest
import weakref

import gast

from tensorflow.python.autograph import operators
from tensorflow.python.autograph import utils
from tensorflow.python.autograph.converters import arg_defaults
from tensorflow.python.autograph.converters import asserts
from tensorflow.python.autograph.converters import break_statements
from tensorflow.python.autograph.converters import call_trees
from tensorflow.python.autograph.converters import conditional_expressions
from tensorflow.python.autograph.converters import continue_statements
from tensorflow.python.autograph.converters import control_flow
from tensorflow.python.autograph.converters import directives
from tensorflow.python.autograph.converters import function_scopes
from tensorflow.python.autograph.converters import lists
from tensorflow.python.autograph.converters import logical_expressions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.converters import slices
from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.core import function_wrappers
from tensorflow.python.autograph.core import naming
from tensorflow.python.autograph.core import unsupported_features_checker
from tensorflow.python.autograph.lang import special_functions
from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import pretty_printer
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.eager import function
from tensorflow.python.util import tf_inspect


class _ConvertedEntityFactoryInfo(
    collections.namedtuple(
        '_ConvertedEntityFactoryInfo',
        ('module_name', 'converted_name', 'factory_factory_name', 'source_map'))
):
  """Holds metadata about a converted entity stored as a dynamic factory.

  The dynamic factory is assumed to be created by _wrap_into_dynamic_factory,
  be named `factory_factory_name` and located inside the module named as
  `module_name`.

  Attributes:
    module_name: Text, the name of the module containing the entity.
    converted_name: Text, the name of the converted entity.
    factory_factory_name: Text, the name of the dynamic factory.
    source_map: Dict.
  """

  def __str__(self):
    return '_ConvertedEntityFactoryInfo({} in {})'.format(
        self.converted_name, self.module_name)

  def get_module(self):
    return sys.modules[self.module_name]

  def get_factory(self):
    assert self.module_name in sys.modules
    factory_factory = getattr(sys.modules[self.module_name],
                              self.factory_factory_name)
    return factory_factory()


# TODO(mdan): Add a garbage collection hook for cleaning up modules.
class _FunctionCache(object):
  """A hierarchical cache that uses the converted entity as weak key.

  The keys soft references (i.e. they are discarded when the key is
  destroyed). The subkeys are normal hashable values.

  This class is generic - see the call site for how the keys and values are
  defined.
  """

  __slots__ = ('_cache',)

  def __init__(self):
    self._cache = weakref.WeakKeyDictionary()

  def _get_key(self, entity):
    raise NotImplementedError('subclasses will override')

  def has(self, entity, subkey):
    key = self._get_key(entity)
    if key not in self._cache:
      return False
    return subkey in self._cache[key]

  def __getitem__(self, entity):
    key = self._get_key(entity)
    if key not in self._cache:
      # The bucket needs to be initialized to support this usage:
      #   cache[key][subkey] = value
      self._cache[key] = {}
    return self._cache[key]

  def __len__(self):
    return len(self._cache)


class _CodeObjectCache(_FunctionCache):
  """A function cache based on code objects (i.e., the source code).

  Multiple functions may share the same code object, but they may share the
  cache because we know they have the exact source code. This properly handles
  functions defined in a loop, bound methods, etc.

  Falls back to the function object, if it doesn't have a code object.
  """

  def _get_key(self, entity):
    if hasattr(entity, '__code__'):
      return entity.__code__
    else:
      return entity


class _UnboundInstanceCache(_FunctionCache):
  """A function cache based on unbound function objects.

  Unlike the _CodeObjectCache, this discriminates between different functions
  even if they have the same code. This properly handles decorators that may
  masquerade as various functions. Bound functions are not discriminated by
  the object they're bound to.
  """

  def _get_key(self, entity):
    if inspect.ismethod(entity):
      return entity.__func__
    return entity


# Using a re-entrant lock to guard against the unlikely possibility that the
# conversion process tiggers additional code execution.
_CACHE_LOCK = threading.RLock()


_CACHE = _CodeObjectCache()
_WHITELIST_CACHE = _UnboundInstanceCache()


# Note: strictly speaking, a simple factory might have been sufficient for
# functions. But the double factory approach allows us to control the closure
# and globals of the converted code in a cleaner fashion.
# TODO(mdan): A simple factory may be sufficient.
def _wrap_into_dynamic_factory(nodes, entity_name, factory_factory_name,
                               factory_name, closure_vars, future_features):
  """Wraps an AST into the body of a dynamic factory.

  This uses the dynamic factory (factory of factory) pattern to achieve the
  following:

   1. The inner factory, dynamically creates the entity represented by nodes.
   2. The entity is parametrized by `ag__`, the internal AutoGraph module.
   3. The outer factory creates the inner factory with a lexical scope
      in which `closure_vars` are bound local variables. This in turn allows the
      caller to control the exact closure (i.e. non-global free variables) for
      the inner factory.

  The AST is expected to define some symbol named by `entity_name`.

  Args:
    nodes: ast.AST
    entity_name: Union[Text, ast.AST]
    factory_factory_name: Text
    factory_name: Text
    closure_vars: Iterable[Text]
    future_features: Iterable[Text], see EntityInfo.future_features.

  Returns:
    ast.AST
  """
  if not isinstance(nodes, (list, tuple)):
    nodes = (nodes,)

  dummy_closure_defs = []
  for var_name in closure_vars:
    template = """
      var_name = None
    """
    dummy_closure_defs.extend(templates.replace(template, var_name=var_name))

  if future_features:
    future_imports = gast.ImportFrom(
        module='__future__',
        names=[gast.alias(name=name, asname=None) for name in future_features],
        level=0)
  else:
    future_imports = []

  # These dummy symbol declarations create local fariables in a function scope,
  # so that the Python parser correctly marks them as free non-global variables
  # upon load (that is, it creates cell slots for each symbol). Their values are
  # not used, as the cells are swapped with the original entity's cells after
  # the code has been loaded.
  template = """
    future_imports
    def factory_factory_name():
      dummy_closure_defs
      def factory_name(ag__, ag_source_map__, ag_module__):
        entity_defs
        entity_name.ag_source_map = ag_source_map__
        entity_name.ag_module = ag_module__
        entity_name = ag__.autograph_artifact(entity_name)
        return entity_name
      return factory_name
  """
  return templates.replace(
      template,
      future_imports=future_imports,
      factory_factory_name=factory_factory_name,
      factory_name=factory_name,
      dummy_closure_defs=dummy_closure_defs,
      entity_defs=nodes,
      entity_name=entity_name)


def _convert_with_cache(entity, program_ctx, free_nonglobal_var_names):
  """Returns a (possibly cached) factory for the converted result of entity."""
  # The cache subkey encompases any conversion options on which the generated
  # code may depend.
  # The cached factory includes the necessary definitions to distinguish
  # between the global and non-global free variables. For this reason, the
  # cache subkey includes the names of the free non-globals.
  subkey = (program_ctx.options, frozenset(free_nonglobal_var_names))

  with _CACHE_LOCK:
    # The cache values are _ConvertedEntityFactoryInfo objects.
    if _CACHE.has(entity, subkey):
      # TODO(mdan): Check whether the module is still loaded.
      converted_entity_info = _CACHE[entity][subkey]
      logging.log(3, 'Cache hit for entity %s subkey %s: %s', entity, subkey,
                  converted_entity_info)
      return converted_entity_info

    logging.log(1, 'Entity %s is not cached for subkey %s', entity, subkey)

    nodes, converted_name, entity_info = convert_entity_to_ast(
        entity, program_ctx)

    namer = naming.Namer(entity_info.namespace)
    factory_factory_name = namer.new_symbol('create_converted_entity_factory',
                                            ())
    factory_name = namer.new_symbol('create_converted_entity', ())
    nodes = _wrap_into_dynamic_factory(nodes, converted_name,
                                       factory_factory_name, factory_name,
                                       free_nonglobal_var_names,
                                       entity_info.future_features)

    module, _, source_map = loader.load_ast(nodes, include_source_map=True)
    module_name = module.__name__

    converted_entity_info = _ConvertedEntityFactoryInfo(
        module_name=module_name,
        converted_name=converted_name,
        factory_factory_name=factory_factory_name,
        source_map=source_map)
    _CACHE[entity][subkey] = converted_entity_info
    return converted_entity_info


def _instantiate(entity, converted_entity_info, free_nonglobal_var_names):
  """Creates a converted instance and binds it to match original entity."""
  factory = converted_entity_info.get_factory()

  # `factory` is currently bound to the empty module it was loaded from.
  # It must instead be bound to the globals and closure from the original
  # entity.
  if tf_inspect.isfunction(entity) or tf_inspect.ismethod(entity):
    entity_globals = entity.__globals__
    entity_closure = entity.__closure__ or ()
  elif hasattr(entity, '__module__'):
    entity_globals = sys.modules[entity.__module__].__dict__
    entity_closure = ()
  assert len(entity_closure) == len(free_nonglobal_var_names)

  # Fit the original entity's cells to match the order of factory's cells.
  original_names_and_cells = dict(zip(free_nonglobal_var_names, entity_closure))
  new_factory_cells = tuple(
      original_names_and_cells[name] for name in factory.__code__.co_freevars)

  bound_factory = types.FunctionType(
      code=factory.__code__,
      globals=entity_globals,
      name=factory.__name__,
      argdefs=(),
      closure=new_factory_cells)

  # Two other free vars: the internal "ag__" module and the source
  # map. These are wired via the parameters of the factory.
  converted_entity = bound_factory(  # pylint:disable=not-callable
      ag_internal, converted_entity_info.source_map,
      converted_entity_info.get_module())

  if tf_inspect.isfunction(entity) or tf_inspect.ismethod(entity):
    # Attach the default argument to the converted function.
    converted_entity.__defaults__ = entity.__defaults__
    if hasattr(entity, '__kwdefaults__'):
      converted_entity.__kwdefaults__ = entity.__kwdefaults__

  return converted_entity


def convert(entity, program_ctx):
  """Converts an entity into an equivalent entity."""

  if tf_inspect.isfunction(entity) or tf_inspect.ismethod(entity):
    if not hasattr(entity, '__code__'):
      raise ValueError('Cannot apply autograph to a function that doesn\'t '
                       'expose a __code__ object. If this is a @tf.function,'
                       ' try passing f.python_function instead.')
    free_nonglobal_var_names = entity.__code__.co_freevars
  else:
    free_nonglobal_var_names = ()

  for i, name in enumerate(free_nonglobal_var_names):
    if (name == 'ag__' and
        entity.__closure__[i].cell_contents is not ag_internal):
      raise ValueError('entity {} uses the reserved symbol "{}"'.format(
          entity, name))
    # TODO(mdan): In extreme cases, other ag__ symbols may also be clobbered.

  converted_entity_info = _convert_with_cache(entity, program_ctx,
                                              free_nonglobal_var_names)

  return _instantiate(entity, converted_entity_info, free_nonglobal_var_names)


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

  if tf_inspect.isgeneratorfunction(o):
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


# TODO(mdan): Rename to convert_*_node to avoid confusion with convert.
def convert_entity_to_ast(o, program_ctx):
  """Compile a Python entity into equivalent TensorFlow.

  Args:
    o: A Python entity.
    program_ctx: A ProgramContext object.

  Returns:
    A tuple (ast, new_name, namespace):
        * ast: An AST representing an entity with interface equivalent to `o`,
            but which when executed it creates TF a graph.
        * new_name: The symbol name under which the new entity can be found.
        * namespace: A dict mapping all symbols visible to the converted entity,
            keyed by their symbol name.

  Raises:
    NotImplementedError: if entity is of a type that is not yet supported.
  """
  logging.log(1, 'Converting %s', o)

  if tf_inspect.isclass(o):
    nodes, name, entity_info = convert_class_to_ast(o, program_ctx)
  elif tf_inspect.isfunction(o):
    nodes, name, entity_info = convert_func_to_ast(o, program_ctx)
  elif tf_inspect.ismethod(o):
    nodes, name, entity_info = convert_func_to_ast(o, program_ctx)
  elif hasattr(o, '__class__'):
    # Note: this should only be raised when attempting to convert the object
    # directly. converted_call should still support it.
    raise NotImplementedError(
        'cannot convert entity "{}": object conversion is not yet'
        ' supported.'.format(o))
  else:
    raise NotImplementedError(
        'Entity "%s" has unsupported type "%s". Only functions and classes are '
        'supported for now.' % (o, type(o)))

  if logging.has_verbosity(2):
    logging.log(2, 'Compiled output of %s:\n\n%s\n', o, parser.unparse(nodes))
  if logging.has_verbosity(4):
    for n in nodes:
      logging.log(4, 'Compiled AST of %s:\n\n%s\n\n', o,
                  pretty_printer.fmt(n, color=False))

  return nodes, name, entity_info


def convert_class_to_ast(c, program_ctx):
  """Specialization of `convert_entity_to_ast` for classes."""
  # TODO(mdan): Revisit this altogether. Not sure we still need it.
  converted_members = {}
  method_filter = lambda m: tf_inspect.isfunction(m) or tf_inspect.ismethod(m)
  members = tf_inspect.getmembers(c, predicate=method_filter)
  if not members:
    raise ValueError('cannot convert %s: no member methods' % c)

  # TODO(mdan): Don't clobber namespaces for each method in one class namespace.
  # The assumption that one namespace suffices for all methods only holds if
  # all methods were defined in the same module.
  # If, instead, functions are imported from multiple modules and then spliced
  # into the class, then each function has its own globals and __future__
  # imports that need to stay separate.

  # For example, C's methods could both have `global x` statements referring to
  # mod1.x and mod2.x, but using one namespace for C would cause a conflict.
  # from mod1 import f1
  # from mod2 import f2
  # class C(object):
  #   method1 = f1
  #   method2 = f2

  class_namespace = {}
  future_features = None
  for _, m in members:
    # Only convert the members that are directly defined by the class.
    if inspect_utils.getdefiningclass(m, c) is not c:
      continue
    (node,), _, entity_info = convert_func_to_ast(
        m, program_ctx=program_ctx, do_rename=False)
    class_namespace.update(entity_info.namespace)
    converted_members[m] = node

    # TODO(mdan): Similarly check the globals.
    if future_features is None:
      future_features = entity_info.future_features
    elif frozenset(future_features) ^ frozenset(entity_info.future_features):
      # Note: we can support this case if ever needed.
      raise ValueError(
          'cannot convert {}: if has methods built with mismatched future'
          ' features: {} and {}'.format(c, future_features,
                                        entity_info.future_features))
  namer = naming.Namer(class_namespace)
  class_name = namer.class_name(c.__name__)

  # Process any base classes: if the superclass if of a whitelisted type, an
  # absolute import line is generated.
  output_nodes = []
  renames = {}
  base_names = []
  for base in c.__bases__:
    if isinstance(object, base):
      base_names.append('object')
      continue
    if is_whitelisted(base):
      alias = namer.new_symbol(base.__name__, ())
      output_nodes.append(
          gast.ImportFrom(
              module=base.__module__,
              names=[gast.alias(name=base.__name__, asname=alias)],
              level=0))
    else:
      raise NotImplementedError(
          'Conversion of classes that do not directly extend classes from'
          ' whitelisted modules is temporarily suspended. If this breaks'
          ' existing code please notify the AutoGraph team immediately.')
    base_names.append(alias)
    renames[qual_names.QN(base.__name__)] = qual_names.QN(alias)

  # Generate the definition of the converted class.
  bases = [
      gast.Name(n, ctx=gast.Load(), annotation=None, type_comment=None)
      for n in base_names]
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

  # TODO(mdan): Find a way better than forging this object.
  entity_info = transformer.EntityInfo(
      source_code=None,
      source_file=None,
      future_features=future_features,
      namespace=class_namespace)

  return output_nodes, class_name, entity_info


def _add_reserved_symbol(namespace, name, entity):
  if name not in namespace:
    namespace[name] = entity
  elif namespace[name] != entity:
    raise ValueError('The name "%s" is reserved and may not be used.' % name)


ag_internal = None


# TODO(mdan): Move into core or replace with an actual importable module.
def _add_self_references(namespace, autograph_module):
  """Adds namespace references to the module that exposes the api itself."""
  global ag_internal
  if ag_internal is None:
    # Craft a module that exposes parts of the external API as well as certain
    # internal modules.
    ag_internal = imp.new_module('autograph')
    ag_internal.__dict__.update(autograph_module.__dict__)
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

  _add_reserved_symbol(namespace, 'ag__', ag_internal)


def convert_func_to_ast(f, program_ctx, do_rename=True):
  """Specialization of `convert_entity_to_ast` for callable functions."""

  future_features = inspect_utils.getfutureimports(f)
  node, source = parser.parse_entity(f, future_features=future_features)
  logging.log(3, 'Source code of %s:\n\n%s\n', f, source)
  # Parsed AST should contain future imports and one function def node.

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
  origin_info.resolve_entity(node, source, f)

  namespace = inspect_utils.getnamespace(f)
  _add_self_references(namespace, program_ctx.autograph_module)
  namer = naming.Namer(namespace)

  if isinstance(node, gast.Lambda):
    new_name = namer.new_symbol('tf__lambda', ())
  elif do_rename:
    new_name = namer.function_name(f.__name__)
  else:
    new_name = f.__name__

  entity_info = transformer.EntityInfo(
      source_code=source,
      source_file='<fragment>',
      future_features=future_features,
      namespace=namespace)
  context = converter.EntityContext(namer, entity_info, program_ctx, new_name)
  node = node_to_graph(node, context)

  if isinstance(node, gast.Lambda):
    node = gast.Assign(
        targets=[
            gast.Name(
                new_name, ctx=gast.Store(), annotation=None, type_comment=None)
        ],
        value=node)
  elif do_rename:
    node.name = new_name
  else:
    assert node.name == new_name

  return (node,), new_name, entity_info


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
  unsupported_features_checker.verify(node)

  node = converter.standard_analysis(node, context, is_initial=True)
  node = converter.apply_(node, context, function_scopes)
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
  node = converter.apply_(node, context, call_trees)
  node = converter.apply_(node, context, control_flow)
  node = converter.apply_(node, context, conditional_expressions)
  node = converter.apply_(node, context, logical_expressions)
  return node
