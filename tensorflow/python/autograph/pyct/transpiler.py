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
"""Generic source code transformation infrastructure."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import threading
import types

import gast

from tensorflow.python.autograph.pyct import ast_util
from tensorflow.python.autograph.pyct import cache
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import loader
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import templates
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.utils import ag_logging as logging


def _wrap_into_factory(nodes, entity_name, inner_factory_name,
                       outer_factory_name, closure_vars, factory_args,
                       future_features):
  """Wraps an AST into the body of a factory with consistent lexical context.

  The AST is expected to define some symbol with a name given by `entity_name`.

  This mechanism ensures that the resulting transformed entity has lexical
  scoping identical to that of the source entity, while allowing extra
  parametrization.

  Two nested factories achieve the following:

   1. The inner factory dynamically creates the entity represented by `nodes`.
   2. The inner factory is parametrized by a custom set of arguments.
   3. The inner factory has a closure identical to that of the transformed
       entity.
   4. The inner factory has local variables named like `args`, which `nodes` may
       use as additional parameters.
   5. The inner factory returns the variables given by `entity_name`.
   6. The outer factory is niladic.
   7. The outer factory has no closure.
   8. The outer factory creates the necessary lexical scope for the inner
       factory, so that the loaded code has the given configuration for
       closure/globals.
   9. The outer factory returns the inner factory.

  Roughly speaking, the following code is generated:

      from __future__ import future_feature_1
      from __future__ import future_feature_2
      ...

      def outer_factory():
        closure_var_1 = None
        closure_var_2 = None
        ...

        def inner_factory(arg_1, arg_2, ...):
          <<nodes>>
          return entity

        return inner_factory

  The lexical scoping is created using dummy symbol declarations which create
  local fariables in the body of the outer factory, so that the Python parser
  correctly marks them as free non-global variables upon load (that is, it
  creates cell slots for each symbol. Thes symbols are initialized with None,
  but their values are not expected to be used; instead, the caller is expected
  to replace them with the cells of the source entity. For more details, see:
  https://docs.python.org/3/reference/executionmodel.html#binding-of-names

  Args:
    nodes: Tuple[ast.AST], the source code to wrap.
    entity_name: Union[Text, ast.AST], the name of the principal entity that
      `nodes` define.
    inner_factory_name: Text, the name of the inner factory.
    outer_factory_name: Text, the name of the outer factory.
    closure_vars: Iterable[Text], names of the closure variables for the inner
      factory.
    factory_args: Iterable[Text], names of additional arguments for the
      inner factory. Useful to configure variables that the converted code can
      use. Typically, these are modules.
    future_features: Iterable[Text], names of future statements to associate the
      code with.

  Returns:
    ast.AST
  """
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

  factory_args = [
      gast.Name(name, ctx=gast.Param(), annotation=None, type_comment=None)
      for name in factory_args
  ]

  template = """
    future_imports
    def outer_factory_name():
      dummy_closure_defs
      def inner_factory_name(factory_args):
        entity_defs
        return entity_name
      return inner_factory_name
  """
  return templates.replace(
      template,
      dummy_closure_defs=dummy_closure_defs,
      entity_defs=nodes,
      entity_name=entity_name,
      factory_args=factory_args,
      future_imports=future_imports,
      inner_factory_name=inner_factory_name,
      outer_factory_name=outer_factory_name)


class _TransformedFnFactory(object):
  """Helper object that wraps a transformed function factory."""

  def __init__(self, name, freevars, extra_locals):
    """Creates a new factory for a transformed function.

    Args:
      name: The function name.
      freevars: The list of non-global free variables for the function.
      extra_locals: Dict[Text, Any], names and values for custom variables that
        are accessible to the generated code as local variables.
    """
    self._name = name
    self._freevars = freevars
    self._extra_locals = extra_locals

    self._unbound_factory = None
    self.module = None
    self.source_map = None

  def create(self,
             nodes,
             namer,
             inner_factory_name='inner_factory',
             outer_factory_name='outer_factory',
             future_features=()):
    """Initializes a transformed function."""
    if self._unbound_factory is not None:
      raise ValueError('double initialization; create a new object instead')

    inner_factory_name = namer.new_symbol(inner_factory_name, ())
    outer_factory_name = namer.new_symbol(outer_factory_name, ())
    nodes = _wrap_into_factory(nodes, self._name, inner_factory_name,
                               outer_factory_name, self._freevars,
                               self._extra_locals.keys(), future_features)

    module, _, source_map = loader.load_ast(
        nodes, include_source_map=True)
    outer_factory = getattr(module, outer_factory_name)
    self._unbound_factory = outer_factory()
    self.module = module
    self.source_map = source_map

  def instantiate(self,
                  globals_,
                  closure,
                  defaults=None,
                  kwdefaults=None):
    """Creates a new instance of the transformed function."""
    if self._unbound_factory is None:
      raise ValueError('call create first')

    factory_code = self._unbound_factory.__code__
    factory_freevars = factory_code.co_freevars
    closure_map = dict(zip(self._freevars, closure))
    factory_closure = tuple(
        closure_map[name] for name in factory_code.co_freevars)
    if len(factory_closure) != len(closure):
      raise ValueError(
          'closure mismatch, requested {}, but source function had {}'.format(
              self._freevars, factory_freevars))

    bound_factory = types.FunctionType(
        code=factory_code,
        globals=globals_,
        name=self._name,
        argdefs=(),
        closure=factory_closure)

    # The lint override is a false positive.
    transformed_entity = bound_factory(**self._extra_locals)  # pylint:disable=not-callable

    if defaults:
      transformed_entity.__defaults__ = defaults
    if kwdefaults:
      transformed_entity.__kwdefaults__ = kwdefaults

    return transformed_entity


class FunctionTranspiler(object):
  """A generic source-to-source transpiler for Python functions.

  Its interface `transform_function` API offers a function-in, function-out
  interface. Internally, it takes care of parsing, caching and variable binding.

  Users typically subclass this, customizing the transform_ast method.

  Usually, instances of this class are singletons, since each instance manages
  its own cache. The caching subkey allows managing multiple types of
  transformation.

  Example:

      class MyTransformer(FunctionTranspiler):

        def transform_ast(self, node, ctx):
          node = <<transform node, usually using ast.NodeTransformer classes>>
          return node

      transformer = MyTransfomer()

      new_f, module, source_map = transformer.transform_function(f, ...)
      # new_f is a function with signature identical to f

  The transformed function has access to the same namespace as the original
  function. To allow access to internal APIs, users may inject additional
  symbols though the `extra_locals` argument of `transform_function`.
  """

  def __init__(self):
    self._cache_lock = threading.RLock()
    self._cache = cache.CodeObjectCache()

  def transform_ast(self, node, user_context):
    """Performs an actual transformation of a function's AST.

    Subclasses must implement this method. They must not call it.

    The method receives the original AST and generates code according to the
    AST that the method returns. For functions, the returned AST is expected to
    contain a function with the exact same arguments and closure. The resulting
    function will receive the globals, closure and argument defaults of the
    input function.

    Args:
      node: One or more ast.AST nodes representing the AST to be transformed.
      user_context: The same value that the caller passed to
        `transform_function`.
    """
    raise NotImplementedError('subclasses must override this')

  def get_transformed_name(self, node):
    """Returns a name for the output function. Subclasses may override this."""
    if isinstance(node, gast.Lambda):
      return 'lam'
    elif isinstance(node, gast.FunctionDef):
      # Note that we need to rename the function, to avoid any namespace
      # clashes.
      return node.name
    else:
      raise ValueError('Unknown node type {}'.format(node))

  def _erase_arg_defaults(self, node):
    """Erase argde fault expressions, which would otherwise be unbound."""
    args = node.args
    for i in range(len(args.defaults)):
      args.defaults[i] = parser.parse_expression('None')
    for i, d in enumerate(args.kw_defaults):
      if d is not None:
        args.kw_defaults[i] = parser.parse_expression('None')
    return node

  def _transform_function(self, fn, user_context):
    """Performs source code transformation on a function."""
    future_features = inspect_utils.getfutureimports(fn)
    node, source = parser.parse_entity(fn, future_features=future_features)
    logging.log(3, 'Source code of %s:\n\n%s\n', fn, source)

    # In general, the output of inspect.getsource is inexact for lambdas
    # because it uses regex matching to adjust the exact location around
    # the line number that CPython records. Then, the entire containing line
    # is returned, which we may have trouble disambiguating.
    # For example:
    #   x, y = lambda: 1, lambda: 2
    is_lambda = fn.__name__ == '<lambda>'
    if is_lambda:
      nodes = ast_util.find_matching_definitions(node, fn)
      if len(nodes) != 1:
        raise ValueError(
            'Unable to identify source code of lambda function {}.'
            ' It was defined in this code:\n'
            '{}\n'
            'This code must contain a single distinguishable lambda.'
            ' To avoid this problem, define each lambda in a separate'
            ' expression.'.format(fn, source))
      node, = nodes

    origin_info.resolve_entity(node, source, fn)

    namespace = inspect_utils.getnamespace(fn)
    namer = naming.Namer(namespace)
    new_name = namer.new_symbol(self.get_transformed_name(node), ())
    entity_info = transformer.EntityInfo(
        name=new_name,
        source_code=source,
        source_file='<fragment>',
        future_features=future_features,
        namespace=namespace)
    context = transformer.Context(entity_info, namer, user_context)

    node = self._erase_arg_defaults(node)
    node = self.transform_ast(node, context)

    if is_lambda:
      node = gast.Assign(
          targets=[
              gast.Name(
                  new_name,
                  ctx=gast.Store(),
                  annotation=None,
                  type_comment=None)
          ],
          value=node)
    else:
      node.name = new_name

    return node, context

  def _cached_factory(self, fn, cache_subkey):
    cached_factory = self._cache[fn][cache_subkey]
    logging.log(3, 'Cache hit for %s subkey %s: %s', fn, cache_subkey,
                cached_factory)
    return cached_factory

  def _transformed_factory(self, fn, cache_subkey, user_context, extra_locals):
    """Returns the transformed function factory for a given input."""
    if self._cache.has(fn, cache_subkey):
      return self._cached_factory(fn, cache_subkey)

    with self._cache_lock:
      # Check again under lock.
      if self._cache.has(fn, cache_subkey):
        return self._cached_factory(fn, cache_subkey)

      logging.log(1, '%s is not cached for subkey %s', fn, cache_subkey)
      nodes, ctx = self._transform_function(fn, user_context)

      if logging.has_verbosity(2):
        logging.log(2, 'Transformed %s:\n\n%s\n', fn, parser.unparse(nodes))

      factory = _TransformedFnFactory(
          ctx.info.name, fn.__code__.co_freevars, extra_locals)
      factory.create(nodes, ctx.namer, future_features=ctx.info.future_features)
      self._cache[fn][cache_subkey] = factory
    return factory

  def transform_function(self, fn, caching_subkey, user_context, extra_locals):
    """Transforms a function.

    The `caching_subkey` argument allows mapping each function to multiple
    outputs in the cache. This is useful for instance when transformers
    can generate multiple variants of output code, typically as a result of
    different transformation flags.

    Args:
      fn: A function or lambda.
      caching_subkey: Used for caching. Calls made for functions with the same
        code object and caching_subkey will return a cached instance on
        subsequent invocations. Using a constant will create unique per-function
        entries.
      user_context: An opaque object (may be none) that is forwarded to
        transform_ast.
      extra_locals: A Dict[Text, Any] containing additional variables to make
        available to the transformed code. These will be visible as local
        variables.
    Returns:
      A tuple:
        * A function or lambda with the same signature and closure as `fn`
        * The temporary module into which the transformed function was loaded
        * The source map as a
            Dict[origin_info.LineLocation, origin_info.OriginInfo]

    """
    factory = self._transformed_factory(fn, caching_subkey, user_context,
                                        extra_locals)

    transformed_fn = factory.instantiate(
        globals_=fn.__globals__,
        closure=fn.__closure__ or (),
        defaults=fn.__defaults__,
        kwdefaults=getattr(fn, '__kwdefaults__', None))
    return transformed_fn, factory.module, factory.source_map
