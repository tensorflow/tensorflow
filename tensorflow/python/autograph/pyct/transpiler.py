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

import inspect
import threading
import types

import gast

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
  local variables in the body of the outer factory, so that the Python parser
  correctly marks them as free non-global variables upon load (that is, it
  creates cell slots for each symbol. These symbols are initialized with None,
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


class _PythonFnFactory(object):
  """Helper object that wraps a Python function factory."""

  def __init__(self, name, freevars, extra_locals):
    """Creates a new factory for a Python function.

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
    """Initializes a function."""
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
    """Creates a new function instance."""
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
    new_fn = bound_factory(**self._extra_locals)  # pylint:disable=not-callable

    if defaults:
      new_fn.__defaults__ = defaults
    if kwdefaults:
      new_fn.__kwdefaults__ = kwdefaults

    return new_fn


class GenericTranspiler(object):
  """A generic transpiler for Python functions.

  Its interface is the `transform` API, which can process Python function
  objects. Internally, it handles parsing.

  Users typically subclass this, customizing the `transform_ast` method. The
  output of transformed_ast is returned directly by `transform`. Existing
  methods like `transform_function` may also be overloaded.

  Example:

      class MyTransformer(GenericTranspiler):

        def transform_ast(self, node, ctx):
          result = <<transform node>>
          return result

      transformer = MyTransfomer()

      result = transformer.transform(f, ...)
      # result is the output
  """

  def get_transformed_name(self, node):
    """Returns a name for the output function. Subclasses may override this."""
    if isinstance(node, gast.Lambda):
      return 'lam'
    elif isinstance(node, gast.FunctionDef):
      return node.name
    raise ValueError('Unknown node type {}'.format(node))

  def transform_ast(self, node, ctx):
    """Performs an actual transformation of a function's AST.

    Subclasses must implement this method, and do not usually call it.

    Args:
      node: One or more ast.AST nodes representing the AST to be transformed.
      ctx: transformer.Context.
    """
    raise NotImplementedError('subclasses must override this')

  def transform(self, obj, user_context):
    """Transforms a Python object.

    Users typically call this method.

    Args:
      obj: A Python object, function, type, etc.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user_context argument.
    Returns:
      The result of calling transform_function.

    Raises:
      NotImplementedError: if the type of obj is not handled.
    """
    if inspect.isfunction(obj) or inspect.ismethod(obj):
      return self.transform_function(obj, user_context)

    raise NotImplementedError('Non-function: {}'.format(type(obj)))

  def _erase_arg_defaults(self, node):
    """Erase arg default expressions, which would otherwise be unbound."""
    args = node.args
    for i in range(len(args.defaults)):
      args.defaults[i] = parser.parse_expression('None')
    for i, d in enumerate(args.kw_defaults):
      if d is not None:
        args.kw_defaults[i] = parser.parse_expression('None')
    return node

  def transform_module(self, mod, user_context):
    """Transforms a module.

    Subclasses may override this method. The return value is opaque.

    The method receives the original AST. The result is passed as-is to the
    output of `transform`.

    Args:
      mod: A Python module.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user_context argument.
    Returns:
      List[Tuple[Any, Any]]. By default it returns the output of transform_ast,
      evaluated on each supported member, other than modules, together with a
      `transformer.Context` containing information about the transformation
      process.
    """
    result = []
    for member in mod.__dict__.values():
      if inspect.ismodule(member):
        continue  # Not transforming modules recursively.
      try:
        result.append(self.transform(member, user_context))
      except NotImplementedError:
        pass  # Skip unsupported elements.
    return result

  def transform_function(self, fn, user_context):
    """Transforms a function.

    Subclasses may override this method. The return value is opaque.

    The method receives the original AST. The result is passed as-is to the
    output of `transform`.

    Args:
      fn: A function or lambda.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user_context argument.
    Returns:
      Tuple[Any, Any]. By default it returns the output of transform_ast,
      together with a `transformer.Context` containing information about the
      transformation process.
    """
    future_features = inspect_utils.getfutureimports(fn)
    node, source = parser.parse_entity(fn, future_features=future_features)
    logging.log(3, 'Source code of %s:\n\n%s\n', fn, source)

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
    result = self.transform_ast(node, context)

    return result, context


class PyToPy(GenericTranspiler):
  """A generic Python-to-Python transpiler.

  Its `transform` method offers a function-in, function-out interface.
  Internally, it takes care of parsing, caching and loading of the translated
  code.

  Users typically subclass this, overriding `transform_ast`.

  Usually, instances of this class are singletons, since each instance manages
  its own cache. The caching can be controlled by overriding `get_caching_key`.

  Example:

      class MyTransformer(PyToPy):

        def transform_ast(self, node, ctx):
          node = <<transform node, usually using ast.NodeTransformer classes>>
          return node

      transformer = MyTransfomer()

      new_f, module, source_map = transformer.transform_function(f, ...)
      # new_f is a function with signature identical to f

  The transformed function has access to the same namespace as the original
  function. To allow access to internal APIs, users may inject additional
  symbols by overriding `get_extra_locals`.
  """

  def __init__(self):
    self._cache_lock = threading.RLock()
    self._cache = cache.CodeObjectCache()

  def get_extra_locals(self):
    """Returns extra static local variables to be made to transformed code.

    Subclasses must override this.

    Returns:
      extra_locals: A Dict[Text, Any] containing additional variables to make
        available to the transformed code.
    """
    raise NotImplementedError('subclasses must override this')

  def get_caching_key(self, user_context):
    """Returns a unique key to use for caching.

    Subclasses must override this.

    Calls made to `transform_function` with functions that have the same code
    object and caching key will return a cached instance on subsequent
    invocations.

    Args:
      user_context: The context object which was passed to `transform`.

    Returns:
      extra_locals: A hashable.
    """
    raise NotImplementedError('subclasses must override this')

  def _cached_factory(self, fn, cache_subkey):
    cached_factory = self._cache[fn][cache_subkey]
    logging.log(3, 'Cache hit for %s subkey %s: %s', fn, cache_subkey,
                cached_factory)
    return cached_factory

  def transform_function(self, fn, user_context):
    """Transforms a function. See GenericTranspiler.trasnform_function.

    This overload wraps the parent's `transform_function`, adding caching and
    facilities to instantiate the output as a Python object. It also
    adds facilities to make new symbols available to the generated Python code,
    visible as local variables - see `get_extra_locals`.

    Args:
      fn: A function or lambda.
      user_context: An opaque object (may be None) that is forwarded to
        transform_ast, through the ctx.user_context argument.
    Returns:
      A tuple:
        * A function or lambda with the same signature and closure as `fn`
        * The temporary module into which the transformed function was loaded
        * The source map as a
            Dict[origin_info.LineLocation, origin_info.OriginInfo]
    """
    cache_subkey = self.get_caching_key(user_context)

    if self._cache.has(fn, cache_subkey):
      # Fast path: use a lock-free check.
      factory = self._cached_factory(fn, cache_subkey)

    else:
      with self._cache_lock:
        # Check again under lock.
        if self._cache.has(fn, cache_subkey):
          factory = self._cached_factory(fn, cache_subkey)

        else:
          logging.log(1, '%s is not cached for subkey %s', fn, cache_subkey)
          # TODO(mdan): Confusing overloading pattern. Fix.
          nodes, ctx = super(PyToPy, self).transform_function(fn, user_context)

          if isinstance(nodes, gast.Lambda):
            nodes = gast.Assign(
                targets=[
                    gast.Name(
                        ctx.info.name,
                        ctx=gast.Store(),
                        annotation=None,
                        type_comment=None)
                ],
                value=nodes)
          else:
            nodes.name = ctx.info.name

          if logging.has_verbosity(2):
            logging.log(2, 'Transformed %s:\n\n%s\n', fn, parser.unparse(nodes))

          factory = _PythonFnFactory(
              ctx.info.name, fn.__code__.co_freevars, self.get_extra_locals())
          factory.create(
              nodes, ctx.namer, future_features=ctx.info.future_features)
          self._cache[fn][cache_subkey] = factory

    transformed_fn = factory.instantiate(
        globals_=fn.__globals__,
        closure=fn.__closure__ or (),
        defaults=fn.__defaults__,
        kwdefaults=getattr(fn, '__kwdefaults__', None))
    return transformed_fn, factory.module, factory.source_map
