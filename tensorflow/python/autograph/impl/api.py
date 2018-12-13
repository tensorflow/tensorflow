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
"""This module contains the user-facing API for AutoGraph."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import sys

from enum import Enum

# pylint:disable=g-bad-import-order
import numpy as np
# pylint:enable=g-bad-import-order


from tensorflow.python.autograph.core import config
from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.pyct import compiler
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.utils import py_func
from tensorflow.python.data.util import nest
from tensorflow.python.framework import tensor_util
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export

# TODO(mdan): Properly document the type hints.
# TODO(mdan): Reduce the type hint information to (module, type).
# (currently we require (module + class name, type))


# TODO(mdan): This should behave like to_graph (e.g. convert statically).
# TODO(znado): Make an alias so can write Verbosity directly without needing
# to write converter.
def convert(
    recursive=False,
    verbose=converter.Verbosity.BRIEF,
    optional_features=converter.Feature.ALL):
  """Decorator that compiles a function to use TensorFlow ops.

  The decorator is dynamic - it recompiles the target whenever the decorated
  function is called. This means the parameter values are known at conversion.
  It also means that repeated calls with different types of parameters will be
  correctly processed.

  Args:
    recursive: bool, whether to recursively convert any functions or classes
      that the converted function may use.
    verbose: converter.Verbosity, the level of verbosity.
    optional_features: converted.Feature, allows toggling optional or
      experimental features. When set to None, only the core features are
      enabled.

  Returns:
    Callable, a decorator that converts the given function into an equivalent
    function that uses TensorFlow ops.
  """

  def decorator(f):
    """Decorator implementation."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
      return converted_call(
          f, None,
          converter.ConversionOptions(
              recursive=recursive,
              verbose=verbose,
              force_conversion=True,
              optional_features=optional_features,
          ), *args, **kwargs)

    wrapper = tf_decorator.make_decorator(f, wrapper)

    # Sometimes the decorator is just desugared, making it impossible to detect.
    # This attribute makes detection easier.
    setattr(wrapper, '__ag_compiled', True)
    return wrapper

  return decorator


class RunMode(Enum):
  """Specifies the way a converted function or method should be executed in TF.

  Attributes:
   * GRAPH: Call this function directly, as-is. This is suitable for functions
       that were already designed for TF graphs and contain ops.
   * PY_FUNC: Wrap this function into a py_func op. This is suitable for code
       that will only run correctly in Python, for example code that renders
       to the display, reads keyboard input, etc.
  """
  GRAPH = 1
  PY_FUNC = 2


def do_not_convert(run_as=RunMode.GRAPH, return_dtypes=None):
  """Decorator that suppresses the conversion of a function.

  See also: docs/pyfunc_dtypes.md

  Args:
    run_as: RunMode, specifies how to use the function in TensorFlow.
    return_dtypes: Optional[Iterable[ Union[tf.DType,
      utils.py_func.MatchDType]]], the return data types of the converted
      function, if run_as is RunMode.PY_FUNC. Ignored otherwise. May be set to
      None if the function has no return values.

  Returns:
    Callable, a decorator that wraps the original function.
  """

  def decorator(f):
    """Decorator implementation."""

    @functools.wraps(f)
    def graph_wrapper(*args, **kwargs):
      return f(*args, **kwargs)

    @functools.wraps(f)
    def py_func_wrapper(*args, **kwargs):
      if kwargs:
        raise NotImplementedError('RunMode.PY_FUNC does not yet support kwargs')
      # TODO(mdan): Add support for kwargs.
      return py_func.wrap_py_func(
          f, return_dtypes, args, kwargs, use_dummy_return=not return_dtypes)

    if run_as == RunMode.GRAPH:
      wrapper = graph_wrapper
    elif run_as == RunMode.PY_FUNC:
      wrapper = py_func_wrapper
    else:
      raise ValueError('unknown value for run_as: %s' % run_as)

    # Sometimes the decorator is just desugared, making it impossible to detect.
    # This attribute makes detection easier.
    setattr(wrapper, '__ag_compiled', True)
    return wrapper

  return decorator


def converted_call(f, owner, options, *args, **kwargs):
  """Compiles a function call inline. For internal use only."""
  logging.vlog(logging.DEBUG, 'Converted call: %s; owner: %s', f, owner)

  if owner is not None:
    if not isinstance(f, str):
      raise ValueError(
          'When owner is specified, the function name must be specified as'
          ' a string: {}'.format(f))

    # Special case when the owner is a 'super' object. In that case lookups of
    # dynamic attributes won't work. See
    # inspect_utils.SuperWrapperForDynamicAttrs.
    if isinstance(owner, super):
      owner = inspect_utils.SuperWrapperForDynamicAttrs(owner)

    f = getattr(owner, f)

  if inspect_utils.isbuiltin(f):
    return py_builtins.overload_of(f)(*args, **kwargs)

  # TODO(mdan): This needs cleanup.
  # In particular, we may want to avoid renaming functions altogether.
  if not options.force_conversion and conversion.is_whitelisted_for_graph(f):

    # Args typically include `self`, as required by the conversion process.
    # When conversion is skipped, `self` is not necessary, because the
    # original bound method is being executed. This code removes it.
    if tf_inspect.ismethod(f) and args:
      f_class = inspect_utils.getmethodclass(f)
      if args[0] is f_class:
        args = args[1:]

    return f(*args, **kwargs)

  # internal_convert_user_code is for example turned off when issuing a dynamic
  # call conversion from generated code while in nonrecursive mode. In that
  # case we evidently don't want to recurse, but we still have to convert
  # things like builtins.
  if not options.internal_convert_user_code:
    return f(*args, **kwargs)

  # Unwrap functools.partial objects
  # TODO(mdan): Consider sharing unwrapping logic with tf_inspect.
  while isinstance(f, functools.partial):
    args = f.args + args
    new_kwargs = {}
    if f.keywords is not None:
      new_kwargs.update(f.keywords)
    new_kwargs.update(kwargs)
    kwargs = new_kwargs
    f = f.func

  if tf_inspect.isfunction(f) or tf_inspect.ismethod(f):
    # Regular functions
    target_entity = f
    arg_map_target = f
    f_class = inspect_utils.getmethodclass(f)

    # TODO(b/119246461): This may be more elegantly handled using __get__?
    if f_class is not None:
      # If this is a method call, it may or may not include self.
      #
      # Example when self is included:
      #   converted_call(to_graph(foo.bar), foo)
      #
      # Example when self is not included:
      #   super(...).foo(args)
      #
      if owner is not None and (not args or args[0] is not owner):
        effective_args = (owner,) + args
      else:
        # When the owner is not specified, use the result of
        # inspect_utils.getmethodclass.
        # TODO(b/119246461): Make sure an owner is always specified.
        if not args or args[0] is not f_class:
          effective_args = (f_class,) + args
        else:
          effective_args = (f_class,) + args[1:]
      partial_types = (f_class,)
    else:
      effective_args = args
      partial_types = ()

  elif tf_inspect.isclass(f):
    # Constructors
    target_entity = f
    arg_map_target = f.__init__
    effective_args = args
    partial_types = ()

  elif hasattr(f, '__call__') and hasattr(f, '__class__'):
    # Callable objects
    target_entity = f.__call__
    arg_map_target = f.__call__
    effective_args = (f,) + args
    partial_types = (f.__class__,)

  else:
    NotImplementedError('unknown callable type "%s"' % type(f))

  arg_values = tf_inspect.getcallargs(arg_map_target, *args, **kwargs)
  arg_types = {}
  for name, arg in arg_values.items():
    arg_class = arg.__class__
    arg_types[name] = (arg_class.__name__, arg_class)

  # When called from within a decorator, this is the only indication that
  # the function is a method - it appears that the decorator is applied
  # before the method is bound.
  if not partial_types:
    if 'self' in arg_values:
      if tf_inspect.isclass(arg_values['self'].__class__):
        partial_types = (arg_values['self'].__class__,)
    elif 'cls' in arg_values:
      if tf_inspect.isclass(arg_values['cls']):
        partial_types = (arg_values['cls'],)

  converted_f = to_graph(
      target_entity,
      recursive=options.recursive,
      verbose=options.verbose,
      arg_values=arg_values,
      arg_types=arg_types,
      strip_decorators=options.strip_decorators,
      optional_features=options.optional_features,
      experimental_partial_types=partial_types)

  result = converted_f(*effective_args, **kwargs)

  # The converted function's closure is simply inserted into the function's
  # module __dict__. Since modules are permanently cached, that results in
  # leaking the entire closure.
  # Normally, it's not safe to delete the module because that may release said
  # closure as well. However, in the case of converted_call we are certain the
  # function will not be executed again, so the closure should no longer be
  # needed so long as the function doesn't return any executable code.
  # TODO(mdan): Attach the closure properly, using cells.
  if all(map(_is_not_callable, nest.flatten(result))):
    del sys.modules[converted_f.__module__]

  return result


def _is_not_callable(obj):
  # TODO(brianklee): Handle case when obj is a tensor dependent on a py_func.
  if isinstance(obj, (int, float, complex, str, bool)):
    return True
  if isinstance(obj, (np.ndarray, np.generic)):
    return True
  if tensor_util.is_tensor(obj):
    return True
  return False


@tf_export('autograph.to_graph')
def to_graph(entity,
             recursive=True,
             verbose=converter.Verbosity.VERBOSE,
             arg_values=None,
             arg_types=None,
             strip_decorators=None,
             optional_features=converter.Feature.ALL,
             experimental_partial_types=None):
  """Converts a Python entity into a TensorFlow graph.

  Also see: `tf.autograph.to_code`, `tf.function`.

  Unlike `tf.function`, `to_graph` is a low-level transpiler that converts
  Python code to TensorFlow graph code. It does not implement any caching,
  variable management or create any actual ops, and is best used where greater
  control over the generated TensorFlow graph is desired. Another difference
  from `tf.function` is that `to_graph` will not wrap the graph into a
  TensorFlow function or a Python callable. Internally, `tf.function` uses
  `to_graph`.

  _Example Usage_

  ```python
    def foo(x):
      if x > 0:
        y = x * x
      else:
        y = -x
      return y

    converted_foo = to_graph(foo)

    x = tf.constant(1)
    y = converted_foo(x)  # converted_foo is a TensorFlow Op-like.
    assert is_tensor(y)
  ```

  Supported Python entities include:
    * functions
    * classes
    * object methods

  Functions are converted into new functions with converted code.

  Classes are converted by generating a new class whose methods use converted
  code.

  Methods are converted into unbound function that have an additional first
  argument called `self`.

  Args:
    entity: Python callable or class to convert.
    recursive: Whether to recursively convert any functions that the
      converted function may call.
    verbose: The level of printing verbosity to use, as a
      `tf.autograph.experimental.Verbosity` value.
    arg_values: Optional dict of value hints for symbols including
      function arguments mapping string names to actual values. For example,
      `arg_values={'a': 1}` will map the variable `a` to the value `1`.
    arg_types: Optional dict of type hints for symbols including function
      arguments. Type hints allow specifying just the type of a variable, rather
      than a specific value.
    strip_decorators: A tuple specifying decorators that should be
      excluded from the compiled output. By default, when converting a function
      before the decorators are applied, the compiled output will include those
      decorators.
    optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of
      optional features in the conversion process.
    experimental_partial_types: A `set` of `type` values, reserved for internal
      use.

  Returns:
    Same as `entity`, the converted Python function or class.

  Raises:
    ValueError: If the entity could not be converted.
  """
  if strip_decorators is None:
    strip_decorators = ()
  strip_decorators += (convert, do_not_convert, converted_call)

  program_ctx = converter.ProgramContext(
      options=converter.ConversionOptions(
          recursive=recursive,
          verbose=verbose,
          strip_decorators=strip_decorators,
          optional_features=optional_features),
      partial_types=experimental_partial_types,
      autograph_module=tf_inspect.getmodule(to_graph),
      uncompiled_modules=config.DEFAULT_UNCOMPILED_MODULES)
  _, name, namespace = conversion.entity_to_graph(entity, program_ctx,
                                                  arg_values, arg_types)

  nodes = []
  for dep in reversed(program_ctx.conversion_order):
    nodes.extend(program_ctx.dependency_cache[dep])

  compiled_module, _ = compiler.ast_to_object(
      nodes,
      source_prefix=program_ctx.required_imports,
      include_source_map=True)

  # The compiled code should see everything the entry entity saw.
  # TODO(mdan): This might not work well if the call tree spans modules?
  for key, val in namespace.items():
    # Avoid overwriting entities that have been transformed.
    if key not in compiled_module.__dict__:
      compiled_module.__dict__[key] = val
  for key, val in program_ctx.additional_symbols.items():
    if key not in compiled_module.__dict__:
      compiled_module.__dict__[key] = val
  compiled = getattr(compiled_module, name)

  if tf_inspect.isfunction(entity):
    compiled.__defaults__ = entity.__defaults__

  if hasattr(compiled, '__globals__'):
    # Remove self to avoid circular references. This will probably only work
    # so long as the function is not reentrant.
    del compiled.__globals__[name]

  # Need this so the source_mapping attribute is available for the context
  # manager to access for runtime errors.
  #
  # Note that compiler.ast_to_object attaches the source map 'ag_source_map__'
  # symbol to the compiled module.
  # TODO(mdan): Record this statically in the generated code.
  # TODO(mdan): Rename this attribute to 'autograph_info__'
  source_map_attribute_name = 'ag_source_map'
  if getattr(compiled, source_map_attribute_name, None) is not None:
    raise ValueError('cannot convert %s because is has an attribute '
                     '"%s", which is reserved for AutoGraph.' %
                     (compiled, source_map_attribute_name))
  setattr(compiled, source_map_attribute_name,
          compiled_module.__dict__['ag_source_map__'])

  return compiled


@tf_export('autograph.to_code')
def to_code(entity,
            recursive=True,
            arg_values=None,
            arg_types=None,
            indentation='  ',
            optional_features=converter.Feature.ALL,
            experimental_partial_types=None):
  """Similar to `to_graph`, but returns Python source code as a string.

  Also see: `tf.autograph.to_graph`.

  `to_graph` returns the Python source code that can be used to generate a
  TensorFlow graph that is functionally identical to the input Python code.

  Args:
    entity: Python callable or class to convert.
    recursive: Whether to recursively convert any functions that the
      converted function may call.
    arg_values: Optional dict of value hints for symbols including
      function arguments mapping string names to actual values. For example,
      `arg_values={'a': 1}` will map the variable `a` to the value `1`.
    arg_types: Optional dict of type hints for symbols including function
      arguments. Type hints allow specifying just the type of a variable, rather
      than a specific value.
    indentation: The string to use for indenting. Typically two or four spaces,
      or just the tab character.
    optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of
      optional features in the conversion process.
    experimental_partial_types: A `set` of `type` values, reserved for internal
      use.

  Returns:
    The converted code as string.
  """
  program_ctx = converter.ProgramContext(
      options=converter.ConversionOptions(
          recursive=recursive,
          verbose=converter.Verbosity.BRIEF,
          strip_decorators=(convert, do_not_convert, converted_call),
          optional_features=optional_features),
      partial_types=experimental_partial_types,
      autograph_module=tf_inspect.getmodule(to_graph),
      uncompiled_modules=config.DEFAULT_UNCOMPILED_MODULES)
  conversion.entity_to_graph(entity, program_ctx, arg_values, arg_types)

  code = '\n'.join(
      compiler.ast_to_source(program_ctx.dependency_cache[dep], indentation)
      for dep in reversed(program_ctx.conversion_order))

  return program_ctx.required_imports + '\n\n' + code
