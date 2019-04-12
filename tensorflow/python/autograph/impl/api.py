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

import collections
import copy
import functools
import inspect
import os
import pdb
import sys
import textwrap

from enum import Enum

# pylint:disable=g-bad-import-order
import six
# pylint:enable=g-bad-import-order


from tensorflow.python.autograph.core import converter
from tensorflow.python.autograph.impl import conversion
from tensorflow.python.autograph.operators import py_builtins
from tensorflow.python.autograph.pyct import errors
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.utils import ag_logging as logging
from tensorflow.python.autograph.utils import py_func
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.python.util.tf_export import tf_export


def is_autograph_strict_conversion_mode():
  return int(os.environ.get('AUTOGRAPH_STRICT_CONVERSION', '0')) > 0


# TODO(mdan): Properly document the type hints.
# TODO(mdan): Reduce the type hint information to (module, type).
# (currently we require (module + class name, type))


# TODO(mdan): This should behave like to_graph (e.g. convert statically).
# TODO(znado): Make an alias so can write Verbosity directly without needing
# to write converter.
def convert(
    recursive=False,
    optional_features=None):
  """Decorator that compiles a function to use TensorFlow ops.

  The decorator is dynamic - it recompiles the target whenever the decorated
  function is called. This means the parameter values are known at conversion.
  It also means that repeated calls with different types of parameters will be
  correctly processed.

  Args:
    recursive: bool, whether to recursively convert any functions or classes
      that the converted function may use.
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
              force_conversion=True,
              optional_features=optional_features,
          ), args, kwargs)

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


def do_not_convert_internal(f):
  """Decorator that marks internal functions which do not need conversion."""
  setattr(f, '__ag_compiled', True)
  return f


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

    setattr(wrapper, '__ag_compiled', True)
    return wrapper

  return decorator


def _call_unconverted(f, args, kwargs):
  """Calls the original function without converting with AutoGraph."""
  if inspect_utils.istfmethodtarget(f):
    return f.__self__.call(args, kwargs)

  if kwargs is not None:
    return f(*args, **kwargs)
  else:
    return f(*args)


def _is_known_loaded_type(f, module_name, entity_name):
  """Tests whether the function or method is an instance of a known type."""
  if (module_name not in sys.modules or
      not hasattr(sys.modules[module_name], entity_name)):
    return False
  type_entity = getattr(sys.modules[module_name], entity_name)
  if isinstance(f, type_entity):
    # The method if of this type. Example:
    #
    # o = ClassType()
    # function(o.method)()
    return True
  # Note: inspect is required here, to avoid unpacking tf.function decorators.
  if inspect.ismethod(f):
    f = six.get_unbound_function(f)
    # The the unbound method if of this type. Example:
    #
    # class ClassType:
    #   @function
    #   def method(self):
    #     ...
    # o = ClassType()
    # o.method()
    if isinstance(f, type_entity):
      return True
  return False


def converted_call(f, owner, options, args, kwargs):
  """Compiles a function call inline. For internal use only."""
  if owner is not None:
    if not isinstance(f, str):
      raise ValueError(
          'When owner is specified, the function name must be specified as'
          ' a string: {}'.format(f))
    owner_attr = f

    # Special case when the owner is a 'super' object. In that case lookups of
    # dynamic attributes won't work. See
    # inspect_utils.SuperWrapperForDynamicAttrs.
    if isinstance(owner, super):
      owner = inspect_utils.SuperWrapperForDynamicAttrs(owner)

    f = getattr(owner, f)

  if logging.has_verbosity(1):
    if owner is not None:
      composite_desc = '("{}" attr of {})'.format(owner_attr, owner)
    else:
      composite_desc = ''

    logging.log(1,
                'Converted call: %s %s\n    args: %s\n    kwargs: %s\n',
                f, composite_desc, args, kwargs)

  if inspect_utils.isbuiltin(f):
    if kwargs:
      return py_builtins.overload_of(f)(*args, **kwargs)
    else:
      return py_builtins.overload_of(f)(*args)

  # TODO(b/122265385): Remove this bypass.
  if (_is_known_loaded_type(f, 'wrapt', 'FunctionWrapper') or
      _is_known_loaded_type(f, 'wrapt', 'BoundFunctionWrapper')):
    logging.warn(
        'Entity {} appears to be decorated by wrapt, which is not yet supported'
        ' by AutoGraph. The function will be called without transformation.'
        ' You may however apply AutoGraph before the decorator.'.format(f))
    logging.log(2, 'Permanently whitelisted: %s: wrapt decorated', f)
    return _call_unconverted(f, args, kwargs)

  if _is_known_loaded_type(f, 'functools', '_lru_cache_wrapper'):
    logging.log(2, 'Permanently whitelisted: %s: lru_cache', f)
    return _call_unconverted(f, args, kwargs)

  # Constructors are permanently whitelisted.
  # TODO(mdan): Toggle as experimental feature instead.
  # TODO(b/124016764): Remove this limitation.
  if tf_inspect.isclass(f):
    logging.log(2, 'Permanently whitelisted: %s: constructor', f)
    return _call_unconverted(f, args, kwargs)

  # Other built-in modules are permanently whitelisted.
  # TODO(mdan): Figure out how to do this consistently for all stdlib modules.
  if any(f in m.__dict__.values() for m in (collections, pdb, copy, inspect)):
    logging.log(2, 'Permanently whitelisted: %s: part of builtin module', f)
    return _call_unconverted(f, args, kwargs)

  if not options.force_conversion and conversion.is_whitelisted_for_graph(f):
    return _call_unconverted(f, args, kwargs)

  # internal_convert_user_code is for example turned off when issuing a dynamic
  # call conversion from generated code while in nonrecursive mode. In that
  # case we evidently don't want to recurse, but we still have to convert
  # things like builtins.
  if not options.internal_convert_user_code:
    return _call_unconverted(f, args, kwargs)

  # TODO(mdan): Move this entire block inside to_graph.
  try:  # Begin of transformation error guards

    # Unwrap functools.partial objects
    # TODO(mdan): Consider sharing unwrapping logic with tf_inspect.
    while isinstance(f, functools.partial):
      args = f.args + args
      new_kwargs = {}
      if f.keywords is not None:
        new_kwargs.update(f.keywords)
      if kwargs is not None:
        new_kwargs.update(kwargs)
      kwargs = new_kwargs
      f = f.func

    if tf_inspect.isfunction(f) or tf_inspect.ismethod(f):
      # Regular functions
      target_entity = f
      f_self = inspect_utils.getmethodself(f)

      # TODO(b/119246461): This may be more elegantly handled using __get__?
      if f_self is not None:
        effective_args = (f_self,) + args
      else:
        effective_args = args

    elif tf_inspect.isclass(f):
      # Constructors
      # Note: Until we support class constructurs, and enable whole-class
      # conversion with an experimental flag, this branch is dead code.
      # TODO(mdan): Consider removing unless there is a compelling use case.
      target_entity = f
      effective_args = args

    elif hasattr(f, '__call__') and hasattr(f, '__class__'):
      # Callable objects
      target_entity = f.__call__
      effective_args = (f,) + args

    else:
      target_entity = f
      raise NotImplementedError('unknown callable type "%s"' % type(f))

    if (not tf_inspect.isclass(target_entity) and
        not hasattr(target_entity, '__code__')):
      logging.log(
          2, 'Permanently whitelisted: %s: native binding', target_entity)
      return _call_unconverted(f, args, kwargs)

    converted_f = to_graph(
        target_entity,
        recursive=options.recursive,
        arg_values=None,
        arg_types=None,
        experimental_optional_features=options.optional_features)

    if logging.has_verbosity(2):
      logging.log(2, 'Defaults of %s : %s', converted_f,
                  converted_f.__defaults__)
      if kwargs is not None:
        callargs = tf_inspect.getcallargs(
            converted_f, *effective_args, **kwargs)
      else:
        callargs = tf_inspect.getcallargs(converted_f, *effective_args)
      formatted_callargs = '\n'.join(
          '    {}: {}'.format(k, v) for k, v in callargs.items())
      logging.log(2, 'Calling %s with\n%s\n', converted_f, formatted_callargs)

  # TODO(mdan): Reduce this list.
  except (errors.AutoGraphError, AssertionError, AttributeError, IndexError,
          KeyError, NameError, NotImplementedError, SyntaxError, TypeError,
          ValueError, IOError) as e:

    logging.log(1, 'Error transforming entity %s', target_entity, exc_info=True)

    if is_autograph_strict_conversion_mode():
      raise

    logging.warn(
        'Entity %s could not be transformed and will be executed as-is.'
        ' Some features (e.g. tensor-dependent conditionals and loops) may not'
        ' work as expected.'
        ' Error details can be found in the logs when running with the env'
        ' variable AUTOGRAPH_VERBOSITY >= 1. Please report this to the'
        ' AutoGraph team. Cause: %s', target_entity, e)

    return _call_unconverted(f, args, kwargs)

  if kwargs is not None:
    result = converted_f(*effective_args, **kwargs)
  else:
    result = converted_f(*effective_args)

  return result


@tf_export('autograph.to_graph')
def to_graph(entity,
             recursive=True,
             arg_values=None,
             arg_types=None,
             experimental_optional_features=None):
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
    arg_values: Optional dict of value hints for symbols including
      function arguments mapping string names to actual values. For example,
      `arg_values={'a': 1}` will map the variable `a` to the value `1`.
    arg_types: Optional dict of type hints for symbols including function
      arguments. Type hints allow specifying just the type of a variable, rather
      than a specific value.
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of
      optional features in the conversion process.

  Returns:
    Same as `entity`, the converted Python function or class.

  Raises:
    ValueError: If the entity could not be converted.
  """
  try:
    # TODO(b/129431421): Remove these args.
    del arg_values
    del arg_types
    program_ctx = converter.ProgramContext(
        options=converter.ConversionOptions(
            recursive=recursive,
            optional_features=experimental_optional_features),
        autograph_module=tf_inspect.getmodule(to_graph))
    return conversion.convert(entity, program_ctx)
  except (ValueError, AttributeError, KeyError, NameError, AssertionError) as e:
    errors.report_internal_error(entity, e)


# TODO(mdan): Remove deprecated indentation arg.
@tf_export('autograph.to_code')
def to_code(entity,
            recursive=True,
            arg_values=None,
            arg_types=None,
            indentation='  ',
            experimental_optional_features=None):
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
    experimental_optional_features: `None`, a tuple of, or a single
      `tf.autograph.experimental.Feature` value. Controls the use of
      optional features in the conversion process.

  Returns:
    The converted code as string.
  """
  # TODO(b/129431421): Remove this arg.
  del indentation
  source = tf_inspect.getsource(
      to_graph(
          entity,
          recursive=recursive,
          arg_values=arg_values,
          arg_types=arg_types,
          experimental_optional_features=experimental_optional_features))
  return textwrap.dedent(source)
