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

from functools import wraps

from enum import Enum

# pylint:disable=g-bad-import-order
import six
# pylint:enable=g-bad-import-order

from tensorflow.contrib.autograph.core import config
from tensorflow.contrib.autograph.core import converter
from tensorflow.contrib.autograph.impl import conversion
from tensorflow.contrib.autograph.operators import py_builtins
from tensorflow.contrib.autograph.pyct import compiler
from tensorflow.contrib.autograph.pyct import inspect_utils
from tensorflow.contrib.autograph.utils import py_func
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

# TODO(mdan): Properly document the type hints.
# TODO(mdan): Reduce the type hint information to (module, type).
# (currently we require (module + class name, type))


# TODO(mdan): This should behave like to_graph (e.g. convert statically).
def convert(recursive=False, verbose=False):
  """Decorator that compiles a function to use TensorFlow ops.

  The decorator is dynamic - it recompiles the target whenever the decorated
  function is called. This means the parameter values are known at conversion.
  It also means that repeated calls with different types of parameters will be
  correctly processed.

  Args:
    recursive: bool, whether to recursively convert any functions or classes
        that the converted function may use.
    verbose: bool, whether to output the compiled code in the logs.

  Returns:
    Callable, a decorator that converts the given function into an equivalent
    function that uses TensorFlow ops.
  """
  def decorator(f):
    """Decorator implementation."""

    @wraps(f)
    def wrapper(*args, **kwargs):
      return converted_call(f, recursive, verbose, True, {}, *args, **kwargs)

    wrapper = tf_decorator.make_decorator(f, wrapper)

    # Sometimes the decorator is just desugared, making it impossible to detect.
    # This attribute makes detection easier.
    setattr(wrapper, '__pyct_is_compile_decorator', True)
    return wrapper

  return decorator


class RunMode(Enum):
  """Specifies the way a converted function or method should be executed in TF.

  The enum values have the following semantics:

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
    return_dtypes: Optional[Iterable[
        Union[tf.DType, utils.py_func.MatchDType]]], the return data types of
        the converted function, if run_as is RunMode.PY_FUNC. Ignored otherwise.
        May be set to None if the function has no return values.

  Returns:
    Callable, a decorator that wraps the original function.
  """

  def decorator(f):
    """Decorator implementation."""

    @wraps(f)
    def graph_wrapper(*args, **kwargs):
      return f(*args, **kwargs)

    @wraps(f)
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
    setattr(wrapper, '__pyct_is_compile_decorator', True)
    return wrapper

  return decorator


# TODO(mdan): Move to a private, undocumented module.
def converted_call(f, recursive, verbose, force_conversion, arg_types, *args,
                   **kwargs):
  """Compiles a function call inline. For internal use only."""
  # TODO(mdan): This needs cleanup.
  # In particular, we may want to avoid renaming functions altogether.
  if not force_conversion and conversion.is_whitelisted_for_graph(f):
    return f(*args, **kwargs)

  unknown_arg_value = object()  # Sentinel for arguments of unknown value

  if inspect_utils.isbuiltin(f):
    return py_builtins.overload_of(f)(*args, **kwargs)

  if tf_inspect.isfunction(f) or tf_inspect.ismethod(f):
    # Regular functions
    target_entity = f
    arg_map_target = f
    effective_args = args
    f_class = inspect_utils.getmethodclass(f)

    if f_class is not None:
      partial_types = (f_class,)
    else:
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
  for name, arg in arg_values.items():
    if arg is unknown_arg_value:
      continue
    arg_class = arg.__class__
    # If arg_value_hints specifies any name, use that instead.
    if name not in arg_types:
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
      recursive=recursive,
      verbose=verbose,
      arg_values=arg_values,
      arg_types=arg_types,
      partial_types=partial_types)
  return converted_f(*effective_args, **kwargs)


# TODO(mdan): Rename: to_ops?
# TODO(mdan): Looki into overloading as function and decorator, like tfe.defun.
# TODO(mdan): Remove partial_types.
def to_graph(e,
             recursive=True,
             verbose=False,
             arg_values=None,
             arg_types=None,
             partial_types=None):
  """Converts a Python entity into equivalent code that uses TensorFlow ops.

  Supported Python entities include:
    * functions
    * classes

  Classes are converted by converting all their methods into a new class.

  Args:
    e: Union[Callable, Type], the Python entity to convert.
    recursive: bool, whether to recursively convert any functions that the
        converted function may call.
    verbose: bool, whether to output the compiled code in the logs.
    arg_values: Optional[Dict[Text, Any]], value hints for symbols including
        function arguments.
    arg_types: Optional[Dict[Text, Type]], type hints for symbols including
        function arguments.
    partial_types: Set[Type], reserved for internal use.

  Returns:
    Union[Callable, Type], the converted entity, which is the same kind as e
    (that is, a function is e is a function, a class if e is a class, etc.) but
    its code has been converted to use TF ops.

  Raises:
    ValueError: If the entity could not be converted.
  """
  program_ctx = converter.ProgramContext(
      recursive=recursive,
      autograph_decorators=(convert, do_not_convert, converted_call),
      partial_types=partial_types,
      autograph_module=tf_inspect.getmodule(to_graph),
      uncompiled_modules=config.DEFAULT_UNCOMPILED_MODULES)
  _, name, namespace = conversion.entity_to_graph(e, program_ctx, arg_values,
                                                  arg_types)

  nodes = []
  for dep in reversed(program_ctx.dependency_cache.values()):
    nodes.extend(dep)
  compiled_module, compiled_src = compiler.ast_to_object(
      nodes,
      source_prefix=program_ctx.required_imports,
      include_source_map=True)

  # The compiled code should see everything the entry entity saw.
  # TODO(mdan): This might not work well if the call tree spans modules?
  for key, val in namespace.items():
    # Avoid overwriting entities that have been transformed.
    if key not in compiled_module.__dict__:
      compiled_module.__dict__[key] = val
  compiled = getattr(compiled_module, name)

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

  if verbose:
    logging.info('Compiled output of %s:\n\n%s\n', e, compiled_src)

  return compiled


def to_code(e,
            recursive=True,
            arg_values=None,
            arg_types=None,
            partial_types=None,
            indentation='  '):
  """Returns the equivalent code that uses TensorFlow ops.

  Also see: `to_graph`, `convert`

  Args:
    e: Union[Callable, Type], the Python entity to convert.
    recursive: bool, whether to recursively convert any functions that the
        converted function may call.
    arg_values: Optional[Dict[Text, Any]], value hints for symbols including
        function arguments.
    arg_types: Optional[Dict[Text, Type]], type hints for symbols including
        function arguments.
    partial_types: Set[Type], reserved for internal use.
    indentation: Text, when to use for each level of indentation.

  Returns:
    Text, the converted code.
  """
  program_ctx = converter.ProgramContext(
      recursive=recursive,
      autograph_decorators=(convert, do_not_convert, converted_call),
      partial_types=partial_types,
      autograph_module=tf_inspect.getmodule(to_graph),
      uncompiled_modules=config.DEFAULT_UNCOMPILED_MODULES)
  conversion.entity_to_graph(e, program_ctx, arg_values, arg_types)

  code = '\n'.join(
      compiler.ast_to_source(dep, indentation)
      for dep in reversed(tuple(six.itervalues(program_ctx.dependency_cache))))

  return program_ctx.required_imports + '\n\n' + code
