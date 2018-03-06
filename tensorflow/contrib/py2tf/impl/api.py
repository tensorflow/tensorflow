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
"""Public API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from functools import wraps

import gast
import six

from tensorflow.contrib.py2tf.impl import config
from tensorflow.contrib.py2tf.impl import conversion
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import inspect_utils
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.utils import builtins
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import tf_inspect

# TODO(mdan): Properly document the type hints.
# TODO(mdan): Reduce the type hint information to (module, type).
# (currently we require (module + class name, type))


def graph_ready(f):
  """No-op decorator that explicitly marks a function as graph-ready.

  Graph-ready functions are assumed to not need any conversion.

  Args:
    f: Any callable.
  Returns:
    f itself.
  """
  setattr(f, '__pyct_is_compile_decorator', True)
  return f


def convert_inline(f, *args, **kwargs):
  """Shorthand to convert and call a function.

  For example, the following two statements are equivalent:

      @convert()
      def foo():
        ...
      foo(bar)

      def foo():
        ...
      convert_inline(foo, bar)

  Args:
    f: Function to convert. Only this call will be converted.
    *args: Passed through to f.
    **kwargs: Passed through to f, with the following exceptions:
        * arg_value_hints: A dict mapping parameter names to objects that can
            hint at the type of those parameters.

  Returns:
    The result of the converted f applied to args and kwargs.
  """
  if 'arg_value_hints' in kwargs:
    arg_value_hints = kwargs['arg_value_hints']
    del kwargs['arg_value_hints']
  else:
    arg_value_hints = None
  if tf_inspect.ismethod(f):
    # When converting methods, the result is still an unbound function.
    args = (f.__self__,) + args
  return convert(arg_value_hints)(f)(*args, **kwargs)


def convert(recursive=False, verbose=False, arg_types=None):
  """Decorator that compiles a function to graph mode.

  The decorator is dynamic - invoking compilation whenever the decorated
  function is called. This means the parameter values are known at compilation.

  Args:
    recursive: Whether to recusrively convert any functions that the decorator
        function may call.
    verbose: Whether to output the compiled code in the logs.
    arg_types: See to_graph.

  Returns:
    A decorator that compiles the given function to graph mode.

  Raises:
    ValueError: If any of the arguments are illegal.
  """
  if arg_types is None:
    arg_types = {}

  def decorator(f):
    """Decorator implementation."""

    @wraps(f)
    def wrapper(*args, **kwargs):
      return converted_call(f, recursive, verbose, arg_types, *args, **kwargs)

    # Sometimes the decorator is just desugared, making it impossible to detect.
    # This attribute makes detection easier.
    setattr(wrapper, '__pyct_is_compile_decorator', True)
    return wrapper

  return decorator


def converted_call(f, recursive, verbose, arg_types, *args, **kwargs):
  """Compiles a function call inline."""
  # TODO(mdan): This needs cleanup.
  # In particular, we may want to avoid renaming functions altogether.

  if conversion.is_whitelisted_for_graph(f):
    return f(*args, **kwargs)

  unknown_arg_value = object()  # Sentinel for arguments of unknown value

  if tf_inspect.isbuiltin(f):
    return builtins.dynamic_builtin(f, *args, **kwargs)

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
    effective_args = (unknown_arg_value,) + args
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


def to_graph(e,
             recursive=True,
             verbose=False,
             arg_values=None,
             arg_types=None,
             partial_types=None):
  """Compile a Python entity into equivalent TensorFlow code.

  Currently supported entities:
    * functions
    * classes

  Classes are handled by converting all their methods into a new class.

  Args:
    e: A Python entity.
    recursive: Whether to recusrively convert any functions that the decorator
        function may call.
    verbose: Whether to output the compiled code in the logs.
    arg_values: A dict containing value hints for symbols like function
        parameters.
    arg_types: A dict containing type hints for symbols like function
        parameters.
    partial_types: A set of types (e.g. classes) that will not be converted
        entirely. Calls to member functions for these types will be renamed
        independently.

  Returns:
    A function with a signature identical to `o`, but which when executed it
  creates TF a graph that has the same functionality as the original entity.
  """
  conversion_map = conversion.ConversionMap(
      recursive=recursive,
      nocompile_decorators=(convert, graph_ready, convert_inline),
      partial_types=partial_types,
      api_module=tf_inspect.getmodule(to_graph))
  _, name = conversion.entity_to_graph(e, conversion_map, arg_values, arg_types)

  module = gast.Module([])
  for import_line in config.COMPILED_IMPORT_STATEMENTS:
    module.body.append(parser.parse_str(import_line))
  for dep in conversion_map.dependency_cache.values():
    module.body.append(dep)
  compiled_node, compiled_src = compiler.ast_to_object(module)

  # The compiled code should see everything the entry function saw.
  # TODO(mdan): This might not work well if the call tree spans modules?
  if tf_inspect.isfunction(e):
    compiled_node.__dict__.update(inspect_utils.getnamespace(e))
  compiled_fn = getattr(compiled_node, name)

  if verbose:
    logging.info('Compiled output of %s:\n\n%s\n', e, compiled_src)

  return compiled_fn


def to_code(e,
            recursive=True,
            arg_values=None,
            arg_types=None,
            partial_types=None,
            indentation='  '):
  """Return the equivalent of an entity in TensorFlow code.

  See `to_graph` for more details.

  Args:
    e: A Python entity.
    recursive: See to_graph.
    arg_values: See to_graph.
    arg_types: See to_graph.
    partial_types: See to_graph.
    indentation: String, when to use for each level of indentation.

  Returns:
    String.
  """
  conversion_map = conversion.ConversionMap(
      recursive=recursive,
      nocompile_decorators=(convert, graph_ready, convert_inline),
      partial_types=partial_types,
      api_module=tf_inspect.getmodule(to_graph))
  conversion.entity_to_graph(e, conversion_map, arg_values, arg_types)

  imports = '\n'.join(config.COMPILED_IMPORT_STATEMENTS)
  code = '\n'.join(
      compiler.ast_to_source(dep, indentation)
      for dep in reversed(tuple(
          six.itervalues(conversion_map.dependency_cache))))

  return imports + '\n\n' + code
