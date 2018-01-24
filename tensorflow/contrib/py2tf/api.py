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

from tensorflow.contrib.py2tf import config
from tensorflow.contrib.py2tf import conversion
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import parser
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


def convert(recursive=False, arg_value_hints=None):
  """Decorator that compiles a function to graph mode.

  The decorator is dynamic - invoking compilation whenever the decorated function
  is called. This means the parameter values are known at compilation.

  Args:
    recursive: Whether to recusrively convert any functions that the decorator
        function may call.
    arg_value_hints: A dict mapping parameter names to objects that can hint
        at the type of those parameters.

  Returns:
    A decorator that compiles the given function to graph mode.

  Raises:
    ValueError: If any of the arguments are illegal.
  """
  if arg_value_hints is None:
    arg_value_hints = {}

  def decorator(f):
    """Decorator implementation."""

    @wraps(f)
    def wrapper(*args, **kwargs):
      """Wrapper that calls the compiled version of the wrapped function."""
      partial_types = ()
      arg_names = tf_inspect.getargspec(f)[0]
      for name, arg in zip(arg_names, args):
        arg_class = arg.__class__
        if tf_inspect.isclass(arg_class):
          # If arg_value_hints specifies any name, use that instead.
          # TODO(mdan): Shouldn't this just be in the func's globals?
          if name not in arg_value_hints:
            arg_value_hints[name] = (arg_class.__name__, arg_class)
          # Annotated methods need to specify that their owner type is partial,
          # otherwise other members they call will not be converted.
          if name == 'self':
            partial_types = (arg_class,)
      wrapped = to_graph(
          f,
          recursive=recursive,
          arg_value_hints=arg_value_hints,
          partial_types=partial_types)
      return wrapped(*args, **kwargs)

    # Sometimes the decorator is just desugared, making it impossible to detect.
    # This attribute makes detection easier.
    setattr(wrapper, '__pyct_is_compile_decorator', True)
    return wrapper

  return decorator


def to_graph(o, recursive=True, arg_value_hints=None, partial_types=None):
  """Compile a Python entity into equivalent TensorFlow code.

  Currently supported entities:
    * functions
    * classes

  Classes are handled by converting all their methods into a new class.

  Args:
    o: A Python function or class.
    recursive: Whether to recusrively convert any functions that the decorator
        function may call.
    arg_value_hints: A dict mapping parameter names to objects that can hint
        at the type of those parameters.
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
      partial_types=partial_types)
  _, name = conversion.object_to_graph(o, conversion_map, arg_value_hints)

  module = gast.Module([])
  for import_line in config.COMPILED_IMPORT_STATEMENTS:
    module.body.append(parser.parse_str(import_line))
  for dep in conversion_map.dependency_cache.values():
    module.body.append(dep)
  compiled_node = compiler.ast_to_object(module)

  # The compiled code should see everything the entry function saw.
  # TODO(mdan): This might not work well if the call tree spans modules?
  if tf_inspect.isfunction(o):
    compiled_node.__dict__.update(six.get_function_globals(o))

  compiled_fn = getattr(compiled_node, name)
  return compiled_fn


def to_code(o,
            recursive=True,
            arg_value_hints=None,
            partial_types=None,
            indentation='  '):
  """Return the equivalent of an entity in TensorFlow code.

  See `to_graph` for more details.

  Args:
    o: A Python function or class.
    recursive: Whether to recusrively convert any functions that the decorator
        function may call.
    arg_value_hints: A dict mapping parameter names to objects that can hint
        at the type of those parameters.
    partial_types: A set of types (e.g. classes) that will not be converted
        entirely. Calls to member functions for these types will be renamed
        independently.
    indentation: String, when to use for each level of indentation.

  Returns:
    String.
  """
  conversion_map = conversion.ConversionMap(
      recursive=recursive,
      nocompile_decorators=(convert, graph_ready, convert_inline),
      partial_types=partial_types)
  conversion.object_to_graph(o, conversion_map, arg_value_hints)

  imports = '\n'.join(config.COMPILED_IMPORT_STATEMENTS)
  code = '\n'.join(
      compiler.ast_to_source(dep, indentation)
      for dep in reversed(tuple(
          six.itervalues(conversion_map.dependency_cache))))

  return imports + '\n\n' + code
