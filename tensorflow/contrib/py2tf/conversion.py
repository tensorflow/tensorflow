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
"""High level conversion support."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gast
import six

from tensorflow.contrib.py2tf import config
from tensorflow.contrib.py2tf import naming
from tensorflow.contrib.py2tf.convert import break_canonicalization
from tensorflow.contrib.py2tf.convert import builtin_functions
from tensorflow.contrib.py2tf.convert import call_trees
from tensorflow.contrib.py2tf.convert import continue_canonicalization
from tensorflow.contrib.py2tf.convert import control_flow
from tensorflow.contrib.py2tf.convert import for_canonicalization
from tensorflow.contrib.py2tf.convert import logical_expressions
from tensorflow.contrib.py2tf.convert import print_functions
from tensorflow.contrib.py2tf.convert import side_effect_guards
from tensorflow.contrib.py2tf.pyct import parser
from tensorflow.contrib.py2tf.pyct.static_analysis import access
from tensorflow.contrib.py2tf.pyct.static_analysis import live_values
from tensorflow.contrib.py2tf.pyct.static_analysis import type_info
from tensorflow.python.util import tf_inspect


class ConversionMap(object):
  """ConversionMaps keep track of converting function hierarchies.

  Attributes:
    dependency_cache: dict[object]: ast; maps original objects to their
        converted AST
    name_map: dict[string]: string; maps original objects to the name of
        their converted counterparts
  """

  def __init__(self):
    self.dependency_cache = {}
    self.name_map = {}

  def new_namer(self, global_symbols):
    return naming.Namer(global_symbols, self.name_map)

  def update_name_map(self, namer):
    for o, name in namer.renamed_calls.items():
      if o in self.name_map:
        if self.name_map[o] != name:
          raise ValueError(
              'Calls to %s were converted using multiple names (%s). This is '
              'possible when an object with one of these names already '
              'existed. To fix, avoid using any of these names.')
      else:
        self.name_map[o] = name

  def add_to_cache(self, original_object, converted_ast):
    self.dependency_cache[original_object] = converted_ast


def object_to_graph(o, conversion_map, value_hints):
  """Compile a Python object into equivalent TensorFlow.

  The function will also recursively compile all the objects that `o`
  references, updating `dependency_cache`.

  This function is reentrant, and relies on dependency_cache to avoid
  generating duplicate code.

  Args:
    o: A Python object.
    conversion_map: A ConversionMap object.
    value_hints: A dict containing value hints for symbols like function
        parameters.

  Returns:
    A tuple (ast, new_name):
        * ast: An AST representing an object with interface equivalent to `o`,
            but which when executed it creates TF a graph.
        * new_name: The symbol name under which the new object can be found.

  Raises:
    ValueError: if the object is not supported.
  """
  if value_hints is None:
    value_hints = {}

  if tf_inspect.isclass(o):
    node, new_name = class_to_graph(o, conversion_map, value_hints)
  elif tf_inspect.isfunction(o):
    node, new_name = function_to_graph(o, conversion_map, value_hints)
  else:
    raise ValueError(
        'Unsupported object type %s. Only functions and classes are supported'
        ' for now.')

  conversion_map.add_to_cache(o, node)
  # Recursively convert remaining dependencies.
  for obj in conversion_map.name_map.keys():
    if obj not in conversion_map.dependency_cache:
      if hasattr(obj, 'im_class'):
        # Class members are converted with their objects.
        continue
      object_to_graph(obj, conversion_map, None)

  return node, new_name


def class_to_graph(c, conversion_map, param_value_hints):
  """Specialization of `object_to_graph` for classes."""
  converted_members = {}
  members = tf_inspect.getmembers(c, predicate=tf_inspect.ismethod)
  if not members:
    raise ValueError('Cannot convert %s: it has no member methods.')

  if 'self' in param_value_hints:
    raise ValueError('Hints may not be provided for reserved name "self".')
  param_value_hints['self'] = (c.__name__, c)

  class_globals = None
  for _, m in members:
    node, _ = function_to_graph(m, conversion_map, param_value_hints, c)
    # TODO(mdan): Do not assume all members have the same view of globals.
    if class_globals is None:
      class_globals = six.get_function_globals(m)
    converted_members[m] = node
  namer = conversion_map.new_namer(class_globals)
  class_name = namer.compiled_class_name(c.__name__, c)
  node = gast.ClassDef(
      class_name,
      bases=[],
      keywords=[],
      body=converted_members.values(),
      decorator_list=[])

  return node, class_name


def function_to_graph(f, conversion_map, param_value_hints, owner_type=None):
  """Specialization of `object_to_graph` for callable functions."""
  node = parser.parse_object(f).body[0]
  node_globals = six.get_function_globals(f)

  # This is needed for non-global functions.
  closure = six.get_function_closure(f)
  if closure:
    for e in closure:
      if callable(e.cell_contents):
        fn = e.cell_contents
        node_globals[fn.__name__] = fn

  namer = conversion_map.new_namer(node_globals)
  node = node_to_graph(node, namer, node_globals, param_value_hints)

  # Simulate a rename to ensure the top level is in the name map. This is needed
  # for top level functions, and it also helps the consistency verification made
  # by update_name_map.
  if owner_type is not None:
    new_name = namer.compiled_function_name(f.__name__, f, owner_type)
  else:
    new_name = namer.compiled_function_name(f.__name__, f)
  node.name = new_name
  conversion_map.update_name_map(namer)
  return node, conversion_map.name_map[f]


def _static_analysis_pass(node, namespace, value_hints):
  node = access.resolve(node)
  node = live_values.resolve(node, namespace, config.PYTHON_LITERALS)
  node = type_info.resolve(node, value_hints)
  return node


def node_to_graph(node, namer, namespace, value_hints):
  """Convert Python code to equivalent TF graph mode code.

  Args:
    node: A Python AST node representing the code to convert.
    namer: A naming.Namer object.
    namespace: Dict mapping symbol names to their corresponding live objects.
    value_hints: A dict containing value hints for symbols like function
        parameters.

  Returns:
    A tuple (node, deps):
        * node: A Python ast node, representing the converted code.
        * deps: A set of strings, the fully qualified names of object
            dependencies that this node has.
  """
  # TODO(mdan): Factor out common elements.
  # These include:
  #   * keeping track of symbols that have been created
  #   * marking nodes (e.g. py_func wrappers) to suppress further processing
  #   * code move between blocks
  #   * insertion of new global references
  #   * visiting blocks in transformers

  # Certain steps, especially canonicalization, insert new symbols into the
  # tree, which must be accounted. Although less efficient, it is most robust
  # to re-run the analysis.

  node = _static_analysis_pass(node, namespace, value_hints)
  node = break_canonicalization.transform(node, namer)

  # Note: sequencing continue canonicalization before for loop one avoids
  # dealing with the extra loop increment operation that the for
  # canonicalization creates.
  node = continue_canonicalization.transform(node, namer)
  namespace['len'] = len

  node = _static_analysis_pass(node, namespace, value_hints)
  node = for_canonicalization.transform(node, namer)
  # for_canonicalization may insert new global references.
  node = builtin_functions.transform(node)
  # builtin_functions may insert new global references.
  namespace['print'] = print

  node = _static_analysis_pass(node, namespace, value_hints)
  node = print_functions.transform(node)
  node = call_trees.transform(node, namer, config.DEFAULT_UNCOMPILED_MODULES)
  node = control_flow.transform(node, namer)
  node = logical_expressions.transform(node)
  node = side_effect_guards.transform(node, namer)

  return node
