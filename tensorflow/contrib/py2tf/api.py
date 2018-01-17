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

import gast
import six

from tensorflow.contrib.py2tf import config
from tensorflow.contrib.py2tf import conversion
from tensorflow.contrib.py2tf.pyct import compiler
from tensorflow.contrib.py2tf.pyct import parser


def to_graph(f, arg_value_hints=None):
  """Compile a Python function into equivalent TensorFlow code.

  Args:
    f: A Python function with arbitrary arguments and return values.
    arg_value_hints: A dict mapping parameter names to objects that can hint
        at the type of those parameters.

  Returns:
    A function with a signature identical to `f`, but which when executed it
  creates TF a graph that has the same functionality as the original function.
  """
  conversion_map = conversion.ConversionMap()
  _, name = conversion.object_to_graph(f, conversion_map, arg_value_hints)

  module = gast.Module([])
  for import_line in config.COMPILED_IMPORT_STATEMENTS:
    module.body.append(parser.parse_str(import_line))
  for dep in conversion_map.dependency_cache.values():
    module.body.append(dep)
  compiled_node = compiler.ast_to_object(module)

  # The compiled code should see everything the entry function saw.
  # TODO(mdan): This might not work well if the call tree spans modules?
  compiled_node.__dict__.update(six.get_function_globals(f))

  compiled_fn = getattr(compiled_node, name)
  return compiled_fn


def to_code(f, arg_value_hints=None, indentation='  '):
  """Return the equivalent of a function in TensorFlow code.

  Args:
    f: A Python function with arbitrary arguments and return values.
    arg_value_hints: A dict mapping parameter names to objects that can hint
        at the type of those parameters.
    indentation: String, when to use for each level of indentation.

  Returns:
    String.
  """
  conversion_map = conversion.ConversionMap()
  conversion.object_to_graph(f, conversion_map, arg_value_hints)

  imports = '\n'.join(config.COMPILED_IMPORT_STATEMENTS)
  code = '\n'.join(
      compiler.ast_to_source(dep, indentation)
      for dep in reversed(tuple(
          six.itervalues(conversion_map.dependency_cache))))

  return imports + '\n\n' + code
