# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
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
"""Converting AST to code.

Adapted from Tangent.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(mdan): Use six for compatibility here.
import atexit
import imp
import os
import tempfile

import astor
import gast

from tensorflow.contrib.autograph.pyct import anno
from tensorflow.contrib.autograph.pyct import ast_util
from tensorflow.contrib.autograph.pyct import origin_info
from tensorflow.contrib.autograph.pyct import parser


def _build_source_map(node, code):
  """Return the Python objects represented by given AST.

  Compiling the AST code this way ensures that the source code is readable by
  e.g. `pdb` or `inspect`.

  Args:
    node: An AST node of the original generated code, before the source code is
      generated.
    code: The string representation of the source code for the newly generated
      code.

  Returns:
    Dict[CodeLocation, OriginInfo], a mapping between the user and AutoGraph
    generated code.
  """
  # After we have the final generated code we reparse it to get the final line
  # numbers. Then we walk through the generated and original ASTs in parallel
  # to build the mapping between the user and generated code.
  new_node = parser.parse_str(code)
  origin_info.resolve(new_node, code)
  source_mapping = {}
  for before, after in ast_util.parallel_walk(node, new_node):
    # Need both checks because if origin information is ever copied over to new
    # nodes then we need to rely on the fact that only the original user code
    # has the origin annotation.
    if (anno.hasanno(before, anno.Basic.ORIGIN) and
        anno.hasanno(after, anno.Basic.ORIGIN)):
      source_info = anno.getanno(before, anno.Basic.ORIGIN)
      new_line_number = anno.getanno(after, anno.Basic.ORIGIN).line_number
      source_mapping[new_line_number] = source_info
  return source_mapping


def ast_to_source(node, indentation='  '):
  """Return the source code of given AST."""
  original_node = node
  if isinstance(node, gast.AST):
    node = gast.gast_to_ast(node)
  generator = astor.codegen.SourceGenerator(indentation, False,
                                            astor.string_repr.pretty_string)
  generator.visit(node)
  generator.result.append('\n')
  # In some versions of Python, literals may appear as actual values. This
  # ensures everything is string.
  code = map(str, generator.result)
  code = astor.source_repr.pretty_source(code).lstrip()
  source_mapping = _build_source_map(original_node, code)

  return code, source_mapping


def ast_to_object(node,
                  indentation='  ',
                  source_prefix=None,
                  delete_on_exit=False):
  """Return the Python objects represented by given AST.

  Compiling the AST code this way ensures that the source code is readable by
  e.g. `pdb` or `inspect`.

  Args:
    node: The code to compile, as an AST object.
    indentation: The string to use for indentation.
    source_prefix: Optional string to print as-is into the source file.
    delete_on_exit: Whether to delete the temporary file used for compilation on
      exit.

  Returns:
    A module object containing the compiled source code.
  Raises:
    ValueError: If ag_source_map__ is already in the namespace of the compiled
    node.
  """
  # code_source_mapping does not yet include the offsets from import statements.
  source, code_source_mapping = ast_to_source(node, indentation=indentation)

  with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    # TODO(znado): move into an _offset_source_map() helper function.
    # Need to offset the generated line numbers by the number of import lines.
    if source_prefix:
      num_import_lines = source_prefix.count('\n') + 1
    else:
      num_import_lines = 0
    source_mapping = {}
    for line_number, original_position in code_source_mapping.items():
      source_map_key = origin_info.CodeLocation(
          file_path=f.name, line_number=line_number + num_import_lines)
      source_mapping[source_map_key] = original_position
    module_name = os.path.basename(f.name[:-3])
    if source_prefix:
      f.write(source_prefix)
      f.write('\n')
    f.write(source)
  if delete_on_exit:
    atexit.register(lambda: os.remove(f.name))
  compiled_node = imp.load_source(module_name, f.name)

  # TODO(znado): Clean this up so we don't need to attach it to the namespace.
  # TODO(znado): This does not work for classes because their methods share a
  # namespace.
  # This attaches the source map which is needed for error handling.  Note that
  # api.to_graph copies this source map into an attribute of the function.
  #
  # We need this so the ag_source_map__ variable is available to the call to
  # rewrite_graph_construction_error in the except block inside each function
  # that handles graph construction errors.
  #
  # We cannot get the rewritten function name until it is too late so templating
  # is hard, and this cleanly fixes the
  # issues encountered with nested functions because this is attached to the
  # outermost one.
  source_map_name = 'ag_source_map__'
  if source_map_name in compiled_node.__dict__:
    raise ValueError('cannot convert %s because is has namespace attribute '
                     '"%s", which is reserved for AutoGraph.' %
                     (compiled_node, source_map_name))
  compiled_node.__dict__[source_map_name] = source_mapping

  return compiled_node, source
