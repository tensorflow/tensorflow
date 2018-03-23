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


def ast_to_source(node, indentation='  '):
  """Return the source code of given AST."""
  if isinstance(node, gast.AST):
    node = gast.gast_to_ast(node)
  generator = astor.codegen.SourceGenerator(indentation, False,
                                            astor.string_repr.pretty_string)
  generator.visit(node)
  generator.result.append('\n')
  # In some versions of Python, literals may appear as actual values. This
  # ensures everything is string.
  code = map(str, generator.result)
  return astor.source_repr.pretty_source(code).lstrip()


def ast_to_object(
    node, indentation='  ', source_prefix=None, delete_on_exit=True):
  """Return the Python objects represented by given AST.

  Compiling the AST code this way ensures that the source code is readable by
  e.g. `pdb` or `inspect`.

  Args:
    node: The code to compile, as an AST object.
    indentation: The string to use for indentation.
    source_prefix: Optional string to print as-is into the source file.
    delete_on_exit: Whether to delete the temporary file used for compilation
        on exit.

  Returns:
    A module object containing the compiled source code.
  """
  source = ast_to_source(node, indentation)

  with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
    module_name = os.path.basename(f.name[:-3])
    if source_prefix:
      f.write(source_prefix)
      f.write('\n')
    f.write(source)
  if delete_on_exit:
    atexit.register(lambda: os.remove(f.name))
  return imp.load_source(module_name, f.name), source
