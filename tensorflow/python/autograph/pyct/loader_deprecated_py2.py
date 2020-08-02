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
"""Converting AST to code and Python entities.

Python 2 compatibility version. Not maintained.

Adapted from Tangent.
"""

# TODO(mdan): Consolidate with parser and rename to parsing.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# TODO(mdan): Use six for compatibility here.
import atexit
import imp
import os
import tempfile

import six

from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.autograph.pyct import parser


def load_source(source, delete_on_exit):
  """Loads the given source code as a Python module."""
  if six.PY2:
    source = source.encode('utf-8')
    f = tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False)
  else:
    f = tempfile.NamedTemporaryFile(  # pylint:disable=unexpected-keyword-arg
        mode='w', suffix='.py', delete=False, encoding='utf-8')

  with f:
    module_name = os.path.basename(f.name[:-3])
    f.write(source)

  if delete_on_exit:
    atexit.register(lambda: os.remove(f.name))
  return imp.load_source(module_name, f.name), f.name


def load_ast(nodes,
             indentation='  ',
             include_source_map=False,
             delete_on_exit=True):
  """Loads the given AST as a Python module.

  Compiling the AST code this way ensures that the source code is readable by
  e.g. `pdb` or `inspect`.

  Args:
    nodes: Union[ast.AST, Iterable[ast.AST]], the code to compile, as an AST
      object.
    indentation: Text, the string to use for indentation.
    include_source_map: bool, whether return a source map.
    delete_on_exit: bool, whether to delete the temporary file used for
      compilation on exit.

  Returns:
    Tuple[module, Text, Dict[LineLocation, OriginInfo]], containing:
    the module containing the unparsed nodes, the source code corresponding to
    nodes, and the source map. Is include_source_map is False, the source map
    will be None.
  """
  if not isinstance(nodes, (list, tuple)):
    nodes = (nodes,)

  source = parser.unparse(nodes, indentation=indentation)
  module, _ = load_source(source, delete_on_exit)

  if include_source_map:
    source_map = origin_info.create_source_map(nodes, source, module.__file__)
  else:
    source_map = None

  # TODO(mdan): Return a structured object.
  return module, source, source_map
