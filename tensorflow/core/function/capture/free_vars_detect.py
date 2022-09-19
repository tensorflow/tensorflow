# Copyright 2022 The TensorFlow Authors. All Rights Reserved.
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
"""An independent module to detect free vars inside a function."""

import types
from typing import List

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity
from tensorflow.python.util import tf_inspect


def _parse_and_analyze(func):
  """Parse and analyze Python Function code."""
  node, source = parser.parse_entity(func, future_features=())
  node = qual_names.resolve(node)
  entity_info = transformer.EntityInfo(
      name=func.__name__,
      source_code=source,
      source_file=None,
      future_features=(),
      namespace={})
  namer = naming.Namer({})
  ctx = transformer.Context(entity_info, namer, None)
  node = activity.resolve(node, ctx)
  return node


def detect_function_free_vars(func: types.FunctionType) -> List[str]:
  """Detect free vars in any Python function."""
  assert isinstance(
      func, types.FunctionType
  ), f"The input should be of Python function type. Got type: {type(func)}."

  node = _parse_and_analyze(func)
  scope = anno.getanno(node, anno.Static.SCOPE)
  free_vars_all = list(scope.free_vars)
  globals_dict = func.__globals__
  filtered = []
  for var in free_vars_all:
    base = str(var.qn[0])
    if base in globals_dict:
      obj = globals_dict[base]
      if tf_inspect.ismodule(obj):
        continue
      if (tf_inspect.isclass(obj) or
          tf_inspect.ismethod(obj) or
          tf_inspect.isfunction(obj)):
        if obj.__module__ != func.__module__:
          continue
      # Only keep free vars without subscript for simplicity
      if not var.has_subscript():
        filtered.append(str(var))
    else:
      if not var.has_subscript():
        filtered.append(str(var))

  return sorted(filtered)

