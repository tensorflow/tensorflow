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

import builtins
import collections
import functools
import inspect
import types

from tensorflow.python.autograph.pyct import anno
from tensorflow.python.autograph.pyct import inspect_utils
from tensorflow.python.autograph.pyct import naming
from tensorflow.python.autograph.pyct import parser
from tensorflow.python.autograph.pyct import qual_names
from tensorflow.python.autograph.pyct import transformer
from tensorflow.python.autograph.pyct.static_analysis import activity

FreeVar = collections.namedtuple("FreeVar", ["name", "is_function", "obj"])


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


def _search_callable_free_vars(fn):
  """Search free vars from a callable object."""
  node = _parse_and_analyze(fn)
  scope = anno.getanno(node, anno.Static.SCOPE)
  free_vars_all = list(scope.free_vars)
  namespace = inspect_utils.getnamespace(fn)
  filtered = []

  for var in free_vars_all:
    base = str(var.qn[0])

    if var.is_simple():
      if base in builtins.__dict__.keys():
        continue
      obj = namespace[base]
    else:
      assert var.is_composite()
      # A compositve qualified name `QN` can be either an attr or a subscript
      if var.has_subscript():
        # For free var with subscripts, both the base and full formats are
        # generated.
        # For example, if the code have `glob[idx]`, `free_vars_all` would
        # contain `glob` as well as `glob[idx]`.
        # The method only keeps the base format for simplicity.
        continue
      else:
        assert var.has_attr()
        # For free vars with multiple attributes like `f.g.h`,
        # just as the subscripts, multiple free vars (QN) are generated:
        # ['f', 'f.g', 'f.g.h']
        # If `f` is `self`, only process the first attribute `f.g`.
        # Otherwise, only process `f`.
        if not var.qn[0].is_composite() and base == "self":
          attr = str(var.qn[1])
          obj = getattr(fn.__self__, attr)
        else:
          continue

    if (inspect.ismodule(obj) or inspect.isclass(obj)):
      continue
    elif inspect.isfunction(obj) or inspect.ismethod(obj):
      while hasattr(fn, "__wrapped__"):
        obj = obj.__wrapped__
      if obj.__module__ != fn.__module__:
        continue
      filtered.append(FreeVar(str(var), True, obj))
    else:
      filtered.append(FreeVar(str(var), False, None))

  filtered = sorted(filtered, key=lambda x: x.name)
  return filtered


def _make_lambda_name(obj):
  source = inspect.getsource(obj)
  name = source.split("=")[0].strip()
  return name


def _make_callable_signature(obj):
  """Generate signature for function/method."""
  if inspect.isclass(obj) or inspect.isfunction(obj):
    if obj.__name__ == "<lambda>":
      return _make_lambda_name(obj)
    return obj.__name__
  elif inspect.ismethod(obj):
    obj_self = obj.__self__
    if isinstance(obj_self, type):
      cls_name = obj_self.__name__
    else:
      cls_name = obj_self.__class__.__name__
    return f"{cls_name}.{obj.__name__}"
  else:
    raise TypeError(
        f"Only class/function/methods are valid inputs, got {type(obj)}")


def _detect_function_free_vars(fn):
  """Detect free vars in any Python function."""
  assert isinstance(fn, types.FunctionType) or isinstance(
      fn, types.MethodType
  ), f"The input should be of Python function type. Got type: {type(fn)}."

  while hasattr(fn, "__wrapped__"):
    fn = fn.__wrapped__

  queue = collections.deque([fn])
  fn_map = dict()

  # Perform BFS over functions to get free vars
  while queue:
    obj = queue.popleft()
    signature = _make_callable_signature(obj)
    if signature not in fn_map:
      free_vars = _search_callable_free_vars(obj)
      if not free_vars:
        continue
      fn_map[signature] = free_vars
      for var in free_vars:
        # Only search callable free vars
        if var.is_function:
          obj = var.obj
          if _make_callable_signature(obj) not in fn_map:
            queue.append(obj)

  # func_name -> namedtupe FreeVar
  return fn_map


def generate_logging(fn, fn_threshold=5, var_threshold=10):
  """Generate loggings of free vars from fn."""
  if fn is None:
    return None
  assert isinstance(fn, types.FunctionType) or isinstance(
      fn, types.MethodType
  ) or isinstance(fn, functools.partial) or isinstance(
      fn, functools.partialmethod
  ), f"The input should be of Python function/method type. Got type: {type(fn)}."

  while hasattr(fn, "__wrapped__"):
    fn = fn.__wrapped__

  if isinstance(fn, functools.partial) or isinstance(fn,
                                                     functools.partialmethod):
    fn = fn.func

  fn_vars_map = _detect_function_free_vars(fn)
  # If not free vars detected, return None
  if not fn_vars_map:
    return None

  logging_txt = []
  tf_fn_name = _make_callable_signature(fn)
  tf_fn_module = fn.__module__

  def one_line_logging(fn_name, free_vars, threshold=10):
    if not free_vars:
      return ""
    log = f"Inside function {fn_name}(): "
    log += ", ".join([var.name for var in free_vars[:threshold]])
    if len(free_vars) > threshold:
      log += "..."
    return log

  # Show the free vars info of the tf.function at the top
  fn_threshold -= 1
  tf_fn_line = one_line_logging(tf_fn_name, fn_vars_map[tf_fn_name],
                                var_threshold)

  # Functions that are defined outside of tf.function
  outside_fn_lines = []
  outside_fn_names = [name for name in fn_vars_map.keys() if name != tf_fn_name]
  outside_fn_names = sorted(outside_fn_names)
  for fn_name in outside_fn_names[:fn_threshold]:
    outside_fn_lines.append(
        one_line_logging(fn_name, fn_vars_map[fn_name], var_threshold))

  if len(fn_vars_map) > fn_threshold:
    ellipsis_line = "..."
  else:
    ellipsis_line = None

  # TODO(panzf): direct users to the manual API after it's exposed to public
  explanation_line = (
      f"Free variables are detected within tf.function {tf_fn_name}() in"
      f"{tf_fn_module}. Free variable usage may cause inconsistant behaviors"
      "between eager mode and tf.function. Please consider refactor the code"
      "if possible. More details are avaiable in"
      "https://www.tensorflow.org/guide/function#limitations.\n"
      "Free variable names inside each function/method are shown below:")

  logging_txt = [explanation_line, tf_fn_line] + outside_fn_lines
  if ellipsis_line:
    logging_txt.append(ellipsis_line)

  return "\n".join(logging_txt)
