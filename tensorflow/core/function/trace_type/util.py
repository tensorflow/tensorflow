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
"""Utilities for the trace_type module."""

from typing import Any, List, Tuple

import numpy as np


# TODO(b/225045380): Depend on the abstracted `leaf` lib from 'nest'.
def is_namedtuple(obj):
  return hasattr(obj, "_fields") and all(
      isinstance(field, str) for field in obj._fields)


# TODO(b/225045380): Depend on the abstracted `leaf` lib from 'nest'.
def is_attrs(obj):
  return hasattr(type(obj), "__attrs_attrs__")


# TODO(b/225045380): Depend on the abstracted `leaf` lib from 'nest'.
def attrs_from_values(attrs_type, values):
  """Creates an instance of the `attr.s` class `attrs_type` from `values`.

  Args:
    attrs_type: a class decorated with `attr.s`.
    values: one value per attribute, in the order of `__attrs_attrs__`.

  Returns:
    An instance of `attrs_type`. Attributes that are keyword-only are passed to
    `__init__` by name. Attributes with `init=False` are not accepted by
    `__init__`, so they are set on the new instance afterwards.
  """
  fields = attrs_type.__attrs_attrs__
  if all(
      getattr(f, "init", True) and not getattr(f, "kw_only", False)
      for f in fields
  ):
    return attrs_type(*values)
  # `alias` is the name `__init__` expects. Older versions of attrs do not
  # have it and strip leading underscores from the attribute name instead.
  init_kwargs = {
      getattr(f, "alias", None) or f.name.lstrip("_"): v
      for f, v in zip(fields, values)
      if getattr(f, "init", True)
  }
  instance = attrs_type(**init_kwargs)
  for f, v in zip(fields, values):
    if not getattr(f, "init", True):
      # Bypasses `__setattr__`, which frozen classes disable.
      object.__setattr__(instance, f.name, v)
  return instance


# TODO(b/225045380): Depend on the abstracted `leaf` lib from 'nest'.
def is_np_ndarray(value):
  return hasattr(value, "__array__") and not (
      # For legacy reasons we do not automatically promote Numpy strings.
      isinstance(value, np.str_)
      # NumPy dtypes have __array__ as unbound methods.
      or isinstance(value, type))


def cast_and_return_whether_casted(
    trace_types, values, context
) -> Tuple[List[Any], bool]:
  did_cast = False
  casted_values = []
  for t, v in zip(trace_types, values):
    casted_v = t.cast(v, context)
    casted_values.append(casted_v)
    if casted_v is not v:
      did_cast = True
  return casted_values, did_cast
