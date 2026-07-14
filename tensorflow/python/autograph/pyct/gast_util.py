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
"""Gast compatibility library. Supports 0.2.2 and 0.3.2."""
# TODO(mdan): Remove this file once it's safe to break compatibility.

import functools

import gast


GAST2 = hasattr(gast, 'Str')
GAST3 = not GAST2


def _is_constant_gast_2(node):
  return isinstance(node, (gast.Num, gast.Str, gast.Bytes, gast.Ellipsis,
                           gast.NameConstant))


def _is_constant_gast_3(node):
  return isinstance(node, gast.Constant)


def is_literal(node):
  """Tests whether node represents a Python literal."""
  # Normal literals, True/False/None/Etc. in Python3
  if is_constant(node):
    return True

  # True/False/None/Etc. in Python2
  if isinstance(node, gast.Name) and node.id in ['True', 'False', 'None']:
    return True

  return False


def _is_ellipsis_gast_2(node):
  return isinstance(node, gast.Ellipsis)


def _is_ellipsis_gast_3(node):
  return isinstance(node, gast.Constant) and node.value == Ellipsis


if GAST2:
  is_constant = _is_constant_gast_2
  is_ellipsis = _is_ellipsis_gast_2

  Module = gast.Module
  Name = gast.Name
  Str = gast.Str

elif GAST3:
  is_constant = _is_constant_gast_3
  is_ellipsis = _is_ellipsis_gast_3

  Module = functools.partial(gast.Module, type_ignores=None)  # pylint:disable=invalid-name
  Name = functools.partial(gast.Name, type_comment=None)  # pylint:disable=invalid-name
  Str = functools.partial(gast.Constant, kind=None)  # pylint:disable=invalid-name

else:
  assert False
