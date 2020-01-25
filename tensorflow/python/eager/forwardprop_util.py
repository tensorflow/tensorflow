# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
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
"""Utilities for managing forward accumulators.

A separate file from forwardprop.py so that functions can use these utilities.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib

from tensorflow.python import pywrap_tfe


class TangentInfo(
    collections.namedtuple("TangentInfo", ["indices", "tangents"])):
  """Packed forward accumulator state. The return value of `pack_tangents`."""

  def __new__(cls, indices=None, tangents=None):
    if indices is None:
      indices = ()
    if tangents is None:
      tangents = []
    return super(TangentInfo, cls).__new__(cls, indices, tangents)


def pack_tangents(tensors):
  """Packs forward accumulator state into a TangentInfo tuple.

  Args:
    tensors: A flat list of Tensors to pack forward accumulator state for.

  Returns:
    A tuple of (indices, tangents):
      indices: A sequence of sequences of two-element tuples. Each forward
        accumulator is represented as a sequence of tuples with (primal_index,
        jvp_index). Both integers index into the concatenated `tensors + jvps`
        array.
      tangents: A flat list of Tensors. Best interpreted as a sequence to be
        appended to `tensors`.
  """
  return TangentInfo(*pywrap_tfe.TFE_Py_PackJVPs(tensors))


@contextlib.contextmanager
def push_forwardprop_state():
  """Temporarily push or pop transient state for accumulators in the active set.

  Allows an accumulator which is currently processing an operation to
  temporarily reset its state. This is useful when building forwardprop versions
  of functions, where an accumulator will trigger function building and then
  must process captured symbolic tensors while building it. Without pushing and
  poping, accumulators ignore operations executed as a direct result of their
  own jvp computations.

  Yields:
    None (used for its side effect).
  """
  try:
    pywrap_tfe.TFE_Py_ForwardAccumulatorPushState()
    yield
  finally:
    pywrap_tfe.TFE_Py_ForwardAccumulatorPopState()
