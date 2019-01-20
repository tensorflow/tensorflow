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
"""Code for backpropagation using the tape utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.ops.unconnected_gradients import UnconnectedGradients
from tensorflow.python.util import compat

VSpace = collections.namedtuple("VSpace", [
    "aggregate_fn", "num_elements_fn", "zeros_fn", "ones_fn", "graph_shape_fn"
])


def imperative_grad(
    tape,
    target,
    sources,
    output_gradients=None,
    unconnected_gradients=UnconnectedGradients.NONE):
  """Computes gradients from the imperatively defined tape on top of the stack.

  Works by filtering the tape, computing how many downstream usages are of each
  tensor and entry, and repeatedly applying backward functions until we have
  gradients for all sources.

  Args:
   tape: the gradient tape which stores the trace.
   target: either a Tensor or list of Tensors to be differentiated.
   sources: list of Tensors for which we want gradients
   output_gradients: if not None, a list of gradient provided for each Target,
    or None if we are to use the target's computed downstream gradient.
   unconnected_gradients: determines the value returned if the target and
    sources are unconnected. When 'none' the value returned is None wheras when
    'zero' a zero tensor in the same shape as the sources is returned.

  Returns:
   the gradient wrt each of the sources.

  Raises:
    ValueError: if the arguments are invalid.
    RuntimeError: if something goes wrong.
  """
  try:
    unconnected_gradients = UnconnectedGradients(unconnected_gradients)
  except ValueError:
    raise ValueError(
        "Unknown value for unconnected_gradients: %r" % unconnected_gradients)

  return pywrap_tensorflow.TFE_Py_TapeGradient(
      tape._tape,  # pylint: disable=protected-access
      target,
      sources,
      output_gradients,
      compat.as_str(unconnected_gradients.value))
