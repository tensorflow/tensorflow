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


VSpace = collections.namedtuple(
    "VSpace", ["aggregate_fn", "num_elements_fn", "zeros", "ones"])


def imperative_grad(
    tape,
    target,
    sources,
    output_gradients=None):
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

  Returns:
   the gradient wrt each of the sources.

  Raises:
    RuntimeError: if something goes wrong.
    ValueError: if there is no sequence of differentiable operations connecting
     a source and any target Tensor. This can happen either if the target is
     not computed based on the source, if the tracing was set up incorrectly,
     or if only non-differentiable functions of the source were used in the
     computation of target.
  """
  return pywrap_tensorflow.TFE_Py_TapeGradient(
      tape._tape,  # pylint: disable=protected-access
      target,
      sources,
      output_gradients)
