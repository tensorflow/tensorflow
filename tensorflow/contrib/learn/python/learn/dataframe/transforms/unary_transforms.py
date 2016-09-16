# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""Builds `Transforms` that wrap unary TensorFlow operations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import series
from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops

# Each entry is a mapping from registered_name to operation. Each operation is
# wrapped in a transform and then registered as a member function
# `Series`.registered_name().
UNARY_TRANSFORMS = [("__neg__", math_ops.neg),
                    ("sign", math_ops.sign),
                    ("inv", math_ops.inv),
                    ("square", math_ops.square),
                    ("round", math_ops.round),
                    ("sqrt", math_ops.sqrt),
                    ("rsqrt", math_ops.rsqrt),
                    ("exp", math_ops.exp),
                    ("log", math_ops.log),
                    ("ceil", math_ops.ceil),
                    ("floor", math_ops.floor),
                    ("cos", math_ops.cos),
                    ("sin", math_ops.sin),
                    ("lgamma", math_ops.lgamma),
                    ("digamma", math_ops.digamma),
                    ("erf", math_ops.erf),
                    ("erfc", math_ops.erfc),
                    ("__invert__", math_ops.logical_not, bool)]

DOC_FORMAT_STRING = (
    "A `Transform` that wraps the `{0}` operation. "
    "Documentation for `{0}`: \n\n {1}"
)


# pylint: disable=unused-argument
def register_unary_op(registered_name, operation, ignore_dtype=None):
  """Creates a `Transform` that wraps a unary tensorflow operation.

  If `registered_name` is specified, the `Transform` is registered as a member
  function of `Series`.

  Args:
    registered_name: the name of the member function of `Series` corresponding
      to the returned `Transform`.
    operation: a unary TensorFlow operation.
    ignore_dtype: an optional dtype, not used here but needed for symmetry with
      test.
  """

  doc = DOC_FORMAT_STRING.format(operation.__name__, operation.__doc__)

  @property
  def name(self):
    return operation.__name__

  @property
  def input_valency(self):
    return 1

  @property
  def _output_names(self):
    return "output"

  def _apply_transform(self, input_tensors, **kwargs):
    input_tensor = input_tensors[0]
    if isinstance(input_tensor, ops.SparseTensor):
      result = ops.SparseTensor(input_tensor.indices,
                                operation(input_tensor.values),
                                input_tensor.shape)
    else:
      result = operation(input_tensor)
    # pylint: disable=not-callable
    return self.return_type(result)

  cls = type(operation.__name__,
             (transform.TensorFlowTransform,),
             {"name": name,
              "__doc__": doc,
              "input_valency": input_valency,
              "_output_names": _output_names,
              "_apply_transform": _apply_transform})

  series.Series.register_unary_op(registered_name)(cls)
