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

"""Transforms Sparse to Dense Tensor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.contrib.learn.python.learn.dataframe import transform
from tensorflow.python.ops import sparse_ops


class Densify(transform.TensorFlowTransform):
  """Transforms Sparse to Dense Tensor."""

  def __init__(self,
               default_value):
    super(Densify, self).__init__()
    self._default_value = default_value

  @transform.parameter
  def default_value(self):
    return self._default_value

  @property
  def name(self):
    return "Densify"

  @property
  def input_valency(self):
    return 1

  @property
  def _output_names(self):
    return "output",

  def _apply_transform(self, input_tensors, **kwargs):
    """Applies the transformation to the `transform_input`.

    Args:
      input_tensors: a list of Tensors representing the input to
        the Transform.
      **kwargs: Additional keyword arguments, unused here.

    Returns:
        A namedtuple of Tensors representing the transformed output.
    """
    s = input_tensors[0]

     # pylint: disable=not-callable
    return self.return_type(sparse_ops.sparse_to_dense(
        s.indices, s.dense_shape, s.values, default_value=self.default_value))
