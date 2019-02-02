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
"""Sparse feature column (deprecated).

This module and all its submodules are deprecated. To UPDATE or USE linear
optimizers, please check its latest version in core:
tensorflow_estimator/python/estimator/canned/linear_optimizer/.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework.ops import internal_convert_to_tensor
from tensorflow.python.framework.ops import name_scope
from tensorflow.python.util import deprecation


class SparseFeatureColumn(object):
  """Represents a sparse feature column.

  Contains three tensors representing a sparse feature column, they are
  example indices (`int64`), feature indices (`int64`), and feature
  values (`float`).
  Feature weights are optional, and are treated as `1.0f` if missing.

  For example, consider a batch of 4 examples, which contains the following
  features in a particular `SparseFeatureColumn`:

  * Example 0: feature 5, value 1
  * Example 1: feature 6, value 1 and feature 10, value 0.5
  * Example 2: no features
  * Example 3: two copies of feature 2, value 1

  This SparseFeatureColumn will be represented as follows:

  ```
   <0, 5,  1>
   <1, 6,  1>
   <1, 10, 0.5>
   <3, 2,  1>
   <3, 2,  1>
  ```

  For a batch of 2 examples below:

  * Example 0: feature 5
  * Example 1: feature 6

  is represented by `SparseFeatureColumn` as:

  ```
   <0, 5,  1>
   <1, 6,  1>

  ```

  @@__init__
  @@example_indices
  @@feature_indices
  @@feature_values
  """

  @deprecation.deprecated(
      None, 'This class is deprecated. To UPDATE or USE linear optimizers, '
      'please check its latest version in core: '
      'tensorflow_estimator/python/estimator/canned/linear_optimizer/.')
  def __init__(self, example_indices, feature_indices, feature_values):
    """Creates a `SparseFeatureColumn` representation.

    Args:
      example_indices: A 1-D int64 tensor of shape `[N]`. Also, accepts
      python lists, or numpy arrays.
      feature_indices: A 1-D int64 tensor of shape `[N]`. Also, accepts
      python lists, or numpy arrays.
      feature_values: An optional 1-D tensor float tensor of shape `[N]`. Also,
      accepts python lists, or numpy arrays.

    Returns:
      A `SparseFeatureColumn`
    """
    with name_scope(None, 'SparseFeatureColumn',
                    [example_indices, feature_indices]):
      self._example_indices = internal_convert_to_tensor(
          example_indices, name='example_indices', dtype=dtypes.int64)
      self._feature_indices = internal_convert_to_tensor(
          feature_indices, name='feature_indices', dtype=dtypes.int64)
    self._feature_values = None
    if feature_values is not None:
      with name_scope(None, 'SparseFeatureColumn', [feature_values]):
        self._feature_values = internal_convert_to_tensor(
            feature_values, name='feature_values', dtype=dtypes.float32)

  @property
  def example_indices(self):
    """The example indices represented as a dense tensor.

    Returns:
      A 1-D Tensor of int64 with shape `[N]`.
    """
    return self._example_indices

  @property
  def feature_indices(self):
    """The feature indices represented as a dense tensor.

    Returns:
      A 1-D Tensor of int64 with shape `[N]`.
    """
    return self._feature_indices

  @property
  def feature_values(self):
    """The feature values represented as a dense tensor.

    Returns:
      May return None, or a 1-D Tensor of float32 with shape `[N]`.
    """
    return self._feature_values
