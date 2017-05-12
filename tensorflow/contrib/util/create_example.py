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
"""Utilities for constructing Example protos.

Takes ndarrays, lists, or tuples for each feature.

@@create_example
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.core.example import example_pb2


def create_example(**features):
  """Constructs a `tf.train.Example` from the given features.

  Args:
    **features: Maps feature name to an integer, float, or string ndarray, or
        another object convertible to an ndarray (list, tuple, etc).

  Returns:
    A `tf.train.Example` with the features.

  Raises:
    ValueError: if a feature is not integer, float, or string.
  """
  example = example_pb2.Example()
  for name in features:
    feature = example.features.feature[name]
    values = np.asarray(features[name])
    # Encode unicode using UTF-8.
    if values.dtype.kind == 'U':
      values = np.vectorize(lambda string: string.encode('utf-8'))(values)

    if values.dtype.kind == 'i':
      feature.int64_list.value.extend(values.astype(np.int64).ravel())
    elif values.dtype.kind == 'f':
      feature.float_list.value.extend(values.astype(np.float32).ravel())
    elif values.dtype.kind == 'S':
      feature.bytes_list.value.extend(values.ravel())
    else:
      raise ValueError('Feature "%s" has unexpected dtype: %s' % (name,
                                                                  values.dtype))
  return example
