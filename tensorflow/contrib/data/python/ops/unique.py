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
"""Unique element dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.data.experimental.ops import unique as experimental_unique
from tensorflow.python.util import deprecation


@deprecation.deprecated(None, "Use `tf.data.experimental.unique()`.")
def unique():
  """Creates a `Dataset` from another `Dataset`, discarding duplicates.

  Use this transformation to produce a dataset that contains one instance of
  each unique element in the input. For example:

  ```python
  dataset = tf.data.Dataset.from_tensor_slices([1, 37, 2, 37, 2, 1])

  # Using `unique()` will drop the duplicate elements.
  dataset = dataset.apply(tf.data.experimental.unique())  # ==> { 1, 37, 2 }
  ```

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  return experimental_unique.unique()
