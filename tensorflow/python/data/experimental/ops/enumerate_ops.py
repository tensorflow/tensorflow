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
"""Enumerate dataset transformations."""
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export


@deprecation.deprecated(None, "Use `tf.data.Dataset.enumerate()`.")
@tf_export("data.experimental.enumerate_dataset")
def enumerate_dataset(start=0):
  """A transformation that enumerates the elements of a dataset.

  It is similar to python's `enumerate`.
  For example:

  ```python
  # NOTE: The following examples use `{ ... }` to represent the
  # contents of a dataset.
  a = { 1, 2, 3 }
  b = { (7, 8), (9, 10) }

  # The nested structure of the `datasets` argument determines the
  # structure of elements in the resulting dataset.
  a.apply(tf.data.experimental.enumerate_dataset(start=5))
  => { (5, 1), (6, 2), (7, 3) }
  b.apply(tf.data.experimental.enumerate_dataset())
  => { (0, (7, 8)), (1, (9, 10)) }
  ```

  Args:
    start: A `tf.int64` scalar `tf.Tensor`, representing the start value for
      enumeration.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """

  def _apply_fn(dataset):
    return dataset.enumerate(start)

  return _apply_fn
