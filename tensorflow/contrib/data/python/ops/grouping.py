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
"""Grouping dataset transformations."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensorflow.python.data.experimental.ops import grouping
from tensorflow.python.util import deprecation


@deprecation.deprecated(None,
                        "Use `tf.data.experimental.group_by_reducer(...)`.")
def group_by_reducer(key_func, reducer):
  """A transformation that groups elements and performs a reduction.

  This transformation maps element of a dataset to a key using `key_func` and
  groups the elements by key. The `reducer` is used to process each group; its
  `init_func` is used to initialize state for each group when it is created, the
  `reduce_func` is used to update the state every time an element is mapped to
  the matching group, and the `finalize_func` is used to map the final state to
  an output value.

  Args:
    key_func: A function mapping a nested structure of tensors
      (having shapes and types defined by `self.output_shapes` and
      `self.output_types`) to a scalar `tf.int64` tensor.
    reducer: An instance of `Reducer`, which captures the reduction logic using
      the `init_func`, `reduce_func`, and `finalize_func` functions.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.
  """
  return grouping.group_by_reducer(key_func, reducer)


@deprecation.deprecated(None,
                        "Use `tf.data.experimental.group_by_window(...)`.")
def group_by_window(key_func,
                    reduce_func,
                    window_size=None,
                    window_size_func=None):
  """A transformation that groups windows of elements by key and reduces them.

  This transformation maps each consecutive element in a dataset to a key
  using `key_func` and groups the elements by key. It then applies
  `reduce_func` to at most `window_size_func(key)` elements matching the same
  key. All except the final window for each key will contain
  `window_size_func(key)` elements; the final window may be smaller.

  You may provide either a constant `window_size` or a window size determined by
  the key through `window_size_func`.

  Args:
    key_func: A function mapping a nested structure of tensors
      (having shapes and types defined by `self.output_shapes` and
      `self.output_types`) to a scalar `tf.int64` tensor.
    reduce_func: A function mapping a key and a dataset of up to `window_size`
      consecutive elements matching that key to another dataset.
    window_size: A `tf.int64` scalar `tf.Tensor`, representing the number of
      consecutive elements matching the same key to combine in a single
      batch, which will be passed to `reduce_func`. Mutually exclusive with
      `window_size_func`.
    window_size_func: A function mapping a key to a `tf.int64` scalar
      `tf.Tensor`, representing the number of consecutive elements matching
      the same key to combine in a single batch, which will be passed to
      `reduce_func`. Mutually exclusive with `window_size`.

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if neither or both of {`window_size`, `window_size_func`} are
      passed.
  """
  return grouping.group_by_window(key_func, reduce_func, window_size,
                                  window_size_func)


@deprecation.deprecated(
    None, "Use `tf.data.experimental.bucket_by_sequence_length(...)`.")
def bucket_by_sequence_length(element_length_func,
                              bucket_boundaries,
                              bucket_batch_sizes,
                              padded_shapes=None,
                              padding_values=None,
                              pad_to_bucket_boundary=False,
                              no_padding=False):
  """A transformation that buckets elements in a `Dataset` by length.

  Elements of the `Dataset` are grouped together by length and then are padded
  and batched.

  This is useful for sequence tasks in which the elements have variable length.
  Grouping together elements that have similar lengths reduces the total
  fraction of padding in a batch which increases training step efficiency.

  Args:
    element_length_func: function from element in `Dataset` to `tf.int32`,
      determines the length of the element, which will determine the bucket it
      goes into.
    bucket_boundaries: `list<int>`, upper length boundaries of the buckets.
    bucket_batch_sizes: `list<int>`, batch size per bucket. Length should be
      `len(bucket_boundaries) + 1`.
    padded_shapes: Nested structure of `tf.TensorShape` to pass to
      `tf.data.Dataset.padded_batch`. If not provided, will use
      `dataset.output_shapes`, which will result in variable length dimensions
      being padded out to the maximum length in each batch.
    padding_values: Values to pad with, passed to
      `tf.data.Dataset.padded_batch`. Defaults to padding with 0.
    pad_to_bucket_boundary: bool, if `False`, will pad dimensions with unknown
      size to maximum length in batch. If `True`, will pad dimensions with
      unknown size to bucket boundary minus 1 (i.e., the maximum length in each
      bucket), and caller must ensure that the source `Dataset` does not contain
      any elements with length longer than `max(bucket_boundaries)`.
    no_padding: `bool`, indicates whether to pad the batch features (features
      need to be either of type `tf.SparseTensor` or of same shape).

  Returns:
    A `Dataset` transformation function, which can be passed to
    `tf.data.Dataset.apply`.

  Raises:
    ValueError: if `len(bucket_batch_sizes) != len(bucket_boundaries) + 1`.
  """
  return grouping.bucket_by_sequence_length(
      element_length_func, bucket_boundaries, bucket_batch_sizes, padded_shapes,
      padding_values, pad_to_bucket_boundary, no_padding)


class Reducer(grouping.Reducer):
  """A reducer is used for reducing a set of elements.

  A reducer is represented as a tuple of the three functions:
    1) initialization function: key => initial state
    2) reduce function: (old state, input) => new state
    3) finalization function: state => result
  """

  @deprecation.deprecated(None, "Use `tf.data.experimental.Reducer(...)`.")
  def __init__(self, init_func, reduce_func, finalize_func):
    super(Reducer, self).__init__(init_func, reduce_func, finalize_func)
