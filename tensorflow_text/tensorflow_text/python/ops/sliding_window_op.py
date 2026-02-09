# coding=utf-8
# Copyright 2025 TF.Text Authors.
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

"""Sliding window op.

Returns a sliding window of data with a specified width.
"""

from __future__ import absolute_import
from __future__ import print_function

from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops.ragged import ragged_tensor


def sliding_window(data, width, axis=-1, name=None):
  """Builds a sliding window for `data` with a specified width.

  Returns a tensor constructed from `data`, where each element in
  dimension `axis` is a slice of `data` starting at the corresponding
  position, with the given width and step size.  I.e.:

  * `result.shape.ndims = data.shape.ndims + 1`
  * `result[i1..iaxis, a] = data[i1..iaxis, a:a+width]`
    (where `0 <= a < data[i1...iaxis].shape[0] - (width - 1)`).

  Note that each result row (along dimension `axis`) has `width - 1` fewer items
  than the corresponding `data` row.  If a `data` row has fewer than `width`
  items, then the corresponding `result` row will be empty.  If you wish for
  the `result` rows to be the same size as the `data` rows, you can use
  `pad_along_dimension` to add `width - 1` padding elements before calling
  this op.

  #### Examples:

  Sliding window (width=3) across a sequence of tokens:

  >>> # input: <string>[sequence_length]
  >>> input = tf.constant(["one", "two", "three", "four", "five", "six"])
  >>> # output: <string>[sequence_length-2, 3]
  >>> sliding_window(data=input, width=3, axis=0)
  <tf.Tensor: shape=(4, 3), dtype=string, numpy=
      array([[b'one', b'two', b'three'],
             [b'two', b'three', b'four'],
             [b'three', b'four', b'five'],
             [b'four', b'five', b'six']], dtype=object)>

  Sliding window (width=2) across the inner dimension of a ragged matrix
  containing a batch of token sequences:

  >>> # input: <string>[num_sentences, (num_words)]
  >>> input = tf.ragged.constant(
  ...     [['Up', 'high', 'in', 'the', 'air'],
  ...      ['Down', 'under', 'water'],
  ...      ['Away', 'to', 'outer', 'space']])
  >>> # output: <string>[num_sentences, (num_word-1), 2]
  >>> sliding_window(input, width=2, axis=-1)
  <tf.RaggedTensor [[[b'Up', b'high'], [b'high', b'in'], [b'in', b'the'],
                     [b'the', b'air']], [[b'Down', b'under'],
                     [b'under', b'water']],
                    [[b'Away', b'to'], [b'to', b'outer'],
                     [b'outer', b'space']]]>

  Sliding window across the second dimension of a 3-D tensor containing
  batches of sequences of embedding vectors:

  >>> # input: <int32>[num_sequences, sequence_length, embedding_size]
  >>> input = tf.constant([
  ...     [[1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1], [5, 5, 1]],
  ...     [[1, 1, 2], [2, 2, 2], [3, 3, 2], [4, 4, 2], [5, 5, 2]]])
  >>> # output: <int32>[num_sequences, sequence_length-1, 2, embedding_size]
  >>> sliding_window(data=input, width=2, axis=1)
  <tf.Tensor: shape=(2, 4, 2, 3), dtype=int32, numpy=
      array([[[[1, 1, 1],
               [2, 2, 1]],
              [[2, 2, 1],
               [3, 3, 1]],
              [[3, 3, 1],
               [4, 4, 1]],
              [[4, 4, 1],
               [5, 5, 1]]],
             [[[1, 1, 2],
               [2, 2, 2]],
              [[2, 2, 2],
               [3, 3, 2]],
              [[3, 3, 2],
               [4, 4, 2]],
              [[4, 4, 2],
               [5, 5, 2]]]], dtype=int32)>

  Args:
    data: `<dtype> [O1...ON, A, I1...IM]`
      A potentially ragged K-dimensional tensor with outer dimensions of size
      `O1...ON`; axis dimension of size `A`; and inner dimensions of size
      `I1...IM`.  I.e. `K = N + 1 + M`, where `N>=0` and `M>=0`.

    width: An integer constant specifying the width of the window. Must be
      greater than zero.

    axis: An integer constant specifying the axis along which sliding window
      is computed. Negative axis values from `-K` to `-1` are supported.

    name: The name for this op (optional).

  Returns:
    A `K+1` dimensional tensor with the same dtype as `data`, where:

    * `result[i1..iaxis, a]` = `data[i1..iaxis, a:a+width]`
    * `result.shape[:axis]` = `data.shape[:axis]`
    * `result.shape[axis]` = `data.shape[axis] - (width - 1)`
    * `result.shape[axis + 1]` = `width`
    * `result.shape[axis + 2:]` = `data.shape[axis + 1:]`
  """
  with ops.name_scope(name, "SlidingWindow", [data, axis]):
    data = ragged_tensor.convert_to_tensor_or_ragged_tensor(data, name="data")

    if not isinstance(axis, int):
      raise TypeError("axis must be an int")

    if not isinstance(width, int):
      raise TypeError("width must be an int")

    if data.shape.ndims is not None and (axis < -data.shape.ndims or
                                         axis >= data.shape.ndims):
      raise errors.InvalidArgumentError(
          None, None, "axis must be between -k <= axis <= -1 OR 0 <= axis < k")

    if width <= 0:
      raise errors.InvalidArgumentError(
          None, None, "width must be an integer greater than 0")

    slices = []
    for start in range(width):
      stop = None if start - width + 1 == 0 else start - width + 1
      if axis >= 0:
        idx = [slice(None)] * axis + [slice(start, stop)]
      else:
        idx = [Ellipsis, slice(start, stop)] + [slice(None)] * (-axis - 1)
      slices.append(data[idx])

    # Stack the slices.
    stack_axis = axis + 1 if axis >= 0 else axis
    return array_ops_stack.stack(slices, stack_axis)
