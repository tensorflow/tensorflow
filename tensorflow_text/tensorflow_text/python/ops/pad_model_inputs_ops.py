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

"""Library of ops to pack model inputs."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import array_ops_stack
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor


def pad_model_inputs(input, max_seq_length, pad_value=0):  # pylint: disable=redefined-builtin
  r"""Pad model input and generate corresponding input masks.

  `pad_model_inputs` performs the final packaging of a model's inputs commonly
  found in text models. This includes padding out (or simply truncating) to a
  fixed-size, max 2-dimensional `Tensor` and generating mask `Tensor`s (of the
  same shape) with values of 0 if the corresponding item is a pad value and 1 if
  it is part of the original input.

  Note that a simple truncation strategy (drop everything after max sequence
  length) is used to force the inputs to the specified shape. This may be
  incorrect and users should instead apply a `Trimmer` upstream to safely
  truncate large inputs.

  >>> input_data = tf.ragged.constant([
  ...            [101, 1, 2, 102, 10, 20, 102],
  ...            [101, 3, 4, 102, 30, 40, 50, 60, 70, 80],
  ...            [101, 5, 6, 7, 8, 9, 102, 70],
  ...        ], np.int32)
  >>> data, mask = pad_model_inputs(input=input_data, max_seq_length=9)
  >>> print("data: %s, mask: %s" % (data, mask))
    data: tf.Tensor(
    [[101   1   2 102  10  20 102   0   0]
     [101   3   4 102  30  40  50  60  70]
     [101   5   6   7   8   9 102  70   0]], shape=(3, 9), dtype=int32),
    mask: tf.Tensor(
    [[1 1 1 1 1 1 1 0 0]
     [1 1 1 1 1 1 1 1 1]
     [1 1 1 1 1 1 1 1 0]], shape=(3, 9), dtype=int32)

  Args:
    input: A `RaggedTensor` or `Tensor` with rank >= 1.
    max_seq_length: An int, or scalar `Tensor`. The "input" `Tensor` will be
      flattened down to 2 dimensions (if needed), and then have its inner
      dimension either padded out or truncated to this size.
    pad_value: An int or scalar `Tensor` specifying the value used for padding.

  Returns:
      A tuple of (padded_input, pad_mask) where:

      padded_input: A `Tensor` corresponding to `inputs` that has been
        padded/truncated out to a fixed size and flattened to max 2
        dimensions.
      pad_mask: A `Tensor` corresponding to `padded_input` whose values are
        0 if the corresponding item is a pad value and 1 if it is not.
  """
  with ops.name_scope("pad_model_inputs"):
    input = ragged_tensor.convert_to_tensor_or_ragged_tensor(input)

    if isinstance(input, tensor.Tensor):
      if input.shape.ndims > 1:
        return pad_model_inputs(ragged_tensor.RaggedTensor.from_tensor(input),
                                max_seq_length, pad_value)
      elif input.shape.ndims == 1:
        input = array_ops_stack.stack([input])
        (padded, mask) = pad_model_inputs(
            ragged_tensor.RaggedTensor.from_tensor(input), max_seq_length,
            pad_value)
        return array_ops.reshape(padded, [-1]), array_ops.reshape(mask, [-1])
      else:
        raise TypeError("Input must be at least rank 1")

    # Verify that everything is a RaggedTensor
    if not isinstance(input, ragged_tensor.RaggedTensor):
      raise TypeError("Expecting a `RaggedTensor` or `Tensor`, instead found: "
                      + str(input))

    # Flatten down to `merge_axis`
    input = input.merge_dims(1, -1) if input.ragged_rank > 1 else input

    # Pad to fixed Tensor
    target_shape = math_ops.cast([-1, max_seq_length], dtypes.int64)
    padded_input = input.to_tensor(shape=target_shape, default_value=pad_value)

    # Get padded input mask
    input_mask = array_ops.ones_like(input)
    padded_input_mask = input_mask.to_tensor(shape=target_shape)

    return padded_input, padded_input_mask
