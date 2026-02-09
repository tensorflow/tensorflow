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

"""BOISE offset converter for string tensors."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.framework import ops
from tensorflow.python.ops.ragged import ragged_conversion_ops
from tensorflow.python.ops.ragged import ragged_tensor

# pylint: disable=g-bad-import-order,unused-import
from tensorflow.python.ops.ragged.ragged_tensor import RaggedTensor
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
gen_boise_offset_converter = load_library.load_op_library(resource_loader.get_path_to_datafile('_boise_offset_converter.so'))


def _validate_input_has_same_type(*args):
  tensor_type = type(args[-1])
  for arg in args[:-1]:
    if not isinstance(arg, tensor_type):
      raise ValueError("Input tensors must be of the same type: %s vs %s." %
                       (type(arg), tensor_type))


def offsets_to_boise_tags(token_begin_offsets,
                          token_end_offsets,
                          span_begin_offsets,
                          span_end_offsets,
                          span_type,
                          use_strict_boundary_mode=False):
  """Converts the given tokens and spans in offsets format into BOISE tags.

  In the BOISE scheme there is a set of 5 labels for each type:
    - (B)egin: meaning the beginning of the span type.
    - (O)utside: meaning the token is outside of any span type
    - (I)nside: the token is inside the span
    - (S)ingleton: the entire span consists of this single token.
    - (E)nd: this token is the end of the span.

  When given the span begin & end offsets along with a set of token begin & end
  offsets, this function helps translate which each token into one of the 5
  labels.

  For example, given the following example string and entity:

    content = "Who let the dogs out"
    entity = "dogs"
    tokens = ["Who", "let", "the", "dogs", "out"]
    token_begin_offsets = [0, 4, 8, 12, 17]
    token_end_offsets = [3, 7, 11, 16, 20]
    span_begin_offsets = [12]
    span_end_offsets = [16]
    span_type = ["animal"]

   Foo will produce the following labels:
     ["O", "O", "O",  "S-animal", "O"]
       |    |    |        |        |
      Who  let  the      dogs     out

   Special Case 1: Loose or Strict Boundary Criteria:
   By default, loose boundary criteria are used to decide token start and end,
   given a entity span. In the above example, say if we have

    span_begin_offsets = [13];
    span_end_offsets = [16];

   we still get ["O", "O", "O", "S-animal", "O"], even though the span
   begin offset (13) is not exactly aligned with the token begin offset (12).
   Partial overlap between a token and a BOISE tag still qualify the token to
   be labeled with this tag.

   You can choose to use strict boundary criteria by passing in
   use_strict_boundary_mode = false argument, with which Foo will produce
   ["O", "O", "O", "O", "O"] for the case described above.

   Special Case 2: One Token Mapped to Multiple BOISE Tags:
   In cases where a token is overlapped with multiple BOISE tags, we label the
   token with the last tag. For example, given the following example inputs:

    std::string content = "Getty Center";
    std::vector<string> tokens = { "Getty Center" };
    std::vector<int> token_begin_offsets = { 0 };
    std::vector<int> token_end_offsets = { 12 };
    std::vector<int> span_begin_offsets = { 0, 6 };
    std::vector<int> span_end_offsets = { 5, 12 };
    std::vector<string> span_type = { "per", "loc" }

   Foo will produce the following labels:
    ["B-loc"]

  ### Example:
  >>> token_begin_offsets = tf.ragged.constant(
  ...   [[0, 4, 8, 12, 17], [0, 4, 8, 12]])
  >>> token_end_offsets = tf.ragged.constant(
  ...   [[3, 7, 11, 16, 20], [3, 7, 11, 16]])
  >>> span_begin_offsets = tf.ragged.constant([[4], [12]])
  >>> span_end_offsets = tf.ragged.constant([[16], [16]])
  >>> span_type = tf.ragged.constant([['animal'], ['loc']])
  >>> boise_tags = tf_text.offsets_to_boise_tags(token_begin_offsets,
  ...   token_end_offsets, span_begin_offsets, span_end_offsets, span_type)
  >>> boise_tags
  <tf.RaggedTensor [[b'O', b'B-animal', b'I-animal', b'E-animal', b'O'],
  [b'O', b'O', b'O', b'S-loc']]>

  Args:
    token_begin_offsets: A `RaggedTensor` or `Tensor` of token begin byte
      offsets of int32 or int64.
    token_end_offsets: A `RaggedTensor` or `Tensor` of token end byte offsets of
      int32 or int64.
    span_begin_offsets: A `RaggedTensor` or `Tensor` of span begin byte offsets
      of int32 or int64.
    span_end_offsets: A `RaggedTensor` or `Tensor` of span end byte offsets of
      int32 or int64.
    span_type: A `RaggedTensor` or `Tensor` of span type strings.
    use_strict_boundary_mode: A bool indicating whether to use the strict
      boundary mode, which excludes a token from a span label when the token
      begin/end byte range partially overlaps with the span range.

  Returns:
    A `RaggedTensor` of BOISE tag strings in the same dimension as the input
    token begin and end offsets.
  """
  name = None
  with ops.name_scope(name, "OffsetsToBoiseTags", [
      token_begin_offsets,
      token_end_offsets,
      span_begin_offsets,
      span_end_offsets,
      span_type,
      use_strict_boundary_mode,
  ]):
    token_begin_offsets_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        token_begin_offsets)
    token_end_offsets_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        token_end_offsets)
    span_begin_offsets_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        span_begin_offsets)
    span_end_offsets_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        span_end_offsets)
    span_type_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        span_type)

    _validate_input_has_same_type(token_begin_offsets_tensor,
                                  token_end_offsets_tensor,
                                  span_begin_offsets_tensor,
                                  span_end_offsets_tensor, span_type_tensor)

    token_begin_offsets_tensor = tf.cast(
        token_begin_offsets_tensor, dtype=tf.int32)
    token_end_offsets_tensor = tf.cast(token_end_offsets_tensor, dtype=tf.int32)
    span_begin_offsets_tensor = tf.cast(
        span_begin_offsets_tensor, dtype=tf.int32)
    span_end_offsets_tensor = tf.cast(span_end_offsets_tensor, dtype=tf.int32)

    if token_begin_offsets_tensor.shape.ndims is None:
      raise ValueError("Rank of input_tensor must be statically known.")
    if ragged_tensor.is_ragged(token_begin_offsets_tensor):
      try:
        boise_tags = gen_boise_offset_converter.tf_text_offsets_to_boise_tags(
            input_token_begin_offsets=token_begin_offsets_tensor.flat_values,
            input_token_end_offsets=token_end_offsets_tensor.flat_values,
            input_span_begin_offsets=span_begin_offsets_tensor.flat_values,
            input_span_end_offsets=span_end_offsets_tensor.flat_values,
            input_span_type=span_type_tensor.flat_values,
            input_token_begin_row_splits=token_begin_offsets_tensor
            .nested_row_splits[-1],
            input_token_end_row_splits=token_end_offsets_tensor
            .nested_row_splits[-1],
            input_span_begin_row_splits=span_begin_offsets_tensor
            .nested_row_splits[-1],
            input_span_end_row_splits=span_end_offsets_tensor
            .nested_row_splits[-1],
            input_span_type_row_splits=span_type_tensor.nested_row_splits[-1],
            input_use_strict_boundary_mode=use_strict_boundary_mode)
        result = token_end_offsets_tensor.with_flat_values(boise_tags)
        return result
      except (tf.errors.InvalidArgumentError,
              tf.errors.FailedPreconditionError) as exc:
        raise ValueError(
            "Input token and span tensor shape don't match.") from exc
    else:
      if token_begin_offsets_tensor.shape.ndims > 1:
        return offsets_to_boise_tags(
            ragged_conversion_ops.from_tensor(token_begin_offsets_tensor),
            ragged_conversion_ops.from_tensor(token_end_offsets_tensor),
            ragged_conversion_ops.from_tensor(span_begin_offsets_tensor),
            ragged_conversion_ops.from_tensor(span_end_offsets_tensor),
            ragged_conversion_ops.from_tensor(span_type_tensor),
            use_strict_boundary_mode)
      elif token_begin_offsets_tensor.shape.ndims == 0:
        result = offsets_to_boise_tags(
            tf.stack([token_begin_offsets_tensor]),
            tf.stack([token_end_offsets_tensor]),
            tf.stack([span_begin_offsets_tensor]),
            tf.stack([span_end_offsets_tensor]), tf.stack([span_type_tensor]),
            use_strict_boundary_mode)
        return result[0]
      else:
        result = offsets_to_boise_tags(
            tf.stack([token_begin_offsets_tensor]),
            tf.stack([token_end_offsets_tensor]),
            tf.stack([span_begin_offsets_tensor]),
            tf.stack([span_end_offsets_tensor]), tf.stack([span_type_tensor]),
            use_strict_boundary_mode)
        return result.merge_dims(0, 1)


def boise_tags_to_offsets(token_begin_offsets, token_end_offsets, boise_tags):
  """Converts the token offsets and BOISE tags into span offsets and span type.

  In the BOISE scheme there is a set of 5 labels for each type:
    - (B)egin: meaning the beginning of the span type.
    - (O)utside: meaning the token is outside of any span type
    - (I)nside: the token is inside the span
    - (S)ingleton: the entire span consists of this single token.
    - (E)nd: this token is the end of the span.

  For example, given the following example string and entity:

    content = "Who let the dogs out"
    entity = "dogs"
    tokens = ["Who", "let", "the", "dogs", "out"]
    token_begin_offsets = [0, 4, 8, 12, 17]
    token_end_offsets = [3, 7, 11, 16, 20]
    span_begin_offsets = [12]
    span_end_offsets = [16]
    span_type = ["animal"]

    BOISE tags are:
     ["O", "O", "O",  "S-animal", "O"]
       |    |    |        |        |
      Who  let  the      dogs     out

  When given the token begin/end offsets and BOISE tags for an input text
  sequence, this function translates them into entity span begin/end offsets
  and span types.

  ### Example:
  >>> token_begin_offsets = tf.ragged.constant(
  ...   [[0, 4, 8, 12, 17], [0, 4, 8, 12]])
  >>> token_end_offsets = tf.ragged.constant(
  ...   [[3, 7, 11, 16, 20], [3, 7, 11, 16]])
  >>> boise_tags = tf.ragged.constant(
  ...   [['O', 'B-animal', 'I-animal', 'E-animal', 'O'],
  ...    ['O', 'O', 'O', 'S-loc']])
  >>> (span_begin_offsets, span_end_offsets, span_type) = (
  ...   tf_text.boise_tags_to_offsets(token_begin_offsets, token_end_offsets,
  ...     boise_tags))
  >>> span_begin_offsets
  <tf.RaggedTensor [[4], [12]]>
  >>> span_end_offsets
  <tf.RaggedTensor [[16], [16]]>
  >>> span_type
  <tf.RaggedTensor [[b'animal'], [b'loc']]>

  Args:
    token_begin_offsets: A `RaggedTensor` or `Tensor` of token begin byte
      offsets of int32 or int64.
    token_end_offsets: A `RaggedTensor` or `Tensor` of token end byte offsets of
      int32 or int64.
    boise_tags: A `RaggedTensor` of BOISE tag strings in the same dimension as
      the token begin and end offsets.

  Returns:
   A tuple containing `span_begin_offsets`, `span_end_offsets` and `span_type`.
   `span_begin_offsets` is a `RaggedTensor` or `Tensor` of span begin byte
      offsets of int32 or int64.
   `span_end_offsets` is a `RaggedTensor` or `Tensor` of span end byte offsets
      of int32 or int64.
   `span_type` is a `RaggedTensor` or `Tensor` of span type strings.
  """
  name = None
  with ops.name_scope(name, "BoiseTagsToOffsets", [
      token_begin_offsets,
      token_end_offsets,
      boise_tags,
  ]):
    (token_begin_offsets_tensor) = (
        ragged_tensor.convert_to_tensor_or_ragged_tensor(token_begin_offsets))
    token_end_offsets_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        token_end_offsets)
    boise_tags_tensor = ragged_tensor.convert_to_tensor_or_ragged_tensor(
        boise_tags)

    _validate_input_has_same_type(token_begin_offsets_tensor,
                                  token_end_offsets_tensor, boise_tags_tensor)

    token_begin_offsets_tensor = tf.cast(
        token_begin_offsets_tensor, dtype=tf.int32)
    token_end_offsets_tensor = tf.cast(token_end_offsets_tensor, dtype=tf.int32)

    if token_begin_offsets_tensor.shape.ndims is None:
      raise ValueError("Rank of input_tensor must be statically known.")
    if ragged_tensor.is_ragged(token_begin_offsets_tensor):
      try:
        (span_begin_offsets, span_end_offsets, span_type, row_splits) = (
            gen_boise_offset_converter.tf_text_boise_tags_to_offsets(
                input_token_begin_offsets=token_begin_offsets_tensor
                .flat_values,
                input_token_end_offsets=token_end_offsets_tensor.flat_values,
                input_boise_tags=boise_tags_tensor.flat_values,
                input_token_begin_row_splits=token_begin_offsets_tensor
                .nested_row_splits[-1],
                input_token_end_row_splits=token_end_offsets_tensor
                .nested_row_splits[-1],
                input_boise_tags_row_splits=boise_tags_tensor
                .nested_row_splits[-1]))
        span_begin_offsets = RaggedTensor.from_nested_row_splits(
            flat_values=span_begin_offsets, nested_row_splits=[row_splits])
        span_end_offsets = RaggedTensor.from_nested_row_splits(
            flat_values=span_end_offsets, nested_row_splits=[row_splits])
        span_type = RaggedTensor.from_nested_row_splits(
            flat_values=span_type, nested_row_splits=[row_splits])
        return span_begin_offsets, span_end_offsets, span_type
      except (tf.errors.InvalidArgumentError,
              tf.errors.FailedPreconditionError) as exc:
        raise ValueError(
            "Input token and BOISE tags tensor shape don't match.") from exc
    else:
      if token_begin_offsets_tensor.shape.ndims > 1:
        return boise_tags_to_offsets(
            ragged_conversion_ops.from_tensor(token_begin_offsets_tensor),
            ragged_conversion_ops.from_tensor(token_end_offsets_tensor),
            ragged_conversion_ops.from_tensor(boise_tags_tensor))
      elif token_begin_offsets_tensor.shape.ndims == 0:
        span_begin_offsets, span_end_offsets, span_type = boise_tags_to_offsets(
            tf.stack([token_begin_offsets_tensor]),
            tf.stack([token_end_offsets_tensor]), tf.stack([boise_tags_tensor]))
        return span_begin_offsets[0], span_end_offsets[0], span_type[0]
      else:
        span_begin_offsets, span_end_offsets, span_type = boise_tags_to_offsets(
            tf.stack([token_begin_offsets_tensor]),
            tf.stack([token_end_offsets_tensor]), tf.stack([boise_tags_tensor]))
        return span_begin_offsets.merge_dims(0, 1), span_end_offsets.merge_dims(
            0, 1), span_type.merge_dims(0, 1)
