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

"""Library of ops for building segments."""
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops


def combine_segments(segments, start_of_sequence_id, end_of_segment_id):
  """Combine one or more input segments for a model's input sequence.

  `combine_segments` combines the tokens of one or more input segments to a
  single sequence of token values and generates matching segment ids.
  `combine_segments` can follow a `Trimmer`, who limit segment lengths and
  emit `RaggedTensor` outputs, and can be followed up by `ModelInputPacker`.

  See `Detailed Experimental Setup` in `BERT: Pre-training of Deep Bidirectional
  Transformers for Language Understanding`
  (https://arxiv.org/pdf/1810.04805.pdf) for more examples of combined
  segments.


  `combine_segments` first flattens and combines a list of one or more
  segments
  (`RaggedTensor`s of n dimensions) together along the 1st axis, then packages
  any special tokens  into a final n dimensional `RaggedTensor`.

  And finally `combine_segments` generates another `RaggedTensor` (with the
  same rank as the final combined `RaggedTensor`) that contains a distinct int
  id for each segment.

  Example usage:

  ```
  segment_a = [[1, 2],
               [3, 4,],
               [5, 6, 7, 8, 9]]

  segment_b = [[10, 20,],
               [30, 40, 50, 60,],
               [70, 80]]
  expected_combined, expected_ids = combine_segments([segment_a, segment_b])

  # segment_a and segment_b have been combined w/ special tokens describing
  # the beginning of a sequence and end of a sequence inserted.
  expected_combined=[
   [101, 1, 2, 102, 10, 20, 102],
   [101, 3, 4, 102, 30, 40, 50, 60, 102],
   [101, 5, 6, 7, 8, 9, 102, 70, 80, 102],
  ]

  # ids describing which items belong to which segment.
  expected_ids=[
   [0, 0, 0, 0, 1, 1, 1],
   [0, 0, 0, 0, 1, 1, 1, 1, 1],
   [0, 0, 0, 0, 0, 0, 0, 1, 1, 1]]
  ```

  Args:
    segments: A list of `RaggedTensor`s with the tokens of the input segments.
      All elements must have the same dtype (int32 or int64), same rank, and
      same dimension 0 (namely batch size). Slice `segments[i][j, ...]`
      contains the tokens of the i-th input segment to the j-th example in the
      batch.
    start_of_sequence_id: a python int or scalar Tensor containing the id used
      to denote the start of a sequence (e.g. `[CLS]` token in BERT
      terminology).
    end_of_segment_id: a python int or scalar Tensor containing the id used to
      denote end of a segment (e.g. the `[SEP]` token in BERT terminology).

  Returns:
    a tuple of (combined_segments, segment_ids), where:

    combined_segments: A `RaggedTensor` with segments combined and special
      tokens inserted.
    segment_ids:  A `RaggedTensor` w/ the same shape as `combined_segments`
      and containing int ids for each item detailing the segment that they
      correspond to.
  """

  # Create special tokens ([CLS] and [SEP]) that will be combined with the
  # segments
  if len(segments) <= 0:
    raise ValueError("`segments` must be a nonempty list.")
  segment_dtype = segments[0].dtype
  if segment_dtype not in (dtypes.int32, dtypes.int64):
    raise ValueError("`segments` must have elements with dtype of int32 or " +
                     "int64")

  start_of_sequence_id = ops.convert_to_tensor(
      start_of_sequence_id, dtype=segment_dtype)
  end_of_segment_id = ops.convert_to_tensor(
      end_of_segment_id, dtype=segment_dtype)

  start_sequence_id = math_ops.cast(start_of_sequence_id, segment_dtype)
  end_segment_id = math_ops.cast(end_of_segment_id, segment_dtype)
  start_seq_tokens = array_ops.tile([start_sequence_id], [segments[0].nrows()])
  end_segment_tokens = array_ops.tile([end_segment_id], [segments[0].nrows()])
  for i in range(segments[0].ragged_rank):
    start_seq_tokens = array_ops.expand_dims(start_seq_tokens, 1)
    end_segment_tokens = array_ops.expand_dims(end_segment_tokens, 1)
  special_token_segment_template = array_ops.ones_like(start_seq_tokens)

  # Combine all segments w/ special tokens
  segments_to_combine = [start_seq_tokens]
  for seg in segments:
    segments_to_combine.append(seg)
    segments_to_combine.append(end_segment_tokens)
  segments_combined = array_ops.concat(segments_to_combine, 1)

  # Create the segment ids, making sure to account for special tokens.
  segment_ids_to_combine = []
  segment_ids_to_combine.append(special_token_segment_template * 0)
  for i, item in enumerate(segments):
    # Add segment id
    segment_id = array_ops.ones_like(item) * i
    segment_ids_to_combine.append(segment_id)

    # Add for SEP
    special_token_segment_id = special_token_segment_template * i
    segment_ids_to_combine.append(special_token_segment_id)

  segment_ids = array_ops.concat(segment_ids_to_combine, 1)
  return segments_combined, segment_ids


def concatenate_segments(segments):
  """Concatenate input segments for a model's input sequence.

  `concatenate_segments` combines the tokens of one or more input segments to a
  single sequence of token values and generates matching segment ids.
  `concatenate_segments` can follow a `Trimmer`, who limit segment lengths and
  emit `RaggedTensor` outputs, and can be followed up by `ModelInputPacker`.

  `concatenate_segments` first flattens and combines a list of one or more
  segments
  (`RaggedTensor`s of n dimensions) together along the 1st axis, then packages
  any special tokens  into a final n dimensional `RaggedTensor`.

  And finally `concatenate_segments` generates another `RaggedTensor` (with the
  same rank as the final combined `RaggedTensor`) that contains a distinct int
  id for each segment.

  Example usage:

  ```
  segment_a = [[1, 2],
               [3, 4,],
               [5, 6, 7, 8, 9]]

  segment_b = [[10, 20,],
               [30, 40, 50, 60,],
               [70, 80]]
  expected_combined, expected_ids = concatenate_segments([segment_a, segment_b])

  # segment_a and segment_b have been concatenated as is.
  expected_combined=[
   [1, 2, 10, 20],
   [3, 4, 30, 40, 50, 60],
   [5, 6, 7, 8, 9, 70, 80],
  ]

  # ids describing which items belong to which segment.
  expected_ids=[
   [0, 0, 1, 1],
   [0, 0, 1, 1, 1, 1],
   [0, 0, 0, 0, 0, 1, 1]]
  ```

  Args:
    segments: A list of `RaggedTensor`s with the tokens of the input segments.
      All elements must have the same dtype (int32 or int64), same rank, and
      same dimension 0 (namely batch size). Slice `segments[i][j, ...]`
      contains the tokens of the i-th input segment to the j-th example in the
      batch.

  Returns:
    a tuple of (combined_segments, segment_ids), where:

    combined_segments: A `RaggedTensor` with segments combined and special
      tokens inserted.
    segment_ids:  A `RaggedTensor` w/ the same shape as `combined_segments`
      and containing int ids for each item detailing the segment that they
      correspond to.
  """

  # Create special tokens ([CLS] and [SEP]) that will be combined with the
  # segments
  if len(segments) <= 0:
    raise ValueError("`segments` must be a nonempty list.")
  segment_dtype = segments[0].dtype
  if segment_dtype not in (dtypes.int32, dtypes.int64):
    raise ValueError("`segments` must have elements with dtype of int32 or " +
                     "int64")

  # Combine all segments.
  segments_to_combine = []
  for seg in segments:
    segments_to_combine.append(seg)
  segments_combined = array_ops.concat(segments_to_combine, 1)

  # Create the segment ids, making sure to account for special tokens.
  segment_ids_to_combine = []
  for i, item in enumerate(segments):
    # Add segment id
    segment_id = array_ops.ones_like(item) * i
    segment_ids_to_combine.append(segment_id)

  segment_ids = array_ops.concat(segment_ids_to_combine, 1)
  return segments_combined, segment_ids
