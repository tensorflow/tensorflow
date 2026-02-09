// Copyright 2025 TF.Text Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

absl::Status WordpieceTokenizeWithOffsetsShapeFn(InferenceContext* c);

REGISTER_OP("WordpieceTokenizeWithOffsets")
    .Input("input_values: string")
    .Input("vocab_lookup_table: resource")
    .Attr("suffix_indicator: string")
    .Attr("max_bytes_per_word: int")
    .Attr("max_chars_per_token: int = 0")
    .Attr("use_unknown_token: bool")
    .Attr("unknown_token: string")
    .Attr("split_unknown_characters: bool = false")
    .Attr("output_row_partition_type: {'row_lengths', 'row_splits'}"
          " = 'row_lengths'")
    .Output("output_values: string")
    .Output("output_row_lengths: int64")
    .Output("start_values: int64")
    .Output("limit_values: int64")
    .SetShapeFn(WordpieceTokenizeWithOffsetsShapeFn)
    .Doc(R"doc(
  Tokenizes tokens into sub-word pieces based off of a vocabulary.

  `wordpiece_tokenize_with_offsets` returns the relative offsets.

  ### Example:

  ```python
  >>> tokens = ['don', '\'t', 'treadness']
  >>> wordpiece, row_lengths, start, end = wordpiece_tokenize_with_offset(
  ...     tokens, vocab, '##', 100, False, '')
  >>> RaggedTensor.from_row_lengths(wordpiece, row_lengths)
  [['don', '\'', 't'], ['tread', '##ness']]
  >>> RaggedTensor.from_row_lengths(start, row_lengths)
  start = [[[0, 3, 4], [0, 5]]]
  >>> RaggedTensor.from_row_lengths(end, row_lengths)
  end = [[[3, 4, 5], [5, 10]]]
  ```

  Args:
    input_values: 1D Tensor of strings to tokenize with.
    vocab_lookup_table: Resource tensor for a lookup table implementing the
        LookupInterface.
    suffix_indicator: Characters prepended to a wordpiece to
      indicate that it is a suffix to another subword.
    max_bytes_per_word: Max size of input token.
    max_chars_per_token: Max size of output tokens. A non-positive value
      means the max size is not known.
    use_unknown_token: Whether unknown_token should be used.
    unknown_token: The value to use when an unknown token is found.
    split_unknown_characters: Whether individual unknown unicode characters
      should be split out as subtokens.
    output_row_partition_type: Indicates what row-partitioning tensor should
      be returned by the op.  If this is set to 'row_splits', then the
      `output_row_lengths` output will contain row-splits instead of
      row-lengths.

  Returns:
    * output_values: 1D tensor containing the wordpieces for all input strings.
      A 2D RaggedTensor can be constructed from this and output_row_lengths.
    * output_row_lengths: 1D int tensor indicating the number of wordpieces
      corresponding with each input string.  If output_row_partition_type is
      row_splits, then this will contain row split offsets instead.
    * start_values: 1D tensor containing the inclusive start byte offset for
      each wordpiece in all input strings.  Corresponds 1:1 with output_values.
      A 2D RaggedTensor can be constructed from this and output_row_lengths.
    * limit_values: 1D tensor containing the exclusive end byte offset for
      each wordpiece in all input strings.  Corresponds 1:1 with output_values.
      A 2D RaggedTensor can be constructed from this and output_row_lengths.
)doc");

absl::Status WordpieceTokenizeWithOffsetsShapeFn(InferenceContext* c) {
  ShapeHandle input_values = c->input(0);
  ShapeHandle vocab_lookup_table = c->input(1);
  string output_row_partition_type;
  TF_RETURN_IF_ERROR(c->WithRank(input_values, 1, &input_values));
  TF_RETURN_IF_ERROR(c->WithRank(vocab_lookup_table, 0, &vocab_lookup_table));
  TF_RETURN_IF_ERROR(c->GetAttr("output_row_partition_type",
                                &output_row_partition_type));
  DimensionHandle num_input_values = c->Dim(input_values, 0);
  c->set_output(0, c->UnknownShapeOfRank(1));  // output_values
  if (output_row_partition_type == "row_lengths") {
    c->set_output(1, c->Vector(num_input_values));  // row_lengths
  } else {
    DimensionHandle num_splits;
    TF_RETURN_IF_ERROR(c->Add(num_input_values, 1, &num_splits));
    c->set_output(1, c->Vector(num_splits));  // row_splits
  }
  c->set_output(2, c->UnknownShapeOfRank(1));  // start_values
  c->set_output(3, c->UnknownShapeOfRank(1));  // limit_values
  return absl::OkStatus();
}

}  // namespace tensorflow
