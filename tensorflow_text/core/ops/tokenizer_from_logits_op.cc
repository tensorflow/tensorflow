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
namespace text {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

absl::Status TokenizerFromLogitsShapeFn(InferenceContext* c);

REGISTER_OP("TokenizerFromLogits")
    .Input("strings: string")
    .Input("logits: float")
    .Input("force_split_at_break_character: bool")
    .Output("output_values: string")
    .Output("row_splits: int64")
    .Output("start_values: int64")
    .Output("limit_values: int64")
    .SetShapeFn(TokenizerFromLogitsShapeFn)
    .Doc(R"doc(
  Segment input string according to the given split(0)/merge(1) labels of each
  character in the input string.

  ### Example:

  ```python
  >>> strings = ["IloveFlume!", "and tensorflow"])
  >>> labels = [
      [
          # I
          0,
          # love
          0, 1, 1, 1,
          # Flume
          0, 1, 1, 1, 1,
          # !
          0,
          # paddings
          0, 0, 0
      ], [
          # and
          0, 1, 1,
          # ' '
          1,
          # tensorflow
          0, 1, 1, 1, 1, 1, 1, 1, 1, 1
      ]]
  >>> tokenizer = TokenizerFromLogits()
  >>> token_values, rows_splits, start_values, limit_values = (
          gen_tokenizer_from_logits.tokenizer_from_logits(strings, labels)
  >>> RaggedTensor.from_row_splits(token_values, row_splits)
  [["I", "love", "Flume", "!"], ["and", "tensorflow"]]
  >>> RaggedTensor.from_row_splits(start_values, row_splits)
  >>> [[0, 1, 5, 10], [0, 4]]
  >>> RaggedTensor.from_row_splits(limit_values, row_splits)
  >>> [[1, 5, 10, 11], [3, 14]]
  ```

  Args:
    strings: 1D Tensor of strings to tokenize with.
    logits: 3D Tensor; logits[i,j,0] is the logit for the split action for j-th
      character of strings[i].  logits[i,j,1] is the logit for the merge action
      for that same character.  For each character, we pick the action with the
      greatest logit.  Split starts a new word at this character and merge adds
      this character to the previous word.  The shape of this tensor should be
      (n, m, 2) where n is the number of strings, and m is greater or equal with
      the number of characters from each strings[i].  As the elements of the
      strings tensor may have different lengths (in UTF-8 chars), padding may be
      required to get a dense vector; for each row, the extra (padding) pairs of
      logits are ignored.
    force_split_at_break_character: bool scalar, indicates whether to force
      start a new word after seeing an ICU defined whitespace character.

  Returns:
    * token_values: 1D tensor containing the tokens for all input strings.
      A 2D RaggedTensor can be constructed from this and row_splits.
    * row_splits: 1D tensor containing row split offsets indicating the
      start and end offsets in the output values for each input string.
    * start_values: 1D tensor containing the inclusive start byte offset for
      each token in all input strings.  Corresponds 1:1 with output_values.
      A 2D RaggedTensor can be constructed from this and row_splits.
    * limit_values: 1D tensor containing the exclusive end byte offset for
      each token in all input strings.  Corresponds 1:1 with output_values.
      A 2D RaggedTensor can be constructed from this and row_splits.
)doc");

absl::Status TokenizerFromLogitsShapeFn(InferenceContext* c) {
  ShapeHandle strings = c->input(0);
  ShapeHandle logits = c->input(1);
  ShapeHandle force_split_at_break_character = c->input(2);
  TF_RETURN_IF_ERROR(c->WithRank(strings, 1, &strings));
  TF_RETURN_IF_ERROR(c->WithRank(logits, 3, &logits));
  TF_RETURN_IF_ERROR(c->WithRank(force_split_at_break_character, 0,
                                 &force_split_at_break_character));
  DimensionHandle num_strings = c->Dim(strings, 0);
  c->set_output(0, c->UnknownShapeOfRank(1));  // output_values
  DimensionHandle num_splits;
  TF_RETURN_IF_ERROR(c->Add(num_strings, 1, &num_splits));
  c->set_output(1, c->Vector(num_splits));  // row_splits
  c->set_output(2, c->UnknownShapeOfRank(1));  // start_values
  c->set_output(3, c->UnknownShapeOfRank(1));  // limit_values
  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow
