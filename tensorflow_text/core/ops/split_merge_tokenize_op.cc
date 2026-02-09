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
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
namespace text {

using shape_inference::DimensionHandle;
using shape_inference::InferenceContext;
using shape_inference::ShapeHandle;

absl::Status SplitMergeTokenizeWithOffsetsShapeFn(InferenceContext* c);

REGISTER_OP("SplitMergeTokenizeWithOffsets")
    .Input("input_values: string")
    .Input("labels: int32")
    .Input("row_splits: int32")
    .Attr("force_split_at_break_character: bool = true")
    .Output("output_values: string")
    .Output("output_row_splits: int64")
    .Output("start_values: int64")
    .Output("limit_values: int64")
    .SetShapeFn(SplitMergeTokenizeWithOffsetsShapeFn)
    .Doc(R"doc(
  Segment input string according to the given split(0)/merge(1) labels of each
  character in the input string.

  ### Example:

  ```python
  >>> strs = ["Itis",
              "thanksgiving"]
  >>> labels = [0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1]
  >>> row_splits = [0, 4, 16]
  >>> words, row_splits, start, end = create_token(strs, labels)
  >>> RaggedTensor.from_row_splits(words, row_splits)
  [['It', 'is'], ['thanks', 'giving']]
  >>> RaggedTensor.from_row_splits(start, row_splits)
  start = [[[0, 2], [0, 6]]]
  >>> RaggedTensor.from_row_splits(end, row_splits)
  end = [[[2, 4], [6, 11]]]
  ```

  Args:
    input_values: 1D Tensor of strings to tokenize with.
    labels: 1D Tensor of split merge labels.
    row_splits: row_splits together with labels forms a 2D ragged tensor, the
      ith row corresponds to the split/merge labels for input_values[i].
    force_split_at_break_character: bool indicates whether to force start a
      new word after seeing a ICU defined whitespace character.

  Returns:
    * output_values: 1D tensor containing the tokens for all input strings.
      A 2D RaggedTensor can be constructed from this and output_row_splits.
    * output_row_splits: 1D tensor containing row split offsets indicating the
      start and end offsets in the output values for each input string.
    * start_values: 1D tensor containing the inclusive start byte offset for
      each token in all input strings.  Corresponds 1:1 with output_values.
      A 2D RaggedTensor can be constructed from this and output_row_splits.
    * limit_values: 1D tensor containing the exclusive end byte offset for
      each token in all input strings.  Corresponds 1:1 with output_values.
      A 2D RaggedTensor can be constructed from this and output_row_splits.
)doc");

absl::Status SplitMergeTokenizeWithOffsetsShapeFn(InferenceContext* c) {
  ShapeHandle input_values = c->input(0);
  ShapeHandle labels = c->input(1);
  ShapeHandle row_splits = c->input(2);
  TF_RETURN_IF_ERROR(c->WithRank(input_values, 1, &input_values));
  TF_RETURN_IF_ERROR(c->WithRank(labels, 1, &labels));
  TF_RETURN_IF_ERROR(c->WithRank(row_splits, 1, &row_splits));
  DimensionHandle num_input_values = c->Dim(input_values, 0);
  c->set_output(0, c->UnknownShapeOfRank(1));  // output_values
  DimensionHandle num_splits;
  TF_RETURN_IF_ERROR(c->Add(num_input_values, 1, &num_splits));
  c->set_output(1, c->Vector(num_splits));  // row_splits
  c->set_output(2, c->UnknownShapeOfRank(1));  // start_values
  c->set_output(3, c->UnknownShapeOfRank(1));  // limit_values
  return absl::OkStatus();
}

}  // namespace text
}  // namespace tensorflow
