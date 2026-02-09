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

#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(FastWordpieceTokenizeWithOffsetsOpTest, ShapeFn) {
  // FastWordpieceTokenizeWithOffsets(input_values, wp_model) ->
  //     [output_values, output_ids, output_row_splits, start_values,
  //      end_values]
  ShapeInferenceTestOp op("FastWordpieceTokenizeWithOffsets");

  INFER_OK(op, "?;?", "[?];[?];[?];[?];[?]");
  INFER_OK(op, "[?];?", "[?];[?];[?];[?];[?]");
  INFER_OK(op, "[5];?", "[?];[?];[6];[?];[?]");
  INFER_OK(op, "[6];[?]", "[?];[?];[7];[?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "[];?");
  INFER_ERROR("Shape must be rank 1", op, "[1,2];?");
  INFER_ERROR("Shape must be rank 1", op, "?;[]");
  INFER_ERROR("Shape must be rank 1", op, "?;[?,?]");
}

TEST(FastWordpieceDetokenizeOpTest, ShapeFn) {
  // FastWordpieceTokenizeWithOffsets(input_values, input_row_splits, wp_model)
  //     -> [output_values]
  ShapeInferenceTestOp op("TFText>FastWordpieceDetokenize");
  INFER_OK(op, "?;?;?", "[?]");
  INFER_OK(op, "[?];[?];?", "[?]");
  INFER_OK(op, "[5];[?];?", "[?]");
  INFER_OK(op, "[6];[?];[?]", "[?]");
  INFER_ERROR("Shape must be rank 1", op, "[];?;?");
  INFER_ERROR("Shape must be rank 1", op, "[1,2];?;?");
  INFER_ERROR("Shape must be rank 1", op, "?;[];?");
  INFER_ERROR("Shape must be rank 1", op, "?;[?,?];?");
}

}  // namespace
}  // namespace tensorflow
