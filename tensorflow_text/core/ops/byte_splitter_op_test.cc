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

TEST(ByteSplitterWithOffsetsOpTest, ShapeFn) {
  // ByteSplitWithOffsets(input_values)
  //     -> [bytes, row_splits, start_offsets, end_offsets]
  ShapeInferenceTestOp op("TFText>ByteSplitWithOffsets");
  INFER_OK(op, "?", "[?];[?];[?];[?]");
  INFER_OK(op, "[?]", "[?];[?];[?];[?]");
  INFER_OK(op, "[4]", "[?];[5];[?];[?]");
  INFER_ERROR("shape must be rank 1", op, "[]");
  INFER_ERROR("shape must be rank 1", op, "[1,2]");
  INFER_ERROR("shape must be rank 1", op, "[?,?]");
}

TEST(ByteSpliByOffsetsOpTest, ShapeFn) {
  // ByteSplitWithOffsets(input_values, starts, ends, input_row_splits)
  //     -> [output_values, output_row_splits]
  ShapeInferenceTestOp op("TFText>ByteSplitByOffsets");
  INFER_OK(op, "?;[?];[?];[?]", "[?];[?]");
  INFER_OK(op, "[?];[?];[?];[?]", "[?];[?]");
  INFER_OK(op, "[?];[?];[?];[5]", "[?];[5]");
  INFER_OK(op, "[?];[3];[?];[?]", "[3];[?]");
  INFER_ERROR("shape must be rank 1", op, "[];[?];[?];[?]");
  INFER_ERROR("shape must be rank 1", op, "[1,2];[?];[?];[?]");
  INFER_ERROR("shape must be rank 1", op, "[?,?];[?];[?];[?]");
}

}  // namespace
}  // namespace tensorflow
