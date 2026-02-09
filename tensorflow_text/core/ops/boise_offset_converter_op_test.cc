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

TEST(OffsetsToBoiseTagsOpTest, ShapeFn) {
  // OffsetsToBoiseTagsOp(input_token_begin_offsets,
  //                      input_token_end_offsets,
  //                      input_span_begin_offsets,
  //                      input_span_end_offsets,
  //                      input_span_type,
  //                      input_token_begin_row_splits,
  //                      input_token_end_row_splits,
  //                      input_span_begin_row_splits,
  //                      input_span_end_row_splits,
  //                      input_span_type_row_splits,
  //                      input_use_strict_boundary_mode)
  //   -> [output_boise_tags]
  ShapeInferenceTestOp op("TFText>OffsetsToBoiseTags");
  INFER_OK(op, "[?];[?];[?];[?];[?];[?];[?];[?];[?];[?];[?]", "[?]");
  INFER_OK(op, "[5];[5];[2];[2];[2];[3];[3];[3];[3];[3];[?]", "[5]");
  INFER_ERROR("Shape must be rank 1", op,
              "[];[1];[1];[1];[?];[?];[?];[?];[?];[?];[?]");
  INFER_ERROR("Shape must be rank 1", op,
              "[1,2];[1];[1];[1];[?];[?];[?];[?];[?];[?];[?]");
  INFER_ERROR("Shape must be rank 1", op,
              "[?,?];[1];[1];[1];[?];[?];[?];[?];[?];[?];[?]");
}

TEST(BoiseTagsToOffsetsOpTest, ShapeFn) {
  // BoiseTagsToOffsetsOp(input_token_begin_offsets,
  //                      input_token_end_offsets,
  //                      input_boise_tags,
  //                      input_token_begin_row_splits,
  //                      input_token_end_row_splits,
  //                      input_boise_tags_row_splits)
  //   -> [output_span_begin_offsets, output_span_end_offsets, output_span_type,
  //   output_row_splits]
  ShapeInferenceTestOp op("TFText>BoiseTagsToOffsets");
  INFER_OK(op, "[?];[?];[?];[?];[?];[?]", "[?];[?];[?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "[];[5];[5];[2];[2];[2]");
  INFER_ERROR("Shape must be rank 1", op, "[3,4];[5];[5];[2];[2];[2]");
  INFER_ERROR("Shape must be rank 1", op, "[?,?];[5];[5];[2];[2];[2]");
}

}  // namespace
}  // namespace tensorflow
