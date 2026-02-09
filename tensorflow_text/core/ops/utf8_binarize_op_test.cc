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
#include "tensorflow/core/framework/attr_value.proto.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(Utf8BinarizeOpTest, ShapeFn) {
  // Utf8Binarize(input_tokens) -> [token, binarization_bits]
  ShapeInferenceTestOp op("TFText>Utf8Binarize");
  (*op.node_def.mutable_attr())["word_length"].set_i(3);
  (*op.node_def.mutable_attr())["bits_per_char"].set_i(4);
  (*op.node_def.mutable_attr())["replacement_char"].set_i(14);
  INFER_OK(op, "?", "[?, 12]");
  INFER_OK(op, "[?]", "[?, 12]");
  INFER_OK(op, "[4]", "[4, 12]");
  INFER_ERROR("Shape must be rank 1", op, "[]");
  INFER_ERROR("Shape must be rank 1", op, "[1,2]");
  INFER_ERROR("Shape must be rank 1", op, "[?,?]");
}

}  // namespace
}  // namespace tensorflow
