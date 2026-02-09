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

TEST(SentenceFragmenterV2OpTest, ShapeFn) {
  // SentenceFragmenterV2(input_values)
  //     -> [starts, ends, props, punc_token, row_lengths]
  ShapeInferenceTestOp op("SentenceFragmentsV2");
  INFER_OK(op, "?", "[?];[?];[?];[?];[?]");
  INFER_OK(op, "[?]", "[?];[?];[?];[?];[?]");
  INFER_ERROR("Shape must be rank 1", op, "[]");
  INFER_ERROR("Shape must be rank 1", op, "[1,2]");
  INFER_ERROR("Shape must be rank 1", op, "[?,?]");
}

}  // namespace
}  // namespace tensorflow
