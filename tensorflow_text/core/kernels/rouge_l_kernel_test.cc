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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(RougeLFMeasureOpTest, ShapeFn) {
  ShapeInferenceTestOp op("RougeL");

  INFER_OK(op, "[?];[3];[?];[3];[]", "[2];[2];[2]");
  INFER_OK(op, "[5];[3];[?];[3];[]", "[2];[2];[2]");
  INFER_OK(op, "[?];[3];[8];[3];[]", "[2];[2];[2]");
  INFER_OK(op, "[5];[3];[8];[3];[]", "[2];[2];[2]");
  INFER_OK(op, "[5];[3];[8];?;[]", "[2];[2];[2]");
  INFER_OK(op, "[5];?;[8];[3];[]", "[2];[2];[2]");
  INFER_OK(op, "[5];[?];[8];[?];[]", "[?];[?];[?]");
  INFER_OK(op, "?;?;?;?;?", "[?];[?];[?]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 3 and 2.", op,
              "[5];[3];[8];[2];[]");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op,
              "[5];[3];[8];[3];[1]");
}

}  // namespace
}  // namespace tensorflow
