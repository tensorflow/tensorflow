/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace tensorflow {

TEST(UniformQuantizedOpsTest, UniformQuantizedDotShapeInference) {
  ShapeInferenceTestOp op("UniformQuantizedDot");
  INFER_OK(op, "[4,2];[2,3];[];[];[];[];[];[]", "[d0_0,d1_1]");
  INFER_OK(op, "[4,2];[2,3];[];[];[3];[3];[];[]", "[d0_0,d1_1]");
  INFER_OK(op, "[4,2];[2,3];[];[];[3];[3];[3];[3]", "[d0_0,d1_1]");

  // Inner dim does not match.
  INFER_ERROR("", op, "[4,2];[6,3];[];[];[];[];[];[]");
  // lhs scales and zero_points must be scalar tensors.
  INFER_ERROR("", op, "[4,2];[2,3];[4];[4];[];[];[];[]");
  // scales and zero_points must have same rank.
  INFER_ERROR("scales and zero_points must have same rank.", op,
              "[4,2];[2,3];[];[];[3];[];[];[]");
  // If rhs scales and zero_points are not scalar tensors, both of their
  // dim_size[0] must be equal to rhs.dim_size[1].
  INFER_ERROR("", op, "[4,2];[2,3];[];[];[6];[6];[];[]");
  // If output scales and zero_points are not scalar tensors, both of their
  // dim_size[0] must be equal to rhs.dim_size[1].
  INFER_ERROR("", op, "[4,2];[2,3];[];[];[];[];[6];[6]");
}

TEST(UniformQuantizedOpsTest, UniformQuantizedDotHybridShapeInference) {
  ShapeInferenceTestOp op("UniformQuantizedDotHybrid");
  INFER_OK(op, "[4,2];[2,3];[];[]", "[d0_0,d1_1]");
  INFER_OK(op, "[4,2];[2,3];[3];[3]", "[d0_0,d1_1]");

  // Inner dim does not match.
  INFER_ERROR("", op, "[4,2];[6,3];[];[]");
  // scales and zero_points must have same rank.
  INFER_ERROR("scales and zero_points must have same rank.", op,
              "[4,2];[2,3];[3];[]");
  // If rhs scales and zero_points are not scalar tensors, both of their
  // dim_size[0] must be equal to rhs.dim_size[1].
  INFER_ERROR("", op, "[4,2];[2,3];[6];[6]");
}

}  // namespace tensorflow
