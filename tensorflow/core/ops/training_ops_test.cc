/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");

You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

// Used for testing the grad+indices handling for SparseApplyXYZ tests.
static void TestGradAndIndicesErrorHandling(ShapeInferenceTestOp op,
                                            string shape_spec_middle,
                                            string shape_spec_end = "") {
  auto shape_spec = [&shape_spec_middle, shape_spec_end](
      const char* var_spec, const char* grad_indices_spec) {
    return strings::StrCat(var_spec, ";", shape_spec_middle, ";",
                           grad_indices_spec, shape_spec_end);
  };

  // mismatch between grad[1] and var[1].
  INFER_ERROR("Dimension 1 in both shapes must be equal", op,
              shape_spec("[?,1]", "[?,2];[?]").c_str());
  // grad[0] and indices[0] must match.
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op,
              shape_spec("?", "[2,?];[1]").c_str());
  // grad is wrong rank.
  INFER_ERROR("must be equal rank", op, shape_spec("[1]", "[?,2];[?]").c_str());
  // indices is wrong rank.
  INFER_ERROR("Shape must be rank 1 but is rank 2", op,
              shape_spec("[?]", "[?];[1,2]").c_str());
}

TEST(TrainingOpsTest, ApplyGradientDescent_ShapeFn) {
  ShapeInferenceTestOp op("ApplyGradientDescent");

  // Output is a merge of inputs 0 and 2 (var and delta).
  INFER_OK(op, "[1,?];[];[?,2]", "[d0_0,d2_1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[];[2]");

  // alpha must be a scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[?];?");
}

TEST(TrainingOpsTest, ApplyProximalGradientDescent_ShapeFn) {
  ShapeInferenceTestOp op("ApplyProximalGradientDescent");

  // Output is a merge of inputs 0 and 4 (var and delta).
  INFER_OK(op, "[1,?];[];[];[];[?,2]", "[d0_0,d4_1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[];[];[];[2]");

  // alpha, l1, and l2 must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyProximalGradientDescent_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyProximalGradientDescent");

  // Output is a merge of inputs 0 (var) and the non-indices part of 4 (delta).
  INFER_OK(op, "[1,?];[];[];[];[?,2];[3]", "[d0_0,d4_1]");

  TestGradAndIndicesErrorHandling(op, "[];[];[]");

  // alpha, l1, and l2 must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?");
}

TEST(TrainingOpsTest, ApplyAdadelta_ShapeFn) {
  ShapeInferenceTestOp op("ApplyAdadelta");

  // Output is a merge of inputs 0, 1, 2, and 6 (var, accum, accum_update,
  // grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[?,?,?,4]",
           "[d0_0,d1_1,d2_2,d6_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[1];[];[];[];[2]");

  // lr, rho, and episilon must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyAdadelta_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyAdadelta");

  // Output is a merge of inputs 0, 1, 2, and non-indices part of 6 (var, accum,
  // accum_update, grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[?,?,?,4];?",
           "[d0_0,d1_1,d2_2,d6_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[1];?");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[1];?");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[?,1];[];[];[];[?,2];?");

  TestGradAndIndicesErrorHandling(op, "?;?;?;?;?");

  // lr, rho, and episilon must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?");
}

TEST(TrainingOpsTest, ApplyAdagrad_ShapeFn) {
  ShapeInferenceTestOp op("ApplyAdagrad");

  // Output is a merge of inputs 0, 1, and 3 (var, accum, grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[?,?,3]", "[d0_0,d1_1,d3_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[];[2]");

  // lr must be a scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyAdagrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyAdagrad");

  // Output is a merge of inputs 0, 1, and non-indices part of 3 (var, accum,
  // grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[?,?,3];?", "[d0_0,d1_1,d3_2]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,2];[];[?,1];?");
  INFER_ERROR("Shapes must be equal rank, but are 2 and 3", op,
              "[?,1];[?,1];[];[?,?,2];?");

  TestGradAndIndicesErrorHandling(op, "?;?");

  // lr must be a scalar.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?");
}

TEST(TrainingOpsTest, ApplyProximalAdagrad_ShapeFn) {
  ShapeInferenceTestOp op("ApplyProximalAdagrad");

  // Output is a merge of inputs 0, 1, and 5 (var, accum, grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[];[];[?,?,3]", "[d0_0,d1_1,d5_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[];[];[];[2]");

  // lr, l1, and l2 must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyProximalAdagrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyProximalAdagrad");

  // Output is a merge of inputs 0, 1, and the non-indices part of 5 (var,
  // accum, grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[];[];[?,?,3];?", "[d0_0,d1_1,d5_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[];[];[?,1];?");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[];[];[];[?,2];?");

  TestGradAndIndicesErrorHandling(op, "?;?;?;?");

  // lr, l1, and l2 must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?");
}

TEST(TrainingOpsTest, ApplyFtrl_ShapeFn) {
  ShapeInferenceTestOp op("ApplyFtrl");

  // Output is a merge of inputs 0, 1, 2, and 3 (var, accum, linear, grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[?,?,?,4];[];[];[];[]",
           "[d0_0,d1_1,d2_2,d3_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[1];[];[];[];[]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[1];[];[];[];[]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[1];[2];[];[];[];[]");

  // lr, l1, l2, and lr_power must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;[?];?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;?;[?]");
}

TEST(TrainingOpsTest, SparseApplyFtrl_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyFtrl");

  // Output is a merge of inputs 0, 1, 2, and non-indices part of 3 (var, accum,
  // linear, grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[?,?,?,4];?;[];[];[];[]",
           "[d0_0,d1_1,d2_2,d3_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[?,1];?;[];[];[];[]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[?,1];?;[];[];[];[]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[?,1];[?,2];?;[];[];[];[]");

  TestGradAndIndicesErrorHandling(op, "?;?", ";?;?;?;?");

  // lr, l1, l2, and lr_power must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;?;[?];?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;?;?;[?]");
}

TEST(TrainingOpsTest, ApplyMomentum_ShapeFn) {
  ShapeInferenceTestOp op("ApplyMomentum");

  // Output is a merge of inputs 0, 1, and 3 (var, accum, grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[?,?,3];[]", "[d0_0,d1_1,d3_2]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[];[1];[]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[];[2];[]");

  // lr and momentum must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?]");
}

TEST(TrainingOpsTest, SparseApplyMomentum_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyMomentum");

  // Output is a merge of inputs 0, 1, and non-indices part of 3 (var, accum,
  // grad).
  INFER_OK(op, "[1,?,?];[?,2,?];[];[?,?,3];?;[]", "[d0_0,d1_1,d3_2]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,2];[];[?,1];?;[]");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[];[?,2];?;[]");

  TestGradAndIndicesErrorHandling(op, "?;?", ";?");

  // lr and momentum must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?]");
}

TEST(TrainingOpsTest, ApplyAdam_ShapeFn) {
  ShapeInferenceTestOp op("ApplyAdam");

  // Output is a merge of inputs 0, 1, 2, and 9 (var, m, v, and grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[];[];[];[?,?,?,4]",
           "[d0_0,d1_1,d2_2,d9_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[1];[];[];[];[];[];[];[2]");

  // beta1_power, beta2_power, lr, beta1, beta2, and epsilon must be scalars.
  const char err[] = "Shape must be rank 0 but is rank 1";
  INFER_ERROR(err, op, "?;?;?;[?];?;?;?;?;?;?");
  INFER_ERROR(err, op, "?;?;?;?;[?];?;?;?;?;?");
  INFER_ERROR(err, op, "?;?;?;?;?;[?];?;?;?;?");
  INFER_ERROR(err, op, "?;?;?;?;?;?;[?];?;?;?");
  INFER_ERROR(err, op, "?;?;?;?;?;?;?;[?];?;?");
  INFER_ERROR(err, op, "?;?;?;?;?;?;?;?;[?];?");
}

TEST(TrainingOpsTest, ApplyRMSProp_ShapeFn) {
  ShapeInferenceTestOp op("ApplyRMSProp");

  // Output is a merge of inputs 0, 1, 2, and 7 (var, ms, mom, and grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[];[?,?,?,4]",
           "[d0_0,d1_1,d2_2,d7_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[];[1]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[1];[];[];[];[];[2]");

  // lr, rho, momentum, and epsilon must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;[?];?");
}

TEST(TrainingOpsTest, SparseApplyRMSProp_ShapeFn) {
  ShapeInferenceTestOp op("SparseApplyRMSProp");

  // Output is a merge of inputs 0, 1, 2, and the non-indices part of 7 (var,
  // ms, mom, and grad).
  INFER_OK(op, "[1,?,?,?];[?,2,?,?];[?,?,3,?];[];[];[];[];[?,?,?,4];?",
           "[d0_0,d1_1,d2_2,d7_3]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[2];[1];[];[];[];[];[1];?");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 1 and 2", op,
              "[1];[1];[2];[];[];[];[];[1];?");
  INFER_ERROR("Dimension 1 in both shapes must be equal, but are 1 and 2", op,
              "[?,1];[?,1];[?,1];[];[];[];[];[?,2];?");

  TestGradAndIndicesErrorHandling(op, "?;?;?;?;?;?");

  // lr, rho, momentum, and epsilon must be scalars.
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;[?];?;?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;[?];?;?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;[?];?;?;?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;?;?;?;?;[?];?;?");
}

}  // end namespace tensorflow
