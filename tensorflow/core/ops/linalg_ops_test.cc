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

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(LinalgOpsTest, MatrixDeterminant_ShapeFn) {
  ShapeInferenceTestOp op("MatrixDeterminant");
  INFER_OK(op, "?", "?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
  INFER_ERROR("Dimensions must be equal, but are 2 and 1", op, "[1,?,3,4,1,2]");

  INFER_OK(op, "[?,?]", "[]");
  INFER_OK(op, "[1,?]", "[]");
  INFER_OK(op, "[?,1]", "[]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[1,?,3,4,?,?]", "[d0_0,d0_1,d0_2,d0_3]");
  INFER_OK(op, "[1,?,3,4,1,?]", "[d0_0,d0_1,d0_2,d0_3]");
  INFER_OK(op, "[1,?,3,4,?,1]", "[d0_0,d0_1,d0_2,d0_3]");
}

TEST(LinalgOpsTest, UnchangedSquare_ShapeFn) {
  for (const char* op_name : {"Cholesky", "CholeskyGrad", "MatrixInverse"}) {
    ShapeInferenceTestOp op(op_name);

    const string extra_shape = (op.name == "CholeskyGrad" ? ";?" : "");

    INFER_OK(op, "?" + extra_shape, "?");
    INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
                "[1]" + extra_shape);
    INFER_ERROR("Dimensions must be equal, but are 1 and 2", op,
                "[1,2]" + extra_shape);

    INFER_OK(op, "[?,?]" + extra_shape, "[d0_0|d0_1,d0_0|d0_1]");
    INFER_OK(op, "[1,?]" + extra_shape, "[d0_0,d0_0]");
    INFER_OK(op, "[?,1]" + extra_shape, "[d0_1,d0_1]");

    // Repeat previous block of tests with input rank > 2.
    INFER_OK(op, "[5,?,7,?,?]" + extra_shape,
             "[d0_0,d0_1,d0_2,d0_3|d0_4,d0_3|d0_4]");
    INFER_OK(op, "[5,?,7,1,?]" + extra_shape, "[d0_0,d0_1,d0_2,d0_3,d0_3]");
    INFER_OK(op, "[5,?,7,?,1]" + extra_shape, "[d0_0,d0_1,d0_2,d0_4,d0_4]");
  }
}

TEST(LinalgOpsTest, SelfAdjointEig_ShapeFn) {
  ShapeInferenceTestOp op("SelfAdjointEig");
  INFER_OK(op, "?", "?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,2]");

  INFER_OK(op, "[?,?]", "[?,d0_0|d0_1]");
  INFER_OK(op, "[1,?]", "[2,d0_0]");
  INFER_OK(op, "[?,1]", "[2,d0_1]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[5,?,7,?,?]", "[d0_0,d0_1,d0_2,?,d0_3|d0_4]");
  INFER_OK(op, "[5,?,7,1,?]", "[d0_0,d0_1,d0_2,2,d0_3]");
  INFER_OK(op, "[5,?,7,?,1]", "[d0_0,d0_1,d0_2,2,d0_4]");
}

TEST(LinalgOpsTest, SelfAdjointEigV2_ShapeFn) {
  ShapeInferenceTestOp op("SelfAdjointEigV2");
  auto set_compute_v = [&op](bool compute_v) {
    // Test for float32
    TF_ASSERT_OK(NodeDefBuilder("test", "Pack")
                     .Input({{"input", 0, DT_FLOAT}})
                     .Attr("compute_v", compute_v)
                     .Finalize(&op.node_def));

    // Test for float16
    TF_ASSERT_OK(NodeDefBuilder("test", "Pack")
                     .Input({{"input", 0, DT_HALF}})
                     .Attr("compute_v", compute_v)
                     .Finalize(&op.node_def));
  };
  set_compute_v(false);
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,2]");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[3,1,2]");

  INFER_OK(op, "?", "?;[0]");
  INFER_OK(op, "[?,?]", "[d0_0|d0_1];[0]");
  INFER_OK(op, "[1,?]", "[d0_0|d0_1];[0]");
  INFER_OK(op, "[?,1]", "[d0_0|d0_1];[0]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[5,?,7,?,?]", "[d0_0,d0_1,d0_2,d0_3|d0_4];[0]");
  INFER_OK(op, "[5,?,7,1,?]", "[d0_0,d0_1,d0_2,d0_3|d0_4];[0]");
  INFER_OK(op, "[5,?,7,?,1]", "[d0_0,d0_1,d0_2,d0_3|d0_4];[0]");

  set_compute_v(true);
  INFER_OK(op, "?", "?;?");
  INFER_OK(op, "[?,?]", "[d0_0|d0_1];[d0_0|d0_1,d0_0|d0_1]");
  INFER_OK(op, "[1,?]", "[d0_0|d0_1];[d0_0|d0_1,d0_0|d0_1]");
  INFER_OK(op, "[?,1]", "[d0_0|d0_1];[d0_0|d0_1,d0_0|d0_1]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[5,?,7,?,?]",
           "[d0_0,d0_1,d0_2,d0_3|d0_4];[d0_0,d0_1,d0_2,d0_3|d0_4,d0_3|d0_4]");
  INFER_OK(op, "[5,?,7,1,?]",
           "[d0_0,d0_1,d0_2,d0_3|d0_4];[d0_0,d0_1,d0_2,d0_3|d0_4,d0_3|d0_4]");
  INFER_OK(op, "[5,?,7,?,1]",
           "[d0_0,d0_1,d0_2,d0_3|d0_4];[d0_0,d0_1,d0_2,d0_3|d0_4,d0_3|d0_4]");
}

TEST(LinalgOpsTest, MatrixSolve_ShapeFn) {
  ShapeInferenceTestOp op("MatrixSolve");
  INFER_OK(op, "?;?", "?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1];?");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,2];?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[5,?,?];[6]");
  INFER_ERROR("Shapes must be equal rank, but are 0 and 1", op,
              "[5,?];[6,?,?]");

  INFER_OK(op, "[?,?];?", "[d0_0|d0_1,?]");

  // Inputs are [...,M,M] and [...,M,K].  Output is [...,M,K].
  // First test where ... is empty.
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");
  INFER_OK(op, "[?,?];[1,?]", "[d1_0,d1_1]");
  INFER_OK(op, "[1,?];[1,?]", "[d0_0|d1_0,d1_1]");
  INFER_OK(op, "[?,1];[1,?]", "[d0_1|d1_0,d1_1]");
  INFER_OK(op, "[1,1];[?,?]", "[d0_0,d1_1]");
  INFER_OK(op, "[1,1];[1,?]", "[d0_0|d0_1|d1_0,d1_1]");
  // Test with ... being 2-d.
  INFER_OK(op, "[10,?,?,?];[?,20,1,?]", "[d0_0,d1_1,d1_2,d1_3]");
  INFER_OK(op, "[10,?,1,?];[?,20,1,?]", "[d0_0,d1_1,d0_2|d1_2,d1_3]");
  INFER_OK(op, "[10,?,?,1];[?,20,1,?]", "[d0_0,d1_1,d0_3|d1_2,d1_3]");
  INFER_OK(op, "[10,?,1,1];[?,20,?,?]", "[d0_0,d1_1,d0_2,d1_3]");
  INFER_OK(op, "[10,?,1,1];[?,20,1,?]", "[d0_0,d1_1,d0_2|d0_3|d1_2,d1_3]");
}

TEST(LinalgOpsTest, MatrixTriangularSolve_ShapeFn) {
  ShapeInferenceTestOp op("MatrixTriangularSolve");
  INFER_OK(op, "?;?", "?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1];?");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,2];?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[5,?,?];[6]");

  // Inputs are [...,M,M] and [...,M,K].  Output is [...,M,K].
  // First test where ... is empty.
  INFER_OK(op, "[?,?];[?,?]", "[d0_0,d1_1]");
  INFER_OK(op, "[?,?];[1,?]", "[d1_0,d1_1]");
  INFER_OK(op, "[1,?];[1,?]", "[d0_0|d1_0,d1_1]");
  INFER_OK(op, "[?,1];[1,?]", "[d0_1|d1_0,d1_1]");
  INFER_OK(op, "[1,1];[?,?]", "[d0_0,d1_1]");
  INFER_OK(op, "[1,1];[1,?]", "[d0_0|d0_1|d1_0,d1_1]");
  // Test with ... being 2-d.
  INFER_OK(op, "[10,?,?,?];[?,20,1,?]", "[d0_0,d1_1,d1_2,d1_3]");
  INFER_OK(op, "[10,?,1,?];[?,20,1,?]", "[d0_0,d1_1,d0_2|d1_2,d1_3]");
  INFER_OK(op, "[10,?,?,1];[?,20,1,?]", "[d0_0,d1_1,d0_3|d1_2,d1_3]");
  INFER_OK(op, "[10,?,1,1];[?,20,?,?]", "[d0_0,d1_1,d0_2,d1_3]");
  INFER_OK(op, "[10,?,1,1];[?,20,1,?]", "[d0_0,d1_1,d0_2|d0_3|d1_2,d1_3]");
}

TEST(LinalgOpsTest, MatrixSolveLs_ShapeFn) {
  ShapeInferenceTestOp op("MatrixSolveLs");
  INFER_OK(op, "?;?;?", "?");
  INFER_OK(op, "?;?;[]", "?");

  // Inputs are [...,M,N], [...,M,K], and scalar regularizer.
  // Output is [...,N,K]
  // Test with no batch dims.
  INFER_OK(op, "[1,?];[1,?];?", "[d0_1,d1_1]");
  INFER_OK(op, "[1,2];[1,3];?", "[d0_1,d1_1]");
  INFER_ERROR("Dimensions must be equal, but are 5 and 6", op, "[5,?];[6,?];?");
  // Test with batch dims.
  INFER_OK(op, "[10,?,1,?];[?,20,1,?];?", "[d0_0,d1_1,d0_3,d1_3]");
  INFER_OK(op, "[10,20,1,2];[10,20,1,3];?", "[d0_0|d1_0,d0_1|d1_1,d0_3,d1_3]");
  INFER_ERROR("Dimensions must be equal, but are 5 and 6", op,
              "[10,?,5,?];[?,20,6,?];?");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 10 and 11", op,
              "[10,?,5,?];[11,?,5,?];?");

  // Rank checks.
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[?];?;?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "?;[?];?");
  INFER_ERROR("Shape must be rank 0 but is rank 1", op, "?;?;[1]");
}

TEST(LinalgOpsTest, Qr_ShapeFn) {
  ShapeInferenceTestOp op("Qr");
  auto set_attrs = [&op](bool full_matrices) {
    // Test float32
    TF_ASSERT_OK(NodeDefBuilder("test", "Qr")
                     .Input({"input", 0, DT_FLOAT})
                     .Attr("full_matrices", full_matrices)
                     .Finalize(&op.node_def));

    // Test float16
    TF_ASSERT_OK(NodeDefBuilder("test", "Qr")
                     .Input({"input", 0, DT_HALF})
                     .Attr("full_matrices", full_matrices)
                     .Finalize(&op.node_def));
  };

  // Defining `P` = min(`M`, `N`), if full_matrices = False, then Q should be
  // `M` x `P` and `R` should be `P` x `N`. Otherwise, Q should be
  // `M` x `M` and `R` should be `M` x `N`.
  //
  // For rank-3 tensors, `M` = d0_1 and `N` = d0_2.
  //
  set_attrs(false);
  INFER_OK(op, "?", "?;?");
  INFER_OK(op, "[?,?,?]", "[d0_0,d0_1,?];[d0_0,?,d0_2]");
  INFER_OK(op, "[4,?,?]", "[d0_0,d0_1,?];[d0_0,?,d0_2]");
  INFER_OK(op, "[4,2,?]", "[d0_0,d0_1,?];[d0_0,?,d0_2]");
  INFER_OK(op, "[4,?,2]", "[d0_0,d0_1,?];[d0_0,?,d0_2]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_1,d0_2];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_1,d0_2];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");

  set_attrs(true);
  INFER_OK(op, "?", "?;?");
  INFER_OK(op, "[?,?,?]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,?,?]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,?]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,?,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1,d0_1];[d0_0,d0_1,d0_2]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
}

TEST(LinalgOpsTest, Svd_ShapeFn) {
  ShapeInferenceTestOp op("Svd");
  auto set_attrs = [&op](bool compute_uv, bool full_matrices) {
    // Test for float32
    TF_ASSERT_OK(NodeDefBuilder("test", "Svd")
                     .Input({"input", 0, DT_FLOAT})
                     .Attr("compute_uv", compute_uv)
                     .Attr("full_matrices", full_matrices)
                     .Finalize(&op.node_def));

    // Test for float16
    TF_ASSERT_OK(NodeDefBuilder("test", "Svd")
                     .Input({"input", 0, DT_HALF})
                     .Attr("compute_uv", compute_uv)
                     .Attr("full_matrices", full_matrices)
                     .Finalize(&op.node_def));
  };

  // Defining `P` = min(`M`, `N`), if full_matrices = False, then U should be
  // `M` x `P` and `V` should be `N` x `P`. Otherwise, U should be
  // `M` x `M` and `V` should be `N` x `N`.
  //
  // For rank-3 tensors, `M` = d0_1 and `N` = d0_2.
  //
  set_attrs(false, false);
  INFER_OK(op, "?", "?;[0];[0]");
  INFER_OK(op, "[?,?,?]", "[d0_0,?];[0];[0]");
  INFER_OK(op, "[4,?,?]", "[d0_0,?];[0];[0]");
  INFER_OK(op, "[4,2,?]", "[d0_0,?];[0];[0]");
  INFER_OK(op, "[4,?,2]", "[d0_0,?];[0];[0]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1];[0];[0]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1];[0];[0]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_2];[0];[0]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_2];[0];[0]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1];[0];[0]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1];[0];[0]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");

  set_attrs(true, false);
  INFER_OK(op, "?", "?;?;?");
  INFER_OK(op, "[?,?,?]", "[d0_0,?];[d0_0,d0_1,?];[d0_0,d0_2,?]");
  INFER_OK(op, "[4,?,?]", "[d0_0,?];[d0_0,d0_1,?];[d0_0,d0_2,?]");
  INFER_OK(op, "[4,2,?]", "[d0_0,?];[d0_0,d0_1,?];[d0_0,d0_2,?]");
  INFER_OK(op, "[4,?,2]", "[d0_0,?];[d0_0,d0_1,?];[d0_0,d0_2,?]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_1]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_1]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_2];[d0_0,d0_1,d0_2];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_2];[d0_0,d0_1,d0_2];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_1]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_1]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");

  set_attrs(true, true);
  INFER_OK(op, "?", "?;?;?");
  INFER_OK(op, "[?,?,?]", "[d0_0,?];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,?,?]", "[d0_0,?];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,2,?]", "[d0_0,?];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,?,2]", "[d0_0,?];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,2,2]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,2,2]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,3,2]", "[d0_0,d0_2];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,3,2]", "[d0_0,d0_2];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[?,2,3]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_OK(op, "[4,2,3]", "[d0_0,d0_1];[d0_0,d0_1,d0_1];[d0_0,d0_2,d0_2]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
}

TEST(LinalgOpsTest, Lu_ShapeFn) {
  ShapeInferenceTestOp op("Lu");
  INFER_OK(op, "?", "?;?");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[1]");
  INFER_ERROR("Dimensions must be equal, but are 1 and 2", op, "[1,?,3,4,1,2]");

  INFER_OK(op, "[?,?]", "[d0_0,d0_0];[d0_0]");
  INFER_OK(op, "[1,?]", "[d0_0,d0_0];[d0_0]");
  INFER_OK(op, "[?,1]", "[d0_1,d0_1];[d0_1]");

  // Repeat previous block of tests with input rank > 2.
  INFER_OK(op, "[1,?,3,4,?,?]",
           "[d0_0,d0_1,d0_2,d0_3,d0_4,d0_4];[d0_0,d0_1,d0_2,d0_3,d0_4]");
  INFER_OK(op, "[1,?,3,4,1,?]",
           "[d0_0,d0_1,d0_2,d0_3,d0_4,d0_4];[d0_0,d0_1,d0_2,d0_3,d0_4]");
  INFER_OK(op, "[1,?,3,4,?,1]",
           "[d0_0,d0_1,d0_2,d0_3,d0_5,d0_5];[d0_0,d0_1,d0_2,d0_3,d0_5]");
}

TEST(LinalgOpsTest, TridiagonalMatMul_ShapeFn) {
  ShapeInferenceTestOp op("TridiagonalMatMul");
  INFER_OK(op, "?;?;?;?", "in3");
  INFER_OK(op, "[1,5];[1,5];[1,5];[?,1]", "in3");
  INFER_OK(op, "[1,5];[1,5];[1,5];[5,1]", "in3");

  INFER_OK(op, "[?,1,?];[?,1,?];[?,1,?];[?,?,?]", "in3");
  INFER_OK(op, "[?,1,5];[?,1,5];[?,1,5];[7,5,2]", "in3");
  INFER_OK(op, "[7,1,5];[7,1,5];[7,1,5];[?,5,2]", "in3");
  INFER_OK(op, "[7,1,5];[7,1,5];[7,1,5];[7,5,2]", "in3");

  INFER_OK(op, "[7,?,1,5];[7,?,1,5];[7,?,1,5];[7,8,5,2]", "in3");
  INFER_OK(op, "[7,8,1,5];[7,8,1,5];[7,8,1,5];[7,8,5,2]", "in3");

  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[3];[3];[3];[5,1]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op,
              "[3,5];[3,5];[3,5];[5]");
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 4 and 8. "
      "Shapes are [6,4] and [6,8].",
      op, "[6,4,3,5];[6,4,3,5];[6,4,3,5];[6,8,5,2]");
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 4 and 8. "
      "Shapes are [?,4] and [6,8].",
      op, "[?,4,3,5];[?,4,3,5];[?,4,3,5];[6,8,5,2]");

  // Diagonals must have the same length.
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 5 and 6. "
      "Shapes are [1,5] and [1,6]",
      op, "[1,5];[1,6];[1,5];[6,2]");

  // Diagonals must be 1-row matrices.
  INFER_ERROR("Dimension must be 1 but is 3", op, "[3,5];[3,5];[3,5];[5,2]");
}

TEST(LinalgOpsTest, TridiagonalSolve_ShapeFn) {
  ShapeInferenceTestOp op("TridiagonalSolve");
  INFER_OK(op, "?;?", "in1");
  INFER_OK(op, "[3,5];[?,1]", "in1");
  INFER_OK(op, "[?,5];[5,1]", "in1");
  INFER_OK(op, "[?,5];[?,?]", "in1");
  INFER_OK(op, "[?,?];[?,?]", "in1");
  INFER_OK(op, "[3,5];[5,1]", "in1");
  INFER_OK(op, "[3,5];[5,2]", "in1");

  INFER_OK(op, "[?,?,?];[?,?,?]", "in1");
  INFER_OK(op, "[?,3,5];[7,5,2]", "in1");
  INFER_OK(op, "[7,3,5];[?,5,2]", "in1");
  INFER_OK(op, "[7,?,5];[?,5,?]", "in1");
  INFER_OK(op, "[7,3,5];[7,5,2]", "in1");

  INFER_OK(op, "[7,?,3,5];[7,8,5,2]", "in1");
  INFER_OK(op, "[7,8,3,5];[7,8,5,2]", "in1");

  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[3];[5,1]");
  INFER_ERROR("Shape must be at least rank 2 but is rank 1", op, "[3,5];[5]");
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 4 and 8. "
      "Shapes are [6,4] and [6,8].",
      op, "[6,4,3,5];[6,8,5,2]");
  INFER_ERROR(
      "Dimension 1 in both shapes must be equal, but are 4 and 8. "
      "Shapes are [?,4] and [6,8].",
      op, "[?,4,3,5];[6,8,5,2]");
  INFER_ERROR("Dimension must be 3 but is 4", op, "[4,5];[5,2]");
  INFER_ERROR("Dimension must be 3 but is 4", op, "[6,4,5];[6,5,2]");
  INFER_ERROR("Dimensions must be equal, but are 9 and 5", op, "[3,9];[5,2]");
  INFER_ERROR("Dimensions must be equal, but are 9 and 5", op,
              "[6,3,9];[6,5,2]");
}

}  // end namespace tensorflow
