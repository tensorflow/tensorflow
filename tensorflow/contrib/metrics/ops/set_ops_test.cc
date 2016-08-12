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
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(SetOpsTest, DenseToDenseShape_InvalidNumberOfInputs) {
  ShapeInferenceTestOp op("DenseToDenseSetOperation");
  op.input_tensors.resize(3);
  INFER_ERROR("Wrong number of inputs passed", op, "?;?;?");
}

TEST(SetOpsTest, DenseToDenseShape) {
  ShapeInferenceTestOp op("DenseToDenseSetOperation");

  // Unknown shapes.
  INFER_OK(op, "?;?", "[?,?];[?];[?]");

  // Invalid rank.
  INFER_ERROR("expected rank >= 2", op, "[?];?");
  INFER_ERROR("expected rank >= 2", op, "?;[?]");
  INFER_ERROR("expected rank >= 2", op, "[2];?");
  INFER_ERROR("expected rank >= 2", op, "?;[2]");

  // Mismatched ranks.
  INFER_ERROR("Ranks do not match", op, "[?,?];[?,?,?]");
  INFER_ERROR("Ranks do not match", op, "[?,?,?];[?,?]");
  INFER_ERROR("Ranks do not match", op, "[2,1];[2,1,2]");
  INFER_ERROR("Ranks do not match", op, "[2,1,2];[2,1]");

  // Rank 2, unknown dims.
  INFER_OK(op, "[?,?];?", "[?,2];[?];[2]");
  INFER_OK(op, "?;[?,?]", "[?,2];[?];[2]");
  INFER_OK(op, "[?,?];[?,?]", "[?,2];[?];[2]");

  // Rank 4, unknown dims.
  INFER_OK(op, "[?,?,?,?];?", "[?,4];[?];[4]");
  INFER_OK(op, "?;[?,?,?,?]", "[?,4];[?];[4]");
  INFER_OK(op, "[?,?,?,?];[?,?,?,?]", "[?,4];[?];[4]");

  // Known dimension 0.
  INFER_OK(op, "[4,?,?,?];?", "[d0_0,4];[d0_0];[4]");
  INFER_OK(op, "?;[4,?,?,?]", "[d1_0,4];[d1_0];[4]");
  INFER_OK(op, "[4,?,?,?];[?,?,?,?]", "[d0_0,4];[d0_0];[4]");
  INFER_OK(op, "[?,?,?,?];[4,?,?,?]", "[d1_0,4];[d1_0];[4]");
  INFER_OK(op, "[4,?,?,?];[4,?,?,?]", "[d0_0,4];[d0_0];[4]");

  // Mismatched known n-1 dims.
  INFER_ERROR("Dimension 0 in both shapes must be equal", op,
              "[4,?,2,?];[3,1,?,5]");
  INFER_ERROR("Dimension 2 in both shapes must be equal", op,
              "[4,3,2,1];[4,3,3,1]");

  // Matched known n-1 dims.
  INFER_OK(op, "[4,5,6,7];[?,?,?,?]", "[d0_0,4];[d0_0];[4]");
  INFER_OK(op, "[4,5,6,7];[?,?,?,4]", "[d0_0,4];[d0_0];[4]");
  INFER_OK(op, "[?,?,?,?];[4,5,6,7]", "[d1_0,4];[d1_0];[4]");
  INFER_OK(op, "[4,?,2,?];[?,1,?,5]", "[d0_0,4];[d0_0];[4]");
  INFER_OK(op, "[4,5,6,7];[4,?,6,?]", "[d0_0,4];[d0_0];[4]");
  INFER_OK(op, "[4,5,6,7];[4,5,6,4]", "[d0_0,4];[d0_0];[4]");
}

TEST(SetOpsTest, DenseToSparseShape_InvalidNumberOfInputs) {
  ShapeInferenceTestOp op("DenseToSparseSetOperation");
  op.input_tensors.resize(5);
  INFER_ERROR("Wrong number of inputs passed", op, "?;?;?;?;?");
}

TEST(SetOpsTest, DenseToSparseShape) {
  ShapeInferenceTestOp op("DenseToSparseSetOperation");
  INFER_OK(op, "?;?;?;?", "[?,?];[?];[?]");

  // Unknown shapes.
  INFER_OK(op, "?;?;?;?", "[?,?];[?];[?]");

  // Invalid rank.
  INFER_ERROR("expected rank >= 2", op, "[?];?;?;?");
  INFER_ERROR("expected rank >= 2", op, "[?];[?,?];[?];[?]");
  INFER_ERROR("expected rank >= 2", op, "[?];[5,3];[5];[3]");
  INFER_ERROR("expected rank >= 2", op, "[2];?;?;?");
  INFER_ERROR("expected rank >= 2", op, "[2];[?,?];[?];[?]");
  INFER_ERROR("expected rank >= 2", op, "[2];[5,3];[5];[3]");

  // Rank 2, unknown dims.
  INFER_OK(op, "[?,?];?;?;?", "[?,2];[?];[2]");
  INFER_OK(op, "[?,?];[?,?];[?];[?]", "[?,2];[?];[2]");
  INFER_OK(op, "[?,?];[5,3];[5];[3]", "[?,2];[?];[2]");

  // Rank 4, unknown dims.
  INFER_OK(op, "[?,?,?,?];?;?;?", "[?,4];[?];[4]");
  INFER_OK(op, "[?,?,?,?];[?,?];[?];[?]", "[?,4];[?];[4]");
  INFER_OK(op, "[?,?,?,?];[5,3];[5];[3]", "[?,4];[?];[4]");

  // Known dimension 0.
  INFER_OK(op, "[4,?,?,?];?;?;?", "[d0_0,4];[d0_0];[4]");
  INFER_OK(op, "[4,?,?,?];[?,?];[?];[?]", "[d0_0,4];[d0_0];[4]");
  INFER_OK(op, "[4,?,?,?];[5,3];[5];[3]", "[d0_0,4];[d0_0];[4]");

  // Invalid input sparse tensor.
  INFER_ERROR("elements in index (5) and values (6) do not match", op,
              "[?,?];[5,3];[6];[3]");
  INFER_ERROR("rank (3) and shape rank (4) do not match", op,
              "[?,?];[5,3];[5];[4]");
}

TEST(SetOpsTest, SparseToSparseShape_InvalidNumberOfInputs) {
  ShapeInferenceTestOp op("SparseToSparseSetOperation");
  op.input_tensors.resize(7);
  INFER_ERROR("Wrong number of inputs passed", op, "?;?;?;?;?;?;?");
}

TEST(SetOpsTest, SparseToSparseShape) {
  ShapeInferenceTestOp op("SparseToSparseSetOperation");
  INFER_OK(op, "?;?;?;?;?;?", "[?,?];[?];[?]");
}

}  // end namespace tensorflow
