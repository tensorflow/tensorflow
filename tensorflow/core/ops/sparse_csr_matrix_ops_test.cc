/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {

TEST(SparseMatrixOpsTest, SparseTensorToCSRSparseMatrix_ShapeFn) {
  ShapeInferenceTestOp op("SparseTensorToCSRSparseMatrix");
  (*op.node_def.mutable_attr())["T"].set_type(DT_FLOAT);
  op.input_tensors.resize(3);
  // inputs: indices, values, dense_shape
  INFER_ERROR("Expected a known rank", op, "?;?;?");
  INFER_ERROR("either 2 or 3", op, "[?,4];?;?");
  INFER_OK(op, "[?,2];?;?", "[]");
  INFER_OK(op, "[?,3];?;?", "[]");
  Tensor dense_shape_t = test::AsTensor<int64_t>({5, 6});
  op.input_tensors[2] = &dense_shape_t;
  INFER_ERROR("Shape must be rank 3 but is rank 2 for", op, "[?,3];?;?");
  INFER_OK(op, "[?,2];?;?", "[]");
}

TEST(SparseMatrixOpsTest, CSRSparseMatrixToSparseTensor_ShapeFn) {
  ShapeInferenceTestOp op("CSRSparseMatrixToSparseTensor");
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  // outputs: indices, values, dense_shape
  shapes_and_types[0].first = "[4,5]";
  INFER_OK(op, "[]", "[?,2];[?];[2]");
  shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[]", "[?,2];[?];[2]");
  shapes_and_types[0].first = "[4,5,6]";
  INFER_OK(op, "[]", "[?,3];[?];[3]");
  shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[]", "[?,3];[?];[3]");
}

TEST(SparseMatrixOpsTest, DenseToCSRSparseMatrix_ShapeFn) {
  ShapeInferenceTestOp op("DenseToCSRSparseMatrix");
  (*op.node_def.mutable_attr())["T"].set_type(DT_FLOAT);
  INFER_ERROR("Expected a known rank", op, "?;?");
  INFER_ERROR("either 2 or 3", op, "[?];?");
  INFER_OK(op, "[?,?];[?,2]", "[]");
  INFER_OK(op, "[?,?,?];[?,3]", "[]");
  INFER_ERROR("indices.shape[1] must match rank of dense; saw: 2 vs. 3", op,
              "[?,?,?];[?,2]");
}

TEST(SparseMatrixOpsTest, CSRSparseMatrixToDense_ShapeFn) {
  ShapeInferenceTestOp op("CSRSparseMatrixToDense");
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  // outputs: dense
  shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[]", "[?,?]");
  shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[]", "[?,?,?]");
}

TEST(SparseMatrixOpsTest, CSRSparseMatrixComponents_ShapeFn) {
  ShapeInferenceTestOp op("CSRSparseMatrixComponents");
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  // inputs: csr_sparse_matrix, index
  // outputs: row_ptrs, col_inds, values
  shapes_and_types[0].first = "[4,5]";
  INFER_OK(op, "[];[]", "[5];[?];[?]");
  shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[];[]", "[?];[?];[?]");
  shapes_and_types[0].first = "[19,34,55]";
  INFER_OK(op, "[];[]", "[35];[?];[?]");
  shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];[]", "[?];[?];[?]");
  shapes_and_types[0].first = "[?,?,?]";
  INFER_ERROR("index must be a scalar", op, "[];?");
}

TEST(SparseMatrixOpsTest, SparseMatrixMatMul_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixMatMul");
  std::vector<ShapeInferenceTestOp::ShapeAndType> a_shapes_and_types(1);
  a_shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&a_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  auto set_options = [&op](bool transpose_a, bool transpose_b, bool adjoint_a,
                           bool adjoint_b, bool transpose_output) {
    TF_ASSERT_OK(NodeDefBuilder("test", "SparseMatrixMatMul")
                     .Input("a", 0, DT_VARIANT)
                     .Input("b", 1, DT_FLOAT)
                     .Attr("transpose_a", transpose_a)
                     .Attr("transpose_b", transpose_b)
                     .Attr("adjoint_a", adjoint_a)
                     .Attr("adjoint_b", adjoint_b)
                     .Attr("transpose_output", transpose_output)
                     .Finalize(&op.node_def));
  };
  // inputs: a <CSR>, b <T>
  // output: matmul(a, b)
  set_options(false, false, false, false, false /*transpose_output*/);
  a_shapes_and_types[0].first = "?";
  INFER_ERROR("a has an unknown rank", op, "[];?");
  a_shapes_and_types[0].first = "[?]";
  INFER_ERROR("must be at least rank 2 but is rank 1", op, "[];?");
  a_shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[];?", "[?,?]");
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];?", "[?,?,?]");
  a_shapes_and_types[0].first = "[?,3,?]";
  INFER_OK(op, "[];[?,?,?]", "[?,3,d1_2]");
  a_shapes_and_types[0].first = "[?,3,?]";
  INFER_OK(op, "[];[?,?,4]", "[?,3,d1_2]");  // [B,3,?] . [B,?,4]
  a_shapes_and_types[0].first = "[?,?,6]";
  INFER_OK(op, "[];[?,6,?]", "[?,?,d1_2]");  // [B,?,6] . [B,6,?]
  a_shapes_and_types[0].first = "[?,?,5]";
  INFER_ERROR("must be equal, but are 5 and 6 for", op, "[];[?,6,?]");

  set_options(false, false, false, false, true /*transpose_output*/);
  a_shapes_and_types[0].first = "[?,3,?]";
  INFER_OK(op, "[];[?,?,4]", "[?,d1_2,3]");
  a_shapes_and_types[0].first = "[3,?]";
  INFER_OK(op, "[];[?,4]", "[d1_1,3]");

  set_options(/*transpose_a=*/true, /*transpose_b=*/true,
              /*adjoint_a=*/false, /*adjoint_b=*/false,
              false /*transpose_output*/);
  // t([B,W,X]) . t([B,Y,Z]) => [B,X,Y]
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];[?,?,?]", "[?,?,d1_1]");

  set_options(/*transpose_a=*/false, /*transpose_b=*/false,
              /*adjoint_a=*/true, /*adjoint_b=*/true,
              false /*transpose_output*/);
  // adj([B,W,X]) . adj([B,Y,Z]) => [B,X,Y]
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];[?,?,?]", "[?,?,d1_1]");

  set_options(true /*transpose_a*/, true /*transpose_b*/,
              /*adjoint_a=*/false, /*adjoint_b=*/false,
              true /*transpose_output*/);
  // t(t([B,W,X]) . t([B,Y,Z])) => [B,Y,X]
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[];[?,?,?]", "[?,d1_1,?]");

  set_options(/*transpose_a=*/true, /*transpose_b=*/false,
              /*adjoint_a=*/true, /*adjoint_b=*/true,
              false /*transpose_output*/);
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_ERROR("Only one of adjoint_a and transpose_a", op, "[];[?,?,?]");
  set_options(/*transpose_a=*/false, /*transpose_b=*/true,
              /*adjoint_a=*/true, /*adjoint_b=*/true,
              false /*transpose_output*/);
  a_shapes_and_types[0].first = "[?,?,?]";
  INFER_ERROR("Only one of adjoint_b and transpose_b", op, "[];[?,?,?]");
}

TEST(SparseMatrixOpsTest, SparseMatrixAdd_ShapeFn) {
  // inputs: a <CSR>, b <CSR>, alpha <scalar>, beta <scalar>
  // output: alpha * a + beta * b
  ShapeInferenceTestOp op("SparseMatrixAdd");
  std::vector<ShapeInferenceTestOp::ShapeAndType> a_shapes_and_types(1);
  std::vector<ShapeInferenceTestOp::ShapeAndType> b_shapes_and_types(1);
  a_shapes_and_types[0].second = DT_FLOAT;
  b_shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&a_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(&b_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  auto set_shapes = [&a_shapes_and_types, &b_shapes_and_types](
                        const string& a_shape, const string& b_shape) {
    a_shapes_and_types[0].first = a_shape;
    b_shapes_and_types[0].first = b_shape;
  };
  // TODO(ebrevdo): Update shape_inference_testutil to be able to properly test
  // output handle shapes and types.
  set_shapes("[?,?]", "[?,?]");
  INFER_OK(op, "[];[];?;?", "[]");  // output handle: [?,?]
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[];?;?", "[]");  // output handle: [?,?,?]
  set_shapes("[3,4]", "[3,4]");
  INFER_OK(op, "[];[];?;?", "[]");  // output handle: [3,4]
  set_shapes("[3,4,5]", "[3,4,5]");
  INFER_OK(op, "[];[];?;?", "[]");  // output handle: [3,4,5]
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[];[];[]", "[]");  // output handle: [?,?,?]
  // non-scalar beta.
  set_shapes("[?,?]", "[?,?]");
  INFER_ERROR("must be rank 0 but is rank 1", op, "[];[];?;[?]");
  // unknown rank b.
  set_shapes("[?,?,?]", "?");
  INFER_ERROR("b has an unknown rank", op, "[];[];?;?");
  // different ranks of a and b.
  set_shapes("[?,?,?]", "[?,?]");
  INFER_ERROR("must be equal", op, "[];[];?;?");
}

TEST(SparseMatrixOpsTest, SparseMatrixSparseMatMul_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixSparseMatMul");
  std::vector<ShapeInferenceTestOp::ShapeAndType> a_shapes_and_types(1);
  std::vector<ShapeInferenceTestOp::ShapeAndType> b_shapes_and_types(1);
  a_shapes_and_types[0].second = DT_FLOAT;
  b_shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&a_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(&b_shapes_and_types);
  auto set_shapes = [&a_shapes_and_types, &b_shapes_and_types](
                        const string& a_shape, const string& b_shape) {
    a_shapes_and_types[0].first = a_shape;
    b_shapes_and_types[0].first = b_shape;
  };
  auto set_options = [&op](bool transpose_a, bool transpose_b, bool adjoint_a,
                           bool adjoint_b) {
    TF_ASSERT_OK(NodeDefBuilder("test", "SparseMatrixMatMul")
                     .Input("a", 0, DT_VARIANT)
                     .Input("b", 1, DT_FLOAT)
                     .Attr("transpose_a", transpose_a)
                     .Attr("transpose_b", transpose_b)
                     .Attr("adjoint_a", adjoint_a)
                     .Attr("adjoint_b", adjoint_b)
                     .Finalize(&op.node_def));
  };
  // inputs: a <CSR>, b <CSR>
  // output: matmul(a, b) <CSR>
  set_options(false, false, false, false);
  set_shapes("?", "?");
  INFER_ERROR("has an unknown rank", op, "[];[]");
  set_shapes("[?]", "[?,?]");
  INFER_ERROR("must be at least rank 2 but is rank 1", op, "[];[]");
  set_shapes("[?,?]", "[?,?]");
  INFER_OK(op, "[];[]", "[]");  // [d0_0,d1_1]"
  set_shapes("[?,?,?]", "[?,?]");
  INFER_ERROR("must be equal rank, but are", op, "[];[]");
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // "[d0_0,d0_1,d1_2]"
  set_shapes("[?,3,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // "[d0_0,d0_1,d1_2]"
  set_shapes("[?,3,?]", "[?,?,4]");
  INFER_OK(op, "[];[]", "[]");  // [d0_0,d0_1,d1_2]"
  set_shapes("[?,?,6]", "[?,6,?]");
  INFER_OK(op, "[];[]", "[]");  // "[d0_0,d0_1,d1_2]"
  set_shapes("[?,?,5]", "[?,6,?]");
  INFER_ERROR("must be equal, but are 5 and 6 for", op, "[];[]");

  set_options(/*transpose_a=*/true, /*transpose_b=*/true, /*adjoint_a=*/false,
              /*adjoint_b=*/false);
  // t([B,W,X]) . t([B,Y,Z]) => [B,X,Y]
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // [d0_0,d0_2,d1_1]"

  set_options(/*transpose_a=*/false, /*transpose_b=*/false, /*adjoint_a=*/true,
              /*adjoint_b=*/true);
  // adj([B,W,X]) . adj([B,Y,Z]) => [B,X,Y]
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // "[d0_0,d0_2,d1_1]"

  set_options(/*transpose_a=*/true, /*transpose_b=*/false,
              /*adjoint_a=*/true, /*adjoint_b=*/true);
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_ERROR("Only one of adjoint_a and transpose_a", op, "[];[]");
  set_options(/*transpose_a=*/false, /*transpose_b=*/true,
              /*adjoint_a=*/true, /*adjoint_b=*/true);
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_ERROR("Only one of adjoint_b and transpose_b", op, "[];[]");
}

TEST(SparseMatrixOpsTest, SparseMatrixTranspose_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixTranspose");
  // inputs: input
  // outputs: output
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  shapes_and_types[0].first = "[3,4,5]";
  INFER_OK(op, "[]", "[]");  // [3,5,4]"
  shapes_and_types[0].first = "[3,4]";
  INFER_OK(op, "[]", "[]");  // "[4, 3]";
  shapes_and_types[0].first = "?";
  INFER_ERROR("input has an unknown rank", op, "[]");
}

TEST(SparseMatrixOpsTest, SparseMatrixSoftmax_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixSoftmax");
  // inputs: logits
  // outputs: softmax
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  shapes_and_types[0].first = "[?,?,?]";
  INFER_OK(op, "[]", "[]");  // "in0"
  shapes_and_types[0].first = "[?,?]";
  INFER_OK(op, "[]", "[]");  // "in0"
  shapes_and_types[0].first = "?";
  INFER_ERROR("logits has an unknown rank", op, "[]");
}

TEST(SparseMatrixOpsTest, SparseMatrixSoftmaxGrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixSoftmaxGrad");
  // inputs: softmax, grad_softmax
  // outputs: gradient
  std::vector<ShapeInferenceTestOp::ShapeAndType> a_shapes_and_types(1);
  std::vector<ShapeInferenceTestOp::ShapeAndType> b_shapes_and_types(1);
  a_shapes_and_types[0].second = DT_FLOAT;
  b_shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&a_shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(&b_shapes_and_types);
  auto set_shapes = [&a_shapes_and_types, &b_shapes_and_types](
                        const string& a_shape, const string& b_shape) {
    a_shapes_and_types[0].first = a_shape;
    b_shapes_and_types[0].first = b_shape;
  };
  set_shapes("[?,?,?]", "[?,?,?]");
  INFER_OK(op, "[];[]", "[]");  // "in0"
  set_shapes("[?,?]", "[?,?]");
  INFER_OK(op, "[];[]", "[]");  // "in0"
  set_shapes("[3,4]", "[5,6]");
  INFER_ERROR("Dimension 0 in both shapes must be equal, but are 3 and 5", op,
              "[];[]");
  set_shapes("?", "[?,?]");
  INFER_ERROR("softmax has an unknown rank", op, "[];[]");
  set_shapes("[?,?,?]", "?");
  INFER_ERROR("grad_softmax has an unknown rank", op, "[];[]");
}

TEST(SparseMatrixOpsTest, SparseMatrixMul_ShapeFn) {
  ShapeInferenceTestOp op("SparseMatrixMul");
  // inputs: a <CSR>, b <dense>
  // output: a * b
  std::vector<ShapeInferenceTestOp::ShapeAndType> shapes_and_types(1);
  shapes_and_types[0].second = DT_FLOAT;
  op.input_resource_handle_shapes_and_types.push_back(&shapes_and_types);
  op.input_resource_handle_shapes_and_types.push_back(nullptr);
  shapes_and_types[0].first = "[3,4]";
  INFER_OK(op, "[];[]", "[]");  // "[3,4]"
  shapes_and_types[0].first = "[5,3,4]";
  INFER_OK(op, "[];[?,1,1]", "[]");  // "[5,3,4]"
  // b not scalar, doesn't match a.
  shapes_and_types[0].first = "[?,?,?]";
  INFER_ERROR("b must be a scalar or shaped [batch_size, 1, 1]", op,
              "[];[3,4]");
  shapes_and_types[0].first = "[3,4]";
  INFER_ERROR("b must be a scalar or shaped", op, "[];[3,4]");
  shapes_and_types[0].first = "[3,4,5]";
  INFER_ERROR("b must be a scalar or shaped", op, "[];[3,4,5]");
  shapes_and_types[0].first = "[3,4,5]";
  INFER_ERROR("must be equal, but are 3 and 4", op, "[];[4,1,1]");
}

}  // namespace tensorflow
