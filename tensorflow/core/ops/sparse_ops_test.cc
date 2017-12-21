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
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

TEST(SparseOpsTest, SparseTensorDenseAdd_ShapeFn) {
  ShapeInferenceTestOp op("SparseTensorDenseAdd");

  // Copies input 3 to output 0.
  INFER_OK(op, "?;?;?;?", "in3");
}

TEST(SparseOpsTest, SparseAdd_ShapeFn) {
  ShapeInferenceTestOp op("SparseAdd");

  INFER_OK(op, "?;?;?;?;?;?;?", "[?,?];[?];[?]");

  // input(2) determines the output[0].
  INFER_OK(op, "?;?;[?];?;?;?;?", "[?,d2_0];[?];in2");
  INFER_OK(op, "?;?;[1];?;?;?;?", "[?,d2_0];[?];in2");
}

TEST(SparseOpsTest, SparseAddGrad_ShapeFn) {
  ShapeInferenceTestOp op("SparseAddGrad");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "?;?;[1];?");
  INFER_ERROR("must be rank 2", op, "?;[1];?;?");

  INFER_OK(op, "?;?;?;?", "[?];[?]");

  // input[1].dim(0) and input[2].dim(0) determine output.
  INFER_OK(op, "?;[?,?];[?,?];?", "[d1_0];[d2_0]");
}

TEST(SparseOpsTest, SparseReorder_ShapeFn) {
  ShapeInferenceTestOp op("SparseReorder");

  // Inputs are input_indices, input_values, and input_shape.

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always matrix and vector.
  INFER_OK(op, "?;?;?", "[?,?];[?]");

  // input_indices and input_values and transferred to outputs 0 and 1.
  INFER_OK(op, "[?,?];[?];?", "in0;in1");
}

TEST(SparseOpsTest, SparseReshape_ShapeFn) {
  ShapeInferenceTestOp op("SparseReshape");

  // Inputs are input_indices, input_shape, and new_shape.

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always matrix and vector.
  INFER_OK(op, "?;?;?", "[?,?];[?]");

  // first output is matrix [input_indices.dim(0), new_shape.dim(0)].
  // new_shape is transferred to second output.
  INFER_OK(op, "[?,?];?;[?]", "[d0_0,d2_0];in2");
}

TEST(SparseOpsTest, SparseSplit_ShapeFn) {
  ShapeInferenceTestOp op("SparseSplit");
  TF_ASSERT_OK(NodeDefBuilder("test", "SparseSplit")
                   .Input({"split_dim", 0, DT_INT64})
                   .Input({"indices", 1, DT_INT64})
                   .Input({"values", 2, DT_INT64})
                   .Input({"shape", 3, DT_INT64})
                   .Attr("num_split", 2)  // each output is copied twice.
                   .Finalize(&op.node_def));

  // output has three shape types, derived from input_shape (which is input(3)).
  // each type is copied #splits times.
  // First output is [?, NumElements(input_shape)].
  // Second output is [?]
  // Third output is input_shape.
  INFER_OK(op, "?;?;?;?", "[?,?];[?,?];[?];[?];in3;in3");
  INFER_OK(op, "?;?;?;[5,4,3,2,1]", "[?,120];[?,120];[?];[?];in3;in3");
}

TEST(SparseOpsTest, SparseToDense_ShapeFn) {
  ShapeInferenceTestOp op("SparseToDense");
  op.input_tensors.resize(4);

  // input[1] is the shape tensor.
  INFER_OK(op, "?;?;?;?", "?");
  INFER_OK(op, "?;[?];?;?", "?");
  INFER_OK(op, "?;[4];?;?", "[?,?,?,?]");
  Tensor in_t = test::AsTensor<int32>({1, 2, 3, 4});
  op.input_tensors[1] = &in_t;
  INFER_OK(op, "?;[4];?;?", "[1,2,3,4]");
}

TEST(SparseOpsTest, SparseReduceSum_ShapeFn) {
  ShapeInferenceTestOp op("SparseReduceSum");

  // Shape fn always yields unknown.
  INFER_OK(op, "?;?;?;?", "?");
}

TEST(SparseOpsTest, SerializeSparse_ShapeFn) {
  ShapeInferenceTestOp op("SerializeSparse");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always vector of size 3.
  INFER_OK(op, "?;?;?", "[3]");
}

TEST(SparseOpsTest, SerializeManySparse_ShapeFn) {
  ShapeInferenceTestOp op("SerializeManySparse");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always matrix of [?,3].
  INFER_OK(op, "?;?;?", "[?,3]");
}

TEST(SparseOpsTest, DeserializeManySparse_ShapeFn) {
  ShapeInferenceTestOp op("DeserializeManySparse");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1]");
  INFER_ERROR("must be 3", op, "[?,4]");

  // output is always [?,?];[?];[?].
  INFER_OK(op, "?", "[?,?];[?];[?]");
  INFER_OK(op, "[?,3]", "[?,?];[?];[?]");
}

TEST(SparseOpsTest, SparseTensorDenseMatMul_ShapeFn) {
  ShapeInferenceTestOp op("SparseTensorDenseMatMul");
  auto set_adjoints = [&op](bool adjoint_a, bool adjoint_b) {
    TF_ASSERT_OK(NodeDefBuilder("test", "SparseTensorDenseMatMul")
                     .Input({"a_indices", 1, DT_INT64})
                     .Input({"a_values", 2, DT_INT64})
                     .Input({"a_shape", 3, DT_INT64})
                     .Input({"b", 3, DT_INT64})
                     .Attr("adjoint_a", adjoint_a)
                     .Attr("adjoint_b", adjoint_b)
                     .Finalize(&op.node_def));
  };

  // Inputs are a_indices, a_values, a_shape, b.
  set_adjoints(false, false);

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?;?");
  INFER_ERROR("must be rank 1", op, "?;?;[];?");
  INFER_ERROR("must be rank 2", op, "?;?;[3];?");
  INFER_ERROR("must be rank 2", op, "?;?;?;[]");

  // second output dim comes from b, depending on adjoint_b value.
  INFER_OK(op, "?;?;?;?", "[?,?]");
  INFER_OK(op, "?;?;?;[?,?]", "[?,d3_1]");    // use d3_1, !adjoint_b.
  INFER_OK(op, "?;?;?;[1,2]", "[?,d3_1]");    // use d3_1, !adjoint_b.
  INFER_OK(op, "?;?;[2];[1,2]", "[?,d3_1]");  // use d3_1, !adjoint_b.

  set_adjoints(false, true);
  INFER_OK(op, "?;?;?;[?,?]", "[?,d3_0]");  // use d3_0, adjoint_b.
  INFER_OK(op, "?;?;?;[1,2]", "[?,d3_0]");  // use d3_0, adjoint_b.

  // first output comes from a, depending on adjoint_a value.
  // When input tensor is known, its values determine output shape.
  Tensor a_shape_t = test::AsTensor<int64>(std::vector<int64>{3, 1});
  op.input_tensors.resize(4);
  op.input_tensors[2] = &a_shape_t;

  // Multiplying matrices of shape [3, 1] x [1, 2]
  set_adjoints(false, false);
  INFER_OK(op, "?;?;[2];[1,2]", "[3,d3_1]");  // use d3_1, !adjoint_b.
  INFER_OK(op, "?;?;?;[1,2]", "[3,d3_1]");    // use d3_1, !adjoint_b.

  set_adjoints(true, false);
  // Trying to multiply matrices of [1, 3] x [1, 2]
  INFER_ERROR("must be equal", op, "?;?;[2];[1,2]");  // adjoint_a, !adjoint_b.

  // Try with shape tensor describing shape of rank 3.
  a_shape_t = test::AsTensor<int64>(std::vector<int64>{3, 1, 2});
  INFER_ERROR("must be rank 2 but is rank 3", op, "?;?;[3];[1,2]");
}

TEST(SparseOpsTest, SparseSoftmax_ShapeFn) {
  ShapeInferenceTestOp op("SparseSoftmax");

  // Inputs are sp_indices, sp_values, sp_shape.

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is values_shape.
  INFER_OK(op, "?;?;?", "[?]");
  INFER_OK(op, "?;[?];?", "in1");
  INFER_OK(op, "?;[5];?", "in1");
}

TEST(SparseOpsTest, SparseSparseMinAndMin_ShapeFn) {
  for (const char* op_name : {"SparseSparseMaximum", "SparseSparseMinimum"}) {
    ShapeInferenceTestOp op(op_name);

    // Rank checks.
    INFER_ERROR("must be rank 2", op, "[1];?;?;?;?;?");  // a_indices
    INFER_ERROR("must be rank 1", op, "?;[];?;?;?;?");   // a_values
    INFER_ERROR("must be rank 1", op, "?;?;[];?;?;?");   // a_shape
    INFER_ERROR("must be rank 2", op, "?;?;?;[];?;?");   // b_indices
    INFER_ERROR("must be rank 1", op, "?;?;?;?;[];?");   // b_values
    INFER_ERROR("must be rank 1", op, "?;?;?;?;?;[]");   // b_shape

    // output is always [?,?];[?]
    INFER_OK(op, "?;?;?;?;?;?", "[?,?];[?]");
    INFER_OK(op, "?;[?];?;?;?;?", "[?,?];[?]");
    INFER_OK(op, "?;[5];?;?;?;?", "[?,?];[?]");
  }
}

TEST(SparseOpsTest, SparseConcat_ShapeFn) {
  ShapeInferenceTestOp op("SparseConcat");
  std::vector<NodeDefBuilder::NodeOut> src_list;
  int n = 2;
  src_list.reserve(n);
  for (int i = 0; i < n; ++i) src_list.emplace_back("a", 0, DT_INT64);
  TF_ASSERT_OK(NodeDefBuilder("test", "SparseConcat")
                   .Input(src_list)
                   .Input(src_list)
                   .Input(src_list)
                   .Attr("N", n)
                   .Finalize(&op.node_def));

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?;?;?;?");  // indices
  INFER_ERROR("must be rank 2", op, "?;[1];?;?;?;?");  // indices
  INFER_ERROR("must be rank 1", op, "?;?;[];?;?;?");   // values
  INFER_ERROR("must be rank 1", op, "?;?;?;[];?;?");   // values
  INFER_ERROR("must be rank 1", op, "?;?;?;?;[];?");   // shapes
  INFER_ERROR("must be rank 1", op, "?;?;?;?;?;[]");   // shapes

  // row count is sum of (indices[i].dim(0) merge values[i].dim(0))
  // ind_cols is merge of (indices[i].dim(1))
  //
  // output 0 is matrix [row_count, ind_cols]
  // output 1 is matrix [row_count]
  // output 2 is merge of all shapes

  // Test merge of shapes.
  INFER_OK(op, "?;?;?;?;?;?", "[?,?];[?];[?]");
  INFER_OK(op, "?;?;?;?;[?];[?]", "[?,?];[?];in4|in5");
  INFER_OK(op, "?;?;?;?;[?];[5]", "[?,?];[?];in5");

  // Test accumulation of row_count and ind_cols from indices.
  INFER_OK(op, "[4,5];[3,?];?;?;?;?", "[7,d0_1];[7];[?]");

  // Test accumulation of row_count and ind_cols from values.
  INFER_OK(op, "?;?;[4];[3];?;?", "[7,?];[7];[?]");

  // Test merge between row_count and ind_cols.
  INFER_OK(op, "[?,2];[3,?];[4];[?];?;?", "[7,d0_1];[7];[?]");

  // Test some errors during merge.
  INFER_ERROR("but are 100 and 200", op, "[100,?];[?,?];[200];[?];?;?");
  INFER_ERROR("but are 2 and 3", op, "[?,2];[?,3];[?];[?];?;?");
  INFER_ERROR("but are 4 and 5", op, "?;?;?;?;[4];[5]");
}

TEST(SparseOpsTest, SparseDenseCwise_ShapeFn) {
  for (const char* op_name :
       {"SparseDenseCwiseMul", "SparseDenseCwiseDiv", "SparseDenseCwiseAdd"}) {
    ShapeInferenceTestOp op(op_name);

    // output is always a vector.
    INFER_OK(op, "?;?;?;?", "[?]");

    // input(0).dim(0) determines output[0].
    INFER_OK(op, "[?,?];?;?;?", "[d0_0]");

    // Rank checks.
    INFER_ERROR("must be rank 2", op, "[1];?;?;?");
  }
}

TEST(SparseOpsTest, AddSparseToTensorsMap_ShapeFn) {
  ShapeInferenceTestOp op("AddSparseToTensorsMap");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always scalar
  INFER_OK(op, "?;?;?", "[]");
}

TEST(SparseOpsTest, AddManySparseToTensorsMap_ShapeFn) {
  ShapeInferenceTestOp op("AddManySparseToTensorsMap");

  // Rank checks.
  INFER_ERROR("must be rank 2", op, "[1];?;?");
  INFER_ERROR("must be rank 1", op, "?;[];?");
  INFER_ERROR("must be rank 1", op, "?;?;[]");

  // output is always matrix of [?].
  INFER_OK(op, "?;?;?", "[?]");
}

TEST(SparseOpsTest, TakeManySparseFromTensorsMap_ShapeFn) {
  ShapeInferenceTestOp op("TakeManySparseFromTensorsMap");

  // Rank checks.
  INFER_ERROR("must be rank 1", op, "[?,1]");

  // output is always [?,?];[?];[?].
  INFER_OK(op, "?", "[?,?];[?];[?]");
  INFER_OK(op, "[?]", "[?,?];[?];[?]");
}

}  // end namespace tensorflow
