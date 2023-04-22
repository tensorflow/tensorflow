/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/shape_inference_testutil.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RaggedTensorToSparseTest : public ::tensorflow::OpsTestBase {
 protected:
  static constexpr int kSparseIndicesOutput = 0;
  static constexpr int kSparseValuesOutput = 1;
  static constexpr int kSparseDenseShapeOutput = 2;
  // Builds the tensorflow test graph for the RaggedTensorToSparse op, and
  // populates the `splits` input with the given values.
  template <typename T>
  void BuildRaggedTensorToSparseGraph(
      const std::vector<std::vector<int64>>& rt_nested_splits,
      const TensorShape& rt_dense_values_shape,
      const std::vector<T>& rt_dense_values) {
    const auto& dtype = DataTypeToEnum<T>::v();
    int64 num_splits = rt_nested_splits.size();
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedTensorToSparse")
                     .Input(FakeInput(num_splits))  // rt_nested_splits
                     .Input(FakeInput(dtype))       // rt_dense_values
                     .Attr("RAGGED_RANK", num_splits)
                     .Attr("T", dtype)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    for (const auto& splits : rt_nested_splits) {
      int64 splits_size = splits.size();
      AddInputFromArray<int64>(TensorShape({splits_size}), splits);
    }
    AddInputFromArray<T>(rt_dense_values_shape, rt_dense_values);
  }
};

TEST_F(RaggedTensorToSparseTest, OneSplits_Values1D) {
  // ragged_tensor=[[1, 2, 3], [], [4, 5], [6]]
  BuildRaggedTensorToSparseGraph<int>({{0, 3, 3, 5, 6}},    // splits
                                      TensorShape({6}),     // values.shape
                                      {1, 2, 3, 4, 5, 6});  // values
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int64>(
      *GetOutput(kSparseIndicesOutput),
      test::AsTensor<int64>({0, 0, 0, 1, 0, 2, 2, 0, 2, 1, 3, 0}, {6, 2}));
  test::ExpectTensorEqual<int>(*GetOutput(kSparseValuesOutput),
                               test::AsTensor<int>({1, 2, 3, 4, 5, 6}));
  test::ExpectTensorEqual<int64>(*GetOutput(kSparseDenseShapeOutput),
                                 test::AsTensor<int64>({4, 3}));
}

TEST_F(RaggedTensorToSparseTest, EmptyRows) {
  // Empty rows at the beginning, middle, and end of the RaggedTensor.
  // ragged_tensor=[[], [1, 2, 3, 4], [], [5, 6], []]
  BuildRaggedTensorToSparseGraph<int>({{0, 0, 4, 4, 6, 6}},  // splits
                                      TensorShape({6}),      // values.shape
                                      {1, 2, 3, 4, 5, 6});   // values
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int64>(
      *GetOutput(kSparseIndicesOutput),
      test::AsTensor<int64>({1, 0, 1, 1, 1, 2, 1, 3, 3, 0, 3, 1}, {6, 2}));
  test::ExpectTensorEqual<int>(*GetOutput(kSparseValuesOutput),
                               test::AsTensor<int>({1, 2, 3, 4, 5, 6}));
  test::ExpectTensorEqual<int64>(*GetOutput(kSparseDenseShapeOutput),
                                 test::AsTensor<int64>({5, 4}));
}

TEST_F(RaggedTensorToSparseTest, OneSplits_Values2D) {
  // ragged_tensor=[[[1, 2], [3, 4], [5, 6]], [], [[7, 8], [9, 10]], [[11, 12]]]
  BuildRaggedTensorToSparseGraph<int>(
      {{0, 3, 3, 5, 6}},                         // splits
      TensorShape({6, 2}),                       // values.shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});  // values
  TF_ASSERT_OK(RunOpKernel());
  std::vector<int64> expected_splits_12_3 = {
      0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 2, 0, 0, 2, 1,
      2, 0, 0, 2, 0, 1, 2, 1, 0, 2, 1, 1, 3, 0, 0, 3, 0, 1};
  std::vector<int> expected_values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  test::ExpectTensorEqual<int64>(
      *GetOutput(kSparseIndicesOutput),
      test::AsTensor<int64>(expected_splits_12_3, {12, 3}));
  test::ExpectTensorEqual<int>(*GetOutput(kSparseValuesOutput),
                               test::AsTensor<int>(expected_values));
  test::ExpectTensorEqual<int64>(*GetOutput(kSparseDenseShapeOutput),
                                 test::AsTensor<int64>({4, 3, 2}));
}

TEST_F(RaggedTensorToSparseTest, TwoSplits_Values1D) {
  // ragged_tensor =
  //        0             1           2
  // -+--------------------------------------
  // 0| [[ [x],         [x x],       [] ],
  // 1|  [                              ],
  // 2|  [ [x x x x x], [x x x]         ],
  // 3|  [ [],          [x x x x]       ]]
  BuildRaggedTensorToSparseGraph<int>(
      {{0, 3, 3, 5, 7}, {0, 1, 3, 3, 8, 11, 11, 15}},        // splits
      TensorShape({15}),                                     // values.shape
      {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});  // values
  TF_ASSERT_OK(RunOpKernel());
  std::vector<int64> expected_splits_15_3 = {
      0, 0, 0, 0, 1, 0, 0, 1, 1, 2, 0, 0, 2, 0, 1, 2, 0, 2, 2, 0, 3, 2, 0,
      4, 2, 1, 0, 2, 1, 1, 2, 1, 2, 3, 1, 0, 3, 1, 1, 3, 1, 2, 3, 1, 3};
  std::vector<int> expected_values = {1, 2,  3,  4,  5,  6,  7, 8,
                                      9, 10, 11, 12, 13, 14, 15};
  test::ExpectTensorEqual<int>(*GetOutput(kSparseValuesOutput),
                               test::AsTensor<int>(expected_values));
  test::ExpectTensorEqual<int64>(
      *GetOutput(kSparseIndicesOutput),
      test::AsTensor<int64>(expected_splits_15_3, {15, 3}));
  test::ExpectTensorEqual<int64>(*GetOutput(kSparseDenseShapeOutput),
                                 test::AsTensor<int64>({4, 3, 5}));
}

TEST_F(RaggedTensorToSparseTest, ShapeFn) {
  // RaggedSplitsToIndices(rt_nested_splits+, rt_dense_values)
  //     -> [sparse_indices, sparse_values, sparse_dense_shape]
  // The output shape will always have the following form:
  //     [nvals, dense_dims];[nvals];[dense_dims]
  ShapeInferenceTestOp op("RaggedTensorToSparse");

  // Tests with len(rt_nested_splits)==0.
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(0);
  INFER_ERROR("Requires RAGGED_RANK>0", op, "?");

  // Tests with len(rt_nested_splits)==1.
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(1);
  INFER_OK(op, "?;?", "[?,?];[?];[?]");          // nvals=?, dense_dims=?
  INFER_OK(op, "?;[?]", "[?,2];[?];[2]");        // nvals=?, dense_dims=2
  INFER_OK(op, "?;[?,?]", "[?,3];[?];[3]");      // nvals=?, dense_dims=3
  INFER_OK(op, "[?];[5]", "[5,2];[5];[2]");      // nvals=5, dense_dims=2
  INFER_OK(op, "[?];[5,2]", "[10,3];[10];[3]");  // nvals=10, dense_dims=3
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[];?");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[5,5];?");
  INFER_ERROR("Shape must be at least rank 1 but is rank 0", op, "?;[]");

  // Tests with len(rt_nested_splits)==2
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(2);
  INFER_OK(op, "?;?;?", "[?,?];[?];[?]");            // nvals=?, dense_dims=?
  INFER_OK(op, "?;?;[?]", "[?,3];[?];[3]");          // nvals=?, dense_dims=3
  INFER_OK(op, "?;?;[?,?]", "[?,4];[?];[4]");        // nvals=?, dense_dims=4
  INFER_OK(op, "[?];[?];[5]", "[5,3];[5];[3]");      // nvals=5, dense_dims=3
  INFER_OK(op, "[?];[?];[5,2]", "[10,4];[10];[4]");  // nvals=10, dense_dims=4
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "?;[5,5];?");

  // Tests with len(rt_nested_splits)==3
  (*op.node_def.mutable_attr())["RAGGED_RANK"].set_i(3);
  INFER_OK(op, "?;?;?;?", "[?,?];[?];[?]");    // nvals=?, dense_dims=?
  INFER_OK(op, "?;?;?;[?]", "[?,4];[?];[4]");  // nvals=?, dense_dims=4
  INFER_OK(op, "?;?;?;[5]", "[5,4];[5];[4]");  // nvals=5, dense_dims=4
}

TEST_F(RaggedTensorToSparseTest, NoSplits) {
  const auto& dtype = DataTypeToEnum<int>::v();
  TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedTensorToSparse")
                   .Input(FakeInput(0))
                   .Input(FakeInput(dtype))
                   .Attr("RAGGED_RANK", 0)
                   .Attr("T", dtype)
                   .Finalize(node_def()));
  EXPECT_TRUE(absl::StartsWith(
      InitOp().error_message(),
      "Value for attr 'RAGGED_RANK' of 0 must be at least minimum 1"));
}

TEST_F(RaggedTensorToSparseTest, InvalidArg_BadSplitStart) {
  BuildRaggedTensorToSparseGraph<int>({{5, 7, 10}},      // splits
                                      TensorShape({0}),  // values.shape
                                      {});               // values
  EXPECT_EQ("First value of ragged splits must be 0.",
            RunOpKernel().error_message());
}

TEST_F(RaggedTensorToSparseTest, InvalidArg_BadSplitLengths1) {
  BuildRaggedTensorToSparseGraph<int>({{0, 5}, {0, 2, 4, 6}},  // splits
                                      TensorShape({0}),        // values.shape
                                      {});                     // values
  EXPECT_EQ(
      "Final value of ragged splits must match the length "
      "the corresponding ragged values.",
      RunOpKernel().error_message());
}

TEST_F(RaggedTensorToSparseTest, InvalidArg_BadSplitLengths2) {
  BuildRaggedTensorToSparseGraph<int>({{0, 5}},          // splits
                                      TensorShape({0}),  // values.shape
                                      {});               // values
  EXPECT_EQ(
      "Final value of ragged splits must match the length "
      "the corresponding ragged values.",
      RunOpKernel().error_message());
}

TEST_F(RaggedTensorToSparseTest, InvalidArg_EmptySplits) {
  BuildRaggedTensorToSparseGraph<int>({{}},              // splits
                                      TensorShape({0}),  // values.shape
                                      {});               // values
  EXPECT_EQ("ragged splits may not be empty.", RunOpKernel().error_message());
}

}  // namespace
}  // namespace tensorflow
