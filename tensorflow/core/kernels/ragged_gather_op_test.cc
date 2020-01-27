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
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RaggedGatherOpTest : public ::tensorflow::OpsTestBase {
 protected:
  // Builds the tensorflow test graph for RaggedGather.
  template <typename VALUE_TYPE, typename INDEX_TYPE>
  void BuildRaggedGatherGraph(
      const TensorShape& indices_shape, const std::vector<INDEX_TYPE>& indices,
      const std::vector<std::vector<int64>>& params_nested_splits,
      const TensorShape& params_dense_values_shape,
      const gtl::ArraySlice<VALUE_TYPE> params_dense_values) {
    const auto& value_dtype = DataTypeToEnum<VALUE_TYPE>::v();
    const auto& index_dtype = DataTypeToEnum<INDEX_TYPE>::v();
    int64 PARAMS_RAGGED_RANK = params_nested_splits.size();
    int64 num_splits = PARAMS_RAGGED_RANK + indices_shape.dims() - 1;
    TF_ASSERT_OK(
        NodeDefBuilder("tested_op", "RaggedGather")
            .Input(FakeInput(PARAMS_RAGGED_RANK))  // params_nested_splits
            .Input(FakeInput(value_dtype))         // params_dense_values
            .Input(FakeInput(index_dtype))         // indices
            .Attr("PARAMS_RAGGED_RANK", PARAMS_RAGGED_RANK)
            .Attr("OUTPUT_RAGGED_RANK", num_splits)
            .Attr("Tvalues", value_dtype)
            .Attr("Tindices", index_dtype)
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
    for (const auto& splits : params_nested_splits) {
      int64 splits_size = splits.size();
      AddInputFromArray<int64>(TensorShape({splits_size}), splits);
    }
    AddInputFromArray<VALUE_TYPE>(params_dense_values_shape,
                                  params_dense_values);
    AddInputFromArray<INDEX_TYPE>(indices_shape, indices);
  }
};

TEST_F(RaggedGatherOpTest, RaggedGather) {
  // indices = [2, 1, 0, 3]
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  // params.shape = [4, None]
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({4}),                     // indices.shape
      {2, 1, 0, 3},                         // indices
      {{0, 3, 3, 7, 9}},                    // params_nested_splits
      TensorShape({9}),                     // params_dense_values.shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9}  // params_dense_values
  );

  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[.4, .5, .6, .7], [.1, .2, .3], [], [.8, .9]]
  test::ExpectTensorEqual<int64>(*GetOutput(0),
                                 test::AsTensor<int64>({0, 4, 4, 7, 9}));
  test::ExpectTensorNear<float>(
      *GetOutput(1),
      test::AsTensor<float>({.4, .5, .6, .7, .1, .2, .3, .8, .9}), 0.1);
}

TEST_F(RaggedGatherOpTest, RaggedGather_3DParams) {
  // indices = [2, 1, 0, 2, 3]
  // params = [[[]], [[.1, 2], [.3]], [], [[.4, .5], [.6, .7, .8]], [[.9]]]
  // params.shape = [5, None, None]
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({5}),                             // indices.shape
      {2, 1, 0, 2, 3},                              // indices
      {{0, 1, 3, 3, 5, 6}, {0, 0, 2, 3, 5, 8, 9}},  // params_nested_splits
      TensorShape({9}),                             // params_dense_values.shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9}          // params_dense_values
  );

  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[], [[.1, 2], [.3]], [[]], [], [[.4, .5], [.6, .7, .8]]]
  test::ExpectTensorEqual<int64>(*GetOutput(0),
                                 test::AsTensor<int64>({0, 0, 2, 3, 3, 5}));
  test::ExpectTensorEqual<int64>(*GetOutput(1),
                                 test::AsTensor<int64>({0, 2, 3, 3, 5, 8}));
  test::ExpectTensorNear<float>(
      *GetOutput(2), test::AsTensor<float>({.1, .2, .3, .4, .5, .6, .7, .8}),
      0.1);
}

TEST_F(RaggedGatherOpTest, RaggedGather_4DParams) {
  // indices = [2, 1, 0, 2]
  // params = [[[]], [[[1, 2], [3, 4], [5, 6]], [[7, 8]]], []]
  // params.shape = [4, None, None, 2]
  BuildRaggedGatherGraph<int32, int32>(
      TensorShape({4}),              // indices.shape
      {2, 1, 0, 2},                  // indices
      {{0, 1, 3, 3}, {0, 0, 3, 4}},  // params_nested_splits
      TensorShape({4, 2}),           // params_dense_values.shape
      {1, 2, 3, 4, 5, 6, 7, 8}       // params_dense_values
  );

  TF_ASSERT_OK(RunOpKernel());

  // Expected: [[],
  //            [[[1, 2], [3, 4], [5, 6]], [[7, 8]]],
  //            [[]],
  //            []]
  test::ExpectTensorEqual<int64>(*GetOutput(0),
                                 test::AsTensor<int64>({0, 0, 2, 3, 3}));
  test::ExpectTensorEqual<int64>(*GetOutput(1),
                                 test::AsTensor<int64>({0, 3, 4, 4}));
  test::ExpectTensorEqual<int32>(
      *GetOutput(2),
      test::AsTensor<int32>({1, 2, 3, 4, 5, 6, 7, 8}, TensorShape({4, 2})));
}

TEST_F(RaggedGatherOpTest, RaggedGather_2DIndices) {
  // indices = [[2, 1], [0, 3]]
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({2, 2}),                  // indices.shape
      {2, 1, 0, 3},                         // indices
      {{0, 3, 3, 7, 9}},                    // params_nested_splits
      TensorShape({9}),                     // params_dense_values.shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9}  // params_dense_values
  );

  TF_ASSERT_OK(RunOpKernel());

  // Expected: [ [ [.4, .5, .6, .7], [.1, .2, .3] ],
  //             [ [],               [.8, .9]     ] ]
  test::ExpectTensorEqual<int64>(*GetOutput(0),
                                 test::AsTensor<int64>({0, 2, 4}));
  test::ExpectTensorEqual<int64>(*GetOutput(1),
                                 test::AsTensor<int64>({0, 4, 4, 7, 9}));
  test::ExpectTensorNear<float>(
      *GetOutput(2),
      test::AsTensor<float>({.4, .5, .6, .7, .1, .2, .3, .8, .9}), 0.1);
}

TEST_F(RaggedGatherOpTest, RaggedGather_ScalarIndices) {
  // indices = 2
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({}),                      // indices.shape
      {2},                                  // indices
      {{0, 3, 3, 7, 9}},                    // params_nested_splits
      TensorShape({9}),                     // params_dense_values.shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9}  // params_dense_values
  );
  TF_ASSERT_OK(RunOpKernel());

  // Expected: [.4, .5, .6, .7]
  test::ExpectTensorNear<float>(*GetOutput(0),
                                test::AsTensor<float>({.4, .5, .6, .7}), 0.1);
}

TEST_F(RaggedGatherOpTest, RaggedGather_OutOfBounds) {
  // indices = [2, 10]
  // params = [[.1, .2, .3], [], [.4, .5, .6, .7], [.8, .9]]
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({2}),                     // indices.shape
      {2, 10},                              // indices
      {{0, 3, 3, 7, 9}},                    // params_nested_splits
      TensorShape({9}),                     // params_dense_values.shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9}  // params_dense_values
  );
  EXPECT_EQ("indices[1] = 10 is not in [0, 4)", RunOpKernel().error_message());
}

TEST_F(RaggedGatherOpTest, InvalidSplitsNotSorted) {
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({2}),                     // indices.shape
      {0, 2},                               // indices
      {{0, 3, 5, 2, 9}},                    // params_nested_splits
      TensorShape({9}),                     // params_dense_values.shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9}  // params_dense_values
  );
  EXPECT_EQ("Ragged splits must be sorted", RunOpKernel().error_message());
}

TEST_F(RaggedGatherOpTest, InvalidSplitsNegative) {
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({2}),                     // indices.shape
      {0, 2},                               // indices
      {{-1, 3, 2, 7, 9}},                   // params_nested_splits
      TensorShape({9}),                     // params_dense_values.shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9}  // params_dense_values
  );
  EXPECT_EQ("Ragged splits must be non-negative",
            RunOpKernel().error_message());
}

TEST_F(RaggedGatherOpTest, InvalidSplitsEmpty) {
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({0}),  // indices.shape
      {},                // indices
      {{}},              // params_nested_splits
      TensorShape({0}),  // params_dense_values.shape
      {}                 // params_dense_values
  );
  EXPECT_EQ("Ragged splits may not be empty", RunOpKernel().error_message());
}

TEST_F(RaggedGatherOpTest, InvalidSplitsTooBig) {
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({2}),                     // indices.shape
      {0, 2},                               // indices
      {{0, 20, 40, 80, 100}},               // params_nested_splits
      TensorShape({9}),                     // params_dense_values.shape
      {.1, .2, .3, .4, .5, .6, .7, .8, .9}  // params_dense_values
  );
  EXPECT_EQ("Ragged splits must not point past values",
            RunOpKernel().error_message());
}

TEST_F(RaggedGatherOpTest, BadValuesShape) {
  BuildRaggedGatherGraph<float, int32>(
      TensorShape({0}),  // indices.shape
      {},                // indices
      {{0}},             // params_nested_splits
      TensorShape({}),   // params_dense_values.shape
      {.1}               // params_dense_values
  );
  EXPECT_EQ("params.rank must be nonzero", RunOpKernel().error_message());
}

TEST_F(RaggedGatherOpTest, ShapeFn) {
  // RaggedGather(param_splits+, param_values, indices) -> [splits+, values]
  ShapeInferenceTestOp op("RaggedGather");

  (*op.node_def.mutable_attr())["PARAMS_RAGGED_RANK"].set_i(1);
  (*op.node_def.mutable_attr())["OUTPUT_RAGGED_RANK"].set_i(1);
  INFER_OK(op, "?;?;?", "[?];?");
  INFER_OK(op, "[?];[?];[?]", "[?];[?]");
  INFER_OK(op, "[?];[?,?,?];[?]", "[?];[?,d1_1,d1_2]");
  INFER_OK(op, "[5];[10];[15]", "[?];[?]");
  INFER_OK(op, "[5];[10,2];[15]", "[?];[?,d1_1]");
  INFER_ERROR("Shape must be rank 1 but is rank 0", op, "[5];[];[]");
  INFER_ERROR("Shape must be rank 1 but is rank 2", op, "[1,2];[];[5]");

  (*op.node_def.mutable_attr())["PARAMS_RAGGED_RANK"].set_i(2);
  (*op.node_def.mutable_attr())["OUTPUT_RAGGED_RANK"].set_i(2);
  INFER_OK(op, "?;?;?;?", "[?];[?];?");
  INFER_OK(op, "[?];[?];[?];[?]", "[?];[?];[?]");
  INFER_OK(op, "[?];[?];[?,?,?];[?]", "[?];[?];[?,d2_1,d2_2]");
  INFER_OK(op, "[5];[10];[15];[20]", "[?];[?];[?]");

  (*op.node_def.mutable_attr())["PARAMS_RAGGED_RANK"].set_i(1);
  (*op.node_def.mutable_attr())["OUTPUT_RAGGED_RANK"].set_i(2);
  INFER_OK(op, "?;?;?", "[?];[?];?");
  INFER_OK(op, "[?];[?];[?,?]", "[?];[?];[?]");
  INFER_OK(op, "[?];[?,?,?];[?,?]", "[?];[?];[?,d1_1,d1_2]");
  INFER_OK(op, "[15];[20];[5,10]", "[?];[?];[?]");
  INFER_OK(op, "[15];[20,2];[5,10]", "[?];[?];[?,d1_1]");

  (*op.node_def.mutable_attr())["PARAMS_RAGGED_RANK"].set_i(1);
  (*op.node_def.mutable_attr())["OUTPUT_RAGGED_RANK"].set_i(0);
  INFER_OK(op, "[?];[?];[]", "[?]");
}

}  // namespace
}  // namespace tensorflow
