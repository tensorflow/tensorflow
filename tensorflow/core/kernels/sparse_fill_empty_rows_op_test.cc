/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

class SparseFillEmptyRowsTest : public OpsTestBase {
 protected:
  void MakeOp(DataType index_type, DataType value_type) {
    TF_ASSERT_OK(NodeDefBuilder("sparsefillemptyrows", "SparseFillEmptyRows")
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(value_type))
                     .Input(FakeInput(index_type))
                     .Input(FakeInput(value_type))
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseFillEmptyRowsTest, SparseFillEmptyRows) {
  MakeOp(DT_INT64, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int64_t>(TensorShape({4, 2}), {0, 1, 0, 3, 2, 0, 3, 1});
  // sparse_values
  AddInputFromArray<float>(TensorShape({4}), {0, 3, 1, 2});
  // dense_shape
  AddInputFromArray<int64_t>(TensorShape({2}), {5, 6});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {4});

  TF_ASSERT_OK(RunOpKernel());

  // Checks the output indices.
  Tensor expected0(allocator(), DT_INT64, {6, 2});
  expected0.tensor<int64_t, 2>()(0, 0) = 0;
  expected0.tensor<int64_t, 2>()(0, 1) = 1;
  expected0.tensor<int64_t, 2>()(1, 0) = 0;
  expected0.tensor<int64_t, 2>()(1, 1) = 3;
  expected0.tensor<int64_t, 2>()(2, 0) = 1;
  expected0.tensor<int64_t, 2>()(2, 1) = 0;
  expected0.tensor<int64_t, 2>()(3, 0) = 2;
  expected0.tensor<int64_t, 2>()(3, 1) = 0;
  expected0.tensor<int64_t, 2>()(4, 0) = 3;
  expected0.tensor<int64_t, 2>()(4, 1) = 1;
  expected0.tensor<int64_t, 2>()(5, 0) = 4;
  expected0.tensor<int64_t, 2>()(5, 1) = 0;

  test::ExpectTensorEqual<int64_t>(expected0, *GetOutput(0));

  // Checks the output values.
  Tensor expected1(allocator(), DT_FLOAT, {6});
  test::FillValues<float>(&expected1, {0, 3, 4, 1, 2, 4});
  test::ExpectTensorEqual<float>(expected1, *GetOutput(1));

  // Checks the empty row indicator.
  Tensor expected2(allocator(), DT_BOOL, {5});
  test::FillValues<bool>(&expected2, {false, true, false, false, true});
  test::ExpectTensorEqual<bool>(expected2, *GetOutput(2));

  // Checks the reverse index map.
  Tensor expected3(allocator(), DT_INT64, {4});
  test::FillValues<int64_t>(&expected3, {0, 1, 3, 4});
  test::ExpectTensorEqual<int64_t>(expected3, *GetOutput(3));
}

TEST_F(SparseFillEmptyRowsTest, IndicesValuesUnmatch) {
  MakeOp(DT_INT64, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int64_t>(TensorShape({4, 2}), {0, 1, 0, 3, 2, 0, 3, 1});
  // sparse_values
  AddInputFromArray<float>(TensorShape({3}), {0, 3, 1});
  // dense_shape
  AddInputFromArray<int64_t>(TensorShape({2}), {5, 6});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {4});

  const auto status = RunOpKernel();
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(),
            "The length of `values` (3) must match the first dimension of "
            "`indices` (4).");
}

TEST_F(SparseFillEmptyRowsTest, IndicesDenseShapeUnmatch) {
  MakeOp(DT_INT64, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int64_t>(TensorShape({4, 0}), {});
  // sparse_values
  AddInputFromArray<float>(TensorShape({4}), {0, 3, 1, 2});
  // dense_shape
  AddInputFromArray<int64_t>(TensorShape({2}), {5, 6});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {4});

  const auto status = RunOpKernel();
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(),
            "The length of `dense_shape` (2) must match the second dimension "
            "of `indices` (0).");
}

class SparseFillEmptyRowsGradTest : public OpsTestBase {
 protected:
  void MakeOp(DataType index_type, DataType value_type) {
    TF_ASSERT_OK(
        NodeDefBuilder("sparsefillemptyrowsgrad", "SparseFillEmptyRowsGrad")
            .Input(FakeInput(index_type))
            .Input(FakeInput(value_type))
            .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

TEST_F(SparseFillEmptyRowsGradTest, SparseFillEmptyRowsGrad) {
  MakeOp(DT_INT64, DT_FLOAT);

  // reverse_index_map
  AddInputFromArray<int64_t>(TensorShape({2}), {2, 1});
  // grad_values
  AddInputFromArray<float>(TensorShape({4}), {0, 1, 2, 3});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_d_values(allocator(), DT_FLOAT, {2});
  test::FillValues<float>(&expected_d_values, {2, 1});
  test::ExpectTensorEqual<float>(expected_d_values, *GetOutput(0));

  Tensor expected_d_default_value(allocator(), DT_FLOAT, {});
  test::FillValues<float>(&expected_d_default_value, {3});
  test::ExpectTensorEqual<float>(expected_d_default_value, *GetOutput(1));
}

TEST_F(SparseFillEmptyRowsGradTest, InvalidReverseIndexMap) {
  MakeOp(DT_INT64, DT_FLOAT);

  // reverse_index_map
  AddInputFromArray<int64_t>(TensorShape({2}), {2, 10});
  // grad_values
  AddInputFromArray<float>(TensorShape({4}), {0, 1, 2, 3});

  const auto status = RunOpKernel();
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(),
            "Elements in reverse index must be in [0, 4) but got 10");
}

TEST_F(SparseFillEmptyRowsGradTest, NegativeReverseIndexMap) {
  MakeOp(DT_INT64, DT_FLOAT);

  // reverse_index_map
  AddInputFromArray<int64_t>(TensorShape({2}), {-1, 1});
  // grad_values
  AddInputFromArray<float>(TensorShape({4}), {0, 1, 2, 3});

  const auto status = RunOpKernel();
  EXPECT_EQ(status.code(), error::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(),
            "Elements in reverse index must be in [0, 4) but got -1");
}

TEST_F(SparseFillEmptyRowsGradTest, EmptyInputs) {
  MakeOp(DT_INT64, DT_FLOAT);

  // reverse_index_map
  AddInputFromArray<int64_t>(TensorShape({0}), {});
  // grad_values
  AddInputFromArray<float>(TensorShape({0}), {});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_d_values(allocator(), DT_FLOAT, {0});
  test::ExpectTensorEqual<float>(expected_d_values, *GetOutput(0));

  Tensor expected_d_default_value(allocator(), DT_FLOAT, {});
  test::FillValues<float>(&expected_d_default_value, {0});
  test::ExpectTensorEqual<float>(expected_d_default_value, *GetOutput(1));
}

TEST_F(SparseFillEmptyRowsGradTest, EmptyReverseIndexMap) {
  MakeOp(DT_INT64, DT_FLOAT);

  // reverse_index_map
  AddInputFromArray<int64_t>(TensorShape({0}), {});
  // grad_values
  AddInputFromArray<float>(TensorShape({3}), {1, 2, 3});

  TF_ASSERT_OK(RunOpKernel());

  Tensor expected_d_values(allocator(), DT_FLOAT, {0});
  test::ExpectTensorEqual<float>(expected_d_values, *GetOutput(0));

  Tensor expected_d_default_value(allocator(), DT_FLOAT, {});
  test::FillValues<float>(&expected_d_default_value, {6});
  test::ExpectTensorEqual<float>(expected_d_default_value, *GetOutput(1));
}

}  // namespace

}  // namespace tensorflow
