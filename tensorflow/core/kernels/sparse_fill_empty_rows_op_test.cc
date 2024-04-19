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
  AddInputFromArray<int64>(TensorShape({4, 2}), {0, 1, 0, 3, 2, 0, 3, 1});
  // sparse_values
  AddInputFromArray<float>(TensorShape({4}), {0, 3, 1, 2});
  // dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {5, 6});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {4});

  TF_ASSERT_OK(RunOpKernel());

  // Checks the output indices.
  Tensor expected0(allocator(), DT_INT64, {6, 2});
  expected0.tensor<int64, 2>()(0, 0) = 0;
  expected0.tensor<int64, 2>()(0, 1) = 1;
  expected0.tensor<int64, 2>()(1, 0) = 0;
  expected0.tensor<int64, 2>()(1, 1) = 3;
  expected0.tensor<int64, 2>()(2, 0) = 1;
  expected0.tensor<int64, 2>()(2, 1) = 0;
  expected0.tensor<int64, 2>()(3, 0) = 2;
  expected0.tensor<int64, 2>()(3, 1) = 0;
  expected0.tensor<int64, 2>()(4, 0) = 3;
  expected0.tensor<int64, 2>()(4, 1) = 1;
  expected0.tensor<int64, 2>()(5, 0) = 4;
  expected0.tensor<int64, 2>()(5, 1) = 0;

  test::ExpectTensorEqual<int64>(expected0, *GetOutput(0));

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
  test::FillValues<int64>(&expected3, {0, 1, 3, 4});
  test::ExpectTensorEqual<int64>(expected3, *GetOutput(3));
}

TEST_F(SparseFillEmptyRowsTest, IndicesValuesUnmatch) {
  MakeOp(DT_INT64, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int64>(TensorShape({4, 2}), {0, 1, 0, 3, 2, 0, 3, 1});
  // sparse_values
  AddInputFromArray<float>(TensorShape({3}), {0, 3, 1});
  // dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {5, 6});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {4});

  EXPECT_THAT(RunOpKernel(),
              testing::StatusIs(error::INVALID_ARGUMENT,
                                "The length of `values` (3) must match the "
                                "first dimension of `indices` (4)."));
}

TEST_F(SparseFillEmptyRowsTest, IndicesDenseShapeUnmatch) {
  MakeOp(DT_INT64, DT_FLOAT);

  // sparse_indices
  AddInputFromArray<int64>(TensorShape({4, 0}), {});
  // sparse_values
  AddInputFromArray<float>(TensorShape({4}), {0, 3, 1, 2});
  // dense_shape
  AddInputFromArray<int64>(TensorShape({2}), {5, 6});
  // default_value
  AddInputFromArray<float>(TensorShape({}), {4});

  EXPECT_THAT(RunOpKernel(),
              testing::StatusIs(error::INVALID_ARGUMENT,
                                "The length of `dense_shape` (2) must match "
                                "the second dimension of `indices` (0)."));
}

}  // namespace

}  // namespace tensorflow
