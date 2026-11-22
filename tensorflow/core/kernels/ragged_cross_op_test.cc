/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class RaggedCrossOpKernelTest : public ::tensorflow::OpsTestBase {
 protected:
  void BuildRaggedCrossGraph(const std::vector<DataType>& ragged_values_types,
                             const std::vector<DataType>& ragged_splits_types,
                             int num_sparse,
                             const std::vector<DataType>& sparse_values_types,
                             const std::vector<DataType>& dense_types,
                             const std::string& input_order,
                             DataType out_values_type,
                             DataType out_row_splits_type) {
    TF_ASSERT_OK(NodeDefBuilder("tested_op", "RaggedCross")
                     .Input(FakeInput(ragged_values_types))
                     .Input(FakeInput(ragged_splits_types))
                     .Input(FakeInput(num_sparse, DT_INT64))
                     .Input(FakeInput(sparse_values_types))
                     .Input(FakeInput(num_sparse, DT_INT64))
                     .Input(FakeInput(dense_types))
                     .Attr("Nsparse", num_sparse)
                     .Attr("input_order", input_order)
                     .Attr("hashed_output", false)
                     .Attr("num_buckets", 0)
                     .Attr("hash_key", 0)
                     .Attr("ragged_values_types", ragged_values_types)
                     .Attr("ragged_splits_types", ragged_splits_types)
                     .Attr("sparse_values_types", sparse_values_types)
                     .Attr("dense_types", dense_types)
                     .Attr("out_values_type", out_values_type)
                     .Attr("out_row_splits_type", out_row_splits_type)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }
};

// Positive control test case: valid sparse input succeeds
TEST_F(RaggedCrossOpKernelTest, ValidSparseInputSucceeds) {
  BuildRaggedCrossGraph(
      /*ragged_values_types=*/{},
      /*ragged_splits_types=*/{},
      /*num_sparse=*/1,
      /*sparse_values_types=*/{DT_INT64},
      /*dense_types=*/{},
      /*input_order=*/"S",
      /*out_values_type=*/DT_INT64,
      /*out_row_splits_type=*/DT_INT64);

  // Sparse input: 2 rows, indices [[0, 0], [1, 0]], values [10, 20], shape [2,
  // 1]
  AddInputFromArray<int64_t>(TensorShape({2, 2}), {0, 0, 1, 0});
  AddInputFromArray<int64_t>(TensorShape({2}), {10, 20});
  AddInputFromArray<int64_t>(TensorShape({2}), {2, 1});

  TF_ASSERT_OK(RunOpKernel());
}

TEST_F(RaggedCrossOpKernelTest, ZeroColumnSparseIndicesRejected) {
  BuildRaggedCrossGraph(
      /*ragged_values_types=*/{},
      /*ragged_splits_types=*/{},
      /*num_sparse=*/1,
      /*sparse_values_types=*/{DT_INT64},
      /*dense_types=*/{},
      /*input_order=*/"S",
      /*out_values_type=*/DT_INT64,
      /*out_row_splits_type=*/DT_INT64);

  // sparse_indices has shape [1, 0] (dim_size(1) == 0, not 2)
  AddInputFromArray<int64_t>(TensorShape({1, 0}), {});
  AddInputFromArray<int64_t>(TensorShape({1}), {42});
  AddInputFromArray<int64_t>(TensorShape({2}), {1, 1});

  absl::Status status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(status.message(),
                                "Expected sparse indices to have 2 columns"));
}

TEST_F(RaggedCrossOpKernelTest, NegativeSparseShapeBatchDimensionRejected) {
  BuildRaggedCrossGraph(
      /*ragged_values_types=*/{},
      /*ragged_splits_types=*/{},
      /*num_sparse=*/1,
      /*sparse_values_types=*/{DT_INT64},
      /*dense_types=*/{},
      /*input_order=*/"S",
      /*out_values_type=*/DT_INT64,
      /*out_row_splits_type=*/DT_INT64);

  // sparse_shape has negative batch dimension: [-1, 1]
  AddInputFromArray<int64_t>(TensorShape({0, 2}), {});
  AddInputFromArray<int64_t>(TensorShape({0}), {});
  AddInputFromArray<int64_t>(TensorShape({2}), {-1, 1});

  absl::Status status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(
      status.message(), "Sparse shape dimensions must be non-negative"));
}

TEST_F(RaggedCrossOpKernelTest, NegativeSparseShapeSecondDimensionRejected) {
  BuildRaggedCrossGraph(
      /*ragged_values_types=*/{},
      /*ragged_splits_types=*/{},
      /*num_sparse=*/1,
      /*sparse_values_types=*/{DT_INT64},
      /*dense_types=*/{},
      /*input_order=*/"S",
      /*out_values_type=*/DT_INT64,
      /*out_row_splits_type=*/DT_INT64);

  // sparse_shape has negative second dimension: [1, -1]
  AddInputFromArray<int64_t>(TensorShape({0, 2}), {});
  AddInputFromArray<int64_t>(TensorShape({0}), {});
  AddInputFromArray<int64_t>(TensorShape({2}), {1, -1});

  absl::Status status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_TRUE(absl::StrContains(
      status.message(), "Sparse shape dimensions must be non-negative"));
}

TEST_F(RaggedCrossOpKernelTest, InvalidSparseShapeLengthRejected) {
  BuildRaggedCrossGraph(
      /*ragged_values_types=*/{},
      /*ragged_splits_types=*/{},
      /*num_sparse=*/1,
      /*sparse_values_types=*/{DT_INT64},
      /*dense_types=*/{},
      /*input_order=*/"S",
      /*out_values_type=*/DT_INT64,
      /*out_row_splits_type=*/DT_INT64);

  // sparse_shape has 1 element instead of 2
  AddInputFromArray<int64_t>(TensorShape({0, 2}), {});
  AddInputFromArray<int64_t>(TensorShape({0}), {});
  AddInputFromArray<int64_t>(TensorShape({1}), {1});

  absl::Status status = RunOpKernel();
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(RaggedCrossOpKernelTest, RaggedSplitsInt32Supported) {
  BuildRaggedCrossGraph(
      /*ragged_values_types=*/{DT_INT64},
      /*ragged_splits_types=*/{DT_INT32},
      /*num_sparse=*/0,
      /*sparse_values_types=*/{},
      /*dense_types=*/{},
      /*input_order=*/"R",
      /*out_values_type=*/DT_INT64,
      /*out_row_splits_type=*/DT_INT64);

  AddInputFromArray<int64_t>(TensorShape({3}), {10, 20, 30});
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 2, 3});

  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int64_t>(*GetOutput(1),
                                   test::AsTensor<int64_t>({0, 2, 3}));
}

TEST_F(RaggedCrossOpKernelTest, MixedSplitsDtypesInt32Output) {
  BuildRaggedCrossGraph({DT_INT64, DT_INT64}, {DT_INT64, DT_INT32},
                        /*num_sparse=*/0, {}, {}, "RR", DT_INT64, DT_INT32);
  AddInputFromArray<int64_t>(TensorShape({3}), {10, 20, 30});
  AddInputFromArray<int64_t>(TensorShape({2}), {1, 2});
  AddInputFromArray<int64_t>(TensorShape({3}), {0, 2, 3});
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 1, 2});
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int32_t>(*GetOutput(1),
                                   test::AsTensor<int32_t>({0, 2, 3}));
}

}  // namespace
}  // namespace tensorflow
