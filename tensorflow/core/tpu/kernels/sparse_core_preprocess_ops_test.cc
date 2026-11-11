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

#include "tensorflow/core/tpu/kernels/sparse_core_preprocess_ops.h"

#include <cstdint>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"

namespace tensorflow {
namespace {

class ConvertToCooTensorOpTest : public OpsTestBase {
 protected:
  void MakeOp(int sample_count, const std::string& combiner) {
    TF_ASSERT_OK(NodeDefBuilder("convert_to_coo_tensor", "ConvertToCooTensor")
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("sample_count", sample_count)
                     .Attr("combiner", combiner)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void ExpectInvalidArgument(absl::string_view message) {
    absl::Status status = RunOpKernel();
    EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
    EXPECT_THAT(status.message(), ::testing::HasSubstr(message));
  }
};

// Valid inputs: one each of the dense, sparse and ragged forms.
TEST_F(ConvertToCooTensorOpTest, DenseInput) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({0}), {});
  AddInputFromArray<int32_t>(TensorShape({2}), {7, 9});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int32_t>(*GetOutput(0),
                                   test::AsTensor<int32_t>({0, 1}));
  test::ExpectTensorEqual<int32_t>(*GetOutput(1),
                                   test::AsTensor<int32_t>({7, 9}));
  test::ExpectTensorEqual<float>(*GetOutput(2),
                                 test::AsTensor<float>({1.0f, 1.0f}));
}

TEST_F(ConvertToCooTensorOpTest, SparseInput) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({3, 2}), {0, 0, 0, 1, 1, 0});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 2, 3});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int32_t>(*GetOutput(0),
                                   test::AsTensor<int32_t>({0, 0, 1}));
  test::ExpectTensorEqual<int32_t>(*GetOutput(1),
                                   test::AsTensor<int32_t>({1, 2, 3}));
  test::ExpectTensorEqual<float>(*GetOutput(2),
                                 test::AsTensor<float>({0.5f, 0.5f, 1.0f}));
}

TEST_F(ConvertToCooTensorOpTest, RaggedInput) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 2, 3});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 2, 3});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int32_t>(*GetOutput(0),
                                   test::AsTensor<int32_t>({0, 0, 1}));
  test::ExpectTensorEqual<int32_t>(*GetOutput(1),
                                   test::AsTensor<int32_t>({1, 2, 3}));
  test::ExpectTensorEqual<float>(*GetOutput(2),
                                 test::AsTensor<float>({0.5f, 0.5f, 1.0f}));
}

// An empty SparseTensor is valid at any rank and needs no shape strides.
TEST_F(ConvertToCooTensorOpTest, EmptyRank3SparseInputSucceeds) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({0, 3}), {});
  AddInputFromArray<int32_t>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->NumElements(), 0);
  EXPECT_EQ(GetOutput(1)->NumElements(), 0);
  EXPECT_EQ(GetOutput(2)->NumElements(), 0);
}

TEST_F(ConvertToCooTensorOpTest, EmptyRank1SparseInputSucceeds) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({0, 1}), {});
  AddInputFromArray<int32_t>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->NumElements(), 0);
  EXPECT_EQ(GetOutput(1)->NumElements(), 0);
  EXPECT_EQ(GetOutput(2)->NumElements(), 0);
}

// Invalid inputs: each is rejected with InvalidArgument.
TEST_F(ConvertToCooTensorOpTest, EmptySparseIndicesWithValuesFails) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({0, 2}), {});
  AddInputFromArray<int32_t>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument("indices rows number the same as the values");
}

TEST_F(ConvertToCooTensorOpTest, SparseZeroColumnIndicesFails) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({4, 0}), {});
  AddInputFromArray<int32_t>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument(
      "second dimension of the indices of SparseTensor should be equal or "
      "greater than 2");
}

TEST_F(ConvertToCooTensorOpTest, SparseIndicesRowCountMismatchFails) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({1, 2}), {0, 0});
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 2});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument("indices rows number the same as the values");
}

TEST_F(ConvertToCooTensorOpTest, SparseNegativeRowIdFails) {
  MakeOp(/*sample_count=*/1, "mean");
  AddInputFromArray<int32_t>(TensorShape({1, 2}), {-1, 0});
  AddInputFromArray<int32_t>(TensorShape({1}), {5});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument("negative row_id");
}

TEST_F(ConvertToCooTensorOpTest, RaggedRowSplitsNotStartingAtZeroFails) {
  MakeOp(/*sample_count=*/1, "mean");
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 1});
  AddInputFromArray<int32_t>(TensorShape({1}), {5});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument("row_splits starting with 0");
}

class ConvertToListOfSparseCoreCooTensorsOpTest : public OpsTestBase {
 protected:
  void MakeOp(int sample_count, const std::string& combiner) {
    TF_ASSERT_OK(NodeDefBuilder("convert_to_list_of_sparse_core_coo_tensors",
                                "ConvertToListOfSparseCoreCooTensors")
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_INT32))
                     .Input(FakeInput(DT_FLOAT))
                     .Attr("sample_count", sample_count)
                     .Attr("num_sc_per_chip", 1)
                     .Attr("row_offset", 0)
                     .Attr("col_offset", 0)
                     .Attr("col_shift", 0)
                     .Attr("num_sc_shards", 1)
                     .Attr("stacked_table_sample_count", sample_count)
                     .Attr("combiner", combiner)
                     .Finalize(node_def()));
    TF_ASSERT_OK(InitOp());
  }

  void ExpectInvalidArgument(absl::string_view message) {
    absl::Status status = RunOpKernel();
    EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
    EXPECT_THAT(status.message(), ::testing::HasSubstr(message));
  }
};

TEST_F(ConvertToListOfSparseCoreCooTensorsOpTest, SparseInput) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({3, 2}), {0, 0, 0, 1, 1, 0});
  AddInputFromArray<int32_t>(TensorShape({3}), {1, 2, 3});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  TF_ASSERT_OK(RunOpKernel());
  test::ExpectTensorEqual<int32_t>(*GetOutput(0),
                                   test::AsTensor<int32_t>({0, 0, 1}));
  test::ExpectTensorEqual<int32_t>(*GetOutput(1),
                                   test::AsTensor<int32_t>({1, 2, 3}));
  test::ExpectTensorEqual<float>(*GetOutput(2),
                                 test::AsTensor<float>({0.5f, 0.5f, 1.0f}));
}

TEST_F(ConvertToListOfSparseCoreCooTensorsOpTest,
       EmptyRank3SparseInputSucceeds) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({0, 3}), {});
  AddInputFromArray<int32_t>(TensorShape({0}), {});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  TF_ASSERT_OK(RunOpKernel());
  EXPECT_EQ(GetOutput(0)->NumElements(), 0);
  EXPECT_EQ(GetOutput(1)->NumElements(), 0);
  EXPECT_EQ(GetOutput(2)->NumElements(), 0);
}

TEST_F(ConvertToListOfSparseCoreCooTensorsOpTest,
       EmptySparseIndicesWithValuesFails) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({0, 2}), {});
  AddInputFromArray<int32_t>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument("indices rows number the same as the values");
}

TEST_F(ConvertToListOfSparseCoreCooTensorsOpTest,
       SparseZeroColumnIndicesFails) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({4, 0}), {});
  AddInputFromArray<int32_t>(TensorShape({4}), {1, 2, 3, 4});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument(
      "second dimension of the indices of SparseTensor should be equal or "
      "greater than 2");
}

TEST_F(ConvertToListOfSparseCoreCooTensorsOpTest, SparseNegativeRowIdFails) {
  MakeOp(/*sample_count=*/1, "mean");
  AddInputFromArray<int32_t>(TensorShape({1, 2}), {-1, 0});
  AddInputFromArray<int32_t>(TensorShape({1}), {5});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument("negative row_id");
}

TEST_F(ConvertToListOfSparseCoreCooTensorsOpTest,
       RaggedRowSplitsNotStartingAtZeroFails) {
  MakeOp(/*sample_count=*/1, "mean");
  AddInputFromArray<int32_t>(TensorShape({2}), {1, 1});
  AddInputFromArray<int32_t>(TensorShape({1}), {5});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument("row_splits starting with 0");
}

TEST_F(ConvertToListOfSparseCoreCooTensorsOpTest,
       RaggedRowSplitsEndMismatchFails) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({3}), {0, 1, 2});
  AddInputFromArray<int32_t>(TensorShape({3}), {5, 6, 7});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument("row_splits ending with the values elements number");
}

TEST_F(ConvertToListOfSparseCoreCooTensorsOpTest,
       RaggedUnsortedRowSplitsFails) {
  MakeOp(/*sample_count=*/3, "mean");
  AddInputFromArray<int32_t>(TensorShape({4}), {0, 2, 1, 2});
  AddInputFromArray<int32_t>(TensorShape({2}), {5, 6});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument("ragged row_splits must be sorted in ascending");
}

TEST_F(ConvertToListOfSparseCoreCooTensorsOpTest, DenseCountMismatchFails) {
  MakeOp(/*sample_count=*/2, "mean");
  AddInputFromArray<int32_t>(TensorShape({0}), {});
  AddInputFromArray<int32_t>(TensorShape({3}), {5, 6, 7});
  AddInputFromArray<float>(TensorShape({}), {1.0f});
  ExpectInvalidArgument(
      "Dense tensor input should have values elements number the same as the "
      "sample count");
}

TEST(ComputeRowIdsBeforePaddingTest, ValidRank3SparseIndicesSucceeds) {
  // Row id = indices(i, 0) * 3 + indices(i, 1) for a [2, 3, N] dense shape.
  Tensor indices =
      test::AsTensor<int32_t>({0, 1, 5, 1, 2, 0}, TensorShape({2, 3}));
  std::vector<int32_t> row_ids(2, -1);
  TF_ASSERT_OK(ComputeRowIdsBeforePadding(indices, /*total_id_count=*/2,
                                          /*sample_count=*/6, row_ids.data(),
                                          /*shape_strides=*/{1, 3, 1}));
  EXPECT_THAT(row_ids, ::testing::ElementsAre(1, 5));
}

TEST(ComputeRowIdsBeforePaddingTest, EmptyRank3SparseIndicesSucceeds) {
  // No shape strides are needed when there are no coordinates.
  Tensor indices(DT_INT32, TensorShape({0, 3}));
  std::vector<int32_t> row_ids;
  TF_EXPECT_OK(ComputeRowIdsBeforePadding(indices, /*total_id_count=*/0,
                                          /*sample_count=*/2, row_ids.data()));
}

TEST(ComputeRowIdsBeforePaddingTest, NegativeSparseRowIdFails) {
  Tensor indices = test::AsTensor<int32_t>({-1, 0}, TensorShape({1, 2}));
  std::vector<int32_t> row_ids(1);
  absl::Status status = ComputeRowIdsBeforePadding(
      indices, /*total_id_count=*/1, /*sample_count=*/1, row_ids.data());
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(), ::testing::HasSubstr("negative row_id"));
}

TEST(ComputeRowIdsBeforePaddingTest, Rank3NegativeSparseRowIdFails) {
  // The positive leading coordinate masks the negative coordinate if the
  // implementation validates only the final flattened row id.
  Tensor indices = test::AsTensor<int32_t>({1, -1, 0}, TensorShape({1, 3}));
  std::vector<int32_t> row_ids(1);
  absl::Status status = ComputeRowIdsBeforePadding(
      indices, /*total_id_count=*/1, /*sample_count=*/4, row_ids.data(),
      /*shape_strides=*/{1, 4, 1});
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(), ::testing::HasSubstr("negative coordinate"));
}

TEST(ComputeRowIdsBeforePaddingTest, Rank3RowIdOverflowFails) {
  // 4 * 2^30 does not fit in int32, so the flattened row id would wrap.
  Tensor indices = test::AsTensor<int32_t>({4, 0, 0}, TensorShape({1, 3}));
  std::vector<int32_t> row_ids(1);
  absl::Status status = ComputeRowIdsBeforePadding(
      indices, /*total_id_count=*/1, /*sample_count=*/1, row_ids.data(),
      /*shape_strides=*/{1, 1 << 30, 1});
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(), ::testing::HasSubstr("int32 range"));
}

TEST(ComputeRowIdsBeforePaddingTest, Rank3RowIdAccumulationOverflowFails) {
  // Each contribution (2^30) fits in int32, but their sum (2 * 2^30) overflows
  // int32.
  Tensor indices = test::AsTensor<int32_t>({1, 1, 0}, TensorShape({1, 3}));
  std::vector<int32_t> row_ids(1);
  absl::Status status = ComputeRowIdsBeforePadding(
      indices, /*total_id_count=*/1, /*sample_count=*/1, row_ids.data(),
      /*shape_strides=*/{1 << 30, 1 << 30, 1});
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(), ::testing::HasSubstr("int32 range"));
}

TEST(ComputeRowIdsBeforePaddingTest, Rank3NegativeShapeStrideFails) {
  // Even when coordinate == 0 (so coordinate * stride == 0), a negative stride
  // must be rejected before the element loop.
  Tensor indices = test::AsTensor<int32_t>({0, 0, 0}, TensorShape({1, 3}));
  std::vector<int32_t> row_ids(1);
  absl::Status status = ComputeRowIdsBeforePadding(
      indices, /*total_id_count=*/1, /*sample_count=*/1, row_ids.data(),
      /*shape_strides=*/{-1, 1, 1});
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(), ::testing::HasSubstr("must be non-negative"));
}

TEST(ComputeRowIdsBeforePaddingTest, EmptySparseIndicesRowCountMismatchFails) {
  Tensor indices(DT_INT32, TensorShape({0, 2}));
  std::vector<int32_t> row_ids(4);
  absl::Status status = ComputeRowIdsBeforePadding(
      indices, /*total_id_count=*/4, /*sample_count=*/2, row_ids.data());
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("indices rows number the same as the "
                                   "values"));
}

TEST(ComputeRowIdsBeforePaddingTest, ZeroColumnSparseIndicesFails) {
  Tensor indices(DT_INT32, TensorShape({4, 0}));
  std::vector<int32_t> row_ids(4);
  absl::Status status = ComputeRowIdsBeforePadding(
      indices, /*total_id_count=*/4, /*sample_count=*/2, row_ids.data());
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(
      status.message(),
      ::testing::HasSubstr("second dimension of the indices of SparseTensor "
                           "should be equal or greater than 2"));
}

TEST(ComputeRowIdsBeforePaddingTest, DenseCountMismatchFails) {
  Tensor indices(DT_INT32, TensorShape({0}));
  std::vector<int32_t> row_ids(3);
  absl::Status status = ComputeRowIdsBeforePadding(
      indices, /*total_id_count=*/3, /*sample_count=*/2, row_ids.data());
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr(
                  "Dense tensor input should have values elements number the "
                  "same as the sample count"));
}

TEST(ComputeRowIdsBeforePaddingTest, RaggedRowSplitsEndMismatchFails) {
  Tensor row_splits = test::AsTensor<int32_t>({0, 1, 2});
  std::vector<int32_t> row_ids(3);
  absl::Status status = ComputeRowIdsBeforePadding(
      row_splits, /*total_id_count=*/3, /*sample_count=*/2, row_ids.data());
  EXPECT_TRUE(absl::IsInvalidArgument(status)) << status;
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr(
                  "row_splits ending with the values elements number"));
}

}  // namespace
}  // namespace tensorflow
