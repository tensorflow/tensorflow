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

#include "tensorflow/core/kernels/sparse/kernels.h"

#include <cstdint>

#include <gmock/gmock.h>
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status_matchers.h"

namespace tensorflow {
namespace {

TEST(SparseTensorToCSRSparseMatrix, SingleBatchConversion) {
  // Single matrix of size 4 x 5.
  const auto indices =
      test::AsTensor<int64_t>({0, 0, 2, 3, 2, 4, 3, 0}, TensorShape({4, 2}));
  Tensor batch_ptr(DT_INT32, {2});
  Tensor csr_col_ind(DT_INT32, {4});
  auto csr_row_ptr = test::AsTensor<int32>({0, 0, 0, 0, 0});

  functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
  TF_EXPECT_OK(coo_to_csr(/*batch_size=*/1, /*num_rows=*/4, /*num_cols=*/5,
                          indices.template matrix<int64_t>(),
                          batch_ptr.vec<int32>(), csr_row_ptr.vec<int32>(),
                          csr_col_ind.vec<int32>()));

  test::ExpectTensorEqual<int32>(batch_ptr, test::AsTensor<int32>({0, 4}));
  test::ExpectTensorEqual<int32>(csr_row_ptr,
                                 test::AsTensor<int32>({0, 1, 1, 3, 4}));
  test::ExpectTensorEqual<int32>(csr_col_ind,
                                 test::AsTensor<int32>({0, 3, 4, 0}));
}

TEST(SparseTensorToCSRSparseMatrix, BatchConversion) {
  // Batch of 3 matrices, each having dimension [3, 4] with 3 non-zero elements.
  const auto indices = test::AsTensor<int64_t>({0, 0, 0,  //
                                                0, 2, 3,  //
                                                2, 0, 1},
                                               TensorShape({3, 3}));
  Tensor batch_ptr(DT_INT32, {4});
  Tensor csr_col_ind(DT_INT32, {3});
  // row pointers have size = batch_size * (num_rows + 1) = 3 * 4 = 12
  Tensor csr_row_ptr(DT_INT32, {12});
  test::FillFn<int32>(&csr_row_ptr, [](int unused) { return 0; });

  functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
  TF_EXPECT_OK(coo_to_csr(/*batch_size=*/3, /*num_rows=*/3, /*num_cols=*/4,
                          indices.template matrix<int64_t>(),
                          batch_ptr.vec<int32>(), csr_row_ptr.vec<int32>(),
                          csr_col_ind.vec<int32>()));

  test::ExpectTensorEqual<int32>(batch_ptr,
                                 test::AsTensor<int32>({0, 2, 2, 3}));
  test::ExpectTensorEqual<int32>(csr_row_ptr,
                                 test::AsTensor<int32>({0, 1, 1, 2,  //
                                                        0, 0, 0, 0,  //
                                                        0, 1, 1, 1}));
  test::ExpectTensorEqual<int32>(csr_col_ind, test::AsTensor<int32>({0, 3, 1}));
}

TEST(SparseTensorToCSRSparseMatrix, InvalidBatchThrowsIllegalArgument) {
  const auto indices =
      test::AsTensor<int64_t>({0, 0, 0,  //
                               4, 2, 3,  // Batch out of bounds.
                               2, 0, 1},
                              TensorShape({3, 3}));
  Tensor batch_ptr(DT_INT32, {4});
  Tensor csr_col_ind(DT_INT32, {3});
  // row pointers have size = batch_size * (num_rows + 1) = 3 * 4 = 12
  Tensor csr_row_ptr(DT_INT32, {12});
  test::FillFn<int32>(&csr_row_ptr, [](int unused) { return 0; });

  functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
  EXPECT_THAT(
      coo_to_csr(/*batch_size=*/3, /*num_rows=*/3, /*num_cols=*/4,
                 indices.template matrix<int64_t>(), batch_ptr.vec<int32>(),
                 csr_row_ptr.vec<int32>(), csr_col_ind.vec<int32>()),
      absl_testing::StatusIs(tsl::error::Code::INVALID_ARGUMENT,
                             ::testing::ContainsRegex(
                                 "Batch index .* is outside of valid range")));
}

TEST(SparseTensorToCSRSparseMatrix, InvalidRowThrowsIllegalArgument) {
  const auto indices = test::AsTensor<int64_t>({0, 0, 0,   //
                                                1, 4, 3,   // Row out of bounds.
                                                2, 0, 1},  //
                                               TensorShape({3, 3}));
  Tensor batch_ptr(DT_INT32, {4});
  Tensor csr_col_ind(DT_INT32, {3});
  // row pointers have size = batch_size * (num_rows + 1) = 3 * 4 = 12
  Tensor csr_row_ptr(DT_INT32, {12});
  test::FillFn<int32>(&csr_row_ptr, [](int unused) { return 0; });

  functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
  EXPECT_THAT(
      coo_to_csr(/*batch_size=*/3, /*num_rows=*/3, /*num_cols=*/4,
                 indices.template matrix<int64_t>(), batch_ptr.vec<int32>(),
                 csr_row_ptr.vec<int32>(), csr_col_ind.vec<int32>()),
      absl_testing::StatusIs(
          tsl::error::Code::INVALID_ARGUMENT,
          ::testing::ContainsRegex("Row index .* is outside of valid range")));
}

TEST(SparseTensorToCSRSparseMatrix, InvalidColThrowsIllegalArgument) {
  const auto indices = test::AsTensor<int64_t>({0, 0, 0,   //
                                                1, 2, 6,   // Col out of bounds.
                                                2, 0, 1},  //
                                               TensorShape({3, 3}));
  Tensor batch_ptr(DT_INT32, {4});
  Tensor csr_col_ind(DT_INT32, {3});
  // row pointers have size = batch_size * (num_rows + 1) = 3 * 4 = 12
  Tensor csr_row_ptr(DT_INT32, {12});
  test::FillFn<int32>(&csr_row_ptr, [](int unused) { return 0; });

  functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
  EXPECT_THAT(
      coo_to_csr(/*batch_size=*/3, /*num_rows=*/3, /*num_cols=*/4,
                 indices.template matrix<int64_t>(), batch_ptr.vec<int32>(),
                 csr_row_ptr.vec<int32>(), csr_col_ind.vec<int32>()),
      absl_testing::StatusIs(tsl::error::Code::INVALID_ARGUMENT,
                             ::testing::ContainsRegex(
                                 "Column index .* is outside of valid range")));
}

TEST(SparseTensorToCSRSparseMatrix, InvalidRankIllegalArgument) {
  const auto indices =
      test::AsTensor<int64_t>({0, 0, 0, 0,           //
                               1, 2, 2, 3,           //
                               2, 0, 1, 2},          //
                              TensorShape({3, 4}));  // Too many columns.
  Tensor batch_ptr(DT_INT32, {4});
  Tensor csr_col_ind(DT_INT32, {3});
  // row pointers have size = batch_size * (num_rows + 1) = 3 * 4 = 12
  Tensor csr_row_ptr(DT_INT32, {12});
  test::FillFn<int32>(&csr_row_ptr, [](int unused) { return 0; });

  functor::SparseTensorToCSRSparseMatrixCPUFunctor coo_to_csr;
  EXPECT_THAT(
      coo_to_csr(/*batch_size=*/3, /*num_rows=*/3, /*num_cols=*/4,
                 indices.template matrix<int64_t>(), batch_ptr.vec<int32>(),
                 csr_row_ptr.vec<int32>(), csr_col_ind.vec<int32>()),
      absl_testing::StatusIs(tsl::error::Code::INVALID_ARGUMENT,
                             ::testing::ContainsRegex(
                                 "Indices must have either 2 or 3 columns.")));
}

}  // namespace
}  // namespace tensorflow

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
