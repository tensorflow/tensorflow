/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/transpose.h"

#include "absl/container/inlined_vector.h"
#include "absl/numeric/int128.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/permutation_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace xla {

class TestTransposePlan : public TransposePlan {
 public:
  using TransposePlan::CoalesceDimensions;
  using TransposePlan::RemoveTrivialDimensions;
};

TEST(TransposeTest, RemoveTrivialDimensions) {
  absl::InlinedVector<int64_t, 4> dims = {4, 5, 1, 3, 1, 2, 5};
  absl::InlinedVector<int64_t, 4> perm = {0, 2, 1, 4, 3, 6, 5};
  absl::InlinedVector<int64_t, 4> lda = {2, 5, 7, 100, 3, 0, 1};
  TestTransposePlan::RemoveTrivialDimensions(dims, perm, lda);
  EXPECT_THAT(dims, testing::ElementsAre(4, 5, 3, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(0, 1, 2, 4, 3));

  dims = {4, 5, 3, 2, 5};
  perm = {4, 3, 2, 1, 0};
  lda = {2, 5, 100, 0, 1};
  TestTransposePlan::RemoveTrivialDimensions(dims, perm, lda);
  EXPECT_THAT(dims, testing::ElementsAre(4, 5, 3, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(4, 3, 2, 1, 0));
}

TEST(TransposeTest, CoalesceDimensions) {
  absl::InlinedVector<int64_t, 4> dims = {4, 5, 1, 3, 1, 2, 5};
  absl::InlinedVector<int64_t, 4> perm = {0, 2, 1, 4, 3, 6, 5};
  absl::InlinedVector<int64_t, 4> lda = {50, 30, 30, 10, 10, 5, 1};

  TestTransposePlan::CoalesceDimensions(dims, perm, lda);
  EXPECT_THAT(dims, testing::ElementsAre(4, 5, 1, 3, 1, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(0, 2, 1, 4, 3, 6, 5));
  EXPECT_THAT(lda, testing::ElementsAre(50, 30, 30, 10, 10, 5, 1));

  dims = {4, 5, 3, 2, 5};
  perm = {4, 1, 2, 3, 0};
  lda = {150, 30, 10, 5, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda);
  EXPECT_THAT(dims, testing::ElementsAre(4, 30, 5));
  EXPECT_THAT(perm, testing::ElementsAre(2, 1, 0));
  EXPECT_THAT(lda, testing::ElementsAre(150, 5, 1));

  dims = {4, 5, 3, 2, 5};
  perm = {0, 1, 2, 3, 4};
  lda = {150, 30, 10, 5, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda);
  EXPECT_THAT(dims, testing::ElementsAre(600));
  EXPECT_THAT(perm, testing::ElementsAre(0));
  EXPECT_THAT(lda, testing::ElementsAre(1));

  dims = {4, 5, 3, 2, 5};
  perm = {4, 1, 2, 3, 0};
  lda = {150, 30, 10, 7, 1};  // Non-standard stridings prevent coalescing.
  TestTransposePlan::CoalesceDimensions(dims, perm, lda);
  EXPECT_THAT(dims, testing::ElementsAre(4, 15, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(3, 1, 2, 0));
  EXPECT_THAT(lda, testing::ElementsAre(150, 10, 7, 1));
}

// Reference implementation: transpose using Eigen.
template <typename T, int NDIMS>
void TransposeUsingEigenNd(const T* input, T* output,
                           absl::Span<int64_t const> dims,
                           absl::Span<int64_t const> dims_out,
                           absl::Span<int64_t const> permutation) {
  typedef Eigen::TensorMap<
      Eigen::Tensor<T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
      Tensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, Eigen::DenseIndex>,
      Eigen::Aligned>
      ConstTensor;

  Eigen::array<int, NDIMS> p;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dims_eigen;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> dims_out_eigen;
  for (int i = 0; i < NDIMS; ++i) {
    p[i] = permutation[i];
    dims_eigen[i] = dims[i];
    dims_out_eigen[i] = dims_out[i];
  }
  auto x = ConstTensor(input, dims_eigen);
  auto y = Tensor(output, dims_out_eigen);
  y = x.shuffle(p);
}

template <typename T>
void TransposeUsingEigen(const T* input, T* output,
                         absl::Span<int64_t const> dims,
                         absl::Span<int64_t const> dims_out,
                         absl::Span<int64_t const> permutation) {
  switch (dims.size()) {
    case 0:
      return;
    case 1:
      TransposeUsingEigenNd<T, 1>(input, output, dims, dims_out, permutation);
      return;
    case 2:
      TransposeUsingEigenNd<T, 2>(input, output, dims, dims_out, permutation);
      return;
    case 3:
      TransposeUsingEigenNd<T, 3>(input, output, dims, dims_out, permutation);
      return;
    case 4:
      TransposeUsingEigenNd<T, 4>(input, output, dims, dims_out, permutation);
      return;
    default:
      LOG(FATAL) << "Unimplemented Eigen transpose rank";
  }
}

struct TransposeTestCase {
  TransposeTestCase(std::vector<int64> dims, std::vector<int64> permutation)
      : dims(std::move(dims)), permutation(std::move(permutation)) {}

  std::vector<int64> dims;
  std::vector<int64> permutation;

  std::string ToString() const {
    return absl::StrFormat("[%s],perm=[%s]", absl::StrJoin(dims, ","),
                           absl::StrJoin(permutation, ","));
  }
};

std::ostream& operator<<(std::ostream& os, const TransposeTestCase& test) {
  os << test.ToString();
  return os;
}

std::vector<TransposeTestCase> GetTransposeTestCases() {
  std::vector<TransposeTestCase> cases = {
      TransposeTestCase(/*dims=*/{1}, /*permutation=*/{0}),
      TransposeTestCase(/*dims=*/{4}, /*permutation=*/{0}),
      TransposeTestCase(/*dims=*/{27}, /*permutation=*/{0}),
      TransposeTestCase(/*dims=*/{2, 2}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{4, 4}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{4, 4}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{4, 4}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{8, 8}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{8, 8}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{16, 16}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{16, 16}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{11, 15}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{11, 15}, /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{0, 1, 2}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{0, 2, 1}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{1, 2, 0}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{1, 0, 2}),
      TransposeTestCase(/*dims=*/{11, 15, 13}, /*permutation=*/{2, 0, 1}),
      TransposeTestCase(/*dims=*/{64, 64, 64}, /*permutation=*/{2, 1, 0}),
      TransposeTestCase(/*dims=*/{4, 8, 16, 32}, /*permutation=*/{3, 1, 0, 2}),
      TransposeTestCase(/*dims=*/{64, 224, 224, 3},
                        /*permutation=*/{3, 1, 2, 0}),
  };
  return cases;
}

class TransposeTest : public ::testing::TestWithParam<TransposeTestCase> {
 protected:
  template <typename T>
  void TestTranspose() {
    const TransposeTestCase test = GetParam();
    std::vector<int64> output_dims = Permute(test.dims, test.permutation);
    TF_ASSERT_OK_AND_ASSIGN(
        auto plan,
        TransposePlan::Create(sizeof(T), test.dims, test.permutation));
    VLOG(1) << plan->ToString();
    xla::Array<T> input(test.dims);
    input.FillIota(0);
    xla::Array<T> expected_output(output_dims);
    TransposeUsingEigen(input.data(), expected_output.data(), test.dims,
                        output_dims, test.permutation);
    xla::Array<T> output(output_dims);
    EXPECT_EQ(plan->NumElems(), output.num_elements());
    plan->Execute(input.data(), output.data());

    EXPECT_EQ(expected_output, output);
  }
};

TEST_P(TransposeTest, TransposeInt8) { TestTranspose<int8>(); }
TEST_P(TransposeTest, TransposeInt16) { TestTranspose<int16>(); }
TEST_P(TransposeTest, TransposeInt32) { TestTranspose<int32>(); }
TEST_P(TransposeTest, TransposeInt64) { TestTranspose<int64>(); }
TEST_P(TransposeTest, TransposeInt128) { TestTranspose<absl::int128>(); }

INSTANTIATE_TEST_SUITE_P(TransposeTestInstance, TransposeTest,
                         ::testing::ValuesIn(GetTransposeTestCases()));

const std::vector<TransposeTestCase>* benchmark_cases = []() {
  return new std::vector<TransposeTestCase>{
      TransposeTestCase(/*dims=*/{256, 256},
                        /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{64, 224, 224, 3},
                        /*permutation=*/{1, 2, 3, 0}),
      TransposeTestCase(/*dims=*/{256, 64, 64, 3},
                        /*permutation=*/{1, 3, 2, 0}),
      TransposeTestCase(/*dims=*/{1024, 1024},
                        /*permutation=*/{1, 0}),
  };
}();

template <typename T>
void BM_Eigen(::testing::benchmark::State& state) {
  const TransposeTestCase& bm = benchmark_cases->at(state.range(0));
  Array<T> input(bm.dims);
  input.FillIota(0);
  std::vector<int64> output_dims = Permute(bm.dims, bm.permutation);
  Array<T> output(output_dims);
  for (auto s : state) {
    TransposeUsingEigen(input.data(), output.data(), bm.dims, output_dims,
                        bm.permutation);
    tensorflow::testing::DoNotOptimize(output);
  }
}
void BM_Eigen_uint8(::testing::benchmark::State& state) {
  BM_Eigen<uint8_t>(state);
}
void BM_Eigen_float(::testing::benchmark::State& state) {
  BM_Eigen<float>(state);
}
BENCHMARK(BM_Eigen_uint8)->Range(0, benchmark_cases->size() - 1);
BENCHMARK(BM_Eigen_float)->Range(0, benchmark_cases->size() - 1);

template <typename T>
void BM_Transpose(::testing::benchmark::State& state) {
  const TransposeTestCase& bm = benchmark_cases->at(state.range(0));
  TF_ASSERT_OK_AND_ASSIGN(
      auto plan, TransposePlan::Create(sizeof(T), bm.dims, bm.permutation));
  Array<T> input(bm.dims);
  input.FillIota(0);
  std::vector<int64> output_dims = Permute(bm.dims, bm.permutation);
  Array<T> output(output_dims);
  for (auto s : state) {
    plan->Execute(input.data(), output.data());
    tensorflow::testing::DoNotOptimize(output);
  }
}
void BM_Transpose_uint8(::testing::benchmark::State& state) {
  BM_Transpose<uint8_t>(state);
}
void BM_Transpose_float(::testing::benchmark::State& state) {
  BM_Transpose<float>(state);
}
BENCHMARK(BM_Transpose_uint8)->Range(0, benchmark_cases->size() - 1);
BENCHMARK(BM_Transpose_float)->Range(0, benchmark_cases->size() - 1);

TEST(TransposePlanCache, Basics) {
  TransposePlanCache cache(2);
  TF_ASSERT_OK_AND_ASSIGN(
      auto p1, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                 /*permutation=*/{2, 1, 0}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto p1a, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                  /*permutation=*/{2, 1, 0}));
  EXPECT_TRUE(p1.get() == p1a.get());
  TF_ASSERT_OK_AND_ASSIGN(
      auto p2, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                 /*permutation=*/{1, 2, 0}));
  EXPECT_TRUE(p1.get() != p2.get());
  TF_ASSERT_OK_AND_ASSIGN(
      auto p3, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                 /*permutation=*/{0, 1, 2}));
  EXPECT_TRUE(p3.get() != p1.get());
  TF_ASSERT_OK_AND_ASSIGN(
      auto p1b, cache.GetOrCreate(/*elem_size_in_bytes=*/4, /*dims=*/{1, 2, 3},
                                  /*permutation=*/{2, 1, 0}));
  EXPECT_TRUE(p1.get() != p1b.get());
}

}  // namespace xla
