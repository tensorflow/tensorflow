/* Copyright 2021 The OpenXLA Authors.

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

#include "xla/pjrt/transpose.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/numeric/int128.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "xla/array.h"
#include "xla/permutation_util.h"
#include "xla/shape_util.h"
#include "xla/test.h"
#include "xla/util.h"
#include "tsl/platform/test_benchmark.h"
#include "tsl/platform/threadpool.h"
#include "tsl/protobuf/error_codes.pb.h"

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
  absl::InlinedVector<int64_t, 4> lda_tile = {1, 1, 1, 1, 1, 1, 1};
  absl::InlinedVector<int64_t, 4> input_tiling = {1, 1, 1, 1, 1, 1, 1};
  absl::InlinedVector<int64_t, 4> output_tiling = {1, 1, 1, 1, 1, 1, 1};
  TestTransposePlan::RemoveTrivialDimensions(dims, perm, lda, lda_tile,
                                             input_tiling, output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 5, 3, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(0, 1, 2, 4, 3));

  dims = {4, 5, 3, 2, 5};
  perm = {4, 3, 2, 1, 0};
  lda = {2, 5, 100, 0, 1};
  lda_tile = {1, 1, 1, 1, 1};
  input_tiling = {1, 1, 1, 1, 1};
  output_tiling = {1, 1, 1, 1, 1};
  TestTransposePlan::RemoveTrivialDimensions(dims, perm, lda, lda_tile,
                                             input_tiling, output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 5, 3, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(4, 3, 2, 1, 0));
}

TEST(TransposeTest, CoalesceDimensions) {
  absl::InlinedVector<int64_t, 4> dims = {4, 5, 1, 3, 1, 2, 5};
  absl::InlinedVector<int64_t, 4> perm = {0, 2, 1, 4, 3, 6, 5};
  absl::InlinedVector<int64_t, 4> lda = {50, 30, 30, 10, 10, 5, 1};
  absl::InlinedVector<int64_t, 4> lda_tile = {1, 1, 1, 1, 1, 1, 1};
  absl::InlinedVector<int64_t, 4> input_tiling = {1, 1, 1, 1, 1, 1, 1};
  absl::InlinedVector<int64_t, 4> output_tiling = {1, 1, 1, 1, 1, 1, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda, lda_tile, input_tiling,
                                        output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 5, 1, 3, 1, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(0, 2, 1, 4, 3, 6, 5));
  EXPECT_THAT(lda, testing::ElementsAre(50, 30, 30, 10, 10, 5, 1));

  dims = {4, 5, 3, 2, 5};
  perm = {4, 1, 2, 3, 0};
  lda = {150, 30, 10, 5, 1};
  lda_tile = {1, 1, 1, 1, 1};
  input_tiling = {1, 1, 1, 1, 1};
  output_tiling = {1, 1, 1, 1, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda, lda_tile, input_tiling,
                                        output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 30, 5));
  EXPECT_THAT(perm, testing::ElementsAre(2, 1, 0));
  EXPECT_THAT(lda, testing::ElementsAre(150, 5, 1));

  dims = {4, 5, 3, 2, 5};
  perm = {0, 1, 2, 3, 4};
  lda = {150, 30, 10, 5, 1};
  lda_tile = {1, 1, 1, 1, 1};
  input_tiling = {1, 1, 1, 1, 1};
  output_tiling = {1, 1, 1, 1, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda, lda_tile, input_tiling,
                                        output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(600));
  EXPECT_THAT(perm, testing::ElementsAre(0));
  EXPECT_THAT(lda, testing::ElementsAre(1));

  dims = {4, 5, 3, 2, 5};
  perm = {4, 1, 2, 3, 0};
  lda = {150, 30, 10, 7, 1};  // Non-standard stridings prevent coalescing.
  lda_tile = {1, 1, 1, 1, 1};
  input_tiling = {1, 1, 1, 1, 1};
  output_tiling = {1, 1, 1, 1, 1};
  TestTransposePlan::CoalesceDimensions(dims, perm, lda, lda_tile, input_tiling,
                                        output_tiling);
  EXPECT_THAT(dims, testing::ElementsAre(4, 15, 2, 5));
  EXPECT_THAT(perm, testing::ElementsAre(3, 1, 2, 0));
  EXPECT_THAT(lda, testing::ElementsAre(150, 10, 7, 1));
}

TEST(TransposeTest, InvalidTilings) {
  TransposePlan::Options options;
  std::vector<int64_t> dims = {3, 4, 5};
  std::vector<int64_t> perm = {0, 1, 2};
  options.elem_size_in_bytes = sizeof(float);
  options.dims = dims;
  options.permutation = perm;
  std::vector<int64_t> input_tiling = {8, 128};
  std::vector<int64_t> output_tiling = {4};
  options.input_layout = TransposePlan::Tiling{input_tiling};
  options.output_tiling = TransposePlan::Tiling{output_tiling};
  auto plan = TransposePlan::Create(options);
  EXPECT_EQ(plan.status().code(), tsl::error::UNIMPLEMENTED);
  EXPECT_THAT(
      plan.status().message(),
      testing::HasSubstr(
          "Only one of the input and output may have a non-trivial tiling"));
}

// Computes the size in elements of a tiled array.
int64_t SizeOfTiledArray(absl::Span<int64_t const> shape,
                         absl::Span<int64_t const> tiling) {
  int64_t size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i >= shape.size() - tiling.size()) {
      size *= RoundUpTo(shape[i], tiling[i - (shape.size() - tiling.size())]);
    } else {
      size *= shape[i];
    }
  }
  return size;
}

// Advances 'indices' in the lexicographical order of the multidimensional
// array with `shape`. Returns false if the end of the array has been reached.
bool BumpIndices(absl::Span<int64_t const> shape, absl::Span<int64_t> indices) {
  CHECK_EQ(shape.size(), indices.size());
  for (int dimno = indices.size() - 1; dimno >= 0; --dimno) {
    if (indices[dimno] + 1 < shape[dimno]) {
      indices[dimno]++;
      // Whenever an index of a dimension is increased, it means that all
      // following dimensions have maxed out, so they must go to 0.
      std::fill(indices.begin() + dimno + 1, indices.end(), 0);
      return true;
    }
  }
  return false;
}

// Converts a multidimensional index `indices` into an array with `shape` and
// tiling `tiling` into a linear offset into a buffer.
int64_t IndexToLinearIndex(absl::Span<int64_t const> shape,
                           absl::Span<int64_t const> tiling,
                           absl::Span<int64_t const> indices) {
  CHECK_LE(tiling.size(), shape.size());
  CHECK_EQ(shape.size(), indices.size());
  int64_t stride = 1;
  int64_t offset = 0;

  auto index_it = indices.rbegin();
  auto tile_it = tiling.rbegin();
  for (; tile_it != tiling.rend(); ++index_it, ++tile_it) {
    offset += (*index_it % *tile_it) * stride;
    stride *= *tile_it;
  }
  index_it = indices.rbegin();
  tile_it = tiling.rbegin();
  auto shape_it = shape.rbegin();
  for (; tile_it != tiling.rend(); ++index_it, ++shape_it, ++tile_it) {
    offset += (*index_it / *tile_it) * stride;
    stride *= CeilOfRatio(*shape_it, *tile_it);
  }
  for (; shape_it != shape.rend(); ++index_it, ++shape_it) {
    offset += *index_it * stride;
    stride *= *shape_it;
  }
  return offset;
}

// Slow reference code that converts an array from an untiled layout into a
// tiled layout.
template <typename T>
std::vector<T> TileArray(const Array<T>& in, absl::Span<int64_t const> tiling) {
  std::vector<T> out(SizeOfTiledArray(in.dimensions(), tiling), -1);
  if (in.num_elements() == 0) {
    return out;
  }
  std::vector<int64_t> indices(in.num_dimensions(), 0);
  do {
    int64_t i = IndexToLinearIndex(in.dimensions(), tiling, indices);
    out.at(i) = in(indices);
  } while (BumpIndices(in.dimensions(), absl::MakeSpan(indices)));
  return out;
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
  TransposeTestCase(std::vector<int64_t> dims, std::vector<int64_t> permutation,
                    std::vector<int64_t> input_tiling = {},
                    std::vector<int64_t> output_tiling = {})
      : dims(std::move(dims)),
        permutation(std::move(permutation)),
        input_tiling(std::move(input_tiling)),
        output_tiling(std::move(output_tiling)) {}

  std::vector<int64_t> dims;
  std::vector<int64_t> permutation;
  std::vector<int64_t> input_tiling;
  std::vector<int64_t> output_tiling;

  std::string ToString() const {
    return absl::StrFormat(
        "[%s],perm=[%s],tiling=[%s]/[%s]", absl::StrJoin(dims, ","),
        absl::StrJoin(permutation, ","), absl::StrJoin(input_tiling, ","),
        absl::StrJoin(output_tiling, ","));
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
      TransposeTestCase(/*dims=*/{1, 1}, /*permutation=*/{0, 1}),
      TransposeTestCase(/*dims=*/{1, 1}, /*permutation=*/{1, 0}),
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
      TransposeTestCase(/*dims=*/{256, 256, 256}, /*permutation=*/{2, 1, 0}),
      TransposeTestCase(/*dims=*/{4, 8, 16, 32}, /*permutation=*/{3, 1, 0, 2}),
      TransposeTestCase(/*dims=*/{64, 224, 224, 3},
                        /*permutation=*/{3, 1, 2, 0}),

      TransposeTestCase(/*dims=*/{3}, /*permutation=*/{0},
                        /*input_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{3}, /*permutation=*/{0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{2, 4, 6}, /*permutation=*/{0, 1, 2},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{2, 3}),
      TransposeTestCase(/*dims=*/{4}, /*permutation=*/{0},
                        /*input_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{5}, /*permutation=*/{0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{8}, /*permutation=*/{0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{8}, /*permutation=*/{0},
                        /*input_tiling=*/{3},
                        /*output_tiling=*/{}),
      TransposeTestCase(/*dims=*/{29}, /*permutation=*/{0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{3}),
      TransposeTestCase(/*dims=*/{12, 7}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{4}),
      TransposeTestCase(/*dims=*/{12, 7}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{}, /*output_tiling=*/{5}),
      TransposeTestCase(/*dims=*/{12, 7}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{2, 4}),
      TransposeTestCase(/*dims=*/{12, 7}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{}, /*output_tiling=*/{5, 2}),
      TransposeTestCase(/*dims=*/{128, 224, 224, 3},
                        /*permutation=*/{3, 1, 2, 0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{8, 128}),
  };
  return cases;
}

class TransposeTest : public ::testing::TestWithParam<TransposeTestCase> {
 protected:
  template <typename T>
  void TestTranspose(int parallelism) {
    const TransposeTestCase test = GetParam();
    tsl::thread::ThreadPool threadpool(tsl::Env::Default(), "Transpose",
                                       parallelism);
    std::vector<int64_t> output_dims = Permute(test.dims, test.permutation);
    TransposePlan::Options options;
    options.elem_size_in_bytes = sizeof(T);
    options.dims = test.dims;
    options.permutation = test.permutation;
    options.input_layout = TransposePlan::Tiling{test.input_tiling};
    options.output_tiling = TransposePlan::Tiling{test.output_tiling};
    options.transformation = TransposePlan::Transformation::kNone;
    options.num_threads = parallelism;
    TF_ASSERT_OK_AND_ASSIGN(auto plan, TransposePlan::Create(options));
    VLOG(1) << plan->ToString();
    xla::Array<T> untiled_input(test.dims);
    untiled_input.FillIota(0);
    xla::Array<T> expected_untiled_output(output_dims);
    TransposeUsingEigen(untiled_input.data(), expected_untiled_output.data(),
                        test.dims, output_dims, test.permutation);

    auto tiled_input = TileArray(untiled_input, test.input_tiling);
    auto expected_tiled_output =
        TileArray(expected_untiled_output, test.output_tiling);

    std::vector<T> output(
        SizeOfTiledArray(plan->OutputDims(), test.output_tiling), -1);
    plan->Execute(
        tiled_input.data(), output.data(),
        [&](std::function<void()> fn) { threadpool.Schedule(std::move(fn)); });

    EXPECT_EQ(expected_tiled_output, output);
  }
};

TEST_P(TransposeTest, TransposeInt8) { TestTranspose<int8_t>(1); }
TEST_P(TransposeTest, TransposeInt16) { TestTranspose<int16_t>(1); }
TEST_P(TransposeTest, TransposeInt32) { TestTranspose<int32_t>(1); }
TEST_P(TransposeTest, TransposeInt64) { TestTranspose<int64_t>(1); }
TEST_P(TransposeTest, TransposeInt128) { TestTranspose<absl::int128>(1); }

TEST_P(TransposeTest, ParallelTransposeInt8) { TestTranspose<int8_t>(16); }
TEST_P(TransposeTest, ParallelTransposeInt32) { TestTranspose<int32_t>(16); }

INSTANTIATE_TEST_SUITE_P(TransposeTestInstance, TransposeTest,
                         ::testing::ValuesIn(GetTransposeTestCases()));

TEST(TransposeTest, NegativeStrides1D) {
  int64_t n = 10;
  std::vector<int32_t> input(n);
  std::vector<int32_t> output(n);
  std::vector<int32_t> expected(n);
  absl::c_iota(input, int32_t{7});
  std::iota(expected.rbegin(), expected.rend(), 7);
  std::vector<int64_t> dims = {n};
  std::vector<int64_t> permutation = {0};
  TransposePlan::Options options;
  options.elem_size_in_bytes = sizeof(int32_t);
  options.dims = dims;
  options.permutation = permutation;
  std::vector<int64_t> strides = {-int64_t{sizeof(int32_t)}};
  options.input_layout = TransposePlan::Striding{strides};
  TF_ASSERT_OK_AND_ASSIGN(auto plan, TransposePlan::Create(options));
  plan->Execute(input.data() + (n - 1), output.data());
  EXPECT_EQ(expected, output);
}

TEST(TransposeTest, NegativeStrides2D) {
  xla::Array<int16_t> input = {
      {1, 2, 3, 4},
      {5, 6, 7, 8},
      {9, 10, 11, 12},
  };
  xla::Array<int16_t> expected = {
      {4, 8, 12},
      {3, 7, 11},
      {2, 6, 10},
      {1, 5, 9},
  };
  xla::Array<int16_t> output({4, 3});
  std::vector<int64_t> dims = {3, 4};
  std::vector<int64_t> permutation = {1, 0};
  TransposePlan::Options options;
  options.elem_size_in_bytes = sizeof(int16_t);
  options.dims = dims;
  options.permutation = permutation;
  std::vector<int64_t> strides = {4 * sizeof(int16_t),
                                  -int64_t{sizeof(int16_t)}};
  options.input_layout = TransposePlan::Striding{strides};
  TF_ASSERT_OK_AND_ASSIGN(auto plan, TransposePlan::Create(options));
  plan->Execute(input.data() + 3, output.data());
  EXPECT_EQ(expected, output);
}

static std::vector<TransposeTestCase> BenchmarkCases() {
  return std::vector<TransposeTestCase>{
      TransposeTestCase(/*dims=*/{256, 256},
                        /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{512, 512},
                        /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{1024, 1024},
                        /*permutation=*/{1, 0}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{0, 2, 1}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{1, 0, 2}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{1, 2, 0}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{2, 0, 1}),
      TransposeTestCase(/*dims=*/{256, 256, 256},
                        /*permutation=*/{2, 1, 0}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{0, 2, 1}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{1, 0, 2}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{1, 2, 0}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{2, 0, 1}),
      TransposeTestCase(/*dims=*/{512, 512, 512},
                        /*permutation=*/{2, 1, 0}),
      TransposeTestCase(/*dims=*/{64, 224, 224, 3},
                        /*permutation=*/{1, 2, 3, 0}),
      TransposeTestCase(/*dims=*/{256, 64, 64, 3},
                        /*permutation=*/{1, 3, 2, 0}),
  };
}

template <typename T>
void BM_Eigen(const TransposeTestCase& bm, int parallelism,
              ::testing::benchmark::State& state) {
  CHECK_EQ(parallelism, 1);
  Array<T> input(bm.dims);
  input.FillIota(0);
  std::vector<int64_t> output_dims = Permute(bm.dims, bm.permutation);
  Array<T> output(output_dims);
  for (auto s : state) {
    TransposeUsingEigen(input.data(), output.data(), bm.dims, output_dims,
                        bm.permutation);
    tsl::testing::DoNotOptimize(output);
  }
}
static void BM_Eigen_uint8(const TransposeTestCase& bm, int parallelism,
                           ::testing::benchmark::State& state) {
  BM_Eigen<uint8_t>(std::move(bm), parallelism, state);
}
static void BM_Eigen_float(const TransposeTestCase& bm, int parallelism,
                           ::testing::benchmark::State& state) {
  BM_Eigen<float>(bm, parallelism, state);
}

template <typename T>
void BM_Transpose(const TransposeTestCase& bm, int parallelism,
                  ::testing::benchmark::State& state) {
  TransposePlan::Options options;
  options.elem_size_in_bytes = sizeof(T);
  options.dims = bm.dims;
  options.permutation = bm.permutation;
  options.input_layout = TransposePlan::Tiling{};
  options.output_tiling = TransposePlan::Tiling{};
  options.transformation = TransposePlan::Transformation::kNone;
  options.num_threads = parallelism;
  TF_ASSERT_OK_AND_ASSIGN(auto plan, TransposePlan::Create(options));
  Array<T> input(bm.dims);
  input.FillIota(0);
  std::vector<int64_t> output_dims = Permute(bm.dims, bm.permutation);
  Array<T> output(output_dims);
  tsl::thread::ThreadPool threadpool(tsl::Env::Default(), "Transpose",
                                     parallelism);
  for (auto s : state) {
    plan->Execute(input.data(), output.data(), [&](std::function<void()> fn) {
      threadpool.Schedule(std::move(fn));
    });
    tsl::testing::DoNotOptimize(output);
  }
}
static void BM_Transpose_uint8(const TransposeTestCase& bm, int parallelism,
                               ::testing::benchmark::State& state) {
  BM_Transpose<uint8_t>(bm, parallelism, state);
}
static void BM_Transpose_float(const TransposeTestCase& bm, int parallelism,
                               ::testing::benchmark::State& state) {
  BM_Transpose<float>(bm, parallelism, state);
}

static void* benchmarks = []() {
  using BenchmarkFn =
      void (*)(const TransposeTestCase&, int, testing::benchmark::State&);
  std::vector<std::tuple<std::string, BenchmarkFn, std::vector<int>>> variants =
      {
          {"BM_Eigen_uint8", BM_Eigen_uint8, {1}},
          {"BM_Transpose_uint8", BM_Transpose_uint8, {1, 4, 8}},  //
          {"BM_Eigen_float", BM_Eigen_float, {1}},
          {"BM_Transpose_float", BM_Transpose_float, {1, 4, 8}},  //
  };
  auto benchmark_cases = BenchmarkCases();
  for (const auto& benchmark_case : benchmark_cases) {
    for (const auto& variant : variants) {
      for (int num_threads : std::get<2>(variant)) {
        std::string name =
            absl::StrCat(std::get<0>(variant), "_threads_", num_threads, "_",
                         absl::StrJoin(benchmark_case.dims, "_"), "_perm_",
                         absl::StrJoin(benchmark_case.permutation, "_"));

        TransposeTestCase testcase = benchmark_case;
        BenchmarkFn fn = std::get<1>(variant);
        benchmark::RegisterBenchmark(
            name.c_str(), [fn, num_threads, testcase](benchmark::State& state) {
              fn(testcase, num_threads, state);
            });
      }
    }
  }
  return nullptr;
}();

TEST(TransposePlanCache, Basics) {
  std::vector<int64_t> dims = {1, 2, 3};
  std::vector<int64_t> permutation_210 = {2, 1, 0};
  std::vector<int64_t> permutation_120 = {1, 2, 0};
  std::vector<int64_t> permutation_012 = {0, 1, 2};
  TransposePlanCache cache(2);
  TransposePlan::Options o;
  o.elem_size_in_bytes = 4;
  o.dims = dims;
  o.permutation = permutation_210;
  TF_ASSERT_OK_AND_ASSIGN(auto p1, cache.GetOrCreate(o));
  TF_ASSERT_OK_AND_ASSIGN(auto p1a, cache.GetOrCreate(o));
  EXPECT_TRUE(p1.get() == p1a.get());
  TransposePlan::Options o2;
  o2.elem_size_in_bytes = 4;
  o2.dims = dims;
  o2.permutation = permutation_120;
  TF_ASSERT_OK_AND_ASSIGN(auto p2, cache.GetOrCreate(o2));
  EXPECT_TRUE(p1.get() != p2.get());
  TransposePlan::Options o3;
  o3.elem_size_in_bytes = 4;
  o3.dims = dims;
  o3.permutation = permutation_012;
  TF_ASSERT_OK_AND_ASSIGN(auto p3, cache.GetOrCreate(o3));
  EXPECT_TRUE(p3.get() != p1.get());
  TF_ASSERT_OK_AND_ASSIGN(auto p1b, cache.GetOrCreate(o));
  EXPECT_TRUE(p1.get() != p1b.get());
}

}  // namespace xla
