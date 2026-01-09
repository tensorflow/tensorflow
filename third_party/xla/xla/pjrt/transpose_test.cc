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
#include <cstring>
#include <functional>
#include <numeric>
#include <ostream>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/flags/flag.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/log/log_streamer.h"
#include "absl/numeric/int128.h"
#include "absl/random/random.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/array.h"
#include "xla/hlo/testlib/test.h"
#include "xla/permutation_util.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/protobuf/error_codes.pb.h"
#include "xla/util.h"

ABSL_FLAG(int, transpose_num_random_testcases, 10,
          "Number of random testcases");

namespace xla {

class TestTransposePlan : public TransposePlan {
 public:
  using Loop = TransposePlan::Loop;
  using TransposePlan::ChooseLoopOrder;
  using TransposePlan::CoalesceLoops;
  using TransposePlan::RemoveTrivialLoops;
  using TransposePlan::set_inner_kernel_is_memcpy;
};

TEST(TransposeTest, ChooseLoopOrderInterleaving) {
  using Loop = TestTransposePlan::Loop;
  std::vector<Loop> loops;
  loops.push_back(Loop{/*dim_in_a=*/0, /*tile_interior=*/false,
                       /*dim_size=*/100,
                       /*tile_size=*/10, /*lda=*/1000, /*ldb=*/1000});
  loops.push_back(Loop{/*dim_in_a=*/0, /*tile_interior=*/true, /*dim_size=*/100,
                       /*tile_size=*/10, /*lda=*/10, /*ldb=*/10});
  loops.push_back(Loop{/*dim_in_a=*/1, /*tile_interior=*/false,
                       /*dim_size=*/100,
                       /*tile_size=*/10, /*lda=*/500, /*ldb=*/500});
  loops.push_back(Loop{/*dim_in_a=*/1, /*tile_interior=*/true, /*dim_size=*/100,
                       /*tile_size=*/10, /*lda=*/5, /*ldb=*/5});

  TestTransposePlan plan;
  plan.set_inner_kernel_is_memcpy(false);
  plan.ChooseLoopOrder(loops);

  // The test verifies that we order tile interiors after tile exteriors.
  ASSERT_EQ(loops.size(), 4);
  EXPECT_EQ(loops[0].dim_in_a, 0);
  EXPECT_EQ(loops[0].tile_interior, false);
  EXPECT_EQ(loops[1].dim_in_a, 1);
  EXPECT_EQ(loops[1].tile_interior, false);
  EXPECT_EQ(loops[2].dim_in_a, 0);
  EXPECT_EQ(loops[2].tile_interior, true);
  EXPECT_EQ(loops[3].dim_in_a, 1);
  EXPECT_EQ(loops[3].tile_interior, true);
}

TEST(TransposeTest, ChooseLoopOrderHardConstraints) {
  using Loop = TestTransposePlan::Loop;
  std::vector<Loop> loops;
  // Dim 0: Exterior
  loops.push_back(Loop{/*dim_in_a=*/0, /*tile_interior=*/false,
                       /*dim_size=*/100,
                       /*tile_size=*/10, /*lda=*/10, /*ldb=*/10});
  // Dim 0: Interior (but with very small stride, so it wants to be outer)
  loops.push_back(Loop{/*dim_in_a=*/0, /*tile_interior=*/true, /*dim_size=*/100,
                       /*tile_size=*/10, /*lda=*/1000, /*ldb=*/1000});

  // Dim 1: Memcpy inner dim (must be last)
  loops.push_back(Loop{/*dim_in_a=*/1, /*tile_interior=*/false,
                       /*dim_size=*/100,
                       /*tile_size=*/100, /*lda=*/1, /*ldb=*/1,
                       /*is_inner_dim_in_a=*/true});

  // Dim 2: Regular loop
  loops.push_back(Loop{/*dim_in_a=*/2, /*tile_interior=*/false,
                       /*dim_size=*/100,
                       /*tile_size=*/100, /*lda=*/50, /*ldb=*/50});

  TestTransposePlan plan;
  plan.set_inner_kernel_is_memcpy(true);
  plan.ChooseLoopOrder(loops);

  ASSERT_EQ(loops.size(), 4);

  EXPECT_EQ(loops[0].dim_in_a, 2);

  EXPECT_EQ(loops[1].dim_in_a, 0);
  EXPECT_EQ(loops[1].tile_interior, false);

  EXPECT_EQ(loops[2].dim_in_a, 0);
  EXPECT_EQ(loops[2].tile_interior, true);

  EXPECT_EQ(loops[3].dim_in_a, 1);
  EXPECT_EQ(loops[3].is_inner_dim_in_a, true);
}

TEST(TransposeTest, RemoveTrivialLoops) {
  using Loop = TestTransposePlan::Loop;
  std::vector<Loop> loops;
  // Exterior loop, trivial (size 1)
  loops.push_back(Loop{/*dim_in_a=*/0, /*tile_interior=*/false, /*dim_size=*/1,
                       /*tile_size=*/1});
  // Exterior loop, trivial (dim_size == tile_size, 1 tile)
  loops.push_back(Loop{/*dim_in_a=*/1, /*tile_interior=*/false, /*dim_size=*/10,
                       /*tile_size=*/10});
  // Exterior loop, non-trivial
  loops.push_back(Loop{/*dim_in_a=*/2, /*tile_interior=*/false, /*dim_size=*/10,
                       /*tile_size=*/2});
  // Interior loop, trivial (size 1)
  loops.push_back(Loop{/*dim_in_a=*/3, /*tile_interior=*/true, /*dim_size=*/10,
                       /*tile_size=*/1});
  // Interior loop, non-trivial
  loops.push_back(Loop{/*dim_in_a=*/4, /*tile_interior=*/true, /*dim_size=*/10,
                       /*tile_size=*/10});
  // Trivial loop (size 1) but preserved because it is inner dim
  loops.push_back(Loop{/*dim_in_a=*/5, /*tile_interior=*/false, /*dim_size=*/1,
                       /*tile_size=*/1, /*lda=*/1, /*ldb=*/1,
                       /*is_inner_dim_in_a=*/true,
                       /*is_inner_dim_in_b=*/false});

  TestTransposePlan::RemoveTrivialLoops(loops);

  ASSERT_EQ(loops.size(), 3);
  // Expect loop 2 (Exterior non-trivial)
  EXPECT_EQ(loops[0].dim_in_a, 2);
  EXPECT_EQ(loops[0].tile_interior, false);
  // Expect loop 4 (Interior non-trivial)
  EXPECT_EQ(loops[1].dim_in_a, 4);
  EXPECT_EQ(loops[1].tile_interior, true);
  // Expect loop 5 (Trivial but preserved)
  EXPECT_EQ(loops[2].dim_in_a, 5);
  EXPECT_EQ(loops[2].is_inner_dim_in_a, true);
}

TEST(TransposeTest, CoalesceLoops) {
  using Loop = TestTransposePlan::Loop;
  std::vector<Loop> loops;

  // Case 1: Compatible untiled loops
  // Outer: size 4, stride 20 (inner size 5 * inner stride 4)
  loops.push_back(Loop{/*dim_in_a=*/0, /*tile_interior=*/false, /*dim_size=*/4,
                       /*tile_size=*/1, /*lda=*/20, /*ldb=*/400});
  // Inner: size 5, stride 4
  loops.push_back(Loop{/*dim_in_a=*/1, /*tile_interior=*/false, /*dim_size=*/5,
                       /*tile_size=*/1, /*lda=*/4, /*ldb=*/80});

  TestTransposePlan::CoalesceLoops(loops);

  ASSERT_EQ(loops.size(), 1);
  EXPECT_EQ(loops[0].dim_size, 20);
  EXPECT_EQ(loops[0].tile_size, 1);
  EXPECT_EQ(loops[0].lda, 4);
  EXPECT_EQ(loops[0].ldb, 80);

  // Case 2: Incompatible strides
  loops.clear();
  loops.push_back(Loop{/*dim_in_a=*/0, /*tile_interior=*/false, /*dim_size=*/4,
                       /*tile_size=*/1, /*lda=*/21,
                       /*ldb=*/400});  // lda mismatch
  loops.push_back(Loop{/*dim_in_a=*/1, /*tile_interior=*/false, /*dim_size=*/5,
                       /*tile_size=*/1, /*lda=*/4, /*ldb=*/80});

  TestTransposePlan::CoalesceLoops(loops);
  EXPECT_EQ(loops.size(), 2);

  // Case 3: Compatible tiled interior
  loops.clear();
  // Outer interior: tile_size 4, lda 16
  loops.push_back(Loop{/*dim_in_a=*/0, /*tile_interior=*/true, /*dim_size=*/100,
                       /*tile_size=*/4, /*lda=*/16, /*ldb=*/320});
  // Inner interior: tile_size 4, lda 4
  loops.push_back(Loop{/*dim_in_a=*/1, /*tile_interior=*/true, /*dim_size=*/100,
                       /*tile_size=*/4, /*lda=*/4, /*ldb=*/80});

  TestTransposePlan::CoalesceLoops(loops);
  ASSERT_EQ(loops.size(), 1);
  EXPECT_EQ(loops[0].tile_size, 16);
  EXPECT_EQ(loops[0].tile_interior, true);
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
  options.input_tiling = TransposePlan::Tiling{input_tiling};
  options.output_tiling = TransposePlan::Tiling{output_tiling};
  auto plan = TransposePlan::Create(options);
  EXPECT_EQ(plan.status().code(), tsl::error::UNIMPLEMENTED);
  EXPECT_THAT(
      plan.status().message(),
      testing::HasSubstr(
          "Only one of the input and output may have a non-trivial tiling"));
}

TEST(TransposeTest, LargeDimensions) {
  std::vector<int64_t> dims = {3ll << 30};
  std::vector<int64_t> permutation = {0};

  TransposePlan::Options options;
  options.elem_size_in_bytes = 8;
  options.dims = dims;
  options.permutation = permutation;
  options.transformation = TransposePlan::Transformation::kNone;
  TF_EXPECT_OK(TransposePlan::Create(options).status());
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

// Helper to pad tiling to match shape rank. (Suffix alignment).
std::vector<int64_t> PadTiling(absl::Span<int64_t const> shape,
                               absl::Span<int64_t const> tiling) {
  CHECK_LE(tiling.size(), shape.size());
  std::vector<int64_t> full_tiling(shape.size(), 1);
  absl::c_copy(tiling, full_tiling.end() - tiling.size());
  return full_tiling;
}

std::vector<int64_t> ComputeDefaultStrides(
    absl::Span<int64_t const> shape, absl::Span<int64_t const> full_tiling,
    int elem_size_bytes) {
  CHECK_EQ(full_tiling.size(), shape.size());
  std::vector<int64_t> strides(shape.size());
  int64_t stride = elem_size_bytes;
  for (int64_t t : full_tiling) {
    stride *= t;
  }

  for (int i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= CeilOfRatio(shape[i], full_tiling[i]);
  }
  return strides;
}

// Converts a multidimensional index `indices` into an array with `shape` and
// tiling `tiling` into a byte offset into a buffer.
// `striding` is the stride between tiles in bytes.
int64_t IndexToByteOffset(absl::Span<int64_t const> shape,
                          absl::Span<int64_t const> full_tiling,
                          absl::Span<int64_t const> indices,
                          absl::Span<int64_t const> striding,
                          int elem_size_bytes) {
  CHECK_EQ(full_tiling.size(), shape.size());
  CHECK_EQ(shape.size(), indices.size());
  CHECK_EQ(shape.size(), striding.size());

  int64_t intra_tile_stride = elem_size_bytes;
  int64_t offset = 0;

  // Strides within a tiling are always the default strides.
  for (int i = shape.size() - 1; i >= 0; --i) {
    offset += (indices[i] % full_tiling[i]) * intra_tile_stride;
    intra_tile_stride *= full_tiling[i];
  }
  // Strides outside a tiling are the input strides.
  for (size_t i = 0; i < shape.size(); ++i) {
    int64_t outer_idx = indices[i] / full_tiling[i];
    offset += outer_idx * striding[i];
  }
  return offset;
}

// Computes the size in bytes of a tiled array.
int64_t SizeOfTiledArray(absl::Span<int64_t const> shape,
                         absl::Span<int64_t const> tiling,
                         absl::Span<int64_t const> striding, int64_t elem_size,
                         int64_t* min_offset = nullptr) {
  for (int64_t dim : shape) {
    if (dim == 0) {
      if (min_offset) {
        *min_offset = 0;
      }
      return 0;
    }
  }
  std::vector<int64_t> full_tiling = PadTiling(shape, tiling);
  int64_t tile_size =
      absl::c_accumulate(full_tiling, int64_t{1}, std::multiplies<int64_t>());
  int64_t min_offset_bytes = 0;
  int64_t max_offset_bytes = 0;
  for (int i = 0; i < shape.size(); ++i) {
    int64_t last_idx = (shape[i] - 1) / full_tiling[i];
    int64_t term = last_idx * striding[i];
    if (striding[i] > 0) {
      max_offset_bytes += term;
    } else {
      min_offset_bytes += term;
    }
  }
  max_offset_bytes += tile_size * elem_size;
  if (min_offset) {
    *min_offset = min_offset_bytes;
  }
  return max_offset_bytes - min_offset_bytes;
}

int64_t SizeOfTiledArray(absl::Span<int64_t const> shape,
                         absl::Span<int64_t const> tiling, int64_t elem_size) {
  std::vector<int64_t> full_tiling = PadTiling(shape, tiling);
  std::vector<int64_t> striding =
      ComputeDefaultStrides(shape, full_tiling, elem_size);
  return SizeOfTiledArray(shape, tiling, striding, elem_size);
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

template <typename T>
void FillRandom(absl::Span<T> input) {
  absl::BitGen gen;
  for (auto& val : input) {
    if constexpr (std::is_same_v<T, absl::int128>) {
      val = absl::MakeInt128(absl::Uniform<uint64_t>(gen),
                             absl::Uniform<uint64_t>(gen));
    } else {
      using U = std::make_unsigned_t<T>;
      val = absl::bit_cast<T>(absl::Uniform<U>(gen));
    }
  }
}

// Reference implementation of transpose that handles tiling and striding.
template <typename T>
void ReferenceTranspose(absl::Span<int64_t const> dims,
                        absl::Span<int64_t const> permutation,
                        absl::Span<int64_t const> input_tiling,
                        absl::Span<int64_t const> input_striding,
                        absl::Span<int64_t const> output_tiling,
                        absl::Span<int64_t const> output_striding,
                        absl::Span<const T> input, absl::Span<T> output,
                        int64_t input_base_offset_bytes = 0) {
  std::vector<int64_t> output_dims = Permute(dims, permutation);
  std::vector<int64_t> indices(dims.size(), 0);
  std::vector<int64_t> output_indices(dims.size());

  const char* input_base =
      reinterpret_cast<const char*>(input.data()) + input_base_offset_bytes;
  char* output_base = reinterpret_cast<char*>(output.data());

  do {
    int64_t input_byte_offset = IndexToByteOffset(dims, input_tiling, indices,
                                                  input_striding, sizeof(T));
    T val;
    std::memcpy(&val, input_base + input_byte_offset, sizeof(T));

    for (size_t i = 0; i < dims.size(); ++i) {
      output_indices[i] = indices[permutation[i]];
    }
    int64_t output_byte_offset = IndexToByteOffset(
        output_dims, output_tiling, output_indices, output_striding, sizeof(T));
    std::memcpy(output_base + output_byte_offset, &val, sizeof(T));
  } while (BumpIndices(dims, absl::MakeSpan(indices)));
}

struct TransposeTestCase {
  TransposeTestCase(std::vector<int64_t> dims, std::vector<int64_t> permutation,
                    std::vector<int64_t> input_tiling = {},
                    std::vector<int64_t> output_tiling = {},
                    std::vector<int64_t> input_striding = {})
      : dims(std::move(dims)),
        permutation(std::move(permutation)),
        input_tiling(std::move(input_tiling)),
        input_striding(std::move(input_striding)),
        output_tiling(std::move(output_tiling)) {}

  std::vector<int64_t> dims;
  std::vector<int64_t> permutation;
  std::vector<int64_t> input_tiling;
  std::vector<int64_t> input_striding;
  std::vector<int64_t> output_tiling;

  std::string ToString() const {
    return absl::StrFormat(
        "[%s],perm=[%s],tiling=[%s]/[%s],striding=[%s]",
        absl::StrJoin(dims, ","), absl::StrJoin(permutation, ","),
        absl::StrJoin(input_tiling, ","), absl::StrJoin(output_tiling, ","),
        input_striding.empty() ? "none" : absl::StrJoin(input_striding, ","));
  }
};

std::ostream& operator<<(std::ostream& os, const TransposeTestCase& test) {
  os << test.ToString();
  return os;
}

std::vector<TransposeTestCase> GetTransposeTestCases() {
  std::vector<TransposeTestCase> cases = {
      TransposeTestCase(/*dims=*/{}, /*permutation=*/{}),
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
      TransposeTestCase(/*dims=*/{4, 6}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{2, 3},
                        /*output_tiling=*/{}, /*input_striding=*/{512, 128}),
      TransposeTestCase(/*dims=*/{13, 9}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{2, 3},
                        /*output_tiling=*/{}, /*input_striding=*/{0, 0}),
      TransposeTestCase(/*dims=*/{128, 224, 224, 3},
                        /*permutation=*/{3, 1, 2, 0},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{8, 128}),
      TransposeTestCase(/*dims=*/{129, 1234567},
                        /*permutation=*/{0, 1},
                        /*input_tiling=*/{},
                        /*output_tiling=*/{8, 128}),
      TransposeTestCase(/*dims=*/{21}, /*permutation=*/{0},
                        /*input_tiling=*/{14},
                        /*output_tiling=*/{}, /*input_striding=*/{1488}),
      TransposeTestCase(/*dims=*/{26, 81, 46}, /*permutation=*/{2, 0, 1},
                        /*input_tiling=*/{6, 5, 6},
                        /*output_tiling=*/{},
                        /*input_striding=*/{5000, 4512, 0}),
      TransposeTestCase(/*dims=*/{2, 33554433}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{33554433},
                        /*output_tiling=*/{}),
      // Negative strides
      TransposeTestCase(/*dims=*/{10}, /*permutation=*/{0},
                        /*input_tiling=*/{}, /*output_tiling=*/{},
                        /*input_striding=*/{-4}),
      TransposeTestCase(/*dims=*/{3, 4}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{}, /*output_tiling=*/{},
                        /*input_striding=*/{16, -4}),
      TransposeTestCase(/*dims=*/{3, 4}, /*permutation=*/{0, 1},
                        /*input_tiling=*/{}, /*output_tiling=*/{},
                        /*input_striding=*/{-16, -4}),

      // Negative strides with tiling
      TransposeTestCase(/*dims=*/{10}, /*permutation=*/{0},
                        /*input_tiling=*/{2}, /*output_tiling=*/{},
                        /*input_striding=*/{-32}),
      TransposeTestCase(/*dims=*/{4, 4}, /*permutation=*/{1, 0},
                        /*input_tiling=*/{2, 2}, /*output_tiling=*/{},
                        /*input_striding=*/{-32, 16}),
      TransposeTestCase(/*dims=*/{97, 44, 98}, /*permutation=*/{0, 2, 1},
                        /*input_tiling=*/{8, 3, 6}, /*output_tiling=*/{},
                        /*input_striding=*/{-18, -4, 0}),
      TransposeTestCase(/*dims=*/{24, 80}, /*permutation=*/{0, 1},
                        /*input_tiling=*/{}, /*output_tiling=*/{15, 1}),
      TransposeTestCase(/*dims=*/{51, 85, 63, 1}, /*permutation=*/{0, 3, 2, 1},
                        /*input_tiling=*/{5, 13, 16}, /*output_tiling=*/{},
                        /*input_striding=*/{}),
      TransposeTestCase(/*dims=*/{30, 94, 89, 18, 1, 1},
                        /*permutation=*/{4, 5, 3, 1, 2, 0},
                        /*input_tiling=*/{15, 14}, /*output_tiling=*/{},
                        /*input_striding=*/{-32, -56, 4248, 4784, 0, 3280}),
      TransposeTestCase(
          /*dims=*/{49, 24, 29, 95}, /*permutation=*/{1, 3, 2, 0},
          /*input_tiling=*/{15, 10, 15}, /*output_tiling=*/{},
          /*input_striding=*/{11488, -80, 0, -128}),

      TransposeTestCase(/*dims=*/{67, 16, 75, 5, 15},
                        /*permutation=*/{3, 1, 2, 0, 4},
                        /*input_tiling=*/{9, 3}, /*output_tiling=*/{},
                        /*input_striding=*/{}),
      TransposeTestCase(/*dims=*/{52, 44, 45, 96, 1, 5},
                        /*permutation=*/{5, 4, 2, 3, 1, 0},
                        /*input_tiling=*/{}, /*output_tiling=*/{},
                        /*input_striding=*/{})};
  return cases;
}

struct RandomTransposeTestCase {
  TransposeTestCase test_case;
  int elem_size;
  int parallelism;
  TransposePlan::Transformation transformation;
  TransposePlan::ChunkContiguity chunk_contiguity;
  bool use_execute_chunk;

  RandomTransposeTestCase(TransposeTestCase tc, int es, int p,
                          TransposePlan::Transformation t,
                          TransposePlan::ChunkContiguity c, bool u)
      : test_case(std::move(tc)),
        elem_size(es),
        parallelism(p),
        transformation(t),
        chunk_contiguity(c),
        use_execute_chunk(u) {}

  std::string ToString() const {
    return absl::StrFormat(
        "%s,elem_size=%d,threads=%d,transformation=%x,contiguity=%d,chunked=%d",
        test_case.ToString(), elem_size, parallelism,
        static_cast<int>(transformation), static_cast<int>(chunk_contiguity),
        use_execute_chunk);
  }
};

std::ostream& operator<<(std::ostream& os,
                         const RandomTransposeTestCase& test) {
  os << test.ToString();
  return os;
}

std::vector<RandomTransposeTestCase> GetRandomTransposeTestCases() {
  std::vector<RandomTransposeTestCase> cases;
#if defined(PLATFORM_GOOGLE)
  auto seed_seq = absl::MakeTaggedSeedSeq("RANDOM_TRANSPOSE_SEED",
                                          absl::LogInfoStreamer().stream());
  absl::BitGen bitgen(seed_seq);
#else
  absl::BitGen bitgen;
#endif

  for (int i = 0; i < absl::GetFlag(FLAGS_transpose_num_random_testcases);
       ++i) {
    bool valid = false;
    for (int retry = 0; retry < 10 && !valid; ++retry) {
      int rank = absl::Uniform<int>(bitgen, 1, 7);
      int elem_size = 1 << absl::Uniform<int>(bitgen, 0, 5);  // 1, 2, 4, 8, 16

      // Total size capped at 100MB per array.
      int64_t max_bytes = 100LL << 20;
      int64_t max_elems = max_bytes / elem_size;
      std::vector<int64_t> dims(rank);
      int64_t total_elems = 1;
      for (int j = 0; j < rank; ++j) {
        dims[j] = absl::Uniform<int>(bitgen, 1, 100);
        if (total_elems * dims[j] > max_elems) {
          dims[j] = 1;
        }
        total_elems *= dims[j];
      }

      std::vector<int64_t> permutation(rank);
      std::iota(permutation.begin(), permutation.end(), 0);
      absl::c_shuffle(permutation, bitgen);

      std::vector<int64_t> input_tiling;
      std::vector<int64_t> output_tiling;
      int tiling_mode = absl::Uniform<int>(bitgen, 0, 3);
      if (tiling_mode == 1) {
        int num_tiled = absl::Uniform<int>(bitgen, 1, std::min(rank, 3) + 1);
        input_tiling.resize(num_tiled);
        for (int j = 0; j < num_tiled; ++j) {
          input_tiling[j] = absl::Uniform<int>(bitgen, 1, 17);
        }
      } else if (tiling_mode == 2) {
        int num_tiled = absl::Uniform<int>(bitgen, 1, std::min(rank, 3) + 1);
        output_tiling.resize(num_tiled);
        for (int j = 0; j < num_tiled; ++j) {
          output_tiling[j] = absl::Uniform<int>(bitgen, 1, 17);
        }
      }

      std::vector<int64_t> striding;
      if (absl::Bernoulli(bitgen, 0.2)) {
        striding.resize(rank);
        for (int j = 0; j < rank; ++j) {
          int r = absl::Uniform<int>(bitgen, -2, 3);
          if (r == 0) {
            striding[j] = 0;
          } else if (r < 0) {
            striding[j] = -static_cast<int64_t>(elem_size) *
                          absl::Uniform<int>(bitgen, 1, 10);
          } else {
            striding[j] = static_cast<int64_t>(elem_size) *
                          absl::Uniform<int>(bitgen, 1, 1000);
          }
        }
      }

      // Check input/output sizes with tiling and striding.
      std::vector<int64_t> full_input_tiling = PadTiling(dims, input_tiling);
      std::vector<int64_t> input_striding = striding;
      if (input_striding.empty()) {
        input_striding =
            ComputeDefaultStrides(dims, full_input_tiling, elem_size);
      }
      if (SizeOfTiledArray(dims, input_tiling, input_striding, elem_size) >
          max_bytes) {
        continue;
      }

      std::vector<int64_t> output_dims = Permute(dims, permutation);
      if (SizeOfTiledArray(output_dims, output_tiling, elem_size) > max_bytes) {
        continue;
      }

      int parallelism = absl::Uniform<int>(bitgen, 1, 17);
      auto chunk_contiguity = static_cast<TransposePlan::ChunkContiguity>(
          absl::Uniform<int>(bitgen, 0, 3));
      bool use_execute_chunk = absl::Bernoulli(bitgen, 0.5);

      cases.push_back(RandomTransposeTestCase(
          TransposeTestCase(dims, permutation, input_tiling, output_tiling,
                            striding),
          elem_size, parallelism, TransposePlan::Transformation::kNone,
          chunk_contiguity, use_execute_chunk));
      valid = true;
    }
  }
  return cases;
}

template <typename T>
void ExecuteTestWithT(const TransposeTestCase& test, int parallelism,
                      TransposePlan::Transformation transformation,
                      TransposePlan::ChunkContiguity chunk_contiguity,
                      bool use_execute_chunk) {
  tsl::thread::ThreadPool threadpool(tsl::Env::Default(), "Transpose",
                                     parallelism);
  std::vector<int64_t> output_dims = Permute(test.dims, test.permutation);
  TransposePlan::Options options;
  options.elem_size_in_bytes = sizeof(T);
  options.dims = test.dims;
  options.permutation = test.permutation;
  if (!test.input_striding.empty()) {
    options.input_striding = TransposePlan::Striding{test.input_striding};
  }
  if (!test.input_tiling.empty()) {
    options.input_tiling = TransposePlan::Tiling{test.input_tiling};
  }
  if (!test.output_tiling.empty()) {
    options.output_tiling = TransposePlan::Tiling{test.output_tiling};
  }
  options.transformation = transformation;
  options.num_threads = parallelism;
  options.chunk_contiguity = chunk_contiguity;
  TF_ASSERT_OK_AND_ASSIGN(auto plan, TransposePlan::Create(options));
  VLOG(1) << plan->ToString();

  int64_t output_size_bytes =
      SizeOfTiledArray(plan->OutputDims(), test.output_tiling, sizeof(T));
  int64_t output_size = CeilOfRatio<int64_t>(output_size_bytes, sizeof(T));
  std::vector<T> output(output_size, -1);

  std::vector<int64_t> input_striding = test.input_striding;
  std::vector<int64_t> input_tiling = PadTiling(test.dims, test.input_tiling);
  if (input_striding.empty()) {
    input_striding = ComputeDefaultStrides(test.dims, input_tiling, sizeof(T));
  }
  std::vector<int64_t> output_tiling =
      PadTiling(output_dims, test.output_tiling);
  std::vector<int64_t> output_striding =
      ComputeDefaultStrides(output_dims, output_tiling, sizeof(T));

  int64_t min_offset_bytes;
  int64_t input_size_bytes = SizeOfTiledArray(
      test.dims, input_tiling, input_striding, sizeof(T), &min_offset_bytes);
  int64_t input_size = CeilOfRatio<int64_t>(input_size_bytes, sizeof(T));
  std::vector<T> input(input_size);
  FillRandom<T>(absl::MakeSpan(input));
  std::vector<T> expected_output(output_size, -1);

  int64_t input_base_offset_bytes = -min_offset_bytes;
  ReferenceTranspose<T>(test.dims, test.permutation, input_tiling,
                        input_striding, output_tiling, output_striding, input,
                        absl::MakeSpan(expected_output),
                        input_base_offset_bytes);

  if (use_execute_chunk) {
    int num_chunks = plan->Parallelism();
    int64_t input_total = input_size * static_cast<int64_t>(sizeof(T));
    int64_t output_total = output_size * static_cast<int64_t>(sizeof(T));

    // For contiguous buffers, sum of chunk sizes should equal total.
    bool has_non_default_striding = !test.input_striding.empty();
    if (chunk_contiguity == TransposePlan::ChunkContiguity::kInput &&
        !has_non_default_striding) {
      int64_t sum = 0;
      for (int i = 0; i < num_chunks; ++i) {
        sum += plan->InputChunkSizeBytes(i);
      }
      EXPECT_EQ(sum, input_total);
    }
    if (chunk_contiguity == TransposePlan::ChunkContiguity::kOutput) {
      int64_t sum = 0;
      for (int i = 0; i < num_chunks; ++i) {
        sum += plan->OutputChunkSizeBytes(i);
      }
      EXPECT_EQ(sum, output_total);
    }

    for (int chunk = 0; chunk < num_chunks; ++chunk) {
      int64_t input_offset = plan->InputChunkOffsetBytes(chunk);
      int64_t input_bytes = plan->InputChunkSizeBytes(chunk);
      int64_t output_offset = plan->OutputChunkOffsetBytes(chunk);
      int64_t output_bytes = plan->OutputChunkSizeBytes(chunk);

      std::vector<char> input_temp(input_bytes);
      std::memcpy(input_temp.data(),
                  reinterpret_cast<const char*>(input.data()) +
                      input_base_offset_bytes + input_offset,
                  input_bytes);

      if (chunk_contiguity == TransposePlan::ChunkContiguity::kOutput) {
        std::vector<char> output_temp(output_bytes, 0xFF);
        plan->ExecuteChunk(chunk, input_temp.data(), output_temp.data(),
                           /*input_is_global=*/false,
                           /*output_is_global=*/false);
        std::memcpy(reinterpret_cast<char*>(output.data()) + output_offset,
                    output_temp.data(), output_bytes);
      } else {
        plan->ExecuteChunk(chunk, input_temp.data(), output.data(),
                           /*input_is_global=*/false,
                           /*output_is_global=*/true);
      }
    }
  } else {
    plan->Execute(
        reinterpret_cast<const char*>(input.data()) + input_base_offset_bytes,
        output.data(),
        [&](std::function<void()> fn) { threadpool.Schedule(std::move(fn)); });
  }

  EXPECT_EQ(output, expected_output);
}

void ExecuteTest(const TransposeTestCase& test, int elem_size, int parallelism,
                 TransposePlan::Transformation transformation,
                 TransposePlan::ChunkContiguity chunk_contiguity,
                 bool use_execute_chunk) {
  switch (elem_size) {
    case 1:
      ExecuteTestWithT<int8_t>(test, parallelism, transformation,
                               chunk_contiguity, use_execute_chunk);
      break;
    case 2:
      ExecuteTestWithT<int16_t>(test, parallelism, transformation,
                                chunk_contiguity, use_execute_chunk);
      break;
    case 4:
      ExecuteTestWithT<int32_t>(test, parallelism, transformation,
                                chunk_contiguity, use_execute_chunk);
      break;
    case 8:
      ExecuteTestWithT<int64_t>(test, parallelism, transformation,
                                chunk_contiguity, use_execute_chunk);
      break;
    case 16:
      ExecuteTestWithT<absl::int128>(test, parallelism, transformation,
                                     chunk_contiguity, use_execute_chunk);
      break;
    default:
      LOG(FATAL) << "Unsupported elem_size: " << elem_size;
  }
}

class TransposeTest : public ::testing::TestWithParam<TransposeTestCase> {
 protected:
  template <typename T>
  void TestTranspose(int parallelism) {
    ExecuteTest(GetParam(), sizeof(T), parallelism,
                TransposePlan::Transformation::kNone,
                TransposePlan::ChunkContiguity::kNone,
                /*use_execute_chunk=*/false);
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
  options.input_striding = TransposePlan::Striding{strides};
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
  options.input_striding = TransposePlan::Striding{strides};
  TF_ASSERT_OK_AND_ASSIGN(auto plan, TransposePlan::Create(options));
  plan->Execute(input.data() + 3, output.data());
  EXPECT_EQ(expected, output);
}

// Test contiguous chunk execution modes: kNone, kInput, kOutput.
// Uses all TransposeTestCases as test inputs to verify chunked execution.
class ChunkedTransposeTest
    : public ::testing::TestWithParam<
          std::tuple<TransposeTestCase, TransposePlan::ChunkContiguity>> {};

TEST_P(ChunkedTransposeTest, ChunkedTranspose) {
  const auto& [test, mode] = GetParam();
  ExecuteTest(test, sizeof(int32_t), 4, TransposePlan::Transformation::kNone,
              mode, /*use_execute_chunk=*/true);
}

INSTANTIATE_TEST_SUITE_P(
    ChunkedTransposeTestInstance, ChunkedTransposeTest,
    ::testing::Combine(
        ::testing::ValuesIn(GetTransposeTestCases()),
        ::testing::Values(TransposePlan::ChunkContiguity::kNone,
                          TransposePlan::ChunkContiguity::kInput,
                          TransposePlan::ChunkContiguity::kOutput)));

class RandomTransposeTest
    : public ::testing::TestWithParam<RandomTransposeTestCase> {};

TEST_P(RandomTransposeTest, RandomTranspose) {
  const RandomTransposeTestCase& test = GetParam();
  LOG(INFO) << "Running random test case: " << test;
  ExecuteTest(test.test_case, test.elem_size, test.parallelism,
              test.transformation, test.chunk_contiguity,
              test.use_execute_chunk);
}

INSTANTIATE_TEST_SUITE_P(RandomTransposeTestInstance, RandomTransposeTest,
                         ::testing::ValuesIn(GetRandomTransposeTestCases()));

TEST(TransposePlanValidationTest, ValidTiling) {
  TransposePlan::Options options;
  options.elem_size_in_bytes = 1;
  std::vector<int64_t> dims = {6, 98, 9};
  options.dims = dims;
  std::vector<int64_t> permutation = {0, 1, 2};
  options.permutation = permutation;
  std::vector<int64_t> output_tiling = {1, 4};
  options.output_tiling = TransposePlan::Tiling{output_tiling};
  auto status_or_plan = TransposePlan::Create(options);
  EXPECT_TRUE(status_or_plan.ok());
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
