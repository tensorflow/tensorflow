// Copyright 2023 The OpenXLA Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "xla/python/ifrt_proxy/common/array_util.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/dtype.h"
#include "xla/python/ifrt/shape.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {
namespace proxy {

namespace {

using ::testing::ElementsAre;
using ::testing::Not;
using ::testing::TestWithParam;
using ::tsl::testing::IsOk;
using ::tsl::testing::IsOkAndHolds;

constexpr DType::Kind kF64 = DType::Kind::kF64;
constexpr DType::Kind kS32 = DType::Kind::kS32;
constexpr DType::Kind kString = DType::Kind::kString;
using Strides = std::vector<int64_t>;

TEST(DefaultByteStrides, ErrorsIfBadDtype) {
  EXPECT_THAT(DefaultByteStrides(DType(kString), Shape({1})), Not(IsOk()));
}

TEST(DefaultByteStrides, HappyCase) {
  EXPECT_THAT(DefaultByteStrides(DType(kF64), Shape({4, 3, 5})),
              IsOkAndHolds(ElementsAre(120, 40, 8)));
}

// TC represents a testcase.
struct TC {
  const std::string test_name;
  const DType::Kind dtype_kind;
  const std::vector<int64_t> shape;
  const std::optional<std::vector<int64_t>> byte_strides;
  const std::optional<size_t> expected_size;
};
std::string PrintToString(const TC& tc) { return tc.test_name; }

class ArrayMemRegionSuccess : public TestWithParam<TC> {};
INSTANTIATE_TEST_SUITE_P(
    Tests, ArrayMemRegionSuccess,
    testing::Values(
        // F64
        TC{"DefaultF64", kF64, {4, 3, 5}, std::nullopt},
        TC{"MajorToMinorStridesF64", kF64, {4, 3, 5}, Strides({120, 40, 8})},
        TC{"NotMajorToMinorF64", kF64, {3, 4, 5}, Strides({40, 120, 8})},
        TC{"TransposedF64", kF64, {5, 3, 4}, Strides({8, 40, 120})},
        // S32
        TC{"DefaultS32", kS32, {4, 3, 5}, std::nullopt},
        TC{"MajorToMinorStridesS32", kS32, {4, 3, 5}, Strides({60, 20, 4})},
        TC{"NotMajorToMinorS32", kS32, {3, 4, 5}, Strides({20, 60, 4})},
        TC{"TransposedS32", kS32, {5, 3, 4}, Strides({4, 20, 60})},
        // Scalar
        TC{"ScalarF64DefaultStrides", kF64, {}, std::nullopt},
        TC{"ScalarF64EmptyStrides", kF64, {}, Strides({})},
        // Zero elements
        TC{"NoColsDefaultStrides", kF64, {5, 0}, std::nullopt},
        TC{"NoColsStridesNonZero", kF64, {5, 0}, Strides({40, 4})},
        TC{"NoColsStridesZero", kF64, {5, 0}, Strides({0, 0})},
        TC{"NoRowsDefaultStrides", kF64, {0, 5}, std::nullopt},
        TC{"NoRowsStridesNonZero", kF64, {0, 5}, Strides({40, 4})},
        TC{"NoRowsStridesZero", kF64, {0, 5}, Strides({0, 0})},
        // Dimension with size 1
        TC{"SingleElementArbitraryStrides", kF64, {1, 1}, Strides({100, 100})},
        TC{"OneRowArbitraryColStride", kF64, {1, 5}, Strides({100, 8})},
        TC{"OneColArbitraryRowStride", kF64, {5, 1}, Strides({8, 100})},
        TC{"OneRowZeroColStride", kF64, {1, 5}, Strides({0, 8})},
        TC{"OneColZeroRowStride", kF64, {5, 1}, Strides({8, 0})},
        // Non-compact strides.
        TC{"NonCompactSingleDimension", kS32, {5}, Strides({16}), 68},
        TC{"NonCompactDim0", kS32, {4, 3, 5}, Strides({120, 20, 4}), 420},
        TC{"PaddedElements", kS32, {4, 3, 5}, Strides({120, 40, 8}), 476}),
    testing::PrintToStringParamName());
TEST_P(ArrayMemRegionSuccess, TestCase) {
  const TC tc = GetParam();
  const DType dtype(tc.dtype_kind);
  const Shape shape(tc.shape);
  const size_t expected_size = tc.expected_size.value_or(
      dtype.byte_size().value() * shape.num_elements());
  std::string data(expected_size, 'a');

  TF_ASSERT_OK_AND_ASSIGN(auto mem_region1,
                          ArrayMemRegion::FromZerothElementPointer(
                              data.data(), dtype, shape, tc.byte_strides));
  EXPECT_EQ(mem_region1.zeroth_element(), data.data());
  // Note: `EXPECT_EQ(mem_region.mem_region(), absl::string_view(data))` can
  // cause asan to complain if the expectation fails.
  EXPECT_EQ(mem_region1.mem_region().data(), data.data());
  EXPECT_EQ(mem_region1.mem_region().size(), data.size());

  TF_ASSERT_OK_AND_ASSIGN(
      auto mem_region2, ArrayMemRegion::FromMinimalMemRegion(data, dtype, shape,
                                                             tc.byte_strides));
  EXPECT_EQ(mem_region2.zeroth_element(), data.data());
  EXPECT_EQ(mem_region2.mem_region().data(), data.data());
  EXPECT_EQ(mem_region2.mem_region().size(), data.size());
}

class ArrayMemRegionFailure : public TestWithParam<TC> {};
INSTANTIATE_TEST_SUITE_P(
    Tests, ArrayMemRegionFailure,
    testing::Values(
        // Will not be supported
        TC{"OneString", kString, {}, std::nullopt},
        TC{"ManyStrings", kString, {5}, std::nullopt},
        // Currently unimplemented
        TC{"NegativeByteStrides", kS32, {4, 3, 5}, Strides({-60, -20, -4})},
        TC{"ZeroByteStride", kS32, {5, 5}, Strides({0, 0})},
        TC{"SmallerByteStrideThanDataType", kS32, {5, 5}, Strides({1, 1})},
        TC{"ByteStrideIndivisibleByDataType", kS32, {5, 5}, Strides({7, 7})},
        // Bad arguments
        TC{"NegativeShapeDimension", kS32, {-5, -5}, Strides({20, 4})}),
    testing::PrintToStringParamName());
TEST_P(ArrayMemRegionFailure, TestCase) {
  const TC tc = GetParam();
  const DType dtype(tc.dtype_kind);
  const Shape shape(tc.shape);
  char const* kSomeAddr = reinterpret_cast<char*>(1UL << 48);

  auto mem_region1 = ArrayMemRegion::FromZerothElementPointer(
      /*zeroth_element=*/kSomeAddr, dtype, shape, tc.byte_strides);
  EXPECT_THAT(mem_region1.status(), Not(IsOk()));

  const size_t kSomeSize = 1024;
  auto mem_region2 = ArrayMemRegion::FromMinimalMemRegion(
      absl::string_view(kSomeAddr, kSomeSize), dtype, shape, tc.byte_strides);
  EXPECT_THAT(mem_region2.status(), Not(IsOk()));
}

TEST(ArrayMemRegion, FromBadMemRegionSizeFails) {
  const DType kDType(kS32);
  const Shape kShape({5, 5});
  const size_t kDataBytes = kDType.byte_size().value() * kShape.num_elements();

  const size_t kExtraSuffixBytes = 10;
  std::string data_with_extra_suffix(kDataBytes + kExtraSuffixBytes, 'a');

  // If we know that the zeroth_element is at the beginning, then we
  // can construct the ArrayMemoryRegion; the constructed ArrayMemoryRegion
  // will not contain the suffix.
  TF_ASSERT_OK_AND_ASSIGN(
      auto mem_region1,
      ArrayMemRegion::FromZerothElementPointer(
          /*zeroth_element=*/data_with_extra_suffix.data(), kDType, kShape,
          /*byte_strides=*/std::nullopt));
  EXPECT_EQ(mem_region1.mem_region().data(), data_with_extra_suffix.data());
  EXPECT_EQ(mem_region1.zeroth_element(), data_with_extra_suffix.data());
  EXPECT_LT(mem_region1.mem_region().size(), data_with_extra_suffix.size());
  EXPECT_EQ(mem_region1.mem_region().size(), kDataBytes);

  // But given the data_with_extra_suffix region, we cannot discover where
  // within it the zeroth-element points to, so we cannot construct an
  // ArrayMemoryRegion from it.
  auto mem_region2 = ArrayMemRegion::FromMinimalMemRegion(
      data_with_extra_suffix, kDType, kShape,
      /*byte_strides=*/std::nullopt);
  EXPECT_THAT(mem_region2.status(), Not(IsOk()));

  // Similarly, if we provided `FromMinimalMemRegion` a `data` that was smaller
  // than what the constructed `ArrayMemoryRegion` should point to, that will
  // be detected as an error.
  std::string data_without_some_bytes(kDataBytes - kExtraSuffixBytes, 'a');
  auto mem_region3 = ArrayMemRegion::FromMinimalMemRegion(
      data_without_some_bytes, kDType, kShape,
      /*byte_strides=*/std::nullopt);
  EXPECT_THAT(mem_region3.status(), Not(IsOk()));
}

}  // namespace

}  // namespace proxy
}  // namespace ifrt
}  // namespace xla
