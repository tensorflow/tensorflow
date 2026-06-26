/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "absl/status/status_macros.h"
#include "xla/backends/gpu/libraries/cub/cub_scratch_size_deviceless_lookup.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/backends/gpu/transforms/sort_rewriter.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/hlo_runner_interface.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

constexpr int kTestDataSize = 10000;

// Common base class to share configuration and helpers.
class CubSortTestBase
    : public HloInterpreterReferenceMixin<HloPjRtGpuTestBase> {
 public:
  void SetUp() override {
    HloInterpreterReferenceMixin<HloPjRtGpuTestBase>::SetUp();
    SortRewriter::SetSortModeForTestingOnly(SortRewriter::Mode::kAlways);
  }

  absl::StatusOr<bool> IsRewrittenToUseCubSort(absl::string_view hlo_text) {
    ASSIGN_OR_RETURN(std::unique_ptr<HloModule> optimized_module,
                     GetOptimizedModule(hlo_text));

    for (const auto& pass_metadata :
         optimized_module->metadata()->proto().pass_metadata()) {
      if (pass_metadata.pass_name() == "sort-rewriter") {
        return pass_metadata.module_changed();
      }
    }
    return false;
  }

  std::optional<std::string> DevicelessSkipReason(
      PrimitiveType key_type, std::optional<PrimitiveType> value_type,
      int batch_size, int64_t num_items = kTestDataSize) {
    std::string device_name = device_description().name();
    auto cub_version = device_description().cub_version();

    int32_t key_type_size = ShapeUtil::ByteSizeOfPrimitiveType(key_type);
    std::optional<int32_t> value_type_size;
    if (value_type.has_value()) {
      value_type_size = ShapeUtil::ByteSizeOfPrimitiveType(*value_type);
    }

    auto lookup_or = CubScratchSizeDevicelessLookup::GetInstance();
    if (!lookup_or.ok()) {
      return absl::StrCat("Failed to get CubScratchSizeDevicelessLookup: ",
                          lookup_or.status().ToString());
    }
    const CubScratchSizeDevicelessLookup& lookup = *lookup_or;

    if (!lookup.CanLookup(cub_version, device_name, key_type_size,
                          value_type_size, num_items, batch_size)) {
      return absl::Substitute(
          "Skipping deviceless CUB sort test because database lacks sizing "
          "entries for device '$0' with key_size=$1, val_size=$2, items=$3, "
          "batch=$4",
          device_name, key_type_size, value_type_size.value_or(0), num_items,
          batch_size);
    }
    return std::nullopt;
  }
};

// ----- Sort keys

class CubSortKeysTest : public CubSortTestBase {};

class CubSortKeysParameterizedTest
    : public CubSortKeysTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, bool, int, bool>> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = CubSortKeysTest::GetDebugOptionsForTest();
    if (std::get<3>(GetParam())) {
      options.set_xla_gpu_deviceless_cub_mode(
          DebugOptions::DEVICELESS_CUB_FORCE_ON_NO_FALLBACK);
    }
    return options;
  }
};

class CubSortKeysSpecialOrderingTest
    : public CubSortTestBase,
      public ::testing::WithParamInterface<bool> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = CubSortTestBase::GetDebugOptionsForTest();
    if (GetParam()) {
      options.set_xla_gpu_deviceless_cub_mode(
          DebugOptions::DEVICELESS_CUB_FORCE_ON_NO_FALLBACK);
    }
    return options;
  }
};

TEST_F(CubSortKeysTest, AlwaysUsesCubSort) {
  EXPECT_EQ(SortRewriter::SortMode(), SortRewriter::Mode::kAlways);
}

TEST_P(CubSortKeysParameterizedTest, CompareToReference) {
  if (std::get<3>(GetParam())) {
    if (auto skip_reason = DevicelessSkipReason(
            std::get<0>(GetParam()), std::nullopt, std::get<2>(GetParam()))) {
      GTEST_SKIP() << *skip_reason;
    }
  }
  int batch_size = std::get<2>(GetParam());
  int segment_size = kTestDataSize / batch_size;

  const char* kHloTpl = R"(
HloModule TestSortKeys

compare {
  %lhs = $0[] parameter(0)
  %rhs = $0[] parameter(1)
  ROOT %comp = pred[] compare(%lhs, %rhs), direction=$1
}

ENTRY main {
  %input = $0[$2,$3] parameter(0)
  ROOT %sort = $0[$2,$3] sort(%input), dimensions={1}, to_apply=compare
})";
  std::string hlo_str = absl::Substitute(
      kHloTpl,
      primitive_util::LowercasePrimitiveTypeName(std::get<0>(GetParam())),
      std::get<1>(GetParam()) ? "LT" : "GT", batch_size, segment_size);

  ASSERT_OK_AND_ASSIGN(bool rewritten, IsRewrittenToUseCubSort(hlo_str));
  EXPECT_TRUE(rewritten);

  EXPECT_TRUE(RunAndCompare(hlo_str, ErrorSpec{0, 0}));
}

TEST_P(CubSortKeysSpecialOrderingTest, CompareToReferenceNumpyOrderGt) {
  if (GetParam()) {
    if (auto skip_reason = DevicelessSkipReason(BF16, std::nullopt, 1, 16)) {
      GTEST_SKIP() << *skip_reason;
    }
  }
  constexpr char kHlo[] = R"(
numpy_order_comparator {
  lhs = bf16[] parameter(0)
  lhs_is_nan = pred[] compare(lhs, lhs), direction=NE
  c_nan = bf16[] constant(nan)
  c_zero = bf16[] constant(0)
  lhs_is_zero = pred[] compare(lhs, c_zero), direction=EQ
  lhs_no_neg_zero = bf16[] select(lhs_is_zero, c_zero, lhs)
  lhs_no_neg_zero_or_nan = bf16[] select(lhs_is_nan, c_nan, lhs_no_neg_zero)
  rhs = bf16[] parameter(1)
  rhs_is_nan = pred[] compare(rhs, rhs), direction=NE
  rhs_is_zero = pred[] compare(rhs, c_zero), direction=EQ
  rhs_no_neg_zero = bf16[] select(rhs_is_zero, c_zero, rhs)
  rhs_no_neg_zero_or_nan = bf16[] select(rhs_is_nan, c_nan, rhs_no_neg_zero)
  ROOT compare.20017 = pred[] compare(lhs_no_neg_zero_or_nan, rhs_no_neg_zero_or_nan), direction=GT, type=TOTALORDER
}

ENTRY main {
  p = bf16[8] parameter(0)
  nans_and_zeros = bf16[8] constant({nan, -nan, nan, -nan, 0.0, -0.0, 0.0, -0.0})
  values = bf16[16] concatenate(p, nans_and_zeros), dimensions={0}
  ROOT sort = bf16[16] sort(values), dimensions={0}, is_stable=true, to_apply=numpy_order_comparator
})";
  ASSERT_OK_AND_ASSIGN(bool rewritten, IsRewrittenToUseCubSort(kHlo));
  EXPECT_TRUE(rewritten);

  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0, 0}));
}

// Verify that Cub Device Radix sort honors XLA's total order semantics:
// -NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN
// https://openxla.org/xla/operation_semantics#element-wise_comparison_operations
//
// Starting with release 1.12.0, Cub Device Radix sort treats +0.0 and -0.0
// equivalently. See https://github.com/NVIDIA/cub/releases/tag/1.12.0
// This test may break when upgrading to a newer version of Cub.
TEST_P(CubSortKeysSpecialOrderingTest, CompareToReferenceTotalOrderLt) {
  if (GetParam()) {
    if (auto skip_reason = DevicelessSkipReason(F32, std::nullopt, 1, 16)) {
      GTEST_SKIP() << *skip_reason;
    }
  }
  constexpr char kHlo[] = R"(
compare {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT comp = pred[] compare(lhs, rhs), direction=LT, type=TOTALORDER
}

ENTRY main {
  p = f32[8] parameter(0)
  nans_and_zeros = f32[8] constant({nan, -nan, nan, -nan, 0.0, -0.0, 0.0, -0.0})
  values = f32[16] concatenate(p, nans_and_zeros), dimensions={0}
  ROOT sort = f32[16] sort(values), dimensions={0}, is_stable=true, to_apply=compare
})";
  ASSERT_OK_AND_ASSIGN(bool rewritten, IsRewrittenToUseCubSort(kHlo));
  EXPECT_TRUE(rewritten);

  EXPECT_TRUE(RunAndCompare(kHlo, ErrorSpec{0, 0}));
}
#if defined(XLA_CUB_TEST_DEVICELESS_ONLY)
constexpr std::optional<bool> kCubTestDevicelessValue = true;
#elif defined(XLA_CUB_TEST_NO_DEVICELESS)
constexpr std::optional<bool> kCubTestDevicelessValue = false;
#else
constexpr std::optional<bool> kCubTestDevicelessValue = std::nullopt;
#endif

inline ::testing::internal::ParamGenerator<bool> GetCubTestDevicelessValues() {
  if (kCubTestDevicelessValue.has_value()) {
    return ::testing::Values(*kCubTestDevicelessValue);
  }
  return ::testing::Bool();
}

INSTANTIATE_TEST_SUITE_P(CubSort, CubSortKeysSpecialOrderingTest,
                         GetCubTestDevicelessValues(),
                         [](const ::testing::TestParamInfo<bool>& info) {
                           return info.param ? "deviceless" : "device";
                         });

// This test verifies an issue where sort was launched on the wrong stream,
// causing subtle timing bugs: b/347239322.
TEST_P(CubSortKeysParameterizedTest, SortWithSlice) {
  if (std::get<3>(GetParam())) {
    if (auto skip_reason = DevicelessSkipReason(
            std::get<0>(GetParam()), std::nullopt, std::get<2>(GetParam()))) {
      GTEST_SKIP() << *skip_reason;
    }
  }
  constexpr char kHloTpl[] = R"(
cmp {
    p0 = $0[] parameter(0)
    p1 = $0[] parameter(1)
    ROOT cmp = pred[] compare(p0, p1), direction=$1
}

ENTRY m {
    param = $0[$2,$3] parameter(0)
    sort = $0[$2,$3] sort(param), dimensions={1}, is_stable=false, to_apply=cmp
    add = $0[$2,$3] add(sort, sort)  // Avoid matching the topk pattern.
    ROOT slice = $0[$2,10] slice(add), slice={[0:$2],[0:10]}
})";

  int batch_size = std::get<2>(GetParam());
  int segment_size = kTestDataSize / batch_size;
  std::string hlo_str = absl::Substitute(
      kHloTpl,
      primitive_util::LowercasePrimitiveTypeName(std::get<0>(GetParam())),
      std::get<1>(GetParam()) ? "LT" : "GT", batch_size, segment_size);
  ASSERT_OK_AND_ASSIGN(bool rewritten, IsRewrittenToUseCubSort(hlo_str));
  EXPECT_TRUE(rewritten);

  EXPECT_TRUE(RunAndCompare(hlo_str, ErrorSpec{0, 0}));
}

INSTANTIATE_TEST_SUITE_P(
    CubSort, CubSortKeysParameterizedTest,
    ::testing::Combine(::testing::Values(F16, F32, F64, S8, S16, S32, S64, U8,
                                         U16, U32, U64),
                       ::testing::Bool(), ::testing::Values(1, 10),
                       GetCubTestDevicelessValues()),
    [](const ::testing::TestParamInfo<CubSortKeysParameterizedTest::ParamType>&
           info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          std::get<1>(info.param) ? "_asc_" : "_desc_", "b",
          std::get<2>(info.param),
          std::get<3>(info.param) ? "_deviceless" : "_device");
    });

// ----- Sort pairs

class CubSortPairsTest : public CubSortTestBase {};

class CubSortPairsParameterizedTest
    : public CubSortPairsTest,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, bool, int, bool>> {
 public:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions options = CubSortPairsTest::GetDebugOptionsForTest();
    if (std::get<4>(GetParam())) {
      options.set_xla_gpu_deviceless_cub_mode(
          DebugOptions::DEVICELESS_CUB_FORCE_ON_NO_FALLBACK);
    }
    return options;
  }
};

TEST_F(CubSortPairsTest, AlwaysUsesCubSort) {
  EXPECT_EQ(SortRewriter::SortMode(), SortRewriter::Mode::kAlways);
}

TEST_P(CubSortPairsParameterizedTest, CompareToReference) {
  if (std::get<4>(GetParam())) {
    if (auto skip_reason = DevicelessSkipReason(std::get<0>(GetParam()),
                                                std::get<1>(GetParam()),
                                                std::get<3>(GetParam()))) {
      GTEST_SKIP() << *skip_reason;
    }
  }
  int batch_size = std::get<3>(GetParam());
  int segment_size = kTestDataSize / batch_size;

  const char* kHloTpl = R"(
HloModule TestSortPairs

compare {
  %lhs = $0[] parameter(0)
  %rhs = $0[] parameter(1)
  // Note that only the keys (first operand of `sort`) are sorted and the values
  // (second operand of `sort`) are ignored. For the case where this sort is
  // part of a TopK decomposition, this works fine, because CUB sort is stable
  // and `values` are actually the unique indices, produced by an iota.
  %v0 = $1[] parameter(2)
  %v1 = $1[] parameter(3)
  ROOT %comp = pred[] compare(%lhs, %rhs), direction=$2
}

ENTRY main {
  %keys = $0[$3,$4] parameter(0)
  %values = $1[$3,$4] parameter(1)
  ROOT %sort = ($0[$3,$4], $1[$3,$4]) sort(%keys, %values),
      dimensions={1}, to_apply=compare
})";
  std::string hlo_str = absl::Substitute(
      kHloTpl,
      primitive_util::LowercasePrimitiveTypeName(std::get<0>(GetParam())),
      primitive_util::LowercasePrimitiveTypeName(std::get<1>(GetParam())),
      std::get<2>(GetParam()) ? "LT" : "GT", batch_size, segment_size);

  ASSERT_OK_AND_ASSIGN(bool rewritten, IsRewrittenToUseCubSort(hlo_str));
  EXPECT_TRUE(rewritten);

  EXPECT_TRUE(RunAndCompare(hlo_str, ErrorSpec{0, 0}));
}

// This test verifies an issue where sort was launched on the wrong stream,
// causing subtle timing bugs: b/347239322.
TEST_P(CubSortPairsParameterizedTest, SortWithSlice) {
  if (std::get<4>(GetParam())) {
    if (auto skip_reason = DevicelessSkipReason(std::get<0>(GetParam()),
                                                std::get<1>(GetParam()),
                                                std::get<3>(GetParam()))) {
      GTEST_SKIP() << *skip_reason;
    }
  }
  constexpr char kHloTpl[] = R"(
compare {
  lhs = $0[] parameter(0)
  rhs = $0[] parameter(1)
  // Note that only the keys (first operand of `sort`) are sorted and the values
  // (second operand of `sort`) are ignored. For the case where this sort is
  // part of a TopK decomposition, this works fine, because CUB sort is stable
  // and `values` are actually the unique indices, produced by an iota.
  v0 = $1[] parameter(2)
  v1 = $1[] parameter(3)
  ROOT comp = pred[] compare(lhs, rhs), direction=$2
}

ENTRY m {
  keys = $0[$3,$4] parameter(0)
  values = $1[$3,$4] parameter(1)
  sort = ($0[$3,$4], $1[$3,$4]) sort(keys, values),
    dimensions={1}, to_apply=compare
  sorted_keys = $0[$3,$4] get-tuple-element(sort), index=0
  sorted_values = $1[$3,$4] get-tuple-element(sort), index=1

  // Avoid matching the topk pattern.
  added_keys = $0[$3,$4] add(sorted_keys, sorted_keys)
  added_values = $1[$3,$4] add(sorted_values, sorted_values)

  sliced_keys = $0[$3,10] slice(added_keys), slice={[0:$3],[0:10]}
  sliced_values = $1[$3,10] slice(added_values), slice={[0:$3],[0:10]}
  ROOT tuple = tuple(sliced_keys, sliced_values)
})";

  int batch_size = std::get<3>(GetParam());
  int segment_size = kTestDataSize / batch_size;
  std::string hlo_str = absl::Substitute(
      kHloTpl,
      primitive_util::LowercasePrimitiveTypeName(std::get<0>(GetParam())),
      primitive_util::LowercasePrimitiveTypeName(std::get<1>(GetParam())),
      std::get<2>(GetParam()) ? "LT" : "GT", batch_size, segment_size);
  ASSERT_OK_AND_ASSIGN(bool rewritten, IsRewrittenToUseCubSort(hlo_str));
  EXPECT_TRUE(rewritten);

  EXPECT_TRUE(RunAndCompare(hlo_str, ErrorSpec{0, 0}));
}

INSTANTIATE_TEST_SUITE_P(
    CubSort, CubSortPairsParameterizedTest,
    ::testing::Combine(::testing::Values(U8, U16, U32, U64, F32),
                       ::testing::Values(F16, F32, F64), ::testing::Bool(),
                       ::testing::Values(1, 10), GetCubTestDevicelessValues()),
    [](const ::testing::TestParamInfo<CubSortPairsParameterizedTest::ParamType>&
           info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          primitive_util::LowercasePrimitiveTypeName(std::get<1>(info.param)),
          std::get<2>(info.param) ? "_asc_" : "_desc_", "b",
          std::get<3>(info.param),
          std::get<4>(info.param) ? "_deviceless" : "_device");
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
