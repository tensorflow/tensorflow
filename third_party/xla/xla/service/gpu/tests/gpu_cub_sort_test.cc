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

#include <memory>
#include <string>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/strings/substitute.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/transforms/sort_rewriter.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {

bool HloWasRewrittenToUseCubSort(const HloModule& module) {
  for (const auto& pass_metadata : module.metadata().proto().pass_metadata()) {
    if (pass_metadata.pass_name() == "sort-rewriter") {
      return pass_metadata.module_changed();
    }
  }
  return false;
}

constexpr int kTestDataSize = 10000;

// ----- Sort keys

class CubSortKeysTest : public HloTestBase,
                        public ::testing::WithParamInterface<
                            std::tuple<PrimitiveType, bool, int>> {
 public:
  void SetUp() override {
    HloTestBase::SetUp();
    SortRewriter::SetSortSizeThresholdForTestingOnly(
        0);  // Always use CUB sort.
  }
};

TEST_F(CubSortKeysTest, AlwaysUsesCubSort) {
  EXPECT_EQ(SortRewriter::SortSizeThreshold(), 0);
}

TEST_P(CubSortKeysTest, CompareToReference) {
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_hlo_module,
                          GetOptimizedModule(hlo_str));
  EXPECT_TRUE(HloWasRewrittenToUseCubSort(*optimized_hlo_module));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_str));
  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{0, 0}));
}

TEST_F(CubSortKeysTest, CompareToReferenceNumpyOrderGt) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_hlo_module,
                          GetOptimizedModule(kHlo));
  EXPECT_TRUE(HloWasRewrittenToUseCubSort(*optimized_hlo_module));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{0, 0}));
}

// Verify that Cub Device Radix sort honors XLA's total order semantics:
// -NaN < -Inf < -Finite < -0 < +0 < +Finite < +Inf < +NaN
// https://openxla.org/xla/operation_semantics#element-wise_comparison_operations
//
// Starting with release 1.12.0, Cub Device Radix sort treats +0.0 and -0.0
// equivalently. See https://github.com/NVIDIA/cub/releases/tag/1.12.0
// This test may break when upgrading to a newer version of Cub.
TEST_F(CubSortKeysTest, CompareToReferenceTotalOrderLt) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_hlo_module,
                          GetOptimizedModule(kHlo));
  EXPECT_TRUE(HloWasRewrittenToUseCubSort(*optimized_hlo_module));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{0, 0}));
}

// This test verifies an issue where sort was launched on the wrong stream,
// causing subtle timing bugs: b/347239322.
TEST_P(CubSortKeysTest, SortWithSlice) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_hlo_module,
                          GetOptimizedModule(hlo_str));
  EXPECT_TRUE(HloWasRewrittenToUseCubSort(*optimized_hlo_module));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_str));
  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{0, 0}));
}

INSTANTIATE_TEST_SUITE_P(
    CubSort, CubSortKeysTest,
    ::testing::Combine(::testing::Values(F16, F32, F64, S8, S16, S32, S64, U8,
                                         U16, U32, U64),
                       ::testing::Bool(), ::testing::Values(1, 10)),
    [](const ::testing::TestParamInfo<CubSortKeysTest::ParamType>& info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          std::get<1>(info.param) ? "_asc_" : "_desc_", "b",
          std::get<2>(info.param));
    });

// ----- Sort pairs

class CubSortPairsTest
    : public HloTestBase,
      public ::testing::WithParamInterface<
          std::tuple<PrimitiveType, PrimitiveType, bool, int>> {
 public:
  void SetUp() override {
    HloTestBase::SetUp();
    SortRewriter::SetSortSizeThresholdForTestingOnly(
        0);  // Always use CUB sort.
  }
};

TEST_F(CubSortPairsTest, AlwaysUsesCubSort) {
  EXPECT_EQ(SortRewriter::SortSizeThreshold(), 0);
}

TEST_P(CubSortPairsTest, CompareToReference) {
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_hlo_module,
                          GetOptimizedModule(hlo_str));
  EXPECT_TRUE(HloWasRewrittenToUseCubSort(*optimized_hlo_module));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_str));
  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{0, 0}));
}

// This test verifies an issue where sort was launched on the wrong stream,
// causing subtle timing bugs: b/347239322.
TEST_P(CubSortPairsTest, SortWithSlice) {
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> optimized_hlo_module,
                          GetOptimizedModule(hlo_str));
  EXPECT_TRUE(HloWasRewrittenToUseCubSort(*optimized_hlo_module));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_str));
  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{0, 0}));
}

INSTANTIATE_TEST_SUITE_P(
    CubSort, CubSortPairsTest,
    ::testing::Combine(::testing::Values(U8, U16, U32, U64),
                       ::testing::Values(F16, F32, F64), ::testing::Bool(),
                       ::testing::Values(1, 10)),
    [](const ::testing::TestParamInfo<CubSortPairsTest::ParamType>& info) {
      return absl::StrCat(
          primitive_util::LowercasePrimitiveTypeName(std::get<0>(info.param)),
          primitive_util::LowercasePrimitiveTypeName(std::get<1>(info.param)),
          std::get<2>(info.param) ? "_asc_" : "_desc_", "b",
          std::get<3>(info.param));
    });

}  // namespace
}  // namespace gpu
}  // namespace xla
