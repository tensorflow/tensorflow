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
#include "xla/service/gpu/gpu_sort_rewriter.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

// ----- Sort keys

class CubSortKeysTest : public HloTestBase,
                        public ::testing::WithParamInterface<
                            std::tuple<PrimitiveType, bool, int>> {};

TEST_P(CubSortKeysTest, CompareToReference) {
  int batch_size = std::get<2>(GetParam());
  int segment_size = GpuSortRewriter::kSortSizeThreshold / batch_size;

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
          std::tuple<PrimitiveType, PrimitiveType, bool, int>> {};

TEST_P(CubSortPairsTest, CompareToReference) {
  int batch_size = std::get<3>(GetParam());
  int segment_size = GpuSortRewriter::kSortSizeThreshold / batch_size;

  const char* kHloTpl = R"(
HloModule TestSortPairs

compare {
  %lhs = $0[] parameter(0)
  %rhs = $0[] parameter(1)
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
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_str));
  EXPECT_TRUE(RunAndCompare(std::move(hlo_module), ErrorSpec{0, 0}));
}

INSTANTIATE_TEST_SUITE_P(
    CubSort, CubSortPairsTest,
    ::testing::Combine(::testing::Values(U16, U32, U64),
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
