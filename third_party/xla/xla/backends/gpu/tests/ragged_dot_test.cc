/* Copyright 2025 The OpenXLA Authors.

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
#include <utility>

#include <gtest/gtest.h>
#include "xla/error_spec.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/tests/hlo_pjrt_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

using RaggedDotTest = HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>;

TEST_F(RaggedDotTest, NonContracting) {
  const char* hlo_text = R"(
HloModule m

ENTRY main {
  p0 = f32[6,4]{1,0} parameter(0)
  p1 = f32[3,4,5]{2,1,0} parameter(1)
  p2 = s64[3]{0} parameter(2)
  ROOT ragged-dot = f32[6,5]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1}, rhs_contracting_dims={1},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      auto fake_arguments,
      MakeFakeArguments(module.get(), /*pseudo_random=*/true,
                        /*use_large_range=*/false,
                        /*treat_gte_as_data_formatting=*/false,
                        /*max_bits_of_precision=*/10));
  // Set group sizes to reasonable numbers for ragged_dim_size=6.
  fake_arguments[2] = LiteralUtil::CreateR1<int64_t>({1, 2, 3});
  EXPECT_TRUE(RunAndCompare(std::move(module),
                            LiteralUtil::MakePointers(fake_arguments),
                            ErrorSpec{0, 0}));
}

TEST_F(RaggedDotTest, NonContractingWithBatchDims) {
  const char* hlo_text = R"(
  HloModule m

  ENTRY main {
    p0 = f32[3,9,4]{2,1,0} parameter(0)
    p1 = f32[3,2,4,8]{3,2,1,0} parameter(1)
    p2 = s64[3,2]{1,0} parameter(2)
    ROOT ragged-dot = f32[3,9,8]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={2},
                      lhs_batch_dims={0}, rhs_batch_dims={0},
                      lhs_ragged_dims={1}, rhs_group_dims={1}
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      auto fake_arguments,
      MakeFakeArguments(module.get(), /*pseudo_random=*/true,
                        /*use_large_range=*/false,
                        /*treat_gte_as_data_formatting=*/false,
                        /*max_bits_of_precision=*/10));
  // Set group sizes to reasonable numbers for ragged_dim_size=9.
  fake_arguments[2] = LiteralUtil::CreateR2<int64_t>({{4, 5}, {7, 2}, {6, 3}});
  EXPECT_TRUE(RunAndCompare(std::move(module),
                            LiteralUtil::MakePointers(fake_arguments),
                            ErrorSpec{0, 0}));
}

TEST_F(RaggedDotTest, NonContractingWithMultipleContractingDims) {
  const char* hlo_text = R"(
HloModule m

ENTRY main {
  p0 = f32[6,4,3]{2,1,0} parameter(0)
  p1 = f32[2,4,3,5]{3,2,1,0} parameter(1)
  p2 = s64[2]{0} parameter(2)
  ROOT ragged-dot = f32[6,5]{1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={1,2}, rhs_contracting_dims={1,2},
                      lhs_ragged_dims={0}, rhs_group_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      auto fake_arguments,
      MakeFakeArguments(module.get(), /*pseudo_random=*/true,
                        /*use_large_range=*/false,
                        /*treat_gte_as_data_formatting=*/false,
                        /*max_bits_of_precision=*/10));
  // Set group sizes to reasonable numbers for ragged_dim_size=6.
  fake_arguments[2] = LiteralUtil::CreateR1<int64_t>({4, 2});
  EXPECT_TRUE(RunAndCompare(std::move(module),
                            LiteralUtil::MakePointers(fake_arguments),
                            ErrorSpec{0, 0}));
}

TEST_F(RaggedDotTest, NonContractingWithExtraLhsDim) {
  const char* hlo_text = R"(
HloModule m

ENTRY main {
  p0 = f32[2,6,4]{2,1,0} parameter(0)
  p1 = f32[3,4,5]{2,1,0} parameter(1)
  p2 = s64[2,3]{1,0} parameter(2)
  ROOT ragged-dot = f32[2,6,5]{2,1,0} ragged-dot(p0, p1, p2),
                      lhs_contracting_dims={2}, rhs_contracting_dims={1},
                      lhs_ragged_dims={1}, rhs_group_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo_text));
  TF_ASSERT_OK_AND_ASSIGN(
      auto fake_arguments,
      MakeFakeArguments(module.get(), /*pseudo_random=*/true,
                        /*use_large_range=*/false,
                        /*treat_gte_as_data_formatting=*/false,
                        /*max_bits_of_precision=*/10));
  // Set group sizes to reasonable numbers for ragged_dim_size=6.
  fake_arguments[2] = LiteralUtil::CreateR2<int64_t>({{1, 2, 3}, {3, 2, 1}});
  EXPECT_TRUE(RunAndCompare(std::move(module),
                            LiteralUtil::MakePointers(fake_arguments),
                            ErrorSpec{0, 0}));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
