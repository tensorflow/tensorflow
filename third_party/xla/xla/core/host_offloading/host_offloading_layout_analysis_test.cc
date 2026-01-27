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

#include "xla/core/host_offloading/host_offloading_layout_analysis.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class NeedsLayoutConversionTest : public HloHardwareIndependentTestBase {};

TEST_F(NeedsLayoutConversionTest, ElementWise) {
  constexpr absl::string_view hlo_string = R"(
HloModule m, entry_computation_layout={(f32[8,128]{1,0})->f32[8,128]{1,0}}

ENTRY main {
  arg0 = f32[8,128]{1,0} parameter(0)
  ROOT multiply = f32[8,128]{1,0} multiply(arg0, arg0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  EXPECT_THAT(HostOffloadingLayoutAnalysis::NeedsLayoutConversion(module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(NeedsLayoutConversionTest, NonElementWise) {
  constexpr absl::string_view hlo_string = R"(
HloModule dot, entry_computation_layout={(f32[2,10]{1,0}, f32[10,2]{1,0})->f32[2]{0}}

ENTRY dot {
  a = f32[2,10]{1,0} parameter(0)
  b = f32[10,2]{1,0} parameter(1)
  ROOT dot = f32[2]{0} dot(a, b), lhs_batch_dims={0}, lhs_contracting_dims={1}, rhs_batch_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  // Dot is not elementwise.
  EXPECT_THAT(HostOffloadingLayoutAnalysis::NeedsLayoutConversion(module.get()),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(NeedsLayoutConversionTest, Tuple) {
  constexpr absl::string_view hlo_string = R"(
HloModule m, entry_computation_layout={(f32[8,128]{1,0})->(f32[8,128]{1,0},f32[8,128]{1,0})}

ENTRY main {
  arg0 = f32[8,128]{1,0} parameter(0)
  ROOT tuple = (f32[8,128]{1,0}, f32[8,128]{1,0}) tuple(arg0, arg0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  EXPECT_THAT(HostOffloadingLayoutAnalysis::NeedsLayoutConversion(module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(NeedsLayoutConversionTest, ScalarBroadcast) {
  constexpr absl::string_view hlo_string = R"(
HloModule m, entry_computation_layout={()->f32[8,128]{1,0}}

ENTRY main {
  constant = f32[] constant(0.1001)
  ROOT broadcast = f32[8,128]{1,0} broadcast(constant), dimensions={}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  EXPECT_THAT(HostOffloadingLayoutAnalysis::NeedsLayoutConversion(module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(NeedsLayoutConversionTest, NonScalarBroadcast) {
  constexpr absl::string_view hlo_string = R"(
HloModule m, entry_computation_layout={()->f32[8,128]{0,1}}

ENTRY main {
  constant = f32[8] constant({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0})
  ROOT broadcast = f32[8,128]{0,1} broadcast(constant), dimensions={}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  // Non-scalar broadcast in column major -- must use layout conversion.
  EXPECT_THAT(HostOffloadingLayoutAnalysis::NeedsLayoutConversion(module.get()),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(NeedsLayoutConversionTest, Call) {
  constexpr absl::string_view hlo_string = R"(
HloModule m, entry_computation_layout={(f32[8,512,128]{2,1,0})->f32[8,512,128]{2,1,0}}

helper {
  ROOT result = f32[8,512,128]{2,1,0} parameter(0)
}

ENTRY main {
  arg0 = f32[8,512,128]{2,1,0} parameter(0)
  call = f32[8,512,128]{2,1,0} call(arg0), to_apply=helper
  ROOT convert = f32[8,512,128]{2,1,0} convert(call)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  EXPECT_THAT(HostOffloadingLayoutAnalysis::NeedsLayoutConversion(module.get()),
              absl_testing::IsOkAndHolds(false));
}

TEST_F(NeedsLayoutConversionTest, MixedElementTypes) {
  constexpr absl::string_view hlo_string = R"(
HloModule m, entry_computation_layout={(bf16[8,512,128]{2,1,0})->f32[8,512,128]{2,1,0}}

ENTRY main {
  arg0 = bf16[8,512,128]{2,1,0} parameter(0)
  ROOT convert = f32[8,512,128]{2,1,0} convert(arg0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  EXPECT_THAT(HostOffloadingLayoutAnalysis::NeedsLayoutConversion(module.get()),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(NeedsLayoutConversionTest, PaddedTensor) {
  constexpr absl::string_view hlo_string = R"(
HloModule m, entry_computation_layout={(s32[8,2]{0,1:T(2,128)})->s32[8,2]{0,1:T(2,128)}}

ENTRY main {
  arg0 = s32[8,2]{0,1} parameter(0)
  ROOT multiply = s32[8,2]{0,1} multiply(arg0, arg0)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnUnverifiedModule(hlo_string));
  // The tiling used here creates padding, which requires layout conversion.
  EXPECT_THAT(HostOffloadingLayoutAnalysis::NeedsLayoutConversion(module.get()),
              absl_testing::IsOkAndHolds(true));
}

}  // namespace
}  // namespace xla
