/* Copyright 2026 The OpenXLA Authors.

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

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/transforms/sort_rewriter.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/pattern_matcher.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/semantic_version.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace gpu {
namespace {
using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;

namespace m = ::xla::match;

class SortRewriterDevicelessTest : public HloHardwareIndependentTestBase {
 protected:
  void SetUp() override {
    HloHardwareIndependentTestBase::SetUp();
    SortRewriter::SetSortModeForTestingOnly(SortRewriter::Mode::kAlways);
  }

  absl::StatusOr<bool> RunPass(HloModule* module,
                               bool is_early_exit_with_layouts = false) {
    stream_executor::DeviceDescription device_desc;
    device_desc.set_name("NVIDIA H100 80GB HBM3");
    device_desc.set_cub_version(stream_executor::SemanticVersion{3, 1, 2});
    return SortRewriter(device_desc, /*is_deviceless=*/true,
                        is_early_exit_with_layouts)
        .Run(module);
  }

  void AssertSortWasRewritten(const HloModule* module) {
    EXPECT_THAT(
        module->entry_computation()->root_instruction(),
        GmockMatch(m::GetTupleElement(
            m::CustomCall({kCubDeviceRadixSortUnassignedScratchSizeTarget},
                          m::Parameter()),
            0)));
  }
};

// F32[1000] sort, keys-only. Handled by lookup table.
constexpr absl::string_view kCompatibleSortHlo = R"hlo(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[1000] parameter(0)
  ROOT %sort = f32[1000] sort(%input), dimensions={0}, to_apply=%compare
})hlo";

// F32[100'000'000'000'000] sort. Too large, should miss in lookup table.
constexpr absl::string_view kIncompatibleSortHlo = R"hlo(
HloModule TestModule

%compare {
  %lhs = f32[] parameter(0)
  %rhs = f32[] parameter(1)
  ROOT %lt = pred[] compare(%lhs, %rhs), direction=LT
}

ENTRY %main {
  %input = f32[100000000000000] parameter(0)
  ROOT %sort = f32[100000000000000] sort(%input), dimensions={0}, to_apply=%compare
})hlo";

TEST_F(SortRewriterDevicelessTest, DisabledModeNoRewrite) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kCompatibleSortHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deviceless_cub_mode(DebugOptions::DEVICELESS_CUB_DISABLED);

  ASSERT_THAT(RunPass(module.get()), IsOkAndHolds(false));
}

TEST_F(SortRewriterDevicelessTest, WithFallbackHitRewrites) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kCompatibleSortHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deviceless_cub_mode(
          DebugOptions::DEVICELESS_CUB_WITH_FALLBACK);

  ASSERT_THAT(RunPass(module.get()), IsOkAndHolds(true));
  AssertSortWasRewritten(module.get());
}

TEST_F(SortRewriterDevicelessTest, WithFallbackMissFallsBack) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kIncompatibleSortHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deviceless_cub_mode(
          DebugOptions::DEVICELESS_CUB_WITH_FALLBACK);

  ASSERT_OK_AND_ASSIGN(bool changed, RunPass(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SortRewriterDevicelessTest, NoFallbackHitRewrites) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kCompatibleSortHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deviceless_cub_mode(
          DebugOptions::DEVICELESS_CUB_NO_FALLBACK);

  ASSERT_THAT(RunPass(module.get()), IsOkAndHolds(true));
  AssertSortWasRewritten(module.get());
}

TEST_F(SortRewriterDevicelessTest, NoFallbackMissReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kIncompatibleSortHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deviceless_cub_mode(
          DebugOptions::DEVICELESS_CUB_NO_FALLBACK);

  ASSERT_THAT(RunPass(module.get()).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(SortRewriterDevicelessTest, EarlyExitWithLayoutsForcesRewrite) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kIncompatibleSortHlo));

  ASSERT_THAT(RunPass(module.get(), /*is_early_exit_with_layouts=*/true),
              IsOkAndHolds(true));
  AssertSortWasRewritten(module.get());
}

TEST_F(SortRewriterDevicelessTest, ForceOnNoFallbackHitRewrites) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kCompatibleSortHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deviceless_cub_mode(
          DebugOptions::DEVICELESS_CUB_FORCE_ON_NO_FALLBACK);

  ASSERT_THAT(RunPass(module.get()), IsOkAndHolds(true));
  AssertSortWasRewritten(module.get());
}

TEST_F(SortRewriterDevicelessTest, ForceOnNoFallbackMissReturnsError) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       ParseAndReturnVerifiedModule(kIncompatibleSortHlo));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_deviceless_cub_mode(
          DebugOptions::DEVICELESS_CUB_FORCE_ON_NO_FALLBACK);

  ASSERT_THAT(RunPass(module.get()).status(),
              StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
