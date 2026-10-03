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

#include "xla/backends/gpu/transforms/gpu_copy_async_wrapper.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"

namespace xla::gpu {
namespace {

using GpuCopyAsyncWrapperTest = HloHardwareIndependentTestBase;

void SetAsyncCopyMinBytes(HloModule* module, int64_t min_bytes) {
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_async_copy_min_bytes(min_bytes);
}

TEST_F(GpuCopyAsyncWrapperTest, WrapsLargeD2DCopyWhenEnabled) {
  // A single D2D copy of 1024 floats (4096 bytes), above the 1024-byte
  // threshold.
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT copy = f32[1024] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  SetAsyncCopyMinBytes(module.get(), /*min_bytes=*/1024);

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: ENTRY %main
    ; CHECK:   [[P0:%[^ ]+]] = f32[1024]{0} parameter(0)
    ; CHECK:   [[START:%[^ ]+]] = (f32[1024]{0}, f32[1024]{0}, u32[]) copy-start([[P0]])
    ; CHECK:   ROOT {{.*}} = f32[1024]{0} copy-done([[START]])
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCopyAsyncWrapperTest, DoesNotWrapByDefault) {
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT copy = f32[1024] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_EQ(module->config().debug_options().xla_gpu_async_copy_min_bytes(),
            -1);

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: ENTRY %main
    ; CHECK-NOT: copy-start
    ; CHECK:   ROOT {{.*}} = f32[1024]{0} copy(
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCopyAsyncWrapperTest, DoesNotWrapWhenThresholdIsUnset) {
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT copy = f32[1024] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  module->mutable_config()
      .mutable_debug_options()
      .clear_xla_gpu_async_copy_min_bytes();
  EXPECT_FALSE(
      module->config().debug_options().has_xla_gpu_async_copy_min_bytes());

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));
}

TEST_F(GpuCopyAsyncWrapperTest, WrapsD2DCopyWithZeroThreshold) {
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[4] parameter(0)
      ROOT copy = f32[4] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  SetAsyncCopyMinBytes(module.get(), /*min_bytes=*/0);

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCopyAsyncWrapperTest, DoesNotWrapCopyBelowSizeThreshold) {
  // 4 floats = 16 bytes — below the 1024-byte threshold.
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[4] parameter(0)
      ROOT copy = f32[4] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  SetAsyncCopyMinBytes(module.get(), /*min_bytes=*/1024);

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: ENTRY %main
    ; CHECK-NOT: copy-start
    ; CHECK:   ROOT {{.*}} = f32[4]{0} copy(
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCopyAsyncWrapperTest, DoesNotWrapCopyBelow64KiBThreshold) {
  // 1024 floats = 4096 bytes — below a 64 KiB threshold.
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT copy = f32[1024] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  SetAsyncCopyMinBytes(module.get(), /*min_bytes=*/64 * 1024);

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));
}

TEST_F(GpuCopyAsyncWrapperTest, WrapsCopyAbove64KiBThreshold) {
  // 32768 floats = 128 KiB — above a 64 KiB threshold.
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[32768] parameter(0)
      ROOT copy = f32[32768] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  SetAsyncCopyMinBytes(module.get(), /*min_bytes=*/64 * 1024);

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCopyAsyncWrapperTest, DoesNotWrapLayoutChangingCopy) {
  // The copy changes the layout from {0,1} to {1,0}: it is a transpose in
  // disguise, not a memcpy, so it must stay synchronous.
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[128,128]{0,1} parameter(0)
      ROOT copy = f32[128,128]{1,0} copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  SetAsyncCopyMinBytes(module.get(), /*min_bytes=*/1024);

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));

  EXPECT_THAT(RunFileCheck(module->ToString({}), R"(
    ; CHECK: ENTRY %main
    ; CHECK-NOT: copy-start
    ; CHECK:   ROOT {{.*}} = f32[128,128]{1,0} copy(
  )"),
              absl_testing::IsOkAndHolds(true));
}

TEST_F(GpuCopyAsyncWrapperTest, PreservesControlDependencies) {
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[1024] parameter(0)
      p1 = f32[1024] parameter(1)
      add = f32[1024] add(p0, p1)
      copy = f32[1024] copy(p0), control-predecessors={add}
      ROOT tuple = (f32[1024], f32[1024]) tuple(add, copy)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  SetAsyncCopyMinBytes(module.get(), /*min_bytes=*/1024);

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));

  const HloInstruction* copy_start = nullptr;
  for (const HloInstruction* instr :
       module->entry_computation()->instructions()) {
    if (instr->opcode() == HloOpcode::kCopyStart) {
      copy_start = instr;
    }
  }
  ASSERT_NE(copy_start, nullptr);
  ASSERT_EQ(copy_start->control_predecessors().size(), 1);
  EXPECT_EQ(copy_start->control_predecessors()[0]->name(), "add");
}

TEST_F(GpuCopyAsyncWrapperTest, IsIdempotent) {
  constexpr char kHlo[] = R"(
    ENTRY main {
      p0 = f32[1024] parameter(0)
      ROOT copy = f32[1024] copy(p0)
    })";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  SetAsyncCopyMinBytes(module.get(), /*min_bytes=*/1024);

  GpuCopyAsyncWrapper wrapper;
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(true));
  // Second run: the kCopy is gone, only copy-start/copy-done remain.
  EXPECT_THAT(wrapper.Run(module.get()), absl_testing::IsOkAndHolds(false));
}

}  // namespace
}  // namespace xla::gpu
