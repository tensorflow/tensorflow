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

#include "xla/service/gpu/gpu_executable_buffer_allocator.h"

#include <cstdint>

#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace vmm_internal {
namespace {

// ---- VmmRemapSkipEnabled ------------------------------------------------
// The flag value comes from
// DebugOptions::xla_gpu_experimental_command_buffer_vmm_skip_remap (default
// true); these helpers only apply the ROCm-only platform gating.

TEST(VmmRemapSkipEnabledTest, NonRocmAlwaysDisabled) {
  EXPECT_FALSE(VmmRemapSkipEnabled("CUDA", /*flag_enabled=*/true));
  EXPECT_FALSE(VmmRemapSkipEnabled("CUDA", /*flag_enabled=*/false));
  EXPECT_FALSE(VmmRemapSkipEnabled("Host", /*flag_enabled=*/true));
}

TEST(VmmRemapSkipEnabledTest, RocmFollowsFlag) {
  EXPECT_TRUE(VmmRemapSkipEnabled("ROCM", /*flag_enabled=*/true));
  EXPECT_FALSE(VmmRemapSkipEnabled("ROCM", /*flag_enabled=*/false));
}

// ---- VmmCopyThresholdBytes ----------------------------------------------
// The flag value comes from
// DebugOptions::xla_gpu_experimental_command_buffer_vmm_copy_threshold_bytes
// (default 0); these helpers apply the ROCm-only gating and clamp negatives to
// 0.

TEST(VmmCopyThresholdBytesTest, NonRocmAlwaysZero) {
  EXPECT_EQ(VmmCopyThresholdBytes("CUDA", 65536), 0u);
  EXPECT_EQ(VmmCopyThresholdBytes("CUDA", 0), 0u);
}

TEST(VmmCopyThresholdBytesTest, RocmZeroOrNegativeIsZero) {
  EXPECT_EQ(VmmCopyThresholdBytes("ROCM", 0), 0u);
  EXPECT_EQ(VmmCopyThresholdBytes("ROCM", -1), 0u);
  EXPECT_EQ(VmmCopyThresholdBytes("ROCM", -65536), 0u);
}

TEST(VmmCopyThresholdBytesTest, RocmPositiveFlagPassesThrough) {
  EXPECT_EQ(VmmCopyThresholdBytes("ROCM", 4096), 4096u);
  EXPECT_EQ(VmmCopyThresholdBytes("ROCM", 65536), 65536u);
}

}  // namespace
}  // namespace vmm_internal
}  // namespace gpu
}  // namespace xla
