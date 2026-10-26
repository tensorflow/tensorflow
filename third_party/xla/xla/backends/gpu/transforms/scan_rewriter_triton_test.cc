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

#include "xla/backends/gpu/transforms/scan_rewriter_triton.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"  // IWYU pragma: keep
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {
namespace {

class ScanRewriterTritonTest : public HloHardwareIndependentTestBase {
 protected:
  se::DeviceDescription device_info_{TestGpuDeviceInfo::RTXA6000DeviceInfo()};
};

TEST_F(ScanRewriterTritonTest, BasicScanRewrittenToTritonFusion) {
  constexpr absl::string_view kHloText = R"(
HloModule basic_scan

combiner {
  in = f32[] parameter(0)
  carry = f32[] parameter(1)
  add = f32[] add(carry, in)
  ROOT t = (f32[], f32[]) tuple(add, add)
}

ENTRY main {
  input = f32[100]{0} parameter(0)
  init = f32[] constant(0)
  scan = (f32[100]{0}, f32[]) scan(input, init), dimensions={0}, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[100]{0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), "__triton");
}

TEST_F(ScanRewriterTritonTest, ScanWithMultipleGte0UsersRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule multi_gte0_scan

combiner {
  in = f32[] parameter(0)
  carry = f32[] parameter(1)
  add = f32[] add(carry, in)
  ROOT t = (f32[], f32[]) tuple(add, add)
}

ENTRY main {
  input = f32[100]{0} parameter(0)
  init = f32[] constant(0)
  scan = (f32[100]{0}, f32[]) scan(input, init), dimensions={0}, num_carries=1, to_apply=combiner, is_associative=true
  gte0_a = f32[100]{0} get-tuple-element(scan), index=0
  gte0_b = f32[100]{0} get-tuple-element(scan), index=0
  ROOT add = f32[100]{0} add(gte0_a, gte0_b)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(root->operand(0)->opcode(), HloOpcode::kFusion);
  EXPECT_EQ(root->operand(0), root->operand(1));
}

TEST_F(ScanRewriterTritonTest, ScanWithUsedCarryNotRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule scan_used_carry

combiner {
  in = f32[] parameter(0)
  carry = f32[] parameter(1)
  add = f32[] add(carry, in)
  ROOT t = (f32[], f32[]) tuple(add, add)
}

ENTRY main {
  input = f32[100]{0} parameter(0)
  init = f32[] constant(0)
  scan = (f32[100]{0}, f32[]) scan(input, init), dimensions={0}, num_carries=1, to_apply=combiner, is_associative=true
  gte0 = f32[100]{0} get-tuple-element(scan), index=0
  gte1 = f32[] get-tuple-element(scan), index=1
  ROOT t = (f32[100]{0}, f32[]) tuple(gte0, gte1)
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ScanRewriterTritonTest, NonAssociativeScanNotRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule non_assoc_scan

combiner {
  in = f32[] parameter(0)
  carry = f32[] parameter(1)
  add = f32[] add(carry, in)
  ROOT t = (f32[], f32[]) tuple(add, add)
}

ENTRY main {
  input = f32[100]{0} parameter(0)
  init = f32[] constant(0)
  scan = (f32[100]{0}, f32[]) scan(input, init), dimensions={0}, num_carries=1, to_apply=combiner, is_associative=false
  ROOT gte0 = f32[100]{0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ScanRewriterTritonTest, ScanAsRootNotRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule scan_as_root

combiner {
  in = f32[] parameter(0)
  carry = f32[] parameter(1)
  add = f32[] add(carry, in)
  ROOT t = (f32[], f32[]) tuple(add, add)
}

ENTRY main {
  input = f32[100]{0} parameter(0)
  init = f32[] constant(0)
  ROOT scan = (f32[100]{0}, f32[]) scan(input, init), dimensions={0}, num_carries=1, to_apply=combiner, is_associative=true
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ScanRewriterTritonTest, ReverseScanNotRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule reverse_scan

combiner {
  in = f32[] parameter(0)
  carry = f32[] parameter(1)
  add = f32[] add(carry, in)
  ROOT t = (f32[], f32[]) tuple(add, add)
}

ENTRY main {
  input = f32[100]{0} parameter(0)
  init = f32[] constant(0)
  scan = (f32[100]{0}, f32[]) scan(input, init), dimensions={0}, is_reverse=true, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[100]{0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ScanRewriterTritonTest, SmallBatchLargeScanDimNotRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule small_batch_large_scan

combiner {
  in = f32[16] parameter(0)
  carry = f32[16] parameter(1)
  add = f32[16] add(carry, in)
  ROOT t = (f32[16], f32[16]) tuple(add, add)
}

ENTRY main {
  input = f32[16,32768]{1,0} parameter(0)
  init = f32[16]{0} parameter(1)
  scan = (f32[16,32768]{1,0}, f32[16]{0}) scan(input, init), dimensions={1}, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[16,32768]{1,0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ScanRewriterTritonTest, LargeBatchLargeScanDimRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule large_batch_large_scan

combiner {
  in = f32[64] parameter(0)
  carry = f32[64] parameter(1)
  add = f32[64] add(carry, in)
  ROOT t = (f32[64], f32[64]) tuple(add, add)
}

ENTRY main {
  input = f32[64,32768]{1,0} parameter(0)
  init = f32[64]{0} parameter(1)
  scan = (f32[64,32768]{1,0}, f32[64]{0}) scan(input, init), dimensions={1}, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[64,32768]{1,0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kFusion);
  ASSERT_OK_AND_ASSIGN(GpuBackendConfig backend_config,
                       root->backend_config<GpuBackendConfig>());
  EXPECT_EQ(backend_config.fusion_backend_config().kind(), "__triton");
  // The tiling is left to the cost model.
  EXPECT_FALSE(
      backend_config.fusion_backend_config().has_block_level_fusion_config());
}

TEST_F(ScanRewriterTritonTest, SmallScanDimAlwaysRewrittenEvenAtBatch1) {
  constexpr absl::string_view kHloText = R"(
HloModule small_scan_dim_batch_1

combiner {
  in = f32[1] parameter(0)
  carry = f32[1] parameter(1)
  add = f32[1] add(carry, in)
  ROOT t = (f32[1], f32[1]) tuple(add, add)
}

ENTRY main {
  input = f32[1,4096]{1,0} parameter(0)
  init = f32[1]{0} parameter(1)
  scan = (f32[1,4096]{1,0}, f32[1]{0}) scan(input, init), dimensions={1}, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[1,4096]{1,0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(ScanRewriterTritonTest, RewriterBailsOutOnPreAmpereCudaGpu) {
  constexpr absl::string_view kHloText = R"(
HloModule non_ampere_scan

combiner {
  in = f32[16] parameter(0)
  carry = f32[16] parameter(1)
  add = f32[16] add(carry, in)
  ROOT t = (f32[16], f32[16]) tuple(add, add)
}

ENTRY main {
  input = f32[16,4096]{1,0} parameter(0)
  init = f32[16]{0} parameter(1)
  scan = (f32[16,4096]{1,0}, f32[16]{0}) scan(input, init), dimensions={1}, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[16,4096]{1,0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  EXPECT_THAT(
      ScanRewriterTriton(
          TestGpuDeviceInfo::RTXA6000DeviceInfo(
              se::CudaComputeCapability{se::CudaComputeCapability::kVolta, 0}))
          .Run(module.get()),
      absl_testing::StatusIs(
          absl::StatusCode::kFailedPrecondition,
          ::testing::HasSubstr("Triton support is only enabled for Ampere GPUs "
                               "(compute capability 8.0) and up, but got")));
}

TEST_F(ScanRewriterTritonTest, Rank3ScanOverLastDimensionRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule rank3_scan

combiner {
  in = f32[3,4] parameter(0)
  carry = f32[3,4] parameter(1)
  add = f32[3,4] add(carry, in)
  ROOT t = (f32[3,4], f32[3,4]) tuple(add, add)
}

ENTRY main {
  input = f32[3,4,5]{2,1,0} parameter(0)
  init = f32[3,4]{1,0} parameter(1)
  scan = (f32[3,4,5]{2,1,0}, f32[3,4]{1,0}) scan(input, init),
    dimensions={2}, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[3,4,5]{2,1,0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_TRUE(changed);
}

TEST_F(ScanRewriterTritonTest, Rank3ScanOverNonLastDimensionNotRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule rank3_scan_non_last

combiner {
  in = f32[3,5] parameter(0)
  carry = f32[3,5] parameter(1)
  add = f32[3,5] add(carry, in)
  ROOT t = (f32[3,5], f32[3,5]) tuple(add, add)
}

ENTRY main {
  input = f32[3,4,5]{2,1,0} parameter(0)
  init = f32[3,5]{1,0} parameter(1)
  scan = (f32[3,4,5]{2,1,0}, f32[3,5]{1,0}) scan(input, init),
    dimensions={1}, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[3,4,5]{2,1,0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(ScanRewriterTritonTest, NonTrailingScanDimNotRewritten) {
  constexpr absl::string_view kHloText = R"(
HloModule non_trailing_scan

combiner {
  in = f32[16] parameter(0)
  carry = f32[16] parameter(1)
  add = f32[16] add(carry, in)
  ROOT t = (f32[16], f32[16]) tuple(add, add)
}

ENTRY main {
  input = f32[4096,16]{1,0} parameter(0)
  init = f32[16]{0} parameter(1)
  scan = (f32[4096,16]{1,0}, f32[16]{0}) scan(input, init), dimensions={0}, num_carries=1, to_apply=combiner, is_associative=true
  ROOT gte0 = f32[4096,16]{1,0} get-tuple-element(scan), index=0
}
)";

  ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHloText));

  ScanRewriterTriton rewriter(device_info_);
  ASSERT_OK_AND_ASSIGN(bool changed, rewriter.Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
