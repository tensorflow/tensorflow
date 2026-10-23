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

#include "xla/backends/gpu/transforms/splitk_rewriter.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tests/test_utils.h"

namespace xla {
namespace gpu {
namespace {

se::DeviceDescription GetDeviceDescription() {
  return se::DeviceDescription::FromProto(
             ParseTextProto<stream_executor::GpuDeviceInfoProto>(
                 "core_count: 132")
                 .value())
      .value();
}

class SplitkRewriterTest : public HloHardwareIndependentTestBase {
 public:
  SplitkRewriterTest() : rewriter_(GetDeviceDescription()) {}

 protected:
  SplitkRewriter rewriter_;
};

TEST_F(SplitkRewriterTest, SmallNonContractingDimensionCauseSplitK) {
  const char* hlo_string = R"(
HloModule module

ENTRY test {
  lhs = f32[16,10240]{1,0} parameter(0)
  rhs = f32[10240,128]{1,0} parameter(1)
  ROOT dot = f32[16,128]{1,0} dot(lhs, rhs),
                              lhs_contracting_dims={1}, rhs_contracting_dims={0}
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: dot({{.*}}), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
CHECK: ROOT {{.*}} = f32[16,128]{1,0} reduce
  )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, PaddingIsInserted) {
  // Huge K dimension to trigger 128 which is the largest possible splitK
  // (hoping to make the test less fragile as heuristic changes).
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    lhs = f32[16,102401]{1,0} parameter(0)
    rhs = f32[102401,128]{1,0} parameter(1)
    ROOT dot = f32[16,128]{1,0} dot(lhs, rhs),
                                lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: f32[16,102912]{1,0} pad(%lhs, %constant), padding=0_0x0_511
    )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, AccumulatorTypeIsDifferentFromOutputType) {
  // Huge K dimension to trigger 128 which is the largest possible splitK
  // (hoping to make the test less fragile as heuristic changes).
  const char* hlo_string = R"(
  HloModule module

  ENTRY test {
    lhs = bf16[16,102400]{1,0} parameter(0)
    rhs = bf16[102400,128]{1,0} parameter(1)
    ROOT dot = bf16[16,128]{1,0} dot(lhs, rhs),
                                lhs_contracting_dims={1}, rhs_contracting_dims={0}
  })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: f32{{.*}} dot(
CHECK: f32{{.*}} reduce(
CHECK: bf16[16,128]{1,0} convert(
)")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, SplitKDotKeepsPhysicalLayoutOfOriginalDot) {
  // The dot result has the non-default layout {0,1} (dimension 0 is minor).
  // The split-K dot must keep this physical layout with the split-K dimension
  // as the most major one, so that the reduction does not transpose.
  const char* hlo_string = R"(
HloModule module

ENTRY test {
  lhs = bf16[1536,10240]{1,0} parameter(0)
  rhs = bf16[256,10240]{1,0} parameter(1)
  ROOT dot = bf16[1536,256]{0,1} dot(lhs, rhs),
                                lhs_contracting_dims={1}, rhs_contracting_dims={1}
})";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: %[[DOT:.*]] = f32[{{[0-9]+}},1536,256]{1,2,0} dot(
CHECK: f32[1536,256]{0,1} reduce(%[[DOT]], {{.*}}), dimensions={0}
CHECK: ROOT {{.*}} = bf16[1536,256]{0,1} convert(
  )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, NoSplitKIfEnoughWork) {
  // Small K is not profitable to split.
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[1024,512]{1,0} parameter(0)
      rhs = f32[512,2048]{1,0} parameter(1)
      ROOT dot = f32[1024,2048]{1,0} dot(lhs, rhs),
                             lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SplitkRewriterTest, DoNotSplitKS32) {
  // We would split K otherwise, but for s32 operands we don't, because neither
  // cuBLAS nor Triton support it.
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = s32[16,10240]{1,0} parameter(0)
      rhs = s32[10240,128]{1,0} parameter(1)
      ROOT dot = s32[16,128]{1,0} dot(lhs, rhs),
                    lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(SplitkRewriterTest, B200FailingModuleTest) {
  se::DeviceDescription b200_desc = TestGpuDeviceInfo::B200SXMDeviceInfo();
  SplitkRewriter splitk_rewriter(b200_desc);

  const char* hlo_string = R"(
HloModule jit_fwd

ENTRY %main.2 (broadcast: f32[2,128,128], b.1: f32[128,2,128]) -> f32[2,128,128] {
  %broadcast = f32[2,128,128]{2,1,0} parameter(0)
  %b.1 = f32[128,2,128]{2,1,0} parameter(1)
  ROOT %dot.1 = f32[2,128,128]{2,1,0} dot(%broadcast, %b.1), lhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_batch_dims={1}, rhs_contracting_dims={0}
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  CHECK_OK(splitk_rewriter.HloModulePass::Run(module.get()));
}

TEST_F(SplitkRewriterTest, ForceSplitK) {
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = f32[1024,512]{1,0} parameter(0)
      rhs = f32[512,2048]{1,0} parameter(1)
      ROOT dot = f32[1024,2048]{1,0} dot(lhs, rhs),
                             lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));

  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_force_split_k(2);

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: dot({{.*}}), lhs_batch_dims={1}, lhs_contracting_dims={2}, rhs_batch_dims={0}, rhs_contracting_dims={1}
CHECK: ROOT {{.*}} = f32[1024,2048]{1,0} reduce
  )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest,
       ConvertedSubByteOperandIsNotPaddedForSubByteAlignment) {
  // K/4 = 2128 is 16B-aligned for the dot's bf16 operand, though not for the
  // pre-convert s4 source.
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = bf16[64,8512]{1,0} parameter(0)
      rhs = s4[8512,8512]{1,0:E(4)} parameter(1)
      rhs_converted = bf16[8512,8512]{1,0} convert(rhs)
      ROOT dot = bf16[64,8512]{1,0} dot(lhs, rhs_converted),
                             lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_force_split_k(4);

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK-NOT: pad(
CHECK: %[[CONVERT:.*]] = bf16[8512,8512]{1,0} convert(%rhs)
CHECK-NOT: pad(
CHECK: bf16[8512,4,2128]{2,1,0} reshape(%[[CONVERT]])
CHECK-NOT: pad(
  )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, ConvertedSubByteOperandIsPaddedForOperandAlignment) {
  // K/4 is not 16B-aligned for bf16, so K is padded for the dot's bf16 operand
  // alignment (to a multiple of 4 * 8 elements), not the pre-convert s4 size.
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = bf16[64,8500]{1,0} parameter(0)
      rhs = s4[8512,8500]{1,0:E(4)} parameter(1)
      rhs_converted = bf16[8512,8500]{1,0} convert(rhs)
      ROOT dot = bf16[64,8512]{1,0} dot(lhs, rhs_converted),
                             lhs_contracting_dims={1}, rhs_contracting_dims={1}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_force_split_k(4);

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK-DAG: bf16[64,8512]{1,0} pad(%lhs, {{.*}}), padding=0_0x0_12
CHECK-DAG: bf16[8512,8512]{1,0} pad(%rhs_converted, {{.*}}), padding=0_0x0_12
  )")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, LayoutChangeOfConvertedSubByteOperandIsNotPadded) {
  // Same as ConvertedSubByteOperandIsNotPaddedForSubByteAlignment, but with
  // the operand relabeled by a layout-changing bitcast.
  const char* hlo_string = R"(
    HloModule module

    ENTRY test {
      lhs = bf16[64,8512]{1,0} parameter(0)
      rhs = s4[8512,8512]{1,0:E(4)} parameter(1)
      rhs_converted = bf16[8512,8512]{1,0} convert(rhs)
      rhs_bitcast = bf16[8512,8512]{0,1} bitcast(rhs_converted)
      ROOT dot = bf16[64,8512]{1,0} dot(lhs, rhs_bitcast),
                             lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_force_split_k(4);

  ASSERT_OK_AND_ASSIGN(bool changed,
                       rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK-NOT: pad(
CHECK: bf16[4,2128,8512]{1,0,2} reshape(%rhs_bitcast)
CHECK-NOT: pad(
  )")
                  .value_or(false));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
