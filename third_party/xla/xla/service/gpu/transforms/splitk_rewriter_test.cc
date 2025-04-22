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

#include "xla/service/gpu/transforms/splitk_rewriter.h"

#include <memory>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace gpu {
namespace {

class SplitkRewriterTest : public HloTestBase {
 public:
  SplitkRewriterTest()
      : rewriter_(se::DeviceDescription(
            ParseTextProto<stream_executor::GpuDeviceInfoProto>(
                "core_count: 132")
                .value())) {}

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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_split_k_rewrite(true);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_split_k_rewrite(true);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: f32[16,102528]{1,0} pad({{.*}}), padding=0_0x127_0
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

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_split_k_rewrite(true);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(RunFileCheck(module->ToString(), R"(
CHECK: f32{{.*}} dot(
CHECK: f32{{.*}} reduce(
CHECK: bf16[16,128]{1,0} convert(
)")
                  .value_or(false));
}

TEST_F(SplitkRewriterTest, NoSplitKIfEnoughWork) {
  // Huge K dimension to trigger 128 which is the largest possible splitK
  // (hoping to make the test less fragile as heuristic changes).
  const char* hlo_string = R"(
    HloModule module
  
    ENTRY test {
      lhs = f32[1024,10240]{1,0} parameter(0)
      rhs = f32[10240,2048]{1,0} parameter(1)
      ROOT dot = f32[1024,2048]{1,0} dot(lhs, rhs),
                                  lhs_contracting_dims={1}, rhs_contracting_dims={0}
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_gpu_experimental_enable_split_k_rewrite(true);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          rewriter_.HloModulePass::Run(module.get()));
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
