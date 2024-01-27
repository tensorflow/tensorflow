/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/custom_kernel_fusion_rewriter.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/kernels/custom_kernel_fusion_pattern.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/test.h"

namespace xla::gpu {

//===----------------------------------------------------------------------===//
// Simple pattern matchers for testing custom kernel_fusion rewriter.
//===----------------------------------------------------------------------===//

struct SimpleGemmPattern : public CustomKernelFusionPattern {
  explicit SimpleGemmPattern(int64_t workspace = 0) : workspace(workspace) {}

  std::optional<Match> TryMatch(const se::DeviceDescription& device,
                                HloInstruction* instr) const override {
    if (auto* dot = DynCast<HloDotInstruction>(instr)) {
      CustomFusionConfig config;
      config.set_name("simple_gemm");
      return Match{config, {instr}, workspace};
    }
    return std::nullopt;
  }

  int64_t workspace;
};

//===----------------------------------------------------------------------===//

class CustomKernelFusionRewriterTest : public HloTestBase {};

TEST_F(CustomKernelFusionRewriterTest, SimpleGemm) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main (p0: f16[15,19], p1: f16[19,17]) -> f16[15,17] {
      %p0 = f16[15,19]{1,0} parameter(0)
      %p1 = f16[19,17]{1,0} parameter(1)
      ROOT %r = f16[15,17]{1,0} dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* expected = R"(
    ; CHECK: %simple_gemm {{.*}} {
    ; CHECK:   [[P0:%[^ ]+]] = f16[15,19]{1,0} parameter(0)
    ; CHECK:   [[P1:%[^ ]+]] = f16[19,17]{1,0} parameter(1)
    ; CHECK:   ROOT [[DOT:%[^ ]+]] = f16[15,17]{1,0} dot([[P0]], [[P1]]),
    ; CHECK:     lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ; CHECK: }

    ; CHECK: ENTRY %main {{.*}} {
    ; CHECK:   ROOT [[FUSION:%[^ ]+]] = f16[15,17]{1,0} fusion
    ; CHECK:     kind=kCustom, calls=%simple_gemm,
    ; CHECK:     backend_config={
    ; CHECK:       "kind":"__custom_fusion",
    ; CHECK:       "custom_fusion_config":{"name":"simple_gemm"}
    ; CHECK:     }
    ; CHECK: }
  )";

  CustomKernelFusionPatternRegistry patterns;
  patterns.Emplace<SimpleGemmPattern>();

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CustomKernelFusionRewriter pass(&device, &patterns);
  RunAndFilecheckHloRewrite(hlo, std::move(pass), expected);
}

TEST_F(CustomKernelFusionRewriterTest, SimpleGemmWithWorkspace) {
  const char* hlo = R"(
    HloModule test

    ENTRY %main (p0: f16[15,19], p1: f16[19,17]) -> f16[15,17] {
      %p0 = f16[15,19]{1,0} parameter(0)
      %p1 = f16[19,17]{1,0} parameter(1)
      ROOT %r = f16[15,17]{1,0} dot(%p0, %p1),
        lhs_contracting_dims={1}, rhs_contracting_dims={0}
    }
  )";

  const char* expected = R"(
    ; CHECK: %simple_gemm {{.*}} {
    ; CHECK:   [[P0:%[^ ]+]] = f16[15,19]{1,0} parameter(0)
    ; CHECK:   [[P1:%[^ ]+]] = f16[19,17]{1,0} parameter(1)
    ; CHECK:   [[DOT:%[^ ]+]] = f16[15,17]{1,0} dot([[P0]], [[P1]]),
    ; CHECK:     lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ; CHECK:   [[WORKSPACE:%[^ ]+]] = u8[1024]{0} custom-call(),
    ; CHECK:     custom_call_target="__custom_kernel_fusion$workspace"
    ; CHECK:   ROOT [[TUPLE:%[^ ]+]] = (f16[15,17]{1,0}, u8[1024]{0})
    ; CHECK:     tuple([[DOT]], [[WORKSPACE]])
    ; CHECK: }

    ; CHECK: ENTRY %main {{.*}} {
    ; CHECK:   [[FUSION:%[^ ]+]] = (f16[15,17]{1,0}, u8[1024]{0}) fusion
    ; CHECK:     kind=kCustom, calls=%simple_gemm,
    ; CHECK:     backend_config={
    ; CHECK:       "kind":"__custom_fusion",
    ; CHECK:       "custom_fusion_config":{"name":"simple_gemm"}
    ; CHECK:     }
    ; CHECK:   ROOT {{.*}} get-tuple-element([[FUSION]]), index=0
    ; CHECK: }
  )";

  CustomKernelFusionPatternRegistry patterns;
  patterns.Emplace<SimpleGemmPattern>(1024);

  auto device = TestGpuDeviceInfo::RTXA6000DeviceInfo();
  CustomKernelFusionRewriter pass(&device, &patterns);
  RunAndFilecheckHloRewrite(hlo, std::move(pass), expected);
}

}  // namespace xla::gpu
