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

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/hlo_pjrt_gpu_test_base.h"
#include "xla/error_spec.h"
#include "xla/stream_executor/device_description.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"

namespace xla::gpu {
namespace {

// Regression tests for kernel cache behavior.
// These tests verify that different emitters (Loop, Triton, etc.) properly
// namespace their cached kernels to avoid cross-emitter cache collisions.
class KernelCacheRegressionTest
    : public HloPjRtInterpreterReferenceMixin<HloPjRtGpuTestBase> {};

TEST_F(KernelCacheRegressionTest,
       LoopAndTritonFusionsWithIdenticalComputation) {
  stream_executor::GpuComputeCapability cc =
      device_description().gpu_compute_capability();
  if (cc.IsCuda() && !cc.cuda_compute_capability()->IsAtLeastAmpere()) {
    GTEST_SKIP() << "Triton requires Ampere or above";
  }
  constexpr absl::string_view kHloText = R"(
HloModule test_module

add_f32 {
  p0 = f32[] parameter(0)
  p1 = f32[] parameter(1)
  ROOT add = f32[] add(p0, p1)
}

fused_computation_triton {
  p0 = f32[32,4]{1,0} parameter(0)
  p1 = f32[32,4]{1,0} parameter(1)
  mul = f32[32,4]{1,0} multiply(p0, p1)
  zero = f32[] constant(0)
  reduce = f32[32]{0} reduce(mul, zero), dimensions={1}, to_apply=add_f32
  ROOT sqrt = f32[32]{0} sqrt(reduce)
}

fused_computation_loop {
  p0 = f32[32,4]{1,0} parameter(0)
  p1 = f32[32,4]{1,0} parameter(1)
  mul = f32[32,4]{1,0} multiply(p0, p1)
  zero = f32[] constant(0)
  reduce = f32[32]{0} reduce(mul, zero), dimensions={1}, to_apply=add_f32
  ROOT sqrt = f32[32]{0} sqrt(reduce)
}

ENTRY main {
  input0 = f32[32,4]{1,0} parameter(0)
  input1 = f32[32,4]{1,0} parameter(1)

  // Triton fusion
  triton_result = f32[32]{0} fusion(input0, input1), kind=kCustom,
    calls=fused_computation_triton,
    backend_config={"fusion_backend_config":{
      "kind":"__triton",
      "block_level_fusion_config":{
        "output_tiles":[{"sizes":["32"]}],
        "num_warps":"1",
        "num_ctas":"1",
        "num_stages":"1",
        "is_tma_allowed":false}}}

  // Loop fusion with identical computation structure
  loop_result = f32[32]{0} fusion(input0, input1), kind=kLoop,
    calls=fused_computation_loop

  // Output both results - they should be identical
  ROOT tuple = (f32[32]{0}, f32[32]{0}) tuple(loop_result, triton_result)
})";

  EXPECT_TRUE(RunAndCompareNoHloPasses(kHloText, ErrorSpec{/*aabs=*/1e-6,
                                                           /*arel=*/1e-6}));
}

}  // namespace
}  // namespace xla::gpu
