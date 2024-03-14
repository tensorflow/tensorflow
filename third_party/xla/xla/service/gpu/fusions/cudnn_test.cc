/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"

namespace xla {
namespace gpu {
namespace {

class CuDnnFusionTest : public GpuCodegenTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = GpuCodegenTest::GetDebugOptionsForTest();
    // Let this group of tests just use first available plan skipping
    // autotuning.
    debug_options.set_xla_gpu_autotune_level(0);
    return debug_options;
  }
  bool IsAtLeastHopperWithCuDnn9() {
    se::StreamExecutor* executor = backend().default_stream_executor();
    return executor->GetDeviceDescription()
               .cuda_compute_capability()
               .IsAtLeastHopper() &&
           GetDnnVersionInfo(executor).major_version() >= 9;
  }

 protected:
  void SetUp() override {
    if (!IsAtLeastHopperWithCuDnn9()) {
      GTEST_SKIP()
          << "cuDNN GEMM fusion is not enabled before Hopper / cuDNN 9.";
    }
  }
};

using CuDnnFusionExecutionTest = CuDnnFusionTest;

TEST_F(CuDnnFusionExecutionTest, DotF32ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT r = f32[32,64] dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

ENTRY e {
  p0 = f32[32,96] parameter(0)
  p1 = f32[96,64] parameter(1)
  ROOT _ = f32[32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotBF16WithCopyExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[96,512,64]{1,2,0} parameter(0)
  cp = bf16[96,512,64]{2,1,0} copy(p0)
  p1 = bf16[96,64,512]{2,1,0} parameter(1)
  ROOT d = bf16[96,512,512]{2,1,0} dot(cp, p1),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[96,512,64]{1,2,0} parameter(0)
  p1 = bf16[96,64,512]{2,1,0} parameter(1)
  ROOT r = bf16[96,512,512]{2,1,0} fusion(p0, p1), kind=kCustom,
    calls=fusion1,
    backend_config={"fusion_backend_config": {kind :"__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-2, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotBF16BF16F32ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[16,32,128] parameter(0)
  p1 = bf16[16,128,64] parameter(1)
  ROOT r = f32[16,32,64] dot(p0, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[16,32,128] parameter(0)
  p1 = bf16[16,128,64] parameter(1)
  ROOT _ = f32[16,32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-6, /*arel=*/1e-6}));
}

TEST_F(CuDnnFusionExecutionTest, DotF32WithOutputSubtractionExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f32[9,32,96] parameter(0)
  p1 = f32[9,96,64] parameter(1)
  d = f32[9,32,64] dot(p0, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
  p2 = f32[9,32,64] parameter(2)
  ROOT s = f32[9,32,64] subtract(p2, d)
}

ENTRY e {
  p0 = f32[9,32,96] parameter(0)
  p1 = f32[9,96,64] parameter(1)
  p2 = f32[9,32,64] parameter(2)
  ROOT _ = f32[9,32,64] fusion(p0, p1, p2), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotWithNonDefaultLayoutsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[32,32]{0,1} parameter(0)
  p1 = bf16[32,32]{1,0} parameter(1)
  ROOT r = bf16[32,32]{0,1} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[32,32]{0,1} parameter(0)
  p1 = bf16[32,32]{1,0} parameter(1)
  ROOT _ = bf16[32,32]{0,1} fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-4, /*arel=*/1e-4}));
}

TEST_F(CuDnnFusionExecutionTest, RHSFusionExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = bf16[5,32,96] parameter(0)
  p1 = s8[5,96,16] parameter(1)
  p1c = bf16[5,96,16] convert(p1)
  ROOT r = bf16[5,32,16] dot(p0, p1c),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = bf16[5,32,96] parameter(0)
  p1 = s8[5,96,16] parameter(1)
  ROOT _ = bf16[5,32,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, SkipNonDefaultPrecision) {
  EXPECT_FALSE(Run(R"(
t {
  p0 = f32[27,23] parameter(0)
  p0c = s8[27,23] convert(p0)
  p0cc = f32[27,23] convert(p0c)
  p1 = f32[23,21] parameter(1)
  ROOT r = f32[27,21] dot(p0cc, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    operand_precision={HIGH, HIGH}
}

ENTRY e {
  p0 = f32[27,23] parameter(0)
  p1 = f32[23,21] parameter(1)
  ROOT r = f32[27,21] fusion(p0, p1), kind=kCustom, calls=t,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})"));
}

TEST_F(CuDnnFusionExecutionTest,
       DotF16NegateNonDefaultDimensionsExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = f16[16,32,96] parameter(0)
  p0n = f16[16,32,96] negate(p0)
  p1 = f16[16,64,96] parameter(1)
  ROOT r = f16[16,32,64] dot(p0n, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
}

ENTRY e {
  p0 = f16[16,32,96] parameter(0)
  p1 = f16[16,64,96] parameter(1)
  ROOT _ = f16[16,32,64] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

TEST_F(CuDnnFusionExecutionTest, DotS8BF16ExecutesCorrectly) {
  EXPECT_TRUE(RunAndCompare(R"(
fusion1 {
  p0 = s8[5,32,96] parameter(0)
  p0c = bf16[5,32,96] convert(p0)
  p1 = bf16[5,96,16] parameter(1)
  ROOT r = bf16[5,32,16] dot(p0c, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={1}
}

ENTRY e {
  p0 = s8[5,32,96] parameter(0)
  p1 = bf16[5,96,16] parameter(1)
  ROOT _ = bf16[5,32,16] fusion(p0, p1), kind=kCustom, calls=fusion1,
    backend_config={"fusion_backend_config": {kind: "__cudnn$fusion"}}
})",
                            ErrorSpec{/*aabs=*/1e-5, /*arel=*/1e-5}));
}

TEST_F(CuDnnFusionExecutionTest, CommandBuffersAreSupported) {
  const std::string kHloText = R"(
HloModule m

%fusion0 {
  %p0 = f32[64,64]{1,0} parameter(0)
  %p1 = f32[64,64]{1,0} parameter(1)
  ROOT %d = f32[64,64]{1,0} dot(%p0, %p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

%fusion_a {
  %p0.2 = f32[64,64]{1,0} parameter(0)
  %p1.2 = f32[64,64]{1,0} parameter(1)
  ROOT %a = f32[64,64]{1,0} add(%p0.2, %p1.2)
}

%fusion1 {
  %p0.1 = f32[64,64]{1,0} parameter(0)
  %p1.1 = f32[64,64]{1,0} parameter(1)
  ROOT %d.1 = f32[64,64]{1,0} dot(%p0.1, %p1.1), lhs_contracting_dims={0}, rhs_contracting_dims={1}
}

%command_buffer {
  %p0.4 = f32[64,64]{1,0} parameter(0)
  %p1.4 = f32[64,64]{1,0} parameter(1)
  %d0.1 = f32[64,64]{1,0} fusion(%p0.4, %p1.4), kind=kCustom, calls=%fusion0,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
  %a.2 = f32[64,64]{1,0} fusion(%d0.1, %d0.1), kind=kLoop, calls=%fusion_a
  ROOT %d1.1 = f32[64,64]{1,0} fusion(%a.2, %p1.4), kind=kCustom, calls=%fusion1,
    backend_config={"fusion_backend_config":{"kind":"__cudnn$fusion","cudnn_fusion_config":{"plan_id":"0"}}}
}

ENTRY %e {
  %p0.3 = f32[64,64]{1,0} parameter(0)
  %p1.3 = f32[64,64]{1,0} parameter(1)
  ROOT %call = f32[64,64]{1,0} call(%p0.3, %p1.3), to_apply=%command_buffer
})";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{/*aabs=*/1e-3, /*arel=*/1e-3}));
}

class CuDnnFusionRewriteTest : public CuDnnFusionTest {
 public:
  DebugOptions GetDebugOptionsForTest() override {
    DebugOptions debug_options = CuDnnFusionTest::GetDebugOptionsForTest();
    // Reset autotuning level to default.
    debug_options.set_xla_gpu_autotune_level(
        GetDebugOptionsFromFlags().xla_gpu_autotune_level());
    debug_options.set_xla_gpu_cudnn_gemm_fusion(true);
    return debug_options;
  }
};

TEST_F(CuDnnFusionRewriteTest,
       DoNotExecuteGemmFusionWithCuDnnWhenNotSupported) {
  // Dimension size 61 does not satisfy the requirement on alignment
  // (multiple of 2).
  MatchOptimizedHlo(R"(
ENTRY e {
  p0 = f16[20,40,61] parameter(0)
  p2 = f16[20,40,61] parameter(2)
  p0n = f16[20,40,61] negate(p2)
  p1 = f16[20,80,61] parameter(1)
  ROOT r = f16[20,40,80] dot(p0n, p1),
    lhs_batch_dims={0}, rhs_batch_dims={0},
    lhs_contracting_dims={2}, rhs_contracting_dims={2}
})",
                    R"(
; CHECK: ENTRY
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: parameter
; CHECK-NEXT: ROOT
; CHECK-SAME: fusion
; CHECK-NOT: cudnn
)");
}

TEST_F(CuDnnFusionRewriteTest, AutotuningPicksCuDnnForS8BF16OnHopper) {
  // The test case relies on measurements by the autotuner and current
  // performance comparison of the backends. May need to be updated if
  // the situation changes.
  MatchOptimizedHlo(R"(
e {
  p0 = bf16[720,720,720] parameter(0)
  p1 = s8[720,720,720] parameter(1)
  c = bf16[720,720,720] convert(p1)
  ROOT d = bf16[720,720,720] dot(p0, c),
    lhs_batch_dims={0}, lhs_contracting_dims={2},
    rhs_batch_dims={0}, rhs_contracting_dims={1}
})",
                    R"(
; CHECK: __cudnn$fusion
)");
}

}  // namespace
}  // namespace gpu
}  // namespace xla
