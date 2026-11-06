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

#include "xla/backends/gpu/transforms/horizontal_fusion.h"

#include <memory>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/error_spec.h"
#include "xla/hlo/testlib/filecheck.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/tests/hlo_interpreter_reference_mixin.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/xla.pb.h"

namespace xla::gpu {
namespace {

using ::absl_testing::IsOkAndHolds;

class HorizontalFusionTest
    : public HloInterpreterReferenceMixin<GpuPjRtCodegenTest> {
 protected:
  DebugOptions GetDebugOptionsForTest() const override {
    DebugOptions debug_options = GpuPjRtCodegenTest::GetDebugOptionsForTest();
    debug_options.set_xla_gpu_experimental_enable_horizontal_fusion(true);
    return debug_options;
  }
};

TEST_F(HorizontalFusionTest, LoopPlusLoopPlusLoopDifferentShapes) {
  constexpr absl::string_view kHloText = R"(
HloModule loop_plus_loop_plus_loop, is_scheduled=true

fused_computation.0 {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  mul0 = f32[1024]{0} multiply(p0, p1)
  ROOT out0 = f32[1024]{0} add(mul0, p0)
}

fused_computation.1 {
  p2 = f32[128,3]{1,0} parameter(0)
  neg1 = f32[128,3]{1,0} negate(p2)
  mul1 = f32[128,3]{1,0} multiply(p2, p2)
  ROOT out1 = f32[128,3]{1,0} add(neg1, mul1)
}

fused_computation.2 {
  p0 = f32[1024]{0} parameter(0)
  p3 = f32[512]{0} parameter(1)
  slice0 = f32[512]{0} slice(p0), slice={[0:512]}
  sub2 = f32[512]{0} subtract(slice0, p3)
  ROOT out2 = f32[512]{0} multiply(sub2, p3)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[1024]{0} parameter(1)
  p2 = f32[128,3]{1,0} parameter(2)
  p3 = f32[512]{0} parameter(3)

  f0 = f32[1024]{0} fusion(p0, p1), kind=kLoop, calls=fused_computation.0
  f1 = f32[128,3]{1,0} fusion(p2), kind=kLoop, calls=fused_computation.1
  f2 = f32[512]{0} fusion(p0, p3), kind=kLoop, calls=fused_computation.2

  ROOT result = (f32[1024]{0}, f32[128,3]{1,0}, f32[512]{0})
      tuple(f0, f1, f2)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion horizontal_fusion(device_description());
  ASSERT_THAT(horizontal_fusion.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: %[[F0:.*]] = f32[1024]{0} fusion({{.*}}), kind=kLoop, calls=%fused_computation.0
// CHECK-DAG: %[[F1:.*]] = f32[128,3]{1,0} fusion({{.*}}), kind=kLoop, calls=%fused_computation.1
// CHECK-DAG: %[[F2:.*]] = f32[512]{0} fusion({{.*}}), kind=kLoop, calls=%fused_computation.2
// CHECK: ROOT {{.*}} = (f32[1024]{0}, f32[128,3]{1,0}, f32[512]{0}) tuple(%[[F0]], %[[F1]], %[[F2]])
// CHECK: ENTRY %main
// CHECK: %[[HF:.*]] = (f32[1024]{0}, f32[128,3]{1,0}, f32[512]{0}) fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
)"));
  EXPECT_TRUE(filecheck_matched);

  TF_EXPECT_OK(CompileAndVerifyIr(module->Clone(), R"(
; CHECK-LABEL: define ptx_kernel void @horizontal_fusion
; CHECK-NOT: _nested_0
; CHECK-NOT: _nested_1
; CHECK-NOT: _nested_2
; CHECK: tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
; CHECK: icmp {{.*}}ult i32
; CHECK: ret void
)",
                                  /*match_optimized_ir=*/true,
                                  /*run_optimization_passes=*/false));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), ErrorSpec{1e-5, 1e-5}));
}

TEST_F(HorizontalFusionTest, SharedInputBranchesEndToEndPipeline) {
  constexpr absl::string_view kHloText = R"(
HloModule shared_input_branches

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)

  neg0 = f32[1024]{0} negate(p0)
  branch0 = f32[1024]{0} cosine(neg0)

  slice0 = f32[512]{0} slice(p0), slice={[0:512]}
  diff1 = f32[512]{0} subtract(slice0, p1)
  branch1 = f32[512]{0} multiply(diff1, p1)

  ROOT out = (f32[1024]{0}, f32[512]{0}) tuple(branch0, branch1)
}
)";

  MatchOptimizedHlo(kHloText, R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: f32[1024]{0} fusion({{.*}}), kind=kLoop
// CHECK-DAG: f32[512]{0} fusion({{.*}}), kind=kLoop
// CHECK: ENTRY %main
// CHECK: fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
)");

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(HorizontalFusionTest, SharedIntermediateInputDoesNotAliasInPlace) {
  constexpr absl::string_view kHloText = R"(
HloModule shared_intermediate_input

ENTRY main {
  p0 = f32[16384]{0} parameter(0)
  t0 = f32[16384]{0} tanh(p0)
  x = f32[16384]{0} opt-barrier(t0)
  y = f32[16384]{0} opt-barrier(x)
  out0 = f32[16384]{0} cosine(x)
  out1 = f32[16384]{0} exponential(y)
  ROOT res = (f32[16384]{0}, f32[16384]{0}) tuple(out0, out1)
}
)";

  EXPECT_TRUE(RunAndCompare(kHloText, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(HorizontalFusionTest, LoopPlusRowReductionMixedThreadCountsAndBarriers) {
  constexpr absl::string_view kHloText = R"(
HloModule loop_plus_row_reduction, is_scheduled=true

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_loop {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  mul0 = f32[512]{0} multiply(p0, p1)
  add0 = f32[512]{0} add(mul0, p0)
  ROOT out_loop = f32[512]{0} cosine(add0)
}

fused_reduce {
  p2 = f32[8,512]{1,0} parameter(0)
  sq = f32[8,512]{1,0} multiply(p2, p2)
  zero = f32[] constant(0)
  ROOT out_red = f32[8]{0} reduce(sq, zero), dimensions={1}, to_apply=add_f32
}

ENTRY main {
  p0 = f32[512]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)
  p2 = f32[8,512]{1,0} parameter(2)

  f_loop = f32[512]{0} fusion(p0, p1), kind=kLoop, calls=fused_loop
  f_red = f32[8]{0} fusion(p2), kind=kInput, calls=fused_reduce

  ROOT result = (f32[512]{0}, f32[8]{0}) tuple(f_loop, f_red)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion horizontal_fusion(device_description());
  ASSERT_THAT(horizontal_fusion.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: %[[F0:.*]] = f32[512]{0} fusion({{.*}}), kind=kLoop, calls=%fused_loop
// CHECK-DAG: %[[F1:.*]] = f32[8]{0} fusion({{.*}}), kind=kInput, calls=%fused_reduce
// CHECK: ROOT {{.*}} = (f32[512]{0}, f32[8]{0}) tuple(%[[F0]], %[[F1]])
// CHECK: ENTRY %main
// CHECK: fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
)"));
  EXPECT_TRUE(filecheck_matched);

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), ErrorSpec{1e-4, 1e-4}));
}

TEST_F(HorizontalFusionTest, TransposePlusReductionSharedMemoryAliasing) {
  constexpr absl::string_view kHloText = R"(
HloModule transpose_plus_reduction, is_scheduled=true

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

fused_transpose {
  p0 = f32[32,64]{1,0} parameter(0)
  p1 = f32[32,64]{1,0} parameter(1)
  sum01 = f32[32,64]{1,0} add(p0, p1)
  ROOT out_transpose = f32[64,32]{1,0} transpose(sum01), dimensions={1,0}
}

fused_reduce {
  p2 = f32[8,1024]{1,0} parameter(0)
  p3 = f32[8,1024]{1,0} parameter(1)
  prod = f32[8,1024]{1,0} multiply(p2, p3)
  zero = f32[] constant(0)
  ROOT out_reduce = f32[8]{0} reduce(prod, zero), dimensions={1}, to_apply=add_f32
}

ENTRY main {
  p0 = f32[32,64]{1,0} parameter(0)
  p1 = f32[32,64]{1,0} parameter(1)
  p2 = f32[8,1024]{1,0} parameter(2)
  p3 = f32[8,1024]{1,0} parameter(3)

  f_transpose = f32[64,32]{1,0} fusion(p0, p1), kind=kInput,
      calls=fused_transpose
  f_reduce = f32[8]{0} fusion(p2, p3), kind=kInput, calls=fused_reduce

  ROOT result = (f32[64,32]{1,0}, f32[8]{0}) tuple(f_transpose, f_reduce)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion horizontal_fusion(device_description());
  ASSERT_THAT(horizontal_fusion.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: %[[F0:.*]] = f32[64,32]{1,0} fusion({{.*}}), kind=kInput, calls=%fused_transpose
// CHECK-DAG: %[[F1:.*]] = f32[8]{0} fusion({{.*}}), kind=kInput, calls=%fused_reduce
// CHECK: ROOT {{.*}} = (f32[64,32]{1,0}, f32[8]{0}) tuple(%[[F0]], %[[F1]])
// CHECK: ENTRY %main
// CHECK: fusion({{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
)"));
  EXPECT_TRUE(filecheck_matched);

  TF_EXPECT_OK(CompileAndVerifyIr(module->Clone(), R"(
; CHECK: @global_smem = external {{.*}}addrspace(3) global [0 x i8]
; CHECK-LABEL: define ptx_kernel void @horizontal_fusion
; CHECK: ret void
)",
                                  /*match_optimized_ir=*/true,
                                  /*run_optimization_passes=*/false));

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), ErrorSpec{1e-4, 1e-4}));
}

TEST_F(HorizontalFusionTest, ProducerConsumerDiamondWithHorizontalLayer) {
  constexpr absl::string_view kHloText = R"(
HloModule diamond_horizontal_layer, is_scheduled=true

add_f32 {
  lhs = f32[] parameter(0)
  rhs = f32[] parameter(1)
  ROOT sum = f32[] add(lhs, rhs)
}

producer_reduce {
  p0 = f32[32,128]{1,0} parameter(0)
  p1 = f32[32,128]{1,0} parameter(1)
  prod = f32[32,128]{1,0} multiply(p0, p1)
  zero = f32[] constant(0)
  ROOT mid = f32[32]{0} reduce(prod, zero), dimensions={1}, to_apply=add_f32
}

consumer_0_loop {
  mid = f32[32]{0} parameter(0)
  p2 = f32[32]{0} parameter(1)
  c0_add = f32[32]{0} add(mid, p2)
  ROOT c0 = f32[32]{0} cosine(c0_add)
}

consumer_1_transpose {
  mid = f32[32]{0} parameter(0)
  p3 = f32[32,64]{1,0} parameter(1)
  mid_bcast = f32[32,64]{1,0} broadcast(mid), dimensions={0}
  c1_add = f32[32,64]{1,0} add(p3, mid_bcast)
  ROOT c1 = f32[64,32]{1,0} transpose(c1_add), dimensions={1,0}
}

consumer_2_reduce {
  mid = f32[32]{0} parameter(0)
  p4 = f32[8,128]{1,0} parameter(1)
  mid_slice = f32[8]{0} slice(mid), slice={[0:8]}
  mid_slice_bcast = f32[8,128]{1,0} broadcast(mid_slice), dimensions={0}
  c2_add = f32[8,128]{1,0} add(p4, mid_slice_bcast)
  zero = f32[] constant(0)
  ROOT c2 = f32[8]{0} reduce(c2_add, zero), dimensions={1}, to_apply=add_f32
}

ENTRY main {
  p0 = f32[32,128]{1,0} parameter(0)
  p1 = f32[32,128]{1,0} parameter(1)
  p2 = f32[32]{0} parameter(2)
  p3 = f32[32,64]{1,0} parameter(3)
  p4 = f32[8,128]{1,0} parameter(4)

  mid = f32[32]{0} fusion(p0, p1), kind=kInput, calls=producer_reduce

  c0 = f32[32]{0} fusion(mid, p2), kind=kLoop, calls=consumer_0_loop
  c1 = f32[64,32]{1,0} fusion(mid, p3), kind=kInput,
      calls=consumer_1_transpose
  c2 = f32[8]{0} fusion(mid, p4), kind=kInput, calls=consumer_2_reduce

  ROOT result = (f32[32]{0}, f32[64,32]{1,0}, f32[8]{0}) tuple(c0, c1, c2)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion horizontal_fusion(device_description());
  ASSERT_THAT(horizontal_fusion.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  ASSERT_OK_AND_ASSIGN(bool filecheck_matched,
                       RunFileCheck(module->ToString(), R"(
// CHECK: %horizontal_fusion_computation
// CHECK-DAG: %[[F0:.*]] = f32[32]{0} fusion({{.*}}), kind=kLoop, calls=%consumer_0_loop
// CHECK-DAG: %[[F1:.*]] = f32[64,32]{1,0} fusion({{.*}}), kind=kInput, calls=%consumer_1_transpose
// CHECK-DAG: %[[F2:.*]] = f32[8]{0} fusion({{.*}}), kind=kInput, calls=%consumer_2_reduce
// CHECK: ROOT {{.*}} = (f32[32]{0}, f32[64,32]{1,0}, f32[8]{0}) tuple(%[[F0]], %[[F1]], %[[F2]])
// CHECK: ENTRY %main
// CHECK: %[[MID:.*]] = f32[32]{0} fusion({{.*}}), kind=kInput, calls=%producer_reduce
// CHECK: %[[HF:.*]] = (f32[32]{0}, f32[64,32]{1,0}, f32[8]{0}) fusion(%[[MID]], {{.*}}), kind=kCustom, calls=%horizontal_fusion_computation
)"));
  EXPECT_TRUE(filecheck_matched);

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), ErrorSpec{1e-4, 1e-4}));
}

TEST_F(HorizontalFusionTest, MultiOutputNestedFusionPlusLoop) {
  constexpr absl::string_view kHloText = R"(
HloModule mof_plus_loop, is_scheduled=true

fused_mof {
  p0 = f32[1024]{0} parameter(0)
  n0 = f32[1024]{0} negate(p0)
  c0 = f32[1024]{0} cosine(p0)
  ROOT t = (f32[1024]{0}, f32[1024]{0}) tuple(n0, c0)
}

fused_single {
  p1 = f32[512]{0} parameter(0)
  mul1 = f32[512]{0} multiply(p1, p1)
  ROOT out1 = f32[512]{0} add(mul1, p1)
}

ENTRY main {
  p0 = f32[1024]{0} parameter(0)
  p1 = f32[512]{0} parameter(1)

  f_mof = (f32[1024]{0}, f32[1024]{0}) fusion(p0), kind=kLoop, calls=fused_mof
  gte0 = f32[1024]{0} get-tuple-element(f_mof), index=0
  gte1 = f32[1024]{0} get-tuple-element(f_mof), index=1
  f_single = f32[512]{0} fusion(p1), kind=kLoop, calls=fused_single

  ROOT result = (f32[1024]{0}, f32[1024]{0}, f32[512]{0})
      tuple(gte0, gte1, f_single)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(kHloText));
  HorizontalFusion horizontal_fusion(device_description());
  ASSERT_THAT(horizontal_fusion.Run(module.get()), IsOkAndHolds(true));
  EXPECT_OK(module->schedule().Verify());

  EXPECT_TRUE(
      RunAndCompareNoHloPasses(std::move(module), ErrorSpec{1e-5, 1e-5}));
}
}  // namespace
}  // namespace xla::gpu
