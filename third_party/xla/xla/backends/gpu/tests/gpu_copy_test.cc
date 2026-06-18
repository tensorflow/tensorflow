/* Copyright 2018 The OpenXLA Authors.

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
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/tests/hlo_pjrt_interpreter_reference_mixin.h"
#include "xla/xla.pb.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla::gpu {
namespace {

class GpuCopyTest
    : public HloPjRtInterpreterReferenceMixin<GpuPjRtCodegenTest> {};

// The GPU backend should not emit a copy kernel for the kCopy instruction in
// this test. Instead, it should generate a CopyThunk which invokes cuMemcpy at
// runtime.
TEST_F(GpuCopyTest, UseMemcpy) {
  HloComputation::Builder builder(TestName());

  Literal literal = LiteralUtil::CreateR2<float>({{1.0, 2.0}, {3.0, 4.0}});
  HloInstruction* constant = builder.AddInstruction(
      HloInstruction::CreateConstant(std::move(literal)));
  builder.AddInstruction(HloInstruction::CreateUnary(
      constant->shape(), HloOpcode::kCopy, constant));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  // There should not be any kernel prefixed "copy".
  ASSERT_OK(CompileAndVerifyIr(std::move(hlo_module),
                               "; CHECK-NOT: define void @_copy",
                               /*match_optimized_ir=*/false));
}

TEST_F(GpuCopyTest, CopyTranspose) {
  const char* hlo_text = R"(
    HloModule Test

    fused_computation {
      param_0 = f32[100,200,300]{2,1,0} parameter(0)
      ROOT b.1 = f32[100,200,300]{2,0,1} copy(f32[100,200,300]{2,1,0} param_0)
    }

    ENTRY main {
      a = f32[100, 200, 300]{2,1,0} parameter(0)
      ROOT wrapped_b = f32[100,200,300]{2,0,1} fusion(f32[100,200,300]{2,1,0} %a), kind=kLoop, calls=fused_computation
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> optimized_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_TRUE(RunAndCompare(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

TEST_F(GpuCopyTest, UseMemcpyForTrivialStaticSliceFusion) {
  constexpr absl::string_view hlo_text = R"(
    HloModule Test

    wrapped_slice_computation {
      param_0 = f32[8,16]{1,0} parameter(0)
      ROOT slice = f32[3,16]{1,0} slice(param_0),
          slice={[2:5], [0:16]}
    }

    ENTRY main {
      param_0 = f32[8,16]{1,0} parameter(0)
      ROOT wrapped_slice = f32[3,16]{1,0} fusion(param_0),
          kind=kLoop, calls=wrapped_slice_computation
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> hlo_module,
                          ParseAndReturnVerifiedModule(hlo_text));

  ASSERT_OK(CompileAndVerifyIr(std::move(hlo_module),
                               "; CHECK-NOT: void @wrapped_slice",
                               /*match_optimized_ir=*/false,
                               /*run_optimization_passes=*/false));
  EXPECT_TRUE(RunAndCompareNoHloPasses(hlo_text, ErrorSpec{1e-5, 1e-5}));
}

constexpr char kSliceMemcpyModuleUnfused[] = R"(
    body {
      p0 = (s32[], s32[4,8,1000000], s32[1,1,1000000]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      input = s32[4,8,1000000] get-tuple-element(p0), index=1

      ivar_copy = s32[] copy(ivar)
      c1 = s32[] constant(1)
      slice = s32[1,1,1000000] dynamic-slice(input, ivar_copy, c1, c1),
          dynamic_slice_sizes={1,1,1000000}

      next_ivar = s32[] add(ivar_copy, c1)
      ROOT result = (s32[], s32[4,8,1000000], s32[1,1,1000000])
          tuple(next_ivar, input, slice)
    }

    compare {
      p0 = s32[] parameter(0)
      c6 = s32[] constant(6)
      ROOT cmp = pred[] compare(p0, c6), direction=LT
    }

    condition {
      p0 = (s32[], s32[4,8,1000000], s32[1,1,1000000]) parameter(0)
      ivar = s32[] get-tuple-element(p0), index=0
      c6 = s32[] constant(6)
      ROOT cmp = pred[] compare(ivar, c6), direction=LT
    }

    ENTRY main {
      p0 = s32[4,8,1000000] parameter(0)
      p1 = s32[1,1,1000000] parameter(1)
      c0 = s32[] constant(0)
      tuple = (s32[], s32[4,8,1000000], s32[1,1,1000000]) tuple(c0, p0, p1)
      ROOT while = (s32[], s32[4,8,1000000], s32[1,1,1000000]) while(tuple),
          condition=condition, body=body
    })";

TEST_F(GpuCopyTest, UseDynamicSliceFusionIntegrationTest) {
  auto compute_capability = device_description().gpu_compute_capability();
  if (auto cc = compute_capability.cuda_compute_capability();
      !cc || !cc->IsAtLeastAmpere()) {
    GTEST_SKIP() << "Test requires at least Ampere.";
  }

  // This is an integration test to verify that the pipeline for rewriting
  // dynamic-slices that depend on while loop iteration variables works as a
  // whole.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> hlo_module,
      ParseAndReturnVerifiedModule(kSliceMemcpyModuleUnfused));

  // Check that there are exactly two fusions:
  // 1. A `compare` fusion for the loop condition.
  // 2. An `add` fusion for the next ivar.
  // If the dynamic-slice fusion rewrite does not trigger, there will be a
  // third fusion for the dynamic-slice.
  ASSERT_OK(CompileAndVerifyIr(std::move(hlo_module), R"(
                       CHECK-NOT: define {{.*}}@

                       CHECK: define {{.*}}@
                       CHECK: getelementptr
                       CHECK-NEXT: load
                       CHECK-NEXT: icmp
                       CHECK-NEXT: zext
                       CHECK-NEXT: getelementptr
                       CHECK-NEXT: store
                       CHECK-NEXT: ret

                       CHECK-NOT: define {{.*}}@
                       CHECK: define {{.*}}@
                       CHECK: getelementptr
                       CHECK-NEXT: load
                       CHECK-NEXT: add
                       CHECK-NEXT: store
                       CHECK-NEXT: ret

                       CHECK-NOT: define {{.*}}@)",
                               /*match_optimized_ir=*/false,
                               /*run_optimization_passes=*/true));
}

TEST_F(GpuCopyTest, UseDynamicSliceFusionIntegrationTestControl) {
  // Control for UseDynamicSliceFusionIntegrationTest. Verify that without
  // fusion-dynamic-memcpy-rewriter, we have a third fusion.
  HloModuleConfig config;
  DebugOptions options = GpuPjRtCodegenTest::GetDebugOptionsForTest();
  options.add_xla_disable_hlo_passes("fusion-dynamic-memcpy-rewriter");
  config.set_debug_options(options);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<VerifiedHloModule> hlo_module,
      ParseAndReturnVerifiedModule(kSliceMemcpyModuleUnfused, config));
  ASSERT_OK(CompileAndVerifyIr(std::move(hlo_module), R"(
                       CHECK-COUNT-3: define {{.*}}@
                       CHECK-NOT: define {{.*}}@)",
                               /*match_optimized_ir=*/false,
                               /*run_optimization_passes=*/true));
}

}  // namespace
}  // namespace xla::gpu
