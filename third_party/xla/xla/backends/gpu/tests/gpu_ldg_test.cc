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

// Tests that we emit ld.global.nc (the PTX instruction corresponding to CUDA's
// __ldg builtin) for reads of buffers that don't change during a kernel's
// execution.

#include <memory>
#include <utility>

#include <gtest/gtest.h>
#include "absl/strings/substitute.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/hlo_module_config.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/test.h"
#include "xla/xla.pb.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

using GpuLdgTest = GpuPjRtCodegenTest;

// Parameters are never overwritten, so parameter reads should get ld.global.nc
// reads.
//
// On the ROCM platform the "ptx" string is not populated for the compiled
// executable, and hence the call to CompileAdnVerifyPtx does not do the
// "VerifyPtx" part, it merely compiles the executable
//
TEST_F(GpuLdgTest, LdgForParamRead) {
  HloComputation::Builder builder(TestName());

  auto shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param, param));
  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  CompileAndOptionallyVerifyPtx(std::move(hlo_module), R"(
    CHECK-NOT: ld.global.b32
    CHECK: ld.global.nc.b32
  )");
}

// Check that reading a buffer produced by a non-parameter HLO also results in
// ld.global.nc, if that buffer isn't modified within the instruction that reads
// it.
//
// On the ROCM platform the "ptx" string is not populated for the compiled
// executable, and hence the call to CompileAdnVerifyPtx does not do the
// "VerifyPtx" part, it merely compiles the executable
//
TEST_F(GpuLdgTest, LdgForNonParamRead) {
  HloComputation::Builder builder(TestName());

  auto shape = ShapeUtil::MakeShape(F32, {2, 2});
  HloInstruction* param =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "x"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param, param));
  HloInstruction* square = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, add, add));
  builder.AddInstruction(HloInstruction::CreateTuple({add, square}));
  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  CompileAndOptionallyVerifyPtx(std::move(hlo_module), R"(
    CHECK: {
    CHECK-NOT: ld.global.b32
    CHECK: ld.global.nc.b32
    CHECK: }
  )");
}

// Check that reading a buffer that's modified in-place does not produce
// ld.global.nc.
//
// We do this by creating a reduce that feeds into an add.  We don't currently
// fuse add into reduce, and the add is elementwise, so it reuses its input
// buffer as its output.
//
// It seems like a fair bet that we won't start fusing add into the output of
// reduce in the foreseeable future.  But if that turns out to be wrong, I give
// you, future reader, permission to delete this test.
//
// On the ROCM platform the "ptx" string is not populated for the compiled
// executable, and hence the call to CompileAdnVerifyPtx does not do the
// "VerifyPtx" part, it merely compiles the executable
//
TEST_F(GpuLdgTest, NoLdgWhenSharingBuffer) {
  auto hlo_module = CreateNewVerifiedModule();
  HloComputation::Builder builder(TestName());

  HloComputation* reduce_computation;
  {
    auto embedded_builder = HloComputation::Builder("add");
    auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "lhs"));
    auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {}), "rhs"));
    embedded_builder.AddInstruction(
        HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
    reduce_computation =
        hlo_module->AddEmbeddedComputation(embedded_builder.Build());
  }

  auto param_shape = ShapeUtil::MakeShape(F32, {32, 32});
  auto reduce_shape = ShapeUtil::MakeShape(F32, {32});
  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "x"));
  HloInstruction* param2 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, reduce_shape, "y"));
  HloInstruction* reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape,
      builder.AddInstruction(HloInstruction::CreateBinary(
          param_shape, HloOpcode::kAdd, param, param)),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      {0}, reduce_computation));
  builder.AddInstruction(HloInstruction::CreateBinary(
      reduce_shape, HloOpcode::kAdd, reduce, param2));

  std::unique_ptr<HloComputation> computation = builder.Build();
  hlo_module->AddEntryComputation(std::move(computation));

  CompileAndOptionallyVerifyPtx(std::move(hlo_module), R"(
    CHECK-LABEL: .entry wrapped_add
    CHECK: {
    CHECK-NOT: ld.global.nc.b32
    CHECK: ld.global.b32
    CHECK: }
  )");
}

class NonInvariantLdgTest : public GpuLdgTest {
 protected:
  std::unique_ptr<VerifiedHloModule> CreateTestModule(bool add_non_invariant) {
    return ParseAndReturnVerifiedModule(absl::Substitute(R"(
add {
  a = f16[3] parameter(0)
  b = f16[3] parameter(1)
  c = f16[3] add(a, b)
}

e {
  a = f16[3] parameter(0)
  b = f16[3] parameter(1)
  c = f16[3] fusion(a, b), kind=kLoop, calls=add, frontend_attributes={$0}
})",
                                                         add_non_invariant
                                                             ? R"(
xla.no_invariant_operands="0")"
                                                             : ""))
        .value();
  }
};

TEST_F(NonInvariantLdgTest, DoNonCoherentLoadsByDefault) {
  CompileAndOptionallyVerifyPtx(CreateTestModule(/*add_non_invariant=*/false),
                                R"(
     CHECK-NOT: ld.global.b16
     CHECK: ld.global.nc.b16
     CHECK: ld.global.nc.b16
   )");
}

TEST_F(NonInvariantLdgTest, DoSelectiveCoherentLoadWithNoInvariantAnnotation) {
  CompileAndOptionallyVerifyPtx(CreateTestModule(/*add_non_invariant=*/true),
                                R"(
     CHECK: ld.global.b16
     CHECK: ld.global.nc.b16
   )");
}

class PdlLdgTest : public GpuLdgTest {
 protected:
  void SetUp() override {
    GpuLdgTest::SetUp();
    if (const auto& cc = device_description().cuda_compute_capability();
        !cc.IsAtLeastHopper()) {
      GTEST_SKIP() << "Test requires Hopper or higher";
    }
  }
  absl::StatusOr<std::unique_ptr<VerifiedHloModule>> CreateTestModule(
      bool enable_pdl_launch) {
    DebugOptions debug_options = GetDebugOptionsForTest();
    debug_options.set_xla_gpu_enable_pdl(true);
    debug_options.set_xla_gpu_enable_pdl_launch(enable_pdl_launch);
    HloModuleConfig config;
    config.set_debug_options(debug_options);
    return ParseAndReturnVerifiedModule(R"(
HloModule m, is_scheduled=true

triton_gemm {
  lhs = f16[15,19] parameter(0)
  rhs = f16[19,17] parameter(1)
  dot = f16[15,17] dot(lhs, rhs),
    lhs_contracting_dims={1}, rhs_contracting_dims={0},
    backend_config={sizes:[64]}
}

add {
  a = f16[15, 17] parameter(0)
  b = f16[15, 17] parameter(1)
  c = f16[15, 17] add(a, b)
}

e {
  lhs = f16[15,19] parameter(0)
  rhs = f16[19,17] parameter(1)
  triton_gemm = f16[15,17] fusion(lhs, rhs), kind=kCustom,
    calls=triton_gemm, backend_config={
      fusion_backend_config:{
        kind:"__triton_nested_gemm_fusion",
        block_level_fusion_config:{
          output_tiles:[{sizes:[64,32]}],
          num_stages:2,
          num_warps:8,
          num_ctas:1
        }
      }
    }
  x = f16[15, 17] parameter(2)
  a = f16[15, 17] fusion(triton_gemm, x), kind=kLoop, calls=add
  t = tuple(triton_gemm, a)
})",
                                        config);
  }
};

TEST_F(PdlLdgTest, DoNonCoherentLoadWhenPrecedentKernelHasNoPdlLaunch) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       CreateTestModule(/*enable_pdl_launch=*/false));
  CompileAndOptionallyVerifyPtx(std::move(module),
                                R"(
    CHECK-LABEL: .entry triton_gemm
    CHECK-NOT: griddepcontrol.launch_dependents
    CHECK-LABEL: .entry loop_add_fusion
    CHECK-NOT: ld.global.b16
    CHECK: ld.global.nc.b16
    CHECK: ld.global.nc.b16
  )");
}

TEST_F(PdlLdgTest, DoCoherentLoadWhenPrecedentKernelHasPdlLaunch) {
  ASSERT_OK_AND_ASSIGN(auto module,
                       CreateTestModule(/*enable_pdl_launch=*/true));
  CompileAndOptionallyVerifyPtx(std::move(module),
                                R"(
    CHECK-LABEL: .entry triton_gemm
    CHECK: griddepcontrol.launch_dependents
    CHECK-LABEL: .entry loop_add_fusion
    CHECK: ld.global.b16
    CHECK: ld.global.nc.b16
  )");
}

}  // namespace
}  // namespace xla::gpu
