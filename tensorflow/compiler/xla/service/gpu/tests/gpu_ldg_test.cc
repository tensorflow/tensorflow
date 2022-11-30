/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/tests/gpu_codegen_test.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace gpu {

class GpuLdgTest : public GpuCodegenTest {};

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
    CHECK-NOT: ld.global.f32
    CHECK: ld.global.nc.f32
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
    CHECK-NOT: ld.global.f32
    CHECK: ld.global.nc.f32
    CHECK: }
  )");
}

// Check that reading a buffer that's modified in-place does not produce
// ld.global.nc.
//
// We do this by creating a reduce that feeds into a sin.  We don't currently
// fuse sin into reduce, and the sin is elementwise, so it reuses its input
// buffer as its output.
//
// It seems like a fair bet that we won't start fusing sin into the output of
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
  HloInstruction* reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      reduce_shape,
      builder.AddInstruction(HloInstruction::CreateBinary(
          param_shape, HloOpcode::kAdd, param, param)),
      builder.AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0))),
      {0}, reduce_computation));
  builder.AddInstruction(
      HloInstruction::CreateUnary(reduce_shape, HloOpcode::kSin, reduce));

  std::unique_ptr<HloComputation> computation = builder.Build();
  hlo_module->AddEntryComputation(std::move(computation));

  CompileAndOptionallyVerifyPtx(std::move(hlo_module), R"(
    CHECK-LABEL: .entry sin
    CHECK: {
    CHECK-NOT: ld.global.nc.f32
    CHECK: ld.global.f32
    CHECK: }
  )");
}

}  // namespace gpu
}  // namespace xla
