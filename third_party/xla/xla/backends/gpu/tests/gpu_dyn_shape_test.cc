/* Copyright 2020 The OpenXLA Authors.

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
#include <utility>

#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "xla/backends/gpu/tests/gpu_pjrt_codegen_test.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

class GpuDynamicShapeTest : public GpuPjRtCodegenTest {};

TEST_F(GpuDynamicShapeTest, DynamicShapeR2) {
  HloComputation::Builder builder(TestName());

  xla::Shape dyn_input_shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 4});
  dyn_input_shape.set_dynamic_dimension(0, true);
  HloInstruction* param_x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dyn_input_shape, "x"));

  builder.AddInstruction(HloInstruction::CreateUnary(
      dyn_input_shape, HloOpcode::kNegate, param_x));
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  EXPECT_OK(CompileAndVerifyIr(std::move(hlo_module),
                               R"(
; CHECK-LABEL: define {{.*}}_kernel void @x_padded(
; CHECK: %[[SRC_INT:.*]] = ptrtoint ptr %{{.*}} to i64
; CHECK: %[[META0:.*]] = add i64 %[[SRC_INT]], 32
; CHECK: %[[PTR0:.*]] = inttoptr i64 %[[META0]] to ptr
; CHECK: %[[DIM0:.*]] = load i32, ptr %[[PTR0]], align 4
; CHECK: %[[META1:.*]] = add i64 %[[SRC_INT]], 36
; CHECK: %[[PTR1:.*]] = inttoptr i64 %[[META1]] to ptr
; CHECK: %[[DIM1:.*]] = load i32, ptr %[[PTR1]], align 4
; CHECK: store i32 %[[DIM0]], ptr
; CHECK: store i32 %[[DIM1]], ptr
; CHECK: %[[TOTAL:.*]] = mul i32 %[[DIM0]], %[[DIM1]]
; CHECK: %[[LIN:.*]] = add i32
; CHECK: icmp ult i32 %[[LIN]], %[[TOTAL]]
; CHECK: load float, ptr
; CHECK: store float
      )",
                               /*match_optimized_ir=*/false));
}

}  // namespace
}  // namespace xla::gpu
