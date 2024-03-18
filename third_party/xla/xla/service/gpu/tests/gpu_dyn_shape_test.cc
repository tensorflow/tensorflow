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

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {
class GpuDynamicShapeTest : public GpuCodegenTest {};

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

  CompileAndVerifyIr(std::move(hlo_module),
                     R"(
; CHECK-DAG: is_thread_0-true
; CHECK-DAG: x.padded{{.*}}.in_dyn_bounds-true
; CHECK-DAG: x.padded{{.*}}.in_bounds-true
; CHECK: %[[dyn_dim_size:.*]] = load i32, ptr
; CHECK: %[[dyn_element_total:.*]] = mul i32 1, %[[dyn_dim_size:.*]]
; CHECK: %[[linear_index:.*]] = add nuw nsw i32
; CHECK: %[[linear_index_in_range:.*]] = icmp ult i32 %[[linear_index:.*]],
; CHECK: store i32 %[[dyn_dim_size:.*]], ptr
      )",
                     /*match_optimized_ir=*/false);
}

}  // namespace gpu
}  // namespace xla
