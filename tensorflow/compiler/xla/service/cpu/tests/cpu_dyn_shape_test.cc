/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/cpu/cpu_compiler.h"
#include "tensorflow/compiler/xla/service/cpu/test_target_triple_helper.h"
#include "tensorflow/compiler/xla/service/cpu/tests/cpu_codegen_test.h"

namespace xla {
namespace cpu {
namespace {

using CpuDynamicShapeTest = CpuCodegenTest;

TEST_F(CpuDynamicShapeTest, DynamicShapeR2) {
  HloComputation::Builder builder(TestName());

  xla::Shape dyn_input_shape = xla::ShapeUtil::MakeShape(xla::F32, {2, 4});
  dyn_input_shape.set_dynamic_dimension(0, true);
  HloInstruction* param_x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, dyn_input_shape, "x"));

  builder.AddInstruction(HloInstruction::CreateUnary(
      dyn_input_shape, HloOpcode::kNegate, param_x));
  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(builder.Build());

  std::string filecheck_pattern = R"(
; CHECK: %[[dyn_dim_size:.*]] = load i32, i32*
; CHECK: %[[i64_dyn_dim_size:.*]] = sext i32 %[[dyn_dim_size:.*]] to i64
; CHECK: icmp uge i64 %[[custom:.*]], %[[i64_dyn_dim_size:.*]]
; CHECK: %[[multiplier:.*]] = mul i64 1, %[[i64_dyn_dim_size:.*]]
; CHECK: mul nuw nsw i64 %[[custom:.*]], %[[multiplier:.*]]
)";

  CpuAotCompilationOptions options{
      /*triple=*/kTargetTripleForHost, /*cpu_name=*/kTargetCpuForHost,
      /*features=*/"",
      /*entry_point_name=*/"entry",
      /*relocation_model=*/CpuAotCompilationOptions::RelocationModel::Static};

  CompileAheadOfTimeAndVerifyIr(std::move(hlo_module), options,
                                filecheck_pattern,
                                /*match_optimized_ir=*/false);
}

}  // namespace
}  // namespace cpu
}  // namespace xla
