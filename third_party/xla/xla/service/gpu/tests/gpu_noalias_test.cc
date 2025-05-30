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

#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {

class GpuNoAliasTest : public GpuCodegenTest {};

TEST_F(GpuNoAliasTest, Concat) {
  HloComputation::Builder builder(TestName());

  auto param_shape = ShapeUtil::MakeValidatedShape(F32, {2, 2}).value();
  HloInstruction* param_x = builder.AddInstruction(
      HloInstruction::CreateParameter(0, param_shape, "x"));
  HloInstruction* param_y = builder.AddInstruction(
      HloInstruction::CreateParameter(1, param_shape, "y"));
  HloInstruction* concat =
      builder.AddInstruction(HloInstruction::CreateConcatenate(
          ShapeUtil::MakeValidatedShape(F32, {2, 4}).value(),
          {param_x, param_y}, 1));
  builder.AddInstruction(HloInstruction::CreateConcatenate(
      ShapeUtil::MakeValidatedShape(F32, {2, 6}).value(), {concat, param_x},
      1));

  std::unique_ptr<HloComputation> computation = builder.Build();

  auto hlo_module = CreateNewVerifiedModule();
  hlo_module->AddEntryComputation(std::move(computation));

  // - After optimizations we have "concatenate(x, y, x)".
  // - We only pass the same parameters once, so the kernel will have these
  // parameters: (x, y, output), and all of them will be noalias.
  auto expected_ir = is_built_with_rocm_ ? R"(
CHECK: define amdgpu_kernel void @{{[a-zA-Z0-9_]+}}(ptr noalias align 16 dereferenceable(16) %{{[a-zA-Z0-9_]+}}, ptr noalias align 16 dereferenceable(16) %{{[a-zA-Z0-9_]+}}, ptr noalias align 128 dereferenceable(48) %{{[a-zA-Z0-9_]+}}) #0
  )"
                                         : R"(
CHECK: define ptx_kernel void @{{[a-zA-Z0-9_]+}}(ptr noalias align 16 dereferenceable(16) %{{[a-zA-Z0-9_]+}}, ptr noalias align 16 dereferenceable(16) %{{[a-zA-Z0-9_]+}}, ptr noalias align 128 dereferenceable(48) %{{[a-zA-Z0-9_]+}})
  )";
  CompileAndVerifyIr(std::move(hlo_module), expected_ir,
                     /*match_optimized_ir=*/false);
}

}  // namespace gpu
}  // namespace xla
