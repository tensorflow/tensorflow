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

#include "xla/error_spec.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/tests/gpu_codegen_test.h"
#include "xla/tests/verified_hlo_module.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {

class GpuCopyTest : public GpuCodegenTest {};

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
  CompileAndVerifyIr(std::move(hlo_module), "; CHECK-NOT: define void @_copy",
                     /*match_optimized_ir=*/false);
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

}  // namespace gpu
}  // namespace xla
