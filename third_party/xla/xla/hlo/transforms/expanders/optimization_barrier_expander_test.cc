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

#include "xla/hlo/transforms/expanders/optimization_barrier_expander.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/shape.h"

namespace xla {
namespace {

class HloBarrierInstruction : public HloInstruction {
 public:
  HloBarrierInstruction(const Shape& shape,
                        absl::Span<HloInstruction* const> operands)
      : HloInstruction(HloOpcode::kOptimizationBarrier, shape) {
    for (HloInstruction* operand : operands) {
      AppendOperand(operand);
    }
  }
};

class OptimizationBarrierExpanderTest : public HloHardwareIndependentTestBase {
};

TEST_F(OptimizationBarrierExpanderTest, RemovesOptimizationBarrier) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  param0 = f32[10] parameter(0)
  add0 = f32[10] add(param0, param0)
  add1 = f32[10] add(param0, add0)
  tuple  = (f32[10], f32[10]) tuple(add0, add1)
  barrier = (f32[10], f32[10]) opt-barrier(tuple)
  gte = f32[10] get-tuple-element(barrier), index=0
  ROOT root = f32[10] add(gte, param0)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  OptimizationBarrierExpander expander;
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(
      FindInstructions(module.get(), HloOpcode::kOptimizationBarrier).empty());
  VLOG(1) << module->ToString();
}

TEST_F(OptimizationBarrierExpanderTest, RemovesOnlySingularOptBarrier) {
  const char* hlo = R"(
HloModule module

ENTRY main {
  param0 = f32[10] parameter(0)
  param1 = f32[10] parameter(1)
  add0 = f32[10] add(param0, param1)
  barrier = f32[10] opt-barrier(add0)
  ROOT add1 = f32[10] add(barrier, param0)
}
)";

  ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                       ParseAndReturnVerifiedModule(hlo));

  OptimizationBarrierExpander expander(
      /*only_remove_singleton_opt_barriers=*/true);
  ASSERT_OK_AND_ASSIGN(bool changed, expander.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_TRUE(
      FindInstructions(module.get(), HloOpcode::kOptimizationBarrier).empty());
  VLOG(1) << module->ToString();
}

}  // namespace
}  // namespace xla
