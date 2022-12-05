/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_runner.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

class DfsHloVisitorWithDefaultTest : public HloTestBase {};

TEST_F(DfsHloVisitorWithDefaultTest, DefaultElementwiseTest) {
  // Verify that HandleElementwiseBinary and HandleElementwiseUnary are called
  // on the appropriate HLO ops (elementwise binary/unary ops).

  class ElementwiseTestVisitor : public DfsHloVisitorWithDefault {
   public:
    Status DefaultAction(HloInstruction* hlo) override {
      // The HLO should be neither an elementwise unary nor binary op. These
      // cases are handled in HandleElementwiseBinary/Unary.
      TF_RET_CHECK(!(hlo->IsElementwise() && hlo->operand_count() == 2))
          << hlo->ToString();
      TF_RET_CHECK(!(hlo->IsElementwise() && hlo->operand_count() == 1))
          << hlo->ToString();
      return OkStatus();
    }

    Status HandleElementwiseBinary(HloInstruction* hlo) override {
      // HLO should be elementwise binary.
      TF_RET_CHECK(hlo->IsElementwise() && hlo->operand_count() == 2)
          << hlo->ToString();
      return OkStatus();
    }
    Status HandleElementwiseUnary(HloInstruction* hlo) override {
      // HLO should be elementwise unary.
      TF_RET_CHECK(hlo->IsElementwise() && hlo->operand_count() == 1)
          << hlo->ToString();
      return OkStatus();
    }
  };

  // HLO module contains are arbitrary mix of elementwise and non-elementwise
  // operations.
  const std::string& hlo_string = R"(
HloModule TestModule

ENTRY TestComputation {
  arg = f32[] parameter(0)
  tuple = (f32[]) tuple(arg)
  gte = f32[] get-tuple-element(tuple), index=0
  abs = f32[] abs(arg)
  add = f32[] add(arg, gte)
  broadcast = f32[42] broadcast(add), dimensions={}
  slice = f32[1] slice(broadcast), slice={[1:2]}
  copy = f32[] copy(arg)
  eq = pred[] compare(arg, gte), direction=EQ
  neg = f32[] negate(arg)
  ROOT convert = f64[] convert(f32[] arg)
})";
  std::unique_ptr<HloModule> module =
      ParseAndReturnVerifiedModule(hlo_string).value();
  ElementwiseTestVisitor visitor;
  TF_EXPECT_OK(module->entry_computation()->Accept(&visitor));
}

}  // namespace
}  // namespace xla
