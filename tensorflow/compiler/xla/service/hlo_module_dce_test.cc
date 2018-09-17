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

#include "tensorflow/compiler/xla/service/hlo_module_dce.h"

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace {

class HloModuleDceTest : public HloTestBase {
 protected:
  HloModuleDceTest() {}

  // Returns whether the given instruction exists in the given computation.
  bool HasInstruction(const HloComputation& computation,
                      const HloInstruction* instruction) {
    return std::find(computation.instructions().begin(),
                     computation.instructions().end(),
                     instruction) != computation.instructions().end();
  }

  // Returns whether the while instruction with name 'while_name' in
  // 'computation' passes through its tuple element at 'tuple_index' from
  // parameter to root instruction.
  bool WhileBodyHasPassThroughTupleElement(const HloComputation* computation,
                                           const string& while_name,
                                           const int64 tuple_index) {
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kWhile &&
          instruction->name() == while_name) {
        auto* while_body_comp = instruction->while_body();
        auto* while_body_param = while_body_comp->parameter_instruction(0);
        auto* while_body_root = while_body_comp->root_instruction();
        if (while_body_root->opcode() != HloOpcode::kTuple) {
          return false;
        }
        auto* operand = while_body_root->operand(tuple_index);
        if (operand->opcode() == HloOpcode::kGetTupleElement &&
            operand->tuple_index() == tuple_index &&
            operand->operand(0) == while_body_param) {
          return true;
        }
        return false;
      }
    }
    return false;
  }
};

// Tests that a while with all outputs live is unmodified.
TEST_F(HloModuleDceTest, WhileWithLiveOutputs) {
  auto module = ParseHloString(R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(5)
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  })")
                    .ValueOrDie();

  HloModuleDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 0));
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 1));
}

// Tests a while loop with one unused output (which is used in the while loop
// body by an instruction with side-effects: rng) is unmodified.
TEST_F(HloModuleDceTest, WhileWithUnusedSideEffectingTupleElement) {
  auto module = ParseHloString(R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], f32[]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = f32[] get-tuple-element(loop_var.1), index=1
    constant.2 = f32[] constant(1.0)
    rng = f32[] rng(constant.2, get-tuple-element.2), distribution=rng_uniform
    add.1 = s32[] add(get-tuple-element.2, constant.2)
    ROOT tuple = (s32[], f32[]) tuple(add, add.1)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], f32[]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.3 = s32[] constant(5)
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.3)
  }
  ENTRY SimpleLoop {
    constant.4 = s32[] constant(0)
    constant.5 = f32[] constant(0.0)
    tuple.1 = (s32[], f32[]) tuple(constant.4, constant.5)
    while = (s32[], f32[]) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT get-tuple-element.4 = s32[] get-tuple-element(while), index=0
  })")
                    .ValueOrDie();

  HloModuleDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 0));
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 1));
}

// Tests that a while loop with one dead tuple element at {1} has its while
// loop body modified to make that tuple element pass-through the while body.
TEST_F(HloModuleDceTest, OneWhileWithDeadTupleElement) {
  auto module = ParseHloString(R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(5)
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    while = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT get-tuple-element.4 = s32[] get-tuple-element(while), index=0
  })")
                    .ValueOrDie();

  HloModuleDCE dce;
  // While tuple element {1} should not be pass-through before ModuleDCE.
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 1));
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 0));
  // While tuple element {1} should now be pass-through after ModuleDCE.
  EXPECT_TRUE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                  "while", 1));
}

// Tests that a tuple element {1} used by condition computation (which appears
// dead in while.body{1} and at while.result{1}) propgates liveness of this
// tuple element to while.body{1} and at while.result{1}.
TEST_F(HloModuleDceTest, OneWhileWithTupleElementUsedByCond) {
  auto module = ParseHloString(R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    multiply = s32[] multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[]) tuple(add, multiply)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=1
    constant.2 = s32[] constant(5)
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[] constant(0)
    tuple.1 = (s32[], s32[]) tuple(constant.3, constant.4)
    while = (s32[], s32[]) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT get-tuple-element.4 = s32[] get-tuple-element(while), index=0
  })")
                    .ValueOrDie();

  HloModuleDCE dce;
  // While tuple element {1} should not be pass-through before ModuleDCE.
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 1));
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 0));
  // While tuple element {1} still be pass-through after ModuleDCE.
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 1));
}

// Tests that HloModuleDCE can remove a dead tuple element at index {1} between
// two dependent while loops.
TEST_F(HloModuleDceTest, TwoWhilesWithDeadTupleElement) {
  auto module = ParseHloString(R"(
  HloModule SimpleLoop
  SimpleLoop.body0 {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[], s32[3]{0}) tuple(add, multiply)
  }
  SimpleLoop.condition0 {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(5)
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  SimpleLoop.body1 {
    loop_var.3 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.4 = s32[] get-tuple-element(loop_var.3), index=0
    constant.3 = s32[] constant(1)
    add.1 = s32[] add(get-tuple-element.4, constant.3)
    get-tuple-element.5 = s32[3]{0} get-tuple-element(loop_var.3), index=1
    multiply.1 = s32[3]{0} multiply(get-tuple-element.5, get-tuple-element.5)
    ROOT tuple.1 = (s32[], s32[3]{0}) tuple(add.1, multiply.1)
  }
  SimpleLoop.condition1 {
    loop_var.4 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.6 = s32[] get-tuple-element(loop_var.4), index=0
    constant.4 = s32[] constant(5)
    ROOT less-than.1 = pred[] less-than(get-tuple-element.6, constant.4)
  }
  ENTRY SimpleLoop {
    constant.5 = s32[] constant(0)
    constant.6 = s32[3]{0} constant({0, 1, 2})
    tuple.2 = (s32[], s32[3]{0}) tuple(constant.5, constant.6)
    while.1 = (s32[], s32[3]{0}) while(tuple.2), condition=
      SimpleLoop.condition0, body=SimpleLoop.body0
    get-tuple-element.7 = s32[] get-tuple-element(while.1), index=0
    tuple.3 = (s32[], s32[3]{0}) tuple(get-tuple-element.7, constant.6)
    while.2 = (s32[], s32[3]{0}) while(tuple.3), condition=
      SimpleLoop.condition1, body=SimpleLoop.body1
    ROOT get-tuple-element.8 = s32[] get-tuple-element(while.2), index=0
  })")
                    .ValueOrDie();

  HloModuleDCE dce;
  // Before HloModuleDCE while.1 and while.2 should not have pass-thru elements.
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while.1", 1));
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while.2", 1));
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());
  // After HloModuleDCE while.1 and while.2 should have pass-thru elements,
  // after being modified to pass through unused tuple element {1}.
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while.1", 0));
  EXPECT_TRUE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                  "while.1", 1));
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while.2", 0));
  EXPECT_TRUE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                  "while.2", 1));
}

// Tests that HloModuleDCE can remove a dead tuple element at while.1{0} and
// while.2{1}, between two dependent while loops.
TEST_F(HloModuleDceTest, TwoWhilesWithDeadTupleElementSwizzled) {
  auto module = ParseHloString(R"(
  HloModule SimpleLoop
  SimpleLoop.body0 {
    loop_var.1 = (s32[3]{0}, s32[]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=1
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=0
    multiply = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple = (s32[3]{0}, s32[]) tuple(multiply, add)
  }
  SimpleLoop.condition0 {
    loop_var.2 = (s32[3]{0}, s32[]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=1
    constant.2 = s32[] constant(5)
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  SimpleLoop.body1 {
    loop_var.3 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.4 = s32[] get-tuple-element(loop_var.3), index=0
    constant.3 = s32[] constant(1)
    add.1 = s32[] add(get-tuple-element.4, constant.3)
    get-tuple-element.5 = s32[3]{0} get-tuple-element(loop_var.3), index=1
    multiply.1 = s32[3]{0} multiply(get-tuple-element.5, get-tuple-element.5)
    ROOT tuple.1 = (s32[], s32[3]{0}) tuple(add.1, multiply.1)
  }
  SimpleLoop.condition1 {
    loop_var.4 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.6 = s32[] get-tuple-element(loop_var.4), index=0
    constant.4 = s32[] constant(5)
    ROOT less-than.1 = pred[] less-than(get-tuple-element.6, constant.4)
  }
  ENTRY SimpleLoop {
    constant.5 = s32[] constant(0)
    constant.6 = s32[3]{0} constant({0, 1, 2})
    tuple.2 = (s32[3]{0}, s32[]) tuple(constant.6, constant.5)
    while.1 = (s32[3]{0}, s32[]) while(tuple.2), condition=
      SimpleLoop.condition0, body=SimpleLoop.body0
    get-tuple-element.7 = s32[] get-tuple-element(while.1), index=1
    tuple.3 = (s32[], s32[3]{0}) tuple(get-tuple-element.7, constant.6)
    while.2 = (s32[], s32[3]{0}) while(tuple.3), condition=
      SimpleLoop.condition1, body=SimpleLoop.body1
    ROOT get-tuple-element.8 = s32[] get-tuple-element(while.2), index=0
  })")
                    .ValueOrDie();

  HloModuleDCE dce;
  // Before HloModuleDCE while.1{0} and while.2{1} should not be pass-thru.
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while.1", 0));
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while.2", 1));
  EXPECT_TRUE(dce.Run(module.get()).ValueOrDie());
  // After HloModuleDCE while.1{0} and while.2{1} not be pass-thru elements.
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while.1", 1));
  EXPECT_TRUE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                  "while.1", 0));
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while.2", 0));
  EXPECT_TRUE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                  "while.2", 1));
}

// Tests that a while whose body has outfeed operations is not DCE-ed.
TEST_F(HloModuleDceTest, WhileWithOutfeed) {
  auto module = ParseHloString(R"(
  HloModule OutfeedLoop
  WhileBody {
    body_param = (s32[]) parameter(0)
    token = token[] after-all()
    constant.2 = s32[] constant(2)
    outfeed_tuple = (s32[]) outfeed(constant.2, token)
    get-tuple-element.1 = s32[] get-tuple-element(body_param), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]) tuple(add)
  }
  WhileCondition {
    cond_param = (s32[]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(cond_param), index=0
    constant.2 = s32[] constant(10)
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    tuple.1 = (s32[]) tuple(constant.3)
    while = (s32[]) while(tuple.1), condition=WhileCondition,
      body=WhileBody
    ROOT rtuple = () tuple()
  })")
                    .ValueOrDie();

  HloModuleDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 0));
}

// Tests that if a loop variable is not referenced outside of a kWhile, the loop
// variable changes are not elided within the loop body, if the condition
// computation uses them.
TEST_F(HloModuleDceTest, WhileWithOnlyLoopVariableBumping) {
  auto module = ParseHloString(R"(
  HloModule InfiniteLoop
  WhileBody {
    body_param = (s32[], s32[]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(body_param), index=0
    get-tuple-element.2 = s32[] get-tuple-element(body_param), index=1
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[], s32[]) tuple(add, get-tuple-element.2)
  }
  WhileCondition {
    cond_param = (s32[], s32[]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(cond_param), index=0
    constant.2 = s32[] constant(10)
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  ENTRY SimpleLoop {
    p0 = (s32[]) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(p0), index=0
    constant.3 = s32[] constant(0)
    tuple.1 = (s32[], s32[]) tuple(constant.3, get-tuple-element.5)
    while = (s32[], s32[]) while(tuple.1), condition=WhileCondition,
      body=WhileBody
    ROOT get-tuple-element.4 = s32[] get-tuple-element(while), index=1
  })")
                    .ValueOrDie();

  HloModuleDCE dce;
  EXPECT_FALSE(dce.Run(module.get()).ValueOrDie());
  EXPECT_FALSE(WhileBodyHasPassThroughTupleElement(module->entry_computation(),
                                                   "while", 0));
}

}  // namespace
}  // namespace xla
