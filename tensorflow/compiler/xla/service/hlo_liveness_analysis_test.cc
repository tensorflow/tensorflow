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

#include "tensorflow/compiler/xla/service/hlo_liveness_analysis.h"

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/test.h"

namespace xla {
namespace {

class HloLivenessAnalysisTest : public HloTestBase {
 protected:
  HloLivenessAnalysisTest() {}

  // Run liveness analysis on the member module. For convenience returns a
  // reference to the generated analysis stored in analysis_.
  const HloLivenessAnalysis& RunLiveness(HloModule* module) {
    liveness_ = HloLivenessAnalysis::Run(*module).value();
    return *liveness_;
  }

  HloInstruction* GetInstruction(HloModule* module, const std::string& name) {
    HloInstruction* to_return = nullptr;
    for (auto* comp : module->computations()) {
      for (auto* inst : comp->instructions()) {
        if (inst->name() == name) {
          to_return = inst;
          break;
        }
      }
    }
    return CHECK_NOTNULL(to_return);
  }

  std::unique_ptr<HloLivenessAnalysis> liveness_;
};

// Test that add instruction at entry root is live at all output shape indices.
TEST_F(HloLivenessAnalysisTest, AddAtEntryRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    ROOT add = s32[] add(constant.1, constant.2)
  })")
                    .value();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
}

// Test that a dead add instruction is marked as dead by analysis.
TEST_F(HloLivenessAnalysisTest, DeadAdd) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    add.1 = s32[] add(constant.1, constant.2)
    ROOT add.2 = s32[] add(constant.1, constant.2)
  })")
                    .value();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add.2"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "add.1"), {}));
}

// Test that all output shape indices of entry root tuple (and defining
// instruction in its output) are marked live.
TEST_F(HloLivenessAnalysisTest, TupleAtEntryRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    ROOT tuple.1 = (s32[], s32[]) tuple(constant.1, constant.2)
  })")
                    .value();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
}

// Tests that all outputs of nested tuple and entry root (and defining
// instruction values appearing in its output) are marked live.
TEST_F(HloLivenessAnalysisTest, NestedTupleAtEntryRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(1)
    constant.2 = s32[] constant(2)
    constant.3 = s32[] constant(3)
    tuple.1 = (s32[], s32[]) tuple(constant.2, constant.3)
    ROOT tuple.2 = (s32[], (s32[], s32[])) tuple(constant.1, tuple.1)
  })")
                    .value();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1, 0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1, 1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.3"), {}));
}

// Tests that GTE at entry root of Tuple instruction only propagates liveness
// to the live elements in tuple.
TEST_F(HloLivenessAnalysisTest, GteOfTuple) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    tuple.1 = (s32[], s32[]) tuple(constant.1, constant.2)
    ROOT get-tuple-element.1 = s32[] get-tuple-element(tuple.1), index=0
  })")
                    .value();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(
      liveness.IsLive(GetInstruction(module.get(), "get-tuple-element.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
}

// Tests that GTE at entry root of nested Tuple instruction only propagates
// liveness to the live elements in tuple.
TEST_F(HloLivenessAnalysisTest, GteOfNestedTuple) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    constant.3 = s32[] constant(2)
    tuple.1 = (s32[], s32[]) tuple(constant.2, constant.3)
    tuple.2 = (s32[], (s32[], s32[])) tuple(constant.1, tuple.1)
    ROOT get-tuple-element.1 = (s32[], s32[]) get-tuple-element(tuple.2), index=1
  })")
                    .value();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(
      liveness.IsLive(GetInstruction(module.get(), "get-tuple-element.1"), {}));
  EXPECT_TRUE(liveness.IsLive(
      GetInstruction(module.get(), "get-tuple-element.1"), {0}));
  EXPECT_TRUE(liveness.IsLive(
      GetInstruction(module.get(), "get-tuple-element.1"), {1}));

  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1, 0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1, 1}));

  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {1}));

  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.3"), {}));
}

// Tests that GTE of GTE (at entry root) of nested Tuple instruction only
// propagates liveness to the live elements in tuple.
TEST_F(HloLivenessAnalysisTest, GteOfGteOfNestedTuple) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    constant.3 = s32[] constant(2)
    tuple.1 = (s32[], s32[]) tuple(constant.2, constant.3)
    tuple.2 = (s32[], (s32[], s32[])) tuple(constant.1, tuple.1)
    get-tuple-element.1 = (s32[], s32[]) get-tuple-element(tuple.2), index=1
    ROOT get-tuple-element.2 = s32[] get-tuple-element(get-tuple-element.1), index=0
  })")
                    .value();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(
      liveness.IsLive(GetInstruction(module.get(), "get-tuple-element.2"), {}));

  EXPECT_TRUE(
      liveness.IsLive(GetInstruction(module.get(), "get-tuple-element.1"), {}));
  EXPECT_TRUE(liveness.IsLive(
      GetInstruction(module.get(), "get-tuple-element.1"), {0}));
  EXPECT_FALSE(liveness.IsLive(
      GetInstruction(module.get(), "get-tuple-element.1"), {1}));

  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1, 0}));
  EXPECT_FALSE(
      liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1, 1}));

  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {1}));

  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "constant.3"), {}));
}

// Test that live/dead while tuple elements are marked live/dead correctly.
TEST_F(HloLivenessAnalysisTest, WhileWithDeadTupleElement) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply.0 = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple.0 = (s32[], s32[3]{0}) tuple(add.0, multiply.0)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[] constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    while.0 = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT get-tuple-element.4 = s32[] get-tuple-element(while.0), index=0
  })")
                    .value();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(
      liveness.IsLive(GetInstruction(module.get(), "get-tuple-element.4"), {}));

  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.0"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.0"), {0}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "while.0"), {1}));

  // While operand.
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.3"), {}));

  // While body.
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.0"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.0"), {0}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "tuple.0"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add.0"), {}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "multiply.0"), {}));
}

// Tests that a tuple element live in while.cond computation, propagates
// liveness to while.body.root/while.result/while.operand (where it is unused).
TEST_F(HloLivenessAnalysisTest, WhileCondPropagatesLiveness) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleLoop
  add_S32 {
    lhs = s32[] parameter(0)
    rhs = s32[] parameter(1)
    ROOT add = s32[] add(lhs, rhs)
  }
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    constant.1 = s32[] constant(1)
    add.0 = s32[] add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(loop_var.1), index=1
    multiply.0 = s32[3]{0} multiply(get-tuple-element.2, get-tuple-element.2)
    ROOT tuple.0 = (s32[], s32[3]{0}) tuple(add.0, multiply.0)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    get-tuple-element.4 = s32[3]{0} get-tuple-element(loop_var.2), index=1
    zero = s32[] constant(0)
    reduce = s32[] reduce(get-tuple-element.4, zero), dimensions={0}, to_apply=add_S32
    add.1 = s32[] add(get-tuple-element.3, reduce)
    constant.2 = s32[] constant(5)
    ROOT less-than = pred[] compare(add.1, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    while.0 = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT get-tuple-element.5 = s32[] get-tuple-element(while.0), index=0
  })")
                    .value();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(
      liveness.IsLive(GetInstruction(module.get(), "get-tuple-element.5"), {}));

  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.0"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.0"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.0"), {1}));

  // While operand.
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.3"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.4"), {}));

  // While body.
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.0"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.0"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.0"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "multiply.0"), {}));
}

// Tests that a use of while.result{0} propagates liveness to
// while.body.param{1} to while.body.root{1}, and then to while.body.param{2}.
TEST_F(HloLivenessAnalysisTest, WhileWithLiveTupleElements) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[], s32[], s32[]) parameter(0)
    get-tuple-element.1 = s32[] get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[] get-tuple-element(loop_var.1), index=1
    add.1 = s32[] add(get-tuple-element.1, get-tuple-element.2)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.1), index=2
    multiply.1 = s32[] multiply(get-tuple-element.3, get-tuple-element.3)
    ROOT tuple.1 = (s32[], s32[], s32[]) tuple(add.1, get-tuple-element.3, multiply.1)
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[], s32[], s32[]) parameter(0)
    get-tuple-element.4 = s32[] get-tuple-element(loop_var.2), index=0
    constant.1 = s32[] constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.4, constant.1), direction=LT
  }
  ENTRY SimpleLoop {
    constant.2 = s32[] constant(0)
    constant.3 = s32[] constant(1)
    constant.4 = s32[] constant(2)
    tuple.2 = (s32[], s32[], s32[]) tuple(constant.2, constant.3, constant.4)
    while.1 = (s32[], s32[], s32[]) while(tuple.2), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT get-tuple-element.5 = s32[] get-tuple-element(while.1), index=0
  })")
                    .value();

  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(
      liveness.IsLive(GetInstruction(module.get(), "get-tuple-element.5"), {}));

  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.1"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.1"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.1"), {2}));
  // While operand.
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.2"), {2}));
  // While body root.
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {2}));
  // While body param.
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "loop_var.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "loop_var.1"), {0}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "loop_var.1"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "loop_var.1"), {2}));
}

TEST_F(HloLivenessAnalysisTest, WhileWithOutfeed) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule OutfeedLoop
  WhileBody {
    body_param = (s32[]) parameter(0)
    token0 = token[] after-all()
    constant.2 = s32[] constant(2)
    outfeed_tuple = (s32[]) outfeed(constant.2, token0)
    get-tuple-element.1 = s32[] get-tuple-element(body_param), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]) tuple(add)
  }
  WhileCondition {
    cond_param = (s32[]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(cond_param), index=0
    constant.2 = s32[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    tuple.1 = (s32[]) tuple(constant.3)
    while = (s32[]) while(tuple.1), condition=WhileCondition,
      body=WhileBody
    ROOT rtuple = () tuple()
  })")
                    .value();

  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.3"), {}));
}

TEST_F(HloLivenessAnalysisTest, NestedWhileWithOutfeed) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule OutfeedLoop
  InnerWhileBody {
    body_param = (s32[]) parameter(0)
    token0 = token[] after-all()
    constant.2 = s32[] constant(2)
    outfeed_tuple = (s32[]) outfeed(constant.2, token0)
    get-tuple-element.1 = s32[] get-tuple-element(body_param), index=0
    constant.1 = s32[] constant(1)
    add = s32[] add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[]) tuple(add)
  }
  InnerWhileCondition {
    cond_param = (s32[]) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(cond_param), index=0
    constant.2 = s32[] constant(10)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  OuterWhileCondition {
    cond_param.2 = (s32[]) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(cond_param.2), index=0
    constant.5 = s32[] constant(5)
    ROOT less-than.2 = pred[] compare(get-tuple-element.5, constant.5), direction=LT
  }
  OuterWhileBody {
    body_param.2 = (s32[]) parameter(0)
    get-tuple-element.8 = s32[] get-tuple-element(body_param.2), index=0
    constant.6 = s32[] constant(0)
    tuple.2 = (s32[]) tuple(constant.6)
    inner_while = (s32[]) while(tuple.2), condition=InnerWhileCondition,
      body=InnerWhileBody
    constant.7 = s32[] constant(1)
    add.2 = s32[] add(get-tuple-element.8, constant.7)
    ROOT rtuple = (s32[]) tuple(add.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    tuple.1 = (s32[]) tuple(constant.3)
    while = (s32[]) while(tuple.1), condition=OuterWhileCondition,
      body=OuterWhileBody
    ROOT rtuple = () tuple()
  })")
                    .value();

  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add.2"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.3"), {}));
}

TEST_F(HloLivenessAnalysisTest, PropagateLivenessFromConditionalComputation) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule main.67

%region_0.10 (Arg_0.11: (s32[], s32[], f32[1024,3], s32[1])) -> (s32[], s32[], f32[1024,3], s32[1]) {
  %Arg_0.11 = (s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) parameter(0)
  %get-tuple-element.17 = s32[] get-tuple-element((s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) %Arg_0.11), index=0, metadata={op_name="while"}
  %constant.13 = s32[] constant(1)
  %add.25 = s32[] add(s32[] %get-tuple-element.17, s32[] %constant.13), metadata={op_name="while/add_1"}
  %get-tuple-element.18 = s32[] get-tuple-element((s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) %Arg_0.11), index=1, metadata={op_name="while"}
  %add.22 = s32[] add(s32[] %get-tuple-element.18, s32[] %constant.13), metadata={op_name="while/add"}
  %get-tuple-element.19 = f32[1024,3]{1,0} get-tuple-element((s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) %Arg_0.11), index=2, metadata={op_name="while"}
  %constant.16 = f32[] constant(0)
  %constant.15 = f32[] constant(1)
  %rng.21 = f32[3]{0} rng(f32[] %constant.16, f32[] %constant.15), distribution=rng_uniform, metadata={op_name="while/random_uniform/RandomUniform"}
  %reshape.23 = f32[1,3]{1,0} reshape(f32[3]{0} %rng.21), metadata={op_name="while/TensorArrayV2Write/TensorListSetItem"}
  %constant.12 = s32[] constant(0)
  %dynamic-update-slice.24 = f32[1024,3]{1,0} dynamic-update-slice(f32[1024,3]{1,0} %get-tuple-element.19, f32[1,3]{1,0} %reshape.23, s32[] %get-tuple-element.18, s32[] %constant.12), metadata={op_name="while/TensorArrayV2Write/TensorListSetItem"}
  %get-tuple-element.20 = s32[1]{0} get-tuple-element((s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) %Arg_0.11), index=3, metadata={op_name="while"}
  ROOT %tuple.26 = (s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) tuple(s32[] %add.25, s32[] %add.22, f32[1024,3]{1,0} %dynamic-update-slice.24, s32[1]{0} %get-tuple-element.20), metadata={op_name="while"}
}

%region_1.27 (Arg_0.28: (s32[], s32[], f32[1024,3], s32[1])) -> pred[] {
  %Arg_0.28 = (s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) parameter(0)
  %get-tuple-element.30 = s32[] get-tuple-element((s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) %Arg_0.28), index=1, metadata={op_name="while"}
  %constant.29 = s32[] constant(1024)
  ROOT %compare.31 = pred[] compare(s32[] %get-tuple-element.30, s32[] %constant.29), direction=LT, metadata={op_name="while/Less"}
}

%region_2.42 (Arg_0.43: (f32[3,32,32,3], token[])) -> (pred[], token[]) {
  %constant.44 = pred[] constant(true)
  %Arg_0.43 = (f32[3,32,32,3]{3,2,1,0}, token[]) parameter(0)
  %get-tuple-element.52 = f32[3,32,32,3]{3,2,1,0} get-tuple-element((f32[3,32,32,3]{3,2,1,0}, token[]) %Arg_0.43), index=0, metadata={op_name="image_sample/write_summary/summary_cond"}
  %constant.49 = f32[] constant(255.5)
  %broadcast.50 = f32[3,32,32,3]{3,2,1,0} broadcast(f32[] %constant.49), dimensions={}, metadata={op_name="image_sample/write_summary/summary_cond/convert_image/Mul"}
  %multiply.53 = f32[3,32,32,3]{3,2,1,0} multiply(f32[3,32,32,3]{3,2,1,0} %get-tuple-element.52, f32[3,32,32,3]{3,2,1,0} %broadcast.50), metadata={op_name="image_sample/write_summary/summary_cond/convert_image/Mul"}
  %constant.47 = f32[] constant(0)
  %broadcast.48 = f32[3,32,32,3]{3,2,1,0} broadcast(f32[] %constant.47), dimensions={}, metadata={op_name="image_sample/write_summary/summary_cond/convert_image/Maximum"}
  %maximum.54 = f32[3,32,32,3]{3,2,1,0} maximum(f32[3,32,32,3]{3,2,1,0} %multiply.53, f32[3,32,32,3]{3,2,1,0} %broadcast.48), metadata={op_name="image_sample/write_summary/summary_cond/convert_image/Maximum"}
  %constant.45 = f32[] constant(255)
  %broadcast.46 = f32[3,32,32,3]{3,2,1,0} broadcast(f32[] %constant.45), dimensions={}, metadata={op_name="image_sample/write_summary/summary_cond/convert_image/Minimum"}
  %minimum.55 = f32[3,32,32,3]{3,2,1,0} minimum(f32[3,32,32,3]{3,2,1,0} %maximum.54, f32[3,32,32,3]{3,2,1,0} %broadcast.46), metadata={op_name="image_sample/write_summary/summary_cond/convert_image/Minimum"}
  %convert.56 = u8[3,32,32,3]{3,2,1,0} convert(f32[3,32,32,3]{3,2,1,0} %minimum.55), metadata={op_name="image_sample/write_summary/summary_cond/convert_image"}
  %get-tuple-element.51 = token[] get-tuple-element((f32[3,32,32,3]{3,2,1,0}, token[]) %Arg_0.43), index=1, metadata={op_name="image_sample/write_summary/summary_cond"}
  %send.57 = (u8[3,32,32,3]{3,2,1,0}, u32[], token[]) send(u8[3,32,32,3]{3,2,1,0} %convert.56, token[] %get-tuple-element.51), channel_id=2, is_host_transfer=true, frontend_attributes={_xla_host_transfer_original_type="u8",_xla_host_transfer_rendezvous="host_compute_channel_0_args_dtoh_0"}, metadata={op_name="image_sample/write_summary/summary_cond/encode_each_image/TensorArrayUnstack/TensorListFromTensor"}
  %send-done.58 = token[] send-done((u8[3,32,32,3]{3,2,1,0}, u32[], token[]) %send.57), channel_id=2, is_host_transfer=true, frontend_attributes={_xla_host_transfer_original_type="u8",_xla_host_transfer_rendezvous="host_compute_channel_0_args_dtoh_0"}, metadata={op_name="image_sample/write_summary/summary_cond/encode_each_image/TensorArrayUnstack/TensorListFromTensor"}
  ROOT %tuple.59 = (pred[], token[]) tuple(pred[] %constant.44, token[] %send-done.58), metadata={op_name="image_sample/write_summary/summary_cond"}
}

%region_3.60 (Arg_0.61: (f32[3,32,32,3], token[])) -> (pred[], token[]) {
  %constant.62 = pred[] constant(false)
  %Arg_0.61 = (f32[3,32,32,3]{3,2,1,0}, token[]) parameter(0)
  %get-tuple-element.63 = token[] get-tuple-element((f32[3,32,32,3]{3,2,1,0}, token[]) %Arg_0.61), index=1, metadata={op_name="image_sample/write_summary/summary_cond"}
  ROOT %tuple.64 = (pred[], token[]) tuple(pred[] %constant.62, token[] %get-tuple-element.63), metadata={op_name="image_sample/write_summary/summary_cond"}
}

ENTRY %main.67 (arg_tuple.1: (s32[])) -> () {
  %arg_tuple.1 = (s32[]{:T(256)}) parameter(0)
  %get-tuple-element.2 = s32[]{:T(256)} get-tuple-element((s32[]{:T(256)}) %arg_tuple.1), index=0
  %constant.3 = s32[] constant(0)
  %compare.8 = pred[]{:T(256)} compare(s32[]{:T(256)} %get-tuple-element.2, s32[] %constant.3), direction=EQ, metadata={op_name="image_sample/write_summary/Equal"}
  %constant.5 = f32[] constant(0)
  %broadcast.6 = f32[1024,3]{1,0} broadcast(f32[] %constant.5), dimensions={}, metadata={op_name="tokens_accumulator"}
  %constant.4 = s32[1]{0} constant({1024})
  %tuple.9 = (s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) tuple(s32[] %constant.3, s32[] %constant.3, f32[1024,3]{1,0} %broadcast.6, s32[1]{0} %constant.4), metadata={op_name="while"}
  %while.32 = (s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) while((s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) %tuple.9), condition=%region_1.27, body=%region_0.10, metadata={op_name="while"}
  %get-tuple-element.33 = f32[1024,3]{1,0} get-tuple-element((s32[], s32[], f32[1024,3]{1,0}, s32[1]{0}) %while.32), index=2, metadata={op_name="while"}
  %transpose.34 = f32[3,1024]{0,1} transpose(f32[1024,3]{1,0} %get-tuple-element.33), dimensions={1,0}, metadata={op_name="transpose.transpose/perm"}
  %reshape.35 = f32[3,32,32,1]{3,2,1,0} reshape(f32[3,1024]{0,1} %transpose.34), metadata={op_name="Reshape"}
  %broadcast.36 = f32[3,32,32,1]{3,2,1,0} broadcast(f32[3,32,32,1]{3,2,1,0} %reshape.35), dimensions={0,1,2,3}, metadata={op_name="Tile"}
  %reshape.37 = f32[3,32,32]{2,1,0} reshape(f32[3,32,32,1]{3,2,1,0} %broadcast.36), metadata={op_name="Tile"}
  %broadcast.38 = f32[3,32,32,3]{3,2,1,0} broadcast(f32[3,32,32]{2,1,0} %reshape.37), dimensions={0,1,2}, metadata={op_name="Tile"}
  %after-all.7 = token[] after-all(), metadata={op_name="image_sample/write_summary/summary_cond"}
  %send.39 = (pred[]{:T(256)}, u32[], token[]) send(pred[]{:T(256)} %compare.8, token[] %after-all.7), channel_id=1, is_host_transfer=true, frontend_attributes={_xla_host_transfer_original_type="pred",_xla_host_transfer_rendezvous="if_predicate_channel_1_dtoh_0"}, metadata={op_name="image_sample/write_summary/summary_cond"}
  %send-done.40 = token[] send-done((pred[]{:T(256)}, u32[], token[]) %send.39), channel_id=1, is_host_transfer=true, frontend_attributes={_xla_host_transfer_original_type="pred",_xla_host_transfer_rendezvous="if_predicate_channel_1_dtoh_0"}, metadata={op_name="image_sample/write_summary/summary_cond"}
  %tuple.41 = (f32[3,32,32,3]{3,2,1,0}, token[]) tuple(f32[3,32,32,3]{3,2,1,0} %broadcast.38, token[] %send-done.40), metadata={op_name="image_sample/write_summary/summary_cond"}
  %conditional.65 = (pred[], token[]) conditional(pred[]{:T(256)} %compare.8, (f32[3,32,32,3]{3,2,1,0}, token[]) %tuple.41, (f32[3,32,32,3]{3,2,1,0}, token[]) %tuple.41), true_computation=%region_2.42, false_computation=%region_3.60, metadata={op_name="image_sample/write_summary/summary_cond"}
  ROOT %tuple.66 = () tuple()
}
)")
                    .value();

  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(
      liveness.IsLive(GetInstruction(module.get(), "conditional.65"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.41"), {}));
  EXPECT_TRUE(liveness.IsLive(
      GetInstruction(module.get(), "get-tuple-element.33"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "while.32"), {}));
  EXPECT_TRUE(liveness.IsLive(
      GetInstruction(module.get(), "dynamic-update-slice.24"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "send.57"), {}));
}

}  // namespace
}  // namespace xla
