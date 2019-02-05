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

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"

namespace xla {
namespace {

class HloLivenessAnalysisTest : public HloTestBase {
 protected:
  HloLivenessAnalysisTest() {}

  // Run liveness analysis on the member module. For convenience returns a
  // reference to the generated analysis stored in analysis_.
  const HloLivenessAnalysis& RunLiveness(HloModule* module) {
    liveness_ = HloLivenessAnalysis::Run(*module).ConsumeValueOrDie();
    return *liveness_;
  }

  HloInstruction* GetInstruction(HloModule* module, const string& name) {
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
  auto module = ParseHloString(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    ROOT add = s32[] add(constant.1, constant.2)
  })")
                    .ValueOrDie();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
}

// Test that a dead add instruction is marked as dead by analysis.
TEST_F(HloLivenessAnalysisTest, DeadAdd) {
  auto module = ParseHloString(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    add.1 = s32[] add(constant.1, constant.2)
    ROOT add.2 = s32[] add(constant.1, constant.2)
  })")
                    .ValueOrDie();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add.2"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "add.1"), {}));
}

// Test that all output shape indices of entry root tuple (and defining
// instruction in its output) are marked live.
TEST_F(HloLivenessAnalysisTest, TupleAtEntryRoot) {
  auto module = ParseHloString(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    ROOT tuple.1 = (s32[], s32[]) tuple(constant.1, constant.2)
  })")
                    .ValueOrDie();
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
  auto module = ParseHloString(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(1)
    constant.2 = s32[] constant(2)
    constant.3 = s32[] constant(3)
    tuple.1 = (s32[], s32[]) tuple(constant.2, constant.3)
    ROOT tuple.2 = (s32[], s32[]) tuple(constant.1, tuple.1)
  })")
                    .ValueOrDie();
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

// Tests that GTE at entry root of Tuple instruction only propgates liveness
// to the live elements in tuple.
TEST_F(HloLivenessAnalysisTest, GteOfTuple) {
  auto module = ParseHloString(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    tuple.1 = (s32[], s32[]) tuple(constant.1, constant.2)
    ROOT get-tuple-element.1 = s32[] get-tuple-element(tuple.1), index=0
  })")
                    .ValueOrDie();
  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(
      liveness.IsLive(GetInstruction(module.get(), "get-tuple-element.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {0}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "tuple.1"), {1}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.1"), {}));
  EXPECT_FALSE(liveness.IsLive(GetInstruction(module.get(), "constant.2"), {}));
}

// Tests that GTE at entry root of nested Tuple instruction only propgates
// liveness to the live elements in tuple.
TEST_F(HloLivenessAnalysisTest, GteOfNestedTuple) {
  auto module = ParseHloString(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    constant.3 = s32[] constant(2)
    tuple.1 = (s32[], s32[]) tuple(constant.2, constant.3)
    tuple.2 = (s32[], s32[]) tuple(constant.1, tuple.1)
    ROOT get-tuple-element.1 = (s32[], s32[]) get-tuple-element(tuple.2), index=1
  })")
                    .ValueOrDie();
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
// propgates liveness to the live elements in tuple.
TEST_F(HloLivenessAnalysisTest, GteOfGteOfNestedTuple) {
  auto module = ParseHloString(R"(
  HloModule SimpleModule
  ENTRY SimpleComputation {
    constant.1 = s32[] constant(0)
    constant.2 = s32[] constant(1)
    constant.3 = s32[] constant(2)
    tuple.1 = (s32[], s32[]) tuple(constant.2, constant.3)
    tuple.2 = (s32[], s32[]) tuple(constant.1, tuple.1)
    get-tuple-element.1 = (s32[], s32[]) get-tuple-element(tuple.2), index=1
    ROOT get-tuple-element.2 = s32[] get-tuple-element(get-tuple-element.1), index=0
  })")
                    .ValueOrDie();
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
  auto module = ParseHloString(R"(
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
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    while.0 = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT get-tuple-element.4 = s32[] get-tuple-element(while.0), index=0
  })")
                    .ValueOrDie();
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
  auto module = ParseHloString(R"(
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
    get-tuple-element.4 = s32[] get-tuple-element(loop_var.2), index=1
    add.1 = s32[] add(get-tuple-element.3, get-tuple-element.4)
    constant.2 = s32[] constant(5)
    ROOT less-than = pred[] less-than(add.1, constant.2)
  }
  ENTRY SimpleLoop {
    constant.3 = s32[] constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[], s32[3]{0}) tuple(constant.3, constant.4)
    while.0 = (s32[], s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
    ROOT get-tuple-element.5 = s32[] get-tuple-element(while.0), index=0
  })")
                    .ValueOrDie();
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
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add.0"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "multiply.0"), {}));
}

// Tests that a use of while.result{0} propagates liveness to
// while.body.param{1} to while.body.root{1}, and then to while.body.param{2}.
TEST_F(HloLivenessAnalysisTest, WhileWithLiveTupleElements) {
  auto module = ParseHloString(R"(
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
    ROOT less-than = pred[] less-than(get-tuple-element.4, constant.1)
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
                    .ValueOrDie();

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
  auto module = ParseHloString(R"(
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

  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.3"), {}));
}

TEST_F(HloLivenessAnalysisTest, NestedWhileWithOutfeed) {
  auto module = ParseHloString(R"(
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
    ROOT less-than = pred[] less-than(get-tuple-element.3, constant.2)
  }
  OuterWhileCondition {
    cond_param.2 = (s32[]) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(cond_param.2), index=0
    constant.5 = s32[] constant(5)
    ROOT less-than.2 = pred[] less-than(get-tuple-element.5, constant.5)
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
                    .ValueOrDie();

  const HloLivenessAnalysis& liveness = RunLiveness(module.get());
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "add.2"), {}));
  EXPECT_TRUE(liveness.IsLive(GetInstruction(module.get(), "constant.3"), {}));
}

}  // namespace
}  // namespace xla
