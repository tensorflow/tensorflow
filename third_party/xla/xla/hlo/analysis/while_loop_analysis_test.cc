/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/hlo/analysis/while_loop_analysis.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_replace.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/service/constant_value.h"
#include "xla/service/value_range.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace {

class WhileLoopAnalysisTest : public HloHardwareIndependentTestBase {
 protected:
  absl::StatusOr<int64_t> MakeWhileLoopAndGetTripCount(int init, int limit,
                                                       int step,
                                                       ComparisonDirection dir);
  absl::StatusOr<Range> MakeWhileLoopAndGetRange(int init, int limit, int step,
                                                 ComparisonDirection dir);
};

absl::StatusOr<int64_t> WhileLoopAnalysisTest::MakeWhileLoopAndGetTripCount(
    int init, int limit, int step, ComparisonDirection dir) {
  std::string hlo_string_template = R"(
  HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val = f32[2] get-tuple-element(p_body), index=0
      index = s32[] get-tuple-element(p_body), index=1
      one = s32[] constant({{STEP}})
      inc = s32[] add(index, one)
      ROOT root = (f32[2], s32[]) tuple(val, inc)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant({{LIMIT}})
      ROOT result = pred[] compare(gte, const), direction={{COMP_DIR}}
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] constant({{INIT}})
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    }
  )";

  std::string hlo_string =
      absl::StrReplaceAll(hlo_string_template,
                          {{"{{INIT}}", absl::StrCat(init)},
                           {"{{LIMIT}}", absl::StrCat(limit)},
                           {"{{STEP}}", absl::StrCat(step)},
                           {"{{COMP_DIR}}", ComparisonDirectionToString(dir)}});

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::optional<int64_t> trip_count = MatchTrivialLoopTripCount(
      while_op, 1,
      Cast<HloConstantInstruction>(
          module->GetComputationWithName("entry")->GetInstructionWithName(
              "param.1"))
          ->literal());

  CHECK(trip_count.has_value());

  return *trip_count;
}

absl::StatusOr<Range> WhileLoopAnalysisTest::MakeWhileLoopAndGetRange(
    int init, int limit, int step, ComparisonDirection dir) {
  std::string hlo_string_template = R"(
  HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val = f32[2] get-tuple-element(p_body), index=0
      index = s32[] get-tuple-element(p_body), index=1
      one = s32[] constant({{STEP}})
      inc = s32[] add(index, one)
      ROOT root = (f32[2], s32[]) tuple(val, inc)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant({{LIMIT}})
      ROOT result = pred[] compare(gte, const), direction={{COMP_DIR}}
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] constant({{INIT}})
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    }
  )";

  std::string hlo_string =
      absl::StrReplaceAll(hlo_string_template,
                          {{"{{INIT}}", absl::StrCat(init)},
                           {"{{LIMIT}}", absl::StrCat(limit)},
                           {"{{STEP}}", absl::StrCat(step)},
                           {"{{COMP_DIR}}", ComparisonDirectionToString(dir)}});

  TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                      ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::optional<Range> range = MatchTrivialLoopRange(while_op);

  CHECK(range.has_value());

  return *range;
}

TEST_F(WhileLoopAnalysisTest, SingleIterationUpperBound) {
  absl::string_view kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val = f32[2] get-tuple-element(p_body), index=0
      const = s32[] constant(-1)
      ROOT root = (f32[2], s32[]) tuple(val, const)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=EQ
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  EXPECT_EQ(*ComputeWhileLoopTripCountUpperBound(while_op), 1);
}

TEST_F(WhileLoopAnalysisTest, SimpleLoopWithCustomCallNonTuple) {
  absl::string_view hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    custom-call.1 = (s32[]{:T(128)}, s32[3]{0}) custom-call(loop_var.1), custom_call_target="CustomCallStart"
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(custom-call.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(custom-call.1), index=1
    output = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    ROOT custom-call.2 = (s32[]{:T(128)}, s32[3]{0}) custom-call(idx, output), custom_call_target="CustomCallEnd"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.5 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.5, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  auto m = ParseAndReturnVerifiedModule(hlo_string).value();
  HloInstruction* while_op = m->entry_computation()->root_instruction();
  EXPECT_EQ(ComputeWhileLoopTripCountUpperBound(while_op), std::nullopt);
}

TEST_F(WhileLoopAnalysisTest, SimpleLoopWithCustomCall) {
  absl::string_view hlo_string = R"(
  HloModule SimpleLoop
  SimpleLoop.body {
    loop_var.1 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    custom-call.1 = (s32[]{:T(128)}, s32[3]{0}) custom-call(loop_var.1), custom_call_target="CustomCallStart"
    get-tuple-element.1 = s32[]{:T(128)} get-tuple-element(custom-call.1), index=0
    constant.1 = s32[]{:T(128)} constant(1)
    idx = s32[]{:T(128)} add(get-tuple-element.1, constant.1)
    get-tuple-element.2 = s32[3]{0} get-tuple-element(custom-call.1), index=1
    output = s32[3]{0} add(get-tuple-element.2, get-tuple-element.2)
    tuple = (s32[]{:T(128)}, s32[3]{0}) tuple(idx, output)
    ROOT custom-call.2 = (s32[]{:T(128)}, s32[3]{0}) custom-call(tuple), custom_call_target="CustomCallEnd"
  }
  SimpleLoop.condition {
    loop_var.2 = (s32[]{:T(128)}, s32[3]{0}) parameter(0)
    get-tuple-element.3 = s32[] get-tuple-element(loop_var.2), index=0
    constant.2 = s32[]{:T(128)} constant(5)
    ROOT less-than = pred[] compare(get-tuple-element.3, constant.2), direction=LT
  }
  ENTRY SimpleLoop {
    constant.3 = s32[]{:T(128)} constant(0)
    constant.4 = s32[3]{0} constant({0, 1, 2})
    tuple.1 = (s32[]{:T(128)}, s32[3]{0}) tuple(constant.3, constant.4)
    ROOT while = (s32[]{:T(128)}, s32[3]{0}) while(tuple.1), condition=
      SimpleLoop.condition, body=SimpleLoop.body
  }
  )";
  auto m = ParseAndReturnVerifiedModule(hlo_string).value();
  HloInstruction* while_op = m->entry_computation()->root_instruction();
  EXPECT_EQ(ComputeWhileLoopTripCountUpperBound(while_op), std::nullopt);
}

TEST_F(WhileLoopAnalysisTest, NoUpperBound) {
  absl::string_view kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val = f32[2] get-tuple-element(p_body), index=0
      const = s32[] constant(42)
      ROOT root = (f32[2], s32[]) tuple(val, const)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=EQ
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  EXPECT_EQ(ComputeWhileLoopTripCountUpperBound(while_op), std::nullopt);
}

int CalculateTripCount(int init, int limit, int step, ComparisonDirection dir) {
  int trip_count = 0;
  if (dir == ComparisonDirection::kLt) {
    for (int i = init; i < limit; i += step) {
      trip_count++;
    }
  } else if (dir == ComparisonDirection::kLe) {
    for (int i = init; i <= limit; i += step) {
      trip_count++;
    }
  } else {
    LOG(FATAL) << "Unknown comparison direction: "
               << ComparisonDirectionToString(dir);
  }
  return trip_count;
}

bool RangeEqualIgnoreBitwidth(const Range& range, int init, int limit,
                              int step) {
  auto range_min = [](const Range& r) {
    return r.min().IsSigned() ? r.min().GetSignedValue()
                              : r.min().GetUnsignedValue();
  };
  auto range_max = [](const Range& r) {
    return r.max()->IsSigned() ? r.max()->GetSignedValue()
                               : r.max()->GetUnsignedValue();
  };
  return range_min(range) == init && range_max(range) == limit &&
         range.step()->GetSignedValue() == step;
}

TEST_F(WhileLoopAnalysisTest, ExactBoundTrivialRange) {
  // LT cases
  EXPECT_TRUE(RangeEqualIgnoreBitwidth(
      MakeWhileLoopAndGetRange(0, 42, 1, ComparisonDirection::kLt).value(), 0,
      41, 1));
  EXPECT_TRUE(RangeEqualIgnoreBitwidth(
      MakeWhileLoopAndGetRange(0, 42, 2, ComparisonDirection::kLt).value(), 0,
      40, 2));
  EXPECT_TRUE(RangeEqualIgnoreBitwidth(
      MakeWhileLoopAndGetRange(0, 42, 5, ComparisonDirection::kLt).value(), 0,
      40, 5));
  EXPECT_TRUE(RangeEqualIgnoreBitwidth(
      MakeWhileLoopAndGetRange(0, 40, 5, ComparisonDirection::kLt).value(), 0,
      35, 5));

  // LE cases
  EXPECT_TRUE(RangeEqualIgnoreBitwidth(
      MakeWhileLoopAndGetRange(0, 42, 1, ComparisonDirection::kLe).value(), 0,
      42, 1));
  EXPECT_TRUE(RangeEqualIgnoreBitwidth(
      MakeWhileLoopAndGetRange(0, 42, 2, ComparisonDirection::kLe).value(), 0,
      42, 2));
  EXPECT_TRUE(RangeEqualIgnoreBitwidth(
      MakeWhileLoopAndGetRange(0, 42, 5, ComparisonDirection::kLe).value(), 0,
      40, 5));
  EXPECT_TRUE(RangeEqualIgnoreBitwidth(
      MakeWhileLoopAndGetRange(0, 40, 5, ComparisonDirection::kLe).value(), 0,
      40, 5));
}

TEST_F(WhileLoopAnalysisTest, ExactBoundTrivialTripCount) {
  // LT cases
  EXPECT_EQ(
      MakeWhileLoopAndGetTripCount(0, 42, 1, ComparisonDirection::kLt).value(),
      CalculateTripCount(0, 42, 1, ComparisonDirection::kLt));
  EXPECT_EQ(
      MakeWhileLoopAndGetTripCount(0, 42, 2, ComparisonDirection::kLt).value(),
      CalculateTripCount(0, 42, 2, ComparisonDirection::kLt));
  EXPECT_EQ(
      MakeWhileLoopAndGetTripCount(0, 42, 5, ComparisonDirection::kLt).value(),
      CalculateTripCount(0, 42, 5, ComparisonDirection::kLt));
  EXPECT_EQ(
      MakeWhileLoopAndGetTripCount(0, 40, 5, ComparisonDirection::kLt).value(),
      CalculateTripCount(0, 40, 5, ComparisonDirection::kLt));

  // LE cases
  EXPECT_EQ(
      MakeWhileLoopAndGetTripCount(0, 42, 1, ComparisonDirection::kLe).value(),
      CalculateTripCount(0, 42, 1, ComparisonDirection::kLe));
  EXPECT_EQ(
      MakeWhileLoopAndGetTripCount(0, 42, 2, ComparisonDirection::kLe).value(),
      CalculateTripCount(0, 42, 2, ComparisonDirection::kLe));
  EXPECT_EQ(
      MakeWhileLoopAndGetTripCount(0, 42, 5, ComparisonDirection::kLe).value(),
      CalculateTripCount(0, 42, 5, ComparisonDirection::kLe));
  EXPECT_EQ(
      MakeWhileLoopAndGetTripCount(0, 40, 5, ComparisonDirection::kLe).value(),
      CalculateTripCount(0, 40, 5, ComparisonDirection::kLe));
}

TEST_F(WhileLoopAnalysisTest, NoAIVNoConstChain) {
  absl::string_view kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[], s32[]) parameter(0)
      val1 = f32[2] get-tuple-element(p_body), index=0
      val2 = s32[] get-tuple-element(p_body), index=1
      val3 = s32[] get-tuple-element(p_body), index=2
      add = s32[] add(val2, val3)
      sub = s32[] subtract(add, val3)
      ROOT root = (f32[2], s32[], s32[]) tuple(val1, add, sub)
    }

    condition {
      p_cond = (f32[2], s32[], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=EQ
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      param.2 = s32[] parameter(2)
      while_init = (f32[2], s32[], s32[]) tuple(param.0, param.1, param.2)
      ROOT while = (f32[2], s32[], s32[]) while(while_init), condition=condition, body=body
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::vector<const HloInstruction*> aux_indices =
      GetAuxiliaryLoopInductionVars(while_op);
  EXPECT_EQ(aux_indices.size(), 0);
}

TEST_F(WhileLoopAnalysisTest, AIVMultiChain) {
  absl::string_view kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val1 = f32[2] get-tuple-element(p_body), index=0
      val2 = s32[] get-tuple-element(p_body), index=1
      const.1 = s32[] constant(42)
      const.2 = s32[] constant(42)
      const.3 = s32[] constant(42)
      add = s32[] add(val2, const.1)
      sub = s32[] subtract(add, const.2)
      mul = s32[] multiply(sub, const.3)
      ROOT root = (f32[2], s32[]) tuple(val1, mul)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=EQ
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::vector<const HloInstruction*> aux_indices =
      GetAuxiliaryLoopInductionVars(while_op);
  EXPECT_EQ(aux_indices.size(), 1);
  EXPECT_EQ(aux_indices[0]->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(aux_indices[0]->tuple_index(), 1);
}

TEST_F(WhileLoopAnalysisTest, NoAIV) {
  absl::string_view kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val1 = f32[2] get-tuple-element(p_body), index=0
      val2 = s32[] get-tuple-element(p_body), index=1
      add = s32[] add(val2, val2)
      const.1 = s32[] constant(42)
      mul = s32[] multiply(add, const.1)
      div = s32[] divide(mul, add)
      ROOT root = (f32[2], s32[]) tuple(val1, div)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=EQ
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::vector<const HloInstruction*> aux_indices =
      GetAuxiliaryLoopInductionVars(while_op);
  EXPECT_EQ(aux_indices.size(), 0);
}

TEST_F(WhileLoopAnalysisTest, AIVNoChain) {
  absl::string_view kHloModule = R"(
    HloModule ModuleWithWhile

    body {
      p_body = (f32[2], s32[]) parameter(0)
      val1 = f32[2] get-tuple-element(p_body), index=0
      val2 = s32[] get-tuple-element(p_body), index=1
      const = s32[] constant(42)
      add = s32[] add(val2, const)
      ROOT root = (f32[2], s32[]) tuple(val1, add)
    }

    condition {
      p_cond = (f32[2], s32[]) parameter(0)
      gte = s32[] get-tuple-element(p_cond), index=1
      const = s32[] constant(42)
      ROOT result = pred[] compare(gte, const), direction=EQ
    }

    ENTRY entry {
      param.0 = f32[2] parameter(0)
      param.1 = s32[] parameter(1)
      while_init = (f32[2], s32[]) tuple(param.0, param.1)
      ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(kHloModule));

  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::vector<const HloInstruction*> aux_indices =
      GetAuxiliaryLoopInductionVars(while_op);
  EXPECT_EQ(aux_indices.size(), 1);
  EXPECT_EQ(aux_indices[0]->opcode(), HloOpcode::kGetTupleElement);
  EXPECT_EQ(aux_indices[0]->tuple_index(), 1);
}

TEST_F(WhileLoopAnalysisTest, NonScalarUpdateOp) {
  absl::string_view hlo = R"(
    HloModule test, replica_count=2
    add {
      param.3 = s32[] parameter(0)
      param.4 = s32[] parameter(1)
      ROOT add = add(param.3, param.4)
    }
    body {
      param.0 = (s32[], s32[]) parameter(0)
      p0.1 = s32[] get-tuple-element(param.0), index=0
      p1.1 = s32[] get-tuple-element(param.0), index=1
      update = s32[] all-reduce(p0.1), replica_groups={{0,1}}, to_apply=add
      ROOT tuple = (s32[], s32[]) tuple(update, p1.1)
    }
    condition {
      param.2 = (s32[], s32[]) parameter(0)
      p0.2 = s32[] get-tuple-element(param.2), index=0
      c4 = s32[] constant(4)
      ROOT compare = pred[] compare(p0.2, c4), direction=LT
    }
    ENTRY entry {
      c0 = s32[] constant(0)
      data = s32[] parameter(0)
      tuple = (s32[], s32[]) tuple(c0, data)
      ROOT while = (s32[], s32[]) while(tuple), body=body, condition=condition
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  const HloInstruction* while_op =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(ComputeWhileLoopTripCount(while_op), std::nullopt);
}

TEST_F(WhileLoopAnalysisTest, UpdateOnIndVarCopySuccess) {
  absl::string_view hlo = R"(
    HloModule test, replica_count=2
    body {
      param.0 = (s32[], s32[]) parameter(0)
      p0.1 = s32[] get-tuple-element(param.0), index=0
      p1.1 = s32[] get-tuple-element(param.0), index=1
      copy0.1 = s32[] copy(p0.1)
      const = s32[] constant(1)
      add = s32[] add(copy0.1, const)
      ROOT tuple = (s32[], s32[]) tuple(add, p1.1)
    }
    condition {
      param.2 = (s32[], s32[]) parameter(0)
      p0.2 = s32[] get-tuple-element(param.2), index=0
      c4 = s32[] constant(4)
      ROOT compare = pred[] compare(p0.2, c4), direction=LT
    }
    ENTRY entry {
      c0 = s32[] constant(0)
      data = s32[] parameter(0)
      tuple = (s32[], s32[]) tuple(c0, data)
      ROOT while = (s32[], s32[]) while(tuple), body=body, condition=condition
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  const HloInstruction* while_op =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(*ComputeWhileLoopTripCount(while_op), 4);
}

TEST_F(WhileLoopAnalysisTest, IndVarInitialiationNotConstantSuccess) {
  absl::string_view hlo = R"(
    HloModule test, replica_count=2
    body {
      param.0 = (s32[], s32[]) parameter(0)
      p0.1 = s32[] get-tuple-element(param.0), index=0
      p1.1 = s32[] get-tuple-element(param.0), index=1
      const = s32[] constant(1)
      add = s32[] add(p0.1, const)
      ROOT tuple = (s32[], s32[]) tuple(add, p1.1)
    }
    condition {
      param.2 = (s32[], s32[]) parameter(0)
      p0.2 = s32[] get-tuple-element(param.2), index=0
      c4 = s32[] constant(4)
      ROOT compare = pred[] compare(p0.2, c4), direction=LT
    }
    ENTRY entry {
      c0 = s32[] constant(0)
      copy0.0 = s32[] copy(c0)
      data = s32[] parameter(0)
      tuple = (s32[], s32[]) tuple(copy0.0, data)
      ROOT while = (s32[], s32[]) while(tuple), body=body, condition=condition
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  const HloInstruction* while_op =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(*ComputeWhileLoopTripCount(while_op), 4);
}

TEST_F(WhileLoopAnalysisTest, FusedUpdateOp) {
  absl::string_view hlo = R"(
  HloModule test, replica_count=2
  add {
    param.3 = s32[] parameter(0)
    param.4 = s32[] parameter(1)
    ROOT add = add(param.3, param.4)
  }
  body {
    param.0 = (s32[], s32[]) parameter(0)
    p0.1 = s32[] get-tuple-element(param.0), index=0
    p1.1 = s32[] get-tuple-element(param.0), index=1
    c1 = s32[] constant(1)
    update = s32[] fusion(p0.1, c1), kind=kInput, calls=add
    ROOT tuple = (s32[], s32[]) tuple(update, p1.1)
  }
  condition {
    param.2 = (s32[], s32[]) parameter(0)
    p0.2 = s32[] get-tuple-element(param.2), index=0
    c4 = s32[] constant(4)
    ROOT compare = pred[] compare(p0.2, c4), direction=LT
  }
  ENTRY entry {
    c0 = s32[] constant(0)
    data = s32[] parameter(0)
    tuple = (s32[], s32[]) tuple(c0, data)
    ROOT while = (s32[], s32[]) while(tuple), body=body, condition=condition
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::optional<int64_t> trip_count = ComputeWhileLoopTripCount(while_op);
  ASSERT_NE(trip_count, std::nullopt);
  EXPECT_EQ(trip_count, 4);
}

TEST_F(WhileLoopAnalysisTest, NonScalarConditionOp) {
  absl::string_view hlo = R"(
    HloModule test, replica_count=2
    add {
      param.3 = s32[] parameter(0)
      param.4 = s32[] parameter(1)
      ROOT add = add(param.3, param.4)
    }
    body {
      param.0 = (s32[], s32[]) parameter(0)
      p0.1 = s32[] get-tuple-element(param.0), index=0
      p1.1 = s32[] get-tuple-element(param.0), index=1
      c1 = s32[] constant(1)
      update = s32[] add(p0.1, c1)
      ROOT tuple = (s32[], s32[]) tuple(update, p1.1)
    }
    fused_computation {
      param.5 = s32[] parameter(0)
      param.6 = s32[] parameter(1)
      all-reduce.1 = s32[] all-reduce(param.5), replica_groups={{0,1}}, to_apply=add
      ROOT compare = pred[] compare(all-reduce.1, param.6), direction=LT
    }
    condition {
      param.2 = (s32[], s32[]) parameter(0)
      p0.2 = s32[] get-tuple-element(param.2), index=0
      c4 = s32[] constant(4)
      ROOT compare = pred[] call(p0.2, c4), to_apply=fused_computation
    }
    ENTRY entry {
      c0 = s32[] constant(0)
      data = s32[] parameter(0)
      tuple = (s32[], s32[]) tuple(c0, data)
      ROOT while = (s32[], s32[]) while(tuple), body=body, condition=condition
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  const HloInstruction* while_op =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(ComputeWhileLoopTripCount(while_op), std::nullopt);
}

TEST_F(WhileLoopAnalysisTest, IndvarWithNonScalarShape) {
  absl::string_view hlo_string = R"(
  HloModule test

  loop.body {
    loop_var.1 = (s32[2]{0:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}) parameter(0)
    get-tuple-element.1 = s32[2]{0:T(128)} get-tuple-element(loop_var.1), index=0
    get-tuple-element.2 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} get-tuple-element(loop_var.1), index=1
    iota = s32[4,3,5]{2,1,0} iota(), iota_dimension=0
    bitcast.12855 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} bitcast(iota)
    add.40974 = s32[1,1,1,4,3,5]{5,4,3,2,1,0} add(get-tuple-element.2, bitcast.12855)
    constant.1 = s32[2]{0:T(128)} constant({1, 1})
    idx = s32[2]{0:T(128)} add(get-tuple-element.1, constant.1)
    ROOT tuple = (s32[2]{0:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}) tuple(idx, add.40974)
  }

  loop.condition {
    loop_var.2 = (s32[2]{0:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}) parameter(0)
    get-tuple-element.3 = s32[2]{0:T(128)} get-tuple-element(loop_var.2), index=0
    slice = s32[1]{0:T(128)} slice(get-tuple-element.3), slice={[0:1]}
    constant.2 = s32[1]{0:T(128)} constant({4})
    less-than = pred[1]{0:T(128)} compare(slice, constant.2), direction=LT
    ROOT bitcast = pred[]{:T(128)} bitcast(less-than)
  }

  ENTRY %main {
    first_idx = s32[2]{0:T(128)} constant({0, 1})
    zeros32 = s32[]{:T(128)} constant(0)
    broadcast = s32[1,1,1,4,3,5]{5,4,3,2,1,0} broadcast(zeros32)
    input = (s32[2]{0:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}) tuple(first_idx, broadcast)
    ROOT while = (s32[2]{0:T(128)}, s32[1,1,1,4,3,5]{5,4,3,2,1,0}) while(input), condition=loop.condition, body=loop.body
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_string));
  const HloInstruction* while_op =
      module->entry_computation()->root_instruction();
  EXPECT_EQ(ComputeWhileLoopTripCount(while_op), std::nullopt);
}

TEST_F(WhileLoopAnalysisTest, FusedConditionOp) {
  const char* hlo = R"(
  HloModule test, replica_count=2
  add {
    param.3 = s32[] parameter(0)
    param.4 = s32[] parameter(1)
    ROOT add = add(param.3, param.4)
  }
  body {
    param.0 = (s32[], s32[]) parameter(0)
    p0.1 = s32[] get-tuple-element(param.0), index=0
    p1.1 = s32[] get-tuple-element(param.0), index=1
    c1 = s32[] constant(1)
    update = s32[] add(p0.1, c1)
    ROOT tuple = (s32[], s32[]) tuple(update, p1.1)
  }
  fused_computation {
    param.5 = s32[] parameter(0)
    param.6 = s32[] parameter(1)
    ROOT compare = pred[] compare(param.5, param.6), direction=LT
  }
  condition {
    param.2 = (s32[], s32[]) parameter(0)
    p0.2 = s32[] get-tuple-element(param.2), index=0
    c4 = s32[] constant(4)
    ROOT compare = pred[] fusion(p0.2, c4), kind=kInput, calls=fused_computation
  }
  ENTRY entry {
    c0 = s32[] constant(0)
    data = s32[] parameter(0)
    tuple = (s32[], s32[]) tuple(c0, data)
    ROOT while = (s32[], s32[]) while(tuple), body=body, condition=condition
  }
  )";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::optional<int64_t> trip_count = ComputeWhileLoopTripCount(while_op);
  ASSERT_NE(trip_count, std::nullopt);
  EXPECT_EQ(trip_count, 4);
}

TEST_F(WhileLoopAnalysisTest, AvoidBruteForceForHugeParams) {
  absl::string_view hlo = R"(
  HloModule test
  fused_comp {
    p.0 = pred[100000000]{0} parameter(0)
    p.1 = s32[] parameter(1)
    dynamic-slice = pred[1]{0} constant({false})
    ROOT dus = pred[100000000]{0} dynamic-update-slice(p.0, dynamic-slice, p.1)
  }
  body {
    param.2 = (pred[100000000], s32[]) parameter(0)
    gte.1 = pred[100000000] get-tuple-element(param.2), index=0
    iter = s32[] get-tuple-element(param.2), index=1
    fusion = pred[100000000] call(gte.1,iter), to_apply=fused_comp
    c.1 = s32[] constant(1)
    add = s32[] add(iter, c.1)
    ROOT tuple = tuple(fusion, add)
  }
  or {
    x = pred[] parameter(0)
    y = pred[] parameter(1)
    ROOT res = pred[] or(x, y)
  }
  condition {
    param.1 = (pred[100000000], s32[]) parameter(0)
    gte = pred[100000000] get-tuple-element(param.1), index=0
    false.1 = pred[] constant(false)
    ROOT any = pred[] reduce(gte, false.1), dimensions={0}, to_apply=or
  }

  ENTRY main {
    true.1 = pred[] constant(true)
    param.0 = pred[100000000] broadcast(true.1)
    c.0 = s32[] constant(0)
    tuple = tuple(param.0, c.0)
    while = while(tuple), body=body, condition=condition
    ROOT iter = s32[] get-tuple-element(while), index=1
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo));
  const HloInstruction* while_op =
      module->entry_computation()->root_instruction()->operand(0);
  std::optional<int64_t> trip_count = ComputeWhileLoopTripCount(while_op);
  EXPECT_EQ(trip_count, std::nullopt);
}

TEST_F(WhileLoopAnalysisTest, LoopFusionForLoopVariable) {
  // This test verifies that fusions in initialization, condition and update are
  // accepted by while loop analysis.
  absl::string_view hlo = R"(
  HloModule test
  fused_add.11 {
    param_0.968 = s32[] parameter(0)
    constant_1239_1 = s32[] constant(1)
    ROOT add.1041.1 = s32[] add(param_0.968, constant_1239_1)
  }
  fused_add.11.clone.2 {
    param_0.2169 = s32[] parameter(0)
    constant_1239_4 = s32[] constant(1)
    ROOT add.1041.4 = s32[] add(param_0.2169, constant_1239_4)
  }
  body {
    param.1 = (s32[], s32[]) parameter(0)
    loop_iter = s32[] get-tuple-element(param.1), index=0
    data = s32[] get-tuple-element(param.1), index=1
    loop_add_fusion.11 = s32[] fusion(loop_iter), kind=kLoop, calls=fused_add.11
    loop_add_fusion.11.double_buffer_clone = s32[] fusion(loop_add_fusion.11), kind=kLoop, calls=fused_add.11.clone.2
    ROOT tuple = (s32[], s32[]) tuple(loop_add_fusion.11.double_buffer_clone, data)
  }
  fused_compare {
    param_0.987 = s32[] parameter(0)
    constant_1238_1 = s32[] constant(7)
    ROOT compare.98.1 = pred[] compare(param_0.987, constant_1238_1), direction=LT
  }
  condition {
    param.2 = (s32[], s32[]) parameter(0)
    loop_iter = s32[] get-tuple-element(param.2), index=0
    ROOT loop_compare_fusion = pred[] fusion(loop_iter), kind=kLoop, calls=fused_compare
  }
  fused_add.12 {
    param_0.968 = s32[] parameter(0)
    constant_1239_1 = s32[] constant(1)
    ROOT add.1041.1 = s32[] add(param_0.968, constant_1239_1)
  }
  ENTRY main {
    data = s32[] parameter(0)
    c.0 = s32[] constant(0)
    c.1 = s32[] constant(1)
    add.1 = s32[] add(c.0, c.1)
    c.0.loop_double_buffer_peeled = s32[] fusion(add.1), kind=kLoop, calls=fused_add.12
    tuple = (s32[], s32[]) tuple(c.0.loop_double_buffer_peeled, data)
    ROOT while = while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* while_op = module->entry_computation()->root_instruction();
  auto loop_induction_variable = GetLoopInductionVarTupleIdx(while_op);
  ASSERT_TRUE(loop_induction_variable.has_value());
  EXPECT_EQ(loop_induction_variable.value(), 0);
}

TEST_F(WhileLoopAnalysisTest, UpdateIsMultipleOperationsWithConstantOperand) {
  absl::string_view hlo = R"(
  HloModule test
  body {
    param.1 = (s32[], s32[8,8]) parameter(0)
    iter.1 = s32[] get-tuple-element(param.1), index=0
    c.1 = s32[] constant(1)
    add.1 = s32[] add(iter.1, c.1)
    add.2 = s32[] add(add.1, c.1)
    data.1 = s32[8,8] get-tuple-element(param.1), index=1
    ROOT tuple = (s32[], s32[8,8]) tuple(add.2, data.1)
  }
  condition {
    param = (s32[], s32[8,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c.10 = s32[] constant(10)
    ROOT compare = pred[] compare(iter, c.10), direction=LT
  }
  ENTRY main {
    c.0 = s32[] constant(0)
    data = s32[8,8] parameter(0)
    tuple = tuple(c.0, data)
    ROOT while = while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::optional<int64_t> indvar_idx = GetLoopInductionVarTupleIdx(while_op);
  ASSERT_NE(indvar_idx, std::nullopt);
  EXPECT_EQ(*indvar_idx, 0);
  std::optional<int64_t> trip_count = ComputeWhileLoopTripCount(while_op);
  EXPECT_EQ(trip_count, std::nullopt);
}

TEST_F(WhileLoopAnalysisTest,
       UpdateIsMultipleOperationsWithoutConstantOperand) {
  absl::string_view hlo = R"(
  HloModule test
  body {
    param.1 = (s32[], s32[8,8]) parameter(0)
    iter.1 = s32[] get-tuple-element(param.1), index=0
    c.1 = s32[] constant(1)
    add.1 = s32[] add(c.1, c.1)
    add.2 = s32[] add(iter.1, add.1)
    data.1 = s32[8,8] get-tuple-element(param.1), index=1
    ROOT tuple = (s32[], s32[8,8]) tuple(add.2, data.1)
  }
  condition {
    param = (s32[], s32[8,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c.10 = s32[] constant(10)
    ROOT compare = pred[] compare(iter, c.10), direction=LT
  }
  ENTRY main {
    c.0 = s32[] constant(0)
    data = s32[8,8] parameter(0)
    tuple = tuple(c.0, data)
    ROOT while = while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::optional<int64_t> indvar_idx = GetLoopInductionVarTupleIdx(while_op);
  ASSERT_NE(indvar_idx, std::nullopt);
  EXPECT_EQ(*indvar_idx, 0);
  std::optional<int64_t> trip_count = ComputeWhileLoopTripCount(while_op);
  EXPECT_EQ(trip_count, std::nullopt);
}

TEST_F(WhileLoopAnalysisTest,
       ConditionIsMultipleOperationsWithConstantOperand) {
  absl::string_view hlo = R"(
  HloModule test
  body {
    param.1 = (s32[], s32[8,8]) parameter(0)
    iter.1 = s32[] get-tuple-element(param.1), index=0
    c.1 = s32[] constant(1)
    add.1 = s32[] add(iter.1, c.1)
    data.1 = s32[8,8] get-tuple-element(param.1), index=1
    ROOT tuple = (s32[], s32[8,8]) tuple(add.1, data.1)
  }
  condition {
    param = (s32[], s32[8,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c.10 = s32[] constant(10)
    add.10 = s32[] add(iter, c.10)
    ROOT compare = pred[] compare(add.10, c.10), direction=LT
  }
  ENTRY main {
    c.0 = s32[] constant(0)
    data = s32[8,8] parameter(0)
    tuple = tuple(c.0, data)
    ROOT while = while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::optional<int64_t> indvar_idx = GetLoopInductionVarTupleIdx(while_op);
  ASSERT_NE(indvar_idx, std::nullopt);
  EXPECT_EQ(*indvar_idx, 0);
  std::optional<int64_t> trip_count = ComputeWhileLoopTripCount(while_op);
  EXPECT_EQ(trip_count, std::nullopt);
}

TEST_F(WhileLoopAnalysisTest,
       ConditionIsMultipleOperationsWithoutConstantOperand) {
  absl::string_view hlo = R"(
  HloModule test
  body {
    param.1 = (s32[], s32[8,8]) parameter(0)
    iter.1 = s32[] get-tuple-element(param.1), index=0
    c.1 = s32[] constant(1)
    add.1 = s32[] add(iter.1, c.1)
    data.1 = s32[8,8] get-tuple-element(param.1), index=1
    ROOT tuple = (s32[], s32[8,8]) tuple(add.1, data.1)
  }
  condition {
    param = (s32[], s32[8,8]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c.5 = s32[] constant(5)
    add.10 = s32[] add(c.5, c.5)
    ROOT compare = pred[] compare(iter, add.10), direction=LT
  }
  ENTRY main {
    c.0 = s32[] constant(0)
    data = s32[8,8] parameter(0)
    tuple = tuple(c.0, data)
    ROOT while = while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(hlo));
  HloInstruction* while_op = module->entry_computation()->root_instruction();
  std::optional<int64_t> indvar_idx = GetLoopInductionVarTupleIdx(while_op);
  ASSERT_NE(indvar_idx, std::nullopt);
  EXPECT_EQ(*indvar_idx, 0);
  std::optional<int64_t> trip_count = ComputeWhileLoopTripCount(while_op);
  EXPECT_EQ(trip_count, std::nullopt);
}

TEST_F(WhileLoopAnalysisTest, GetIndvarIndexShouldWorkWhenParamIsCopied) {
  absl::string_view hlo = R"(
    HloModule test

    fused_copy {
      param.1 = (s32[],s32[]) parameter(0)
      ROOT copy = (s32[], s32[]) copy(param.1)
    }

    body {
      param.1 = (s32[], s32[]) parameter(0)
      copy_fusion = (s32[], s32[]) fusion(param.1), kind=kInput, calls=fused_copy
      iter.1 = s32[] get-tuple-element(copy_fusion), index=0
      c.1 = s32[] constant(1)
      add.1 = s32[] add(iter.1, c.1)
      data.1 = s32[] get-tuple-element(copy_fusion), index=1
      ROOT tuple = (s32[], s32[]) tuple(add.1, data.1)
    }

    condition {
      param = (s32[], s32[]) parameter(0)
      iter = s32[] get-tuple-element(param), index=0
      c.10 = s32[] constant(10)
      ROOT compare = pred[] compare(iter, c.10), direction=LT
    }

    ENTRY main {
      c0 = s32[] constant(0)
      data = s32[] parameter(0)
      tuple = (s32[], s32[]) tuple(c0, data)
      ROOT while = (s32[], s32[]) while(tuple), body=body, condition=condition
    }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo));
  HloInstruction* while_op = m->entry_computation()->root_instruction();
  ASSERT_EQ(while_op->opcode(), HloOpcode::kWhile);
  EXPECT_EQ(GetLoopInductionVarTupleIdx(while_op), 0);
}

TEST_F(WhileLoopAnalysisTest,
       MatchTrivialLoopCountFailsWhenIndvarIsNotIncrementedByConstant) {
  absl::string_view hlo_with_constant = R"(
  HloModule test
  body {
    param.1 = (s32[], s32[]) parameter(0)
    iter.1 = s32[] get-tuple-element(param.1), index=0
    data.1 = s32[] get-tuple-element(param.1), index=1
    c.1 = s32[] constant(1)
    add.1 = s32[] add(iter.1, c.1)
    ROOT tuple = (s32[], s32[]) tuple(add.1, data.1)
  }
  condition {
    param = (s32[], s32[]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c.10 = s32[] constant(10)
    ROOT compare = pred[] compare(iter, c.10), direction=LT
  }
  ENTRY main {
    c0 = s32[] constant(0)
    data = s32[] parameter(0)
    tuple = (s32[], s32[]) tuple(c0, data)
    ROOT while = (s32[], s32[]) while(tuple), body=body, condition=condition
  })";
  absl::string_view hlo_without_constant = R"(
  HloModule test
  body {
    param.1 = (s32[], s32[]) parameter(0)
    iter.1 = s32[] get-tuple-element(param.1), index=0
    data.1 = s32[] get-tuple-element(param.1), index=1
    add.1 = s32[] add(iter.1, iter.1)
    ROOT tuple = (s32[], s32[]) tuple(add.1, data.1)
  }
  condition {
    param = (s32[], s32[]) parameter(0)
    iter = s32[] get-tuple-element(param), index=0
    c.10 = s32[] constant(10)
    ROOT compare = pred[] compare(iter, c.10), direction=LT
  }
  ENTRY main {
    c1 = s32[] constant(1)
    data = s32[] parameter(0)
    tuple = (s32[], s32[]) tuple(c1, data)
    ROOT while = (s32[], s32[]) while(tuple), body=body, condition=condition
  })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_with_constant,
                          ParseAndReturnVerifiedModule(hlo_with_constant));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m_without_constant,
                          ParseAndReturnVerifiedModule(hlo_without_constant));
  HloInstruction* while_op_with_constant =
      m_with_constant->entry_computation()->root_instruction();
  HloInstruction* while_op_without_constant =
      m_without_constant->entry_computation()->root_instruction();
  std::optional<int64_t> trip_count_with_constant = MatchTrivialLoopTripCount(
      while_op_with_constant, 0, LiteralUtil::CreateR0<int32_t>(0));
  EXPECT_EQ(trip_count_with_constant, 10);
  std::optional<int64_t> trip_count_without_constant =
      MatchTrivialLoopTripCount(while_op_without_constant, 0,
                                LiteralUtil::CreateR0<int32_t>(0));
  EXPECT_EQ(trip_count_without_constant, std::nullopt);
}

TEST_F(WhileLoopAnalysisTest,
       MatchTrivialLoopCountWithSimpleArithmeticOnIndvar) {
  absl::string_view hlo_string = R"(
HloModule ModuleWithWhile
body {
  p_body = (f32[2], s32[]) parameter(0)
  val = f32[2] get-tuple-element(p_body), index=0
  index = s32[] get-tuple-element(p_body), index=1
  one = s32[] constant(1)
  inc = s32[] add(index, one)
  ROOT root = (f32[2], s32[]) tuple(val, inc)
}
condition {
  p_cond = (f32[2], s32[]) parameter(0)
  gte = s32[] get-tuple-element(p_cond), index=1
  const = s32[] constant(10)
  ROOT result = pred[] compare(gte, const), direction=LE
}
ENTRY entry {
  param.0 = f32[2] parameter(0)
  const.0 = s32[] constant(1)
  const.1 = s32[] constant(1)
  add.0 = s32[] add(const.0, const.1)
  while_init = (f32[2], s32[]) tuple(param.0, add.0)
  ROOT while = (f32[2], s32[]) while(while_init), condition=condition, body=body
}
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> m,
                          ParseAndReturnVerifiedModule(hlo_string));

  HloInstruction* while_op = m->entry_computation()->root_instruction();
  std::optional<Range> range = MatchTrivialLoopRange(while_op);

  EXPECT_TRUE(range.has_value());
  EXPECT_EQ(range->min().GetSignedValue(), 2);
  EXPECT_TRUE(range->max().has_value());
  EXPECT_EQ(range->max().value().GetSignedValue(), 10);
  EXPECT_TRUE(range->step().has_value());
  EXPECT_EQ(range->step().value().GetSignedValue(), 1);
}
}  // namespace
}  // namespace xla
