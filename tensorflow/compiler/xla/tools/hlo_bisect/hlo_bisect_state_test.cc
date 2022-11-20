/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/tools/hlo_bisect/hlo_bisect_state.h"

#include <initializer_list>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/service/pattern_matcher_gmock.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace bisect {
namespace {

namespace m = match;

using HloBisectStateTest = HloTestBase;

// Simple test bug checker, verifies the presence of the given instructions in
// the entry computation.
class TestBugSearch : public BugCheckerInterface {
 public:
  TestBugSearch(std::initializer_list<HloOpcode> opcodes) : opcodes_(opcodes) {}

  StatusOr<bool> Run(const HloModule& module) override {
    auto has_opcode = [&](HloOpcode opcode) {
      return absl::c_any_of(module.entry_computation()->instructions(),
                            [opcode](const HloInstruction* instr) {
                              return instr->opcode() == opcode;
                            });
    };
    return absl::c_all_of(opcodes_, has_opcode);
  }

  absl::flat_hash_map<std::string, Literal> GetResults() override { return {}; }

 private:
  std::vector<HloOpcode> opcodes_;
};

Literal CreateLiteral(float value) {
  Literal result = Literal::CreateFromShape(ShapeUtil::MakeShape(F32, {}));
  result.PopulateWithValue(value);
  return result;
}

TEST_F(HloBisectStateTest, TrimByOutputs) {
  const char* kModuleStr = R"(
    HloModule test_module
    ENTRY test_computation {
      p1 = s32[8] parameter(0)
      p2 = s32[8] parameter(1)
      a = s32[8] add(p1, p2)
      b = s32[8] multiply(p1, p2)
      c = s32[8] subtract(p1, p2)
      ROOT sum = tuple(a, b, c)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  TestBugSearch bug_checker({HloOpcode::kMultiply});
  HloBisectState bisect(std::move(module), &bug_checker);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, bisect.TrimEntryComputation());
  EXPECT_TRUE(changed);
  auto reduced_module = std::move(bisect).GetResult();
  EXPECT_THAT(reduced_module->entry_computation()->root_instruction(),
              GmockMatch(m::Multiply(m::Parameter(0), m::Parameter(1))));
}

TEST_F(HloBisectStateTest, TrimByInstructions) {
  const char* kModuleStr = R"(
    HloModule axpy_module
    ENTRY axpy_computation {
      alpha = f32[] parameter(0)
      broadcast = f32[10] broadcast(alpha), dimensions={}
      x = f32[10] parameter(1)
      ax = f32[10] multiply(broadcast, x)
      y = f32[10] parameter(2)
      ROOT add = f32[10] add(ax, y)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  TestBugSearch bug_checker({HloOpcode::kMultiply, HloOpcode::kBroadcast});
  HloBisectState bisect(std::move(module), &bug_checker);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, bisect.TrimEntryComputation());
  EXPECT_TRUE(changed);
  auto reduced_module = std::move(bisect).GetResult();
  EXPECT_THAT(
      reduced_module->entry_computation()->root_instruction(),
      GmockMatch(m::Multiply(m::Broadcast(m::Parameter(0)), m::Parameter(1))));
}

TEST_F(HloBisectStateTest, TrimByUsingRandomConstants) {
  const char* kModuleStr = R"(
    HloModule test_module
    ENTRY test_computation {
      p1 = f32[4] parameter(0)
      p2 = f32[4] parameter(1)
      a = f32[4] multiply(p1, p2)
      b = f32[4] add(p1, p2)
      ROOT result = f32[4] power(a, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  TestBugSearch bug_checker({HloOpcode::kPower});
  HloBisectState bisect(std::move(module), &bug_checker);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, bisect.TrimEntryComputation());
  EXPECT_TRUE(changed);
  auto reduced_module = std::move(bisect).GetResult();
  EXPECT_THAT(reduced_module->entry_computation()->root_instruction(),
              GmockMatch(m::Power(m::Constant(), m::Constant())));
}

TEST_F(HloBisectStateTest, TrimByUsingReferenceConstants) {
  class TestBugSearchWithReferenceConstants : public TestBugSearch {
   public:
    TestBugSearchWithReferenceConstants()
        : TestBugSearch({HloOpcode::kPower}) {}

    absl::flat_hash_map<std::string, Literal> GetResults() override {
      absl::flat_hash_map<std::string, Literal> results;
      results["a"] = CreateLiteral(2.0f);
      results["b"] = CreateLiteral(3.0f);
      return results;
    }
  };

  const char* kModuleStr = R"(
    HloModule test_module
    ENTRY test_computation {
      p1 = f32[] parameter(0)
      p2 = f32[] parameter(1)
      a = f32[] multiply(p1, p2)
      b = f32[] add(p1, p2)
      ROOT result = f32[] power(a, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  TestBugSearchWithReferenceConstants bug_checker;
  HloBisectState bisect(std::move(module), &bug_checker);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, bisect.TrimEntryComputation());
  EXPECT_TRUE(changed);
  auto reduced_module = std::move(bisect).GetResult();
  EXPECT_THAT(reduced_module->entry_computation()->root_instruction(),
              GmockMatch(m::Power(m::Constant(), m::Constant())));
}

TEST_F(HloBisectStateTest, TrimByOutputsLostBug) {
  class CustomBugSearch : public TestBugSearch {
   public:
    CustomBugSearch() : TestBugSearch({HloOpcode::kConstant}) {}
    StatusOr<bool> Run(const HloModule& module) override {
      TF_ASSIGN_OR_RETURN(bool has_constants, TestBugSearch::Run(module));
      int program_size = module.entry_computation()->instruction_count();
      return program_size == 5 && !has_constants;
    }
  };
  const char* kModuleStr = R"(
    HloModule test_module
    ENTRY test_computation {
      p1 = s32[8] parameter(0)
      p2 = s32[8] parameter(1)
      a = s32[8] add(p1, p2)
      b = s32[8] multiply(p1, p2)
      ROOT sum = tuple(a, b)
    }
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(kModuleStr));
  CustomBugSearch bug_checker;
  HloBisectState bisect(std::move(module), &bug_checker);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, bisect.TrimEntryComputation());
  EXPECT_FALSE(changed);
}

}  // namespace
}  // namespace bisect
}  // namespace xla
