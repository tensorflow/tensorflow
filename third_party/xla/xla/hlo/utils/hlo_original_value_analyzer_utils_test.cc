/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/test.h"
#include "xla/shape_util.h"

namespace xla {
namespace {

TEST(ScopeInstructionTest, ToString) {
  EXPECT_EQ(ScopeInstruction::Create("loop", 0).ToString(), "loop");
  EXPECT_EQ(ScopeInstruction::Create("loop", 3).ToString(), "loop#3");
  EXPECT_EQ(ScopeInstruction::Create("loop", -1).ToString(), "loop#*");
  EXPECT_EQ(ScopeInstruction::Create("loop", -2).ToString(), "loop#$");
}

TEST(ScopeInstructionTest, FromString) {
  auto s1 = ScopeInstruction::FromString("loop");
  EXPECT_EQ(s1.instruction_name, "loop");
  EXPECT_EQ(s1.iteration_index, 0);

  auto s2 = ScopeInstruction::FromString("loop#3");
  EXPECT_EQ(s2.instruction_name, "loop");
  EXPECT_EQ(s2.iteration_index, 3);

  auto s3 = ScopeInstruction::FromString("loop#*");
  EXPECT_EQ(s3.instruction_name, "loop");
  EXPECT_EQ(s3.iteration_index, -1);

  auto s4 = ScopeInstruction::FromString("loop#$");
  EXPECT_EQ(s4.instruction_name, "loop");
  EXPECT_EQ(s4.iteration_index, -2);

  EXPECT_DEATH(ScopeInstruction::FromString("loop#abc"),
               "Failed to parse scope instruction");
}

TEST(TensorKeyTest, ToString) {
  EXPECT_EQ(TensorKey::Create("res", {}).ToString(), "res ({})");
  EXPECT_EQ(TensorKey::Create("res", {1, 0}).ToString(), "res ({1,0})");
}

TEST(RelativeScopedTensorKeyTest, FromStringAndToString) {
  auto r1 = RelativeScopedTensorKey::FromString("res", {0});
  EXPECT_EQ(r1.tensor_key.instruction_name, "res");
  EXPECT_TRUE(r1.scope_instructions.empty());
  EXPECT_EQ(r1.ToString(), "res ({0})");

  auto r2 = RelativeScopedTensorKey::FromString("loop#3/res", {1});
  ASSERT_EQ(r2.scope_instructions.size(), 1);
  EXPECT_EQ(r2.scope_instructions[0].instruction_name, "loop");
  EXPECT_EQ(r2.scope_instructions[0].iteration_index, 3);
  EXPECT_EQ(r2.tensor_key.instruction_name, "res");
  EXPECT_EQ(r2.ToString(), "loop#3/res ({1})");

  auto r3 = RelativeScopedTensorKey::FromString("a/b/c", {});
  ASSERT_EQ(r3.scope_instructions.size(), 2);
  EXPECT_EQ(r3.ToString(), "a/b/c ({})");
}

TEST(AbsoluteScopedTensorKeyTest, Create) {
  auto rel = RelativeScopedTensorKey::FromString("sub/res", {0});
  // sub is NOT in call_map, so it's treated as a normal scope.
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;

  auto abs = AbsoluteScopedTensorKey::Create(
      {ScopeInstruction::Create("main", 0)}, rel, call_map);
  ASSERT_EQ(abs.scope_instructions.size(), 2);
  EXPECT_EQ(abs.scope_instructions[0].instruction_name, "main");
  EXPECT_EQ(abs.scope_instructions[1].instruction_name, "sub");
  EXPECT_EQ(abs.tensor_key.instruction_name, "res");
}

TEST(AbsoluteScopedTensorKeyTest, CreateWithCallMap) {
  auto rel = RelativeScopedTensorKey::FromString("call/res", {0});
  // 'call' is in call_map, so it's expanded.
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  call_map["call"] = {ScopeInstruction::Create("expanded", 1)};

  auto abs = AbsoluteScopedTensorKey::Create(
      {ScopeInstruction::Create("main", 0)}, rel, call_map);
  ASSERT_EQ(abs.scope_instructions.size(), 2);
  EXPECT_EQ(abs.scope_instructions[0].instruction_name, "main");
  EXPECT_EQ(abs.scope_instructions[1].instruction_name, "expanded");
  EXPECT_EQ(abs.scope_instructions[1].iteration_index, 1);
  EXPECT_EQ(abs.tensor_key.instruction_name, "res");
}

TEST(AbsoluteScopedTensorKeyTest, CreateWithWildcardReplacementSize1) {
  auto rel = RelativeScopedTensorKey::FromString("res", {0});
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  call_map["call"] = {ScopeInstruction::Create("expanded", -2)};

  auto abs = AbsoluteScopedTensorKey::Create(
      {ScopeInstruction::Create("call", 5)}, rel, call_map);
  ASSERT_EQ(abs.scope_instructions.size(), 1);
  EXPECT_EQ(abs.scope_instructions[0].instruction_name, "expanded");
  EXPECT_EQ(abs.scope_instructions[0].iteration_index, 5);
}

TEST(AbsoluteScopedTensorKeyTest, CreateWithWildcardReplacementSizeN) {
  auto rel = RelativeScopedTensorKey::FromString("res", {0});
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  call_map["call"] = {ScopeInstruction::Create("expanded1", 0),
                      ScopeInstruction::Create("expanded2", -2)};

  auto abs = AbsoluteScopedTensorKey::Create(
      {ScopeInstruction::Create("call", 5)}, rel, call_map);
  ASSERT_EQ(abs.scope_instructions.size(), 2);
  EXPECT_EQ(abs.scope_instructions[0].instruction_name, "expanded1");
  EXPECT_EQ(abs.scope_instructions[0].iteration_index, 0);
  EXPECT_EQ(abs.scope_instructions[1].instruction_name, "expanded2");
  EXPECT_EQ(abs.scope_instructions[1].iteration_index, 5);
}

TEST(AbsoluteScopedTensorKeyTest, CreatePreservesSpecificIndex) {
  auto rel = RelativeScopedTensorKey::FromString("call/res", {0});
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  call_map["call"] = {ScopeInstruction::Create("expanded", 3)};

  auto abs = AbsoluteScopedTensorKey::Create(
      {ScopeInstruction::Create("main", 5)}, rel, call_map);
  ASSERT_EQ(abs.scope_instructions.size(), 2);
  EXPECT_EQ(abs.scope_instructions[0].instruction_name, "main");
  EXPECT_EQ(abs.scope_instructions[1].instruction_name, "expanded");
  EXPECT_EQ(abs.scope_instructions[1].iteration_index, 3);
}

TEST(AbsoluteScopedTensorKeyTest, CreatePreservesWildcard) {
  auto rel = RelativeScopedTensorKey::FromString("call/res", {0});
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  call_map["call"] = {ScopeInstruction::Create("expanded", -1)};

  auto abs = AbsoluteScopedTensorKey::Create(
      {ScopeInstruction::Create("main", 5)}, rel, call_map);
  ASSERT_EQ(abs.scope_instructions.size(), 2);
  EXPECT_EQ(abs.scope_instructions[0].instruction_name, "main");
  EXPECT_EQ(abs.scope_instructions[1].instruction_name, "expanded");
  EXPECT_EQ(abs.scope_instructions[1].iteration_index, -1);
}

TEST(HloOriginalValueAnalyzerUtilsTest, GetShardingFromUnshardRecoveryModule) {
  auto module_or_status = ParseAndReturnUnverifiedModule(R"(
HloModule recovery_module

ENTRY main {
  param = f32[2,2] parameter(0), sharding={replicated}
  ROOT copy = f32[2,2] copy(param)
}
)");
  ASSERT_TRUE(module_or_status.ok());
  auto module = std::move(module_or_status).value();

  auto sharding = GetShardingFromUnshardRecoveryModule(*module);
  ASSERT_TRUE(sharding.has_value());
  EXPECT_TRUE(sharding->IsReplicated());
}

}  // namespace
}  // namespace xla
