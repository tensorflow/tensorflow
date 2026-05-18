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

#include "xla/hlo/transforms/simplifiers/alias_anti_dependency_inserter.h"

#include <memory>

#include "absl/algorithm/container.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/literal_util.h"
#include "xla/shape_util.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

using AliasAntiDependencyInserterTest = HloHardwareIndependentTestBase;

TEST_F(AliasAntiDependencyInserterTest, NoOpOnIndependentNodes) {
  const Shape vec = ShapeUtil::MakeShape(xla::F32, {4});
  auto builder = HloComputation::Builder(TestName());
  auto p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec, "p0"));
  auto p1 =
      builder.AddInstruction(HloInstruction::CreateParameter(1, vec, "p1"));
  builder.AddInstruction(
      HloInstruction::CreateBinary(vec, HloOpcode::kAdd, p0, p1));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  AliasAntiDependencyInserter inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&inserter, module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(AliasAntiDependencyInserterTest,
       InsertsControlDependencyOnMutatingWriter) {
  const Shape vec = ShapeUtil::MakeShape(xla::F32, {4});
  const Shape svec = ShapeUtil::MakeShape(xla::F32, {1});

  auto builder = HloComputation::Builder(TestName());
  auto p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec, "p0"));
  auto update = builder.AddInstruction(
      HloInstruction::CreateParameter(1, svec, "update"));
  auto start_indices = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0)));

  // Reader of p0
  auto reader = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kNegate, p0));

  // In-place writer of p0 (modifies p0 in place)
  auto writer = builder.AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
      vec, p0, update, {start_indices}));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  AliasAntiDependencyInserter inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&inserter, module.get()));
  EXPECT_TRUE(changed);

  // Assert that a control dependency from reader to writer was added!
  EXPECT_TRUE(absl::c_linear_search(reader->control_successors(), writer));
  EXPECT_TRUE(absl::c_linear_search(writer->control_predecessors(), reader));

  // Clean up
  TF_ASSERT_OK(inserter.RemoveAddedControlDependencies());
  EXPECT_FALSE(absl::c_linear_search(reader->control_successors(), writer));
  EXPECT_FALSE(absl::c_linear_search(writer->control_predecessors(), reader));
}

TEST_F(AliasAntiDependencyInserterTest,
       NoControlDependencyOnNonMutatingWriter) {
  const Shape vec = ShapeUtil::MakeShape(xla::F32, {4});

  auto builder = HloComputation::Builder(TestName());
  auto p0 =
      builder.AddInstruction(HloInstruction::CreateParameter(0, vec, "p0"));

  // Reader of p0
  auto reader = builder.AddInstruction(
      HloInstruction::CreateUnary(vec, HloOpcode::kNegate, p0));

  // Non-mutating alias of p0
  auto bitcast = builder.AddInstruction(HloInstruction::CreateBitcast(vec, p0));

  auto module = CreateNewVerifiedModule();
  module->AddEntryComputation(builder.Build());

  AliasAntiDependencyInserter inserter;
  TF_ASSERT_OK_AND_ASSIGN(bool changed, RunHloPass(&inserter, module.get()));
  EXPECT_FALSE(changed);

  (void)reader;
  (void)bitcast;
}

}  // namespace
}  // namespace xla
