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

#include "xla/hlo/transforms/simplifiers/instruction_hoister.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/service/hlo_verifier.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace {

class InstructionHoisterTest : public HloHardwareIndependentTestBase {
 protected:
  void VerifySchedule(HloModule* module) {
    HloVerifier verifier(/*layout_sensitive=*/false,
                         /*allow_mixed_precision=*/true);
    TF_ASSERT_OK(verifier.Run(module).status());
  }
};

TEST_F(InstructionHoisterTest, HoistBitcasts) {
  const absl::string_view hlo_string = R"hlo(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[10] parameter(0)
  neg = f32[10] negate(p0)
  add = f32[10] add(neg, p0)
  bitcast = f32[10] bitcast(neg)
  ROOT root = f32[10] multiply(add, bitcast)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionHoister hoister(/*hoist_parameters=*/false,
                             /*host_constants=*/false,
                             /*hoist_bitcasts=*/true,
                             /*hoist_gtes=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);
  VerifySchedule(module.get());

  const HloInstructionSequence& sequence =
      module->schedule().sequence(module->entry_computation());

  // The expected sequence: p0, neg, bitcast, add, root
  // 'bitcast' should be hoisted to immediately follow its producer 'neg'.
  EXPECT_EQ(sequence.instructions()[1]->opcode(), HloOpcode::kNegate);
  EXPECT_EQ(sequence.instructions()[2]->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(sequence.instructions()[2]->operand(0), sequence.instructions()[1]);
}

TEST_F(InstructionHoisterTest, HoistBitcastChains) {
  const absl::string_view hlo_string = R"hlo(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[10] parameter(0)
  neg = f32[10] negate(p0)
  add = f32[10] add(neg, p0)
  bitcast1 = f32[10] bitcast(neg)
  bitcast2 = f32[10] bitcast(bitcast1)
  ROOT root = f32[10] multiply(add, bitcast2)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionHoister hoister(/*hoist_parameters=*/false,
                             /*host_constants=*/false,
                             /*hoist_bitcasts=*/true,
                             /*hoist_gtes=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);
  VerifySchedule(module.get());

  const HloInstructionSequence& sequence =
      module->schedule().sequence(module->entry_computation());

  // The expected sequence: p0, neg, bitcast1, bitcast2, add, root
  EXPECT_EQ(sequence.instructions()[1]->opcode(), HloOpcode::kNegate);
  EXPECT_EQ(sequence.instructions()[2]->opcode(), HloOpcode::kBitcast);
  EXPECT_EQ(sequence.instructions()[3]->opcode(), HloOpcode::kBitcast);
}

TEST_F(InstructionHoisterTest, HoistGTEs) {
  const absl::string_view hlo_string = R"hlo(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = f32[10] parameter(0)
  neg = f32[10] negate(p0)
  tuple = (f32[10], f32[10]) tuple(neg, p0)
  add = f32[10] add(p0, p0)
  gte = f32[10] get-tuple-element(tuple), index=0
  ROOT root = f32[10] multiply(add, gte)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionHoister hoister(/*hoist_parameters=*/false,
                             /*host_constants=*/false,
                             /*hoist_bitcasts=*/false,
                             /*hoist_gtes=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);
  VerifySchedule(module.get());

  const HloInstructionSequence& sequence =
      module->schedule().sequence(module->entry_computation());

  // The expected sequence should have 'gte' immediately after 'tuple'.
  int tuple_index = -1;
  int gte_index = -1;
  for (int i = 0; i < sequence.size(); ++i) {
    if (sequence.instructions()[i]->opcode() == HloOpcode::kTuple) {
      tuple_index = i;
    } else if (sequence.instructions()[i]->opcode() ==
               HloOpcode::kGetTupleElement) {
      gte_index = i;
    }
  }
  EXPECT_GT(tuple_index, -1);
  EXPECT_EQ(gte_index, tuple_index + 1);
}

TEST_F(InstructionHoisterTest, DefaultDisabled) {
  const absl::string_view hlo_string = R"hlo(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = (f32[10], f32[10]) parameter(0)
  gte0 = f32[10] get-tuple-element(p0), index=0
  neg = f32[10] negate(gte0)
  add = f32[10] add(gte0, gte0)
  bitcast = (f32[10], f32[10]) bitcast(neg)
  gte1 = f32[10] get-tuple-element(p0), index=1
  ROOT root = f32[10] multiply(add, gte1)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  // Explicitly disable everything to verify bitcasts and gtes aren't hoisted by
  // default.
  InstructionHoister hoister(/*hoist_parameters=*/false,
                             /*host_constants=*/false,
                             /*hoist_bitcasts=*/false,
                             /*hoist_gtes=*/false);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));
  EXPECT_FALSE(changed);
}

TEST_F(InstructionHoisterTest, HoistGTEAndBitcast) {
  const absl::string_view hlo_string = R"hlo(
HloModule module, is_scheduled=true

ENTRY entry {
  p0 = (f32[10], f32[10]) parameter(0)
  gte_p0 = f32[10] get-tuple-element(p0), index=0
  neg = f32[10] negate(gte_p0)
  tuple = (f32[10], f32[10]) tuple(neg, gte_p0)
  gte_neg = f32[10] get-tuple-element(tuple), index=0
  add = f32[10] add(gte_p0, gte_p0)
  mul = f32[10] multiply(add, add)
  bitcast = f32[10] bitcast(gte_neg)
  ROOT root = f32[10] multiply(mul, bitcast)
}
)hlo";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  InstructionHoister hoister(/*hoist_parameters=*/true,
                             /*host_constants=*/true,
                             /*hoist_bitcasts=*/true,
                             /*hoist_gtes=*/true);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, hoister.Run(module.get()));
  EXPECT_TRUE(changed);
  VerifySchedule(module.get());

  const HloInstructionSequence& sequence =
      module->schedule().sequence(module->entry_computation());

  // Expected sequence should have gte_neg after tuple, and bitcast after
  // gte_neg
  int tuple_index = -1;
  int gte_neg_index = -1;
  int bitcast_index = -1;
  for (int i = 0; i < sequence.size(); ++i) {
    if (sequence.instructions()[i]->opcode() == HloOpcode::kTuple) {
      tuple_index = i;
    } else if (sequence.instructions()[i]->name() == "gte_neg") {
      gte_neg_index = i;
    } else if (sequence.instructions()[i]->opcode() == HloOpcode::kBitcast) {
      bitcast_index = i;
    }
  }
  EXPECT_GT(tuple_index, -1);
  EXPECT_EQ(gte_neg_index, tuple_index + 1);
  EXPECT_EQ(bitcast_index, gte_neg_index + 1);
}

}  // namespace
}  // namespace xla
