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
#include "tensorflow/compiler/xla/service/hlo_rematerialization_test_utils.h"

#include <memory>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/utils/hlo_matchers.h"
#include "tensorflow/compiler/xla/iterator_util.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/tsl/lib/core/status_test_util.h"

namespace xla {
namespace {

using ::testing::UnorderedElementsAre;

class HloRematerializationTestUtilsTest : public RematerializationTestBase {};

TEST_F(HloRematerializationTestUtilsTest, MakeRematerializableComputation) {
  auto computation = MakeRematerializableComputation();

  // Prescriptive check to verify that all expected instructions appear.
  std::vector<HloInstruction*> instructions(computation->instructions().begin(),
                                            computation->instructions().end());
  EXPECT_EQ(instructions[0]->name(), "param");
  EXPECT_EQ(instructions[1]->name(), "reshape");
  EXPECT_THAT(instructions[1]->operands(),
              UnorderedElementsAre(instructions[0]));
  EXPECT_EQ(instructions[2]->name(), "broadcast");
  EXPECT_THAT(instructions[2]->operands(),
              UnorderedElementsAre(instructions[1]));
  EXPECT_EQ(instructions[3]->name(), "negate");
  EXPECT_THAT(instructions[3]->operands(),
              UnorderedElementsAre(instructions[2]));
  EXPECT_EQ(instructions[4]->name(), "concatenate");
  EXPECT_THAT(instructions[4]->operands(),
              UnorderedElementsAre(instructions[3], instructions[3]));
  EXPECT_EQ(instructions[5]->name(), "slice");
  EXPECT_THAT(instructions[5]->operands(),
              UnorderedElementsAre(instructions[4]));
  EXPECT_EQ(instructions[6]->name(), "concatenate");
  EXPECT_THAT(instructions[6]->operands(),
              UnorderedElementsAre(instructions[2], instructions[5]));
  EXPECT_EQ(instructions[7]->name(), "slice");
  EXPECT_THAT(instructions[7]->operands(),
              UnorderedElementsAre(instructions[6]));
}

TEST_F(HloRematerializationTestUtilsTest,
       MakeRematerializableWhileComputation) {
  auto while_condition = MakeConditionComputation();
  auto body_computation = MakeRematerializableComputation();
  auto computation = MakeRematerializableWhileComputation(
      while_condition.get(), body_computation.get());

  // Prescriptive check to verify that all expected instructions appear.
  std::vector<HloInstruction*> instructions(computation->instructions().begin(),
                                            computation->instructions().end());
  EXPECT_EQ(instructions[0]->name(), "param");
  EXPECT_EQ(instructions[1]->name(), "reshape");
  EXPECT_THAT(instructions[1]->operands(),
              UnorderedElementsAre(instructions[0]));
  EXPECT_EQ(instructions[2]->name(), "broadcast");
  EXPECT_THAT(instructions[2]->operands(),
              UnorderedElementsAre(instructions[1]));
  EXPECT_EQ(instructions[3]->name(), "slice");
  EXPECT_THAT(instructions[3]->operands(),
              UnorderedElementsAre(instructions[2]));
  // We also check the condition and body for the while instruction.
  EXPECT_EQ(instructions[4]->name(), "while");
  EXPECT_THAT(instructions[4]->operands(),
              UnorderedElementsAre(instructions[3]));
  EXPECT_EQ(instructions[4]->while_condition()->name(),
            "MakeRematerializableWhileComputation.cond");
  EXPECT_EQ(instructions[4]->while_body()->name(),
            "MakeRematerializableWhileComputation");
  EXPECT_EQ(instructions[5]->name(), "concatenate");
  EXPECT_THAT(instructions[5]->operands(),
              UnorderedElementsAre(instructions[2], instructions[4]));
  EXPECT_EQ(instructions[6]->name(), "slice");
  EXPECT_THAT(instructions[6]->operands(),
              UnorderedElementsAre(instructions[5]));
}

}  // namespace
}  // namespace xla
