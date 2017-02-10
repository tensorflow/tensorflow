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

#include "tensorflow/compiler/xla/service/hlo_rematerialization.h"

#include <memory>
#include <string>

#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {
namespace {

class HloOrderingTest : public HloTestBase {};

TEST_F(HloOrderingTest, SimpleRematerialization) {
  // Construct a simple computation which requires rematerialization to get
  // below the desired memory usage.
  //
  //   F32[1024] %big_param = {...}
  //   F32[] %scalar_param = 42.0;
  //   F32[1024] %bcast = broadcast(%scalar_param)
  //   F32[1024] %add = add(%big_param, %bcast)
  //   F32[1] %slice = slice(%add)
  //   %tuple = tuple(%slice, %bcast)
  //
  // At the program point between %add and %slice, large values %bcast, %add,
  // and %big_param are live. Rematerializing %bcast right before the tuple
  // makes it so only two large values are ever simultaneously live.
  auto builder = HloComputation::Builder(TestName());
  const Shape big_shape = ShapeUtil::MakeShape(xla::F32, {1024});
  auto big_param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, big_shape, "big_param"));
  auto scalar_param = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(xla::F32, {}), "scalar_param"));
  auto bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(big_shape, scalar_param, {}));
  auto add = builder.AddInstruction(HloInstruction::CreateBinary(
      big_shape, HloOpcode::kAdd, big_param, bcast));
  auto slice = builder.AddInstruction(HloInstruction::CreateSlice(
      ShapeUtil::MakeShape(xla::F32, {1}), add, /*start_indices=*/{0},
      /*limit_indices=*/{1}));

  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({slice, bcast}));

  HloModule module(TestName());
  auto computation = module.AddEntryComputation(builder.Build());

  SequentialHloOrdering::HloModuleSequence sequence;
  TF_ASSIGN_OR_ASSERT_OK(
      bool changed,
      HloRematerialization::RematerializeAndSchedule(
          [](const Shape& shape) {
            return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
          },
          /*memory_limit_bytes=*/2500 * sizeof(float), &module, &sequence));
  EXPECT_TRUE(changed);

  // Operand one of the tuple root should be a rematerialized broadcast.
  EXPECT_EQ(tuple, computation->root_instruction());
  const HloInstruction* remat_bcast = tuple->operand(1);
  EXPECT_EQ(HloOpcode::kBroadcast, remat_bcast->opcode());
  EXPECT_NE(bcast, tuple->operand(1));

  // The rematerialized broadcast should be immediate before the root in the
  // sequence.
  EXPECT_EQ(sequence.at(computation)[computation->instruction_count() - 1],
            tuple);
  EXPECT_EQ(sequence.at(computation)[computation->instruction_count() - 2],
            remat_bcast);
}

}  // namespace

}  // namespace xla
