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

class HloOrderingTest : public HloTestBase {
 protected:
  // Creates and returns a computation which can benefit from
  // rematerialization. The computation looks like:
  //
  //   F32[1] %param = {...}
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1024] %negate = negate(%bcast)
  //   F32[2048] %concat_1 = concat({%negate, %negate})
  //   F32[1] %slice_1 = slice(%concat_1, {0:1})
  //   F32[1025] %concat_2 = concat({%bcast, %slice_1})
  //   F32[1] %slice_2 = slice(%concat_2, {0:1});
  //
  // The instruction %bcast can be rematerialized before its use at %concat_2
  // to reduce peak memory usage. This avoids %bcast and %concat_1 being
  // simultaneously live. Peak memory use is about 16KB before rematerialization
  // (during execution of %concat_1) and about 12KB after rematerializing %bcast
  // for its use in %concat_2.
  std::unique_ptr<HloComputation> MakeRematerializableComputation(
      const string& suffix = "") {
    auto builder = HloComputation::Builder(TestName() + suffix);
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "param"));
    auto bcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec1024_shape_, param, {}));
    auto negate = builder.AddInstruction(
        HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, bcast));
    auto concat_1 = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {2048}), {negate, negate},
        /*dimension=*/0));
    auto slice_1 = builder.AddInstruction(HloInstruction::CreateSlice(
        vec1_shape_, concat_1, /*start_indices=*/{0},
        /*limit_indices=*/{1}));
    auto concat_2 = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {1025}), {bcast, slice_1},
        /*dimension=*/0));
    // Add a final slice to make the parameter shape match the output shape
    // which is necessary to use this computation in a while.
    builder.AddInstruction(HloInstruction::CreateSlice(vec1_shape_, concat_2,
                                                       /*start_indices=*/{0},
                                                       /*limit_indices=*/{1}));
    return builder.Build();
  }

  // Creates and returns a computation which includes a while and can benefit
  // from rematerialization. The computation looks like:
  //
  //   F32[1] %param = {...}
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1] %slice_1 = slice(%bcast, {0:1})
  //   F32[1] %while = while(%slice_1, while_body, while_cond)
  //   F32[1025] %concat = concat({%bcast, %while})
  //   F32[1] %slice_2 = slice(%concat, {0:1});
  //
  // The instruction %bcast can be rematerialized before its use at %concat to
  // reduce peak memory usage. This avoids %bcast being live during execution of
  // the while. Peak memory use is maximum of 8K and 4K plus the memory use of
  // the while subcomputations.
  std::unique_ptr<HloComputation> MakeRematerializableWhileComputation(
      HloComputation* while_cond, HloComputation* while_body,
      const string& suffix = "") {
    auto builder = HloComputation::Builder(TestName() + suffix);
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "param"));
    auto bcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec1024_shape_, param, {}));
    auto slice_1 = builder.AddInstruction(
        HloInstruction::CreateSlice(vec1_shape_, bcast, /*start_indices=*/{0},
                                    /*limit_indices=*/{1}));
    auto while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
        vec1_shape_, while_cond, while_body, slice_1));
    auto concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {1025}), {bcast, while_inst},
        /*dimension=*/0));
    builder.AddInstruction(HloInstruction::CreateSlice(vec1_shape_, concat,
                                                       /*start_indices=*/{0},
                                                       /*limit_indices=*/{1}));
    return builder.Build();
  }

  // Create and return a trivial computation appropriate for use as a while
  // condition.
  std::unique_ptr<HloComputation> MakeConditionComputation() {
    auto builder = HloComputation::Builder(TestName() + ".cond");
    builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1_shape_, "param"));
    builder.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
    return builder.Build();
  }

  // Return the byte size of the top-level buffer of the given shape.
  static int64 ByteSizeOf(const Shape& shape) {
    return ShapeUtil::ByteSizeOf(shape, sizeof(void*));
  }

  // Various shapes used in the canned computations.
  const Shape vec1_shape_ = ShapeUtil::MakeShape(xla::F32, {1});
  const Shape vec1024_shape_ = ShapeUtil::MakeShape(xla::F32, {1024});
};

// Test rematerialization of a single computation produced by
// MakeRematerializableComputation.
TEST_F(HloOrderingTest, SingleComputation) {
  HloModule module(TestName());
  HloComputation* computation =
      module.AddEntryComputation(MakeRematerializableComputation());

  // Find and save the original broadcast instruction which should be
  // rematerialized.
  const HloInstruction* slice = computation->root_instruction();
  ASSERT_EQ(HloOpcode::kSlice, slice->opcode());
  const HloInstruction* concat = slice->operand(0);
  ASSERT_EQ(HloOpcode::kConcatenate, concat->opcode());
  const HloInstruction* bcast = concat->operand(0);
  ASSERT_EQ(HloOpcode::kBroadcast, bcast->opcode());

  SequentialHloOrdering::HloModuleSequence sequence;
  // Computation requires 16KB without rematerialization, but uses only 12KB
  // with rematerialization so pick a memory limit between these values (14KB).
  TF_ASSIGN_OR_ASSERT_OK(
      bool changed, HloRematerialization::RematerializeAndSchedule(
                        ByteSizeOf,
                        /*memory_limit_bytes=*/14 * 1024, &module, &sequence));
  EXPECT_TRUE(changed);

  // Root should not have changed.
  EXPECT_EQ(computation->root_instruction(), slice);

  // The broadcast should have been rematerialized.
  const HloInstruction* remat_bcast = concat->operand(0);
  EXPECT_EQ(HloOpcode::kBroadcast, remat_bcast->opcode());
  EXPECT_NE(bcast, remat_bcast);

  // The rematerialized broadcast should be immediate before the concat in the
  // sequence.
  EXPECT_EQ(sequence.at(computation)[computation->instruction_count() - 2],
            concat);
  EXPECT_EQ(sequence.at(computation)[computation->instruction_count() - 3],
            remat_bcast);
}

// Test rematerialization of a single computation produced by
// MakeRematerializableComputation but with a sufficiently high memory limit
// such that no instructions are rematerialized.
TEST_F(HloOrderingTest, SingleComputationNoRematerialization) {
  HloModule module(TestName());
  HloComputation* computation =
      module.AddEntryComputation(MakeRematerializableComputation());

  EXPECT_EQ(computation->instruction_count(), 7);

  SequentialHloOrdering::HloModuleSequence sequence;
  TF_ASSIGN_OR_ASSERT_OK(
      bool changed, HloRematerialization::RematerializeAndSchedule(
                        ByteSizeOf,
                        /*memory_limit_bytes=*/20 * 1024, &module, &sequence));

  // No instructions should have been materialized.
  EXPECT_FALSE(changed);
  EXPECT_EQ(computation->instruction_count(), 7);
}

// Test rematerialization of a computation which calls another computation via a
// while. Both the entry computation and while body computation can have memory
// usage reduced via rematerialization however the memory limit is set such that
// only one computation needs to have an instruction rematerialized. The entry
// computation should be the one chosen because rematerialization in the while
// will presumably be more expensive.
TEST_F(HloOrderingTest, RematerializeAroundWhile) {
  HloModule module(TestName());

  auto cond_builder = HloComputation::Builder(TestName() + ".cond");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloComputation* while_cond =
      module.AddEmbeddedComputation(cond_builder.Build());

  HloComputation* body_computation = module.AddEmbeddedComputation(
      MakeRematerializableComputation(/*suffix=*/".body"));
  HloComputation* entry_computation =
      module.AddEntryComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/body_computation));

  EXPECT_EQ(entry_computation->instruction_count(), 6);
  EXPECT_EQ(body_computation->instruction_count(), 7);

  // The body computation uses 16KB and the entry computation uses 2KB at the
  // while so the peak memory use of the module is 18KB. Set the memory limit a
  // bit lower (17KB) to force rematerialization of the entry computation.
  SequentialHloOrdering::HloModuleSequence sequence;
  TF_ASSIGN_OR_ASSERT_OK(
      bool changed, HloRematerialization::RematerializeAndSchedule(
                        ByteSizeOf,
                        /*memory_limit_bytes=*/17 * 1024, &module, &sequence));
  EXPECT_TRUE(changed);

  // Only the entry computation should have a rematerialized instruction added.
  EXPECT_EQ(entry_computation->instruction_count(), 7);
  EXPECT_EQ(body_computation->instruction_count(), 7);
}

// Test rematerialization of a computation which calls another computation via a
// while. Both the entry computation and while body computation should have
// computations rematerialized.
TEST_F(HloOrderingTest, RematerializeEntryAndWhileBody) {
  HloModule module(TestName());

  auto cond_builder = HloComputation::Builder(TestName() + ".cond");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloComputation* while_cond =
      module.AddEmbeddedComputation(cond_builder.Build());

  HloComputation* body_computation = module.AddEmbeddedComputation(
      MakeRematerializableComputation(/*suffix=*/".body"));
  HloComputation* entry_computation =
      module.AddEntryComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/body_computation));

  EXPECT_EQ(entry_computation->instruction_count(), 6);
  EXPECT_EQ(body_computation->instruction_count(), 7);

  SequentialHloOrdering::HloModuleSequence sequence;
  TF_ASSIGN_OR_ASSERT_OK(
      bool changed, HloRematerialization::RematerializeAndSchedule(
                        ByteSizeOf,
                        /*memory_limit_bytes=*/15 * 1024, &module, &sequence));
  EXPECT_TRUE(changed);

  // Both computations should have a rematerialized instruction added.
  EXPECT_EQ(entry_computation->instruction_count(), 7);
  EXPECT_EQ(body_computation->instruction_count(), 8);
}

// Test rematerialization of a doubly nested computation. All computations
// should have an instruction rematerialized.
TEST_F(HloOrderingTest, RematerializeNestedComputations) {
  HloModule module(TestName());

  auto cond_builder = HloComputation::Builder(TestName() + ".cond");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloComputation* while_cond =
      module.AddEmbeddedComputation(cond_builder.Build());

  HloComputation* inner_computation = module.AddEmbeddedComputation(
      MakeRematerializableComputation(/*suffix=*/".inner"));
  HloComputation* middle_computation =
      module.AddEmbeddedComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/inner_computation,
          /*suffix=*/".middle"));
  HloComputation* entry_computation =
      module.AddEntryComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/middle_computation));

  EXPECT_EQ(entry_computation->instruction_count(), 6);
  EXPECT_EQ(middle_computation->instruction_count(), 6);
  EXPECT_EQ(inner_computation->instruction_count(), 7);

  // If all computations are maximally rematerialized then peak memory usage is
  // ~12K so pick something slightly larger.
  SequentialHloOrdering::HloModuleSequence sequence;
  TF_ASSIGN_OR_ASSERT_OK(
      bool changed, HloRematerialization::RematerializeAndSchedule(
                        ByteSizeOf,
                        /*memory_limit_bytes=*/13 * 1024, &module, &sequence));
  EXPECT_TRUE(changed);

  // All computations should have a rematerialized instruction added.
  EXPECT_EQ(entry_computation->instruction_count(), 7);
  EXPECT_EQ(middle_computation->instruction_count(), 7);
  EXPECT_EQ(inner_computation->instruction_count(), 8);
}

}  // namespace

}  // namespace xla
