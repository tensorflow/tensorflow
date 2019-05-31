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
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_ordering.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::_;

class HloRematerializationTest : public HloTestBase {
 protected:
  // Creates and returns a computation which can benefit from
  // rematerialization. The computation looks like:
  //
  //   F32[1] %param = {...}
  //   F32[] %reshape = reshape(F32[], param)
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
    auto reshape = builder.AddInstruction(
        HloInstruction::CreateReshape(scalar_shape_, param));
    auto bcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec1024_shape_, reshape, {}));
    auto negate = builder.AddInstruction(
        HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, bcast));
    auto concat_1 = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {2048}), {negate, negate},
        /*dimension=*/0));
    auto slice_1 = builder.AddInstruction(HloInstruction::CreateSlice(
        vec1_shape_, concat_1, /*start_indices=*/{0},
        /*limit_indices=*/{1},
        /*strides=*/{1}));
    auto concat_2 = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {1025}), {bcast, slice_1},
        /*dimension=*/0));
    // Add a final slice to make the parameter shape match the output shape
    // which is necessary to use this computation in a while.
    builder.AddInstruction(HloInstruction::CreateSlice(vec1_shape_, concat_2,
                                                       /*start_indices=*/{0},
                                                       /*limit_indices=*/{1},
                                                       /*strides=*/{1}));
    return builder.Build();
  }

  // Creates and returns a computation which includes a while and can benefit
  // from rematerialization. The computation looks like:
  //
  //   F32[] %param = {...}
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
    auto reshape = builder.AddInstruction(
        HloInstruction::CreateReshape(scalar_shape_, param));
    auto bcast = builder.AddInstruction(
        HloInstruction::CreateBroadcast(vec1024_shape_, reshape, {}));
    auto slice_1 = builder.AddInstruction(
        HloInstruction::CreateSlice(vec1_shape_, bcast, /*start_indices=*/{0},
                                    /*limit_indices=*/{1},
                                    /*strides=*/{1}));
    auto while_inst = builder.AddInstruction(HloInstruction::CreateWhile(
        vec1_shape_, while_cond, while_body, slice_1));
    auto concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {1025}), {bcast, while_inst},
        /*dimension=*/0));
    builder.AddInstruction(HloInstruction::CreateSlice(vec1_shape_, concat,
                                                       /*start_indices=*/{0},
                                                       /*limit_indices=*/{1},
                                                       /*strides=*/{1}));
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

  StatusOr<bool> RunHloRematerialization(int64 memory_limit_bytes,
                                         HloModule* module) {
    TF_EXPECT_OK(verifier().Run(module).status());
    HloMemoryScheduler scheduler(
        [](const BufferValue& buffer) { return ByteSizeOf(buffer.shape()); },
        DefaultMemoryScheduler);
    TF_EXPECT_OK(scheduler.Run(module).status());
    HloRematerialization remat(ByteSizeOf, memory_limit_bytes,
                               /*sizes=*/nullptr);
    return remat.Run(module);
  }

  // Various shapes used in the canned computations.
  const Shape scalar_shape_ = ShapeUtil::MakeShape(xla::F32, {});
  const Shape vec1_shape_ = ShapeUtil::MakeShape(xla::F32, {1});
  const Shape vec1024_shape_ = ShapeUtil::MakeShape(xla::F32, {1024});
};

// Test rematerialization of a single computation produced by
// MakeRematerializableComputation.
TEST_F(HloRematerializationTest, SingleComputation) {
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(MakeRematerializableComputation());

  // Find and save the original broadcast instruction which should be
  // rematerialized.
  const HloInstruction* slice = computation->root_instruction();
  ASSERT_THAT(slice, op::Slice(op::Concatenate(op::Broadcast(_), _)));
  const HloInstruction* concat = slice->operand(0);
  const HloInstruction* bcast = concat->operand(0);

  // Computation requires 16KB without rematerialization, but uses only 12KB
  // with rematerialization so pick a memory limit between these values (14KB).
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/14 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // Root should not have changed.
  EXPECT_EQ(computation->root_instruction(), slice);

  // The broadcast should have been rematerialized.
  const HloInstruction* remat_bcast = concat->operand(0);
  EXPECT_THAT(remat_bcast, op::Broadcast(::testing::Ne(bcast)));

  // The rematerialized broadcast should be immediate before the concat in the
  // sequence.
  EXPECT_EQ(module->schedule()
                .sequence(computation)
                .instructions()[computation->instruction_count() - 2],
            concat);
  EXPECT_EQ(module->schedule()
                .sequence(computation)
                .instructions()[computation->instruction_count() - 3],
            remat_bcast);
}

// Test rematerialization of a single computation produced by
// MakeRematerializableComputation but with a sufficiently high memory limit
// such that no instructions are rematerialized.
TEST_F(HloRematerializationTest, SingleComputationNoRematerialization) {
  auto module = CreateNewVerifiedModule();
  HloComputation* computation =
      module->AddEntryComputation(MakeRematerializableComputation());

  EXPECT_EQ(computation->instruction_count(), 8);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/20 * 1024, module.get()));

  // No instructions should have been materialized.
  EXPECT_FALSE(changed);
  EXPECT_EQ(computation->instruction_count(), 8);
}

// Test rematerialization of a computation which calls another computation via a
// while. Both the entry computation and while body computation can have memory
// usage reduced via rematerialization however the memory limit is set such that
// only one computation needs to have an instruction rematerialized. The entry
// computation should be the one chosen because rematerialization in the while
// will presumably be more expensive.
TEST_F(HloRematerializationTest, RematerializeAroundWhile) {
  auto module = CreateNewVerifiedModule();

  auto cond_builder = HloComputation::Builder(TestName() + ".cond");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloComputation* while_cond =
      module->AddEmbeddedComputation(cond_builder.Build());

  HloComputation* body_computation = module->AddEmbeddedComputation(
      MakeRematerializableComputation(/*suffix=*/".body"));
  HloComputation* entry_computation =
      module->AddEntryComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/body_computation));

  EXPECT_EQ(entry_computation->instruction_count(), 7);
  EXPECT_EQ(body_computation->instruction_count(), 8);

  // The body computation uses 16KB and the entry computation uses 2KB at the
  // while so the peak memory use of the module is 18KB. Set the memory limit a
  // bit lower (17KB) to force rematerialization of the entry computation.
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/17 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // Only the entry computation should have a rematerialized instruction added.
  EXPECT_EQ(entry_computation->instruction_count(), 8);
  EXPECT_EQ(body_computation->instruction_count(), 8);
}

// Test rematerialization of a computation which calls another computation via a
// while. Both the entry computation and while body computation should have
// computations rematerialized.
TEST_F(HloRematerializationTest, RematerializeEntryAndWhileBody) {
  auto module = CreateNewVerifiedModule();

  auto cond_builder = HloComputation::Builder(TestName() + ".cond");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloComputation* while_cond =
      module->AddEmbeddedComputation(cond_builder.Build());

  HloComputation* body_computation = module->AddEmbeddedComputation(
      MakeRematerializableComputation(/*suffix=*/".body"));
  HloComputation* entry_computation =
      module->AddEntryComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/body_computation));

  EXPECT_EQ(entry_computation->instruction_count(), 7);
  EXPECT_EQ(body_computation->instruction_count(), 8);

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/15 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // Both computations should have rematerialized instructions added.
  EXPECT_EQ(entry_computation->instruction_count(), 9);
  EXPECT_EQ(body_computation->instruction_count(), 9);
}

// Test rematerialization of a doubly nested computation. All computations
// should have an instruction rematerialized.
TEST_F(HloRematerializationTest, RematerializeNestedComputations) {
  auto module = CreateNewVerifiedModule();

  auto cond_builder = HloComputation::Builder(TestName() + ".cond");
  cond_builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1_shape_, "param"));
  cond_builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloComputation* while_cond =
      module->AddEmbeddedComputation(cond_builder.Build());

  HloComputation* inner_computation = module->AddEmbeddedComputation(
      MakeRematerializableComputation(/*suffix=*/".inner"));
  HloComputation* middle_computation =
      module->AddEmbeddedComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/inner_computation,
          /*suffix=*/".middle"));
  HloComputation* entry_computation =
      module->AddEntryComputation(MakeRematerializableWhileComputation(
          while_cond, /*while_body=*/middle_computation));

  EXPECT_EQ(entry_computation->instruction_count(), 7);
  EXPECT_EQ(middle_computation->instruction_count(), 7);
  EXPECT_EQ(inner_computation->instruction_count(), 8);

  // If all computations are maximally rematerialized then peak memory usage is
  // ~12K so pick something slightly larger.
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/13 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // All computations should have rematerialized instructions added.
  EXPECT_EQ(entry_computation->instruction_count(), 9);
  EXPECT_EQ(middle_computation->instruction_count(), 9);
  EXPECT_EQ(inner_computation->instruction_count(), 9);
}

TEST_F(HloRematerializationTest, RngNotRematerialized) {
  // Test that a single rng is not rematerialized:
  //
  // Entry computation:
  //   F32[] %param = {...}
  //   F32[1024] rng = rng(param)
  //   F32[1024] tanh = tanh(rng)
  //   F32[1024] exp = exp(rng)
  //   F32[1024] add_0 = add(rng, tanh)              // LIVE: add_0 + rng +
  //                                                 //       tanh + exp
  //
  //   F32[1024] add_1 = add(rng, add(exp, add_0))   // LIVE: add_1 + add_0 +
  //                                                 //       rng + tanh + exp
  //
  //   F32[1024] add_2 = add(rng, add(tanh, add_1))  // LIVE: add_2 + add_1 +
  //                                                 //       rng + tanh + exp
  auto module = CreateNewVerifiedModule();

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto rng = builder.AddInstruction(HloInstruction::CreateRng(
      vec1024_shape_, RandomDistribution::RNG_UNIFORM, {param, param}));
  auto tanh = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kTanh, rng));
  auto exp = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kExp, rng));
  auto add_0 = builder.AddInstruction(
      HloInstruction::CreateBinary(vec1024_shape_, HloOpcode::kAdd, rng, tanh));
  auto add_1 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, rng,
      builder.AddInstruction(HloInstruction::CreateBinary(
          vec1024_shape_, HloOpcode::kAdd, exp, add_0))));
  builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, rng,
      builder.AddInstruction(HloInstruction::CreateBinary(
          vec1024_shape_, HloOpcode::kAdd, tanh, add_1))));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  auto count_rngs = [](const HloComputation* computation) {
    int64 rng_count = 0;
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kRng) {
        ++rng_count;
      }
    }
    return rng_count;
  };
  // Before rematerialization there should be a single broadcast rng in
  // the graph.
  ASSERT_EQ(count_rngs(entry_computation), 1);
  const int64 original_instruction_count =
      entry_computation->instruction_count();
  // Pick a memory limit some where between 24KB (initial peak memory including
  // parameter and output) and 20KB (peak memory possible with
  // rematerialization).
  TF_ASSERT_OK_AND_ASSIGN(
      bool changed,
      RunHloRematerialization(
          /*memory_limit_bytes=*/4 * ByteSizeOf(vec1024_shape_), module.get()));
  EXPECT_TRUE(changed);
  // The rng should not have been rematerialized.
  EXPECT_EQ(count_rngs(entry_computation), 1);
  // There should have been rematerialization.
  EXPECT_GT(entry_computation->instruction_count(), original_instruction_count);
}

TEST_F(HloRematerializationTest, InstructionRematerializedMultipleTimes) {
  // Test that a single instruction is rematerialized several times. Module:
  //
  // Entry computation:
  //   F32[] %param = {...}
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1024] %add_1 = add(%bcast, bcast)
  //   F32[1024] %call_1 = call(Subcomputation, {%add_1})
  //   F32[1024] %add_2 = add(%bcast, call_1)
  //   F32[1024] %call_2 = call(SubComputation, {%add_2})
  //   F32[1024] %add_3 = add(%bcast, call_2)
  //   F32[1024] %call_3 = call(Subcomputation, {%add_3})
  //   F32[1024] %add_4 = add(%bcast, call_3)
  //
  // Subcomputation:
  //   F32[1024] %param = {...}
  //   F32[2048] %concat = concat({%param, %param})
  //   F32[1024] %slice = slice(%concat)
  //
  // The value %bcast is live across each call of Subcomputation (which requires
  // 8KB) though the value is not used in the calls. Rematerializing %bcast
  // across these calls reduces peak memory use from ~20KB down to ~16KB.
  auto module = CreateNewVerifiedModule();

  HloComputation* subcomputation = nullptr;
  {
    auto builder = HloComputation::Builder(TestName() + ".subcomputation");
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1024_shape_, "param"));
    auto concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {2048}), {param, param},
        /*dimension=*/0));
    builder.AddInstruction(HloInstruction::CreateSlice(
        vec1024_shape_, concat, /*start_indices=*/{0},
        /*limit_indices=*/{1024}, /*strides=*/{1}));
    subcomputation = module->AddEmbeddedComputation(builder.Build());
  }

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(vec1024_shape_, param, {}));
  auto add_1 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, bcast));
  auto call_1 = builder.AddInstruction(
      HloInstruction::CreateCall(vec1024_shape_, {add_1}, subcomputation));
  auto add_2 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, call_1));
  auto call_2 = builder.AddInstruction(
      HloInstruction::CreateCall(vec1024_shape_, {add_2}, subcomputation));
  auto add_3 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, call_2));
  auto call_3 = builder.AddInstruction(
      HloInstruction::CreateCall(vec1024_shape_, {add_3}, subcomputation));
  auto add_4 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, call_3));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  auto count_broadcasts = [](const HloComputation* computation) {
    int64 bcast_count = 0;
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kBroadcast) {
        bcast_count++;
      }
    }
    return bcast_count;
  };

  // Before rematerialization there should be a single broadcast instruction in
  // the graph.
  EXPECT_EQ(count_broadcasts(entry_computation), 1);
  EXPECT_EQ(entry_computation->instruction_count(), 9);

  EXPECT_EQ(add_2->operand(0), bcast);
  EXPECT_EQ(add_3->operand(0), bcast);
  EXPECT_EQ(add_4->operand(0), bcast);

  // Pick a memory limit some where between 24KB (initial peak memory including
  // parameter and output) and 20KB (peak memory possible with
  // rematerialization).
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/22 * 1024, module.get()));
  EXPECT_TRUE(changed);

  // The broadcast should have been rematerialized 3 times.
  EXPECT_EQ(count_broadcasts(entry_computation), 4);
  EXPECT_EQ(entry_computation->instruction_count(), 12);

  // The operands of add_2, add_3, and add_4 should all be rematerialized
  // broadcasts.
  EXPECT_NE(add_2->operand(0), bcast);
  EXPECT_THAT(add_2->operand(0), op::Broadcast(param));
  EXPECT_NE(add_3->operand(0), bcast);
  EXPECT_THAT(add_3->operand(0), op::Broadcast(param));
  EXPECT_NE(add_4->operand(0), bcast);
  EXPECT_THAT(add_4->operand(0), op::Broadcast(param));
}

TEST_F(HloRematerializationTest, CopyNotRematerialized) {
  // Test that copies are not rematerialized.
  auto module = CreateNewVerifiedModule();

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, vec1024_shape_, "param"));

  auto copy = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kCopy, param));

  auto negate_a_1 = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, copy));

  auto negate_a_2 = builder.AddInstruction(HloInstruction::CreateUnary(
      vec1024_shape_, HloOpcode::kNegate, negate_a_1));

  auto negate_b_1 = builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, copy));

  auto negate_b_2 = builder.AddInstruction(HloInstruction::CreateUnary(
      vec1024_shape_, HloOpcode::kNegate, negate_b_1));

  builder.AddInstruction(HloInstruction::CreateTuple({negate_a_2, negate_b_2}));

  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/1 * 1024, module.get()));

  auto count_copies = [](const HloComputation* computation) {
    int64 copy_count = 0;
    for (auto* instruction : computation->instructions()) {
      if (instruction->opcode() == HloOpcode::kCopy) {
        copy_count++;
      }
    }
    return copy_count;
  };
  EXPECT_TRUE(changed);

  EXPECT_EQ(count_copies(entry_computation), 1);
}

class IndirectUseTest : public HloRematerializationTest,
                        public ::testing::WithParamInterface<bool> {};

TEST_P(IndirectUseTest, IndirectUseNotRematerialized) {
  // Test that an rematerializable instruction is not rematerialized if it has
  // an indirect use. Test is parameterized on whether the value has an indirect
  // use, and the instruction should be rematerialized iff the value has no
  // indirect use. Module:
  //
  // Entry computation:
  //   F32[] %param = {...}
  //   F32[1024] %bcast = broadcast(%param)
  //   F32[1024] %add_1 = add(%bcast, bcast)
  //   F32[1024] %call = call(Subcomputation, {%add_1})
  //   F32[1024] %add_2 = add(%bcast, call)
  //   {F32[1024], F32[1024]} %tuple = tuple(%bcast, %add_2)
  //   F32[1024] %gte = GetTupleElememt(%tuple, 0)
  //   F32[1024] %negate = negate(%gte)
  //
  // Subcomputation:
  //   F32[1024] %param = {...}
  //   F32[2048] %concat = concat({%param, %param})
  //   F32[1024] %slice = slice(%concat)
  //
  // The value %bcast is live across the call and rematerialization of %bcast
  // across that point would reduce peak memory use by 4KB. However, %bcast is
  // used indirectly in the %negate so rematerialization should not happen.
  //
  // This test is parameterized on whether the broadcast has an indirect use or
  // not. The indirect use is controlled by the index of the GetTupleElement
  // instruction. If the element is 0, then the %negate operand aliases %bcast
  // (ie %bcast is used indirectly by %negate), otherwise the %negate operand
  // aliases %add_2.
  const bool indirectly_used = GetParam();
  auto module = CreateNewVerifiedModule();

  HloComputation* subcomputation = nullptr;
  {
    auto builder = HloComputation::Builder(TestName() + ".subcomputation");
    auto param = builder.AddInstruction(
        HloInstruction::CreateParameter(0, vec1024_shape_, "param"));
    auto concat = builder.AddInstruction(HloInstruction::CreateConcatenate(
        ShapeUtil::MakeShape(xla::F32, {2048}), {param, param},
        /*dimension=*/0));
    builder.AddInstruction(HloInstruction::CreateSlice(
        vec1024_shape_, concat, /*start_indices=*/{0},
        /*limit_indices=*/{1024}, /*strides=*/{1}));
    subcomputation = module->AddEmbeddedComputation(builder.Build());
  }

  auto builder = HloComputation::Builder(TestName());
  auto param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, scalar_shape_, "param"));
  auto bcast = builder.AddInstruction(
      HloInstruction::CreateBroadcast(vec1024_shape_, param, {}));
  auto add_1 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, bcast));
  auto call_1 = builder.AddInstruction(
      HloInstruction::CreateCall(vec1024_shape_, {add_1}, subcomputation));
  auto add_2 = builder.AddInstruction(HloInstruction::CreateBinary(
      vec1024_shape_, HloOpcode::kAdd, bcast, call_1));
  auto tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({bcast, add_2}));
  auto gte = builder.AddInstruction(HloInstruction::CreateGetTupleElement(
      vec1024_shape_, tuple, indirectly_used ? 0 : 1));
  builder.AddInstruction(
      HloInstruction::CreateUnary(vec1024_shape_, HloOpcode::kNegate, gte));
  HloComputation* entry_computation =
      module->AddEntryComputation(builder.Build());

  EXPECT_EQ(entry_computation->instruction_count(), 8);

  // Pick a memory limit some where between 24KB (initial peak memory including
  // parameter and output) and 20KB (peak memory possible with
  // rematerialization).
  TF_ASSERT_OK_AND_ASSIGN(bool changed,
                          RunHloRematerialization(
                              /*memory_limit_bytes=*/22 * 1024, module.get()));
  // Rematerialization should only occur if the rematerializable instruction has
  // no indirect uses.
  if (indirectly_used) {
    EXPECT_FALSE(changed);
    EXPECT_EQ(entry_computation->instruction_count(), 8);
  } else {
    EXPECT_TRUE(changed);
    EXPECT_EQ(entry_computation->instruction_count(), 9);
  }
}

INSTANTIATE_TEST_SUITE_P(IndirectUseTestInstantiation, IndirectUseTest,
                         ::testing::Values(true, false));

}  // namespace

}  // namespace xla
