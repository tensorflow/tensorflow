/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/collectives/collective_permute_combiner.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace {

using std::nullopt;
constexpr int64_t kMaxCombineCount = 256;

int64_t CollectivePermuteCount(const HloModule& module) {
  int64_t count = 0;
  for (HloComputation* computation : module.computations()) {
    if (computation->IsFusionComputation()) {
      continue;
    }
    for (HloInstruction* hlo : computation->instructions()) {
      if (hlo->opcode() == HloOpcode::kCollectivePermute) {
        ++count;
      }
    }
  }
  return count;
}

// inputs[i] will be some op producing a shape of size sizes_in_kib[i] which
// feeds into collective permute op in collective_permutes[i]. Returns a tuple
// of the collective_permutes.
HloInstruction* MakeCollectivePermutes(
    std::vector<int64_t> sizes_in_kib, std::vector<HloInstruction*>* inputs,
    std::vector<std::pair<int64_t, int64_t>> source_target_pairs,
    HloComputation::Builder* b) {
  std::vector<HloInstruction*> collective_permutes;
  for (int i = 0; i < sizes_in_kib.size(); i++) {
    auto constant = b->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.3)));
    Shape shape = ShapeUtil::MakeShape(
        F32, {static_cast<int32_t>(sizes_in_kib.at(i) * 1024 / sizeof(float))});
    auto input =
        b->AddInstruction(HloInstruction::CreateBroadcast(shape, constant, {}));
    inputs->push_back(input);
    collective_permutes.push_back(
        b->AddInstruction(HloInstruction::CreateCollectivePermute(
            shape, input, source_target_pairs, /*channel_id=*/nullopt)));
  }
  return b->AddInstruction(HloInstruction::CreateTuple(collective_permutes));
}

using CollectivePermuteCombinerTest = HloHardwareIndependentTestBase;

// Tests combination of several CollectivePermute instructions.
TEST_F(CollectivePermuteCombinerTest, CombineCollectivePermutes) {
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder b(TestName());
  std::vector<HloInstruction*> inputs;
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs{
      {0, 1}, {1, 2}, {2, 3}};
  auto root = MakeCollectivePermutes({1, 2, 10, 7, 6}, &inputs,
                                     source_target_pairs, &b);
  auto computation = module->AddEntryComputation(b.Build());

  // Run the CollectivePermute combiner optimization pass.
  CollectivePermuteCombiner combine(10 * 1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(CollectivePermuteCount(*module), inputs.size());
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  ASSERT_EQ(CollectivePermuteCount(*module), 1);
  EXPECT_TRUE(changed);

  ASSERT_EQ(root, computation->root_instruction());
  ASSERT_EQ(inputs.size(), root->operands().size());

  HloInstruction* combined = nullptr;
  for (int64_t i = 0; i < root->operands().size(); ++i) {
    HloInstruction* hlo = root->mutable_operand(i);
    ASSERT_TRUE(hlo->opcode() == HloOpcode::kGetTupleElement);
    EXPECT_EQ(hlo->tuple_index(), i);
    EXPECT_TRUE(ShapeUtil::Equal(inputs[i]->shape(), hlo->shape()));

    if (combined == nullptr) {
      // Verify the combined collective permute instruction.
      combined = hlo->mutable_operand(0);
      ASSERT_TRUE(combined->opcode() == HloOpcode::kCollectivePermute);
      EXPECT_TRUE(ShapeUtil::Equal(root->shape(), combined->shape()));
      ASSERT_EQ(combined->operands().size(), inputs.size());
    }
    EXPECT_EQ(combined, hlo->operand(0));
    EXPECT_TRUE(ShapeUtil::Equal(inputs[i]->shape(), hlo->shape()));
    EXPECT_EQ(combined->operand(i), inputs[i]);
    EXPECT_EQ(1, inputs[i]->users().size());
  }
  ASSERT_NE(combined, nullptr);
}

// Tests that the combination threshold is respected.
TEST_F(CollectivePermuteCombinerTest, RespectThreshold) {
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder b(TestName());
  std::vector<HloInstruction*> inputs;
  std::vector<std::pair<int64_t, int64_t>> source_target_pairs{
      {0, 1}, {1, 2}, {2, 3}};
  CHECK_NOTNULL(
      MakeCollectivePermutes({2, 10, 7}, &inputs, source_target_pairs, &b));
  module->AddEntryComputation(b.Build());

  // Run the CollectivePermute combiner optimization pass with threshold less
  // than the combined size of the collective permute ops so that the
  // combination cannot occur.
  {
    CollectivePermuteCombiner combine((2 + 10) * 1024 - 1, kMaxCombineCount);
    ASSERT_EQ(CollectivePermuteCount(*module), inputs.size());
    TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
    EXPECT_EQ(CollectivePermuteCount(*module), inputs.size());
    EXPECT_FALSE(changed);
  }

  // Run the CollectivePermute combiner optimization pass again with a slightly
  // higher threshold so that all collective permute ops are combined.
  {
    CollectivePermuteCombiner combine((2 + 10 + 7) * 1024, kMaxCombineCount);
    ASSERT_EQ(CollectivePermuteCount(*module), inputs.size());
    TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
    EXPECT_EQ(CollectivePermuteCount(*module), 1);
    EXPECT_TRUE(changed);
  }
}

// Tests that dependent collective permutes are not combined.
TEST_F(CollectivePermuteCombinerTest, NoDependentCombination) {
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder b(TestName());
  auto constant = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(42.3)));
  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs{{0, 1},
                                                                     {1, 2}};
  auto cp = b.AddInstruction(HloInstruction::CreateCollectivePermute(
      constant->shape(), constant, source_target_pairs,
      /*channel_id=*/nullopt));
  b.AddInstruction(HloInstruction::CreateCollectivePermute(
      constant->shape(), cp, source_target_pairs, /*channel_id=*/nullopt));

  module->AddEntryComputation(b.Build());

  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(CollectivePermuteCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(CollectivePermuteCount(*module), 2);
  EXPECT_FALSE(changed);
}

TEST_F(CollectivePermuteCombinerTest, CombineCollectivePermutesHLO) {
  const char* const hlo_string = R"(
HloModule CombineCollectivePermutes, entry_computation_layout={()->(f32[256]{0}, f32[512]{0}, f32[2560]{0}, f32[1792]{0}, f32[1536]{0})}

ENTRY %CombineCollectivePermutes () -> (f32[256], f32[512], f32[2560], f32[1792], f32[1536]) {
  %constant = f64[] constant(42.3)
  %broadcast = f32[256]{0} broadcast(f64[] %constant), dimensions={}
  %collective-permute = f32[256]{0} collective-permute(f32[256]{0} %broadcast), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.1 = f64[] constant(42.3)
  %broadcast.1 = f32[512]{0} broadcast(f64[] %constant.1), dimensions={}
  %collective-permute.1 = f32[512]{0} collective-permute(f32[512]{0} %broadcast.1), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.2 = f64[] constant(42.3)
  %broadcast.2 = f32[2560]{0} broadcast(f64[] %constant.2), dimensions={}
  %collective-permute.2 = f32[2560]{0} collective-permute(f32[2560]{0} %broadcast.2), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.3 = f64[] constant(42.3)
  %broadcast.3 = f32[1792]{0} broadcast(f64[] %constant.3), dimensions={}
  %collective-permute.3 = f32[1792]{0} collective-permute(f32[1792]{0} %broadcast.3), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.4 = f64[] constant(42.3)
  %broadcast.4 = f32[1536]{0} broadcast(f64[] %constant.4), dimensions={}
  %collective-permute.4 = f32[1536]{0} collective-permute(f32[1536]{0} %broadcast.4), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1
  
  ROOT %tuple = (f32[256]{0}, f32[512]{0}, f32[2560]{0}, f32[1792]{0}, f32[1536]{0}) tuple(f32[256]{0} %collective-permute, f32[512]{0} %collective-permute.1, f32[2560]{0} %collective-permute.2, f32[1792]{0} %collective-permute.3, f32[1536]{0} %collective-permute.4)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  const int64_t total_count = 5;
  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(CollectivePermuteCount(*module), total_count);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_EQ(CollectivePermuteCount(*module), 1);
  EXPECT_TRUE(changed);
}

TEST_F(CollectivePermuteCombinerTest, RespectMaxCombineCount) {
  const char* const hlo_string = R"(
HloModule CombineCollectivePermutes, entry_computation_layout={()->(f32[256]{0}, f32[512]{0}, f32[2560]{0}, f32[1792]{0}, f32[1536]{0})}

ENTRY %CombineCollectivePermutes () -> (f32[256], f32[512], f32[2560], f32[1792], f32[1536]) {
  %constant = f64[] constant(42.3)
  %broadcast = f32[256]{0} broadcast(f64[] %constant), dimensions={}
  %collective-permute = f32[256]{0} collective-permute(f32[256]{0} %broadcast), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.1 = f64[] constant(42.3)
  %broadcast.1 = f32[512]{0} broadcast(f64[] %constant.1), dimensions={}
  %collective-permute.1 = f32[512]{0} collective-permute(f32[512]{0} %broadcast.1), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.2 = f64[] constant(42.3)
  %broadcast.2 = f32[2560]{0} broadcast(f64[] %constant.2), dimensions={}
  %collective-permute.2 = f32[2560]{0} collective-permute(f32[2560]{0} %broadcast.2), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.3 = f64[] constant(42.3)
  %broadcast.3 = f32[1792]{0} broadcast(f64[] %constant.3), dimensions={}
  %collective-permute.3 = f32[1792]{0} collective-permute(f32[1792]{0} %broadcast.3), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.4 = f64[] constant(42.3)
  %broadcast.4 = f32[1536]{0} broadcast(f64[] %constant.4), dimensions={}
  %collective-permute.4 = f32[1536]{0} collective-permute(f32[1536]{0} %broadcast.4), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1
  
  ROOT %tuple = (f32[256]{0}, f32[512]{0}, f32[2560]{0}, f32[1792]{0}, f32[1536]{0}) tuple(f32[256]{0} %collective-permute, f32[512]{0} %collective-permute.1, f32[2560]{0} %collective-permute.2, f32[1792]{0} %collective-permute.3, f32[1536]{0} %collective-permute.4)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  const int64_t total_count = 5;
  const int64_t max_count = 3;
  CollectivePermuteCombiner combine(1024 * 1024, max_count);
  ASSERT_EQ(CollectivePermuteCount(*module), total_count);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  // Expect (total_count // max_count) combined collective permute ops
  EXPECT_EQ(CollectivePermuteCount(*module),
            (total_count + max_count - 1) / max_count);
  EXPECT_TRUE(changed);
}

TEST_F(CollectivePermuteCombinerTest, SourceTargetPairsPreventCombining) {
  const char* const hlo_string = R"(
HloModule CombineCollectivePermutes, entry_computation_layout={()->(f32[256]{0}, f32[512]{0}, f32[2560]{0}, f32[1792]{0}, f32[1536]{0})}

ENTRY %CombineCollectivePermutes () -> (f32[256], f32[512], f32[2560], f32[1792], f32[1536]) {
  %constant = f64[] constant(42.3)
  %broadcast = f32[256]{0} broadcast(f64[] %constant), dimensions={}
  %collective-permute = f32[256]{0} collective-permute(f32[256]{0} %broadcast), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.1 = f64[] constant(42.3)
  %broadcast.1 = f32[512]{0} broadcast(f64[] %constant.1), dimensions={}
  %collective-permute.1 = f32[512]{0} collective-permute(f32[512]{0} %broadcast.1), source_target_pairs={{0,1},{1,2}}, channel_id=1

  %constant.2 = f64[] constant(42.3)
  %broadcast.2 = f32[2560]{0} broadcast(f64[] %constant.2), dimensions={}
  %collective-permute.2 = f32[2560]{0} collective-permute(f32[2560]{0} %broadcast.2), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.3 = f64[] constant(42.3)
  %broadcast.3 = f32[1792]{0} broadcast(f64[] %constant.3), dimensions={}
  %collective-permute.3 = f32[1792]{0} collective-permute(f32[1792]{0} %broadcast.3), source_target_pairs={{0,1},{1,2}}, channel_id=1

  %constant.4 = f64[] constant(42.3)
  %broadcast.4 = f32[1536]{0} broadcast(f64[] %constant.4), dimensions={}
  %collective-permute.4 = f32[1536]{0} collective-permute(f32[1536]{0} %broadcast.4), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1
  
  ROOT %tuple = (f32[256]{0}, f32[512]{0}, f32[2560]{0}, f32[1792]{0}, f32[1536]{0}) tuple(f32[256]{0} %collective-permute, f32[512]{0} %collective-permute.1, f32[2560]{0} %collective-permute.2, f32[1792]{0} %collective-permute.3, f32[1536]{0} %collective-permute.4)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  const int64_t total_count = 5;
  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(CollectivePermuteCount(*module), total_count);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  // Expect two combined collective permute ops since there are two types of
  // source_target_paris in HLO
  EXPECT_EQ(CollectivePermuteCount(*module), 2);
  EXPECT_TRUE(changed);
}

TEST_F(CollectivePermuteCombinerTest, IgnoreChannelId) {
  const char* const hlo_string = R"(
HloModule CombineCollectivePermutes, entry_computation_layout={()->(f32[256]{0}, f32[512]{0}, f32[2560]{0}, f32[1792]{0}, f32[1536]{0})}

ENTRY %CombineCollectivePermutes () -> (f32[256], f32[512], f32[2560], f32[1792], f32[1536]) {
  %constant = f64[] constant(42.3)
  %broadcast = f32[256]{0} broadcast(f64[] %constant), dimensions={}
  %collective-permute = f32[256]{0} collective-permute(f32[256]{0} %broadcast), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.1 = f64[] constant(42.3)
  %broadcast.1 = f32[512]{0} broadcast(f64[] %constant.1), dimensions={}
  %collective-permute.1 = f32[512]{0} collective-permute(f32[512]{0} %broadcast.1), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1

  %constant.2 = f64[] constant(42.3)
  %broadcast.2 = f32[2560]{0} broadcast(f64[] %constant.2), dimensions={}
  %collective-permute.2 = f32[2560]{0} collective-permute(f32[2560]{0} %broadcast.2), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=2

  %constant.3 = f64[] constant(42.3)
  %broadcast.3 = f32[1792]{0} broadcast(f64[] %constant.3), dimensions={}
  %collective-permute.3 = f32[1792]{0} collective-permute(f32[1792]{0} %broadcast.3), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=2

  %constant.4 = f64[] constant(42.3)
  %broadcast.4 = f32[1536]{0} broadcast(f64[] %constant.4), dimensions={}
  %collective-permute.4 = f32[1536]{0} collective-permute(f32[1536]{0} %broadcast.4), source_target_pairs={{0,1},{1,2},{2,3}}, channel_id=1
  
  ROOT %tuple = (f32[256]{0}, f32[512]{0}, f32[2560]{0}, f32[1792]{0}, f32[1536]{0}) tuple(f32[256]{0} %collective-permute, f32[512]{0} %collective-permute.1, f32[2560]{0} %collective-permute.2, f32[1792]{0} %collective-permute.3, f32[1536]{0} %collective-permute.4)
})";
  HloModuleConfig config = GetModuleConfigForTest();
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  const int64_t total_count = 5;
  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(CollectivePermuteCount(*module), total_count);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  // Expect one combined collective permute op since channel_id is ignored
  EXPECT_EQ(CollectivePermuteCount(*module), 1);
  EXPECT_TRUE(changed);
}

// Tests that independent collective permutes separated by interleaved
// dependency chains are combined. Two chains: cp_A→cp_B and cp_C→cp_D.
// Topological order: A, B, C, D. With the fix (continue instead of break),
// A and C get combined, B and D get combined → 2 ops.
// Without the fix (break), A becomes a singleton, B+C get combined, D becomes
// a singleton → 3 ops.
TEST_F(CollectivePermuteCombinerTest,
       CombineAcrossInterleavedDependencyChains) {
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder b(TestName());
  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs{{0, 1}};
  Shape shape = ShapeUtil::MakeShape(F32, {256});

  // Chain 1: constant1 → broadcast1 → cp_A → cp_B
  auto constant1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(1.0)));
  auto input_a =
      b.AddInstruction(HloInstruction::CreateBroadcast(shape, constant1, {}));
  auto cp_a = b.AddInstruction(HloInstruction::CreateCollectivePermute(
      shape, input_a, source_target_pairs, /*channel_id=*/std::nullopt));
  auto cp_b = b.AddInstruction(HloInstruction::CreateCollectivePermute(
      shape, cp_a, source_target_pairs, /*channel_id=*/std::nullopt));

  // Chain 2: constant3 → broadcast3 → cp_C → cp_D
  auto constant3 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(3.0)));
  auto input_c =
      b.AddInstruction(HloInstruction::CreateBroadcast(shape, constant3, {}));
  auto cp_c = b.AddInstruction(HloInstruction::CreateCollectivePermute(
      shape, input_c, source_target_pairs, /*channel_id=*/std::nullopt));
  auto cp_d = b.AddInstruction(HloInstruction::CreateCollectivePermute(
      shape, cp_c, source_target_pairs, /*channel_id=*/std::nullopt));

  b.AddInstruction(HloInstruction::CreateTuple({cp_a, cp_b, cp_c, cp_d}));
  module->AddEntryComputation(b.Build());

  ASSERT_EQ(CollectivePermuteCount(*module), 4);

  // With the fix (continue instead of break), A and C get combined, B and D get
  // combined → 2 ops.
  // Without the fix (break), A becomes a singleton, B+C get combined, D becomes
  // a singleton → 3 ops.
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_enable_enzyme_comms_opt(true);

  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_EQ(CollectivePermuteCount(*module), 2);
}

TEST_F(CollectivePermuteCombinerTest,
       CombineAcrossInterleavedDependencyChainsDefault) {
  auto module = CreateNewVerifiedModule();

  HloComputation::Builder b(TestName());
  const std::vector<std::pair<int64_t, int64_t>> source_target_pairs{{0, 1}};
  Shape shape = ShapeUtil::MakeShape(F32, {256});

  // Chain 1: constant1 → broadcast1 → cp_A → cp_B
  auto constant1 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(1.0)));
  auto input_a =
      b.AddInstruction(HloInstruction::CreateBroadcast(shape, constant1, {}));
  auto cp_a = b.AddInstruction(HloInstruction::CreateCollectivePermute(
      shape, input_a, source_target_pairs, /*channel_id=*/std::nullopt));
  auto cp_b = b.AddInstruction(HloInstruction::CreateCollectivePermute(
      shape, cp_a, source_target_pairs, /*channel_id=*/std::nullopt));

  // Chain 2: constant3 → broadcast3 → cp_C → cp_D
  auto constant3 = b.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(3.0)));
  auto input_c =
      b.AddInstruction(HloInstruction::CreateBroadcast(shape, constant3, {}));
  auto cp_c = b.AddInstruction(HloInstruction::CreateCollectivePermute(
      shape, input_c, source_target_pairs, /*channel_id=*/std::nullopt));
  auto cp_d = b.AddInstruction(HloInstruction::CreateCollectivePermute(
      shape, cp_c, source_target_pairs, /*channel_id=*/std::nullopt));

  b.AddInstruction(HloInstruction::CreateTuple({cp_a, cp_b, cp_c, cp_d}));
  module->AddEntryComputation(b.Build());

  ASSERT_EQ(CollectivePermuteCount(*module), 4);

  // Default behavior is break, so it should result in 3 ops.
  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));

  EXPECT_TRUE(changed);
  EXPECT_EQ(CollectivePermuteCount(*module), 3);
}

TEST_F(CollectivePermuteCombinerTest, DeduplicateCombinedOperandsEnabled) {
  const char* const hlo_string = R"(
HloModule Deduplicate, entry_computation_layout={()->(f32[256]{0}, f32[256]{0}, f32[256]{0})}

ENTRY %Deduplicate () -> (f32[256], f32[256], f32[256]) {
  %constant = f32[256]{0} constant({...})
  %op1 = f32[256]{0} copy(%constant)
  %op2 = f32[256]{0} copy(%constant)
  %cp1 = f32[256]{0} collective-permute(%op1), source_target_pairs={{0,1}}, channel_id=1
  %cp2 = f32[256]{0} collective-permute(%op2), source_target_pairs={{0,1}}, channel_id=1
  %cp3 = f32[256]{0} collective-permute(%op1), source_target_pairs={{0,1}}, channel_id=1
  ROOT %tuple = (f32[256], f32[256], f32[256]) tuple(%cp1, %cp2, %cp3)
})";
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().set_xla_enable_enzyme_comms_opt(true);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CollectivePermuteCount(*module), 1);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* combined = root->operand(0)->operand(0);
  ASSERT_EQ(combined->opcode(), HloOpcode::kCollectivePermute);
  // Identical operands %op1 are deduplicated, so we have %op1 and %op2.
  EXPECT_EQ(combined->operand_count(), 2);

  // %cp1 -> GTE(0), %cp2 -> GTE(1), %cp3 -> GTE(0)
  EXPECT_EQ(root->operand(0)->tuple_index(), 0);
  EXPECT_EQ(root->operand(1)->tuple_index(), 1);
  EXPECT_EQ(root->operand(2)->tuple_index(), 0);
}

TEST_F(CollectivePermuteCombinerTest, DeduplicateCombinedOperandsDisabled) {
  const char* const hlo_string = R"(
HloModule Deduplicate, entry_computation_layout={()->(f32[256]{0}, f32[256]{0}, f32[256]{0})}

ENTRY %Deduplicate () -> (f32[256], f32[256], f32[256]) {
  %constant = f32[256]{0} constant({...})
  %op1 = f32[256]{0} copy(%constant)
  %op2 = f32[256]{0} copy(%constant)
  %cp1 = f32[256]{0} collective-permute(%op1), source_target_pairs={{0,1}}, channel_id=1
  %cp2 = f32[256]{0} collective-permute(%op2), source_target_pairs={{0,1}}, channel_id=1
  %cp3 = f32[256]{0} collective-permute(%op1), source_target_pairs={{0,1}}, channel_id=1
  ROOT %tuple = (f32[256], f32[256], f32[256]) tuple(%cp1, %cp2, %cp3)
})";
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().set_xla_enable_enzyme_comms_opt(false);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CollectivePermuteCount(*module), 1);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* combined = root->operand(0)->operand(0);
  ASSERT_EQ(combined->opcode(), HloOpcode::kCollectivePermute);
  // No deduplication.
  EXPECT_EQ(combined->operand_count(), 3);

  EXPECT_EQ(root->operand(0)->tuple_index(), 0);
  EXPECT_EQ(root->operand(1)->tuple_index(), 1);
  EXPECT_EQ(root->operand(2)->tuple_index(), 2);
}

TEST_F(CollectivePermuteCombinerTest, DeduplicateToOneOperandEnabled) {
  const char* const hlo_string = R"(
HloModule Deduplicate, entry_computation_layout={()->(f32[256]{0}, f32[256]{0})}

ENTRY %Deduplicate () -> (f32[256], f32[256]) {
  %constant = f32[256]{0} constant({...})
  %op1 = f32[256]{0} copy(%constant)
  %cp1 = f32[256]{0} collective-permute(%op1), source_target_pairs={{0,1}}, channel_id=1
  %cp2 = f32[256]{0} collective-permute(%op1), source_target_pairs={{0,1}}, channel_id=1
  ROOT %tuple = (f32[256], f32[256]) tuple(%cp1, %cp2)
})";
  HloModuleConfig config = GetModuleConfigForTest();
  config.mutable_debug_options().set_xla_enable_enzyme_comms_opt(true);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string, config));

  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);
  EXPECT_EQ(CollectivePermuteCount(*module), 1);

  const HloInstruction* root = module->entry_computation()->root_instruction();
  // Both %cp1 and %cp2 should be replaced by the same non-tuple %combined.
  const HloInstruction* combined = root->operand(0);
  EXPECT_EQ(combined, root->operand(1));
  ASSERT_EQ(combined->opcode(), HloOpcode::kCollectivePermute);
  EXPECT_EQ(combined->operand_count(), 1);
  EXPECT_TRUE(combined->shape().IsArray());
}

TEST_F(CollectivePermuteCombinerTest,
       PreservesFrontendAttributesAndMergedMetadata) {
  const char* const hlo_string = R"(
HloModule CombineCollectivePermutes

ENTRY %CombineCollectivePermutes () -> (f32[256], f32[512]) {
  %constant = f64[] constant(42.3)
  %broadcast = f32[256]{0} broadcast(f64[] %constant), dimensions={}
  %collective-permute = f32[256]{0} collective-permute(f32[256]{0} %broadcast), source_target_pairs={{0,1},{1,0}}, channel_id=1, frontend_attributes={RESOURCE_TYPE="2",_scheduling_group="ring_attn_fwd_0"}, metadata={op_type="cp" op_name="model/ring/cp_0"}

  %constant.1 = f64[] constant(42.3)
  %broadcast.1 = f32[512]{0} broadcast(f64[] %constant.1), dimensions={}
  %collective-permute.1 = f32[512]{0} collective-permute(f32[512]{0} %broadcast.1), source_target_pairs={{0,1},{1,0}}, channel_id=1, frontend_attributes={RESOURCE_TYPE="2",_scheduling_group="ring_attn_fwd_0"}, metadata={op_type="cp" op_name="model/ring/cp_1"}

  ROOT %tuple = (f32[256]{0}, f32[512]{0}) tuple(%collective-permute, %collective-permute.1)
})";
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          ParseAndReturnVerifiedModule(hlo_string));

  CollectivePermuteCombiner combine(1024 * 1024, kMaxCombineCount);
  ASSERT_EQ(CollectivePermuteCount(*module), 2);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, combine.Run(module.get()));
  EXPECT_TRUE(changed);
  ASSERT_EQ(CollectivePermuteCount(*module), 1);

  const HloInstruction* combined =
      FindInstruction(module.get(), HloOpcode::kCollectivePermute);
  ASSERT_NE(combined, nullptr);

  // Frontend attributes: shared keys with same values preserved.
  ASSERT_TRUE(combined->has_frontend_attributes());
  const auto& attrs = combined->frontend_attributes().map();
  EXPECT_EQ(attrs.at("RESOURCE_TYPE"), "2");
  EXPECT_EQ(attrs.at("_scheduling_group"), "ring_attn_fwd_0");

  // Metadata: common prefix "model/ring/" extracted, suffixes joined.
  const OpMetadata& md = combined->metadata();
  EXPECT_EQ(md.op_type(), "cp");
  EXPECT_EQ(md.op_name(), "model/ring/(cp_0:cp_1)");
}

}  // namespace
}  // namespace xla
