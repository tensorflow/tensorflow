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

#include "xla/service/gpu/fusion_deduplication_cache.h"

#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace xla {
namespace gpu {
namespace {

HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer,
                     bool allow_multi_output = false) {
  HloComputation* computation = consumer->parent();
  HloInstruction* fusion_instruction = consumer;

  if (consumer->opcode() != HloOpcode::kFusion) {
    fusion_instruction =
        computation->AddInstruction(HloInstruction::CreateFusion(
            consumer->shape(), HloInstruction::FusionKind::kLoop, consumer));
    TF_CHECK_OK(computation->ReplaceInstruction(consumer, fusion_instruction));
  }

  if (producer->opcode() == HloOpcode::kFusion) {
    if (allow_multi_output) {
      fusion_instruction->MergeFusionInstructionIntoMultiOutput(producer);
    } else {
      fusion_instruction->MergeFusionInstruction(producer);
    }
  } else {
    if (allow_multi_output) {
      fusion_instruction->FuseInstructionIntoMultiOutput(producer);
    } else {
      fusion_instruction->FuseInstruction(producer);
    }
  }

  // In case of multi-output fusion, `producer` would already be deleted.
  if (!allow_multi_output && producer->user_count() == 0) {
    TF_CHECK_OK(computation->RemoveInstruction(producer));
  }

  return fusion_instruction;
}

bool IsFusible(const HloInstruction& instruction) { return true; }

using FusionDeduplicationCacheTest = HloTestBase;

TEST_F(FusionDeduplicationCacheTest, IdenticalInstructions_EqualId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      add1 = f32[8] add(p0, p1)
      ROOT add2 = f32[8] add(add1, p1)
    })"));

  FusionDeduplicationCache cache =
      FusionDeduplicationCache::Create(*module, IsFusible);

  const HloInstruction* add2 = module->entry_computation()->root_instruction();
  const HloInstruction* add1 = add2->operand(0);
  EXPECT_EQ(cache.GetInstructionId(add1), cache.GetInstructionId(add2));
}

TEST_F(FusionDeduplicationCacheTest,
       IdenticalInstructionsInDifferentComputations_EqualId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    computation.1 {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      ROOT add1 = f32[8] add(p0, p1)
    }

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      ROOT add2 = f32[8] add(p0, p0)
    })"));

  FusionDeduplicationCache cache =
      FusionDeduplicationCache::Create(*module, IsFusible);

  const HloInstruction* add1 =
      module->GetComputationWithName("computation.1")->root_instruction();
  const HloInstruction* add2 = module->entry_computation()->root_instruction();
  EXPECT_EQ(cache.GetInstructionId(add1), cache.GetInstructionId(add2));
}

TEST_F(FusionDeduplicationCacheTest, IdenticalFusionInstructions_EqualId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      log1 = f32[8] log(p0)
      add1 = f32[8] add(log1, p1)
      log2 = f32[8] log(add1)
      ROOT add2 = f32[8] add(log2, p0)
    })"));
  HloComputation* entry_computation = module->entry_computation();

  auto* add1 = entry_computation->GetInstructionWithName("add1");
  auto* add2 = entry_computation->GetInstructionWithName("add2");
  auto* log1 = entry_computation->GetInstructionWithName("log1");
  auto* log2 = entry_computation->GetInstructionWithName("log2");

  FusionDeduplicationCache cache =
      FusionDeduplicationCache::Create(*module, IsFusible);
  EXPECT_EQ(cache.GetInstructionId(add1), cache.GetInstructionId(add2));
  EXPECT_EQ(cache.GetInstructionId(log1), cache.GetInstructionId(log2));
  EXPECT_NE(cache.GetInstructionId(add1), cache.GetInstructionId(log1));

  EXPECT_EQ(cache.GetFusionId(log1, add1), cache.GetFusionId(log2, add2));

  HloInstruction* fusion1 = Fuse(log1, add1);
  cache.UpdateFusedInstructionId(fusion1, log1, add1,
                                 /*consumer_operand_index=*/0);

  HloInstruction* fusion2 = Fuse(log2, add2);
  cache.UpdateFusedInstructionId(fusion2, log2, add2,
                                 /*consumer_operand_index=*/0);

  EXPECT_EQ(cache.GetInstructionId(fusion1), cache.GetInstructionId(fusion2));
}

TEST_F(FusionDeduplicationCacheTest,
       IdenticalMultiOutputFusionInstructions_EqualId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      log1 = f32[8] log(p0)
      add1 = f32[8] add(log1, p1)
      log2 = f32[8] log(add1)
      add2 = f32[8] add(log2, p0)
      ROOT tuple = (f32[8], f32[8], f32[8], f32[8]) tuple(log1, add1, log2, add2)
    })"));
  HloComputation* entry_computation = module->entry_computation();

  auto* add1 = entry_computation->GetInstructionWithName("add1");
  auto* add2 = entry_computation->GetInstructionWithName("add2");
  auto* log1 = entry_computation->GetInstructionWithName("log1");
  auto* log2 = entry_computation->GetInstructionWithName("log2");

  FusionDeduplicationCache cache =
      FusionDeduplicationCache::Create(*module, IsFusible);
  EXPECT_EQ(cache.GetInstructionId(add1), cache.GetInstructionId(add2));
  EXPECT_EQ(cache.GetInstructionId(log1), cache.GetInstructionId(log2));
  EXPECT_NE(cache.GetInstructionId(add1), cache.GetInstructionId(log1));

  EXPECT_EQ(cache.GetFusionId(log1, add1), cache.GetFusionId(log2, add2));

  HloInstruction* fusion1 = Fuse(log1, add1, /*allow_multi_output=*/true);
  cache.UpdateFusedInstructionId(fusion1, log1, add1,
                                 /*consumer_operand_index=*/0,
                                 /*allow_multi_output=*/true);

  HloInstruction* fusion2 = Fuse(log2, add2);
  cache.UpdateFusedInstructionId(fusion2, log2, add2,
                                 /*consumer_operand_index=*/0,
                                 /*allow_multi_output=*/true);

  EXPECT_EQ(cache.GetInstructionId(fusion1), cache.GetInstructionId(fusion2));
}

TEST_F(FusionDeduplicationCacheTest,
       MultiOutputFusionVsSingleOutputFusion_DifferentId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      log1 = f32[8] log(p0)
      add1 = f32[8] add(log1, p1)
      log2 = f32[8] log(add1)
      add2 = f32[8] add(log2, p0)
      ROOT tuple = (f32[8], f32[8], f32[8], f32[8]) tuple(log1, add1, log2, add2)
    })"));
  HloComputation* entry_computation = module->entry_computation();

  auto* add1 = entry_computation->GetInstructionWithName("add1");
  auto* add2 = entry_computation->GetInstructionWithName("add2");
  auto* log1 = entry_computation->GetInstructionWithName("log1");
  auto* log2 = entry_computation->GetInstructionWithName("log2");

  FusionDeduplicationCache cache =
      FusionDeduplicationCache::Create(*module, IsFusible);
  EXPECT_EQ(cache.GetInstructionId(add1), cache.GetInstructionId(add2));
  EXPECT_EQ(cache.GetInstructionId(log1), cache.GetInstructionId(log2));
  EXPECT_NE(cache.GetInstructionId(add1), cache.GetInstructionId(log1));

  EXPECT_EQ(cache.GetFusionId(log1, add1), cache.GetFusionId(log2, add2));

  HloInstruction* fusion1 = Fuse(log1, add1, /*allow_multi_output=*/true);
  cache.UpdateFusedInstructionId(fusion1, log1, add1,
                                 /*consumer_operand_index=*/0,
                                 /*allow_multi_output=*/true);

  HloInstruction* fusion2 = Fuse(log2, add2);
  cache.UpdateFusedInstructionId(fusion2, log2, add2,
                                 /*consumer_operand_index=*/0,
                                 /*allow_multi_output=*/false);

  EXPECT_NE(cache.GetInstructionId(fusion1), cache.GetInstructionId(fusion2));
}

TEST_F(FusionDeduplicationCacheTest,
       DifferentConsumerOperandIndex_DifferentId) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = f32[8] parameter(1)
      log1 = f32[8] log(p0)
      add1 = f32[8] add(log1, p1)
      log2 = f32[8] log(add1)
      ROOT add2 = f32[8] add(p0, log2)
    })"));
  HloComputation* entry_computation = module->entry_computation();

  auto* add1 = entry_computation->GetInstructionWithName("add1");
  auto* add2 = entry_computation->GetInstructionWithName("add2");
  auto* log1 = entry_computation->GetInstructionWithName("log1");
  auto* log2 = entry_computation->GetInstructionWithName("log2");

  FusionDeduplicationCache cache =
      FusionDeduplicationCache::Create(*module, IsFusible);

  EXPECT_NE(cache.GetFusionId(log1, add1), cache.GetFusionId(log2, add2));

  HloInstruction* fusion1 = Fuse(log1, add1);
  cache.UpdateFusedInstructionId(fusion1, log1, add1,
                                 /*consumer_operand_index=*/0);

  HloInstruction* fusion2 = Fuse(log2, add2);
  cache.UpdateFusedInstructionId(fusion2, log2, add2,
                                 /*consumer_operand_index=*/1);

  EXPECT_NE(cache.GetInstructionId(fusion1), cache.GetInstructionId(fusion2));
}

TEST_F(FusionDeduplicationCacheTest, OnlyFusibleInstructionsAreCached) {
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY main {
      p0 = f32[8] parameter(0)
      p1 = (f32[8], f32[8]) parameter(1)
      gte = f32[8] get-tuple-element(p1), index=0
      add = f32[8] add(p0, gte)
      ROOT mul = f32[8] multiply(add, p0)
    })"));

  HloComputation* entry_computation = module->entry_computation();

  auto* add = entry_computation->GetInstructionWithName("add");
  auto* mul = entry_computation->GetInstructionWithName("mul");

  FusionDeduplicationCache cache = FusionDeduplicationCache::Create(
      *module, [&](const HloInstruction& instruction) {
        return instruction.opcode() == HloOpcode::kAdd ||
               instruction.opcode() == HloOpcode::kMultiply;
      });

  // kParameter and kGetTupleElement are not fusible, so assignment of fusion
  // IDs started from `add`.
  EXPECT_EQ(cache.GetInstructionId(add), 0);
  EXPECT_EQ(cache.GetInstructionId(mul), 1);
}

}  // namespace
}  // namespace gpu
}  // namespace xla
