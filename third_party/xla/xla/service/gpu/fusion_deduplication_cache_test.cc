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

HloInstruction* Fuse(HloInstruction* producer, HloInstruction* consumer) {
  HloComputation* computation = consumer->parent();
  HloInstruction* fusion_instruction = consumer;

  if (consumer->opcode() != HloOpcode::kFusion) {
    fusion_instruction =
        computation->AddInstruction(HloInstruction::CreateFusion(
            consumer->shape(), HloInstruction::FusionKind::kLoop, consumer));
    TF_CHECK_OK(computation->ReplaceInstruction(consumer, fusion_instruction));
  }

  if (producer->opcode() == HloOpcode::kFusion) {
    fusion_instruction->MergeFusionInstruction(producer);
  } else {
    fusion_instruction->FuseInstruction(producer);
  }

  if (producer->user_count() == 0) {
    TF_CHECK_OK(computation->RemoveInstruction(producer));
  }

  return fusion_instruction;
}

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

  FusionDeduplicationCache cache = FusionDeduplicationCache::Create(*module);

  const HloInstruction* add2 = module->entry_computation()->root_instruction();
  const HloInstruction* add1 = add2->operand(0);
  EXPECT_EQ(cache.GetInstructionId(*add1), cache.GetInstructionId(*add2));
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

  FusionDeduplicationCache cache = FusionDeduplicationCache::Create(*module);

  const HloInstruction* add1 =
      module->GetComputationWithName("computation.1")->root_instruction();
  const HloInstruction* add2 = module->entry_computation()->root_instruction();
  EXPECT_EQ(cache.GetInstructionId(*add1), cache.GetInstructionId(*add2));
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

  FusionDeduplicationCache cache = FusionDeduplicationCache::Create(*module);
  EXPECT_EQ(cache.GetInstructionId(*add1), cache.GetInstructionId(*add2));
  EXPECT_EQ(cache.GetInstructionId(*log1), cache.GetInstructionId(*log2));
  EXPECT_NE(cache.GetInstructionId(*add1), cache.GetInstructionId(*log1));

  EXPECT_EQ(cache.GetFusionId(*log1, *add1), cache.GetFusionId(*log2, *add2));

  HloInstruction* fusion1 = Fuse(log1, add1);
  cache.UpdateFusedInstructionId(*fusion1, *log1, *add1,
                                 /*consumer_operand_index=*/0);

  HloInstruction* fusion2 = Fuse(log2, add2);
  cache.UpdateFusedInstructionId(*fusion2, *log2, *add2,
                                 /*consumer_operand_index=*/0);

  EXPECT_EQ(cache.GetInstructionId(*fusion1), cache.GetInstructionId(*fusion2));
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

  FusionDeduplicationCache cache = FusionDeduplicationCache::Create(*module);

  EXPECT_NE(cache.GetFusionId(*log1, *add1), cache.GetFusionId(*log2, *add2));

  HloInstruction* fusion1 = Fuse(log1, add1);
  cache.UpdateFusedInstructionId(*fusion1, *log1, *add1,
                                 /*consumer_operand_index=*/0);

  HloInstruction* fusion2 = Fuse(log2, add2);
  cache.UpdateFusedInstructionId(*fusion2, *log2, *add2,
                                 /*consumer_operand_index=*/1);

  EXPECT_NE(cache.GetInstructionId(*fusion1), cache.GetInstructionId(*fusion2));
}

}  // namespace
}  // namespace gpu
}  // namespace xla
