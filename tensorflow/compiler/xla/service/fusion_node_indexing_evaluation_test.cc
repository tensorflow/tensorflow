/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/fusion_node_indexing_evaluation.h"

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/instruction_fusion.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/platform/test.h"

namespace xla {

using FusionNodeIndexingEvaluationTest = HloTestBase;

// Subclass of InstructionFusion exposing the protected methods Fuse and
// FuseInstruction for testing. Also adds the FusionNodeIndexingEvaluation to
// track the code duplication due to indexing HloInstructions with
// different index values.
class InstructionFusionForTesting : public InstructionFusion {
 public:
  explicit InstructionFusionForTesting(HloModule* module)
      : InstructionFusion(InstructionFusion::IsExpensive) {
    module_ = module;
    computation_ = module->entry_computation();
  }

  HloInstruction* FuseInstruction(HloInstruction* fusion_instruction,
                                  HloInstruction* producer) override {
    auto evaluation = fusion_node_evaluations_.find(fusion_instruction);
    if (evaluation == fusion_node_evaluations_.end()) {
      evaluation =
          fusion_node_evaluations_
              .emplace(fusion_instruction,
                       FusionNodeIndexingEvaluation(fusion_instruction))
              .first;
    }
    auto indexing_users = evaluation->second.RemoveFusionOperand(producer);
    HloInstruction* new_producer =
        InstructionFusion::FuseInstruction(fusion_instruction, producer);
    evaluation->second.UpdateEvaluationCache(new_producer, indexing_users);
    return new_producer;
  }

  HloInstruction* Fuse(HloInstruction* producer,
                       HloInstruction* consumer) override {
    return InstructionFusion::Fuse(producer, consumer);
  }

  int64 EvaluateEmittedInstructions(const HloInstruction* producer,
                                    const HloInstruction* consumer) {
    if (consumer->opcode() != HloOpcode::kFusion) {
      return 0;
    }
    if (fusion_node_evaluations_.find(consumer) ==
        fusion_node_evaluations_.end()) {
      fusion_node_evaluations_.emplace(consumer,
                                       FusionNodeIndexingEvaluation(consumer));
    }
    return fusion_node_evaluations_.at(consumer).EvaluateEmittedInstructions(
        producer);
  }

 private:
  absl::flat_hash_map<const HloInstruction*, FusionNodeIndexingEvaluation>
      fusion_node_evaluations_;
};

TEST_F(FusionNodeIndexingEvaluationTest, FuseTwoInstructions) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY entry_computation {
    p0 = f32[4,3]{1,0} parameter(0)
    add = f32[4,3]{1,0} add(p0, p0)
    ROOT sub = f32[4,3]{1,0} subtract(add, p0)
  })")
                    .ValueOrDie();
  HloInstruction* sub = module->entry_computation()->root_instruction();
  HloInstruction* add = sub->mutable_operand(0);
  InstructionFusionForTesting(module.get()).Fuse(add, sub);
}

TEST_F(FusionNodeIndexingEvaluationTest, FuseThreeInstructions) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
  ENTRY entry_computation {
    p0 = f32[4]{0} parameter(0)
    slice1 = f32[3]{0} slice(p0), slice={[0:3]}
    slice2 = f32[3]{0} slice(p0), slice={[0:3]}
    ROOT sub = f32[3]{0} subtract(slice1, slice2)
  })")
                    .ValueOrDie();
  HloInstruction* sub = module->entry_computation()->root_instruction();
  InstructionFusionForTesting instruction_fusion(module.get());
  HloInstruction* slice1 = sub->mutable_operand(0);
  HloInstruction* slice2 = sub->mutable_operand(1);
  auto fusion = instruction_fusion.Fuse(slice1, sub);
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(slice2, fusion), 1);
  instruction_fusion.Fuse(slice2, fusion);
}

TEST_F(FusionNodeIndexingEvaluationTest, ExponentialDuplicationPattern) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
  ENTRY entry_computation {
    p0 = f32[4]{0} parameter(0)
    p1 = f32[4]{0} parameter(1)
    add0 = f32[4]{0} add(p0, p1)
    slice1.0 = f32[3]{0} slice(add0), slice={[0:3]}
    slice1.1 = f32[3]{0} slice(add0), slice={[1:4]}
    add1 = f32[3]{0} add(slice1.0, slice1.1)
    slice2.0 = f32[2]{0} slice(add1), slice={[0:2]}
    slice2.1 = f32[2]{0} slice(add1), slice={[1:3]}
    ROOT add2 = f32[2]{0} add(slice2.0, slice2.1)
  })")
                    .ValueOrDie();
  // This corresponds to the following graph:
  //              add0
  //            /      \
  //       slice1.0  slice1.1
  //            \      /
  //              add1
  //            /      \
  //       slice2.0  slice2.1
  //            \      /
  //              add2
  // This pattern can be arbitrarily extended. In this example, add2, slice2.0,
  // slice2.1 each get emitted once because they can be indexed with the same
  // index vector. Since add1 has a different shape than its two users, it
  // needs to be emitted twice. slice1.0 and slice1.1 each also get emitted
  // twice because they get passed both different index vectors from add1. add0
  // then gets emitted 4 times.
  HloInstruction* add2 = module->entry_computation()->root_instruction();
  InstructionFusionForTesting instruction_fusion(module.get());
  HloInstruction* slice2_0 = add2->mutable_operand(0);
  HloInstruction* slice2_1 = add2->mutable_operand(1);
  auto fusion = instruction_fusion.Fuse(slice2_0, add2);
  // So far we have fused add2 and slice2.0. So when we also fuse slice2.1, we
  // expect to emit it 1 time.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(slice2_1, fusion),
            1);
  instruction_fusion.Fuse(slice2_1, fusion);
  HloInstruction* add1 = fusion->mutable_operand(0);
  EXPECT_EQ(add1->opcode(), HloOpcode::kAdd);
  // If we fuse add1 into 'fusion', it needs to be emitted twice.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(add1, fusion), 2);
  instruction_fusion.Fuse(add1, fusion);
  HloInstruction* slice1_0 = fusion->mutable_operand(0);
  EXPECT_EQ(slice1_0->opcode(), HloOpcode::kSlice);
  // If we fuse slice1.0 into 'fusion', it needs to be emitted twice.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(slice1_0, fusion),
            2);
  instruction_fusion.Fuse(slice1_0, fusion);
  HloInstruction* slice1_1 = fusion->mutable_operand(0);
  EXPECT_EQ(slice1_1->opcode(), HloOpcode::kSlice);
  // If we fuse slice1.1 into 'fusion', it needs to be emitted twice.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(slice1_1, fusion),
            2);
  instruction_fusion.Fuse(slice1_1, fusion);
  HloInstruction* add0 = fusion->mutable_operand(0);
  EXPECT_EQ(add0->opcode(), HloOpcode::kAdd);
  // If we fuse add0 into 'fusion', it needs to be emitted four times.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(add0, fusion), 4);
  instruction_fusion.Fuse(add0, fusion);
}

TEST_F(FusionNodeIndexingEvaluationTest, RecomputeCache) {
  // This is the same HloModule as in ExponentialDuplicationPattern above, but
  // starting with the fusion node as it is before 'add0' is fused in.
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule test_module
%fused_computation (param_0.5: f32[4]) -> f32[2] {
  %param_0.5 = f32[4]{0} parameter(0)
  %slice1.2 = f32[3]{0} slice(f32[4]{0} %param_0.5), slice={[0:3]}
  %slice1.3 = f32[3]{0} slice(f32[4]{0} %param_0.5), slice={[1:4]}
  %add1.1 = f32[3]{0} add(f32[3]{0} %slice1.2, f32[3]{0} %slice1.3)
  %slice2.2 = f32[2]{0} slice(f32[3]{0} %add1.1), slice={[0:2]}
  %slice2.3 = f32[2]{0} slice(f32[3]{0} %add1.1), slice={[1:3]}
  ROOT %add2.1 = f32[2]{0} add(f32[2]{0} %slice2.2, f32[2]{0} %slice2.3)
}

ENTRY entry_computation {
  p0 = f32[4]{0} parameter(0)
  p1 = f32[4]{0} parameter(1)
  add0 = f32[4]{0} add(p0, p1)
  ROOT %fusion = f32[2]{0} fusion(add0), kind=kLoop, calls=%fused_computation
})")
                    .ValueOrDie();
  HloInstruction* fusion = module->entry_computation()->root_instruction();
  InstructionFusionForTesting instruction_fusion(module.get());
  HloInstruction* add0 = fusion->mutable_operand(0);
  EXPECT_EQ(add0->opcode(), HloOpcode::kAdd);
  // Here, the cache for the fusion node needs to be recomputed. Make sure we
  // still get the same evaluation as before when we incrementally build the
  // cache.
  EXPECT_EQ(instruction_fusion.EvaluateEmittedInstructions(add0, fusion), 4);
}

}  // namespace xla
