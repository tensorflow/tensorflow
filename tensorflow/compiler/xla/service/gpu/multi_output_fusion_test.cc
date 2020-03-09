/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/multi_output_fusion.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace gpu {

namespace op = xla::testing::opcode_matchers;

using MultiOutputFusionTest = HloTestBase;

const char kModulePrefix[] = R"(
    HloModule test_module

    scalar_add_computation {
      scalar_lhs.0 = f32[] parameter(0)
      scalar_rhs.0 = f32[] parameter(1)
      ROOT add.0 = f32[] add(scalar_lhs.0, scalar_rhs.0)
    }
    scalar_mul_computation {
      scalar_lhs.1 = f32[] parameter(0)
      scalar_rhs.1 = f32[] parameter(1)
      ROOT mul.1 = f32[] multiply(scalar_lhs.1, scalar_rhs.1)
    })";

static int64 CountMultiOutputFusions(const HloModule* module) {
  int multi_output_fusion_count = 0;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instr : computation->instructions()) {
      if (instr->IsMultiOutputFusion()) {
        multi_output_fusion_count++;
      }
    }
  }
  return multi_output_fusion_count;
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingReduceAndReduceFusion) {
  // Fusion with reduce instruction root and a sibling reduce instruction
  // sharing the same input param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[512]{0} reduce(mul, const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      const.2 = f32[] constant(1)
      fusion = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation
      reduce.2 = f32[512]{0} reduce(p1, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT root = (f32[512]{0}, f32[512]{0}) tuple(fusion, reduce.2)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce()));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDifferentReduceInputShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[6400]{0} parameter(1)
      mul = f32[6400]{0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[] reduce(mul, const.1), dimensions={0}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = f32[6400]{0} parameter(1)
      r1 = f32[64,100]{0,1} reshape(p1.2)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[] reduce(r1, const.2), dimensions={1,0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[6400]{0} parameter(1)
      fusion.1 = f32[] fusion(p0, p1), kind=kInput, calls=fused_computation_1
      fusion.2 = f32[] fusion(p0, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[], f32[]) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDifferentReduceOutputShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[10,10]{1,0} parameter(1)
      mul = f32[10,10]{1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[] reduce(mul, const.1), dimensions={0,1}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = f32[10,10]{1,0} parameter(1)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[10]{0} reduce(p1.2, const.2), dimensions={0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1.3 = f32[10,10]{1,0} parameter(1)
      fusion.1 = f32[] fusion(p0, p1.3), kind=kInput, calls=fused_computation_1
      p2 = f32[] parameter(2)
      fusion.2 = f32[10]{0} fusion(p2, p1.3), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[], f32[10]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingReduceFusions) {
  // Two sibling fusions with reduce instruction roots sharing the same input
  // param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      ROOT reduce.1 = f32[512]{0} reduce(mul, const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    fused_computation_2 {
      p1.2 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[512]{0} reduce(p1.2, const.2), dimensions={0,2,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      fusion.1 = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation_1
      fusion.2 = f32[512] fusion(p0, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[512]{0}, f32[512]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce()));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingReduceAndReduceMultiOutputFusion) {
  // Multi-output fusion with two reduce instructions root and a sibling reduce
  // instruction sharing the same input param.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation (p0: f32[128,512,28,28]) -> (f32[512], f32[512]) {
      const.1 = f32[] constant(1)
      p0.1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      mul = f32[128,512,28,28]{3,2,1,0} multiply(f32[128,512,28,28]{3,2,1,0} p0.1, f32[128,512,28,28]{3,2,1,0} p0.1)
      reduce.1 = f32[512]{0} reduce(f32[128,512,28,28]{3,2,1,0} mul, f32[] const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
      reduce.2 = f32[512]{0} reduce(f32[128,512,28,28]{3,2,1,0} p0.1, f32[] const.1), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT tuple = (f32[512]{0}, f32[512]{0}) tuple(f32[512]{0} reduce.1, f32[512]{0} reduce.2)
    }

    ENTRY entry (p0: f32[128,512,28,28]) -> (f32[512], f32[512], f32[512]) {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      const = f32[] constant(1)
      fusion = (f32[512]{0}, f32[512]{0}) fusion(f32[128,512,28,28]{3,2,1,0} p0), kind=kInput, calls=fused_computation
      get-tuple-element = f32[512]{0} get-tuple-element((f32[512]{0}, f32[512]{0}) fusion), index=0
      get-tuple-element.1 = f32[512]{0} get-tuple-element((f32[512]{0}, f32[512]{0}) fusion), index=1
      reduce.3 = f32[512]{0} reduce(p0, const), dimensions={0,2,3}, to_apply=scalar_add_computation
      ROOT root = (f32[512]{0}, f32[512]{0}, f32[512]{0}) tuple(f32[512]{0} get-tuple-element, f32[512]{0} get-tuple-element.1, f32[512]{0} reduce.3)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce(), op::Reduce()));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingFusionCheckAgainstReduceOperand) {
  // Verify that if we already have a multi-output fusion that we prefer to pick
  // a reduce op from its operands for checking shape compatibility.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p1.1 = f32[10,10]{1,0} parameter(1)
      mul = f32[10,10]{1,0} multiply(p1.1, p1.1)
      const.1 = f32[] parameter(0)
      reduce.1 = f32[] reduce(p1.1, const.1), dimensions={0,1}, to_apply=scalar_add_computation
      ROOT tuple = (f32[10,10], f32[]) tuple(mul, reduce.1)
    }

    fused_computation_2 {
      p1.2 = f32[10,10]{1,0} parameter(1)
      const.2 = f32[] parameter(0)
      ROOT reduce.2 = f32[10] reduce(p1.2, const.2), dimensions={0}, to_apply=scalar_mul_computation
    }

    ENTRY entry {
      p0 = f32[] parameter(0)
      p1 = f32[10,10]{1,0} parameter(1)
      p2 = f32[] parameter(2)
      fusion.1 = (f32[10,10], f32[]) fusion(p0, p1), kind=kInput, calls=fused_computation_1
      get-tuple-element.1 = f32[10,10] get-tuple-element((f32[10,10], f32[]) fusion.1), index=0
      get-tuple-element.2 = f32[] get-tuple-element((f32[10,10], f32[]) fusion.1), index=1
      fusion.2 = f32[10] fusion(p2, p1), kind=kInput, calls=fused_computation_2
      ROOT root = (f32[10,10], f32[], f32[10]) tuple(get-tuple-element.1, get-tuple-element.2, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionTwoLoops) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = f32[6400]{0} parameter(0)
      const.2 = f32[] constant(1)
      broadcast = f32[6400]{0} broadcast(const.2), dimensions={}
      ROOT div = f32[6400]{0} divide(p0.2, broadcast)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (f32[6400]{0}, f32[6400]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Multiply(), op::Divide()));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionLoopReduceToInputFusion) {
  // Fusing a reduce into a loop fusion would require changing the fusion kind.
  // That's not supported yet.
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      const.2 = f32[] constant(0)
      reduce = f32[] reduce(p0, const.2), dimensions={0}, to_apply=scalar_add_computation
      ROOT root = (f32[6400]{0}, f32[]) tuple(fusion.1, reduce)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionLoopElementwise) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      const.2 = f32[] constant(1)
      broadcast = f32[6400]{0} broadcast(const.2), dimensions={}
      div = f32[6400]{0} divide(p0, broadcast)
      ROOT root = (f32[6400]{0}, f32[6400]{0}) tuple(fusion.1, div)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Multiply(), op::Divide()));
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingLoopsDifferentShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      ROOT mul = f32[8,1,5,16,1,2]{5,4,3,2,1,0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      const.2 = f32[] constant(0)
      ROOT reduce = f32[1,5,1,2]{3,2,1,0} reduce(p0.2, const.2), dimensions={0,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      fusion.1 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[1,5,1,2]{3,2,1,0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (f32[8,1,5,16,1,2]{5,4,3,2,1,0}, f32[1,5,1,2]{3,2,1,0}) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionSiblingLoopAndMultiOutputLoop) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      mul = f32[8,1,5,16,1,1]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      exp = f32[8,1,5,16,1,1]{5,4,3,2,1,0} exponential(p0.1)
      ROOT tuple = (f32[8,1,5,16,1,1]{5,4,3,2,1,0},
        f32[8,1,5,16,1,1]{5,4,3,2,1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      const.2 = f32[] constant(0)
      broadcast = f32[8,1,5,16,1,1]{5,4,3,2,1,0} broadcast(const.2),
        dimensions={}
      ROOT add = f32[8,1,5,16,1,1]{5,4,3,2,1,0} add(p0.2, broadcast)
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      fusion.1 = (f32[8,1,5,16,1,1]{5,4,3,2,1,0},
        f32[8,1,5,16,1,1]{5,4,3,2,1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      fusion.2 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      gte0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=1
      ROOT root = (f32[8,1,5,16,1,1]{5,4,3,2,1,0},
        f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0})
        tuple(gte0, gte1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Multiply(), op::Exp(), op::Add()));
}

TEST_F(MultiOutputFusionTest,
       MultiOutputFusionSiblingLoopAndMultiOutputLoopDifferentShapes) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      mul = f32[8,1,5,16,1,2]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      exp = f32[8,1,5,16,1,2]{5,4,3,2,1,0} exponential(p0.1)
      ROOT tuple = (f32[8,1,5,16,1,2]{5,4,3,2,1,0},
        f32[8,1,5,16,1,2]{5,4,3,2,1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      const.2 = f32[] constant(0)
      ROOT reduce = f32[1,5,1,2]{3,2,1,0} reduce(p0.2, const.2),
        dimensions={0,3}, to_apply=scalar_add_computation
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} parameter(0)
      fusion.1 = (f32[8,1,5,16,1,2]{5,4,3,2,1,0},
        f32[8,1,5,16,1,2]{5,4,3,2,1,0}) fusion(p0), kind=kLoop,
        calls=fused_computation_1
      fusion.2 = f32[1,5,1,2]{3,2,1,0} fusion(p0), kind=kLoop,
        calls=fused_computation_2
      gte0 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[8,1,5,16,1,2]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=1
      ROOT root = (f32[8,1,5,16,1,2]{5,4,3,2,1,0},
        f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[1,5,1,2]{3,2,1,0})
        tuple(gte0, gte1, fusion.2)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionElementwiseAndReduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      exp = f32[32,32,32]{2,1,0} exponential(p0)
      reduce = f32[32,32]{1,0} reduce(exp, c0), dimensions={2},
        to_apply=scalar_add_computation
      ROOT root = (f32[32,32]{1,0}, f32[32,32,32]{2,1,0}) tuple(reduce, exp)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement()));
  const HloInstruction* fusion = root->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Exp()));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionLoopFusionAndReduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0.1 = f32[32,32,32]{2,1,0} parameter(0)
      p1.1 = f32[32,32,32]{2,1,0} parameter(1)
      ROOT add = f32[32,32,32]{2,1,0} add(p0.1, p1.1)
    }

    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      add = f32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_add
      reduce = f32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
      ROOT root = (f32[32,32]{1,0}, f32[32,32,32]{2,1,0}) tuple(reduce, add)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement()));
  const HloInstruction* fusion = root->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Add()));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionLoopFusionAndReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_select {
      p1.1 = f32[32,32,32]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      broadcast = f32[32,32,32]{2,1,0} broadcast(f32[] c0), dimensions={}
      greater-than = pred[32,32,32]{2,1,0} compare(f32[32,32,32]{2,1,0} p1.1,
        f32[32,32,32]{2,1,0} broadcast), direction=GT
      p0.1 = f32[32,32,32]{2,1,0} parameter(0)
      ROOT select = f32[32,32,32]{2,1,0} select(pred[32,32,32]{2,1,0}
        greater-than, f32[32,32,32]{2,1,0} p0.1, f32[32,32,32]{2,1,0} broadcast)
    }

    fused_reduce {
      p0.2 = f32[32,32,32]{2,1,0} parameter(0)
      c1 = f32[] constant(0)
      r1 = f32[32,32]{1,0} reduce(p0.2, c1), dimensions={2},
        to_apply=scalar_add_computation
      mul = f32[32,32,32]{2,1,0} multiply(p0.2, p0.2)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={2},
        to_apply=scalar_add_computation
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      select = f32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(select), kind=kInput,
        calls=fused_reduce
      gte0 = f32[32,32]{1,0} get-tuple-element(fusion), index=0
      gte1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
      ROOT root = (f32[32,32]{1,0}, f32[32,32]{1,0}, f32[32,32,32]{2,1,0})
        tuple(gte1, gte1, select)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                              op::GetTupleElement()));
  const HloInstruction* fusion = root->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce(), op::Select()));
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionDoNotFuseLoopReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_element_wise {
      p0.1 = f32[2,2,2]{2,1,0} parameter(0)
      p1.1 = f32[2,2,2]{2,1,0} parameter(1)
      ROOT root = f32[2,2,2]{2,1,0} add(p0.1, p1.1)
    }

    fused_reduce {
      p0.2 = f32[2,2,2]{2,1,0} parameter(0)
      mul = f32[2,2,2]{2,1,0} multiply(f32[2,2,2]{2,1,0} p0.2,
        f32[2,2,2]{2,1,0} p0.2)
      broadcast = f32[2,2,2,2]{3,2,1,0} broadcast(mul), dimensions={3,2,1}
      c1 = f32[] constant(0)
      ROOT reduce = f32[2,2]{1,0} reduce(f32[2,2,2,2]{3,2,1,0} broadcast,
        f32[] c1), dimensions={1,3}, to_apply=scalar_add_computation
    }

    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{2,1,0} parameter(1)
      element_wise = f32[2,2,2]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_element_wise
      fusion = f32[2,2]{1,0} fusion(element_wise), kind=kLoop, calls=fused_reduce
      ROOT root = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(fusion, element_wise)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest,
       ProducerConsumerFusionFp16LoopFusionAndReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_select {
      p1.1 = f16[32,32,32]{2,1,0} parameter(1)
      c0 = f16[] constant(0)
      broadcast = f16[32,32,32]{2,1,0} broadcast(f16[] c0), dimensions={}
      greater-than = pred[32,32,32]{2,1,0} compare(f16[32,32,32]{2,1,0} p1.1,
        f16[32,32,32]{2,1,0} broadcast), direction=GT
      p0.1 = f16[32,32,32]{2,1,0} parameter(0)
      ROOT select = f16[32,32,32]{2,1,0} select(pred[32,32,32]{2,1,0}
        greater-than, f16[32,32,32]{2,1,0} p0.1, f16[32,32,32]{2,1,0} broadcast)
    }
    fused_reduce {
      p0.2 = f16[32,32,32]{2,1,0} parameter(0)
      convert = f32[32,32,32]{2,1,0} convert(p0.2)
      c1 = f32[] constant(0)
      r1 = f32[32,32]{1,0} reduce(convert, c1), dimensions={2},
        to_apply=scalar_add_computation
      mul = f32[32,32,32]{2,1,0} multiply(convert, convert)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={2},
        to_apply=scalar_add_computation
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }
    ENTRY reduce {
      p0 = f16[32,32,32]{2,1,0} parameter(0)
      p1 = f16[32,32,32]{2,1,0} parameter(1)
      select = f16[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(select), kind=kInput,
        calls=fused_reduce
      gte0 = f32[32,32]{1,0} get-tuple-element(fusion), index=0
      gte1 = f32[32,32]{1,0} get-tuple-element(fusion), index=1
      ROOT root = (f32[32,32]{1,0}, f32[32,32]{1,0}, f16[32,32,32]{2,1,0})
        tuple(gte1, gte1, select)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::GetTupleElement(), op::GetTupleElement(),
                              op::GetTupleElement()));
  const HloInstruction* fusion = root->operand(0)->operand(0);
  ASSERT_TRUE(fusion->IsMultiOutputFusion());
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Reduce(), op::Reduce(), op::Select()));
}

TEST_F(MultiOutputFusionTest,
       ProducerConsumerFusionReduceUnfriendlyLoopFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    mixed_input_layouts_computation {
      p0.1 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      p1.1 = f16[128,1024,32,32]{3,2,1,0} parameter(1)
      copy = f16[128,1024,32,32]{1,3,2,0} copy(p1.1)
      c0 = f16[] constant(0)
      broadcast = f16[128,1024,32,32]{1,3,2,0} broadcast(c0), dimensions={}
      greater-than = pred[128,1024,32,32]{1,3,2,0} compare(copy, broadcast), direction=GT
      ROOT root = f16[128,1024,32,32]{1,3,2,0} select(greater-than, p0.1, broadcast)
    }
    fused_reduce {
      p0.2 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      convert = f32[128,1024,32,32]{1,3,2,0} convert(p0.2)
      c0.2 = f32[] constant(0)
      ROOT reduce = f32[1024]{0} reduce(convert, c0.2), dimensions={0,2,3}, to_apply=scalar_add_computation
    }
    ENTRY reduce {
      p0 = f16[128,1024,32,32]{3,2,1,0} parameter(0)
      p1 = f16[128,1024,32,32]{1,3,2,0} parameter(1)
      loop_fusion = f16[128,1024,32,32]{1,3,2,0} fusion(p0, p1), kind=kLoop, calls=mixed_input_layouts_computation
      reduce_fusion = f32[1024]{0} fusion(loop_fusion), kind=kInput, calls=fused_reduce
      ROOT root = (f32[1024]{0}, f16[128,1024,32,32]{1,3,2,0}) tuple(reduce_fusion, loop_fusion)
    })"))
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

TEST_F(MultiOutputFusionTest, ProducerConsumerFusionAvoidsCycles) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      ROOT add = f32[32,32,32]{2,1,0} add(p0, p1)
    }

    fused_mul {
      p2 = f32[64,64,64]{2,1,0} parameter(0)
      p3 = f32[64,64,64]{2,1,0} parameter(1)
      ROOT multiply = f32[64,64,64]{2,1,0} multiply(p2, p3)
    }

    fused_reduce_1 {
      p4 = f32[32,32,32]{2,1,0} parameter(0)
      p5 = f32[64,64,64]{2,1,0} parameter(1)
      slice = f32[32,32,32]{2,1,0} slice(p5), slice={[0:32], [0:32], [0:32]}
      add = f32[32,32,32]{2,1,0} add(p4, slice)
      c0 = f32[] constant(0)
      ROOT r1 = f32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
    }

    fused_reduce_2 {
      p6 = f32[32,32,32]{2,1,0} parameter(0)
      p7 = f32[64,64,64]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      pad = f32[64,64,64]{2,1,0} pad(p6, c0), padding=16_16x16_16x16_16
      mul = f32[64,64,64]{2,1,0} multiply(pad, p7)
      ROOT r1 = f32[64,64]{1,0} reduce(mul, c0), dimensions={2},
        to_apply=scalar_add_computation
    }

    ENTRY reduce {
      p8 = f32[32,32,32]{2,1,0} parameter(0)
      p9 = f32[64,64,64]{2,1,0} parameter(1)
      // `add` and `mul` can be multi-output fused with `reduce1` and `reduce2`,
      // respectively. However, both isn't possible, because multi-output fusion
      // will introduce an extra dependency from `neg` to `abs` or vice versa.
      // Hence, the second multi-output fusion would introduce a cycle.
      add = f32[32,32,32]{2,1,0} fusion(p8, p8), kind=kLoop, calls=fused_add
      mul = f32[64,64,64]{2,1,0} fusion(p9, p9), kind=kLoop, calls=fused_mul

      reduce1 = f32[32,32]{1,0} fusion(add, mul), kind=kInput,
          calls=fused_reduce_1
      reduce2 = f32[64,64]{1,0} fusion(add, mul), kind=kInput,
          calls=fused_reduce_2
      ROOT root = (f32[32,32,32]{2,1,0}, f32[32,32]{1,0}, f32[64,64]{1,0},
                   f32[64,64,64]{2,1,0}) tuple(add, reduce1, reduce2, mul)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  EXPECT_EQ(1, CountMultiOutputFusions(module.get()));
}

TEST_F(MultiOutputFusionTest, PreferFuseProducerIntoFusionConsumer) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      ROOT add = f32[32,32,32]{2,1,0} add(p0, p1)
    }
    fused_reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[64,64,64]{2,1,0} parameter(1)
      slice = f32[32,32,32]{2,1,0} slice(p1), slice={[0:32], [0:32], [0:32]}
      add = f32[32,32,32]{2,1,0} add(p0, slice)
      c0 = f32[] constant(0)
      ROOT r1 = f32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
    }
    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[64,64,64]{2,1,0} parameter(1)
      add = f32[32,32,32]{2,1,0} fusion(p0, p0), kind=kLoop, calls=fused_add
      c0 = f32[] constant(0)
      reduce2 = f32[32,32]{1,0} reduce(add, c0), dimensions={2},
        to_apply=scalar_add_computation
      reduce = f32[32,32]{1,0} fusion(add, p1), kind=kInput, calls=fused_reduce
      ROOT root = (f32[32,32,32]{2,1,0}, f32[32,32]{1,0}, f32[32,32]{1,0})
                  tuple(add, reduce, reduce2)
    })"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  int multi_output_fusion_count = 0;
  for (auto* computation : module->MakeNonfusionComputations()) {
    for (auto* instr : computation->instructions()) {
      if (instr->IsMultiOutputFusion()) {
        multi_output_fusion_count++;
      }
    }
  }
  EXPECT_EQ(1, multi_output_fusion_count);
}

// Check that we limit the number of operands to fusions we create.
TEST_F(MultiOutputFusionTest, AvoidsLargeFusion) {
  constexpr int64 kNumParams = 200;
  ASSERT_GT(kNumParams, kMaxOperandsAndOutputsPerFusion);

  // Compute
  //   p0 * p1,
  //   p0 * p1 + p1 * p2
  //   p0 * p1 + p1 * p2 + p2 * p3
  //   ...
  // where each of the (pi * pj)'s is represented as a fusion node so that
  // multi-output fusion will pay attention to it.
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {10, 100});

  std::vector<HloInstruction*> params;
  for (int64 i = 0; i < kNumParams; ++i) {
    params.push_back(
        b.AddInstruction(HloInstruction::CreateParameter(i, shape, "p")));
  }

  // Creates a fusion node that calculates x*y.
  auto make_fusion = [&](HloInstruction* x, HloInstruction* y) {
    HloComputation::Builder sub_builder("subcomp");
    auto* p0 = sub_builder.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "p"));
    auto* p1 = sub_builder.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "p"));
    sub_builder.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kMultiply, p0, p1));
    HloComputation* subcomp =
        module->AddEmbeddedComputation(sub_builder.Build());
    return HloInstruction::CreateFusion(
        shape, HloInstruction::FusionKind::kLoop, {x, y}, subcomp);
  };

  auto* sum = b.AddInstruction(make_fusion(params[0], params[1]));
  for (int64 i = 2; i < kNumParams; ++i) {
    sum = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, sum,
        b.AddInstruction(make_fusion(params[i - 1], params[i]))));
  }
  auto computation = module->AddEntryComputation(b.Build());
  EXPECT_TRUE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
  SCOPED_TRACE(module->ToString());
  for (const HloInstruction* instr : computation->instructions()) {
    EXPECT_LE(instr->operand_count() + ShapeUtil::SubshapeCount(instr->shape()),
              kMaxOperandsAndOutputsPerFusion)
        << instr->ToString();
  }
}

TEST_F(MultiOutputFusionTest, MultiOutputFusionDUS) {
  auto module = ParseAndReturnVerifiedModule(R"(HloModule dus_mof
    fusion.1 {
      p.0 = f16[50,96,1024]{2,1,0} parameter(0)
      p.1 = f16[1,96,1024]{2,1,0} parameter(1)
      c.0 = s32[3]{0} constant({0, 0, 0})
      ROOT %dynamic-update-slice = f16[50,96,1024]{2,1,0} dynamic-update-slice(p.0, p.1, c.0)
    }

    fusion.2 {
      p.0 = f16[50,96,1024]{2,1,0} parameter(0)
      p.1 = f16[1,96,1024]{2,1,0} parameter(1)
      c.0 = s32[3]{0} constant({0, 0, 0})
      ROOT %dynamic-update-slice = f16[50,96,1024]{2,1,0} dynamic-update-slice(p.0, p.1, c.0)
    }

    ENTRY entry {
      p.00 = f16[50,96,1024]{2,1,0} parameter(0)
      p.01 = f16[50,96,1024]{2,1,0} parameter(1)
      p.1 = f16[1,96,1024]{2,1,0} parameter(2)

      f1 = f16[50,96,1024] fusion(p.00, p.1), kind=kLoop, calls=fusion.1
      f2 = f16[50,96,1024] fusion(p.01, p.1), kind=kLoop, calls=fusion.2
      ROOT tuple = (f16[50,96,1024],f16[50,96,1024]) tuple(f1, f2)
    })")
                    .ValueOrDie();
  ASSERT_FALSE(GpuMultiOutputFusion().Run(module.get()).ValueOrDie());
}

// Check that we don't fuse too many reductions together.
TEST_F(MultiOutputFusionTest, SharedMemoryBudget) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY computation {
      zero = f32[] constant(0)
      param0 = f32[64,64] parameter(0)
      param1 = f32[64,64] parameter(1)
      param2 = f32[64,64] parameter(2)
      param3 = f32[64,64] parameter(3)
      param4 = f32[64,64] parameter(4)
      param5 = f32[64,64] parameter(5)
      param6 = f32[64,64] parameter(6)
      param7 = f32[64,64] parameter(7)
      param8 = f32[64,64] parameter(8)
      param9 = f32[64,64] parameter(9)
      add0 = f32[64,64] add(param0, param1)
      add1 = f32[64,64] add(param1, param2)
      add2 = f32[64,64] add(param2, param3)
      add3 = f32[64,64] add(param3, param4)
      add4 = f32[64,64] add(param4, param5)
      add5 = f32[64,64] add(param5, param6)
      add6 = f32[64,64] add(param6, param7)
      add7 = f32[64,64] add(param7, param8)
      add8 = f32[64,64] add(param8, param9)
      add9 = f32[64,64] add(param9, param0)
      out0 = f32[64] reduce(f32[64,64] add0, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      out1 = f32[64] reduce(f32[64,64] add1, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      out2 = f32[64] reduce(f32[64,64] add2, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      out3 = f32[64] reduce(f32[64,64] add3, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      out4 = f32[64] reduce(f32[64,64] add4, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      out5 = f32[64] reduce(f32[64,64] add5, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      out6 = f32[64] reduce(f32[64,64] add6, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      out7 = f32[64] reduce(f32[64,64] add7, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      out8 = f32[64] reduce(f32[64,64] add8, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      out9 = f32[64] reduce(f32[64,64] add9, f32[] zero), dimensions={0},
        to_apply=scalar_add_computation
      ROOT out = (f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64], f32[64]) tuple(f32[64] out0, f32[64] out1, f32[64] out2, f32[64] out3, f32[64] out4, f32[64] out5, f32[64] out6, f32[64] out7, f32[64] out8, f32[64] out9)
    }
  )"))
                    .ValueOrDie();
  ASSERT_TRUE(GpuMultiOutputFusion().Run(module.get()).ConsumeValueOrDie());

  EXPECT_EQ(2, CountMultiOutputFusions(module.get()));
}

}  // namespace gpu
}  // namespace xla
