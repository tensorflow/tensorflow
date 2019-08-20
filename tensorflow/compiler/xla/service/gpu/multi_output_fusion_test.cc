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
#include "tensorflow/compiler/xla/service/gpu/fusion_merger.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/variadic_op_splitter.h"
#include "tensorflow/compiler/xla/service/hlo_cse.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/service/hlo_pass_fix.h"
#include "tensorflow/compiler/xla/service/hlo_pass_pipeline.h"
#include "tensorflow/compiler/xla/service/layout_assignment.h"
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

// This test a case where we want to postpone some fusion up to the
// MOF fusion.
TEST_F(MultiOutputFusionTest, MultiOutputFusionPostpone) {
  auto module = ParseAndReturnVerifiedModule(R"(HloModule postpone_mof
%scalar_ge_of_first32bits_computation (scalar_lhs: u64[], scalar_rhs: u64[]) -> u64[] {
  %scalar_lhs = u64[] parameter(0)
  %convert.2 = u16[] convert(u64[] %scalar_lhs)
  %bitcast-convert.2 = f16[] bitcast-convert(u16[] %convert.2)
  %scalar_rhs = u64[] parameter(1)
  %convert.3 = u16[] convert(u64[] %scalar_rhs)
  %bitcast-convert.3 = f16[] bitcast-convert(u16[] %convert.3)
  %compare = pred[] compare(f16[] %bitcast-convert.2, f16[] %bitcast-convert.3), direction=GE
  ROOT %select = u64[] select(pred[] %compare, u64[] %scalar_lhs, u64[] %scalar_rhs)
}

%write (scalar_lhs.1: f16[], scalar_rhs.1: f16[]) -> f16[] {
  %scalar_lhs.1 = f16[] parameter(0)
  ROOT %scalar_rhs.1 = f16[] parameter(1)
}

ENTRY %computation.49 (arg0.1: f16[128,64,112,112]) -> f16[128,64,112,112] {
  %constant.360 = f16[] constant(0)
  %broadcast.6 = f16[102760448]{0} broadcast(f16[] %constant.360), dimensions={}
  %arg0.1 = f16[128,64,112,112]{3,2,1,0} parameter(0), parameter_replication={false}
  %bitcast-convert = u16[128,64,112,112]{3,2,1,0} bitcast-convert(f16[128,64,112,112]{3,2,1,0} %arg0.1)
  %convert = u64[128,64,112,112]{3,2,1,0} convert(u16[128,64,112,112]{3,2,1,0} %bitcast-convert)
  %iota = u64[102760448]{0} iota(), iota_dimension=0
  %constant = u64[] constant(32)
  %broadcast.9 = u64[102760448]{0} broadcast(u64[] %constant), dimensions={}
  %shift-left.1 = u64[102760448]{0} shift-left(u64[102760448]{0} %iota, u64[102760448]{0} %broadcast.9)
  %bitcast = u64[128,64,112,112]{3,2,1,0} bitcast(u64[102760448]{0} %shift-left.1)
  %add = u64[128,64,112,112]{3,2,1,0} add(u64[128,64,112,112]{3,2,1,0} %convert, u64[128,64,112,112]{3,2,1,0} %bitcast)
  %constant.8 = u64[] constant(64512)
  %reduce-window = u64[128,64,56,56]{3,2,1,0} reduce-window(u64[128,64,112,112]{3,2,1,0} %add, u64[] %constant.8), window={size=1x1x2x2 stride=1x1x2x2}, to_apply=%scalar_ge_of_first32bits_computation
  %broadcast.1 = u64[128,64,56,56]{3,2,1,0} broadcast(u64[] %constant), dimensions={}
  %shift-right-logical = u64[128,64,56,56]{3,2,1,0} shift-right-logical(u64[128,64,56,56]{3,2,1,0} %reduce-window, u64[128,64,56,56]{3,2,1,0} %broadcast.1)
  %convert.4 = u32[128,64,56,56]{3,2,1,0} convert(u64[128,64,56,56]{3,2,1,0} %shift-right-logical)
  %iota.1 = u32[128,64,56,56]{3,2,1,0} iota(), iota_dimension=0
  %constant.1 = u32[] constant(802816)
  %broadcast.2 = u32[128,64,56,56]{3,2,1,0} broadcast(u32[] %constant.1), dimensions={}
  %multiply = u32[128,64,56,56]{3,2,1,0} multiply(u32[128,64,56,56]{3,2,1,0} %iota.1, u32[128,64,56,56]{3,2,1,0} %broadcast.2)
  %subtract = u32[128,64,56,56]{3,2,1,0} subtract(u32[128,64,56,56]{3,2,1,0} %convert.4, u32[128,64,56,56]{3,2,1,0} %multiply)
  %iota.2 = u32[128,64,56,56]{3,2,1,0} iota(), iota_dimension=1
  %constant.2 = u32[] constant(12544)
  %broadcast.3 = u32[128,64,56,56]{3,2,1,0} broadcast(u32[] %constant.2), dimensions={}
  %multiply.1 = u32[128,64,56,56]{3,2,1,0} multiply(u32[128,64,56,56]{3,2,1,0} %iota.2, u32[128,64,56,56]{3,2,1,0} %broadcast.3)
  %subtract.1 = u32[128,64,56,56]{3,2,1,0} subtract(u32[128,64,56,56]{3,2,1,0} %subtract, u32[128,64,56,56]{3,2,1,0} %multiply.1)
  %convert.6 = u16[128,64,56,56]{3,2,1,0} convert(u32[128,64,56,56]{3,2,1,0} %subtract.1)
  %constant.3 = u16[] constant(112)
  %broadcast.4 = u16[128,64,56,56]{3,2,1,0} broadcast(u16[] %constant.3), dimensions={}
  %divide = u16[128,64,56,56]{3,2,1,0} divide(u16[128,64,56,56]{3,2,1,0} %convert.6, u16[128,64,56,56]{3,2,1,0} %broadcast.4)
  %iota.3 = u16[128,64,56,56]{3,2,1,0} iota(), iota_dimension=2
  %subtract.2 = u16[128,64,56,56]{3,2,1,0} subtract(u16[128,64,56,56]{3,2,1,0} %divide, u16[128,64,56,56]{3,2,1,0} %iota.3)
  %constant.4 = u16[] constant(2)
  %broadcast.5 = u16[128,64,56,56]{3,2,1,0} broadcast(u16[] %constant.4), dimensions={}
  %multiply.2 = u16[128,64,56,56]{3,2,1,0} multiply(u16[128,64,56,56]{3,2,1,0} %subtract.2, u16[128,64,56,56]{3,2,1,0} %broadcast.5)
  %remainder = u16[128,64,56,56]{3,2,1,0} remainder(u16[128,64,56,56]{3,2,1,0} %convert.6, u16[128,64,56,56]{3,2,1,0} %broadcast.4)
  %iota.4 = u16[128,64,56,56]{3,2,1,0} iota(), iota_dimension=3
  %subtract.3 = u16[128,64,56,56]{3,2,1,0} subtract(u16[128,64,56,56]{3,2,1,0} %remainder, u16[128,64,56,56]{3,2,1,0} %iota.4)
  %add.1 = u16[128,64,56,56]{3,2,1,0} add(u16[128,64,56,56]{3,2,1,0} %multiply.2, u16[128,64,56,56]{3,2,1,0} %subtract.3)
  %convert.7 = u8[128,64,56,56]{3,2,1,0} convert(u16[128,64,56,56]{3,2,1,0} %add.1)
  %convert.8 = u16[128,64,56,56]{3,2,1,0} convert(u8[128,64,56,56]{3,2,1,0} %convert.7)
  %constant.5 = u16[] constant(1)
  %broadcast.7 = u16[128,64,56,56]{3,2,1,0} broadcast(u16[] %constant.5), dimensions={}
  %shift-right-logical.1 = u16[128,64,56,56]{3,2,1,0} shift-right-logical(u16[128,64,56,56]{3,2,1,0} %convert.8, u16[128,64,56,56]{3,2,1,0} %broadcast.7)
  %add.2 = u16[128,64,56,56]{3,2,1,0} add(u16[128,64,56,56]{3,2,1,0} %shift-right-logical.1, u16[128,64,56,56]{3,2,1,0} %iota.3)
  %multiply.3 = u16[128,64,56,56]{3,2,1,0} multiply(u16[128,64,56,56]{3,2,1,0} %add.2, u16[128,64,56,56]{3,2,1,0} %broadcast.4)
  %add.3 = u16[128,64,56,56]{3,2,1,0} add(u16[128,64,56,56]{3,2,1,0} %multiply.3, u16[128,64,56,56]{3,2,1,0} %iota.4)
  %and = u16[128,64,56,56]{3,2,1,0} and(u16[128,64,56,56]{3,2,1,0} %convert.8, u16[128,64,56,56]{3,2,1,0} %broadcast.7)
  %add.4 = u16[128,64,56,56]{3,2,1,0} add(u16[128,64,56,56]{3,2,1,0} %add.3, u16[128,64,56,56]{3,2,1,0} %and)
  %convert.9 = u32[128,64,56,56]{3,2,1,0} convert(u16[128,64,56,56]{3,2,1,0} %add.4)
  %add.5 = u32[128,64,56,56]{3,2,1,0} add(u32[128,64,56,56]{3,2,1,0} %convert.9, u32[128,64,56,56]{3,2,1,0} %multiply)
  %add.6 = u32[128,64,56,56]{3,2,1,0} add(u32[128,64,56,56]{3,2,1,0} %add.5, u32[128,64,56,56]{3,2,1,0} %multiply.1)
  %bitcast.1 = u32[25690112]{0} bitcast(u32[128,64,56,56]{3,2,1,0} %add.6)
  %convert.5 = u16[128,64,56,56]{3,2,1,0} convert(u64[128,64,56,56]{3,2,1,0} %reduce-window)
  %bitcast-convert.4 = f16[128,64,56,56]{3,2,1,0} bitcast-convert(u16[128,64,56,56]{3,2,1,0} %convert.5)
  %bitcast.2 = f16[25690112]{0} bitcast(f16[128,64,56,56]{3,2,1,0} %bitcast-convert.4)
  %scatter = f16[102760448]{0} scatter(f16[102760448]{0} %broadcast.6, u32[25690112]{0} %bitcast.1, f16[25690112]{0} %bitcast.2), update_window_dims={}, inserted_window_dims={0}, scatter_dims_to_operand_dims={0}, index_vector_dim=1, to_apply=%write
  ROOT %bitcast.3 = f16[128,64,112,112]{3,2,1,0} bitcast(f16[102760448]{0} %scatter)
})")
                    .ValueOrDie();
  // Same passes as in gpu_compiler.cc
  HloPassFix<HloPassPipeline> fusion("fusion");
  // HloPassPipeline fusion("fusion");

  // We try to split variadic ops with many parameters into several such ops
  // to avoid exceeding the parameter space.
  fusion.AddPass<VariadicOpSplitter>();
  /* TODO(b/117531509): Use LayoutAssignment::InstructionCanChangeLayout after
   * fixing the ticket. */
  fusion.AddInvariantChecker<HloVerifier>(
      /*layout_sensitive=*/true,
      /*allow_mixed_precision=*/false,
      LayoutAssignment::InstructionCanChangeLayout);
  fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/false);
  fusion.AddPass<GpuInstructionFusion>(/*may_duplicate=*/true);
  fusion.AddPass<FusionMerger>();
  fusion.AddPass<GpuMultiOutputFusion>();
  fusion.AddPass<HloCSE>(/*is_layout_sensitive=*/true,
                         /*only_fusion_computations=*/true);
  fusion.AddPass<HloDCE>();

  ASSERT_TRUE(fusion.Run(module.get()).ValueOrDie());
  // fusion.Run(module.get());
  ASSERT_TRUE(module.get()->entry_computation() != nullptr);
  auto computation = module.get()->entry_computation();
  printf("!!!!!%s\n", module.get()->ToString().c_str());
  bool found_ifusion = false;
  bool found_lfusion = false;
  for (auto instr : computation->instructions()) {
    if (instr->IsLoopFusion()) {
      ASSERT_TRUE(instr->shape().IsTuple());
      ASSERT_EQ(instr->shape().tuple_shapes(0).element_type(), U8);
      ASSERT_EQ(instr->shape().tuple_shapes(1).element_type(), U16);
      found_lfusion = true;
    } else if (instr->IsInputFusion()) {
      ASSERT_EQ(instr->shape().element_type(), F16);
      found_ifusion = true;
    }
  }
  ASSERT_EQ(found_ifusion, true);
  ASSERT_EQ(found_lfusion, true);
}

}  // namespace gpu
}  // namespace xla
