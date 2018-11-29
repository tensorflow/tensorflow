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

#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"

#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"

namespace xla {
namespace gpu {

using GpuFusibleTest = HloTestBase;

const char kModulePrefix[] = R"(
    HloModule test_module
    scalar_add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    })";

TEST_F(GpuFusibleTest,
       LayoutsAreReduceInputFusionFriendly_ElementwiseProducer) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      exp = f32[2,2,2]{2,1,0} exponential(p0)
      ROOT reduce = f32[2,2]{1,0} reduce(exp, c0), dimensions={2}, to_apply=scalar_add
    })"))
                    .ValueOrDie();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kReduce);
  const HloInstruction* exp =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(exp->opcode(), HloOpcode::kExp);
  EXPECT_TRUE(LayoutsAreReduceInputFusionFriendly(*exp, *reduce));
}

TEST_F(GpuFusibleTest,
       LayoutsAreReduceInputFusionFriendly_MixedLayoutProducer) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    mixed_input_layouts_computation {
      p0.1 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      p1.1 = f16[128,1024,32,32]{3,2,1,0} parameter(1)
      copy = f16[128,1024,32,32]{1,3,2,0} copy(p1.1)
      c0 = f16[] constant(0)
      broadcast = f16[128,1024,32,32]{1,3,2,0} broadcast(c0), dimensions={}
      greater-than = pred[128,1024,32,32]{1,3,2,0} greater-than(copy, broadcast)
      ROOT root = f16[128,1024,32,32]{1,3,2,0} select(greater-than, p0.1, broadcast)
    }
    fused_reduce {
      p0.2 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      convert = f32[128,1024,32,32]{1,3,2,0} convert(p0.2)
      c0.2 = f32[] constant(0)
      ROOT reduce = f32[1024]{0} reduce(convert, c0.2), dimensions={0,2,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      p1 = f16[128,1024,32,32]{3,2,1,0} parameter(1)
      loop_fusion = f16[128,1024,32,32]{1,3,2,0} fusion(p0, p1), kind=kLoop, calls=mixed_input_layouts_computation
      reduce_fusion = f32[1024]{0} fusion(loop_fusion), kind=kInput, calls=fused_reduce
      ROOT root = (f32[1024]{0}, f16[128,1024,32,32]{1,3,2,0}) tuple(reduce_fusion, loop_fusion)
    })"))
                    .ValueOrDie();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* reduce_fusion =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(reduce_fusion->fused_expression_root()->opcode(),
            HloOpcode::kReduce);
  const HloInstruction* loop_fusion =
      module->entry_computation()->root_instruction()->operand(1);
  ASSERT_EQ(loop_fusion->fused_expression_root()->opcode(), HloOpcode::kSelect);
  EXPECT_FALSE(
      LayoutsAreReduceInputFusionFriendly(*loop_fusion, *reduce_fusion));
}

TEST_F(GpuFusibleTest, LayoutsAreReduceInputFusionFriendly_CopyProducer) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_reduce {
      p0.1 = f32[128,1024,32,32]{1,3,2,0} parameter(0)
      c0.1 = f32[] constant(0)
      ROOT reduce = f32[1024]{0} reduce(p0.1, c0.1), dimensions={0,2,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f16[128,1024,32,32]{3,2,1,0} parameter(0)
      copy = f32[128,1024,32,32]{1,3,2,0} copy(p0)
      ROOT reduce_fusion = f32[1024]{0} fusion(copy), kind=kInput, calls=fused_reduce
    })"))
                    .ValueOrDie();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->fused_expression_root()->opcode(), HloOpcode::kReduce);
  const HloInstruction* copy =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(copy->opcode(), HloOpcode::kCopy);
  EXPECT_FALSE(LayoutsAreReduceInputFusionFriendly(*copy, *reduce));
}

TEST_F(GpuFusibleTest,
       LayoutsAreReduceInputFusionFriendly_LayoutChangingFusionProducer) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    layout_changing_computation {
      p0.1 = f16[128,1024,32,32]{3,2,1,0} parameter(0)
      p1.1 = f16[128,1024,32,32]{3,2,1,0} parameter(1)
      c0 = f16[] constant(0)
      broadcast = f16[128,1024,32,32]{3,2,1,0} broadcast(c0), dimensions={}
      greater-than = pred[128,1024,32,32]{3,2,1,0} greater-than(p1.1, broadcast)
      select = f16[128,1024,32,32]{3,2,1,0} select(greater-than, p0.1, broadcast)
      ROOT root = f16[128,1024,32,32]{1,3,2,0} copy(select)
    }
    fused_reduce {
      p0.2 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      convert = f32[128,1024,32,32]{1,3,2,0} convert(p0.2)
      c0.2 = f32[] constant(0)
      ROOT reduce = f32[1024]{0} reduce(convert, c0.2), dimensions={0,2,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f16[128,1024,32,32]{3,2,1,0} parameter(0)
      p1 = f16[128,1024,32,32]{3,2,1,0} parameter(1)
      loop_fusion = f16[128,1024,32,32]{1,3,2,0} fusion(p0, p1), kind=kLoop, calls=layout_changing_computation
      ROOT reduce_fusion = f32[1024]{0} fusion(loop_fusion), kind=kInput, calls=fused_reduce
    })"))
                    .ValueOrDie();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* reduce_fusion =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce_fusion->fused_expression_root()->opcode(),
            HloOpcode::kReduce);
  const HloInstruction* loop_fusion =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(loop_fusion->fused_expression_root()->opcode(), HloOpcode::kCopy);
  EXPECT_FALSE(
      LayoutsAreReduceInputFusionFriendly(*loop_fusion, *reduce_fusion));
}

TEST_F(GpuFusibleTest,
       LayoutsAreReduceInputFusionFriendly_ConsiderMaximumRanksParamsOnly) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    broadcasting_computation {
      p0.1 = f32[128,1024,32,32]{1,3,2,0} parameter(0)
      p1.1 = f32[128]{0} parameter(1)
      broadcast = f32[128,1024,32,32]{1,3,2,0} broadcast(p1.1), dimensions={0}
      ROOT add = f32[128,1024,32,32]{1,3,2,0} add(p0.1, broadcast)
    }
    ENTRY entry {
      p0 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      p1 = f16[128]{0} parameter(1)
      loop_fusion = f32[128,1024,32,32]{1,3,2,0} fusion(p0, p1), kind=kLoop, calls=broadcasting_computation
      c0.2 = f32[] constant(0)
      ROOT reduce = f32[128,1024]{0,1} reduce(loop_fusion, c0.2), dimensions={0,2,3}, to_apply=scalar_add
    })"))
                    .ValueOrDie();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kReduce);
  const HloInstruction* loop_fusion =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(loop_fusion->fused_expression_root()->opcode(), HloOpcode::kAdd);
  EXPECT_TRUE(LayoutsAreReduceInputFusionFriendly(*loop_fusion, *reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_ReductionToVector) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      c0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      // Reduction-to-vector lowered by IrEmitterUnnested.
      ROOT reduce = f32[512]{0} reduce(p1, c0), dimensions={0,2,3}, to_apply=scalar_add
    })"))
                    .ValueOrDie();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kReduce);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_TRUE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_ElementalReduction) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      c0 = f32[] parameter(0)
      p1 = f32[8,512,5,16,1,1]{5,4,3,2,1,0} parameter(1)
      // Reduction lowered by GpuElementalIrEmitter.
      ROOT reduce = f32[8,512,5,1,1]{4,3,2,1,0} reduce(p1, c0), dimensions={3}, to_apply=scalar_add
    })"))
                    .ValueOrDie();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kReduce);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_FALSE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_SingleOutputInputReduceFusion) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      ROOT reduce = f32[128,512]{1,0} reduce(p1, c0), dimensions={2,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = f32[128,512]{1,0} fusion(p0), kind=kInput, calls=fused_reduction
    })"))
                    .ValueOrDie();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_TRUE(IsReduceInputFusion(*reduce));
  EXPECT_TRUE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_SingleOutputLoopReduceFusion) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] parameter(0)
      p1 = f32[8,512,5,16,1,1]{5,4,3,2,1,0} parameter(1)
      ROOT reduce = f32[8,5,1,1]{3,2,1,0} reduce(p1, c0), dimensions={1,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f32[8,512,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      ROOT fusion = f32[8,5,1,1]{3,2,1,0} fusion(p0), kind=kLoop, calls=fused_reduction
    })"))
                    .ValueOrDie();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_FALSE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_MultiOutputInputReduceFusion) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      reduce.0 = f32[128,512]{1,0} reduce(p1, c0), dimensions={2,3}, to_apply=scalar_add
      reduce.1 = f32[128,512]{1,0} reduce(p1, c0), dimensions={2,3}, to_apply=scalar_add
      ROOT root = (f32[128,512]{1,0}, f32[128,512]{1,0}) tuple(reduce.0, reduce.1)
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = (f32[128,512]{1,0}, f32[128,512]{1,0}) fusion(p0), kind=kInput, calls=fused_reduction
    })"))
                    .ValueOrDie();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_TRUE(IsReduceInputFusion(*reduce));
  EXPECT_TRUE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest,
       IsReduceInputFusion_MultiOutputInputReduceFusionWithExtraOutputs) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      reduce = f32[128,512]{1,0} reduce(p1, c0), dimensions={2,3}, to_apply=scalar_add
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1, p1)
      ROOT root = (f32[128,512]{1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(reduce, mul)
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = (f32[128,512]{1,0}, f32[128,512,28,28]{3,2,1,0}) fusion(p0), kind=kInput, calls=fused_reduction
    })"))
                    .ValueOrDie();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_TRUE(IsReduceInputFusion(*reduce));
  EXPECT_TRUE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_MultiOutputLoopReduceFusion) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      reduce.0 = f32[512,28]{1,0} reduce(p1, c0), dimensions={0,2}, to_apply=scalar_add
      reduce.1 = f32[512,28]{1,0} reduce(p1, c0), dimensions={0,2}, to_apply=scalar_add
      ROOT root = (f32[512,28]{1,0}, f32[512,28]{1,0}) tuple(reduce.0, reduce.1)
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = (f32[512,28]{1,0}, f32[512,28]{1,0}) fusion(p0), kind=kLoop, calls=fused_reduction
    })"))
                    .ValueOrDie();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_FALSE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest,
       IsReduceInputFusion_MultiOutputLoopFusionReduceAndElementwiseOp) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      reduce = f32[512,28]{1,0} reduce(p1, c0), dimensions={0,2}, to_apply=scalar_add
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1, p1)
      ROOT root = (f32[512,28]{1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(reduce, mul)
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = (f32[512,28]{1,0}, f32[128,512,28,28]{3,2,1,0}) fusion(p0), kind=kLoop, calls=fused_reduction
    })"))
                    .ValueOrDie();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_FALSE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_LoopFusions) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = f32[6400]{0} parameter(0)
      const.2 = f32[] constant(1)
      ROOT div = f32[6400]{0} divide(p0.2, const.2)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (f32[6400]{0}, f32[6400]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_IgnoreFpPrecision) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = f32[6400]{0} parameter(0)
      ROOT convert = f16[6400]{0} convert(p0.2)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (f32[6400]{0}, f32[6400]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .ValueOrDie();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_Reduce) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      const.2 = f32[] constant(0)
      reduce = f32[] reduce(p0, const.2), dimensions={0}, to_apply=scalar_add
      ROOT root = (f32[6400]{0}, f32[]) tuple(fusion.1, reduce)
    })"))
                    .ValueOrDie();
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion, *reduce));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_Elementwise) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      const.2 = f32[] constant(1)
      div = f32[6400]{0} divide(p0, const.2)
      ROOT root = (f32[6400]{0}, f32[6400]{0}) tuple(fusion.1, div)
    })"))
                    .ValueOrDie();
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* div =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion, *div));
}

TEST_F(GpuFusibleTest,
       ShapesCompatibleForMultiOutputFusion_MultiOutputLoopFusion) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      mul = f32[8,1,5,16,1,1]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      exp = f32[8,1,5,16,1,1]{5,4,3,2,1,0} exponential(p0.1)
      ROOT tuple = (f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      const.2 = f32[] constant(0)
      ROOT add = f32[8,1,5,16,1,1]{5,4,3,2,1,0} add(p0.2, const.2)
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      fusion.1 = (f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}) fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} fusion(p0), kind=kLoop, calls=fused_computation_2
      gte0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=1
      ROOT root = (f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}) tuple(gte0, gte1, fusion.2)
    })"))
                    .ValueOrDie();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1)->operand(0);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_UnfusedOps) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      exp = f32[2,2,2]{2,1,0} exponential(p0)
      reduce = f32[2,2]{1,0} reduce(exp, c0), dimensions={2}, to_apply=scalar_add
      ROOT root = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(reduce, exp)
    })"))
                    .ValueOrDie();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* exp =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*reduce, *exp));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_DifferentLayouts) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{0,1,2} parameter(1)
      c0 = f32[] constant(0)
      exp = f32[2,2,2]{2,1,0} exponential(p0)
      reduce = f32[2,2]{0,1} reduce(p1, c0), dimensions={2}, to_apply=scalar_add
      ROOT root = (f32[2,2]{0,1}, f32[2,2,2]{2,1,0}) tuple(reduce, exp)
    })"))
                    .ValueOrDie();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* exp =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_FALSE(ShapesCompatibleForMultiOutputFusion(*reduce, *exp));
}

TEST_F(GpuFusibleTest,
       ShapesCompatibleForMultiOutputFusion_MultiOutputReduceFusion) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_select {
      p1.1 = f32[2,2,2]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      broadcast = f32[2,2,2]{2,1,0} broadcast(f32[] c0), dimensions={}
      greater-than = pred[2,2,2]{2,1,0} greater-than(f32[2,2,2]{2,1,0} p1.1, f32[2,2,2]{2,1,0} broadcast)
      p0.1 = f32[2,2,2]{2,1,0} parameter(0)
      ROOT select = f32[2,2,2]{2,1,0} select(pred[2,2,2]{2,1,0} greater-than, f32[2,2,2]{2,1,0} p0.1, f32[2,2,2]{2,1,0} broadcast)
    }

    fused_reduce {
      p0.2 = f32[2,2,2]{2,1,0} parameter(0)
      c1 = f32[] constant(0)
      r1 = f32[2,2]{1,0} reduce(p0.2, c1), dimensions={2}, to_apply=scalar_add
      mul = f32[2,2,2]{2,1,0} multiply(p0.2, p0.2)
      r2 = f32[2,2]{1,0} reduce(mul, c1), dimensions={2}, to_apply=scalar_add
      ROOT tuple = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{2,1,0} parameter(1)
      select = f32[2,2,2]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      fusion = (f32[2,2]{1,0}, f32[2,2]{1,0}) fusion(select), kind=kInput, calls=fused_reduce
      gte0 = f32[2,2]{1,0} get-tuple-element(fusion), index=0
      gte1 = f32[2,2]{1,0} get-tuple-element(fusion), index=1
      ROOT root = (f32[2,2]{1,0}, f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(gte1, gte1, select)
    })"))
                    .ValueOrDie();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1)->operand(0);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_ReduceFusions) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_reduce_1 {
      p0.1 = f32[2,2,2]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce = f32[2,2]{1,0} reduce(f32[2,2,2]{2,1,0} p0.1, f32[] c0), dimensions={0}, to_apply=scalar_add
    }

    fused_reduce_2 {
      p0.2 = f32[2,2,2]{2,1,0} parameter(0)
      mul = f32[2,2,2]{2,1,0} multiply(f32[2,2,2]{2,1,0} p0.2, f32[2,2,2]{2,1,0} p0.2)
      c1 = f32[] constant(0)
      ROOT reduce = f32[2,2]{1,0} reduce(f32[2,2,2]{2,1,0} mul, f32[] c1), dimensions={0}, to_apply=scalar_add
    }

    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{2,1,0} parameter(1)
      reduce_1 = f32[2,2]{1,0} fusion(p0), kind=kLoop, calls=fused_reduce_1
      reduce_2 = f32[2,2]{1,0} fusion(p1), kind=kLoop, calls=fused_reduce_2
      ROOT root = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(reduce_1, reduce_2)
    })"))
                    .ValueOrDie();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest,
       ShapesCompatibleForMultiOutputFusion_NoReductionToVector) {
  auto module = ParseHloString(absl::StrCat(kModulePrefix, R"(
    fused_element_wise {
      p0.1 = f32[2,2,2]{2,1,0} parameter(0)
      p1.1 = f32[2,2,2]{2,1,0} parameter(1)
      ROOT add = f32[2,2,2]{2,1,0} add(p0.1, p1.1)
    }

    fused_reduce {
      p0.2 = f32[2,2,2]{2,1,0} parameter(0)
      mul = f32[2,2,2]{2,1,0} multiply(f32[2,2,2]{2,1,0} p0.2, f32[2,2,2]{2,1,0} p0.2)
      c1 = f32[] constant(0)
      // Note that reduce is not a reduction-to-vector.
      ROOT reduce = f32[2,2]{1,0} reduce(f32[2,2,2]{2,1,0} mul, f32[] c1), dimensions={1}, to_apply=scalar_add
    }

    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{2,1,0} parameter(1)
      element_wise = f32[2,2,2]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_element_wise
      fusion = (f32[2,2]{1,0}, f32[2,2]{1,0}) fusion(element_wise), kind=kLoop, calls=fused_reduce
      ROOT root = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(fusion, element_wise)
    })"))
                    .ValueOrDie();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_FALSE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

}  // namespace gpu
}  // namespace xla
