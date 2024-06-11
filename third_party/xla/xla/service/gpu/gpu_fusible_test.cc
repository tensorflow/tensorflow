/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/service/gpu/gpu_fusible.h"

#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_parser.h"
#include "xla/tests/hlo_test_base.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

using ::testing::ElementsAre;

using GpuFusibleTest = HloTestBase;

const char kModulePrefix[] = R"(
    HloModule test_module
    scalar_add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    })";

TEST_F(GpuFusibleTest, IsPhysicallyTransposing_ElementwiseProducer) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      exp = f32[2,2,2]{2,1,0} exponential(p0)
      ROOT reduce = f32[2,2]{1,0} reduce(exp, c0), dimensions={2}, to_apply=scalar_add
    })"))
                    .value();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* exp =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(exp->opcode(), HloOpcode::kExp);
  EXPECT_FALSE(IsPhysicallyTransposing(*exp));
}

TEST_F(GpuFusibleTest, IsPhysicallyTransposing_MixedLayoutProducer) {
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
      ROOT reduce = f32[1024]{0} reduce(convert, c0.2), dimensions={0,2,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      p1 = f16[128,1024,32,32]{3,2,1,0} parameter(1)
      loop_fusion = f16[128,1024,32,32]{1,3,2,0} fusion(p0, p1), kind=kLoop, calls=mixed_input_layouts_computation
      reduce_fusion = f32[1024]{0} fusion(loop_fusion), kind=kInput, calls=fused_reduce
      ROOT root = (f32[1024]{0}, f16[128,1024,32,32]{1,3,2,0}) tuple(reduce_fusion, loop_fusion)
    })"))
                    .value();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* loop_fusion =
      module->entry_computation()->root_instruction()->operand(1);
  ASSERT_EQ(loop_fusion->fused_expression_root()->opcode(), HloOpcode::kSelect);
  EXPECT_TRUE(IsPhysicallyTransposing(*loop_fusion));
}

TEST_F(GpuFusibleTest,
       IsPhysicallyTransposing_MixedLayoutProducerWithTrivialDim) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    mixed_input_layouts_computation {
      p0.1 = f16[128,1,32,32]{1,3,2,0} parameter(0)
      p1.1 = f16[128,1,32,32]{3,2,1,0} parameter(1)
      bitcast = f16[128,1,32,32]{1,3,2,0} bitcast(p1.1)
      c0 = f16[] constant(0)
      broadcast = f16[128,1,32,32]{1,3,2,0} broadcast(c0), dimensions={}
      greater-than = pred[128,1,32,32]{1,3,2,0} compare(bitcast, broadcast), direction=GT
      ROOT root = f16[128,1,32,32]{1,3,2,0} select(greater-than, p0.1, broadcast)
    }
    fused_reduce {
      p0.2 = f16[128,1,32,32]{1,3,2,0} parameter(0)
      convert = f32[128,1,32,32]{1,3,2,0} convert(p0.2)
      c0.2 = f32[] constant(0)
      ROOT reduce = f32[1]{0} reduce(convert, c0.2), dimensions={0,2,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f16[128,1,32,32]{1,3,2,0} parameter(0)
      p1 = f16[128,1,32,32]{3,2,1,0} parameter(1)
      loop_fusion = f16[128,1,32,32]{1,3,2,0} fusion(p0, p1), kind=kLoop, calls=mixed_input_layouts_computation
      reduce_fusion = f32[1]{0} fusion(loop_fusion), kind=kInput, calls=fused_reduce
      ROOT root = (f32[1]{0}, f16[128,1,32,32]{1,3,2,0}) tuple(reduce_fusion, loop_fusion)
    })"))
                    .value();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* loop_fusion =
      module->entry_computation()->root_instruction()->operand(1);
  ASSERT_EQ(loop_fusion->fused_expression_root()->opcode(), HloOpcode::kSelect);
  EXPECT_FALSE(IsPhysicallyTransposing(*loop_fusion));
}

TEST_F(GpuFusibleTest, IsPhysicallyTransposing_CopyProducer) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
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
                    .value();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* copy =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(copy->opcode(), HloOpcode::kCopy);
  EXPECT_TRUE(IsPhysicallyTransposing(*copy));
}

TEST_F(GpuFusibleTest, IsPhysicallyTransposing_PhysicalTranspose) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_reduce {
      p0.1 = f32[1024,128,32,32]{3,2,1,0} parameter(0)
      c0.1 = f32[] constant(0)
      ROOT reduce = f32[1024]{0} reduce(p0.1, c0.1), dimensions={1,2,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f16[128,1024,32,32]{3,2,1,0} parameter(0)
      copy = f32[1024,128,32,32]{3,2,1,0} transpose(p0), dimensions={1,0,2,3}
      ROOT reduce_fusion = f32[1024]{0} fusion(copy), kind=kInput, calls=fused_reduce
    })"))
                    .value();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* transpose =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(transpose->opcode(), HloOpcode::kTranspose);
  EXPECT_TRUE(IsPhysicallyTransposing(*transpose));
}

TEST_F(GpuFusibleTest, IsPhysicallyTransposing_LayoutChangingFusionProducer) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    layout_changing_computation {
      p0.1 = f16[128,1024,32,32]{3,2,1,0} parameter(0)
      p1.1 = f16[128,1024,32,32]{3,2,1,0} parameter(1)
      c0 = f16[] constant(0)
      broadcast = f16[128,1024,32,32]{3,2,1,0} broadcast(c0), dimensions={}
      greater-than = pred[128,1024,32,32]{3,2,1,0} compare(p1.1, broadcast), direction=GT
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
                    .value();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* loop_fusion =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(loop_fusion->fused_expression_root()->opcode(), HloOpcode::kCopy);
  EXPECT_TRUE(IsPhysicallyTransposing(*loop_fusion));
}

TEST_F(GpuFusibleTest,
       IsPhysicallyTransposing_ConsiderMaximumTrueRanksParamsOnly) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    broadcasting_computation {
      p0.1 = f32[128,1024,32,32]{1,3,2,0} parameter(0)
      p1.1 = f32[1,128,1,1]{3,2,1,0} parameter(1)
      reshape = f32[128]{0} reshape(p1.1)
      broadcast = f32[128,1024,32,32]{1,3,2,0} broadcast(reshape), dimensions={0}
      ROOT add = f32[128,1024,32,32]{1,3,2,0} add(p0.1, broadcast)
    }
    ENTRY entry {
      p0 = f32[128,1024,32,32]{1,3,2,0} parameter(0)
      p1 = f32[1,128,1,1]{3,2,1,0} parameter(1)
      loop_fusion = f32[128,1024,32,32]{1,3,2,0} fusion(p0, p1), kind=kLoop, calls=broadcasting_computation
      c0.2 = f32[] constant(0)
      ROOT reduce = f32[1024]{0} reduce(loop_fusion, c0.2), dimensions={0,2,3}, to_apply=scalar_add
    })"))
                    .value();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* loop_fusion =
      module->entry_computation()->root_instruction()->operand(0);
  ASSERT_EQ(loop_fusion->fused_expression_root()->opcode(), HloOpcode::kAdd);
  EXPECT_FALSE(IsPhysicallyTransposing(*loop_fusion));
}

TEST_F(GpuFusibleTest, TransposesMinorDimension) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      default_layout = f32[10,20,30,40]{3,2,1,0} parameter(0)
      non_default_layout = f32[10,20,30,40]{1,2,3,0} parameter(1)

      transpose_minor_default = f32[10,20,40,30]{3,2,1,0} transpose(default_layout), dimensions={0,1,3,2}
      no_transpose_minor_default = f32[10,20,40,30]{2,3,1,0} transpose(default_layout), dimensions={0,1,3,2}
      transpose_major_default = f32[10,30,20,40]{3,2,1,0} transpose(default_layout), dimensions={0,2,1,3}

      transpose_minor_non_default = f32[10,30,20,40]{1,2,3,0} transpose(non_default_layout), dimensions={0,2,1,3}
      no_transpose_minor_non_default = f32[10,20,40,30]{1,2,0,3} transpose(non_default_layout), dimensions={0,1,3,2}
      transpose_major_non_default = f32[10,20,40,30]{1,2,3,0} transpose(non_default_layout), dimensions={0,1,3,2}

      ROOT r = tuple(transpose_minor_default, no_transpose_minor_default, transpose_major_default,
                     transpose_minor_non_default, no_transpose_minor_non_default, transpose_major_non_default)
    })"));

  auto* tuple = (*module)->entry_computation()->root_instruction();
  EXPECT_TRUE(TransposesMinorDimension(tuple->operand(0)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(1)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(2)));
  EXPECT_TRUE(TransposesMinorDimension(tuple->operand(3)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(4)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(5)));
}

TEST_F(GpuFusibleTest, TransposesMinorDimensionSkipTrivialDimensions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      default_layout = f32[10,20,1,1]{3,2,1,0} parameter(0)
      non_default_layout = f32[10,20,1,1]{1,2,3,0} parameter(1)

      // Only trivial dimensions are swapped.
      transpose_minor_default = f32[10,20,1,1]{3,2,1,0} transpose(default_layout), dimensions={0,1,3,2}
      // The first non-trivial dimension is still the same in input and output.
      transpose_nontrivial_minor_default = f32[10,1,20,1]{3,2,1,0} transpose(default_layout), dimensions={0,2,1,3}
      no_transpose_minor_default = f32[10,20,1,1]{2,3,1,0} transpose(default_layout), dimensions={0,1,3,2}
      // We swap the most major dimension with a trivial dimension.
      transpose_one_major_default = f32[1,20,10,1]{3,2,1,0} transpose(default_layout), dimensions={2,1,0,3}
      // The first two non-trivial dimensions are swapped.
      transpose_two_major_default = f32[20,10,1,1]{3,2,1,0} transpose(default_layout), dimensions={1,0,2,3}

      transpose_minor_non_default = f32[10,1,20,1]{1,2,3,0} transpose(non_default_layout), dimensions={0,2,1,3}
      no_transpose_minor_non_default = f32[10,20,1,1]{1,2,0,3} transpose(non_default_layout), dimensions={0,1,3,2}
      transpose_major_non_default = f32[10,20,1,1]{1,2,3,0} transpose(non_default_layout), dimensions={0,1,3,2}

      ROOT r = tuple(transpose_minor_default, transpose_nontrivial_minor_default, no_transpose_minor_default, transpose_one_major_default, transpose_two_major_default,
                     transpose_minor_non_default, no_transpose_minor_non_default, transpose_major_non_default)
    })"));

  auto* tuple = (*module)->entry_computation()->root_instruction();
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(0)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(1)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(2)));
  EXPECT_TRUE(TransposesMinorDimension(tuple->operand(3)));
  EXPECT_TRUE(TransposesMinorDimension(tuple->operand(4)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(5)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(6)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(7)));
}

TEST_F(GpuFusibleTest, CopyTransposesMinorDimension) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      default_layout = f32[10,20,30,40]{3,2,1,0} parameter(0)
      non_default_layout = f32[10,20,30,40]{1,2,3,0} parameter(1)

      copy_transpose_minor_default = f32[10,20,30,40]{2,3,1,0} copy(default_layout)
      copy_no_transpose_minor_default = f32[10,20,30,40]{3,2,1,0} copy(default_layout)

      copy_transpose_minor_non_default = f32[10,20,30,40]{2,1,3,0} copy(non_default_layout)
      copy_no_transpose_minor_non_default = f32[10,20,30,40]{1,2,3,0} copy(non_default_layout)

      ROOT r = tuple(copy_transpose_minor_default, copy_no_transpose_minor_default,
                     copy_transpose_minor_non_default, copy_no_transpose_minor_non_default)
    })"));

  auto* tuple = (*module)->entry_computation()->root_instruction();
  EXPECT_TRUE(TransposesMinorDimension(tuple->operand(0)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(1)));
  EXPECT_TRUE(TransposesMinorDimension(tuple->operand(2)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(3)));
}

TEST_F(GpuFusibleTest, CopyTransposesMinorDimensionSkipTrivialDimensions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      default_layout = f32[10,20,1,1]{3,2,1,0} parameter(0)
      non_default_layout = f32[10,20,1,1]{1,2,3,0} parameter(1)

      copy_transpose_minor_default = f32[10,20,1,1]{2,3,1,0} copy(default_layout)
      copy_no_transpose_minor_default = f32[10,20,1,1]{3,2,1,0} copy(default_layout)

      copy_transpose_minor_non_default = f32[10,20,1,1]{2,0,3,1} copy(non_default_layout)
      copy_no_transpose_minor_non_default = f32[10,20,1,1]{1,2,3,0} copy(non_default_layout)

      ROOT r = tuple(copy_transpose_minor_default, copy_no_transpose_minor_default,
                     copy_transpose_minor_non_default, copy_no_transpose_minor_non_default)
    })"));

  auto* tuple = (*module)->entry_computation()->root_instruction();
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(0)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(1)));
  EXPECT_TRUE(TransposesMinorDimension(tuple->operand(2)));
  EXPECT_FALSE(TransposesMinorDimension(tuple->operand(3)));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_ReductionToVector) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      c0 = f32[] parameter(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(1)
      // Reduction-to-vector lowered by IrEmitterUnnested.
      ROOT reduce = f32[512]{0} reduce(p1, c0), dimensions={0,2,3}, to_apply=scalar_add
    })"))
                    .value();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kReduce);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_TRUE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_ElementalReduction) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      c0 = f32[] parameter(0)
      p1 = f32[8,512,5,16,1,1]{5,4,3,2,1,0} parameter(1)
      // Reduction lowered by GpuElementalIrEmitter.
      ROOT reduce = f32[512,5,1,1]{3,2,1,0} reduce(p1, c0), dimensions={3,0},
        to_apply=scalar_add
    })"))
                    .value();
  SCOPED_TRACE(module->ToString());
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kReduce);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_FALSE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_SingleOutputInputReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] constant(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT reduce = f32[128,512]{1,0} reduce(p1, c0), dimensions={2,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = f32[128,512]{1,0} fusion(p0), kind=kInput, calls=fused_reduction
    })"))
                    .value();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_TRUE(IsReduceInputFusion(*reduce));
  EXPECT_TRUE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_SingleOutputLoopReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] constant(0)
      p1 = f32[8,512,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      ROOT reduce = f32[8,5,1,1]{3,2,1,0} reduce(p1, c0), dimensions={1,3}, to_apply=scalar_add
    }
    ENTRY entry {
      p0 = f32[8,512,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      ROOT fusion = f32[8,5,1,1]{3,2,1,0} fusion(p0), kind=kLoop, calls=fused_reduction
    })"))
                    .value();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_FALSE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_MultiOutputInputReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] constant(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      reduce.0 = f32[128,512]{1,0} reduce(p1, c0), dimensions={2,3}, to_apply=scalar_add
      reduce.1 = f32[128,512]{1,0} reduce(p1, c0), dimensions={2,3}, to_apply=scalar_add
      ROOT root = (f32[128,512]{1,0}, f32[128,512]{1,0}) tuple(reduce.0, reduce.1)
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = (f32[128,512]{1,0}, f32[128,512]{1,0}) fusion(p0), kind=kInput, calls=fused_reduction
    })"))
                    .value();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_TRUE(IsReduceInputFusion(*reduce));
  EXPECT_TRUE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest,
       IsReduceInputFusion_MultiOutputInputReduceFusionWithExtraOutputs) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] constant(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      reduce = f32[128,512]{1,0} reduce(p1, c0), dimensions={2,3}, to_apply=scalar_add
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1, p1)
      ROOT root = (f32[128,512]{1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(reduce, mul)
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = (f32[128,512]{1,0}, f32[128,512,28,28]{3,2,1,0}) fusion(p0), kind=kInput, calls=fused_reduction
    })"))
                    .value();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_TRUE(IsReduceInputFusion(*reduce));
  EXPECT_TRUE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, IsReduceInputFusion_MultiOutputLoopReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] constant(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      reduce.0 = f32[512,28]{1,0} reduce(p1, c0), dimensions={0,2}, to_apply=scalar_add
      reduce.1 = f32[512,28]{1,0} reduce(p1, c0), dimensions={0,2}, to_apply=scalar_add
      ROOT root = (f32[512,28]{1,0}, f32[512,28]{1,0}) tuple(reduce.0, reduce.1)
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = (f32[512,28]{1,0}, f32[512,28]{1,0}) fusion(p0), kind=kLoop, calls=fused_reduction
    })"))
                    .value();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_FALSE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest,
       IsReduceInputFusion_MultiOutputLoopFusionReduceAndElementwiseOp) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_reduction {
      c0 = f32[] constant(0)
      p1 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      reduce = f32[512,28]{1,0} reduce(p1, c0), dimensions={0,2}, to_apply=scalar_add
      mul = f32[128,512,28,28]{3,2,1,0} multiply(p1, p1)
      ROOT root = (f32[512,28]{1,0}, f32[128,512,28,28]{3,2,1,0}) tuple(reduce, mul)
    }
    ENTRY entry {
      p0 = f32[128,512,28,28]{3,2,1,0} parameter(0)
      ROOT fusion = (f32[512,28]{1,0}, f32[128,512,28,28]{3,2,1,0}) fusion(p0), kind=kLoop, calls=fused_reduction
    })"))
                    .value();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction();
  ASSERT_EQ(reduce->opcode(), HloOpcode::kFusion);
  EXPECT_FALSE(IsReduceInputFusion(*reduce));
  EXPECT_FALSE(IsInputFusibleReduction(*reduce));
}

TEST_F(GpuFusibleTest, CustomFusionIsNotFusibleAsConsumer) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

triton_fusion {
  p0 = f16[20,3]{1,0} parameter(0)
  p1 = f16[3,40]{1,0} parameter(1)
  dot = f16[20,40]{1,0} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT c = f16[20,40]{0,1} copy(dot)
}

ENTRY e {
  p0 = f16[20,3]{1,0} parameter(0)
  n = f16[20,3]{1,0} negate(p0)
  p1 = f16[3,40]{1,0} parameter(1)
  ROOT r = f16[20,40]{0,1} fusion(n, p1),
    kind=kCustom,
    calls=triton_fusion
})"));
  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_FALSE(IsFusibleAsMultiOutputFusionRoot(*root));
}

TEST_F(GpuFusibleTest, FusionHeroesAreCompatible_TransposeFusionCompatible) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[64,32]{1,0} parameter(0)
      neg = f32[64,32]{1,0} negate(p0.1)
      ROOT transpose = f32[32,64]{1,0} transpose(neg), dimensions={1,0}
    }

    fused_computation_2 {
      p0.2 = f32[32,64]{1,0} parameter(0)
      neg = f32[32,64]{1,0} negate(p0.2)
      ROOT add = f32[32,64]{1,0} add(neg, neg)
    }

    ENTRY entry {
      p0 = f32[64,32]{1,0} parameter(0)
      fusion.1 = f32[32,64]{1,0} fusion(p0), kind=kLoop, calls=fused_computation_1
      ROOT fusion.2 = f32[32,64]{1,0} fusion(fusion.1), kind=kLoop, calls=fused_computation_2
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction();
  const HloInstruction* fusion_2 = fusion_1->operand(0);
  EXPECT_TRUE(FusionHeroesAreCompatible(fusion_1->fused_expression_root(),
                                        fusion_2->fused_expression_root()));
  EXPECT_TRUE(FusionHeroesAreCompatible(fusion_2->fused_expression_root(),
                                        fusion_1->fused_expression_root()));
}

TEST_F(GpuFusibleTest, FusionHeroesAreCompatible_TransposeFusionNotCompatible) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[64,32]{1,0} parameter(0)
      neg = f32[64,32]{1,0} negate(p0.1)
      ROOT transpose = f32[32,64]{1,0} transpose(neg), dimensions={1,0}
    }

    fused_computation_2 {
      p0.2 = f32[32,64]{1,0} parameter(0)
      broadcast = f32[32,64,4]{2,1,0} broadcast(p0.2), dimensions={0,1}
      ROOT add = f32[32,64,4]{2,1,0} add(broadcast, broadcast)
    }

    ENTRY entry {
      p0 = f32[64,32]{1,0} parameter(0)
      fusion.1 = f32[32,64]{1,0} fusion(p0), kind=kLoop, calls=fused_computation_1
      ROOT fusion.2 = f32[32,64,4]{2,1,0} fusion(fusion.1), kind=kLoop, calls=fused_computation_2
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction();
  const HloInstruction* fusion_2 = fusion_1->operand(0);
  EXPECT_FALSE(FusionHeroesAreCompatible(fusion_1->fused_expression_root(),
                                         fusion_2->fused_expression_root()));
  EXPECT_FALSE(FusionHeroesAreCompatible(fusion_2->fused_expression_root(),
                                         fusion_1->fused_expression_root()));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_LoopFusions) {
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
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_IgnoreFpPrecision) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
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
      fusion.2 = f16[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (f32[6400]{0}, f16[6400]{0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_BitcastCompatible) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[6400]{0} parameter(0)
      ROOT mul = f32[6400]{0} multiply(p0.1, p0.1)
    }

    fused_computation_2 {
      p0.2 = f32[6400]{0} parameter(0)
      bitcast = f32[1,6400]{1,0} bitcast(p0.2)
      ROOT convert = f16[1,6400]{1,0} convert(bitcast)
    }

    ENTRY entry {
      p0 = f32[6400]{0} parameter(0)
      fusion.1 = f32[6400]{0} fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f16[1,6400]{1,0} fusion(p0), kind=kLoop, calls=fused_computation_2
      ROOT root = (f32[6400]{0}, f16[1,6400]{1,0}) tuple(fusion.1, fusion.2)
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_Reduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
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
                    .value();
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion, *reduce));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_Elementwise) {
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
                    .value();
  const HloInstruction* fusion =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* div =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion, *div));
}

TEST_F(GpuFusibleTest,
       ShapesCompatibleForMultiOutputFusion_MultiOutputLoopFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      mul = f32[8,1,5,16,1,1]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      exp = f32[8,1,5,16,1,1]{5,4,3,2,1,0} exponential(p0.1)
      ROOT tuple = (f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      const.2 = f32[] constant(0)
      broadcast = f32[8,1,5,16,1,1]{5,4,3,2,1,0} broadcast(const.2), dimensions={}
      ROOT add = f32[8,1,5,16,1,1]{5,4,3,2,1,0} add(p0.2, broadcast)
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      fusion.1 = (f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}) fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} fusion(p0), kind=kLoop, calls=fused_computation_2
      gte0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=1
      ROOT root = (f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}) tuple(gte0, gte1, fusion.2)
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(2);
  EXPECT_NE(fusion_1, fusion_2);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest,
       ShapesCompatibleForMultiOutputFusion_DifferentElementType) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_computation_1 {
      p0.1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      mul = f32[8,1,5,16,1,1]{5,4,3,2,1,0} multiply(p0.1, p0.1)
      exp = f32[8,1,5,16,1,1]{5,4,3,2,1,0} exponential(p0.1)
      ROOT tuple = (f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}) tuple(mul, exp)
    }

    fused_computation_2 {
      p0.2 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      const.2 = f32[] constant(0)
      broadcast = f32[8,1,5,16,1,1]{5,4,3,2,1,0} broadcast(const.2), dimensions={}
      add = f32[8,1,5,16,1,1]{5,4,3,2,1,0} add(p0.2, broadcast)
      ROOT convert = s32[8,1,5,16,1,1]{5,4,3,2,1,0} convert(add)
    }

    ENTRY entry {
      p0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} parameter(0)
      fusion.1 = (f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}) fusion(p0), kind=kLoop, calls=fused_computation_1
      fusion.2 = s32[8,1,5,16,1,1]{5,4,3,2,1,0} fusion(p0), kind=kLoop, calls=fused_computation_2
      gte0 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=0
      gte1 = f32[8,1,5,16,1,1]{5,4,3,2,1,0} get-tuple-element(fusion.1), index=1
      ROOT root = (f32[8,1,5,16,1,1]{5,4,3,2,1,0}, f32[8,1,5,16,1,1]{5,4,3,2,1,0}, s32[8,1,5,16,1,1]{5,4,3,2,1,0}) tuple(gte0, gte1, fusion.2)
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(2);
  EXPECT_NE(fusion_1, fusion_2);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_UnfusedOps) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      exp = f32[32,32,32]{2,1,0} exponential(p0)
      reduce = f32[32,32]{1,0} reduce(exp, c0), dimensions={2},
        to_apply=scalar_add
      ROOT root = (f32[32,32]{1,0}, f32[32,32,32]{2,1,0}) tuple(reduce, exp)
    })"))
                    .value();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* exp =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*reduce, *exp));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_DifferentLayouts) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{0,1,2} parameter(1)
      c0 = f32[] constant(0)
      exp = f32[2,2,2]{2,1,0} exponential(p0)
      reduce = f32[2,2]{0,1} reduce(p1, c0), dimensions={2}, to_apply=scalar_add
      ROOT root = (f32[2,2]{0,1}, f32[2,2,2]{2,1,0}) tuple(reduce, exp)
    })"))
                    .value();
  const HloInstruction* reduce =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* exp =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_FALSE(ShapesCompatibleForMultiOutputFusion(*reduce, *exp));
}

TEST_F(
    GpuFusibleTest,
    ShapesCompatibleForMultiOutputFusion_SiblingTransposeFusionsNotCompatible) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_021_transpose {
      param_0 = f32[20,20,20]{2,1,0} parameter(0)
      transpose = f32[20,20,20]{2,1,0} transpose(param_0), dimensions={0,2,1}
      ROOT bitcast = f32[8000]{0} bitcast(transpose)
    }

    fused_220_transpose {
      param_0 = f32[20,20,20]{2,1,0} parameter(0)
      transpose = f32[20,20,20]{2,1,0} transpose(param_0), dimensions={2,1,0}
      ROOT bitcast = f32[8000]{0} bitcast(transpose)
    }

    ENTRY reduce {
      p0 = f32[20,20,20]{2,1,0} parameter(0)
      fusion = f32[8000]{0} fusion(p0), kind=kInput, calls=fused_021_transpose
      fusion.1 = f32[8000]{0} fusion(p0), kind=kInput, calls=fused_220_transpose
      ROOT root = (f32[8000]{0}, f32[8000]{0}) tuple(fusion, fusion.1)
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_FALSE(
      FusionHeroesAreCompatible(fusion_1->fused_expression_root()->operand(0),
                                fusion_2->fused_expression_root()->operand(0)));
  EXPECT_FALSE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest,
       ShapesCompatibleForMultiOutputFusion_SiblingTransposeFusionsCompatible) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_1230_transpose {
      param_0 = f32[1,20,20]{2,1,0} parameter(0)
      bitcast.1 = f32[20,2,2,5]{3,2,1,0} bitcast(param_0)
      transpose = f32[2,2,5,20]{3,2,1,0} transpose(bitcast.1), dimensions={1,2,3,0}
      ROOT bitcast.2 = f32[400]{0} bitcast(transpose)
    }

    fused_021_transpose {
      param_0 = f32[1,20,20]{2,1,0} parameter(0)
      transpose = f32[1,20,20]{2,1,0} transpose(param_0), dimensions={0,2,1}
      ROOT bitcast = f32[400]{0} bitcast(transpose)
    }

    ENTRY reduce {
      p0 = f32[1,20,20]{2,1,0} parameter(0)
      fusion = f32[400]{0} fusion(p0), kind=kInput, calls=fused_1230_transpose
      fusion.1 = f32[400]{0} fusion(p0), kind=kInput, calls=fused_021_transpose
      ROOT root = (f32[400]{0}, f32[400]{0}) tuple(fusion, fusion.1)
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest,
       ShapesCompatibleForMultiOutputFusion_MultiOutputReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_select {
      p1.1 = f32[2,2,2]{2,1,0} parameter(1)
      c0 = f32[] constant(0)
      broadcast = f32[2,2,2]{2,1,0} broadcast(f32[] c0), dimensions={}
      greater-than = pred[2,2,2]{2,1,0} compare(f32[2,2,2]{2,1,0} p1.1, f32[2,2,2]{2,1,0} broadcast), direction=GT
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
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0)->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1)->operand(0);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, ShapesCompatibleForMultiOutputFusion_ReduceFusions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
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
      ROOT root = (f32[2,2]{1,0}, f32[2,2]{1,0}) tuple(reduce_1, reduce_2)
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest,
       ShapesCompatibleForMultiOutputFusion_DifferentReduceDimensions) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_reduce_1 {
      p0.1 = f32[32,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce = f32[32,32]{1,0} reduce(f32[32,32,32]{2,1,0} p0.1, f32[] c0),
        dimensions={0}, to_apply=scalar_add
    }

    fused_reduce_2 {
      p0.2 = f32[32,32,32]{2,1,0} parameter(0)
      mul = f32[32,32,32]{2,1,0} multiply(f32[32,32,32]{2,1,0} p0.2,
        f32[32,32,32]{2,1,0} p0.2)
      c1 = f32[] constant(0)
      ROOT reduce = f32[32,32]{1,0} reduce(f32[32,32,32]{2,1,0} mul, f32[] c1),
        dimensions={2}, to_apply=scalar_add
    }

    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      reduce_1 = f32[32,32]{1,0} fusion(p0), kind=kLoop, calls=fused_reduce_1
      reduce_2 = f32[32,32]{1,0} fusion(p1), kind=kLoop, calls=fused_reduce_2
      ROOT root = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(reduce_1, reduce_2)
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_FALSE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest,
       ShapesCompatibleForMultiOutputFusion_NoReductionToVector) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_element_wise {
      p0.1 = f32[32,32,32]{2,1,0} parameter(0)
      p1.1 = f32[32,32,32]{2,1,0} parameter(1)
      ROOT add = f32[32,32,32]{2,1,0} add(p0.1, p1.1)
    }

    fused_reduce {
      p0.2 = f32[32,32,32]{2,1,0} parameter(0)
      mul = f32[32,32,32]{2,1,0} multiply(f32[32,32,32]{2,1,0} p0.2,
        f32[32,32,32]{2,1,0} p0.2)
      broadcast = f32[32,32,32,32]{3,2,1,0} broadcast(mul), dimensions={3,2,1}
      c1 = f32[] constant(0)
      // Note that reduce is not a reduction-to-vector.
      ROOT reduce = f32[32,32]{1,0} reduce(f32[32,32,32,32]{3,2,1,0} broadcast,
        f32[] c1), dimensions={1,3}, to_apply=scalar_add
    }

    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      element_wise = f32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop,
        calls=fused_element_wise
      fusion = f32[32,32]{1,0} fusion(element_wise),
        kind=kLoop, calls=fused_reduce
      ROOT root = (f32[32,32]{1,0}, f32[32,32,32]{2,1,0})
        tuple(fusion, element_wise)
    })"))
                    .value();
  const HloInstruction* fusion_1 =
      module->entry_computation()->root_instruction()->operand(0);
  const HloInstruction* fusion_2 =
      module->entry_computation()->root_instruction()->operand(1);
  EXPECT_FALSE(ShapesCompatibleForMultiOutputFusion(*fusion_1, *fusion_2));
}

TEST_F(GpuFusibleTest, IsFusibleAsMultiOutputFusionRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    })")
                    .value();

  const HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_TRUE(IsFusibleAsMultiOutputFusionRoot(*root));
}

TEST_F(GpuFusibleTest, ScatterIsNotFusibleAsMultiOutputFusionRoot) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY Scatter {
      p0 = s32[3,3] parameter(0)
      operand = s32[3,3] add(p0, p0)
      p1 = s32[2] parameter(1)
      indices = s32[2] add(p1, p1)
      p2 = s32[2,3] parameter(2)
      updates = s32[2,3] add(p2, p2)
      ROOT scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
    })")
                    .value();

  const HloInstruction* scatter_inst =
      module->entry_computation()->root_instruction();
  EXPECT_FALSE(IsFusibleAsMultiOutputFusionRoot(*scatter_inst));
}

TEST_F(GpuFusibleTest, ProducerConsumerFusionElementwiseAndReduce) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      c0 = f32[] constant(0)
      exp = f32[32,32,32]{2,1,0} exponential(p0)
      reduce = f32[32,32]{1,0} reduce(exp, c0), dimensions={2},
        to_apply=scalar_add
      ROOT root = (f32[32,32]{1,0}, f32[32,32,32]{2,1,0}) tuple(reduce, exp)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* consumer = root->operand(0);
  const HloInstruction* producer = root->operand(1);
  EXPECT_TRUE(IsProducerMultiOutputFusible(*producer));
  EXPECT_TRUE(IsFusibleAsMultiOutputFusionRoot(*consumer));
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*producer, *consumer));
}

TEST_F(GpuFusibleTest, ProducerConsumerFusionTransposeAndLoopFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0.1 = f32[32,31,30]{2,1,0} parameter(0)
      p1.1 = f32[32,31,30]{2,1,0} parameter(1)
      neg = f32[32,31,30]{2,1,0} negate(p0.1)
      ROOT add = f32[32,31,30]{2,1,0} add(neg, p1.1)
    }

    ENTRY reduce {
      p0 = f32[32,31,30]{2,1,0} parameter(0)
      p1 = f32[32,30,31]{2,1,0} parameter(1)
      transpose =  f32[32,31,30]{2,1,0} transpose(p1), dimensions={0,2,1}
      ROOT add = f32[32,31,30]{2,1,0} fusion(p0, transpose), kind=kLoop, calls=fused_add
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* consumer = root;
  const HloInstruction* producer = root->operand(1);
  EXPECT_TRUE(IsProducerConsumerFusible(*producer, *consumer));
}

TEST_F(GpuFusibleTest, ProducerConsumerFusionReduceAndLoopFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_add {
      p0.1 = f32[32,31,30]{2,1,0} parameter(0)
      p1.1 = f32[32,31,30]{2,1,0} parameter(1)
      neg = f32[32,31,30]{2,1,0} negate(p0.1)
      ROOT add = f32[32,31,30]{2,1,0} add(neg, p1.1)
    }

    ENTRY reduce {
      p0 = f32[32,31,30]{2,1,0} parameter(0)
      p1 = f32[32,31,30,29]{3,2,1,0} parameter(1)
      c0 = f32[] constant(0.0)
      reduce =  f32[32,31,30]{2,1,0} reduce(p1, c0), dimensions={3}, to_apply=scalar_add
      ROOT add = f32[32,31,30]{2,1,0} fusion(p0, reduce), kind=kLoop, calls=fused_add
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* consumer = root;
  const HloInstruction* producer = root->operand(1);
  EXPECT_TRUE(IsProducerConsumerFusible(*producer, *consumer));
}

TEST_F(GpuFusibleTest, ProducerConsumerFusionLoopFusionAndReduce) {
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
        to_apply=scalar_add
      ROOT root = (f32[32,32]{1,0}, f32[32,32,32]{2,1,0}) tuple(reduce, add)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* consumer = root->operand(0);
  const HloInstruction* producer = root->operand(1);
  EXPECT_TRUE(IsProducerMultiOutputFusible(*producer));
  EXPECT_TRUE(IsFusibleAsMultiOutputFusionRoot(*consumer));
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*producer, *consumer));
}

TEST_F(GpuFusibleTest, ProducerConsumerFusionLoopFusionAndReduceFusion) {
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
        to_apply=scalar_add
      mul = f32[32,32,32]{2,1,0} multiply(p0.2, p0.2)
      r2 = f32[32,32]{1,0} reduce(mul, c1), dimensions={2},
        to_apply=scalar_add
      ROOT tuple = (f32[32,32]{1,0}, f32[32,32]{1,0}) tuple(r1, r2)
    }

    ENTRY reduce {
      p0 = f32[32,32,32]{2,1,0} parameter(0)
      p1 = f32[32,32,32]{2,1,0} parameter(1)
      select = f32[32,32,32]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_select
      fusion = (f32[32,32]{1,0}, f32[32,32]{1,0}) fusion(select), kind=kInput,
        calls=fused_reduce
      ROOT root = ((f32[32,32]{1,0}, f32[32,32]{1,0}), f32[32,32,32]{2,1,0}) tuple(fusion, select)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* consumer = root->operand(0);
  const HloInstruction* producer = root->operand(1);
  EXPECT_TRUE(IsProducerMultiOutputFusible(*producer));
  EXPECT_TRUE(IsFusibleAsMultiOutputFusionRoot(*consumer));
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*producer, *consumer));
}

TEST_F(GpuFusibleTest, ProducerConsumerFusionDoNotFuseLoopReduceFusion) {
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
        f32[] c1), dimensions={1,3}, to_apply=scalar_add
    }

    ENTRY reduce {
      p0 = f32[2,2,2]{2,1,0} parameter(0)
      p1 = f32[2,2,2]{2,1,0} parameter(1)
      element_wise = f32[2,2,2]{2,1,0} fusion(p0, p1), kind=kLoop, calls=fused_element_wise
      fusion = f32[2,2]{1,0} fusion(element_wise), kind=kLoop, calls=fused_reduce
      ROOT root = (f32[2,2]{1,0}, f32[2,2,2]{2,1,0}) tuple(fusion, element_wise)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* consumer = root->operand(0);
  const HloInstruction* producer = root->operand(1);
  EXPECT_TRUE(IsProducerMultiOutputFusible(*producer));
  EXPECT_TRUE(IsFusibleAsMultiOutputFusionRoot(*consumer));
  EXPECT_FALSE(ShapesCompatibleForMultiOutputFusion(*producer, *consumer));
}

TEST_F(GpuFusibleTest, ProducerConsumerFusionReduceUnfriendlyLoopFusion) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    mixed_input_layouts_computation {
      p0.1 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      p1.1 = f16[128,1024,33,33]{3,2,1,0} parameter(1)
      copy = f16[128,1024,33,33]{1,3,2,0} copy(p1.1)
      slice = f16[128,1024,32,32]{1,3,2,0} slice(copy), slice={[0:128],[0:1024],[0:32],[0:32]}
      c0 = f16[] constant(0)
      broadcast = f16[128,1024,32,32]{1,3,2,0} broadcast(c0), dimensions={}
      greater-than = pred[128,1024,32,32]{1,3,2,0} compare(slice, broadcast), direction=GT
      ROOT root = f16[128,1024,32,32]{1,3,2,0} select(greater-than, p0.1, broadcast)
    }
    fused_reduce {
      p0.2 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      convert = f32[128,1024,32,32]{1,3,2,0} convert(p0.2)
      c0.2 = f32[] constant(0)
      ROOT reduce = f32[1024]{0} reduce(convert, c0.2), dimensions={0,2,3}, to_apply=scalar_add
    }
    ENTRY reduce {
      p0 = f16[128,1024,32,32]{1,3,2,0} parameter(0)
      p1 = f16[128,1024,33,33]{3,2,1,0} parameter(1)
      loop_fusion = f16[128,1024,32,32]{1,3,2,0} fusion(p0, p1), kind=kLoop, calls=mixed_input_layouts_computation
      reduce_fusion = f32[1024]{0} fusion(loop_fusion), kind=kInput, calls=fused_reduce
      ROOT root = (f32[1024]{0}, f16[128,1024,32,32]{1,3,2,0}) tuple(reduce_fusion, loop_fusion)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* consumer = root->operand(0);
  const HloInstruction* producer = root->operand(1);
  EXPECT_FALSE(IsProducerMultiOutputFusible(*producer));
  EXPECT_TRUE(IsFusibleAsMultiOutputFusionRoot(*consumer));
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*producer, *consumer));
}

TEST_F(GpuFusibleTest, ProducerConsumerFusionInPlaceOperation) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    %fusion {
      %param_0 = s32[4,4]{1,0} parameter(0)
      %copy = s32[4,4]{0,1} copy(%param_0)
      ROOT %transpose = s32[4,4]{1,0} transpose(%copy), dimensions={1,0}
    }

    ENTRY %main {
      %param_0 = s32[4,4]{1,0} parameter(0)
      %constant_0 = s32[] constant(0)
      %constant_1 = s32[] constant(1)
      %constant_1x1_1 = s32[1,1] constant({ {1} })
      %updated = s32[4,4]{1,0} dynamic-update-slice(%param_0, %constant_1x1_1, %constant_1, %constant_0)
      %transpose = s32[4,4]{0,1} fusion(%updated), kind=kLoop, calls=fusion
      ROOT %tuple = tuple(%updated, %transpose)
    })"))
                    .value();
  const HloInstruction* tuple = module->entry_computation()->root_instruction();
  EXPECT_EQ(tuple->opcode(), HloOpcode::kTuple);
  const HloInstruction* dus = tuple->operand(0);
  EXPECT_EQ(dus->opcode(), HloOpcode::kDynamicUpdateSlice);
  const HloInstruction* transpose = tuple->operand(1);
  EXPECT_EQ(transpose->opcode(), HloOpcode::kFusion);
  EXPECT_FALSE(IsProducerMultiOutputFusible(*dus));
  EXPECT_TRUE(IsFusibleAsMultiOutputFusionRoot(*transpose));
  EXPECT_TRUE(ShapesCompatibleForMultiOutputFusion(*dus, *transpose));
}

TEST_F(GpuFusibleTest, NonscalarConstantsNotFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY BroadcastIntoReduce {
      constant = f32[16] constant({0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15})
      broadcast = f32[16,16,16,16]{3,2,1,0} broadcast(constant), dimensions={0}
      constant.1 = f32[] constant(0)
      reduce = f32[] reduce(broadcast, constant.1), dimensions={0,1,2,3},
                                                         to_apply=add
      ROOT root = (f32[], f32[], f32[16,16,16,16], f32[16]) tuple(reduce, constant.1, broadcast, constant)
    })")
                    .value();
  // Do not fuse if producer is a non-scalar constant or consumer is non-fusion
  // node.
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* consumer = root->operand(0);
  const HloInstruction* producer = root->operand(1);
  const HloInstruction* consumer2 = root->operand(2);
  const HloInstruction* producer2 = root->operand(3);
  EXPECT_FALSE(
      static_cast<bool>(IsProducerConsumerFusible(*producer, *consumer)));
  EXPECT_FALSE(
      static_cast<bool>(IsProducerConsumerFusible(*producer2, *consumer2)));
}

TEST_F(GpuFusibleTest, FuseLayoutChangingOpWithElementwise) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    ENTRY entry {
      p0 = f32[16,16,16,16]{3,2,1,0} parameter(0)
      copy = f32[16,16,16,16]{0,1,2,3} copy(p0)
      ROOT add = f32[16,16,16,16]{0,1,2,3} add(copy, copy)
    })")
                    .value();

  const HloInstruction* consumer =
      module->entry_computation()->root_instruction();
  const HloInstruction* producer = consumer->operand(0);
  EXPECT_TRUE(
      static_cast<bool>(IsProducerConsumerFusible(*producer, *consumer)));
}

TEST_F(GpuFusibleTest, FuseReduceWithUnaryElementwise) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY main.12 {
      Arg_0.1 = f32[2048]{0} parameter(0)
      constant.4 = f32[] constant(0.0)
      reduce.10 = f32[] reduce(Arg_0.1, constant.4), dimensions={0}, to_apply=scalar_add
      ROOT exp = f32[] exponential(reduce.10)
    })"))
                    .value();

  const HloInstruction* consumer =
      module->entry_computation()->root_instruction();
  const HloInstruction* producer = consumer->operand(0);
  EXPECT_TRUE(
      static_cast<bool>(IsProducerConsumerFusible(*producer, *consumer)));
}

TEST_F(GpuFusibleTest, DoNotFuseReduceWithRacesWithUnaryElementwise) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY main.12 {
      Arg_0.1 = f32[196608]{0} parameter(0)
      constant.4 = f32[] constant(0.0)
      reduce.10 = f32[] reduce(Arg_0.1, constant.4), dimensions={0}, to_apply=scalar_add
      ROOT exp = f32[] exponential(reduce.10)
    })"))
                    .value();

  const HloInstruction* consumer =
      module->entry_computation()->root_instruction();
  const HloInstruction* producer = consumer->operand(0);
  EXPECT_FALSE(
      static_cast<bool>(IsProducerConsumerFusible(*producer, *consumer)));
}

TEST_F(GpuFusibleTest, CreatesHeavyComputation_NonfusionInstr) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      p_0 = f32[20,50] parameter(0)

      constant_1 = f32[] constant(1)
      reduce-window_1 = f32[21,41] reduce-window(p_0, constant_1),
        window={size=20x10 pad=0_20x0_0}, to_apply=scalar_add

      constant_2 = f32[] constant(2)
      reduce-window_2 = f32[21,41] reduce-window(p_0, constant_2),
        window={size=20x10 pad=0_20x0_0}, to_apply=scalar_add

      ROOT root = (f32[21,41], f32[21,41])
        tuple(reduce-window_1, reduce-window_2)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* producer = root->operand(0);
  const HloInstruction* consumer = root->operand(1);
  EXPECT_TRUE(CreatesHeavyComputation(*producer, *consumer));
}

TEST_F(GpuFusibleTest, DoesNotCreateHeavyComputation_NonfusionInstr) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      p_0 = f32[3,5] parameter(0)
      constant = f32[] constant(1)
      broadcast = f32[3, 5] broadcast(f32[] constant), dimensions={}
      scaled_p_0 = f32[3,5] multiply(f32[3, 5] broadcast, f32[3,5]{1, 0} p_0)

      p_1 = f32[2,5] parameter(1)
      reduce-window = f32[3,5] reduce-window(p_1, constant),
        window={size=2x1 pad=0_2x0_0}, to_apply=scalar_add

      ROOT root = (f32[3,5], f32[3,5]) tuple(reduce-window, scaled_p_0)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* producer = root->operand(0);
  const HloInstruction* consumer = root->operand(1);
  EXPECT_FALSE(CreatesHeavyComputation(*producer, *consumer));
}

TEST_F(GpuFusibleTest,
       DoesNotCreateHeavyComputation_NonoverlappingReduceWindows) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      p_0 = f32[2,5] parameter(0)

      constant_1 = f32[] constant(1)
      reduce-window_1 = f32[3,5] reduce-window(p_0, constant_1),
        window={size=2x1 pad=0_2x0_0}, to_apply=scalar_add

      constant_2 = f32[] constant(2)
      reduce-window_2 = f32[2,3] reduce-window(p_0, constant_2),
        window={size=2x1 pad=0_2x0_0 stride=2x2}, to_apply=scalar_add

      ROOT root = (f32[3,5], f32[2,3]) tuple(reduce-window_1, reduce-window_2)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* producer = root->operand(0);
  const HloInstruction* consumer = root->operand(1);
  EXPECT_FALSE(CreatesHeavyComputation(*producer, *consumer));
}

TEST_F(GpuFusibleTest, CreatesHeavyComputation_ReduceWindowGather) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    ENTRY entry {
      p0 = s32[512,512,2] parameter(0)
      p1 = f32[1,1,512,512] parameter(1)
      constant_1 = f32[] constant(0)
      reduce-window.1 = reduce-window(p1, constant_1),
        window={size=1x1x16x16 stride=1x1x16x16}, to_apply=scalar_add
      ROOT ret = gather(reduce-window.1, p0), offset_dims={0,1,2,3},
        collapsed_slice_dims={}, start_index_map={1,2},
        index_vector_dim=2, slice_sizes={1,1,1,1}
    })"))
                    .value();
  auto gather = module->entry_computation()->root_instruction();
  auto reduce_window = gather->operand(0);
  EXPECT_EQ(gather->opcode(), HloOpcode::kGather);
  EXPECT_EQ(reduce_window->opcode(), HloOpcode::kReduceWindow);
  EXPECT_FALSE(IfFusedReadsElementsMultipleTimes(*reduce_window));
  EXPECT_TRUE(IsExpensiveToRepeat(*reduce_window));
  EXPECT_TRUE(IfFusedReadsElementsMultipleTimes(*gather));
  EXPECT_TRUE(CreatesHeavyComputation(*reduce_window, *gather));
}

TEST_F(GpuFusibleTest, CreatesHeavyComputation_FusionInstr) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_producer {
      operand = f32[20,20] parameter(0)
      constant = f32[] constant(1)
      ROOT reduce-window = f32[11,11] reduce-window(operand, constant),
        window={size=20x20 pad=0_10x0_10}, to_apply=scalar_add
    }

    fused_consumer {
      operand_0 = f32[11,11] parameter(0)
      operand_1 = f32[11,11] parameter(1)
      constant = f32[] constant(1)
      reduce-window = f32[11,11] reduce-window(operand_1, constant),
        window={size=2x2 pad=0_1x0_1}, to_apply=scalar_add
      ROOT scaled_operand_1 =
        f32[11,11] multiply(f32[11,11] operand_0, f32[11,11] reduce-window)
    }

    ENTRY entry {
      p0 = f32[20,20] parameter(0)
      p1 = f32[11,11] parameter(1)
      producer = f32[11,11] fusion(p0), kind=kLoop, calls=fused_producer
      consumer = f32[11,11] fusion(p1, producer), kind=kLoop, calls=fused_consumer
      ROOT root = (f32[11,11], f32[11,11]) tuple(producer, consumer)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* producer = root->operand(0);
  const HloInstruction* consumer = root->operand(1);
  EXPECT_TRUE(CreatesHeavyComputation(*producer, *consumer));
}

TEST_F(GpuFusibleTest, DoesNotCreateHeavyComputation_FusionInstr) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_producer {
      p_0 = f32[2,2] parameter(0)
      constant = f32[] constant(1)
      ROOT reduce-window = f32[2,2] reduce-window(p_0, constant),
        window={size=2x2 pad=0_1x0_1}, to_apply=scalar_add
    }

    fused_consumer {
      p_0 = f32[2,2] parameter(0)

      p_1 = f32[2,2] parameter(1)
      constant = f32[] constant(1)
      reduce-window = f32[2,2] reduce-window(p_1, constant),
        window={size=2x2 pad=0_1x0_1}, to_apply=scalar_add

      ROOT scaled_p_1 = f32[2,2] multiply(f32[2, 2] p_0, f32[2,2] reduce-window)
    }

    ENTRY entry {
      p_0 = f32[2,2] parameter(0)
      producer = f32[2,2] fusion(p_0), kind=kLoop, calls=fused_producer
      consumer = f32[2,2] fusion(producer, p_0), kind=kLoop, calls=fused_consumer
      ROOT root = (f32[2,2], f32[2,2]) tuple(producer, consumer)
    })"))
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* producer = root->operand(0);
  const HloInstruction* consumer = root->operand(1);
  EXPECT_FALSE(CreatesHeavyComputation(*producer, *consumer));
}

TEST_F(GpuFusibleTest, ChooseFusionKind) {
  auto module = ParseAndReturnVerifiedModule(R"(
HloModule module

ENTRY computation {
    p = f32[5000,6000]{1,0} parameter(0)
    c = f32[6000,5000] transpose(p), dimensions={1,0}
    ROOT r = f32[300,20,5000] reshape(c)
}
)")
                    .value();
  const HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* producer = root->operand(0);
  EXPECT_EQ(ChooseFusionKind(*producer, *root),
            HloInstruction::FusionKind::kInput);
}

TEST_F(GpuFusibleTest, GetFusionRoots1) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    fusion {
      p0 = s32[] parameter(0)
      custom-call = (bf16[], s32[]) custom-call(p0), custom_call_target="my_custom_call"
      get-tuple-element.0 = bf16[] get-tuple-element(custom-call), index=0
      get-tuple-element.1 = s32[] get-tuple-element(custom-call), index=1
      ROOT tuple = (bf16[], s32[], s32[]) tuple(get-tuple-element.0, get-tuple-element.1, p0)
    }

    ENTRY entry{
      p0 = s32[] parameter(0)
      ROOT fusion = (bf16[], s32[], s32[]) fusion(p0), kind=kCustom, calls=fusion
    }
  )")
                    .value();
  auto called_computations =
      module->entry_computation()->root_instruction()->called_computations();
  ASSERT_EQ(called_computations.size(), 1);
  auto fusion = called_computations.front();
  auto roots = GetFusionRoots(*fusion);
  auto custom_call = fusion->root_instruction()->operand(0)->operand(0);
  auto parameter = fusion->root_instruction()->operand(2);
  std::vector<const HloInstruction*> expected_roots{custom_call, custom_call,
                                                    parameter};
  EXPECT_EQ(roots, expected_roots);
}

TEST_F(GpuFusibleTest, GetFusionRoots2) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    fusion {
      p0 = s32[] parameter(0)
      custom-call.1 = bf16[] custom-call(p0), custom_call_target="my_custom_call1"
      custom-call.2 = bf16[] custom-call(p0), custom_call_target="my_custom_call2"
      ROOT tuple = (bf16[], bf16[], s32[]) tuple(custom-call.1, custom-call.2, p0)
    }

    ENTRY entry{
      p0 = s32[] parameter(0)
      ROOT fusion = (bf16[], bf16[], s32[]) fusion(p0), kind=kCustom, calls=fusion
    }
  )")
                    .value();
  auto called_computations =
      module->entry_computation()->root_instruction()->called_computations();
  ASSERT_EQ(called_computations.size(), 1);
  auto fusion = called_computations.front();
  auto roots = GetFusionRoots(*fusion);
  auto custom_call1 = fusion->root_instruction()->operand(0);
  auto custom_call2 = fusion->root_instruction()->operand(1);
  auto parameter = fusion->root_instruction()->operand(2);
  std::vector<const HloInstruction*> expected_roots{custom_call1, custom_call2,
                                                    parameter};
  EXPECT_EQ(roots, expected_roots);
}

TEST_F(GpuFusibleTest, GetFusionRoots3) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    fusion {
      p0 = s32[] parameter(0)
      custom-call = (bf16[], s32[]) custom-call(p0), custom_call_target="my_custom_call"
      get-tuple-element.0 = bf16[] get-tuple-element(custom-call), index=0
      custom-call.2 = bf16[] custom-call(p0), custom_call_target="my_custom_call2"
      get-tuple-element.1 = s32[] get-tuple-element(custom-call), index=1
      ROOT tuple = (bf16[], bf16[], s32[], s32[]) tuple(get-tuple-element.0, custom-call.2, get-tuple-element.1, p0)
    }

    ENTRY entry{
      p0 = s32[] parameter(0)
      ROOT fusion = (bf16[], bf16[], s32[], s32[]) fusion(p0), kind=kCustom, calls=fusion
    }
  )")
                    .value();
  auto called_computations =
      module->entry_computation()->root_instruction()->called_computations();
  ASSERT_EQ(called_computations.size(), 1);
  auto fusion = called_computations.front();
  auto roots = GetFusionRoots(*fusion);
  auto custom_call1 = fusion->root_instruction()->operand(0)->operand(0);
  auto custom_call2 = fusion->root_instruction()->operand(1);
  auto parameter = fusion->root_instruction()->operand(3);
  std::vector<const HloInstruction*> expected_roots{custom_call1, custom_call2,
                                                    custom_call1, parameter};
  EXPECT_EQ(roots, expected_roots);
}

TEST_F(GpuFusibleTest, GetFusionRootsWithGTEMakeTupleSequence) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    fusion {
      p0 = s32[] parameter(0)
      p1 = s32[32] parameter(1)
      custom-call = (bf16[], s32[], u32[]) custom-call(p1), custom_call_target="my_custom_call"
      get-tuple-element.0 = bf16[] get-tuple-element(custom-call), index=0
      get-tuple-element.1 = s32[] get-tuple-element(custom-call), index=1
      bitcast = s32[1] bitcast(get-tuple-element.1)
      dynamic-update-slice = s32[32] dynamic-update-slice(p1, bitcast, p0)
      get-tuple-element.2 = u32[] get-tuple-element(custom-call), index=2
      ROOT tuple = (bf16[], s32[32], u32[]) tuple(get-tuple-element.0, dynamic-update-slice, get-tuple-element.2)
    }

    ENTRY entry{
      p0 = s32[] parameter(0)
      bitcast = s32[32] bitcast(p0)
      ROOT fusion = (bf16[], s32[32], u32[]) fusion(p0, bitcast), kind=kCustom, calls=fusion
    }
  )")
                    .value();

  auto called_computations =
      module->entry_computation()->root_instruction()->called_computations();
  ASSERT_EQ(called_computations.size(), 1);
  auto fusion = called_computations.front();
  auto roots = GetFusionRoots(*fusion);
  auto custom_call = fusion->root_instruction()->operand(0)->operand(0);
  auto dus = fusion->root_instruction()->operand(1);
  std::vector<const HloInstruction*> expected_result{custom_call, dus,
                                                     custom_call};
  EXPECT_EQ(roots, expected_result);
}

TEST_F(GpuFusibleTest, GetFusionRootsWithMakeTupleGTESequence) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    fusion {
      p0 = s32[] parameter(0)
      p1 = s32[32] parameter(1)
      custom-call = (bf16[], s32[], u32[]) custom-call(p1), custom_call_target="my_custom_call"
      get-tuple-element.0 = bf16[] get-tuple-element(custom-call), index=0
      get-tuple-element.1 = s32[] get-tuple-element(custom-call), index=1
      bitcast = s32[1] bitcast(get-tuple-element.1)
      dynamic-update-slice = s32[32] dynamic-update-slice(p1, bitcast, p0)
      get-tuple-element.2 = u32[] get-tuple-element(custom-call), index=2
      tuple = (bf16[], s32[32], u32[]) tuple(get-tuple-element.0, dynamic-update-slice, get-tuple-element.2)
      get-tuple-element.3 = bf16[] get-tuple-element(tuple), index=0
      get-tuple-element.4 = u32[] get-tuple-element(tuple), index=2
      ROOT tuple2 = (bf16[], s32[32], u32[]) tuple(get-tuple-element.3, dynamic-update-slice, get-tuple-element.4)
    }

    ENTRY entry{
      p0 = s32[] parameter(0)
      bitcast = s32[32] bitcast(p0)
      ROOT fusion = (bf16[], s32[32], u32[]) fusion(p0, bitcast), kind=kCustom, calls=fusion
    }
  )")
                    .value();

  auto called_computations =
      module->entry_computation()->root_instruction()->called_computations();
  ASSERT_EQ(called_computations.size(), 1);
  auto fusion = called_computations.front();
  auto roots = GetFusionRoots(*fusion);
  auto tuple_inst = fusion->root_instruction()->operand(0)->operand(0);
  auto custom_call = tuple_inst->operand(0)->operand(0);
  auto dus = fusion->root_instruction()->operand(1);
  std::vector<const HloInstruction*> expected_result{custom_call, dus,
                                                     custom_call};
  EXPECT_EQ(roots, expected_result);
}

TEST_F(GpuFusibleTest, GetFusibleComputations) {
  auto module = ParseAndReturnVerifiedModule(absl::StrCat(kModulePrefix, R"(
    fused_reduce {
      p0 = f32[128,1024] parameter(0)
      c0 = f32[] constant(0)
      ROOT reduce = f32[128]{0} reduce(p0, c0), dimensions={1}, to_apply=scalar_add
    }
    body_a {
      p0 = f32[128,1024] parameter(0)
      ROOT reduce_fusion = f32[128] fusion(p0), kind=kInput, calls=fused_reduce
    }
    body_b {
      p0 = f32[128,1024] parameter(0)
      c0 = f32[] constant(0)
      ROOT bc = f32[128] broadcast(c0), dimensions={}
    }
    ENTRY main {
      p0 = s32[] parameter(0)
      p1 = f32[128,1024] parameter(1)
      ROOT conditional = f32[128] conditional(p0, p1, p1),
        branch_computations={body_a, body_b}
    })"))
                    .value();

  // fused_reduce is already fused, scalar_add is not fusible.
  auto fusible = GetFusibleComputations(*module, {});
  EXPECT_THAT(fusible, ElementsAre(module->GetComputationWithName("body_a"),
                                   module->GetComputationWithName("body_b"),
                                   module->entry_computation()));
}

}  // namespace gpu
}  // namespace xla
