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

#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"

#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_parser.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/test_utils.h"
#include "tensorflow/compiler/xla/util.h"

namespace op = xla::testing::opcode_matchers;

namespace xla {
namespace gpu {

using InstructionFusionTest = HloTestBase;

TEST_F(InstructionFusionTest,
       CostlyProducerAndOperandElementReusingConsumerNotFused) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5.0f)));
  HloInstruction* log1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kLog, const0));
  HloInstruction* broadcast2 =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(F32, {1}), log1, {}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(broadcast2, computation->root_instruction());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
  EXPECT_EQ(broadcast2, computation->root_instruction());
}

TEST_F(InstructionFusionTest,
       NonCostlyProducerAndOperandElementReusingConsumerFused) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5)));
  HloInstruction* negate1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kNegate, const0));
  HloInstruction* broadcast2 =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(S32, {1}), negate1, {}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(broadcast2, computation->root_instruction());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest,
       CostlyProducerAndNonOperandElementReusingConsumerFused_Reshape) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5.0f)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kExp, const0));
  HloInstruction* reshape2 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(F32, {}), exp1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(reshape2, computation->root_instruction());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest,
       CostlyProducerAndNonOperandElementReusingConsumerFused_Transpose) {
  HloComputation::Builder builder(TestName());
  HloInstruction* const0 = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5.0f)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kExp, const0));
  HloInstruction* transpose2 = builder.AddInstruction(
      HloInstruction::CreateTranspose(ShapeUtil::MakeShape(F32, {}), exp1, {}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose2, computation->root_instruction());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, PotentialBitcastReshapeOfDotFused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(F32, {1, 1}), "0"));
  auto dot1 = builder.AddInstruction(
      CreateCanonicalDot(ShapeUtil::MakeShape(F32, {1, 1}), param0, param0));
  auto reshape2 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(F32, {1, 1, 1}), dot1));
  auto log = builder.AddInstruction(HloInstruction::CreateUnary(
      reshape2->shape(), xla::HloOpcode::kLog, reshape2));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(log, computation->root_instruction());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
}

TEST_F(InstructionFusionTest, PotentialBitcastTransposeOfDotUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(S32, {1, 1}), "0"));
  auto dot1 = builder.AddInstruction(
      CreateCanonicalDot(ShapeUtil::MakeShape(S32, {1, 1}), param0, param0));
  auto transpose2 = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(S32, {1, 1}), dot1, {0, 1}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose2, computation->root_instruction());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

// Tests that broadcasts fused into a fusion with a reduce root.
TEST_F(InstructionFusionTest, BroadcastIntoReduce) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY BroadcastIntoReduce {
      constant = f32[] constant(1)
      broadcast = f32[16,16,16,16]{3,2,1,0} broadcast(constant), dimensions={}
      constant.1 = f32[] constant(0)
      ROOT reduce = f32[] reduce(broadcast, constant.1), dimensions={0,1,2,3},
                                                         to_apply=add
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Reduce(op::Broadcast(op::Constant()), op::Constant()));
}

TEST_F(InstructionFusionTest, DoNotFuseLayoutChangingOpWithReduce) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY entry {
      p0 = f32[16,16,16,16]{3,2,1,0} parameter(0)
      copy = f32[16,16,16,16]{0,1,2,3} copy(p0)
      constant.1 = f32[] constant(0)
      ROOT reduce = f32[16] reduce(copy, constant.1), dimensions={0,1,2}, to_apply=add
    })")
                    .ValueOrDie();

  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

TEST_F(InstructionFusionTest, DoNotFuseLayoutChangingOpWithReduceFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    fused_reduce {
      p0.1 = f32[16,16,16,16]{0,1,2,3} parameter(0)
      mul = f32[16,16,16,16]{0,1,2,3} multiply(p0.1, p0.1)
      c0.1 = f32[] constant(0)
      ROOT root = f32[] reduce(mul, c0.1), dimensions={0,1,2,3}, to_apply=add
    }

    ENTRY entry {
      p0 = f32[16,16,16,16]{3,2,1,0} parameter(0)
      copy = f32[16,16,16,16]{0,1,2,3} copy(p0)
      fusion = f32[] fusion(copy), kind=kInput, calls=fused_reduce
      ROOT root = (f32[]) tuple(fusion)
    })")
                    .ValueOrDie();

  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

TEST_F(InstructionFusionTest, FuseLayoutChangingOpWithElementwise) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    ENTRY entry {
      p0 = f32[16,16,16,16]{3,2,1,0} parameter(0)
      copy = f32[16,16,16,16]{0,1,2,3} copy(p0)
      ROOT add = f32[16,16,16,16]{0,1,2,3} add(copy, copy)
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(), op::Add(op::Copy(), op::Copy()));
}

TEST_F(InstructionFusionTest, BitcastIntoAdd) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY BroadcastIntoAdd {
      p0 = f32[4,1,1]{2,1,0} parameter(0)
      p1 = f32[4,1]{1,0} parameter(1)
      bitcast = f32[4,1]{1,0} bitcast(p0)
      ROOT add = f32[4,1] add(bitcast, p1)
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Add(op::Bitcast(op::Parameter()), op::Parameter()));
}

TEST_F(InstructionFusionTest, AddIntoBitcast) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY BroadcastIntoAdd {
      p0 = f32[4,1]{1,0} parameter(0)
      p1 = f32[4,1]{1,0} parameter(1)
      add = f32[4,1] add(p0, p1)
      ROOT bitcast = f32[4,1,1] bitcast(add)
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Bitcast(op::Add(op::Parameter(), op::Parameter())));
}

TEST_F(InstructionFusionTest, DontFuseGTE) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY DontFuseGTE {
    p0 = (f32[10], f32[10]) parameter(0)
    gte0 = f32[10] get-tuple-element(p0), index=0
    gte1 = f32[10] get-tuple-element(p0), index=1
    ROOT add = f32[10] add(gte0, gte1)
  })")
                    .ValueOrDie();

  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

// Compute sum(1/p0), where p0 has type f32, twice.  Check that the division is
// duplicated and fused into both reduces.
TEST_F(InstructionFusionTest, FloatingPointDivIsCheap) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  Add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }
  ENTRY TestComputation {
    zero = f32[] constant(0)
    p0 = f32[100] parameter(0)
    p1 = f32[100] parameter(1)
    recip = f32[100] divide(p1, p0)
    sum1 = f32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    sum2 = f32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    ROOT root = (f32[], f32[]) tuple(sum1, sum2)
  })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::Fusion(), op::Fusion()))
      << module->ToString();
}

// Compute sum(100/p0), where p0 has type s32, twice.  Check that the division
// is *not* duplicated and fused into both reduces, because we say that integer
// division is not cheap.
TEST_F(InstructionFusionTest, IntegerDivIsNotCheap) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  Add {
    lhs = s32[] parameter(0)
    rhs = s32[] parameter(1)
    ROOT add = s32[] add(lhs, rhs)
  }
  ENTRY TestComputation {
    zero = s32[] constant(0)
    p0 = s32[100] parameter(0)
    p1 = s32[100] parameter(1)
    recip = s32[100] divide(p1, p0)
    sum1 = s32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    sum2 = s32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    ROOT mul = (s32[], s32[]) tuple(sum1, sum2)
  })")
                    .ValueOrDie();

  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie())
      << module->ToString();
}

TEST_F(InstructionFusionTest, DotOutputFusionImpossible) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  ENTRY NoOutputFusion {
    alpha = f32[] constant(3)
    broadcast = f32[4,4]{1,0} broadcast(alpha), dimensions={}
    p0 = f32[4,3]{1,0} parameter(0)
    p1 = f32[3,4]{1,0} parameter(1)
    dot = f32[4,4]{1,0} dot(p0, p1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    d = f32[4,4]{1,0} multiply(dot, dot)
    ROOT mul = f32[4,4] multiply(d, broadcast)
  })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);
  EXPECT_THAT(root->fused_expression_root(),
              op::Multiply(op::Multiply(op::Parameter(), op::Parameter()),
                           op::Broadcast(op::Constant())));
}

// Counts the HLO ops with a given op code in the specified module.
static int Count(const HloModule& module, HloOpcode op) {
  int count = 0;
  for (const auto* computation : module.computations()) {
    for (const auto* instruction : computation->instructions()) {
      if (instruction->opcode() == op) {
        ++count;
      }
    }
  }
  return count;
}

// Returns an HLO instruction from the given computation with the op code.
static StatusOr<const HloInstruction*> FindHloInstruction(
    const HloComputation& computation, HloOpcode op) {
  for (const auto* instruction : computation.instructions()) {
    if (instruction->opcode() == op) {
      return instruction;
    }
  }
  return NotFound(
      "Computation '%s' does not contain an instruction with op code '%s'.",
      computation.name(), HloOpcodeString(op));
}

TEST_F(InstructionFusionTest, MultiOutputFusion) {
  // sub --> add --> tuple
  //  \---------------/
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    ENTRY OutputFusion {
     p0 = f32[4,3]{1,0} parameter(0)
     p1 = f32[4,3]{1,0} parameter(1)
     p2 = f32[4,3]{1,0} parameter(2)
     sub = f32[4,3]{1,0} subtract(p0, p2)
     add = f32[4,3]{1,0} add(sub, p1)
     ROOT tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}) tuple(sub, add)
    })")
                    .ValueOrDie();

  // Multi-output fusion is disabled here and performed in the
  // GpuMultiOutputFusion pass instead.
  ASSERT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

TEST_F(InstructionFusionTest, FuseScalarConstant) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module

  ENTRY FuseScalarConstant {
    p0 = f32[] parameter(0)
    c0 = f32[] constant(1)
    add1 = f32[] add(p0, c0)
    b0 = f32[2]{0} broadcast(add1), dimensions={}
    c1 = f32[2]{0} constant({1, 2})
    ROOT add2 = f32[2]{0} add(b0, c1)
  })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Add(op::Broadcast(op::Add(op::Parameter(), op::Constant())),
                      op::Parameter()));
}

// Check that we limit the number of operands to fusions we create.
TEST_F(InstructionFusionTest, AvoidsLargeFusion) {
  constexpr int64_t kNumParams = 200;
  ASSERT_GT(kNumParams, MaxOperandsAndOutputsPerFusion());

  // Compute p0 + p1 + ... + pN.
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {10, 100});
  auto param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p"));
  auto sum = param0;
  for (int64_t i = 1; i < kNumParams; ++i) {
    auto param =
        b.AddInstruction(HloInstruction::CreateParameter(i, shape, "p"));
    sum = b.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, sum, param));
  }
  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(b.Build());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  SCOPED_TRACE(module->ToString());
  for (const HloInstruction* instr : computation->instructions()) {
    EXPECT_LE(instr->operand_count(), MaxOperandsAndOutputsPerFusion())
        << instr->ToString();
  }
}

TEST_F(InstructionFusionTest, FuseIntoScatter) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY FuseIntoScatter {
      p0 = s32[3,3] parameter(0)
      operand = s32[3,3] add(p0, p0)
      p1 = s32[2] parameter(1)
      indices = s32[2] add(p1, p1)
      p2 = s32[2,3] parameter(2)
      updates = s32[2,3] add(p2, p2)
      scatter = s32[3,3] scatter(operand, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
      ROOT add = s32[3,3] add(scatter, scatter)
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Fusion(), op::Fusion()));
  EXPECT_EQ(root->operand(0)->fusion_kind(),
            HloInstruction::FusionKind::kInput);
  EXPECT_THAT(root->operand(0)->fused_expression_root(),
              op::Scatter(op::Add(), op::Add(), op::Add()));
}

TEST_F(InstructionFusionTest, NonscalarConstantsNotFused) {
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
      ROOT reduce = f32[] reduce(broadcast, constant.1), dimensions={0,1,2,3},
                                                         to_apply=add
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  // The f32[16] constant should not be fused into the reduce, but the f32[]
  // constant should be.
  auto* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_instructions_computation()->root_instruction(),
              op::Reduce(op::Broadcast(op::Parameter()), op::Constant()));
}

TEST_F(InstructionFusionTest, FuseReverse) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY Reverse {
      p0 = f32[50,96,1024]{2,1,0} parameter(0)
      add = f32[50,96,1024]{2,1,0} add(p0, p0)
      ROOT reverse = f32[50,96,1024] reverse(add), dimensions={0}
    })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_THAT(root->fused_expression_root(),
              op::Reverse(op::Add(op::Parameter(), op::Parameter())));
}

TEST_F(InstructionFusionTest, GpuIsExpensiveF32) {
  auto m = CreateNewVerifiedModule();
  Shape r0f32 = ShapeUtil::MakeShape(F32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f32, "param0"));

  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kDivide, param0, one));
  HloInstruction* rem = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f32, HloOpcode::kRemainder, param0, one));

  EXPECT_FALSE(GpuInstructionFusion::IsExpensive(*div));
  EXPECT_TRUE(GpuInstructionFusion::IsExpensive(*rem));
}

TEST_F(InstructionFusionTest, GpuIsExpensiveS32) {
  auto m = CreateNewVerifiedModule();
  Shape r0s32 = ShapeUtil::MakeShape(S32, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0s32, "param0"));

  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0s32, HloOpcode::kDivide, param0, one));
  HloInstruction* rem = builder.AddInstruction(
      HloInstruction::CreateBinary(r0s32, HloOpcode::kRemainder, param0, one));

  EXPECT_FALSE(GpuInstructionFusion::IsExpensive(*div));
  EXPECT_FALSE(GpuInstructionFusion::IsExpensive(*rem));
}

TEST_F(InstructionFusionTest, GpuIsExpensiveBroadcastS32) {
  auto m = CreateNewVerifiedModule();
  Shape r1s32 = ShapeUtil::MakeShape(S32, {10});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r1s32, "param0"));

  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* one_broad =
      builder.AddInstruction(HloInstruction::CreateBroadcast(r1s32, one, {}));

  HloInstruction* div = builder.AddInstruction(HloInstruction::CreateBinary(
      r1s32, HloOpcode::kDivide, param0, one_broad));
  HloInstruction* rem = builder.AddInstruction(HloInstruction::CreateBinary(
      r1s32, HloOpcode::kRemainder, param0, one_broad));

  EXPECT_FALSE(GpuInstructionFusion::IsExpensive(*div));
  EXPECT_FALSE(GpuInstructionFusion::IsExpensive(*rem));
}

TEST_F(InstructionFusionTest, FloatingPointExpIsCheap) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module
  Add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }
  ENTRY TestComputation {
    zero = f32[] constant(0)
    p0 = f32[100] parameter(0)
    recip = f32[100] exponential(p0)
    sum1 = f32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    sum2 = f32[] reduce(recip, zero), dimensions={0}, to_apply=Add
    ROOT root = (f32[], f32[]) tuple(sum1, sum2)
  })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Tuple(op::Fusion(), op::Fusion()))
      << module->ToString();
}

}  // namespace gpu
}  // namespace xla
