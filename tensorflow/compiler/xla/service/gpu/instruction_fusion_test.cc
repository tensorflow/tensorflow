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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kExp, const0));
  HloInstruction* broadcast2 =
      builder.AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeShape(S32, {1}), exp1, {0}));

  auto module = CreateNewModule();
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
          ShapeUtil::MakeShape(S32, {1}), negate1, {0}));

  auto module = CreateNewModule();
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kExp, const0));
  HloInstruction* reshape2 = builder.AddInstruction(
      HloInstruction::CreateReshape(ShapeUtil::MakeShape(S32, {}), exp1));

  auto module = CreateNewModule();
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
      HloInstruction::CreateConstant(LiteralUtil::CreateR0(5)));
  HloInstruction* exp1 = builder.AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeShape(S32, {}), HloOpcode::kExp, const0));
  HloInstruction* transpose2 = builder.AddInstruction(
      HloInstruction::CreateTranspose(ShapeUtil::MakeShape(S32, {}), exp1, {}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose2, computation->root_instruction());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  EXPECT_THAT(computation->root_instruction(), op::Fusion());
}

TEST_F(InstructionFusionTest, PotentialBitcastReshapeOfDotUnfused) {
  HloComputation::Builder builder(TestName());
  auto param0 = builder.AddInstruction(HloInstruction::CreateParameter(
      0, ShapeUtil::MakeShape(S32, {1, 1}), "0"));
  auto dot1 = builder.AddInstruction(
      CreateCanonicalDot(ShapeUtil::MakeShape(S32, {1, 1}), param0, param0));
  auto reshape2 = builder.AddInstruction(HloInstruction::CreateReshape(
      ShapeUtil::MakeShape(S32, {1, 1, 1}), dot1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(reshape2, computation->root_instruction());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
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

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_EQ(transpose2, computation->root_instruction());
  EXPECT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie());
}

// Tests that broadcasts fused into a fusion with a reduce root.
TEST_F(InstructionFusionTest, BroadcastIntoReduce) {
  auto module = ParseHloString(R"(
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
  auto module = ParseHloString(R"(
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
  auto module = ParseHloString(R"(
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
  auto module = ParseHloString(R"(
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
  auto module = ParseHloString(R"(
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
  auto module = ParseHloString(R"(
    HloModule test_module

    ENTRY BroadcastIntoAdd {
      p0 = f32[4,1,1]{2,1,0} parameter(0)
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
  auto module = ParseHloString(R"(
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

TEST_F(InstructionFusionTest, DotOutputFusion) {
  auto module = ParseHloString(R"(
  HloModule test_module
  ENTRY OutputFusion {
    alpha = f32[] constant(3)
    broadcast = f32[4,4]{1,0} broadcast(alpha), dimensions={}
    p0 = f32[4,3]{1,0} parameter(0)
    p1 = f32[4,3]{1,0} parameter(1)
    transpose = f32[3,4]{1,0} transpose(p1), dimensions={1, 0}
    dot = f32[4,4]{1,0} dot(p0, transpose), lhs_contracting_dims={1}, rhs_contracting_dims={0}
    ROOT mul = f32[4,4] multiply(dot, broadcast)
  })")
                    .ValueOrDie();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Fusion());
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kOutput);
  EXPECT_THAT(
      root->fused_expression_root(),
      op::Multiply(op::Dot(op::Parameter(), op::Transpose(op::Parameter())),
                   op::Broadcast(op::Constant())));
}

// Compute sum(1/p0), where p0 has type f32, twice.  Check that the division is
// duplicated and fused into both reduces.
TEST_F(InstructionFusionTest, FloatingPointDivIsCheap) {
  auto module = ParseHloString(R"(
  HloModule test_module
  Add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }
  ENTRY TestComputation {
    zero = f32[] constant(0)
    one = f32[] constant(1)
    p0 = f32[100] parameter(0)
    recip = f32[100] divide(one, p0)
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
  auto module = ParseHloString(R"(
  HloModule test_module
  Add {
    lhs = s32[] parameter(0)
    rhs = s32[] parameter(1)
    ROOT add = s32[] add(lhs, rhs)
  }
  ENTRY TestComputation {
    zero = s32[] constant(0)
    one_hundred = s32[] constant(100)
    p0 = s32[100] parameter(0)
    recip = s32[100] divide(one_hundred, p0)
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
  auto module = ParseHloString(R"(
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
  auto module = ParseHloString(R"(
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

  ASSERT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  SCOPED_TRACE(module->ToString());

  // Expect that there is one multi-output fusion and subtract has not been
  // duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1);
  EXPECT_EQ(Count(*module, HloOpcode::kSubtract), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      const HloInstruction* fusion,
      FindHloInstruction(*module->entry_computation(), HloOpcode::kFusion));
  EXPECT_THAT(
      fusion->fused_expression_root(),
      op::Tuple(op::Add(op::Subtract(), op::Parameter()), op::Subtract()));
}

TEST_F(InstructionFusionTest, MultiOutputFusionExpensiveOp) {
  // tanh --> add --> tuple
  //  \---------------/
  auto module = ParseHloString(R"(
    HloModule test_module
    ENTRY OutputFusion {
     p0 = f32[4,3]{1,0} parameter(0)
     p1 = f32[4,3]{1,0} parameter(1)
     tanh = f32[4,3]{1,0} tanh(p0)
     add = f32[4,3]{1,0} add(tanh, p1)
     ROOT tuple = (f32[4,3]{1,0}, f32[4,3]{1,0}) tuple(tanh, add)
    })")
                    .ValueOrDie();

  // TODO(tjoerg): Allow multi-output fusion for expensive operations like tanh.
  ASSERT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie())
      << module->ToString();
}

TEST_F(InstructionFusionTest, MultiOutputFusion2) {
  // sub --> add1 --\--------\
  //  \----------> add2 --> tuple
  auto module = ParseHloString(R"(
    HloModule test_module
    ENTRY OutputFusion {
     p0 = f32[4,3]{1,0} parameter(0)
     p1 = f32[4,3]{1,0} parameter(1)
     p2 = f32[4,3]{1,0} parameter(2)
     sub = f32[4,3]{1,0} subtract(p0, p2)
     add1 = f32[4,3]{1,0} add(sub, p1)
     add2 = f32[4,3]{1,0} add(sub, add1)
     ROOT tuple = (f32[4,3]{1,0}) tuple(add1, add2)
    })")
                    .ValueOrDie();

  ASSERT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  SCOPED_TRACE(module->ToString());

  // Expect that there is one multi-output fusion and subtract has not been
  // duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1);
  EXPECT_EQ(Count(*module, HloOpcode::kSubtract), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      const HloInstruction* fusion,
      FindHloInstruction(*module->entry_computation(), HloOpcode::kFusion));
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Add(op::Subtract(), op::Add()),
                        op::Add(op::Subtract(), op::Parameter())));
}

TEST_F(InstructionFusionTest, MultiOutputFusion3) {
  // sub --> add1 ----\--------\
  //  \ --> add2 --> add3 --> tuple
  auto module = ParseHloString(R"(
    HloModule test_module
    ENTRY OutputFusion {
     p0 = f32[4,3]{1,0} parameter(0)
     p1 = f32[4,3]{1,0} parameter(1)
     p2 = f32[4,3]{1,0} parameter(2)
     p3 = f32[4,3]{1,0} parameter(3)
     sub = f32[4,3]{1,0} subtract(p0, p2)
     add1 = f32[4,3]{1,0} add(sub, p1)
     add2 = f32[4,3]{1,0} add(p2, sub)
     add3 = f32[4,3]{1,0} add(add1, add2)
     ROOT tuple = (f32[4,3]{1,0}) tuple(add3, add2)
    })")
                    .ValueOrDie();

  ASSERT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  SCOPED_TRACE(module->ToString());

  // Expect that there is one multi-output fusion and subtract has not been
  // duplicated.
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1);
  EXPECT_EQ(Count(*module, HloOpcode::kSubtract), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      const HloInstruction* fusion,
      FindHloInstruction(*module->entry_computation(), HloOpcode::kFusion));
  EXPECT_THAT(fusion->fused_expression_root(),
              op::Tuple(op::Add(op::Add(), op::Add()),
                        op::Add(op::Parameter(), op::Subtract())));
}

TEST_F(InstructionFusionTest, NoCyclesDueToMultiOutputFusion) {
  // sub --> mul ---\
  //  \--> call --> add --> tuple
  auto module = ParseHloString(R"(
  HloModule test_module
  ENTRY OutputFusion {
    c = f32[] constant(42)
    p0 = f32[4,3]{1,0} parameter(0)
    p1 = f32[4,3]{1,0} parameter(1)
    sub = f32[4,3]{1,0} subtract(p0, p1)
    mul = f32[4,3]{1,0} multiply(sub, c)
    call = f32[4,3]{1,0} custom-call(sub), custom_call_target="foo"
    add = f32[4,3]{1,0} add(mul, call)
    ROOT tuple = (f32[4,3]{1,0}) tuple(add)
  })")
                    .ValueOrDie();

  ASSERT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  // Visit instructions in post order to detect cycles.
  // TODO(tjoerg): Add cycle detection to the HloVerifier.
  class DummyVisitor : public DfsHloVisitorWithDefault {
   public:
    DummyVisitor() {}
    Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
      return Status::OK();
    }
  } visitor;
  for (const HloComputation* computation : module->MakeComputationPostOrder()) {
    // Accept will return a FailedPrecondition when a cycle is detected.
    EXPECT_TRUE(computation->root_instruction()->Accept(&visitor).ok());
  }
}

TEST_F(InstructionFusionTest, NoMultiOutputFusionWithIncompatibleShapes) {
  // sub[2,3] --> add[4,3] --> tuple([2,3], [4,3])
  //  \-------------------------/
  auto module = ParseHloString(R"(
    HloModule test_module
    ENTRY OutputFusion {
     p0 = f32[2,3]{1,0} parameter(0)
     p1 = f32[4,3]{1,0} parameter(1)
     p2 = f32[2,3]{1,0} parameter(2)
     sub = f32[2,3]{1,0} subtract(p0, p2)
     add = f32[4,3]{1,0} add(sub, p1)
     ROOT tuple = (f32[2,3]{1,0}, f32[4,3]{1,0}) tuple(sub, add)
    })")
                    .ValueOrDie();

  // Multi-output fusion requires shapes to be compatible. Since `sub` and `add`
  // have incompatible shapes, expect that no multi-output fusion happens.
  ASSERT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie())
      << module->ToString();
}

TEST_F(InstructionFusionTest, FuseIntoInputFusionInstruction) {
  auto module = ParseHloString(R"(
  HloModule test_module

  add_computation {
    add_lhs = f32[] parameter(0)
    add_rhs = f32[] parameter(1)
    ROOT add_root = f32[] add(add_lhs, add_rhs)
  }

  fused_computation {
    p1 = f32[10] parameter(0)
    zero = f32[] constant(0)
    ROOT f2_root = f32[] reduce(p1, zero), dimensions={0},
           to_apply=add_computation
  }

  ENTRY entry {
    p0 = f32[10] parameter(0)
    mul = f32[10] multiply(p0, p0)
    fusion = f32[] fusion(mul), kind=kInput, calls=fused_computation
    ROOT tuple = (f32[10], f32[]) tuple(fusion, mul)
  })")
                    .ValueOrDie();

  // Multi-output fusion is not supported for non-loop fusions at present. Since
  // `fused_computation` is a input fusion, expect no multi-output fusion to
  // happen.
  ASSERT_FALSE(GpuInstructionFusion(/*may_duplicate=*/true)
                   .Run(module.get())
                   .ValueOrDie())
      << module->ToString();
}

TEST_F(InstructionFusionTest, FuseScalarConstant) {
  auto module = ParseHloString(R"(
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
  constexpr int64 kNumParams = 200;
  ASSERT_GT(kNumParams, GpuInstructionFusion::kMaxOperandsAndOutputsPerFusion);

  // Compute p0 + p1 + ... + pN.
  HloComputation::Builder b(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {10, 100});
  auto param0 =
      b.AddInstruction(HloInstruction::CreateParameter(0, shape, "p"));
  auto sum = param0;
  for (int64 i = 1; i < kNumParams; ++i) {
    auto param =
        b.AddInstruction(HloInstruction::CreateParameter(i, shape, "p"));
    sum = b.AddInstruction(
        HloInstruction::CreateBinary(shape, HloOpcode::kAdd, sum, param));
  }
  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(b.Build());
  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/true)
                  .Run(module.get())
                  .ValueOrDie());
  SCOPED_TRACE(module->ToString());
  for (const HloInstruction* instr : computation->instructions()) {
    EXPECT_LE(instr->operand_count(),
              GpuInstructionFusion::kMaxOperandsAndOutputsPerFusion)
        << instr->ToString();
  }
}

}  // namespace gpu
}  // namespace xla
