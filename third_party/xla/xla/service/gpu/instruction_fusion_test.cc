/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/gpu/instruction_fusion.h"

#include <cstdint>
#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/gpu/gpu_device_info_for_tests.h"
#include "xla/service/gpu/gpu_fusible.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/pattern_matcher_gmock.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/hlo_test_base.h"
#include "xla/tests/test_utils.h"
#include "xla/tests/verified_hlo_module.h"
#include "xla/util.h"
#include "tsl/platform/statusor.h"

namespace m = ::xla::match;

namespace xla {
namespace gpu {

class InstructionFusionTest : public HloTestBase {
 public:
  GpuInstructionFusion duplicating_instruction_fusion_{
      /*may_duplicate=*/true, TestGpuDeviceInfo::RTXA6000DeviceInfo()};
};

TEST_F(InstructionFusionTest, NoFusionIntoCustomFusionConsumer) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<VerifiedHloModule> module,
                          ParseAndReturnVerifiedModule(R"(
HloModule m

c {
  p0 = bf16[3000,53]{1,0} parameter(0)
  p1 = bf16[22,53]{1,0} parameter(1)
  d = bf16[3000,22]{1,0} dot(p0, p1),
    lhs_contracting_dims={1}, rhs_contracting_dims={1}
  r = bf16[1,1,3000,22]{3,2,1,0} reshape(d)
  ROOT c = bf16[1,1,3000,22]{2,1,3,0} copy(r)
}

ENTRY e {
  p1 = bf16[3000,53]{1,0} parameter(1)
  p0 = bf16[22,53]{1,0} parameter(0)
  cp0 = bf16[22,53]{1,0} convert(p0)
  ROOT f = bf16[1,1,3000,22]{2,1,3,0} fusion(p1, cp0), kind=kCustom, calls=c
})"));

  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
}

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
  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
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
  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());
  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Fusion()));
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
  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());
  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Fusion()));
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
  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());
  EXPECT_THAT(computation->root_instruction(), GmockMatch(m::Fusion()));
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
  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());
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
  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
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
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_THAT(
      root->fused_expression_root(),
      GmockMatch(m::Reduce(m::Broadcast(m::Constant()), m::Constant())));
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
                    .value();

  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
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
                    .value();

  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
}

TEST_F(InstructionFusionTest, DoNotRepeatLargeReduceWindow) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = f32[] parameter(0)
      rhs = f32[] parameter(1)
      ROOT add = f32[] add(lhs, rhs)
    }

    ENTRY entry {
      p0 = s32[512,512,2] parameter(0)
      p1 = f32[1,1,512,512] parameter(1)
      constant_1 = f32[] constant(1)
      reduce-window.1 = reduce-window(p1, constant_1),
        window={size=1x1x9x9}, to_apply=add
      ROOT ret = gather(reduce-window.1, p0), offset_dims={0,1,2,3},
        collapsed_slice_dims={}, start_index_map={1,2},
        index_vector_dim=2, slice_sizes={1,1,1,1}
    })")
                    .value();

  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
}

TEST_F(InstructionFusionTest, FuseLayoutChangingOpWithElementwise) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    ENTRY entry {
      p0 = f32[16,16,16,16]{3,2,1,0} parameter(0)
      copy = f32[16,16,16,16]{0,1,2,3} copy(p0)
      ROOT add = f32[16,16,16,16]{0,1,2,3} add(copy, copy)
    })")
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_THAT(root->fused_expression_root(),
              GmockMatch(m::Add(m::Copy(), m::Copy())));
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
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_THAT(root->fused_expression_root(),
              GmockMatch(m::Add(m::Bitcast(m::Parameter()), m::Parameter())));
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
                    .value();

  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
}

TEST_F(InstructionFusionTest, ConvertIntoBitcastBothConsumedByTuple) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test

  ENTRY main {
    param_0 = f32[2048,16000]{1,0} parameter(0)
    convert = bf16[2048,16000]{1,0} convert(param_0)
    bitcast = bf16[16000,1,2048]{2,1,0} bitcast(convert)
    ROOT tuple.143 = (bf16[16000,1,2048]{2,1,0}, bf16[2048,16000]{1,0}) tuple(bitcast, convert)
  })")
                    .value();
  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
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
                    .value();

  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
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
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(m::Fusion(), m::Fusion())))
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
                    .value();

  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value())
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
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kLoop);
  EXPECT_THAT(
      root->fused_expression_root(),
      GmockMatch(m::Multiply(m::Multiply(m::Parameter(), m::Parameter()),
                             m::Broadcast(m::Constant()))));
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
                    .value();

  // Multi-output fusion is disabled here and performed in the
  // GpuMultiOutputFusion pass instead.
  ASSERT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
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
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_THAT(
      root->fused_expression_root(),
      GmockMatch(m::Add(m::Broadcast(m::Add(m::Parameter(), m::Constant())),
                        m::Parameter())));
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
  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());
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
      p1 = s32[2] parameter(1)
      indices = s32[2] add(p1, p1)
      p2 = s32[2,3] parameter(2)
      updates = s32[2,3] add(p2, p2)
      scatter = s32[3,3] scatter(p0, indices, updates),
          to_apply=add,
          update_window_dims={1},
          inserted_window_dims={0},
          scatter_dims_to_operand_dims={0},
          index_vector_dim=1
      ROOT add = s32[3,3] add(scatter, scatter)
    })")
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Add(m::Fusion(&fusion), m::Fusion())));
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kInput);
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Scatter(m::Parameter(), m::Add(), m::Add())));
}

TEST_F(InstructionFusionTest, DontFuseIntoFirstOperandOfScatter) {
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
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  const HloInstruction* fusion = nullptr;
  ASSERT_THAT(root, GmockMatch(m::Add(m::Fusion(&fusion), m::Fusion())));
  EXPECT_EQ(fusion->fusion_kind(), HloInstruction::FusionKind::kInput);
  EXPECT_THAT(fusion->fused_expression_root(),
              GmockMatch(m::Scatter(m::Parameter(), m::Add(), m::Add())));
}

TEST_F(InstructionFusionTest, ScatterOpShouldNotFuseWithSharedOperand) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module

  add {
    lhs = f32[] parameter(0)
    rhs = f32[] parameter(1)
    ROOT add = f32[] add(lhs, rhs)
  }

  ENTRY Test {
    parameter.0 = f32[8,8] parameter(0)
    parameter.1 = s32[7] parameter(1)
    indices = s32[7] add(parameter.1, parameter.1)
    slice = f32[7,8] slice(parameter.0), slice={[0:7],[0:8]}
    ROOT scatter = f32[8,8] scatter(parameter.0, indices, slice),
        to_apply=add,
        update_window_dims={1},
        inserted_window_dims={0},
        scatter_dims_to_operand_dims={0},
        index_vector_dim=1
  })")
                    .value();
  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());
  // Verify that we don't fuse scatter and slice together since
  // scatter modifies the input buffer in-place, which is also used
  // as slice's input, and we don't know where the scatter indices point to.
  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(
      root, GmockMatch(m::Fusion(m::Parameter(), m::Slice(), m::Parameter())));
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
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());
  // The f32[16] constant should not be fused into the reduce, but the f32[]
  // constant should be.
  auto* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_THAT(
      root->fused_instructions_computation()->root_instruction(),
      GmockMatch(m::Reduce(m::Broadcast(m::Parameter()), m::Constant())));
}

TEST_F(InstructionFusionTest, FuseReverse) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    ENTRY Reverse {
      p0 = f32[50,96,1024]{2,1,0} parameter(0)
      add = f32[50,96,1024]{2,1,0} add(p0, p0)
      ROOT reverse = f32[50,96,1024] reverse(add), dimensions={0}
    })")
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_THAT(root->fused_expression_root(),
              GmockMatch(m::Reverse(m::Add(m::Parameter(), m::Parameter()))));
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
  HloInstruction* sqrt = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kSqrt, param0));
  HloInstruction* rsqrt = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kRsqrt, param0));
  HloInstruction* exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f32, HloOpcode::kExp, param0));

  EXPECT_FALSE(GpuInstructionFusion::IsExpensive(*div));
  EXPECT_TRUE(GpuInstructionFusion::IsExpensive(*rem));
  EXPECT_FALSE(GpuInstructionFusion::IsExpensive(*sqrt));
  EXPECT_FALSE(GpuInstructionFusion::IsExpensive(*rsqrt));
  EXPECT_FALSE(GpuInstructionFusion::IsExpensive(*exp));
}

TEST_F(InstructionFusionTest, GpuIsExpensiveF64) {
  auto m = CreateNewVerifiedModule();
  Shape r0f64 = ShapeUtil::MakeShape(F64, {});
  HloComputation::Builder builder(TestName());
  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, r0f64, "param0"));

  HloInstruction* one = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1.0f)));
  HloInstruction* div = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f64, HloOpcode::kDivide, param0, one));
  HloInstruction* rem = builder.AddInstruction(
      HloInstruction::CreateBinary(r0f64, HloOpcode::kRemainder, param0, one));
  HloInstruction* sqrt = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f64, HloOpcode::kSqrt, param0));
  HloInstruction* rsqrt = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f64, HloOpcode::kRsqrt, param0));
  HloInstruction* exp = builder.AddInstruction(
      HloInstruction::CreateUnary(r0f64, HloOpcode::kExp, param0));

  EXPECT_TRUE(GpuInstructionFusion::IsExpensive(*div));
  EXPECT_TRUE(GpuInstructionFusion::IsExpensive(*rem));
  EXPECT_TRUE(GpuInstructionFusion::IsExpensive(*sqrt));
  EXPECT_TRUE(GpuInstructionFusion::IsExpensive(*rsqrt));
  EXPECT_TRUE(GpuInstructionFusion::IsExpensive(*exp));
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
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, GmockMatch(m::Tuple(m::Fusion(), m::Fusion())))
      << module->ToString();
}

TEST_F(InstructionFusionTest, SmallReducedDimensionIsNotLoweredToLoop) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module

    add {
      lhs = s32[] parameter(0)
      rhs = s32[] parameter(1)
      ROOT add = s32[] add(lhs, rhs)
    }

    ENTRY FuseSmallReduction {
      p0 = s32[1048576,4] parameter(0)
      p1 = s32[1048576,4] parameter(1)
      sum = s32[1048576,4] add(p0, p1)
      init = s32[] constant(0)
      ROOT reduce = s32[1048576] reduce(sum, init), dimensions={1}, to_apply=add
    })")
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* root = module->entry_computation()->root_instruction();
  ASSERT_THAT(root, GmockMatch(m::Fusion()));
  EXPECT_EQ(root->fusion_kind(), HloInstruction::FusionKind::kInput);
}

TEST_F(InstructionFusionTest, IotaIntoVariadicReduction) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule m

  f {
    tmp_0 = f32[] parameter(0)
    tmp_1 = f32[] parameter(1)
    tmp_2 = pred[] compare(tmp_0, tmp_1), direction=GE
    tmp_3 = f32[] select(tmp_2, tmp_0, tmp_1)
    tmp_4 = pred[] compare(tmp_0, tmp_1), direction=EQ
    tmp_5 = s32[] parameter(2)
    tmp_6 = s32[] parameter(3)
    tmp_7 = s32[] minimum(tmp_5, tmp_6)
    tmp_8 = s32[] select(tmp_2, tmp_5, tmp_6)
    tmp_9 = s32[] select(tmp_4, tmp_7, tmp_8)
    ROOT tmp_10 = (f32[], s32[]) tuple(tmp_3, tmp_9)
  }

  minmax {
    tmp_0 = f32[] parameter(0)
    tmp_1 = f32[] parameter(2)
    tmp_2 = s32[] parameter(1)
    tmp_3 = s32[] parameter(3)
    ROOT tmp_4 = (f32[], s32[]) fusion(tmp_0, tmp_1, tmp_2, tmp_3), kind=kLoop, calls=f
  }

  ENTRY e {
    tmp_0 = f32[554112,10]{1,0} parameter(0)
    tmp_1 = s32[554112,10]{1,0} iota(), iota_dimension=1
    tmp_2 = f32[] constant(-inf)
    tmp_3 = s32[] constant(0)
    ROOT tmp_4 = (f32[554112]{0}, s32[554112]{0}) reduce(tmp_0, tmp_1, tmp_2, tmp_3), dimensions={1}, to_apply=minmax
  })")
                    .value();

  EXPECT_TRUE(GpuInstructionFusion(/*may_duplicate=*/false,
                                   TestGpuDeviceInfo::RTXA6000DeviceInfo())
                  .Run(module.get())
                  .value());
  ASSERT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Fusion(m::Parameter())));
  EXPECT_THAT(
      module->entry_computation()->root_instruction()->fused_expression_root(),
      GmockMatch(
          m::Reduce(m::Parameter(), m::Iota(), m::Constant(), m::Constant())));
}

TEST_F(InstructionFusionTest, InputReductionFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
    HloModule test_module
    add.clone.13 {
      x.27 = f32[] parameter(0)
      y.27 = f32[] parameter(1)
      ROOT add.1036 = f32[] add(x.27, y.27)
    }
    add.clone.14 {
      x.28 = f32[] parameter(0)
      y.28 = f32[] parameter(1)
      ROOT add.1037 = f32[] add(x.28, y.28)
    }
    add {
      x = bf16[] parameter(0)
      convert.448 = f32[] convert(x)
      y = bf16[] parameter(1)
      convert.449 = f32[] convert(y)
      add.597 = f32[] add(convert.448, convert.449)
      ROOT convert.450 = bf16[] convert(add.597)
    }
    ENTRY FuseSmallReduction {
      param_2.7 = bf16[8,16,64,2048]{3,2,1,0} parameter(2)
      convert.1395 = f32[8,16,64,2048]{3,2,1,0} convert(param_2.7)
      param_0.85 = bf16[8,16,64,2048]{3,2,1,0} parameter(0)
      convert.1393 = f32[8,16,64,2048]{3,2,1,0} convert(param_0.85)
      multiply.1652 = f32[8,16,64,2048]{3,2,1,0} multiply(convert.1395, convert.1393)
      convert.1392 = bf16[8,16,64,2048]{3,2,1,0} convert(multiply.1652)
      bitcast.15934 = bf16[128,64,2048]{2,1,0} bitcast(convert.1392)
      convert.1391 = f32[128,64,2048]{2,1,0} convert(bitcast.15934)
      param_1.15 = bf16[] parameter(1)
      convert.1394 = f32[] convert(param_1.15)
      reduce.462 = f32[128,64]{1,0} reduce(convert.1391, convert.1394), dimensions={2}, to_apply=add.clone.13
      reduce.121 = f32[64]{0} reduce(reduce.462, convert.1394), dimensions={0}, to_apply=add.clone.14
      ROOT convert.890 = bf16[64]{0} convert(reduce.121)
    })")
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* fused_convert_fusion =
      module->entry_computation()->root_instruction();

  ASSERT_THAT(fused_convert_fusion, GmockMatch(m::Fusion()));
  SCOPED_TRACE(module->ToString());
  EXPECT_EQ(fused_convert_fusion->fusion_kind(),
            HloInstruction::FusionKind::kInput);
}

TEST_F(InstructionFusionTest, DotStrengthReductionFusion) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module

scalar_add_computation {
  scalar_rhs = f32[] parameter(1)
  scalar_lhs = f32[] parameter(0)
  ROOT add.1 = f32[] add(scalar_lhs, scalar_rhs)
}

ENTRY main {
  param_1.3 = f16[16,64,96,6,2,16]{5,4,3,2,1,0} parameter(1)
  param_0.6 = f16[16,64,96,1,2,16]{5,4,3,2,1,0} parameter(0)
  bitcast.26 = f16[16,64,96,2,16]{4,3,2,1,0} bitcast(param_0.6)
  broadcast.4 = f16[16,64,96,6,2,16]{5,4,3,2,1,0} broadcast(bitcast.26), dimensions={0,1,2,4,5}
  multiply.4 = f16[16,64,96,6,2,16]{5,4,3,2,1,0} multiply(broadcast.4, param_1.3)
  convert.8 = f32[16,64,96,6,2,16]{5,4,3,2,1,0} convert(multiply.4)
  constant_2 = f32[] constant(0)
  reduce.3 = f32[16,64,96,6,2]{3,4,2,1,0} reduce(convert.8, constant_2), dimensions={5}, to_apply=scalar_add_computation
  bitcast.25 = f32[16,64,96,2,6]{4,3,2,1,0} bitcast(reduce.3)
  convert.7 = f16[16,64,96,2,6]{4,3,2,1,0} convert(bitcast.25)
  ROOT bitcast.24 = f16[16,64,96,2,1,6]{5,4,3,2,1,0} bitcast(convert.7)
})")
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  const HloInstruction* fused_convert_fusion =
      module->entry_computation()->root_instruction()->operand(0);

  ASSERT_THAT(fused_convert_fusion, GmockMatch(m::Fusion()));
  SCOPED_TRACE(module->ToString());
  EXPECT_EQ(fused_convert_fusion->fusion_kind(),
            HloInstruction::FusionKind::kInput);
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1);
}

TEST_F(InstructionFusionTest, ReductionFusionOtherUnaryElementwiseOpsAreFused) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module

scalar_add_computation {
  scalar_rhs = f32[] parameter(1)
  scalar_lhs = f32[] parameter(0)
  ROOT add.1 = f32[] add(scalar_lhs, scalar_rhs)
}

ENTRY main {
  param_0 = f16[64,96,6,16]{3,2,1,0} parameter(0)
  constant_2 = f32[] constant(0)
  reduce.3 = f32[64,6,16]{2,1,0} reduce(param_0, constant_2), dimensions={1}, to_apply=scalar_add_computation
  negate = f32[64,6,16]{2,1,0} negate(reduce.3)
  ROOT sine = f16[64,6,16]{2,1,0} sine(negate)
})")
                    .value();

  EXPECT_TRUE(duplicating_instruction_fusion_.Run(module.get()).value());

  HloInstruction* fused_convert_fusion =
      module->entry_computation()->root_instruction();

  ASSERT_THAT(fused_convert_fusion, GmockMatch(m::Fusion()));
  SCOPED_TRACE(module->ToString());
  EXPECT_EQ(fused_convert_fusion->fusion_kind(),
            HloInstruction::FusionKind::kInput);
  EXPECT_EQ(Count(*module, HloOpcode::kFusion), 1);
}

TEST_F(InstructionFusionTest, DoNotFuseInsideReducer) {
  auto module = ParseAndReturnVerifiedModule(R"(
  HloModule test_module

scalar_add_computation {
  scalar_rhs = f32[] parameter(1)
  scalar_lhs = f32[] parameter(0)
  add.1 = f32[] add(scalar_lhs, scalar_rhs)
  ROOT add.2 = f32[] add(add.1, scalar_rhs)
}

ENTRY main {
  param_0 = f16[64,96] parameter(0)
  constant_2 = f32[] constant(0)
  ROOT reduce = f32[64] reduce(param_0, constant_2), dimensions={1}, to_apply=scalar_add_computation
})")
                    .value();

  EXPECT_FALSE(duplicating_instruction_fusion_.Run(module.get()).value());
  SCOPED_TRACE(module->ToString());
}

}  // namespace gpu
}  // namespace xla
