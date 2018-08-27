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

#include "tensorflow/compiler/xla/service/bfloat16_conversion_folding.h"
#include "tensorflow/compiler/xla/service/bfloat16_support.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

class TestBFloat16Support : public BFloat16Support {
 public:
  TestBFloat16Support() {}
  ~TestBFloat16Support() override {}

  bool SupportsBF16Operand(const HloInstruction& hlo,
                           int64 operand_index) const override {
    if (hlo.opcode() == HloOpcode::kAdd ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kCrossReplicaSum) {
      return true;
    }
    return false;
  }

  bool SupportsBF16Output(const HloInstruction& hlo) const override {
    if (hlo.opcode() == HloOpcode::kAdd ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kCrossReplicaSum) {
      return true;
    }
    return false;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    if (hlo.opcode() == HloOpcode::kAdd || hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kCrossReplicaSum) {
      return true;
    }
    return false;
  }
};

class BFloat16ConversionFoldingTest : public HloTestBase {
 protected:
  bool FoldConversions(HloModule* module) {
    TestBFloat16Support bfloat16_support_;
    BFloat16ConversionFolding fold(&bfloat16_support_);
    StatusOr<bool> result = fold.Run(module);
    EXPECT_IS_OK(result.status());
    return result.ValueOrDie();
  }
};

TEST_F(BFloat16ConversionFoldingTest, FoldIfSupported) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kAdd, a, b));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, add0));
  HloInstruction* convert1 = builder.AddInstruction(
      HloInstruction::CreateConvert(f32_shape, convert0));

  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kAdd, convert1, c));
  builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, add1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), add1);
  EXPECT_EQ(add0->shape().element_type(), BF16);
  EXPECT_EQ(add1->shape().element_type(), BF16);
  EXPECT_EQ(add1->operand(0), add0);
}

TEST_F(BFloat16ConversionFoldingTest, DoNotFoldIfUnsupported) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* mul0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kMultiply, a, b));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, mul0));
  HloInstruction* convert1 = builder.AddInstruction(
      HloInstruction::CreateConvert(f32_shape, convert0));

  HloInstruction* mul1 = builder.AddInstruction(HloInstruction::CreateBinary(
      f32_shape, HloOpcode::kMultiply, convert1, c));
  HloInstruction* convert2 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, mul1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), convert2);
  EXPECT_EQ(mul0->shape().element_type(), F32);
  EXPECT_EQ(mul1->shape().element_type(), F32);
  EXPECT_EQ(mul1->operand(0), convert1);
}

TEST_F(BFloat16ConversionFoldingTest, DoNotFoldUnsupportedMixedPrecision) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* sub0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kSubtract, a, b));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, sub0));
  HloInstruction* convert1 = builder.AddInstruction(
      HloInstruction::CreateConvert(f32_shape, convert0));

  HloInstruction* sub1 = builder.AddInstruction(HloInstruction::CreateBinary(
      f32_shape, HloOpcode::kSubtract, convert1, c));
  HloInstruction* convert2 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, sub1));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), convert2);
  EXPECT_EQ(sub0->shape().element_type(), F32);
  EXPECT_EQ(sub1->shape().element_type(), F32);
  EXPECT_EQ(sub1->operand(0), convert1);
}

TEST_F(BFloat16ConversionFoldingTest, DoNotFoldTuple) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(f32_shape, b));

  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({a, convert0}));
  HloInstruction* gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, tuple, 0));
  HloInstruction* convert1 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, gte));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), convert1);
  EXPECT_EQ(gte->shape().element_type(), F32);
  EXPECT_EQ(tuple->operand(1), convert0);
}

TEST_F(BFloat16ConversionFoldingTest, FoldCrossReplicaSumTupleOutput) {
  auto builder = HloComputation::Builder(TestName());

  auto module = CreateNewModule();
  HloComputation::Builder sum_builder("add");
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "y"));
  sum_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, x, y));
  HloComputation* sum = module->AddEmbeddedComputation(sum_builder.Build());

  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_shape, "a"));
  HloInstruction* convert_a =
      builder.AddInstruction(HloInstruction::CreateConvert(f32_shape, a));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32_shape, "b"));

  HloInstruction* crs =
      builder.AddInstruction(HloInstruction::CreateCrossReplicaSum(
          ShapeUtil::MakeTupleShape({f32_shape, f32_shape}), {convert_a, b},
          sum, /*replica_groups=*/{}, /*barrier=*/"",
          /*all_reduce_id=*/absl::nullopt));
  HloInstruction* gte_a = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, crs, 0));
  HloInstruction* gte_b = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, crs, 1));
  HloInstruction* convert_gte_b =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, gte_b));
  HloInstruction* tuple = builder.AddInstruction(
      HloInstruction::CreateTuple({gte_a, convert_gte_b}));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), tuple);
  EXPECT_EQ(tuple->operand(0), gte_a);
  EXPECT_EQ(tuple->operand(1), gte_b);
  EXPECT_EQ(gte_a->shape().element_type(), F32);
  EXPECT_EQ(gte_b->shape().element_type(), BF16);
  EXPECT_EQ(crs->operand(0), a);
  EXPECT_EQ(crs->operand(1), b);
  EXPECT_EQ(a->shape().element_type(), BF16);
  EXPECT_EQ(b->shape().element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(crs->shape(), {0}).element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(crs->shape(), {1}).element_type(), BF16);
}

}  // namespace xla
