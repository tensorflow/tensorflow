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

#include "xla/hlo/transforms/simplifiers/bfloat16_conversion_folding.h"

#include <cstdint>
#include <optional>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/float_support.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class TestBFloat16Support : public FloatSupport {
 public:
  TestBFloat16Support() : FloatSupport(BF16) {}
  ~TestBFloat16Support() override {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    if (hlo.opcode() == HloOpcode::kAdd ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllReduce) {
      return true;
    }
    return false;
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    if (hlo.opcode() == HloOpcode::kAdd ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllReduce) {
      return true;
    }
    return false;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    if (hlo.opcode() == HloOpcode::kAdd || hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllReduce) {
      return true;
    }
    return false;
  }
};

class BFloat16ConversionFoldingTest : public HloHardwareIndependentTestBase {
 protected:
  BFloat16ConversionFoldingTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/true) {}

  bool FoldConversions(HloModule* module) {
    TestBFloat16Support bfloat16_support_;
    BFloat16ConversionFolding fold(&bfloat16_support_);
    absl::StatusOr<bool> result = fold.Run(module);
    EXPECT_IS_OK(result.status());
    return result.value();
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

  auto module = CreateNewVerifiedModule();
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

  auto module = CreateNewVerifiedModule();
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

  auto module = CreateNewVerifiedModule();
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

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(FoldConversions(module.get()));

  EXPECT_EQ(computation->root_instruction(), convert1);
  EXPECT_EQ(gte->shape().element_type(), F32);
  EXPECT_EQ(tuple->operand(1), convert0);
}

TEST_F(BFloat16ConversionFoldingTest, DoNotFoldAsyncOp) {
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  auto module = CreateNewVerifiedModule();

  auto async_computation_builder = HloComputation::Builder("async_computation");
  HloInstruction* async_a = async_computation_builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "async_a"));
  HloInstruction* async_b = async_computation_builder.AddInstruction(
      HloInstruction::CreateParameter(1, f32_shape, "async_b"));
  HloInstruction* add =
      async_computation_builder.AddInstruction(HloInstruction::CreateBinary(
          f32_shape, HloOpcode::kAdd, async_a, async_b));
  HloComputation* async_computation =
      module->AddEmbeddedComputation(async_computation_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(f32_shape, b));
  HloInstruction* async_start =
      builder.AddInstruction(HloInstruction::CreateAsyncStart(
          ShapeUtil::MakeTupleShape(
              {ShapeUtil::MakeTupleShape({f32_shape, f32_shape}), f32_shape,
               ShapeUtil::MakeScalarShape(U32)}),
          {a, convert0}, async_computation));
  HloInstruction* async_done = builder.AddInstruction(
      HloInstruction::CreateAsyncDone(f32_shape, async_start));
  HloInstruction* convert1 = builder.AddInstruction(
      HloInstruction::CreateConvert(bf16_shape, async_done));
  HloComputation* computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(FoldConversions(module.get()));

  EXPECT_EQ(async_computation->root_instruction(), add);
  EXPECT_EQ(computation->root_instruction(), convert1);
  EXPECT_EQ(async_done->shape().element_type(), F32);
  EXPECT_EQ(async_start->operand(1), convert0);
}

TEST_F(BFloat16ConversionFoldingTest, FoldAllReduceTupleOutput) {
  auto builder = HloComputation::Builder(TestName());

  auto module = CreateNewVerifiedModule();
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

  HloInstruction* crs = builder.AddInstruction(HloInstruction::CreateAllReduce(
      ShapeUtil::MakeTupleShape({f32_shape, f32_shape}), {convert_a, b}, sum,
      /*device_list=*/CollectiveDeviceList(),
      /*constrain_layout=*/false,
      /*channel_id=*/std::nullopt, /*use_global_device_ids=*/false));
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
