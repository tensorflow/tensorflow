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

#include "xla/hlo/transforms/simplifiers/float_normalization.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/service/float_support.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_verifier.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {
namespace m = match;

class TestFloatSupport : public FloatSupport {
 public:
  explicit TestFloatSupport(PrimitiveType low_precision_type,
                            PrimitiveType high_precision_type)
      : FloatSupport(low_precision_type, high_precision_type) {}
  ~TestFloatSupport() override = default;

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    if (hlo.opcode() == HloOpcode::kAdd ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kReduce ||
        hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllToAll) {
      return true;
    }
    if (hlo.opcode() == HloOpcode::kDot) {
      // Test that only the first operand of kDot supports low-precision.
      return operand_index == 0;
    }
    return false;
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    if (hlo.opcode() == HloOpcode::kAdd || hlo.opcode() == HloOpcode::kReduce ||
        hlo.opcode() == HloOpcode::kSubtract ||
        hlo.opcode() == HloOpcode::kDot || hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllToAll) {
      return true;
    }
    return false;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    if (hlo.opcode() == HloOpcode::kAdd || hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement) {
      return true;
    }
    return false;
  }
};

// The test float class that doesn't support any compute ops for low-precision
// but supports some collectives.
class TestFloatNoComputeSupport : public FloatSupport {
 public:
  explicit TestFloatNoComputeSupport(PrimitiveType low_precision_type,
                                     PrimitiveType high_precision_type)
      : FloatSupport(low_precision_type, high_precision_type) {}
  ~TestFloatNoComputeSupport() override = default;

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    if (hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllToAll ||
        hlo.opcode() == HloOpcode::kAllReduce ||
        hlo.opcode() == HloOpcode::kReduceScatter) {
      return true;
    }
    return false;
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    if (hlo.opcode() == HloOpcode::kTuple ||
        hlo.opcode() == HloOpcode::kGetTupleElement ||
        hlo.opcode() == HloOpcode::kAllToAll ||
        hlo.opcode() == HloOpcode::kAllReduce ||
        hlo.opcode() == HloOpcode::kReduceScatter) {
      return true;
    }
    return false;
  }
};

class FloatNormalizationTest : public HloHardwareIndependentTestBase {
 protected:
  FloatNormalizationTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/true) {}

  bool Normalize(HloModule* module, PrimitiveType low_precision_type = BF16,
                 PrimitiveType high_precision_type = F32) {
    TestFloatSupport float_support(low_precision_type, high_precision_type);
    FloatNormalization normalization(&float_support);
    absl::StatusOr<bool> result = normalization.Run(module);
    EXPECT_IS_OK(result.status());

    HloVerifier verifier(/*layout_sensitive=*/false,
                         /*allow_mixed_precision=*/true);
    EXPECT_IS_OK(verifier.Run(module).status());

    return result.value();
  }
};

class FloatNormalizationF8Test
    : public FloatNormalizationTest,
      public ::testing::WithParamInterface<PrimitiveType> {};

INSTANTIATE_TEST_SUITE_P(FloatNormalizationF8Suite, FloatNormalizationF8Test,
                         ::testing::Values(F4E2M1FN, F8E3M4, F8E4M3,
                                           F8E4M3B11FNUZ, F8E4M3FN, F8E4M3FNUZ,
                                           F8E5M2, F8E5M2FNUZ, F8E8M0FNU));

TEST_F(FloatNormalizationTest, NoopIfSupported) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kAdd, a, b));

  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kAdd, add0, c));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction(), add1);
  EXPECT_EQ(add0->shape().element_type(), BF16);
  EXPECT_EQ(add1->shape().element_type(), F32);
}

TEST_F(FloatNormalizationTest, ResolveIfUnsupportedBF16) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* mul0 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kMultiply, a, b));

  HloInstruction* mul1 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kMultiply, mul0, c));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(computation->root_instruction()->operand(0), mul1);
  EXPECT_EQ(mul0->shape().element_type(), F32);
  EXPECT_EQ(mul1->shape().element_type(), F32);
  EXPECT_EQ(mul1->operand(0)->opcode(), HloOpcode::kConvert);
}

TEST_F(FloatNormalizationTest, ResolveUnsupportedMixedPrecisionSubtraction) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f32_shape, "c"));

  HloInstruction* sub0 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kSubtract, a, b));

  HloInstruction* sub1 = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kSubtract, sub0, c));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(computation->root_instruction()->operand(0), sub1);
  EXPECT_EQ(sub0->shape().element_type(), F32);
  EXPECT_EQ(sub1->shape().element_type(), F32);
  EXPECT_EQ(sub1->operand(0)->opcode(), HloOpcode::kConvert);
}

TEST_F(FloatNormalizationTest, ResolveUnsupportedMixedPrecisionReduce) {
  Shape f32_input_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape f32_output_shape = ShapeUtil::MakeShape(F32, {4});

  Shape bf16_scalar_shape = ShapeUtil::MakeShape(BF16, {});

  auto reduce_comp_builder = HloComputation::Builder("reduce_comp");
  auto reduce_comp_param0 = reduce_comp_builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_scalar_shape, "param0"));
  auto reduce_comp_param1 = reduce_comp_builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_scalar_shape, "param1"));
  reduce_comp_builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_scalar_shape, HloOpcode::kAdd,
                                   reduce_comp_param0, reduce_comp_param1));

  auto module = CreateNewVerifiedModule();
  auto reduce_computation =
      module->AddEmbeddedComputation(reduce_comp_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  HloInstruction* input = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_input_shape, "a"));
  HloInstruction* init = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_scalar_shape, "init"));
  HloInstruction* reduce = builder.AddInstruction(HloInstruction::CreateReduce(
      f32_output_shape, input, init, {0}, reduce_computation));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction(), reduce);
  EXPECT_EQ(reduce->called_computations().size(), 1);
  EXPECT_EQ(reduce->called_computations()[0]->num_parameters(), 2);
  EXPECT_EQ(reduce->called_computations()[0]
                ->parameter_instruction(0)
                ->shape()
                .element_type(),
            F32);
  EXPECT_EQ(reduce->called_computations()[0]
                ->parameter_instruction(1)
                ->shape()
                .element_type(),
            F32);
  EXPECT_EQ(reduce->called_computations()[0]
                ->root_instruction()
                ->shape()
                .element_type(),
            F32);
  EXPECT_EQ(reduce->shape().element_type(), F32);
  EXPECT_EQ(reduce->operand(0), input);
  EXPECT_EQ(input->shape().element_type(), F32);
  EXPECT_EQ(reduce->operand(1)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(reduce->operand(1)->shape().element_type(), F32);
}

TEST_F(FloatNormalizationTest, ResolveMixedPrecisionTupleAllReduce) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder sum_builder("sum");
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(F32, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(F32, {}), "y"));
  sum_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kAdd, x, y));
  HloComputation* reduction =
      module->AddEmbeddedComputation(sum_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));

  HloInstruction* crs = builder.AddInstruction(HloInstruction::CreateAllReduce(
      ShapeUtil::MakeTupleShape({f32_shape, bf16_shape}), {a, b}, reduction,
      /*device_list=*/CollectiveDeviceList(),
      /*constrain_layout=*/false,
      /*channel_id=*/std::nullopt,
      /*use_global_device_ids=*/false));
  builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(bf16_shape, crs, 1));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->shape().element_type(), BF16);
  EXPECT_EQ(crs->operand(1)->shape().element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(crs->shape(), {1}).element_type(), F32);
}

TEST_F(FloatNormalizationTest, ResolveMixedPrecisionTupleAllToAllToBF16) {
  auto module = CreateNewVerifiedModule(TestName(), /*replica_count=*/2);

  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));

  std::vector<ReplicaGroup> replica_groups(1);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(1);
  HloInstruction* a2a = builder.AddInstruction(HloInstruction::CreateAllToAll(
      ShapeUtil::MakeTupleShape({bf16_shape, bf16_shape}), {a, a},
      CollectiveDeviceList(replica_groups), /*constrain_layout=*/false,
      std::nullopt));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction(), a2a);
  EXPECT_EQ(ShapeUtil::GetSubshape(a2a->shape(), {0}).element_type(), BF16);
  EXPECT_EQ(ShapeUtil::GetSubshape(a2a->shape(), {1}).element_type(), BF16);
  EXPECT_EQ(a2a->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(a2a->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(a2a->operand(1)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(a2a->operand(1)->shape().element_type(), BF16);
}

TEST_F(FloatNormalizationTest, ResolveMixedPrecisionTupleAllToAllToF32) {
  auto module = CreateNewVerifiedModule(TestName(), /*replica_count=*/2);

  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "a"));

  std::vector<ReplicaGroup> replica_groups(1);
  replica_groups[0].add_replica_ids(0);
  replica_groups[0].add_replica_ids(1);
  HloInstruction* a2a = builder.AddInstruction(HloInstruction::CreateAllToAll(
      ShapeUtil::MakeTupleShape({bf16_shape, f32_shape}), {a, a},
      CollectiveDeviceList(replica_groups), /*constrain_layout=*/false,
      std::nullopt));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(ShapeUtil::GetSubshape(a2a->shape(), {0}).element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(a2a->shape(), {1}).element_type(), F32);
  EXPECT_EQ(a2a->operand(0)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(a2a->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(a2a->operand(1)->opcode(), HloOpcode::kParameter);
  EXPECT_EQ(a2a->operand(1)->shape().element_type(), F32);
}

TEST_F(FloatNormalizationTest, ResolveMixedPrecisionTupleSort) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {1024});
  Shape s32_shape = ShapeUtil::MakeShape(BF16, {1024});

  HloInstruction* key = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "key"));
  HloInstruction* value = builder.AddInstruction(
      HloInstruction::CreateParameter(1, s32_shape, "value"));

  TF_ASSERT_OK_AND_ASSIGN(
      auto* sort,
      MakeSortHlo(ShapeUtil::MakeTupleShape({bf16_shape, s32_shape}),
                  {key, value}, 0, /*is_stable=*/false, &builder,
                  module.get()));
  builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(bf16_shape, sort, 0));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->shape().element_type(), BF16);
  EXPECT_EQ(sort->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(sort->shape(), {0}).element_type(), F32);
}

TEST_F(FloatNormalizationTest, ResolveMixedPrecisionTupleSortRoot) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {1024});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {1024});

  HloInstruction* key = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "key"));
  HloInstruction* value = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "value"));

  TF_ASSERT_OK_AND_ASSIGN(
      auto* sort,
      MakeSortHlo(ShapeUtil::MakeTupleShape({bf16_shape, f32_shape}),
                  {key, value}, 0, /*is_stable=*/false, &builder,
                  module.get()));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(sort->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(ShapeUtil::GetSubshape(sort->shape(), {0}).element_type(), F32);
  EXPECT_NE(computation->root_instruction(), sort);
  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(sort->to_apply()->parameter_instruction(1)->shape().element_type(),
            F32);
  // Make sure that no convert to BF16 was added to the 'to_apply' comparison
  // computation.
  auto users = sort->to_apply()->parameter_instruction(1)->users();
  for (auto user : users) {
    EXPECT_NE(user->opcode(), HloOpcode::kConvert);
  }
}

// Tests that the normalization should not cause unsupported mixed precision due
// to resolving unsupported BF16 operand.
TEST_F(FloatNormalizationTest, DoNotAddUnsupportedMixedPrecision) {
  auto builder = HloComputation::Builder(TestName());
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {4, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  PrecisionConfig precision_config;
  precision_config.mutable_operand_precision()->Resize(
      2, PrecisionConfig::DEFAULT);
  HloInstruction* dot = builder.AddInstruction(
      HloInstruction::CreateDot(bf16_shape, a, b, dot_dnums, precision_config));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get()));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(dot->shape().element_type(), F32);
  EXPECT_EQ(dot->operand(0)->shape().element_type(), F32);
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(dot->operand(1)->shape().element_type(), F32);
  EXPECT_EQ(dot->operand(1)->opcode(), HloOpcode::kConvert);
}

TEST_F(FloatNormalizationTest, DoNotChangeBitcastConvert) {
  auto builder = HloComputation::Builder(TestName());
  Shape u16_shape = ShapeUtil::MakeShape(U16, {4, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {4, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, u16_shape, "a"));

  builder.AddInstruction(HloInstruction::CreateBitcastConvert(bf16_shape, a));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(Normalize(module.get()));
  auto root = computation->root_instruction();

  EXPECT_EQ(root->opcode(), HloOpcode::kBitcastConvert);
  EXPECT_EQ(root->shape().element_type(), BF16);
  EXPECT_EQ(root->operand(0)->shape().element_type(), U16);
}

TEST_P(FloatNormalizationF8Test, ResolveIfUnsupportedF8) {
  PrimitiveType f8_type = GetParam();
  auto builder = HloComputation::Builder(TestName());
  Shape f16_shape = ShapeUtil::MakeShape(F16, {2, 4});
  Shape f8_shape = ShapeUtil::MakeShape(f8_type, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f16_shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, f8_shape, "b"));
  HloInstruction* c = builder.AddInstruction(
      HloInstruction::CreateParameter(2, f16_shape, "c"));

  HloInstruction* mul0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f8_shape, HloOpcode::kMultiply, a, b));

  HloInstruction* mul1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f8_shape, HloOpcode::kMultiply, mul0, c));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get(), f8_type, F16));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(computation->root_instruction()->operand(0), mul1);
  EXPECT_EQ(mul0->shape().element_type(), F16);
  EXPECT_EQ(mul1->shape().element_type(), F16);
  EXPECT_EQ(mul1->operand(0)->opcode(), HloOpcode::kConvert);
}

class FloatNormalizationNoComputeSupportTest : public FloatNormalizationTest {
 protected:
  bool Normalize(HloModule* module, PrimitiveType low_precision_type = BF16,
                 PrimitiveType high_precision_type = F32) {
    TestFloatNoComputeSupport float_support(low_precision_type,
                                            high_precision_type);
    FloatNormalization normalization(&float_support);

    absl::StatusOr<bool> result = normalization.Run(module);
    EXPECT_IS_OK(result.status());

    HloVerifier verifier(/*layout_sensitive=*/false,
                         /*allow_mixed_precision=*/true);
    EXPECT_IS_OK(verifier.Run(module).status());

    return result.value();
  }
};

TEST_F(FloatNormalizationNoComputeSupportTest,
       NoNormalizationForToApplyMultiOutputAllReduce) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder sum_builder("sum");
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(BF16, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(BF16, {}), "y"));
  sum_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(BF16, {}), HloOpcode::kAdd, x, y));
  HloComputation* reduction =
      module->AddEmbeddedComputation(sum_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  Shape bf16_shape_a = ShapeUtil::MakeShape(BF16, {2, 4});
  Shape bf16_shape_b = ShapeUtil::MakeShape(BF16, {16, 16});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_shape_a, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape_b, "b"));

  HloInstruction* crs = builder.AddInstruction(HloInstruction::CreateAllReduce(
      ShapeUtil::MakeTupleShape({bf16_shape_a, bf16_shape_b}), {a, b},
      reduction,
      /*device_list=*/CollectiveDeviceList(),
      /*constrain_layout=*/false,
      /*channel_id=*/std::nullopt,
      /*use_global_device_ids=*/false));
  builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(bf16_shape_b, crs, 1));

  auto computation = module->AddEntryComputation(builder.Build());
  // Since we skip processing to_apply region, nothing should change in the
  // original HLO.
  EXPECT_FALSE(Normalize(module.get()));
  EXPECT_EQ(computation->root_instruction()->shape().element_type(), BF16);
  EXPECT_EQ(crs->operand(1)->shape().element_type(), BF16);
  EXPECT_EQ(crs->to_apply()->root_instruction()->opcode(), HloOpcode::kAdd);
  EXPECT_EQ(ShapeUtil::GetSubshape(crs->shape(), {1}).element_type(), BF16);
}

TEST_F(FloatNormalizationNoComputeSupportTest,
       NormalizationClonesSharedApplyAllReduceAndReduce) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder sum_builder("sum");
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(BF16, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(BF16, {}), "y"));
  sum_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(BF16, {}), HloOpcode::kAdd, x, y));
  HloComputation* reduction =
      module->AddEmbeddedComputation(sum_builder.Build());

  auto builder = HloComputation::Builder(TestName());

  Shape bf16_shape_a = ShapeUtil::MakeShape(BF16, {2, 4});
  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_shape_a, "a"));

  Shape bf16_shape_b = ShapeUtil::MakeShape(BF16, {2, 4, 2});
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape_b, "b"));

  Shape bf16_scalar_shape = ShapeUtil::MakeShape(BF16, {});
  HloInstruction* init = builder.AddInstruction(
      HloInstruction::CreateParameter(2, bf16_scalar_shape, "init"));

  HloInstruction* all_reduce = builder.AddInstruction(
      HloInstruction::CreateAllReduce(bf16_shape_a, {a}, reduction,
                                      /*device_list=*/CollectiveDeviceList(),
                                      /*constrain_layout=*/false,
                                      /*channel_id=*/std::nullopt,
                                      /*use_global_device_ids=*/false));

  HloInstruction* reduce = builder.AddInstruction(
      HloInstruction::CreateReduce(bf16_shape_a, b, init, {2}, reduction));
  builder.AddInstruction(HloInstruction::CreateBinary(
      bf16_shape_a, HloOpcode::kAdd, all_reduce, reduce));

  auto computation = module->AddEntryComputation(builder.Build());
  // Verify that the shared computation was cloned, the all-reduce instruction
  // got the unchanged bf16 add, while the reduction was promoted to f32
  // together with its called computation.
  EXPECT_TRUE(Normalize(module.get()));
  EXPECT_EQ(computation->root_instruction()->shape().element_type(), BF16);
  EXPECT_EQ(all_reduce->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(all_reduce->to_apply()->root_instruction()->opcode(),
            HloOpcode::kAdd);
  EXPECT_EQ(all_reduce->to_apply()->root_instruction()->shape().element_type(),
            BF16);
  EXPECT_EQ(reduce->called_computations().size(), 1);
  EXPECT_EQ(reduce->called_computations()[0]
                ->root_instruction()
                ->shape()
                .element_type(),
            F32);
  EXPECT_EQ(reduce->called_computations()[0]->root_instruction()->opcode(),
            HloOpcode::kConvert);
  EXPECT_EQ(reduce->shape().element_type(), F32);
}

TEST_F(FloatNormalizationNoComputeSupportTest,
       NoNormalizationForToApplyAllReduce) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder sum_builder("sum");
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(BF16, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(BF16, {}), "y"));
  sum_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(BF16, {}), HloOpcode::kAdd, x, y));
  HloComputation* reduction =
      module->AddEmbeddedComputation(sum_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  Shape bf16_shape_a = ShapeUtil::MakeShape(BF16, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_shape_a, "a"));

  HloInstruction* crs = builder.AddInstruction(
      HloInstruction::CreateAllReduce(bf16_shape_a, {a}, reduction,
                                      /*device_list=*/CollectiveDeviceList(),
                                      /*constrain_layout=*/false,
                                      /*channel_id=*/std::nullopt,
                                      /*use_global_device_ids=*/false));

  auto computation = module->AddEntryComputation(builder.Build());
  // Since we skip processing to_apply region, nothing should change in the
  // original HLO.
  EXPECT_FALSE(Normalize(module.get()));
  EXPECT_EQ(computation->root_instruction()->shape().element_type(), BF16);
  EXPECT_EQ(crs->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(crs->to_apply()->root_instruction()->opcode(), HloOpcode::kAdd);
}

TEST_F(FloatNormalizationNoComputeSupportTest,
       NoNormalizationForToApplyReduceScatter) {
  auto module = CreateNewVerifiedModule();
  HloComputation::Builder sum_builder("sum");
  auto x = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/0, ShapeUtil::MakeShape(BF16, {}), "x"));
  auto y = sum_builder.AddInstruction(HloInstruction::CreateParameter(
      /*parameter_number=*/1, ShapeUtil::MakeShape(BF16, {}), "y"));
  sum_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(BF16, {}), HloOpcode::kAdd, x, y));
  HloComputation* reduction =
      module->AddEmbeddedComputation(sum_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  Shape bf16_shape_a = ShapeUtil::MakeShape(BF16, {2, 4});
  Shape bf16_shape_scattered = ShapeUtil::MakeShape(BF16, {1, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_shape_a, "a"));

  HloInstruction* crs =
      builder.AddInstruction(HloInstruction::CreateReduceScatter(
          bf16_shape_scattered, {a}, reduction,
          /*device_list=*/CollectiveDeviceList(),
          /*constrain_layout=*/false,
          /*channel_id=*/std::nullopt,
          /*use_global_device_ids=*/false, /*scatter_dimension*/ 0));

  auto computation = module->AddEntryComputation(builder.Build());
  // Since we skip processing to_apply region, nothing should change in the
  // original HLO.
  EXPECT_FALSE(Normalize(module.get()));
  EXPECT_EQ(computation->root_instruction()->shape().element_type(), BF16);
  EXPECT_EQ(crs->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(crs->to_apply()->root_instruction()->opcode(), HloOpcode::kAdd);
}

TEST_F(FloatNormalizationTest, ConvertBeforeTuple) {
  auto builder = HloComputation::Builder(TestName());
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {2, 4});
  Shape f32_shape = ShapeUtil::MakeShape(F32, {2, 4});

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateParameter(0, bf16_shape, "a"));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateParameter(1, bf16_shape, "b"));

  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(bf16_shape, HloOpcode::kMultiply, a, b));

  HloInstruction* convert =
      builder.AddInstruction(HloInstruction::CreateConvert(f32_shape, add));

  builder.AddInstruction(HloInstruction::CreateVariadic(
      ShapeUtil::MakeTupleShape({f32_shape, bf16_shape}), HloOpcode::kTuple,
      {convert, add}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(Normalize(module.get(), BF16));

  EXPECT_EQ(computation->root_instruction()->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(computation->root_instruction()->operand(0)->shape().element_type(),
            F32);
  EXPECT_EQ(
      computation->root_instruction()->shape().tuple_shapes(0).element_type(),
      F32);
}

TEST_F(FloatNormalizationTest, KeepEntryInputOutputAlias) {
  const std::string hlo_text = R"(
    HloModule m,
      input_output_alias={ {}: (1, {}, must-alias) },
      entry_computation_layout={(bf16[1,64], bf16[1,64])->bf16[1,64]}

    ENTRY e {
      arg_0 = bf16[1,64] parameter(0)
      output_param = bf16[1,64] parameter(1)
      constant = bf16[] constant(2)
      broadcast = bf16[1,64] broadcast(constant), dimensions={}
      ROOT multiply = bf16[1,64] multiply(arg_0, broadcast)
    })";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));

  EXPECT_TRUE(Normalize(module.get(), BF16));

  HloInputOutputAliasConfig& alias_config = module->input_output_alias_config();
  ASSERT_FALSE(alias_config.ParameterHasAlias(0, {}));
  ASSERT_TRUE(alias_config.ParameterHasAlias(1, {}));
  ASSERT_TRUE(alias_config.OutputHasAlias({}));
  EXPECT_EQ(alias_config.GetAliasedParameter({})->parameter_number, 1);
  EXPECT_EQ(alias_config.GetAliasedParameter({})->kind,
            HloInputOutputAliasConfig::AliasKind::kMustAlias);
}

TEST_F(FloatNormalizationTest, AllowExcessPrecisionTrue) {
  const std::string hlo_text = R"(
    HloModule dot_with_convert

    ENTRY main {
      Arg_0 = bf16[2,2]{1,0} parameter(0)
      dot = bf16[2,2]{1,0} dot(Arg_0, Arg_0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      convert = f32[2,2]{1,0} convert(dot)
      ROOT tuple = (f32[2,2]{1,0}, bf16[2,2]{1,0}) tuple(convert, dot)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_allow_excess_precision(true);

  EXPECT_TRUE(Normalize(module.get(), BF16));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Tuple(m::Dot(), m::Convert())));
}

TEST_F(FloatNormalizationTest, AllowExcessPrecisionFalse) {
  const std::string hlo_text = R"(
    HloModule dot_with_convert

    ENTRY main {
      Arg_0 = bf16[2,2]{1,0} parameter(0)
      dot = bf16[2,2]{1,0} dot(Arg_0, Arg_0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
      convert = f32[2,2]{1,0} convert(dot)
      ROOT tuple = (f32[2,2]{1,0}, bf16[2,2]{1,0}) tuple(convert, dot)
    })";
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(hlo_text));
  module->mutable_config()
      .mutable_debug_options()
      .set_xla_allow_excess_precision(false);

  EXPECT_TRUE(Normalize(module.get(), BF16));
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      GmockMatch(m::Tuple(m::Convert(m::Convert(m::Dot())), m::Convert())));
}

}  // namespace
}  // namespace xla
