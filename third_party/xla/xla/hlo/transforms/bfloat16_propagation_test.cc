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

#include "xla/hlo/transforms/bfloat16_propagation.h"

#include <cstdint>
#include <memory>
#include <string>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/comparison_util.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/replica_group.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/literal_util.h"
#include "xla/service/float_support.h"
#include "xla/service/hlo_verifier.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/statusor.h"

namespace xla {

// A class specifying the BF16 support used to test the propagation pass. It
// specifies that BF16 and mixed precision are supported in all HloInstructions,
// and that kDot reduces its operands precision to BF16.
class TestBFloat16Support : public FloatSupport {
 public:
  TestBFloat16Support() : FloatSupport(BF16) {}
  ~TestBFloat16Support() override {}

  bool SupportsLowPrecisionOperand(const HloInstruction& hlo,
                                   int64_t operand_index) const override {
    return true;
  }

  bool SupportsLowPrecisionOutput(const HloInstruction& hlo) const override {
    return true;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    return true;
  }

  bool EffectiveOperandPrecisionIsLowPrecision(
      const HloInstruction& hlo, int64_t operand_index) const override {
    return hlo.opcode() == HloOpcode::kDot;
  }
};

class BFloat16PropagationTest : public HloHardwareIndependentTestBase {
 protected:
  BFloat16PropagationTest()
      : HloHardwareIndependentTestBase(
            /*verifier_layout_sensitive=*/false,
            /*allow_mixed_precision_in_hlo_verifier=*/true) {}

  // Runs the propagation pass on the given module, and returns whether the
  // module is changed after this pass.
  bool PropagatePrecision(HloModule* module) {
    TestBFloat16Support bfloat16_support;
    BFloat16Propagation propagation(&bfloat16_support, &alias_info_);
    absl::StatusOr<bool> result = propagation.Run(module);
    EXPECT_IS_OK(result.status());
    return result.value();
  }

  // Returns whether the given HloInstruction's output element type is BF16 or
  // the only use of it is converting to BF16.
  bool OutputsBF16(const HloInstruction* inst) {
    if (inst->shape().element_type() == BF16) {
      return true;
    }
    return inst->user_count() == 1 &&
           inst->users()[0]->opcode() == HloOpcode::kConvert &&
           inst->users()[0]->shape().element_type() == BF16;
  }

  std::unique_ptr<HloInstruction> CreateDot(const Shape& shape,
                                            HloInstruction* lhs,
                                            HloInstruction* rhs) {
    DotDimensionNumbers dot_dnums;
    dot_dnums.add_lhs_contracting_dimensions(1);
    dot_dnums.add_rhs_contracting_dimensions(0);
    return HloInstruction::CreateDot(shape, lhs, rhs, dot_dnums,
                                     DefaultPrecisionConfig(2));
  }
  AliasInfo alias_info_;
};

// Tests that BF16 can propagate through select over non-tuple buffers, but not
// through add where reducing operand precision can affect the result.
TEST_F(BFloat16PropagationTest, PropagateThroughSelectButNotAdd) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});

  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* c =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape, "c"));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a, b));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add0, b));
  HloInstruction* pred = builder.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {2, 4}), a, b, ComparisonDirection::kEq));
  HloInstruction* sel = builder.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kSelect, pred, c, add1));
  HloInstruction* xpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {4, 2}), sel, {1, 0}));
  HloInstruction* dot = builder.AddInstruction(
      CreateDot(ShapeUtil::MakeShape(F32, {4, 4}), xpose, a));
  HloInstruction* root = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {4, 4}), HloOpcode::kAdd, dot, dot));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), root);
  EXPECT_TRUE(OutputsBF16(xpose));
  EXPECT_TRUE(OutputsBF16(sel));
  EXPECT_TRUE(OutputsBF16(add1));
  EXPECT_FALSE(OutputsBF16(add0));
  EXPECT_FALSE(OutputsBF16(a));
  EXPECT_FALSE(OutputsBF16(b));
  EXPECT_FALSE(OutputsBF16(c));
}

TEST_F(BFloat16PropagationTest, PropagateThroughMaxPoolReduceWindow) {
  auto module = CreateNewVerifiedModule();

  auto sub_builder = HloComputation::Builder("max");
  HloInstruction* p0 = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(0, ShapeUtil::MakeShape(F32, {}), "a"));
  HloInstruction* p1 = sub_builder.AddInstruction(
      HloInstruction::CreateParameter(1, ShapeUtil::MakeShape(F32, {}), "b"));
  sub_builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {}), HloOpcode::kMaximum, p0, p1));
  auto max_computation = module->AddEmbeddedComputation(sub_builder.Build());

  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});

  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* c =
      builder.AddInstruction(HloInstruction::CreateParameter(2, shape, "c"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a, b));
  Window window;
  WindowDimension dim;
  dim.set_size(2);
  dim.set_stride(1);
  dim.set_padding_high(1);
  dim.set_window_dilation(1);
  dim.set_base_dilation(1);
  *window.add_dimensions() = dim;
  *window.add_dimensions() = dim;
  HloInstruction* rw =
      builder.AddInstruction(HloInstruction::CreateReduceWindow(
          shape, add,
          builder.AddInstruction(
              HloInstruction::CreateConstant(LiteralUtil::Zero(F32))),
          window, max_computation));
  HloInstruction* xpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {4, 2}), c, {1, 0}));
  HloInstruction* dot = builder.AddInstruction(
      CreateDot(ShapeUtil::MakeShape(F32, {4, 4}), xpose, rw));
  HloInstruction* root = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {4, 4}), HloOpcode::kAdd, dot, dot));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), root);
  EXPECT_TRUE(OutputsBF16(add));
  EXPECT_TRUE(OutputsBF16(xpose));
  EXPECT_TRUE(OutputsBF16(rw));
}

// Tests that side-effecting all-reduce should not be changed.
TEST_F(BFloat16PropagationTest, DoNotChangeAllReduce) {
  auto module = CreateNewVerifiedModule();

  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});
  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  auto rb = HloComputation::Builder(TestName());
  rb.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kAdd,
      rb.AddInstruction(HloInstruction::CreateParameter(0, shape, "p0")),
      rb.AddInstruction(HloInstruction::CreateParameter(1, shape, "p1"))));
  auto reduction = module->AddEmbeddedComputation(rb.Build());
  HloInstruction* all_reduce =
      builder.AddInstruction(HloInstruction::CreateAllReduce(
          ShapeUtil::MakeTupleShape({shape, shape}), {a, b}, reduction,
          std::make_shared<CollectiveDeviceList>(),
          /*constrain_layout=*/false,
          /*channel_id=*/1, /*use_global_device_ids=*/false));
  HloInstruction* gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, all_reduce, 0));
  HloInstruction* gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, all_reduce, 1));
  HloInstruction* dot = builder.AddInstruction(CreateDot(shape, gte0, gte1));
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, dot, dot));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(PropagatePrecision(module.get()));
  EXPECT_EQ(computation->root_instruction(), root);
}

// Tests that if a constant is converted to BF16 then its literal must also be
// converted.
TEST_F(BFloat16PropagationTest, ConvertConstantLiteral) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});
  Array2D<float> array_a(4, 4);
  array_a.FillUnique(1.0f);
  Array2D<float> array_b(4, 4);
  array_b.FillUnique(10.0f);

  HloInstruction* a = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateFromArray(array_a)));
  HloInstruction* b = builder.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateFromArray(array_b)));
  HloInstruction* dot = builder.AddInstruction(CreateDot(shape, a, b));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_TRUE(OutputsBF16(dot->operand(0)));
  EXPECT_TRUE(OutputsBF16(dot->operand(1)));
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(dot->operand(1)->opcode(), HloOpcode::kConstant);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::ConvertF32ToBF16(LiteralUtil::CreateFromArray(array_a)),
      dot->operand(0)->literal()));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      LiteralUtil::ConvertF32ToBF16(LiteralUtil::CreateFromArray(array_b)),
      dot->operand(1)->literal()));
}

// Tests that BF16 can be propagated through nested tuples.
TEST_F(BFloat16PropagationTest, PropagateThroughTuples) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});

  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a, b));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a, a));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, b, b));
  HloInstruction* xpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {4, 2}), add1, {1, 0}));

  HloInstruction* tuple0 =
      builder.AddInstruction(HloInstruction::CreateTuple({add0, add1, add2}));
  HloInstruction* tuple1 =
      builder.AddInstruction(HloInstruction::CreateTuple({tuple0, xpose}));

  HloInstruction* lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(xpose->shape(), tuple1, 1));
  HloInstruction* rhs =
      builder.AddInstruction(HloInstruction::CreateGetTupleElement(
          add0->shape(),
          builder.AddInstruction(HloInstruction::CreateGetTupleElement(
              tuple0->shape(), tuple1, 0)),
          0));
  HloInstruction* dot = builder.AddInstruction(
      CreateDot(ShapeUtil::MakeShape(F32, {4, 4}), lhs, rhs));

  HloInstruction* output_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({dot, add2}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), output_tuple);
  EXPECT_TRUE(OutputsBF16(xpose));
  EXPECT_TRUE(OutputsBF16(add0));
  EXPECT_TRUE(OutputsBF16(add1));
  EXPECT_FALSE(OutputsBF16(add2));
}

// Tests that even if an instruction does not define a buffer in its output, its
// shape must match the defining instruction.
TEST_F(BFloat16PropagationTest, SameValueReferencedTwice) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});

  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a, b));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a, a));

  HloInstruction* lhs = builder.AddInstruction(HloInstruction::CreateTranspose(
      ShapeUtil::MakeShape(F32, {4, 2}), add1, {1, 0}));

  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
  HloInstruction* rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(add1->shape(), tuple, 1));

  // lhs is the transpose of add1, and rhs is a get-tuple-element aliasing add1.
  HloInstruction* dot = builder.AddInstruction(
      CreateDot(ShapeUtil::MakeShape(F32, {4, 4}), lhs, rhs));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_TRUE(OutputsBF16(add1));
  EXPECT_TRUE(OutputsBF16(lhs));

  // add0 and rhs have been eliminated by simplification and DCE.
}

// Tests that a non-fusion computation's root should not be changed.
TEST_F(BFloat16PropagationTest, DoNotChangeComputationRoot) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a, b));

  HloInstruction* dot = builder.AddInstruction(CreateDot(shape, add, add));

  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({add, dot}));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), tuple);
  EXPECT_FALSE(OutputsBF16(add));
}

// Tests that BF16 is propagated properly through fused computations.
TEST_F(BFloat16PropagationTest, PropagateThroughFusion) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param, param));

  auto builder_f0 = HloComputation::Builder("fusion0");
  HloInstruction* a_f0 =
      builder_f0.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b_f0 =
      builder_f0.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* tuple_f0 =
      builder_f0.AddInstruction(HloInstruction::CreateTuple({a_f0, b_f0}));
  auto comp_f0 = module->AddEmbeddedComputation(builder_f0.Build());
  auto fusion0 = builder.AddInstruction(HloInstruction::CreateFusion(
      tuple_f0->shape(), HloInstruction::FusionKind::kCustom, {add, add},
      comp_f0));

  auto builder_f1 = HloComputation::Builder("fusion1");
  HloInstruction* p_f1 = builder_f1.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_f0->shape(), "param"));
  HloInstruction* a_f1 = builder_f1.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p_f1, 0));
  HloInstruction* b_f1 = builder_f1.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, p_f1, 1));
  HloInstruction* dot = builder_f1.AddInstruction(CreateDot(shape, a_f1, b_f1));
  auto comp_f1 = module->AddEmbeddedComputation(builder_f1.Build());
  auto fusion1 = builder.AddInstruction(HloInstruction::CreateFusion(
      dot->shape(), HloInstruction::FusionKind::kCustom, {fusion0}, comp_f1));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), fusion1);
  EXPECT_TRUE(OutputsBF16(add));
  EXPECT_TRUE(OutputsBF16(a_f0));
  EXPECT_TRUE(OutputsBF16(b_f0));
  EXPECT_TRUE(OutputsBF16(a_f1));
  EXPECT_TRUE(OutputsBF16(b_f1));
}

// Tests that BF16 is propagated properly through called fused computations.
TEST_F(BFloat16PropagationTest, PropagateThroughCalledFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule main

ENTRY main {
  arg.0 = f32[4,4] parameter(0)
  add.0 = f32[4,4] add(arg.0, arg.0)
  call.0 = call(add.0, add.0), to_apply={
    arg.0 = f32[4,4] parameter(0)
    arg.1 = f32[4,4] parameter(1)
    ROOT fusion.0 = (f32[4,4], f32[4,4]) fusion(arg.0, arg.1), kind=kCustom, calls={
      arg.0 = f32[4,4] parameter(0)
      arg.1 = f32[4,4] parameter(1)
      ROOT tuple.0 = tuple(arg.0, arg.1)
    }
  }
  ROOT fusion.1 = f32[4,4] fusion(call.0), kind=kCustom, calls={
    arg.0 = (f32[4,4], f32[4,4]) parameter(0)
    gte.0 = get-tuple-element(arg.0), index=0
    gte.1 = get-tuple-element(arg.0), index=1
    ROOT dot.0 = dot(gte.0, gte.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  EXPECT_TRUE(PropagatePrecision(module.get()));

  HloInstruction* add0 = FindInstruction(module.get(), "add.0");
  ASSERT_NE(add0, nullptr);
  EXPECT_TRUE(OutputsBF16(add0));
  HloInstruction* call = FindInstruction(module.get(), "call.0");
  ASSERT_NE(call, nullptr);
  HloInstruction* arg0 = call->to_apply()->parameter_instruction(0);
  EXPECT_TRUE(OutputsBF16(arg0));
  HloInstruction* arg1 = call->to_apply()->parameter_instruction(1);
  EXPECT_TRUE(OutputsBF16(arg1));
  HloInstruction* gte0 = FindInstruction(module.get(), "gte.0");
  ASSERT_NE(gte0, nullptr);
  EXPECT_TRUE(OutputsBF16(gte0));
  HloInstruction* gte1 = FindInstruction(module.get(), "gte.1");
  ASSERT_NE(gte1, nullptr);
  EXPECT_TRUE(OutputsBF16(gte1));
}

// Tests that BF16 is propagated properly through async fused computations.
TEST_F(BFloat16PropagationTest, PropagateThroughAsyncFusion) {
  constexpr absl::string_view kHlo = R"(
HloModule main

ENTRY main {
  arg.0 = f32[4,4] parameter(0)
  add.0 = f32[4,4] add(arg.0, arg.0)
  fusion-start.0 = ((f32[4,4], f32[4,4]), (f32[4,4], f32[4,4]), s32[]) fusion-start(add.0, add.0), kind=kCustom, calls={
    arg.0 = f32[4,4] parameter(0)
    arg.1 = f32[4,4] parameter(1)
    ROOT tuple.0 = tuple(arg.0, arg.1)
  }, async_execution_thread="main"
  fusion-done.0 = (f32[4,4], f32[4,4]) fusion-done(fusion-start.0)
  ROOT fusion.1 = f32[4,4] fusion(fusion-done.0), kind=kCustom, calls={
    arg.0 = (f32[4,4], f32[4,4]) parameter(0)
    gte.0 = get-tuple-element(arg.0), index=0
    gte.1 = get-tuple-element(arg.0), index=1
    ROOT dot.0 = dot(gte.0, gte.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  }
}
  )";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  EXPECT_TRUE(PropagatePrecision(module.get()));

  HloInstruction* add0 = FindInstruction(module.get(), "add.0");
  ASSERT_NE(add0, nullptr);
  EXPECT_TRUE(OutputsBF16(add0));
  HloInstruction* fusion0 = FindInstruction(module.get(), "fusion-start.0");
  HloInstruction* async_arg0 =
      fusion0->async_wrapped_computation()->parameter_instruction(0);
  EXPECT_TRUE(OutputsBF16(async_arg0));
  HloInstruction* async_arg1 =
      fusion0->async_wrapped_computation()->parameter_instruction(1);
  EXPECT_TRUE(OutputsBF16(async_arg1));
  HloInstruction* arg0 = fusion0->async_wrapped_instruction()
                             ->called_computations()[0]
                             ->parameter_instruction(0);
  EXPECT_TRUE(OutputsBF16(arg0));
  HloInstruction* arg1 = fusion0->async_wrapped_instruction()
                             ->called_computations()[0]
                             ->parameter_instruction(1);
  EXPECT_TRUE(OutputsBF16(arg1));
  HloInstruction* gte0 = FindInstruction(module.get(), "gte.0");
  ASSERT_NE(gte0, nullptr);
  EXPECT_TRUE(OutputsBF16(gte0));
  HloInstruction* gte1 = FindInstruction(module.get(), "gte.1");
  ASSERT_NE(gte1, nullptr);
  EXPECT_TRUE(OutputsBF16(gte1));
}

// Tests that a fusion with a bitcast-convert as its root is changed via adding
// extra convert, instead of changing the type in-place.
TEST_F(BFloat16PropagationTest, FusionWithBitcastConvertRoot) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape u32_shape = ShapeUtil::MakeShape(U32, {4, 4});
  Shape f32_shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, u32_shape, "param"));

  auto builder_f = HloComputation::Builder("fusion");
  HloInstruction* a_f = builder_f.AddInstruction(
      HloInstruction::CreateParameter(0, u32_shape, "a"));
  HloInstruction* bc_f = builder_f.AddInstruction(
      HloInstruction::CreateBitcastConvert(f32_shape, a_f));
  auto comp_f = module->AddEmbeddedComputation(builder_f.Build());
  auto fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      f32_shape, HloInstruction::FusionKind::kLoop, {param}, comp_f));
  auto dot = builder.AddInstruction(CreateDot(f32_shape, fusion, fusion));

  auto computation = module->AddEntryComputation(builder.Build());
  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_EQ(bc_f->shape(), f32_shape);
  EXPECT_TRUE(OutputsBF16(bc_f));
}

// Tests that changes to BF16 that cannot be propagated outside a fusion are
// discarded.
TEST_F(BFloat16PropagationTest, DiscardFusionInternalBF16Changes) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param, param));

  auto builder_f = HloComputation::Builder("fusion");
  HloInstruction* a_f =
      builder_f.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b_f =
      builder_f.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* add_f = builder_f.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a_f, b_f));
  HloInstruction* dot_f =
      builder_f.AddInstruction(CreateDot(shape, add_f, add_f));
  auto comp_f = module->AddEmbeddedComputation(builder_f.Build());
  auto fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      dot_f->shape(), HloInstruction::FusionKind::kCustom, {add, add}, comp_f));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(PropagatePrecision(module.get()));
  EXPECT_EQ(computation->root_instruction(), fusion);
}

// Tests that if 1) the root instruction of a fusion is a tuple, 2) the fusion
// outputs are only used by a dot, and 3) one element of the tuple is used by
// an add in the fusion computation, then the propagation pass should create a
// convert in the fusion computation to keep the add's operand in F32 but change
// the fusion output to BF16. E.g., the following fusion computation
//   (F32, F32) fusion_computation(F32 a, F32 b)
//     = tuple(F32 a, F32 add(F32 a, F32 b))
// will be changed to
//   (BF16, BF16) fusion_computation(F32 a, F32 b)
//     = tuple(BF16 convert(a), BF16 add(F32 a, F32 b))
TEST_F(BFloat16PropagationTest, ConvertTupleFusionElementIfUsedByAdd) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param, param));

  auto builder_f = HloComputation::Builder("fusion0");
  HloInstruction* a_f =
      builder_f.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b_f =
      builder_f.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* add_f = builder_f.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a_f, b_f));
  HloInstruction* tuple_f =
      builder_f.AddInstruction(HloInstruction::CreateTuple({a_f, add_f}));
  auto comp_f = module->AddEmbeddedComputation(builder_f.Build());
  auto fusion = builder.AddInstruction(HloInstruction::CreateFusion(
      tuple_f->shape(), HloInstruction::FusionKind::kCustom, {add, add},
      comp_f));

  HloInstruction* gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion, 0));
  HloInstruction* gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, fusion, 1));
  HloInstruction* dot = builder.AddInstruction(CreateDot(shape, gte0, gte1));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_TRUE(OutputsBF16(gte0));
  EXPECT_TRUE(OutputsBF16(gte1));
  EXPECT_FALSE(OutputsBF16(a_f));
  EXPECT_FALSE(OutputsBF16(b_f));
  EXPECT_TRUE(OutputsBF16(add_f));
  auto new_fusion_root = comp_f->root_instruction();
  EXPECT_EQ(new_fusion_root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(new_fusion_root->operand(1), add_f);
  EXPECT_EQ(new_fusion_root->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_TRUE(OutputsBF16(new_fusion_root->operand(0)));
}

// Tests that BF16 is propagated properly through a while computation with
// non-tuple input/output.
TEST_F(BFloat16PropagationTest, PropagateThroughSimpleWhile) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "param1"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));

  auto builder_cond = HloComputation::Builder("cond");
  auto cond_param = builder_cond.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "cond_param"));
  auto cond_dot =
      builder_cond.AddInstruction(CreateDot(shape, cond_param, cond_param));
  auto cond_root = builder_cond.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}),
      builder_cond.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond.AddInstruction(
              HloInstruction::CreateSlice(ShapeUtil::MakeShape(F32, {1, 1}),
                                          cond_dot, {0, 0}, {1, 1}, {1, 1})))),
      builder_cond.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond.AddInstruction(
              HloInstruction::CreateSlice(ShapeUtil::MakeShape(F32, {1, 1}),
                                          cond_dot, {1, 1}, {2, 2}, {1, 1})))),
      ComparisonDirection::kGt));
  auto cond = module->AddEmbeddedComputation(builder_cond.Build());

  auto builder_body = HloComputation::Builder("body");
  auto body_param = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "body_param"));
  auto body_dot =
      builder_body.AddInstruction(CreateDot(shape, body_param, body_param));
  auto body = module->AddEmbeddedComputation(builder_body.Build());

  auto while_hlo = builder.AddInstruction(
      HloInstruction::CreateWhile(shape, cond, body, add));

  auto dot = builder.AddInstruction(CreateDot(shape, while_hlo, while_hlo));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_TRUE(
      ShapeUtil::Equal(cond_root->shape(), ShapeUtil::MakeShape(PRED, {})));
  EXPECT_TRUE(OutputsBF16(add));
  EXPECT_TRUE(OutputsBF16(body_dot));
  EXPECT_TRUE(OutputsBF16(body_param));
  EXPECT_TRUE(OutputsBF16(cond_param));
  EXPECT_FALSE(OutputsBF16(dot));
}

// Tests that if the while condition prevents using BF16, no changes should be
// made to the while body and thus the fusion node inside it.
TEST_F(BFloat16PropagationTest,
       ConditionPreventsPropagationForFusionInsideWhile) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "param1"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));

  auto builder_cond = HloComputation::Builder("cond");
  auto cond_param = builder_cond.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "cond_param"));
  builder_cond.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}),
      builder_cond.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond.AddInstruction(HloInstruction::CreateSlice(
              ShapeUtil::MakeShape(F32, {1, 1}), cond_param, {0, 0}, {1, 1},
              {1, 1})))),
      builder_cond.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond.AddInstruction(HloInstruction::CreateSlice(
              ShapeUtil::MakeShape(F32, {1, 1}), cond_param, {1, 1}, {2, 2},
              {1, 1})))),
      ComparisonDirection::kGt));
  auto cond = module->AddEmbeddedComputation(builder_cond.Build());

  auto builder_body = HloComputation::Builder("body");
  auto body_param = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "body_param"));
  auto body_transpose = builder_body.AddInstruction(
      HloInstruction::CreateTranspose(shape, body_param, {0, 1}));

  auto builder_f = HloComputation::Builder("fusion");
  HloInstruction* a_f =
      builder_f.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  builder_f.AddInstruction(HloInstruction::CreateTranspose(shape, a_f, {0, 1}));
  auto comp_f = module->AddEmbeddedComputation(builder_f.Build());
  auto body_fusion = builder_body.AddInstruction(HloInstruction::CreateFusion(
      shape, HloInstruction::FusionKind::kCustom, {body_transpose}, comp_f));
  auto body = module->AddEmbeddedComputation(builder_body.Build());

  auto while_hlo = builder.AddInstruction(
      HloInstruction::CreateWhile(shape, cond, body, add));

  auto dot = builder.AddInstruction(CreateDot(shape, while_hlo, while_hlo));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(PropagatePrecision(module.get()));
  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_FALSE(OutputsBF16(add));
  EXPECT_FALSE(OutputsBF16(body_fusion));
  EXPECT_FALSE(OutputsBF16(body_param));
  EXPECT_FALSE(OutputsBF16(body_transpose));
  EXPECT_FALSE(OutputsBF16(a_f));
}

// Tests that BF16 is propagated properly through while computations with
// tuple-shaped input/output.
TEST_F(BFloat16PropagationTest, PropagateThroughTupleWhile) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "param1"));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));

  auto builder_cond = HloComputation::Builder("cond");
  auto cond_param = builder_cond.AddInstruction(
      HloInstruction::CreateParameter(0, tuple->shape(), "cond_param"));
  auto cond_lhs = builder_cond.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, cond_param, 0));
  auto cond_rhs = builder_cond.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, cond_param, 1));
  // This add should prevent RHS from using BF16
  auto cond_add_rhs = builder_cond.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, cond_rhs, cond_rhs));
  auto cond_dot =
      builder_cond.AddInstruction(CreateDot(shape, cond_lhs, cond_add_rhs));
  builder_cond.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}),
      builder_cond.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond.AddInstruction(
              HloInstruction::CreateSlice(ShapeUtil::MakeShape(F32, {1, 1}),
                                          cond_dot, {0, 0}, {1, 1}, {1, 1})))),
      builder_cond.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond.AddInstruction(
              HloInstruction::CreateSlice(ShapeUtil::MakeShape(F32, {1, 1}),
                                          cond_dot, {1, 1}, {2, 2}, {1, 1})))),
      ComparisonDirection::kGt));
  auto cond = module->AddEmbeddedComputation(builder_cond.Build());

  auto builder_body = HloComputation::Builder("body");
  auto body_param = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, tuple->shape(), "body_param"));
  auto body_lhs = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  auto body_rhs = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 1));
  auto body_dot1 =
      builder_body.AddInstruction(CreateDot(shape, body_lhs, body_rhs));
  auto body_dot2 =
      builder_body.AddInstruction(CreateDot(shape, body_rhs, body_lhs));
  auto body_transpose = builder_body.AddInstruction(
      HloInstruction::CreateTranspose(shape, body_dot2, {0, 1}));
  builder_body.AddInstruction(
      HloInstruction::CreateTuple({body_dot1, body_transpose}));
  auto body = module->AddEmbeddedComputation(builder_body.Build());

  auto while_hlo = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple->shape(), cond, body, tuple));

  auto lhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, while_hlo, 0));
  auto rhs = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, while_hlo, 1));
  auto dot = builder.AddInstruction(CreateDot(shape, lhs, rhs));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_TRUE(OutputsBF16(lhs));
  EXPECT_FALSE(OutputsBF16(rhs));
  EXPECT_TRUE(OutputsBF16(body_dot1));
  EXPECT_TRUE(OutputsBF16(body_lhs));
  EXPECT_FALSE(OutputsBF16(body_rhs));
  EXPECT_FALSE(OutputsBF16(body_dot2));
  EXPECT_FALSE(OutputsBF16(body_transpose));
  EXPECT_TRUE(OutputsBF16(cond_lhs));
  EXPECT_FALSE(OutputsBF16(cond_rhs));
  EXPECT_TRUE(OutputsBF16(add0));
  EXPECT_FALSE(OutputsBF16(add1));
}

// Tests that BF16 is not propagated through multiple whiles that invoke the
// same computation as long as one while prevents the propagation.
TEST_F(BFloat16PropagationTest, DoNotPropagateWhilesCallingSameComputation) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* param0 = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param0"));
  HloInstruction* param1 = builder.AddInstruction(
      HloInstruction::CreateParameter(1, shape, "param1"));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  HloInstruction* add2 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  HloInstruction* add3 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param0, param1));
  HloInstruction* tuple0 =
      builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
  HloInstruction* tuple1 =
      builder.AddInstruction(HloInstruction::CreateTuple({add2, add3}));

  // Condition computation for the first while.
  auto builder_cond0 = HloComputation::Builder("cond0");
  auto cond0_param = builder_cond0.AddInstruction(
      HloInstruction::CreateParameter(0, tuple0->shape(), "cond0_param"));
  auto cond0_lhs = builder_cond0.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, cond0_param, 0));
  auto cond0_rhs = builder_cond0.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, cond0_param, 1));
  // This add should prevent RHS from using BF16
  auto cond0_add_rhs =
      builder_cond0.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, cond0_rhs, cond0_rhs));
  auto cond0_dot =
      builder_cond0.AddInstruction(CreateDot(shape, cond0_lhs, cond0_add_rhs));
  builder_cond0.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}),
      builder_cond0.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond0.AddInstruction(
              HloInstruction::CreateSlice(ShapeUtil::MakeShape(F32, {1, 1}),
                                          cond0_dot, {0, 0}, {1, 1}, {1, 1})))),
      builder_cond0.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond0.AddInstruction(
              HloInstruction::CreateSlice(ShapeUtil::MakeShape(F32, {1, 1}),
                                          cond0_dot, {1, 1}, {2, 2}, {1, 1})))),
      ComparisonDirection::kGt));
  auto cond0 = module->AddEmbeddedComputation(builder_cond0.Build());

  // Condition computation for the second while.
  auto builder_cond1 = HloComputation::Builder("cond1");
  auto cond1_param = builder_cond1.AddInstruction(
      HloInstruction::CreateParameter(0, tuple1->shape(), "cond1_param"));
  auto cond1_lhs = builder_cond1.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, cond1_param, 0));
  auto cond1_rhs = builder_cond1.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, cond1_param, 1));
  // This add should prevent LHS from using BF16
  auto cond1_add_lhs =
      builder_cond1.AddInstruction(HloInstruction::CreateBinary(
          shape, HloOpcode::kAdd, cond1_lhs, cond1_lhs));
  auto cond1_dot =
      builder_cond1.AddInstruction(CreateDot(shape, cond1_add_lhs, cond1_rhs));
  builder_cond1.AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::MakeShape(PRED, {}),
      builder_cond1.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond1.AddInstruction(
              HloInstruction::CreateSlice(ShapeUtil::MakeShape(F32, {1, 1}),
                                          cond1_dot, {0, 0}, {1, 1}, {1, 1})))),
      builder_cond1.AddInstruction(HloInstruction::CreateReshape(
          ShapeUtil::MakeShape(F32, {}),
          builder_cond1.AddInstruction(
              HloInstruction::CreateSlice(ShapeUtil::MakeShape(F32, {1, 1}),
                                          cond1_dot, {1, 1}, {2, 2}, {1, 1})))),
      ComparisonDirection::kGt));
  auto cond1 = module->AddEmbeddedComputation(builder_cond1.Build());

  // Body computation shared by both whiles.
  auto builder_body = HloComputation::Builder("body");
  auto body_param = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, tuple0->shape(), "body_param"));
  auto body_lhs = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  auto body_rhs = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 1));
  auto body_dot =
      builder_body.AddInstruction(CreateDot(shape, body_lhs, body_rhs));
  builder_body.AddInstruction(
      HloInstruction::CreateTuple({body_dot, body_rhs}));
  auto body = module->AddEmbeddedComputation(builder_body.Build());

  auto while0 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple0->shape(), cond0, body, tuple0));
  auto while1 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple1->shape(), cond1, body, tuple1));

  auto lhs = builder.AddInstruction(
      CreateDot(shape,
                builder.AddInstruction(
                    HloInstruction::CreateGetTupleElement(shape, while0, 0)),
                builder.AddInstruction(
                    HloInstruction::CreateGetTupleElement(shape, while0, 1))));
  auto rhs = builder.AddInstruction(
      CreateDot(shape,
                builder.AddInstruction(
                    HloInstruction::CreateGetTupleElement(shape, while1, 0)),
                builder.AddInstruction(
                    HloInstruction::CreateGetTupleElement(shape, while1, 1))));
  auto dot = builder.AddInstruction(CreateDot(shape, lhs, rhs));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));
  EXPECT_FALSE(OutputsBF16(body_dot));
  EXPECT_FALSE(OutputsBF16(body_rhs));
  EXPECT_FALSE(OutputsBF16(body_lhs));
  EXPECT_FALSE(OutputsBF16(cond0_lhs));
  EXPECT_FALSE(OutputsBF16(cond0_rhs));
  EXPECT_FALSE(OutputsBF16(cond1_lhs));
  EXPECT_FALSE(OutputsBF16(cond1_rhs));
  EXPECT_TRUE(OutputsBF16(cond0_add_rhs));
  EXPECT_TRUE(OutputsBF16(cond1_add_lhs));
  EXPECT_EQ(computation->root_instruction(), dot);
}

// Tests that if this pass turns an F32 -> BF16 conversion into a no-op (BF16 ->
// BF16 conversion), then it will remove that conversion.
TEST_F(BFloat16PropagationTest, NoopConversionRemoved) {
  auto builder = HloComputation::Builder(TestName());
  Shape f32_shape = ShapeUtil::MakeShape(F32, {4, 4});
  Shape bf16_shape = ShapeUtil::MakeShape(BF16, {4, 4});

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, f32_shape, "param"));
  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kAdd, param, param));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(f32_shape, HloOpcode::kAdd, param, param));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({add0, add1}));
  HloInstruction* gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, tuple, 0));
  HloInstruction* gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, tuple, 1));
  HloInstruction* convert0 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, gte0));
  HloInstruction* convert1 =
      builder.AddInstruction(HloInstruction::CreateConvert(bf16_shape, gte1));
  HloInstruction* add2 = builder.AddInstruction(HloInstruction::CreateBinary(
      bf16_shape, HloOpcode::kAdd, convert0, convert1));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), add2);
  EXPECT_EQ(add2->operand(0), add0);
  EXPECT_EQ(add2->operand(1), add1);
  EXPECT_EQ(add0->shape().element_type(), BF16);
  EXPECT_EQ(add1->shape().element_type(), BF16);
}

TEST_F(BFloat16PropagationTest, TupleDomain) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* a_trans =
      builder.AddInstruction(HloInstruction::CreateTranspose(shape, a, {0, 1}));
  HloInstruction* b_trans =
      builder.AddInstruction(HloInstruction::CreateTranspose(shape, b, {0, 1}));
  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({a_trans, b_trans}));
  HloInstruction* domain = builder.AddInstruction(
      HloInstruction::CreateDomain(tuple->shape(), tuple, nullptr, nullptr));
  HloInstruction* a_gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, domain, 0));
  HloInstruction* b_gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, domain, 1));
  HloInstruction* dot = builder.AddInstruction(CreateDot(shape, a_gte, b_gte));
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, dot, dot));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));
  EXPECT_EQ(computation->root_instruction(), root);

  // test BF16 propagated through domain
  EXPECT_EQ(ShapeUtil::GetTupleElementShape(domain->shape(), 0).element_type(),
            BF16);
  EXPECT_EQ(ShapeUtil::GetTupleElementShape(domain->shape(), 1).element_type(),
            BF16);

  EXPECT_TRUE(OutputsBF16(a_trans));
  EXPECT_TRUE(OutputsBF16(b_trans));
  EXPECT_TRUE(OutputsBF16(a_gte));
  EXPECT_TRUE(OutputsBF16(b_gte));
  EXPECT_FALSE(OutputsBF16(a));
  EXPECT_FALSE(OutputsBF16(b));
}

// Tests that BF16 is not propagated through a domain in case its input cannot
// be propagated. In the case below the input of the domain is the parameter
// tuple which cannot be propagated, so the domain instruction is not propagated
// either.
TEST_F(BFloat16PropagationTest, TupleDomainNoPropagation) {
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {4, 4});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({shape, shape});

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));
  HloInstruction* domain = builder.AddInstruction(
      HloInstruction::CreateDomain(param->shape(), param, nullptr, nullptr));
  HloInstruction* a_gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, domain, 0));
  HloInstruction* b_gte = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, domain, 1));
  HloInstruction* a_trans = builder.AddInstruction(
      HloInstruction::CreateTranspose(shape, a_gte, {0, 1}));
  HloInstruction* b_trans = builder.AddInstruction(
      HloInstruction::CreateTranspose(shape, b_gte, {0, 1}));
  HloInstruction* dot =
      builder.AddInstruction(CreateDot(shape, a_trans, b_trans));
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, dot, dot));

  auto module = CreateNewVerifiedModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), root);
  EXPECT_TRUE(OutputsBF16(a_trans));
  EXPECT_TRUE(OutputsBF16(b_trans));
  EXPECT_FALSE(OutputsBF16(a_gte));
  EXPECT_FALSE(OutputsBF16(b_gte));
  EXPECT_FALSE(OutputsBF16(domain));
  EXPECT_FALSE(OutputsBF16(param));
}

TEST_F(BFloat16PropagationTest, ConditionalSeparateBranchOperands) {
  const std::string module_str = R"(
HloModule module

true_branch {
  true_param = f32[4096,4096] parameter(0)
  ROOT max = f32[4096,4096] maximum(true_param, true_param)
}

false_branch {
  false_param = f32[4096,4096] parameter(0)
  ROOT add = f32[4096,4096] add(false_param, false_param)
}

ENTRY entry {
  param0 = f32[4096,4096] parameter(0)
  param1 = f32[4096,4096] parameter(1)
  copy0 = f32[4096,4096] copy(param0)
  copy1 = f32[4096,4096] copy(param1)
  param2 = pred[] parameter(2)
  conditional = f32[4096,4096] conditional(param2, copy0, copy1),
    true_computation=true_branch, false_computation=false_branch
  ROOT dot = f32[4096,4096] dot(conditional, conditional),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  auto cond = FindInstruction(module.get(), "conditional");
  auto copy0 = FindInstruction(module.get(), "copy0");
  auto copy1 = FindInstruction(module.get(), "copy1");
  EXPECT_TRUE(OutputsBF16(cond));
  EXPECT_TRUE(OutputsBF16(copy0));
  EXPECT_FALSE(OutputsBF16(copy1));
}

TEST_F(BFloat16PropagationTest, ConditionalSharedBranchOperands) {
  const std::string module_str = R"(
HloModule module

true_branch {
  true_param = f32[4096,4096] parameter(0)
  ROOT max = f32[4096,4096] maximum(true_param, true_param)
}

false_branch {
  false_param = f32[4096,4096] parameter(0)
  ROOT add = f32[4096,4096] add(false_param, false_param)
}

ENTRY entry {
  param0 = f32[4096,4096] parameter(0)
  copy0 = f32[4096,4096] copy(param0)
  param1 = pred[] parameter(1)
  conditional = f32[4096,4096] conditional(param1, copy0, copy0),
    true_computation=true_branch, false_computation=false_branch
  ROOT dot = f32[4096,4096] dot(conditional, conditional),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  auto cond = FindInstruction(module.get(), "conditional");
  auto copy0 = FindInstruction(module.get(), "copy0");
  EXPECT_TRUE(OutputsBF16(cond));
  EXPECT_FALSE(OutputsBF16(copy0));
}

TEST_F(BFloat16PropagationTest, ConditionalAliasingOutputs) {
  const std::string module_str = R"(
HloModule module

true_branch {
  true_param = f32[4096,4096] parameter(0)
  max = f32[4096,4096] maximum(true_param, true_param)
  ROOT true_tuple = (f32[4096,4096], f32[4096,4096]) tuple(max, max)
}

false_branch {
  false_param = f32[4096,4096] parameter(0)
  min = f32[4096,4096] minimum(false_param, false_param)
  max2 = f32[4096,4096] maximum(false_param, false_param)
  ROOT false_tuple = (f32[4096,4096], f32[4096,4096]) tuple(min, max2)
}

ENTRY entry {
  param0 = f32[4096,4096] parameter(0)
  copy0 = f32[4096,4096] copy(param0)
  param1 = pred[] parameter(1)
  conditional = (f32[4096,4096], f32[4096,4096]) conditional(param1, copy0, copy0),
    true_computation=true_branch, false_computation=false_branch
  gte0 = f32[4096,4096] get-tuple-element(conditional), index=0
  gte1 = f32[4096,4096] get-tuple-element(conditional), index=1
  dot = f32[4096,4096] dot(gte0, gte1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tuple = (f32[4096,4096], f32[4096,4096]) tuple(dot, gte1)
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  EXPECT_FALSE(PropagatePrecision(module.get()));
}

TEST_F(BFloat16PropagationTest, DynamicUpdateSlice) {
  // This test is crafted so that the DUS has an F32 input (due to parameter)
  // and BF16 output (due to dot). But we should enforce DUS operand 0 and
  // output to get the same precision since it's an in-place operation.
  const std::string module_str = R"(
HloModule Module

ENTRY main {
  param = f32[128,128] parameter(0)
  constant.1 = f32[] constant(0)
  broadcast.6 = f32[128,1] broadcast(constant.1), dimensions={}
  constant.3 = s32[] constant(0)
  dynamic-update-slice = f32[128,128] dynamic-update-slice(param, broadcast.6, constant.3, constant.3)
  ROOT dot = f32[128,128] dot(dynamic-update-slice, dynamic-update-slice), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  EXPECT_FALSE(PropagatePrecision(module.get()));

  HloInstruction* dus = module->entry_computation()->GetInstructionWithName(
      "dynamic-update-slice");
  EXPECT_FALSE(OutputsBF16(dus));
}

TEST_F(BFloat16PropagationTest, DynamicSliceWithHostMemory) {
  // In the case of dynamic-slice from host memory, we should not propagate
  // BF16.
  const std::string module_str = R"(
  HloModule Module

  ENTRY main {
    param = f32[128,128]{1,0:S(5)} parameter(0)
    constant.3 = s32[] constant(0)
    dynamic-slice = f32[128,8] dynamic-slice(param, constant.3, constant.3), dynamic_slice_sizes={128,8}
    ROOT dot = f32[128,128] dot(dynamic-slice, dynamic-slice), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  }
  )";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  EXPECT_FALSE(PropagatePrecision(module.get()));

  HloInstruction* dus =
      module->entry_computation()->GetInstructionWithName("dynamic-slice");
  EXPECT_FALSE(OutputsBF16(dus));
}

// This test demonstrates the need for invoking the ResolveAliasingBuffer
// multiple times via a fixed-point algorithm. The key was the aliasing of the
// two output buffers of the conditional, at subshape 0 (first element). This
// aliasing is not resolved until after the gte0 variable is already processed,
// triggering incorrect type for gte0 if not repeating the aliasing analysis.
TEST_F(BFloat16PropagationTest, ConditionalGTEWithFusion) {
  const std::string module_str = R"(
HloModule module

%add.0 (x: f32[4096,4096], y: f32[4096,4096]) -> f32[4096,4096] {
  x.1 = f32[4096,4096] parameter(0)
  y.1 = f32[4096,4096] parameter(1)
  ROOT dot1 = f32[4096,4096] dot(x.1, y.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

%add.1 (x: f32[4096,4096], y: f32[4096,4096]) -> f32[4096,4096] {
  x.1 = f32[4096,4096] parameter(0)
  y.1 = f32[4096,4096] parameter(1)
  ROOT dot1 = f32[4096,4096] dot(x.1, y.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

%add.2 (x: f32[4096,4096], y: f32[4096,4096]) -> f32[4096,4096] {
  x.1 = f32[4096,4096] parameter(0)
  y.1 = f32[4096,4096] parameter(1)
  ROOT dot1 = f32[4096,4096] dot(x.1, y.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

%add.3 (x: f32[4096,4096], y: f32[4096,4096]) -> f32[4096,4096] {
  x.1 = f32[4096,4096] parameter(0)
  y.1 = f32[4096,4096] parameter(1)
  ROOT dot1 = f32[4096,4096] dot(x.1, y.1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

true_branch {
  true_param = f32[4096,4096] parameter(0)
  constant.1 = f32[4096,4096] constant(0)
  add0 = f32[4096,4096] fusion(true_param,true_param), kind=kLoop, calls=add.0
  constant.2 = f32[4096,4096] constant(0)
  ROOT tuple.2 = (f32[4096,4096], f32[4096,4096], f32[4096,4096]) tuple(true_param,add0,constant.2)
}

false_branch {
  false_param = f32[4096,4096] parameter(0)
  add3 = f32[4096,4096] fusion(false_param,false_param), kind=kLoop, calls=add.1
  constant.1 = f32[4096,4096] constant(0)
  ROOT tuple.2 = (f32[4096,4096], f32[4096,4096], f32[4096,4096]) tuple(add3, add3,constant.1)
}

ENTRY entry {
  param0 = f32[4096,4096] parameter(0)
  copy0 = f32[4096,4096] copy(param0)
  param1 = pred[] parameter(1)
  conditional = (f32[4096,4096], f32[4096,4096], f32[4096,4096]) conditional(param1, param0, copy0),
    true_computation=true_branch, false_computation=false_branch
  gte = f32[4096,4096] get-tuple-element(conditional), index=0
  gte1 = f32[4096,4096] get-tuple-element(conditional), index=1
  gte2 = f32[4096,4096] get-tuple-element(conditional), index=2
  add2 = f32[4096,4096] fusion(gte, gte1), kind=kLoop, calls=add.2
  ROOT add3 = f32[4096,4096] fusion(add2, gte2), kind=kLoop, calls=add.3
  }
)";

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                          ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));
  VLOG(2) << module->ToString() << "\n";
  EXPECT_TRUE(HloVerifier(/*layout_sensitive=*/false,
                          /*allow_mixed_precision=*/true)
                  .Run(module.get())
                  .status()
                  .ok());
  auto gte = FindInstruction(module.get(), "gte");
  auto gte1 = FindInstruction(module.get(), "gte1");
  auto gte2 = FindInstruction(module.get(), "gte2");
  EXPECT_FALSE(OutputsBF16(gte));
  EXPECT_FALSE(OutputsBF16(gte1));
  EXPECT_TRUE(OutputsBF16(gte2));
}

// =============================================================================
// Tests for HloScanInstruction propagation. The scan path mirrors kWhile
// (carry slots alias init -> body parameter -> body root -> result carry) and
// adds a body-internal duplicator pre-pass to break shared dataflow values
// across tuple-root slots (e.g. the canonical JAX cumsum pattern emits
// `tuple(add, add)` to use one value for both the per-step output and the
// next carry).
// =============================================================================

// Tests that BF16 is propagated end-to-end through an associative scan whose
// downstream consumer wants BF16. The scan body uses a single `add` value for
// both the per-step output slot and the carry slot; the body-internal
// duplicator should insert a self-convert on one of them so that both slots
// can independently lower to BF16.
TEST_F(BFloat16PropagationTest, PropagateThroughAssociativeScanIntoDot) {
  constexpr absl::string_view kHlo = R"(
HloModule scan_into_dot

%combine (carry: f32[4,4], slice: f32[4,4]) -> (f32[4,4], f32[4,4]) {
  %slice = f32[4,4] parameter(1)
  %carry = f32[4,4] parameter(0)
  %add = f32[4,4] add(%slice, %carry)
  ROOT %tup = (f32[4,4], f32[4,4]) tuple(%add, %add)
}

ENTRY main {
  %arg = f32[3,4,4] parameter(0)
  %zero = f32[] constant(0)
  %init = f32[4,4] broadcast(%zero), dimensions={}
  %scan = (f32[3,4,4], f32[4,4]) scan(%arg, %init),
      dimensions={0}, num_carries=1, is_associative=true, to_apply=%combine
  %scan_out = f32[3,4,4] get-tuple-element(%scan), index=0
  %slice = f32[1,4,4] slice(%scan_out), slice={[2:3], [0:4], [0:4]}
  %lhs = f32[4,4] reshape(%slice)
  ROOT %dot = f32[4,4] dot(%lhs, %lhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  // Dot demands BF16 operands (TestBFloat16Support reports
  // EffectiveOperandPrecisionIsLowPrecision == true for kDot), so the slice
  // and reshape feeding it should be BF16, the per-step scan output slot 0
  // should be BF16, and the corresponding body root slot 0 should also be
  // BF16.
  HloInstruction* lhs = FindInstruction(module.get(), "lhs");
  ASSERT_NE(lhs, nullptr);
  EXPECT_TRUE(OutputsBF16(lhs));
  HloInstruction* scan_out = FindInstruction(module.get(), "scan_out");
  ASSERT_NE(scan_out, nullptr);
  EXPECT_TRUE(OutputsBF16(scan_out));

  // Inspect the scan body: at least slot 0 of the body root tuple should be
  // BF16 (the one feeding the per-step output that downstream consumes).
  HloInstruction* scan = FindInstruction(module.get(), "scan");
  ASSERT_NE(scan, nullptr);
  ASSERT_EQ(scan->opcode(), HloOpcode::kScan);
  HloInstruction* body_root = scan->to_apply()->root_instruction();
  ASSERT_EQ(body_root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(body_root->shape().tuple_shapes(0).element_type(), BF16);
}

// Tests the body-internal duplicator: a body with `tuple(add, add)` should
// have a self-convert inserted on the duplicate slot so that the two body
// root slots reference distinct dataflow values. After propagation, when
// only the per-step output is consumed in BF16 but the carry isn't, the two
// slots can resolve to different precisions (BF16 output, F32 carry).
TEST_F(BFloat16PropagationTest, ScanBodyDuplicatorBreaksTupleAliasing) {
  constexpr absl::string_view kHlo = R"(
HloModule scan_duplicator

%combine (carry: f32[4,4], slice: f32[4,4]) -> (f32[4,4], f32[4,4]) {
  %slice = f32[4,4] parameter(1)
  %carry = f32[4,4] parameter(0)
  %add = f32[4,4] add(%slice, %carry)
  ROOT %tup = (f32[4,4], f32[4,4]) tuple(%add, %add)
}

ENTRY main {
  %arg = f32[3,4,4] parameter(0)
  %zero = f32[] constant(0)
  %init = f32[4,4] broadcast(%zero), dimensions={}
  ROOT %scan = (f32[3,4,4], f32[4,4]) scan(%arg, %init),
      dimensions={0}, num_carries=1, is_associative=true, to_apply=%combine
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  // No downstream consumer demands BF16, so propagation should be a no-op.
  // The duplicator-inserted self-convert is a no-op kConvert and the
  // SkipNoopConversions cleanup at the end of the pass should remove it.
  EXPECT_FALSE(PropagatePrecision(module.get()));

  HloInstruction* scan = FindInstruction(module.get(), "scan");
  ASSERT_NE(scan, nullptr);
  HloInstruction* body_root = scan->to_apply()->root_instruction();
  ASSERT_EQ(body_root->opcode(), HloOpcode::kTuple);
  // After cleanup both slots should reference the original `add` (the
  // no-op convert was removed).
  EXPECT_EQ(body_root->operand(0), body_root->operand(1));
}

// An associative scan with two distinct body root slots (`sum` and `prod`)
// is tracked per slot. Both slots end up BF16 because no downstream
// consumer demands F32.
TEST_F(BFloat16PropagationTest, ScanWithDistinctRootSlotsLowersPerSlot) {
  constexpr absl::string_view kHlo = R"(
HloModule scan_distinct_slots

%combine (carry: f32[4,4], slice: f32[4,4]) -> (f32[4,4], f32[4,4]) {
  %slice = f32[4,4] parameter(1)
  %carry = f32[4,4] parameter(0)
  %sum = f32[4,4] add(%slice, %carry)
  %prod = f32[4,4] multiply(%slice, %carry)
  ROOT %tup = (f32[4,4], f32[4,4]) tuple(%sum, %prod)
}

ENTRY main {
  %arg = f32[3,4,4] parameter(0)
  %one = f32[] constant(1)
  %init = f32[4,4] broadcast(%one), dimensions={}
  %scan = (f32[3,4,4], f32[4,4]) scan(%arg, %init),
      dimensions={0}, num_carries=1, is_associative=true, to_apply=%combine
  %out = f32[3,4,4] get-tuple-element(%scan), index=0
  %slice = f32[1,4,4] slice(%out), slice={[2:3], [0:4], [0:4]}
  %lhs = f32[4,4] reshape(%slice)
  ROOT %dot = f32[4,4] dot(%lhs, %lhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  // The dot demands BF16 on slot 0, so the per-step output goes BF16. The
  // carry chain (slot 1) has no BF16 demand and stays F32 end to end.
  HloInstruction* scan = FindInstruction(module.get(), "scan");
  ASSERT_NE(scan, nullptr);
  ASSERT_EQ(scan->opcode(), HloOpcode::kScan);
  HloInstruction* body_root = scan->to_apply()->root_instruction();
  ASSERT_EQ(body_root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(body_root->shape().tuple_shapes(0).element_type(), BF16);
  EXPECT_EQ(body_root->shape().tuple_shapes(1).element_type(), F32);
  // The two slots are backed by distinct dataflow values (no duplicator).
  EXPECT_NE(body_root->operand(0), body_root->operand(1));
  // Slot 0's op was upgraded to BF16, directly or via a convert from
  // ResolveInconsistentScans.
  EXPECT_EQ(body_root->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(body_root->operand(1)->shape().element_type(), F32);
}

// Non-associative scans must be expanded by ScanExpander before this pass;
// a bare one must die on the CHECK in ShouldKeepPrecisionUnchanged instead
// of producing an inconsistent module.
TEST_F(BFloat16PropagationTest, NonAssociativeScanCrashesWithoutScanExpander) {
  constexpr absl::string_view kHlo = R"(
HloModule non_associative_scan

%combine (carry: f32[4,4], slice: f32[4,4]) -> (f32[4,4], f32[4,4]) {
  %slice = f32[4,4] parameter(1)
  %carry = f32[4,4] parameter(0)
  %add = f32[4,4] add(%slice, %carry)
  ROOT %tup = (f32[4,4], f32[4,4]) tuple(%add, %add)
}

ENTRY main {
  %arg = f32[3,4,4] parameter(0)
  %zero = f32[] constant(0)
  %init = f32[4,4] broadcast(%zero), dimensions={}
  %scan = (f32[3,4,4], f32[4,4]) scan(%arg, %init),
      dimensions={0}, num_carries=1, is_associative=false, to_apply=%combine
  %scan_out = f32[3,4,4] get-tuple-element(%scan), index=0
  %slice = f32[1,4,4] slice(%scan_out), slice={[2:3], [0:4], [0:4]}
  %lhs = f32[4,4] reshape(%slice)
  ROOT %dot = f32[4,4] dot(%lhs, %lhs),
      lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  EXPECT_DEATH(PropagatePrecision(module.get()),
               "Non-associative kScan reached BFloat16Propagation");
}

// A 1-carry/0-output associative scan has non-tuple body root and result
// shapes, so the carry slot is ShapeIndex{}. AddScanCarryAlignEdges must
// not CHECK-fail while indexing them.
TEST_F(BFloat16PropagationTest, ScanCarryAlignmentHandlesNonTupleRoot) {
  // 1 input + 1 carry, 0 outputs: scan result and body root are single
  // arrays holding the carry directly.
  constexpr absl::string_view kHlo = R"(
HloModule scan_no_outputs

%combine (carry: f32[4,4], slice: f32[4,4]) -> f32[4,4] {
  %carry = f32[4,4] parameter(0)
  %slice = f32[4,4] parameter(1)
  ROOT %add = f32[4,4] add(%carry, %slice)
}

ENTRY main {
  %arg = f32[3,4,4] parameter(0)
  %zero = f32[] constant(0)
  %init = f32[4,4] broadcast(%zero), dimensions={}
  ROOT %scan = f32[4,4] scan(%arg, %init),
      dimensions={0}, num_carries=1, is_associative=true, to_apply=%combine
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  // No BF16 demand on the carry chain: it stays F32, and the pass must not
  // crash on the non-tuple shapes.
  PropagatePrecision(module.get());

  HloInstruction* scan = FindInstruction(module.get(), "scan");
  ASSERT_NE(scan, nullptr);
  ASSERT_EQ(scan->opcode(), HloOpcode::kScan);
  // Body root and scan result must remain non-tuple F32 arrays.
  EXPECT_FALSE(scan->shape().IsTuple());
  EXPECT_EQ(scan->shape().element_type(), F32);
  HloInstruction* body_root = scan->to_apply()->root_instruction();
  EXPECT_FALSE(body_root->shape().IsTuple());
  EXPECT_EQ(body_root->shape().element_type(), F32);
}

// Runs the pass with the real xla::FloatSupport (not the fixture's
// blanket-true overrides), to catch regressions that drop kScan from
// SupportsLowPrecisionOutput / SupportsLowPrecisionOperand.
TEST_F(BFloat16PropagationTest,
       AssociativeScanBF16PromotedUnderProductionFloatSupport) {
  constexpr absl::string_view kHlo = R"(
HloModule scan_production_float_support

%combine (carry: f32[4,4], slice: f32[4,4]) -> (f32[4,4], f32[4,4]) {
  %slice = f32[4,4] parameter(1)
  %carry = f32[4,4] parameter(0)
  %add = f32[4,4] add(%slice, %carry)
  ROOT %tup = (f32[4,4], f32[4,4]) tuple(%add, %add)
}

ENTRY main {
  %arg = f32[3,4,4] parameter(0)
  %zero = f32[] constant(0)
  %init = f32[4,4] broadcast(%zero), dimensions={}
  %scan = (f32[3,4,4], f32[4,4]) scan(%arg, %init),
      dimensions={0}, num_carries=1, is_associative=true, to_apply=%combine
  %scan_out = f32[3,4,4] get-tuple-element(%scan), index=0
  // Explicit F32->BF16 convert that propagation should subsume by lowering
  // the scan's per-step output to BF16 directly.
  ROOT %bf = bf16[3,4,4] convert(%scan_out)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));

  // With the production FloatSupport the per-step output must be promoted
  // to BF16 (the explicit convert is then removed as a no-op).
  FloatSupport production_bf16_support(BF16);
  BFloat16Propagation propagation(&production_bf16_support, &alias_info_);
  TF_ASSERT_OK_AND_ASSIGN(bool changed, propagation.Run(module.get()));
  EXPECT_TRUE(changed) << "Production FloatSupport should propagate BF16 "
                          "through associative scan; if this fails, kScan "
                          "may have been dropped from "
                          "FloatSupport::SupportsLowPrecisionOutput / "
                          "SupportsLowPrecisionOperand.";

  HloInstruction* scan = FindInstruction(module.get(), "scan");
  ASSERT_NE(scan, nullptr);
  ASSERT_EQ(scan->opcode(), HloOpcode::kScan);
  ASSERT_TRUE(scan->shape().IsTuple());
  // Slot 0 (per-step output) should now be BF16; the carry (slot 1) is left
  // F32 because no consumer demanded BF16 of the carry chain.
  EXPECT_EQ(scan->shape().tuple_shapes(0).element_type(), BF16);
  EXPECT_EQ(scan->shape().tuple_shapes(1).element_type(), F32);

  // The post-pass module must verify cleanly under mixed precision.
  EXPECT_TRUE(HloVerifier(/*layout_sensitive=*/false,
                          /*allow_mixed_precision=*/true)
                  .Run(module.get())
                  .status()
                  .ok());
}

// A fully BF16 carry chain must stay BF16. Carry alignment only acts when
// the carry init is F32 (rewriting a BF16 init could affect its other
// consumers), so the pass must not introduce a carry_init vs body_parameter
// mismatch here.
TEST_F(BFloat16PropagationTest, ScanCarryAlignmentSkipsBf16CarryInit) {
  constexpr absl::string_view kHlo = R"(
HloModule scan_bf16_carry_init

%combine (carry: bf16[4,4], slice: bf16[4,4]) -> (bf16[4,4], bf16[4,4]) {
  %carry = bf16[4,4] parameter(0)
  %slice = bf16[4,4] parameter(1)
  %add = bf16[4,4] add(%carry, %slice)
  ROOT %tup = (bf16[4,4], bf16[4,4]) tuple(%add, %add)
}

ENTRY main {
  %arg = bf16[3,4,4] parameter(0)
  %init = bf16[4,4] parameter(1)
  %scan = (bf16[3,4,4], bf16[4,4]) scan(%arg, %init),
      dimensions={0}, num_carries=1, is_associative=true, to_apply=%combine
  %out = bf16[3,4,4] get-tuple-element(%scan), index=0
  // Force the per-step output up to F32 downstream so propagation has work
  // to do, but leave the carry chain entirely BF16.
  ROOT %f = f32[3,4,4] convert(%out)
}
)";
  TF_ASSERT_OK_AND_ASSIGN(auto module, ParseAndReturnVerifiedModule(kHlo));
  PropagatePrecision(module.get());

  HloInstruction* scan = FindInstruction(module.get(), "scan");
  ASSERT_NE(scan, nullptr);
  ASSERT_EQ(scan->opcode(), HloOpcode::kScan);
  ASSERT_TRUE(scan->shape().IsTuple());

  // The carry slot must remain BF16 across init / body parameter / body
  // root / scan result.
  EXPECT_EQ(scan->operand(1)->shape().element_type(), BF16);  // carry init.
  HloComputation* body = scan->to_apply();
  EXPECT_EQ(body->parameter_instruction(1)->shape().element_type(), BF16);
  HloInstruction* body_root = body->root_instruction();
  ASSERT_EQ(body_root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(body_root->shape().tuple_shapes(1).element_type(), BF16);
  EXPECT_EQ(scan->shape().tuple_shapes(1).element_type(), BF16);

  // A downgrade would leave scan operand BF16 vs body parameter F32, which
  // the verifier rejects.
  EXPECT_TRUE(HloVerifier(/*layout_sensitive=*/false,
                          /*allow_mixed_precision=*/true)
                  .Run(module.get())
                  .status()
                  .ok());
}

TEST_F(BFloat16PropagationTest,
       DoNotPropagateThroughBitcastDifferentElementTypes) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape u32_shape = ShapeUtil::MakeShape(U32, {4, 4});
  Shape f32_shape = ShapeUtil::MakeShape(F32, {4, 4});

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, u32_shape, "param"));
  HloInstruction* bitcast =
      builder.AddInstruction(HloInstruction::CreateBitcast(f32_shape, param));
  auto dot = builder.AddInstruction(CreateDot(f32_shape, bitcast, bitcast));

  auto computation = module->AddEntryComputation(builder.Build());
  PropagatePrecision(module.get());

  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_EQ(bitcast->shape().element_type(), F32);
  EXPECT_EQ(param->shape().element_type(), U32);
  EXPECT_FALSE(OutputsBF16(bitcast));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

TEST_F(BFloat16PropagationTest, BitcastDifferentElementTypesInWhileLoop) {
  auto module = CreateNewVerifiedModule();
  auto builder = HloComputation::Builder(TestName());
  Shape u32_shape = ShapeUtil::MakeShape(U32, {4, 4});
  Shape f32_shape = ShapeUtil::MakeShape(F32, {4, 4});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({u32_shape, f32_shape});

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "param"));

  auto builder_cond = HloComputation::Builder("cond");
  builder_cond.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "cond_param"));
  builder_cond.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  auto cond = module->AddEmbeddedComputation(builder_cond.Build());

  auto builder_body = HloComputation::Builder("body");
  auto body_param = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, tuple_shape, "body_param"));
  auto u32_elem = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(u32_shape, body_param, 0));
  auto f32_elem = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, body_param, 1));
  auto bitcast = builder_body.AddInstruction(
      HloInstruction::CreateBitcast(f32_shape, u32_elem));
  auto body_dot =
      builder_body.AddInstruction(CreateDot(f32_shape, bitcast, f32_elem));
  builder_body.AddInstruction(
      HloInstruction::CreateTuple({u32_elem, body_dot}));
  auto body = module->AddEmbeddedComputation(builder_body.Build());

  auto while_hlo = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, cond, body, param));

  auto while_out_f32 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(f32_shape, while_hlo, 1));
  auto entry_dot = builder.AddInstruction(
      CreateDot(f32_shape, while_out_f32, while_out_f32));
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), entry_dot);
  EXPECT_EQ(bitcast->shape().element_type(), F32);
  EXPECT_EQ(u32_elem->shape().element_type(), U32);
  EXPECT_FALSE(OutputsBF16(bitcast));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

TEST_F(BFloat16PropagationTest, BitcastDifferentElementTypesIsWhileRoot) {
  const std::string module_str = R"(
HloModule module

cond {
  cond_param = f32[4,4] parameter(0)
  ROOT cond_root = pred[] constant(true)
}

body {
  body_param = f32[4,4] parameter(0)
  bitcast_in = u32[4,4] bitcast(body_param)
  ROOT bitcast_out = f32[4,4] bitcast(bitcast_in)
}

ENTRY entry {
  param = f32[4,4] parameter(0)
  while = f32[4,4] while(param), condition=cond, body=body
  ROOT root_dot = f32[4,4] dot(while, while),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_FALSE(PropagatePrecision(module.get()));
  auto bitcast_out = FindInstruction(module.get(), "bitcast_out");
  ASSERT_NE(bitcast_out, nullptr);
  EXPECT_FALSE(OutputsBF16(bitcast_out));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

TEST_F(BFloat16PropagationTest, BitcastBf16ToF32InWhileLoop) {
  const std::string module_str = R"(
HloModule module

cond {
  cond_param = f32[4,4] parameter(0)
  ROOT cond_root = pred[] constant(true)
}

body {
  body_param = f32[4,4] parameter(0)
  bitcast_in = bf16[4,4] bitcast(body_param)
  ROOT bitcast_out = f32[4,4] bitcast(bitcast_in)
}

ENTRY entry {
  param = f32[4,4] parameter(0)
  while = f32[4,4] while(param), condition=cond, body=body
  ROOT root_dot = f32[4,4] dot(while, while),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_FALSE(PropagatePrecision(module.get()));
  auto bitcast_out = FindInstruction(module.get(), "bitcast_out");
  ASSERT_NE(bitcast_out, nullptr);
  EXPECT_FALSE(OutputsBF16(bitcast_out));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// A while body rotates its tuple state through copies and a fusion; slot 0
// of the while output is pinned F32 by the module root. The F32 requirement
// must ripple around the whole rotation ring.
TEST_F(BFloat16PropagationTest, WhileWithRotatingTupleState) {
  constexpr absl::string_view module_str = R"hlo(
HloModule while_ring_ripple

fused_scale {
  fparam = f32[4,4] parameter(0)
  ROOT fmul = f32[4,4] multiply(fparam, fparam)
}

cond {
  cond_param = (f32[4,4], f32[4,4], f32[4,4]) parameter(0)
  ROOT cond_root = pred[] constant(true)
}

body {
  body_param = (f32[4,4], f32[4,4], f32[4,4]) parameter(0)
  gte0 = f32[4,4] get-tuple-element(body_param), index=0
  gte1 = f32[4,4] get-tuple-element(body_param), index=1
  gte2 = f32[4,4] get-tuple-element(body_param), index=2
  fus = f32[4,4] fusion(gte0), kind=kLoop, calls=fused_scale
  new0 = f32[4,4] copy(gte2)
  new1 = f32[4,4] copy(fus)
  new2 = f32[4,4] copy(gte1)
  ROOT body_root = (f32[4,4], f32[4,4], f32[4,4]) tuple(new0, new1, new2)
}

ENTRY entry {
  a = f32[4,4] parameter(0)
  b = f32[4,4] parameter(1)
  init = (f32[4,4], f32[4,4], f32[4,4]) tuple(a, a, b)
  while = (f32[4,4], f32[4,4], f32[4,4]) while(init), condition=cond, body=body
  out0 = f32[4,4] get-tuple-element(while), index=0
  out1 = f32[4,4] get-tuple-element(while), index=1
  out2 = f32[4,4] get-tuple-element(while), index=2
  dot1 = f32[4,4] dot(out1, out1),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dot2 = f32[4,4] dot(out2, out2),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  x = f32[4,4] add(a, b)
  dot3 = f32[4,4] dot(x, x),
    lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = (f32[4,4], f32[4,4], f32[4,4], f32[4,4]) tuple(out0, dot1, dot2, dot3)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  // The F32 requirement on slot 0 must have rippled around the whole
  // rotation ring: every while state slot, body root slot, and the fusion
  // stay F32.
  HloInstruction* while_hlo = FindInstruction(module.get(), "while");
  ASSERT_NE(while_hlo, nullptr);
  for (int64_t i = 0; i < 3; ++i) {
    EXPECT_EQ(while_hlo->shape().tuple_shapes(i).element_type(), F32)
        << "while slot " << i;
    EXPECT_EQ(while_hlo->while_body()
                  ->root_instruction()
                  ->shape()
                  .tuple_shapes(i)
                  .element_type(),
              F32)
        << "body root slot " << i;
  }
  HloInstruction* fus = FindInstruction(module.get(), "fus");
  ASSERT_NE(fus, nullptr);
  EXPECT_FALSE(OutputsBF16(fus));
  // The independent chain into dot3 still becomes BF16.
  HloInstruction* x = FindInstruction(module.get(), "x");
  ASSERT_NE(x, nullptr);
  EXPECT_TRUE(OutputsBF16(x));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// Three nested whiles with rotating tuple state. Slot 0 is pinned F32 by
// the entry root; slots 1 and 2 rotate and feed dots, so they may go BF16
// only if every nesting level agrees.
TEST_F(BFloat16PropagationTest, NestedWhileRotatingPinnedSlot) {
  constexpr absl::string_view module_str = R"hlo(
HloModule NestedWhileRotatingPinnedSlot

cond3 {
  p3c = (f32[4,4], f32[4,4], f32[4,4], s32[]) parameter(0)
  i3 = s32[] get-tuple-element(p3c), index=3
  limit3 = s32[] constant(10)
  ROOT lt3 = pred[] compare(i3, limit3), direction=LT
}

body3 {
  p3 = (f32[4,4], f32[4,4], f32[4,4], s32[]) parameter(0)
  a3 = f32[4,4] get-tuple-element(p3), index=0
  b3 = f32[4,4] get-tuple-element(p3), index=1
  c3 = f32[4,4] get-tuple-element(p3), index=2
  i3b = s32[] get-tuple-element(p3), index=3
  one3 = s32[] constant(1)
  ni3 = s32[] add(i3b, one3)
  na3 = f32[4,4] maximum(a3, a3)
  prod3 = f32[4,4] dot(b3, c3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT t3 = (f32[4,4], f32[4,4], f32[4,4], s32[]) tuple(na3, c3, prod3, ni3)
}

cond2 {
  p2c = (f32[4,4], f32[4,4], f32[4,4], s32[]) parameter(0)
  i2 = s32[] get-tuple-element(p2c), index=3
  limit2 = s32[] constant(10)
  ROOT lt2 = pred[] compare(i2, limit2), direction=LT
}

body2 {
  p2 = (f32[4,4], f32[4,4], f32[4,4], s32[]) parameter(0)
  w3 = (f32[4,4], f32[4,4], f32[4,4], s32[]) while(p2), condition=cond3, body=body3
  a2 = f32[4,4] get-tuple-element(w3), index=0
  b2 = f32[4,4] get-tuple-element(w3), index=1
  c2 = f32[4,4] get-tuple-element(w3), index=2
  i2b = s32[] get-tuple-element(w3), index=3
  ROOT t2 = (f32[4,4], f32[4,4], f32[4,4], s32[]) tuple(a2, c2, b2, i2b)
}

cond1 {
  p1c = (f32[4,4], f32[4,4], f32[4,4], s32[]) parameter(0)
  i1 = s32[] get-tuple-element(p1c), index=3
  limit1 = s32[] constant(10)
  ROOT lt1 = pred[] compare(i1, limit1), direction=LT
}

body1 {
  p1 = (f32[4,4], f32[4,4], f32[4,4], s32[]) parameter(0)
  w2 = (f32[4,4], f32[4,4], f32[4,4], s32[]) while(p1), condition=cond2, body=body2
  a1 = f32[4,4] get-tuple-element(w2), index=0
  b1 = f32[4,4] get-tuple-element(w2), index=1
  c1 = f32[4,4] get-tuple-element(w2), index=2
  i1b = s32[] get-tuple-element(w2), index=3
  ROOT t1 = (f32[4,4], f32[4,4], f32[4,4], s32[]) tuple(a1, c1, b1, i1b)
}

ENTRY main {
  pa = f32[4,4] parameter(0)
  pb = f32[4,4] parameter(1)
  pc = f32[4,4] parameter(2)
  zero = s32[] constant(0)
  init = (f32[4,4], f32[4,4], f32[4,4], s32[]) tuple(pa, pb, pc, zero)
  w1 = (f32[4,4], f32[4,4], f32[4,4], s32[]) while(init), condition=cond1, body=body1
  ga = f32[4,4] get-tuple-element(w1), index=0
  gb = f32[4,4] get-tuple-element(w1), index=1
  gc = f32[4,4] get-tuple-element(w1), index=2
  dbc = f32[4,4] dot(gb, gc), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = (f32[4,4], f32[4,4]) tuple(ga, dbc)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  for (const char* name : {"w1", "w2", "w3"}) {
    HloInstruction* w = FindInstruction(module.get(), name);
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->shape().tuple_shapes(0).element_type(), F32) << name;
    EXPECT_EQ(w->shape().tuple_shapes(1).element_type(), BF16) << name;
    EXPECT_EQ(w->shape().tuple_shapes(2).element_type(), BF16) << name;
  }
  HloInstruction* na3 = FindInstruction(module.get(), "na3");
  ASSERT_NE(na3, nullptr);
  EXPECT_FALSE(OutputsBF16(na3));
  HloInstruction* prod3 = FindInstruction(module.get(), "prod3");
  ASSERT_NE(prod3, nullptr);
  EXPECT_TRUE(OutputsBF16(prod3));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// Two whiles call the same body and condition. One result feeds a dot, the
// other the F32 entry root; the shared body couples them, so both settle
// F32. An uncoupled chain still becomes BF16.
TEST_F(BFloat16PropagationTest, TwoWhilesSharedBodyAndCondition) {
  constexpr absl::string_view module_str = R"hlo(
HloModule TwoWhilesSharedBodyAndCondition

shared_cond {
  pc = (f32[4,4], s32[]) parameter(0)
  ic = s32[] get-tuple-element(pc), index=1
  limit = s32[] constant(5)
  ROOT lt = pred[] compare(ic, limit), direction=LT
}

shared_body {
  pb = (f32[4,4], s32[]) parameter(0)
  xb = f32[4,4] get-tuple-element(pb), index=0
  ib = s32[] get-tuple-element(pb), index=1
  oneb = s32[] constant(1)
  nib = s32[] add(ib, oneb)
  xx = f32[4,4] dot(xb, xb), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tb = (f32[4,4], s32[]) tuple(xx, nib)
}

ENTRY main {
  p0 = f32[4,4] parameter(0)
  p1 = f32[4,4] parameter(1)
  p2 = f32[4,4] parameter(2)
  zero = s32[] constant(0)
  ta = (f32[4,4], s32[]) tuple(p0, zero)
  tbt = (f32[4,4], s32[]) tuple(p1, zero)
  wa = (f32[4,4], s32[]) while(ta), condition=shared_cond, body=shared_body
  wb = (f32[4,4], s32[]) while(tbt), condition=shared_cond, body=shared_body
  ga = f32[4,4] get-tuple-element(wa), index=0
  gb = f32[4,4] get-tuple-element(wb), index=0
  dota = f32[4,4] dot(ga, ga), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  extra = f32[4,4] maximum(p2, p2)
  dotex = f32[4,4] dot(extra, extra), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = (f32[4,4], f32[4,4], f32[4,4]) tuple(dota, gb, dotex)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  for (const char* name : {"wa", "wb"}) {
    HloInstruction* w = FindInstruction(module.get(), name);
    ASSERT_NE(w, nullptr);
    EXPECT_EQ(w->shape().tuple_shapes(0).element_type(), F32) << name;
  }
  HloInstruction* xx = FindInstruction(module.get(), "xx");
  ASSERT_NE(xx, nullptr);
  EXPECT_FALSE(OutputsBF16(xx));
  HloInstruction* extra = FindInstruction(module.get(), "extra");
  ASSERT_NE(extra, nullptr);
  EXPECT_TRUE(OutputsBF16(extra));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// The branches of a conditional share an operand and contain fusions; the
// false branch anchors F32 internally. The conditional output still becomes
// BF16 for the downstream dot.
TEST_F(BFloat16PropagationTest, ConditionalSharedOperandsWithFusions) {
  constexpr absl::string_view module_str = R"hlo(
HloModule ConditionalSharedOperandsWithFusions

fused_dot {
  gp0 = f32[4,4] parameter(0)
  gp1 = f32[4,4] parameter(1)
  ROOT gdot = f32[4,4] dot(gp0, gp1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}

fused_mul {
  fp0 = f32[4,4] parameter(0)
  fp1 = f32[4,4] parameter(1)
  ROOT fmul = f32[4,4] multiply(fp0, fp1)
}

true_branch {
  tparam = f32[4,4] parameter(0)
  tfus = f32[4,4] fusion(tparam, tparam), kind=kLoop, calls=fused_dot
  ROOT tmax = f32[4,4] maximum(tfus, tfus)
}

false_branch {
  fparam = f32[4,4] parameter(0)
  ffus = f32[4,4] fusion(fparam, fparam), kind=kLoop, calls=fused_mul
  ROOT fadd = f32[4,4] add(ffus, fparam)
}

ENTRY main {
  p0 = f32[4,4] parameter(0)
  pr = pred[] parameter(1)
  copy0 = f32[4,4] copy(p0)
  cnd = f32[4,4] conditional(pr, copy0, copy0), true_computation=true_branch, false_computation=false_branch
  ROOT dotc = f32[4,4] dot(cnd, cnd), lhs_contracting_dims={1}, rhs_contracting_dims={0}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  HloInstruction* cnd = FindInstruction(module.get(), "cnd");
  ASSERT_NE(cnd, nullptr);
  EXPECT_TRUE(OutputsBF16(cnd));
  HloInstruction* tfus = FindInstruction(module.get(), "tfus");
  ASSERT_NE(tfus, nullptr);
  EXPECT_TRUE(OutputsBF16(tfus));
  HloInstruction* ffus = FindInstruction(module.get(), "ffus");
  ASSERT_NE(ffus, nullptr);
  EXPECT_FALSE(OutputsBF16(ffus));
  HloInstruction* fadd = FindInstruction(module.get(), "fadd");
  ASSERT_NE(fadd, nullptr);
  EXPECT_TRUE(OutputsBF16(fadd));
  HloInstruction* copy0 = FindInstruction(module.get(), "copy0");
  ASSERT_NE(copy0, nullptr);
  EXPECT_FALSE(OutputsBF16(copy0));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// A kLoop fusion with a tuple root: slot 0 forwards a parameter, slot 1 is
// compute. The fused parameter stays F32 while the fusion output goes BF16;
// ResolveInconsistentFusions patches the boundary with a convert.
TEST_F(BFloat16PropagationTest, TupleFusionPassthroughInsideCall) {
  constexpr absl::string_view module_str = R"hlo(
HloModule TupleFusionPassthroughInsideCall

fused_tuple {
  fp0 = f32[4,4] parameter(0)
  fp1 = f32[4,4] parameter(1)
  fdot = f32[4,4] dot(fp1, fp1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT ftup = (f32[4,4], f32[4,4]) tuple(fp0, fdot)
}

callee {
  cp0 = f32[4,4] parameter(0)
  cp1 = f32[4,4] parameter(1)
  fus = (f32[4,4], f32[4,4]) fusion(cp0, cp1), kind=kLoop, calls=fused_tuple
  g0 = f32[4,4] get-tuple-element(fus), index=0
  g1 = f32[4,4] get-tuple-element(fus), index=1
  d = f32[4,4] dot(g0, g1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT ct = (f32[4,4], f32[4,4]) tuple(d, g1)
}

ENTRY main {
  p0 = f32[4,4] parameter(0)
  p1 = f32[4,4] parameter(1)
  c = (f32[4,4], f32[4,4]) call(p0, p1), to_apply=callee
  gg0 = f32[4,4] get-tuple-element(c), index=0
  gg1 = f32[4,4] get-tuple-element(c), index=1
  dd = f32[4,4] dot(gg0, gg1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT outm = f32[4,4] maximum(dd, dd)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  HloInstruction* fus = FindInstruction(module.get(), "fus");
  ASSERT_NE(fus, nullptr);
  EXPECT_EQ(fus->shape().tuple_shapes(0).element_type(), BF16);
  EXPECT_EQ(fus->shape().tuple_shapes(1).element_type(), BF16);
  // The fused parameter kept F32, so the passthrough slot got a convert.
  HloInstruction* fused_root =
      fus->fused_instructions_computation()->root_instruction();
  ASSERT_EQ(fused_root->opcode(), HloOpcode::kTuple);
  EXPECT_EQ(fused_root->operand(0)->opcode(), HloOpcode::kConvert);
  EXPECT_EQ(fused_root->operand(0)->shape().element_type(), BF16);
  EXPECT_EQ(fus->fused_parameter(0)->shape().element_type(), F32);
  HloInstruction* c = FindInstruction(module.get(), "c");
  ASSERT_NE(c, nullptr);
  EXPECT_EQ(c->shape().tuple_shapes(0).element_type(), BF16);
  EXPECT_EQ(c->shape().tuple_shapes(1).element_type(), BF16);
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// Three DUS in-place chains: one anchored F32 by the entry parameter, one
// with a late F32 anchor that must ripple down the chain, one inside a
// fusion that settles BF16 end to end.
TEST_F(BFloat16PropagationTest, DusChainsInsideAndOutsideFusion) {
  constexpr absl::string_view module_str = R"hlo(
HloModule DusChainsInsideAndOutsideFusion

fused_dus {
  fb = f32[4,4] parameter(0)
  fu = f32[1,4] parameter(1)
  fi = s32[] parameter(2)
  ROOT fd = f32[4,4] dynamic-update-slice(fb, fu, fi, fi)
}

ENTRY main {
  base = f32[4,4] parameter(0)
  upd = f32[1,4] parameter(1)
  i0 = s32[] constant(0)
  dus1 = f32[4,4] dynamic-update-slice(base, upd, i0, i0)
  dus2 = f32[4,4] dynamic-update-slice(dus1, upd, i0, i0)
  dot1 = f32[4,4] dot(dus2, dus2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  seed = f32[4,4] maximum(base, base)
  updm = f32[1,4] maximum(upd, upd)
  dus3 = f32[4,4] dynamic-update-slice(seed, updm, i0, i0)
  dus4 = f32[4,4] dynamic-update-slice(dus3, updm, i0, i0)
  dot2 = f32[4,4] dot(dus4, dus4), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  anchor = f32[4,4] add(dus3, dus3)
  seed2 = f32[4,4] maximum(base, base)
  updm2 = f32[1,4] maximum(upd, upd)
  fusw = f32[4,4] fusion(seed2, updm2, i0), kind=kLoop, calls=fused_dus
  dot3 = f32[4,4] dot(fusw, fusw), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = (f32[4,4], f32[4,4], f32[4,4], f32[4,4]) tuple(dot1, dot2, dot3, anchor)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  for (const char* name : {"dus1", "dus2", "dus3", "dus4", "anchor"}) {
    HloInstruction* inst = FindInstruction(module.get(), name);
    ASSERT_NE(inst, nullptr);
    EXPECT_FALSE(OutputsBF16(inst)) << name;
  }
  for (const char* name : {"fusw", "seed2", "updm2"}) {
    HloInstruction* inst = FindInstruction(module.get(), name);
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(OutputsBF16(inst)) << name;
  }
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// A mixed type bitcast (BF16 -> F32) in a while body must keep its
// precision unchanged while a same type bitcast's value goes BF16.
TEST_F(BFloat16PropagationTest, BitcastMixedTypesInWhileBody) {
  constexpr absl::string_view module_str = R"hlo(
HloModule BitcastMixedTypesInWhileBody

wcond {
  pc = (f32[4,4], bf16[4,4], s32[]) parameter(0)
  ic = s32[] get-tuple-element(pc), index=2
  lim = s32[] constant(8)
  ROOT lt = pred[] compare(ic, lim), direction=LT
}

wbody {
  pb = (f32[4,4], bf16[4,4], s32[]) parameter(0)
  xb = f32[4,4] get-tuple-element(pb), index=0
  yb = bf16[4,4] get-tuple-element(pb), index=1
  ib = s32[] get-tuple-element(pb), index=2
  one = s32[] constant(1)
  ni = s32[] add(ib, one)
  ycast = f32[4,4] bitcast(yb)
  nx = f32[4,4] maximum(xb, ycast)
  nxb = f32[4,4] bitcast(nx)
  prod = f32[4,4] dot(nxb, nxb), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tb = (f32[4,4], bf16[4,4], s32[]) tuple(prod, yb, ni)
}

ENTRY main {
  pa = f32[4,4] parameter(0)
  cb = bf16[4,4] convert(pa)
  zero = s32[] constant(0)
  init = (f32[4,4], bf16[4,4], s32[]) tuple(pa, cb, zero)
  w = (f32[4,4], bf16[4,4], s32[]) while(init), condition=wcond, body=wbody
  g0 = f32[4,4] get-tuple-element(w), index=0
  g1 = bf16[4,4] get-tuple-element(w), index=1
  dfin = f32[4,4] dot(g0, g0), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = (f32[4,4], bf16[4,4]) tuple(dfin, g1)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  HloInstruction* w = FindInstruction(module.get(), "w");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->shape().tuple_shapes(0).element_type(), BF16);
  EXPECT_EQ(w->shape().tuple_shapes(1).element_type(), BF16);
  // The mixed type bitcast keeps its F32 output while its surroundings
  // become BF16.
  HloInstruction* ycast = FindInstruction(module.get(), "ycast");
  ASSERT_NE(ycast, nullptr);
  EXPECT_FALSE(OutputsBF16(ycast));
  for (const char* name : {"nx", "nxb", "prod"}) {
    HloInstruction* inst = FindInstruction(module.get(), name);
    ASSERT_NE(inst, nullptr);
    EXPECT_TRUE(OutputsBF16(inst)) << name;
  }
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// Two kCall sites share a callee. One result feeds a dot, the other the F32
// entry root; the shared callee keeps both calls and their operands F32. A
// chain that avoids the callee becomes BF16.
TEST_F(BFloat16PropagationTest, SharedCallTwoSitesAsymmetricDemands) {
  constexpr absl::string_view module_str = R"hlo(
HloModule SharedCallTwoSitesAsymmetricDemands

shared_fn {
  sp = f32[4,4] parameter(0)
  sd = f32[4,4] dot(sp, sp), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT smax = f32[4,4] maximum(sd, sd)
}

ENTRY main {
  p0 = f32[4,4] parameter(0)
  p1 = f32[4,4] parameter(1)
  opA = f32[4,4] maximum(p0, p0)
  callA = f32[4,4] call(opA), to_apply=shared_fn
  dotA = f32[4,4] dot(callA, callA), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  dotOpA = f32[4,4] dot(opA, opA), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  callB = f32[4,4] call(p1), to_apply=shared_fn
  q = f32[4,4] maximum(p1, p1)
  dotQ = f32[4,4] dot(q, q), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = (f32[4,4], f32[4,4], f32[4,4], f32[4,4]) tuple(dotA, callB, dotQ, dotOpA)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  for (const char* name : {"callA", "callB", "opA", "smax", "sd"}) {
    HloInstruction* inst = FindInstruction(module.get(), name);
    ASSERT_NE(inst, nullptr);
    EXPECT_FALSE(OutputsBF16(inst)) << name;
  }
  HloInstruction* q = FindInstruction(module.get(), "q");
  ASSERT_NE(q, nullptr);
  EXPECT_TRUE(OutputsBF16(q));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// A while body swaps its two float slots through a nested tuple, so the
// entry root's F32 pin on one output slot must ripple through the
// forwarding chain and the loop backedge into the other slot. An
// independent chain still becomes BF16.
TEST_F(BFloat16PropagationTest, ForwardingChainsLatePinRipple) {
  constexpr absl::string_view module_str = R"hlo(
HloModule ForwardingChainsLatePinRipple

fcond {
  pc = (f32[4,4], f32[4,4], s32[]) parameter(0)
  ic = s32[] get-tuple-element(pc), index=2
  lim = s32[] constant(6)
  ROOT lt = pred[] compare(ic, lim), direction=LT
}

fbody {
  pb = (f32[4,4], f32[4,4], s32[]) parameter(0)
  x = f32[4,4] get-tuple-element(pb), index=0
  y = f32[4,4] get-tuple-element(pb), index=1
  i = s32[] get-tuple-element(pb), index=2
  one = s32[] constant(1)
  ni = s32[] add(i, one)
  t1 = (f32[4,4], f32[4,4]) tuple(y, x)
  xg = f32[4,4] get-tuple-element(t1), index=0
  yg = f32[4,4] get-tuple-element(t1), index=1
  nx = f32[4,4] dot(xg, xg), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT tb = (f32[4,4], f32[4,4], s32[]) tuple(nx, yg, ni)
}

ENTRY main {
  pa = f32[4,4] parameter(0)
  pb0 = f32[4,4] parameter(1)
  zero = s32[] constant(0)
  init = (f32[4,4], f32[4,4], s32[]) tuple(pa, pb0, zero)
  w = (f32[4,4], f32[4,4], s32[]) while(init), condition=fcond, body=fbody
  g0 = f32[4,4] get-tuple-element(w), index=0
  g1 = f32[4,4] get-tuple-element(w), index=1
  t2 = (f32[4,4], f32[4,4]) tuple(g0, g1)
  g2 = f32[4,4] get-tuple-element(t2), index=0
  g3 = f32[4,4] get-tuple-element(t2), index=1
  d = f32[4,4] dot(g2, g2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ex = f32[4,4] maximum(pa, pa)
  dex = f32[4,4] dot(ex, ex), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  ROOT out = (f32[4,4], f32[4,4], f32[4,4]) tuple(d, g3, dex)
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  EXPECT_TRUE(PropagatePrecision(module.get()));

  HloInstruction* w = FindInstruction(module.get(), "w");
  ASSERT_NE(w, nullptr);
  EXPECT_EQ(w->shape().tuple_shapes(0).element_type(), F32);
  EXPECT_EQ(w->shape().tuple_shapes(1).element_type(), F32);
  HloInstruction* body_root = w->while_body()->root_instruction();
  EXPECT_EQ(body_root->shape().tuple_shapes(0).element_type(), F32);
  EXPECT_EQ(body_root->shape().tuple_shapes(1).element_type(), F32);
  HloInstruction* ex = FindInstruction(module.get(), "ex");
  ASSERT_NE(ex, nullptr);
  EXPECT_TRUE(OutputsBF16(ex));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// A shared callee feeds a dot in the while body (BF16 friendly) and a two
// operand sort in the while condition. The sort propagates precision but its
// output at the read position is a tuple, which can never be BF16, so the
// callee root value must stay F32. The body is processed before the
// condition, so the callee root carries a stale BF16 mark when the graph is
// built; the sort use must seed the value F32 rather than add an edge on the
// tuple position, which can never fire.
TEST_F(BFloat16PropagationTest, SharedCalleeVariadicSortInWhileCondition) {
  constexpr absl::string_view module_str = R"hlo(
HloModule SharedCalleeVariadicSortInWhileCondition

shared {
  sp = f32[2,4] parameter(0)
  ROOT sout = f32[2,4] add(sp, sp)
}

comparator {
  ca = f32[] parameter(0)
  cb = f32[] parameter(1)
  cc = f32[] parameter(2)
  cd = f32[] parameter(3)
  ROOT ccmp = pred[] compare(ca, cb), direction=GT
}

body {
  bstate = (f32[2,4], f32[2,2]) parameter(0)
  bgte0 = f32[2,4] get-tuple-element(bstate), index=0
  callA = f32[2,4] call(bgte0), to_apply=shared
  bdot = f32[2,2] dot(callA, callA), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  ROOT btuple = (f32[2,4], f32[2,2]) tuple(bgte0, bdot)
}

cond {
  cstate = (f32[2,4], f32[2,2]) parameter(0)
  cgte0 = f32[2,4] get-tuple-element(cstate), index=0
  callB = f32[2,4] call(cgte0), to_apply=shared
  csort = (f32[2,4], f32[2,4]) sort(callB, callB), dimensions={1}, to_apply=comparator
  sgte = f32[2,4] get-tuple-element(csort), index=0
  cslice = f32[1,1] slice(sgte), slice={[0:1], [0:1]}
  creshape = f32[] reshape(cslice)
  czero = f32[] constant(0)
  ROOT cgreater = pred[] compare(creshape, czero), direction=GT
}

ENTRY main {
  p0 = f32[2,4] parameter(0)
  p1 = f32[2,2] parameter(1)
  init = (f32[2,4], f32[2,2]) tuple(p0, p1)
  w = (f32[2,4], f32[2,2]) while(init), condition=cond, body=body
  ROOT out = f32[2,2] get-tuple-element(w), index=1
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  PropagatePrecision(module.get());

  HloInstruction* callA = FindInstruction(module.get(), "callA");
  HloInstruction* callB = FindInstruction(module.get(), "callB");
  ASSERT_NE(callA, nullptr);
  ASSERT_NE(callB, nullptr);
  EXPECT_FALSE(OutputsBF16(callA));
  EXPECT_FALSE(OutputsBF16(callB));
  HloInstruction* shared_root =
      module->GetComputationWithName("shared")->root_instruction();
  EXPECT_FALSE(OutputsBF16(shared_root));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// Same staleness window as above, but the condition side user is a widening
// f32 to f64 convert. The convert propagates precision, but its output can
// never be BF16, so the callee root value must stay F32.
TEST_F(BFloat16PropagationTest, SharedCalleeWideningConvertInWhileCondition) {
  constexpr absl::string_view module_str = R"hlo(
HloModule SharedCalleeWideningConvertInWhileCondition

shared {
  sp = f32[2,4] parameter(0)
  ROOT sout = f32[2,4] add(sp, sp)
}

body {
  bstate = (f32[2,4], f32[2,2]) parameter(0)
  bgte0 = f32[2,4] get-tuple-element(bstate), index=0
  callA = f32[2,4] call(bgte0), to_apply=shared
  bdot = f32[2,2] dot(callA, callA), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  ROOT btuple = (f32[2,4], f32[2,2]) tuple(bgte0, bdot)
}

cond {
  cstate = (f32[2,4], f32[2,2]) parameter(0)
  cgte0 = f32[2,4] get-tuple-element(cstate), index=0
  callB = f32[2,4] call(cgte0), to_apply=shared
  cconv = f64[2,4] convert(callB)
  cslice = f64[1,1] slice(cconv), slice={[0:1], [0:1]}
  creshape = f64[] reshape(cslice)
  czero = f64[] constant(0)
  ROOT cgreater = pred[] compare(creshape, czero), direction=GT
}

ENTRY main {
  p0 = f32[2,4] parameter(0)
  p1 = f32[2,2] parameter(1)
  init = (f32[2,4], f32[2,2]) tuple(p0, p1)
  w = (f32[2,4], f32[2,2]) while(init), condition=cond, body=body
  ROOT out = f32[2,2] get-tuple-element(w), index=1
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  PropagatePrecision(module.get());

  HloInstruction* callA = FindInstruction(module.get(), "callA");
  HloInstruction* callB = FindInstruction(module.get(), "callB");
  ASSERT_NE(callA, nullptr);
  ASSERT_NE(callB, nullptr);
  EXPECT_FALSE(OutputsBF16(callA));
  EXPECT_FALSE(OutputsBF16(callB));
  HloInstruction* shared_root =
      module->GetComputationWithName("shared")->root_instruction();
  EXPECT_FALSE(OutputsBF16(shared_root));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// A DUS whose in place operand is a type punning bitcast of a u32 value. The
// u32 value can never be BF16, so it forces the DUS output to stay F32 even
// though the downstream dot would allow BF16 (an in place output must keep
// the precision of the buffer it updates).
TEST_F(BFloat16PropagationTest, InPlaceOperandAliasesNonFloatValue) {
  constexpr absl::string_view module_str = R"hlo(
HloModule InPlaceOperandAliasesNonFloatValue

ENTRY main {
  u = u32[2,4] parameter(0)
  upd = f32[2,1] parameter(1)
  pun = f32[2,4] bitcast(u)
  zero = s32[] constant(0)
  dus = f32[2,4] dynamic-update-slice(pun, upd, zero, zero)
  ROOT dot = f32[2,2] dot(dus, dus), lhs_contracting_dims={1}, rhs_contracting_dims={1}
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  PropagatePrecision(module.get());

  HloInstruction* dus = FindInstruction(module.get(), "dus");
  ASSERT_NE(dus, nullptr);
  EXPECT_FALSE(OutputsBF16(dus));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// Same staleness window, but the condition side user is a same type bitcast
// into host memory space. The bitcast forwards the callee root value, so its
// read position holds that value itself; the host guard keeps the bitcast
// unmarked, an unmarked self read can never turn BF16, and the old pass
// therefore pinned the value F32. The use must seed the value instead of
// adding an edge on the value's own position, which only the value itself
// could reach.
TEST_F(BFloat16PropagationTest, SameTypeHostBitcastReadInWhileCondition) {
  constexpr absl::string_view module_str = R"hlo(
HloModule SameTypeHostBitcastReadInWhileCondition

shared {
  sp = f32[2,4] parameter(0)
  ROOT sout = f32[2,4] add(sp, sp)
}

body {
  bstate = (f32[2,4], f32[2,2]) parameter(0)
  bgte0 = f32[2,4] get-tuple-element(bstate), index=0
  callA = f32[2,4] call(bgte0), to_apply=shared
  bdot = f32[2,2] dot(callA, callA), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  ROOT btuple = (f32[2,4], f32[2,2]) tuple(bgte0, bdot)
}

cond {
  cstate = (f32[2,4], f32[2,2]) parameter(0)
  cgte0 = f32[2,4] get-tuple-element(cstate), index=0
  callB = f32[2,4] call(cgte0), to_apply=shared
  cbc = f32[2,4]{1,0:S(5)} bitcast(callB)
  cdot = f32[2,2] dot(cbc, cbc), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  cslice = f32[1,1] slice(cdot), slice={[0:1], [0:1]}
  creshape = f32[] reshape(cslice)
  czero = f32[] constant(0)
  ROOT cgreater = pred[] compare(creshape, czero), direction=GT
}

ENTRY main {
  p0 = f32[2,4] parameter(0)
  p1 = f32[2,2] parameter(1)
  init = (f32[2,4], f32[2,2]) tuple(p0, p1)
  w = (f32[2,4], f32[2,2]) while(init), condition=cond, body=body
  ROOT out = f32[2,2] get-tuple-element(w), index=1
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  PropagatePrecision(module.get());

  HloInstruction* callA = FindInstruction(module.get(), "callA");
  HloInstruction* callB = FindInstruction(module.get(), "callB");
  HloInstruction* cbc = FindInstruction(module.get(), "cbc");
  ASSERT_NE(callA, nullptr);
  ASSERT_NE(callB, nullptr);
  ASSERT_NE(cbc, nullptr);
  EXPECT_FALSE(OutputsBF16(callA));
  EXPECT_FALSE(OutputsBF16(callB));
  EXPECT_FALSE(OutputsBF16(cbc));
  HloInstruction* shared_root =
      module->GetComputationWithName("shared")->root_instruction();
  EXPECT_FALSE(OutputsBF16(shared_root));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

// Same staleness window as the B1 regression tests above, but the condition
// side user is a plain add, which does not propagate operand precision to
// its output. Such a use fails the users rule statically and must seed the
// value F32.
TEST_F(BFloat16PropagationTest, SharedCalleeNonForwardingUserInWhileCondition) {
  constexpr absl::string_view module_str = R"hlo(
HloModule SharedCalleeNonForwardingUserInWhileCondition

shared {
  sp = f32[2,4] parameter(0)
  ROOT sout = f32[2,4] add(sp, sp)
}

body {
  bstate = (f32[2,4], f32[2,2]) parameter(0)
  bgte0 = f32[2,4] get-tuple-element(bstate), index=0
  callA = f32[2,4] call(bgte0), to_apply=shared
  bdot = f32[2,2] dot(callA, callA), lhs_contracting_dims={1}, rhs_contracting_dims={1}
  ROOT btuple = (f32[2,4], f32[2,2]) tuple(bgte0, bdot)
}

cond {
  cstate = (f32[2,4], f32[2,2]) parameter(0)
  cgte0 = f32[2,4] get-tuple-element(cstate), index=0
  callB = f32[2,4] call(cgte0), to_apply=shared
  cadd = f32[2,4] add(callB, callB)
  cslice = f32[1,1] slice(cadd), slice={[0:1], [0:1]}
  creshape = f32[] reshape(cslice)
  czero = f32[] constant(0)
  ROOT cgreater = pred[] compare(creshape, czero), direction=GT
}

ENTRY main {
  p0 = f32[2,4] parameter(0)
  p1 = f32[2,2] parameter(1)
  init = (f32[2,4], f32[2,2]) tuple(p0, p1)
  w = (f32[2,4], f32[2,2]) while(init), condition=cond, body=body
  ROOT out = f32[2,2] get-tuple-element(w), index=1
}
)hlo";
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<HloModule> module,
                       ParseAndReturnVerifiedModule(module_str));
  PropagatePrecision(module.get());

  HloInstruction* callA = FindInstruction(module.get(), "callA");
  HloInstruction* callB = FindInstruction(module.get(), "callB");
  ASSERT_NE(callA, nullptr);
  ASSERT_NE(callB, nullptr);
  EXPECT_FALSE(OutputsBF16(callA));
  EXPECT_FALSE(OutputsBF16(callB));
  EXPECT_OK(HloVerifier(/*layout_sensitive=*/false,
                        /*allow_mixed_precision=*/true)
                .Run(module.get())
                .status());
}

}  // namespace xla
