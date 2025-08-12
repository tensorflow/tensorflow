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
#include "xla/hlo/ir/collective_device_list.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
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
    BFloat16Propagation propagation(&bfloat16_support);
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
          /*device_list=*/CollectiveDeviceList(), /*constrain_layout=*/false,
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

// Tests that bf16 is not propagated through a domain in case its input cannot
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
  // This test is crafted so that the DUS has an f32 input (due to parameter)
  // and bf16 output (due to dot). But we should enforce DUS operand 0 and
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
  // bf16.
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

}  // namespace xla
