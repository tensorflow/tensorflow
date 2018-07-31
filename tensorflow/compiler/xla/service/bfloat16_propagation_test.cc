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

#include "tensorflow/compiler/xla/service/bfloat16_propagation.h"
#include "tensorflow/compiler/xla/service/bfloat16_support.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// A class specifying the BF16 support used to test the propagation pass. It
// specifies that BF16 and mixed precision are supported in all HloInstructions,
// and that kDot reduces its operands precision to BF16.
class TestBFloat16Support : public BFloat16Support {
 public:
  TestBFloat16Support() {}
  ~TestBFloat16Support() override {}

  bool SupportsBF16Operand(const HloInstruction& hlo,
                           int64 operand_index) const override {
    return true;
  }

  bool SupportsBF16Output(const HloInstruction& hlo) const override {
    return true;
  }

  bool SupportsMixedPrecisions(const HloInstruction& hlo) const override {
    return true;
  }

  bool EffectiveOperandPrecisionIsBF16(const HloInstruction& hlo,
                                       int64 operand_index) const override {
    return hlo.opcode() == HloOpcode::kDot;
  }
};

class BFloat16PropagationTest : public HloTestBase {
 protected:
  // Runs the propagation pass on the given module, and returns whether the
  // module is changed after this pass.
  bool PropagatePrecision(HloModule* module) {
    TestBFloat16Support bfloat16_support;
    BFloat16Propagation propagation(&bfloat16_support);
    StatusOr<bool> result = propagation.Run(module);
    EXPECT_IS_OK(result.status());
    return result.ValueOrDie();
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
  HloInstruction* pred = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kEq, a, b));
  HloInstruction* sel = builder.AddInstruction(
      HloInstruction::CreateTernary(shape, HloOpcode::kSelect, pred, c, add1));
  HloInstruction* xpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {4, 2}), sel, {1, 0}));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {4, 4}), HloOpcode::kDot, xpose, a));
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, dot, dot));

  auto module = CreateNewModule();
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
  HloInstruction* dot = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kDot, a, b));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_TRUE(OutputsBF16(dot->operand(0)));
  EXPECT_TRUE(OutputsBF16(dot->operand(1)));
  EXPECT_EQ(dot->operand(0)->opcode(), HloOpcode::kConstant);
  EXPECT_EQ(dot->operand(1)->opcode(), HloOpcode::kConstant);
  EXPECT_TRUE(LiteralTestUtil::Equal(
      *LiteralUtil::ConvertF32ToBF16(*LiteralUtil::CreateFromArray(array_a)),
      dot->operand(0)->literal()));
  EXPECT_TRUE(LiteralTestUtil::Equal(
      *LiteralUtil::ConvertF32ToBF16(*LiteralUtil::CreateFromArray(array_b)),
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
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {4, 4}), HloOpcode::kDot, lhs, rhs));

  HloInstruction* output_tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({dot, add2}));

  auto module = CreateNewModule();
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
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {4, 4}), HloOpcode::kDot, lhs, rhs));

  auto module = CreateNewModule();
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
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});

  HloInstruction* a =
      builder.AddInstruction(HloInstruction::CreateParameter(0, shape, "a"));
  HloInstruction* b =
      builder.AddInstruction(HloInstruction::CreateParameter(1, shape, "b"));
  HloInstruction* add = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, a, b));

  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {4, 4}), HloOpcode::kDot, add, add));

  HloInstruction* tuple =
      builder.AddInstruction(HloInstruction::CreateTuple({add, dot}));

  auto module = CreateNewModule();
  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_FALSE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), tuple);
  EXPECT_FALSE(OutputsBF16(add));
}

// Tests that BF16 is propagated properly through fused computations.
TEST_F(BFloat16PropagationTest, PropagateThroughFusion) {
  auto module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});

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
  HloInstruction* dot = builder_f1.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {4, 4}), HloOpcode::kDot, a_f1, b_f1));
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

// Tests that changes to BF16 that cannot be propagated outside a fusion are
// discarded.
TEST_F(BFloat16PropagationTest, DiscardFusionInternalBF16Changes) {
  auto module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});

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
  HloInstruction* dot_f = builder_f.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {4, 4}), HloOpcode::kDot, add_f, add_f));
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
  auto module = CreateNewModule();
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
  HloInstruction* dot = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kDot, gte0, gte1));

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

// A select over tuples does not define the leaf buffers, so the types in
// on_true and on_false must match, so that as long as one of them is F32, the
// other must be F32 as well.
TEST_F(BFloat16PropagationTest, SelectOverTuples) {
  auto module = CreateNewModule();
  auto builder = HloComputation::Builder(TestName());
  Shape shape = ShapeUtil::MakeShape(F32, {2, 4});

  HloInstruction* param = builder.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "param"));
  HloInstruction* pred = builder.AddInstruction(HloInstruction::CreateParameter(
      1, ShapeUtil::MakeShape(PRED, {}), "pred"));

  HloInstruction* add0 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, param, param));
  HloInstruction* add1 = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, add0, param));
  HloInstruction* tuple0 =
      builder.AddInstruction(HloInstruction::CreateTuple({param, add0}));
  HloInstruction* tuple1 =
      builder.AddInstruction(HloInstruction::CreateTuple({param, add1}));
  HloInstruction* sel = builder.AddInstruction(HloInstruction::CreateTernary(
      tuple0->shape(), HloOpcode::kTupleSelect, pred, tuple0, tuple1));
  HloInstruction* gte0 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, sel, 0));
  HloInstruction* gte1 = builder.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, sel, 1));
  HloInstruction* xpose =
      builder.AddInstruction(HloInstruction::CreateTranspose(
          ShapeUtil::MakeShape(F32, {4, 2}), gte0, {1, 0}));
  HloInstruction* dot = builder.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(F32, {4, 4}), HloOpcode::kDot, xpose, gte1));

  auto computation = module->AddEntryComputation(builder.Build());

  EXPECT_TRUE(PropagatePrecision(module.get()));

  EXPECT_EQ(computation->root_instruction(), dot);
  EXPECT_FALSE(OutputsBF16(add0));
  EXPECT_FALSE(OutputsBF16(add1));
  EXPECT_FALSE(OutputsBF16(gte0));
  EXPECT_FALSE(OutputsBF16(gte1));
  EXPECT_TRUE(OutputsBF16(xpose));
}

// Tests that BF16 is propagated properly through a while computation with
// non-tuple input/output.
TEST_F(BFloat16PropagationTest, PropagateThroughSimpleWhile) {
  auto module = CreateNewModule();
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
  auto cond_dot = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kDot, cond_param, cond_param));
  auto cond_root = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt,
      builder_cond.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond_dot, {0, 0}, {1, 1}, {1, 1})),
      builder_cond.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond_dot, {1, 1}, {2, 2}, {1, 1}))));
  auto cond = module->AddEmbeddedComputation(builder_cond.Build());

  auto builder_body = HloComputation::Builder("body");
  auto body_param = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, shape, "body_param"));
  auto body_dot = builder_body.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kDot, body_param, body_param));
  auto body = module->AddEmbeddedComputation(builder_body.Build());

  auto while_hlo = builder.AddInstruction(
      HloInstruction::CreateWhile(shape, cond, body, add));

  auto dot = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kDot, while_hlo, while_hlo));
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
  auto module = CreateNewModule();
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
  builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt,
      builder_cond.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond_param, {0, 0}, {1, 1}, {1, 1})),
      builder_cond.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond_param, {1, 1}, {2, 2}, {1, 1}))));
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

  auto dot = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kDot, while_hlo, while_hlo));
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
  auto module = CreateNewModule();
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
  auto cond_dot = builder_cond.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kDot, cond_lhs, cond_add_rhs));
  builder_cond.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt,
      builder_cond.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond_dot, {0, 0}, {1, 1}, {1, 1})),
      builder_cond.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond_dot, {1, 1}, {2, 2}, {1, 1}))));
  auto cond = module->AddEmbeddedComputation(builder_cond.Build());

  auto builder_body = HloComputation::Builder("body");
  auto body_param = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, tuple->shape(), "body_param"));
  auto body_lhs = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  auto body_rhs = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 1));
  auto body_dot1 = builder_body.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kDot, body_lhs, body_rhs));
  auto body_dot2 = builder_body.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kDot, body_rhs, body_lhs));
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
  auto dot = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kDot, lhs, rhs));
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
  auto module = CreateNewModule();
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
  auto cond0_dot = builder_cond0.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kDot, cond0_lhs, cond0_add_rhs));
  builder_cond0.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt,
      builder_cond0.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond0_dot, {0, 0}, {1, 1}, {1, 1})),
      builder_cond0.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond0_dot, {1, 1}, {2, 2}, {1, 1}))));
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
  auto cond1_dot = builder_cond1.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kDot, cond1_add_lhs, cond1_rhs));
  builder_cond1.AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt,
      builder_cond1.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond1_dot, {0, 0}, {1, 1}, {1, 1})),
      builder_cond1.AddInstruction(HloInstruction::CreateSlice(
          ShapeUtil::MakeShape(F32, {}), cond1_dot, {1, 1}, {2, 2}, {1, 1}))));
  auto cond1 = module->AddEmbeddedComputation(builder_cond1.Build());

  // Body computation shared by both whiles.
  auto builder_body = HloComputation::Builder("body");
  auto body_param = builder_body.AddInstruction(
      HloInstruction::CreateParameter(0, tuple0->shape(), "body_param"));
  auto body_lhs = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 0));
  auto body_rhs = builder_body.AddInstruction(
      HloInstruction::CreateGetTupleElement(shape, body_param, 1));
  auto body_dot = builder_body.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kDot, body_lhs, body_rhs));
  builder_body.AddInstruction(
      HloInstruction::CreateTuple({body_dot, body_rhs}));
  auto body = module->AddEmbeddedComputation(builder_body.Build());

  auto while0 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple0->shape(), cond0, body, tuple0));
  auto while1 = builder.AddInstruction(
      HloInstruction::CreateWhile(tuple1->shape(), cond1, body, tuple1));

  auto lhs = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kDot,
      builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(shape, while0, 0)),
      builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(shape, while0, 1))));
  auto rhs = builder.AddInstruction(HloInstruction::CreateBinary(
      shape, HloOpcode::kDot,
      builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(shape, while1, 0)),
      builder.AddInstruction(
          HloInstruction::CreateGetTupleElement(shape, while1, 1))));
  auto dot = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kDot, lhs, rhs));
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

  auto module = CreateNewModule();
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
  HloInstruction* dot = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kDot, a_gte, b_gte));
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, dot, dot));

  auto module = CreateNewModule();
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
  HloInstruction* dot = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kDot, a_trans, b_trans));
  HloInstruction* root = builder.AddInstruction(
      HloInstruction::CreateBinary(shape, HloOpcode::kAdd, dot, dot));

  auto module = CreateNewModule();
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

}  // namespace xla
