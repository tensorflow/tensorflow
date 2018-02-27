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
  EXPECT_TRUE(OutputsBF16(add0));
  EXPECT_TRUE(OutputsBF16(add1));
  EXPECT_TRUE(OutputsBF16(lhs));
  // rhs is a get-tuple-element, which does not define a buffer, but its shape
  // should also be adjusted accordingly.
  EXPECT_TRUE(OutputsBF16(rhs));
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
      tuple0->shape(), HloOpcode::kSelect, pred, tuple0, tuple1));
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

}  // namespace xla
