/* Copyright 2018 Graphcore Ltd

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

#include "tensorflow/compiler/plugin/poplar/driver/hlo_hash.h"

#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/tests/hlo_test_base.h"
#include "tensorflow/core/lib/core/status_test_util.h"

namespace xla {
namespace poplarplugin {
namespace {

using HloHashTest = HloTestBase;

void CreateWhileLoop(HloModule* module, std::string test_name) {
  Shape scalar_shape = ShapeUtil::MakeShape(F32, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({scalar_shape, scalar_shape});
  /* Create while condition */
  HloComputation* comp_cond;
  {
    auto builder_cond = HloComputation::Builder(test_name);
    auto tuple = builder_cond.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "cond_tuple"));
    auto limit0 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(10)));
    auto limit1 = builder_cond.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(12)));
    auto c0 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(F32, {}), tuple, 0));
    auto c1 = builder_cond.AddInstruction(HloInstruction::CreateGetTupleElement(
        ShapeUtil::MakeShape(F32, {}), tuple, 1));
    auto lt0 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kLt, c0, limit0));
    auto lt1 = builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kGt, c1, limit1));
    builder_cond.AddInstruction(HloInstruction::CreateBinary(
        ShapeUtil::MakeShape(PRED, {}), HloOpcode::kAnd, lt0, lt1));

    comp_cond = module->AddEmbeddedComputation(builder_cond.Build());
  }

  /* Create while body */
  HloComputation* comp_body;
  {
    auto builder_body = HloComputation::Builder(test_name);
    auto tuple = builder_body.AddInstruction(
        HloInstruction::CreateParameter(0, tuple_shape, "body_tuple"));
    auto c0 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 0));
    auto c1 = builder_body.AddInstruction(
        HloInstruction::CreateGetTupleElement(scalar_shape, tuple, 1));
    auto one = builder_body.AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(1)));
    auto new_c0 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c0->shape(), HloOpcode::kAdd, c0, one));
    auto new_c1 = builder_body.AddInstruction(
        HloInstruction::CreateBinary(c1->shape(), HloOpcode::kAdd, c1, one));

    builder_body.AddInstruction(HloInstruction::CreateTuple({new_c0, new_c1}));

    comp_body = module->AddEmbeddedComputation(builder_body.Build());
  }

  /* Create main computation */
  auto builder_main = HloComputation::Builder(test_name);
  auto c0 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));
  auto c1 = builder_main.AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<float>(0)));

  auto init =
      builder_main.AddInstruction(HloInstruction::CreateTuple({c0, c1}));

  auto main = builder_main.AddInstruction(
      HloInstruction::CreateWhile(tuple_shape, comp_cond, comp_body, init));

  builder_main.AddInstruction(HloInstruction::CreateTuple({main}));

  module->AddEntryComputation(builder_main.Build());
}

TEST_F(HloHashTest, SameWhileLoopsSameHash) {
  auto module0 = CreateNewModule();
  CreateWhileLoop(module0.get(), TestName());
  auto module1 = CreateNewModule();
  CreateWhileLoop(module1.get(), TestName());
  HloHash hash0(module0.get());
  HloHash hash1(module1.get());
  EXPECT_EQ(hash0.GetHash(), hash1.GetHash());
  EXPECT_EQ(hash0.GetProtoStr(), hash1.GetProtoStr());
}

TEST_F(HloHashTest, DifferentNamesSameHash) {
  std::string hlo_string0 = R"(
HloModule top

%max7 {
  %x.7.0 = f32[] parameter(0)
  %y.7.1 = f32[] parameter(1)
  ROOT %maximum.7.2 = f32[] maximum(f32[] %x.7.0, f32[] %y.7.1)
}

%add8 {
  %x.8.0 = f32[] parameter(0)
  %y.8.1 = f32[] parameter(1)
  ROOT %add.8.2 = f32[] add(f32[] %x.8.0, f32[] %y.8.1)
}

%_pop_op_relu {
  %constant.9.10.clone = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.9.11.clone = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %arg_0 = f32[1,12]{1,0} parameter(0)
  ROOT %maximum.9.12.clone = f32[1,12]{1,0} maximum(f32[1,12]{1,0} %broadcast.9.11.clone, f32[1,12]{1,0} %arg_0), metadata={op_type="Relu" op_name="Relu"}
}

%_pop_op_relugrad {
  %arg_0.1 = f32[1,12]{1,0} parameter(0)
  %constant.9.10.clone.1 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.9.11.clone.1 = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone.1), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %greater-than.9.44.clone = pred[1,12]{1,0} greater-than(f32[1,12]{1,0} %arg_0.1, f32[1,12]{1,0} %broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %arg_1 = f32[1,12]{1,0} parameter(1)
  ROOT %select.9.45.clone = f32[1,12]{1,0} select(pred[1,12]{1,0} %greater-than.9.44.clone, f32[1,12]{1,0} %arg_1, f32[1,12]{1,0} %broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
}

%_pop_op_wide_const {
  %constant.9.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %broadcast.9.39.clone = f32[12,12]{1,0} broadcast(f32[] %constant.9.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
}

%_pop_op_wide_const.1 {
  %constant.9.6.clone.1 = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %broadcast.9.48.clone = f32[4,12]{1,0} broadcast(f32[] %constant.9.6.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
}

ENTRY %cluster_1 {
  %arg2.9.2 = f32[12,12]{1,0} parameter(2), metadata={op_name="XLA_Args"}
  %call.2 = f32[12,12]{1,0} call(), to_apply=%_pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %arg0.9.0 = f32[1,4]{1,0} parameter(0), metadata={op_name="XLA_Args"}
  %arg3.9.3 = f32[4,12]{1,0} parameter(3), metadata={op_name="XLA_Args"}
  %dot.9.9 = f32[1,12]{1,0} dot(f32[1,4]{1,0} %arg0.9.0, f32[4,12]{1,0} %arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  %call = f32[1,12]{1,0} call(f32[1,12]{1,0} %dot.9.9), to_apply=%_pop_op_relu, metadata={op_type="Relu" op_name="Relu"}
  %transpose.9.35 = f32[12,1]{0,1} transpose(f32[1,12]{1,0} %call), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %dot.9.13 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %call, f32[12,12]{1,0} %arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  %constant.9.14 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %reduce.9.15 = f32[1]{0} reduce(f32[1,12]{1,0} %dot.9.13, f32[] %constant.9.14), dimensions={1}, to_apply=%max7, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.9.16 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.15), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %subtract.9.17 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %dot.9.13, f32[1,12]{1,0} %broadcast.9.16), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %exponential.9.18 = f32[1,12]{1,0} exponential(f32[1,12]{1,0} %subtract.9.17), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %constant.9.10 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %reduce.9.21 = f32[1]{0} reduce(f32[1,12]{1,0} %exponential.9.18, f32[] %constant.9.10), dimensions={1}, to_apply=%add8, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.9.32 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.21), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %divide.9.33 = f32[1,12]{1,0} divide(f32[1,12]{1,0} %exponential.9.18, f32[1,12]{1,0} %broadcast.9.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %arg1.9.1 = f32[1,12]{1,0} parameter(1), metadata={op_name="XLA_Args"}
  %subtract.9.34 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %divide.9.33, f32[1,12]{1,0} %arg1.9.1), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %dot.9.36 = f32[12,12]{1,0} dot(f32[12,1]{0,1} %transpose.9.35, f32[1,12]{1,0} %subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %multiply.9.40 = f32[12,12]{1,0} multiply(f32[12,12]{1,0} %call.2, f32[12,12]{1,0} %dot.9.36), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %subtract.9.41 = f32[12,12]{1,0} subtract(f32[12,12]{1,0} %arg2.9.2, f32[12,12]{1,0} %multiply.9.40), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %call.3 = f32[4,12]{1,0} call(), to_apply=%_pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %transpose.9.46 = f32[4,1]{0,1} transpose(f32[1,4]{1,0} %arg0.9.0), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %transpose.9.37 = f32[12,12]{0,1} transpose(f32[12,12]{1,0} %arg2.9.2), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %dot.9.38 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %subtract.9.34, f32[12,12]{0,1} %transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %call.1 = f32[1,12]{1,0} call(f32[1,12]{1,0} %call, f32[1,12]{1,0} %dot.9.38), to_apply=%_pop_op_relugrad, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %dot.9.47 = f32[4,12]{1,0} dot(f32[4,1]{0,1} %transpose.9.46, f32[1,12]{1,0} %call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %multiply.9.49 = f32[4,12]{1,0} multiply(f32[4,12]{1,0} %call.3, f32[4,12]{1,0} %dot.9.47), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  %subtract.9.50 = f32[4,12]{1,0} subtract(f32[4,12]{1,0} %arg3.9.3, f32[4,12]{1,0} %multiply.9.49), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  ROOT %tuple.9.55 = (f32[12,12]{1,0}, f32[4,12]{1,0}) tuple(f32[12,12]{1,0} %subtract.9.41, f32[4,12]{1,0} %subtract.9.50), metadata={op_name="XLA_Retvals"}
}

)";

  std::string hlo_string1 = R"(
HloModule top

%__max7 {
  %__x.7.0 = f32[] parameter(0)
  %__y.7.1 = f32[] parameter(1)
  ROOT %__maximum.7.2 = f32[] maximum(f32[] %__x.7.0, f32[] %__y.7.1)
}

%__add8 {
  %__x.8.0 = f32[] parameter(0)
  %__y.8.1 = f32[] parameter(1)
  ROOT %__add.8.2 = f32[] add(f32[] %__x.8.0, f32[] %__y.8.1)
}

%___pop_op_relu {
  %__constant.9.10.clone = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %__broadcast.9.11.clone = f32[1,12]{1,0} broadcast(f32[] %__constant.9.10.clone), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %__arg_0 = f32[1,12]{1,0} parameter(0)
  ROOT %__maximum.9.12.clone = f32[1,12]{1,0} maximum(f32[1,12]{1,0} %__broadcast.9.11.clone, f32[1,12]{1,0} %__arg_0), metadata={op_type="Relu" op_name="Relu"}
}

%___pop_op_relugrad {
  %__arg_0.1 = f32[1,12]{1,0} parameter(0)
  %__constant.9.10.clone.1 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %__broadcast.9.11.clone.1 = f32[1,12]{1,0} broadcast(f32[] %__constant.9.10.clone.1), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %__greater-than.9.44.clone = pred[1,12]{1,0} greater-than(f32[1,12]{1,0} %__arg_0.1, f32[1,12]{1,0} %__broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %__arg_1 = f32[1,12]{1,0} parameter(1)
  ROOT %__select.9.45.clone = f32[1,12]{1,0} select(pred[1,12]{1,0} %__greater-than.9.44.clone, f32[1,12]{1,0} %__arg_1, f32[1,12]{1,0} %__broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
}

%___pop_op_wide_const {
  %__constant.9.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %__broadcast.9.39.clone = f32[12,12]{1,0} broadcast(f32[] %__constant.9.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
}

%___pop_op_wide_const.1 {
  %__constant.9.6.clone.1 = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %__broadcast.9.48.clone = f32[4,12]{1,0} broadcast(f32[] %__constant.9.6.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
}

ENTRY %__cluster_1 {
  %__arg2.9.2 = f32[12,12]{1,0} parameter(2), metadata={op_name="XLA_Args"}
  %__call.2 = f32[12,12]{1,0} call(), to_apply=%___pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %__arg0.9.0 = f32[1,4]{1,0} parameter(0), metadata={op_name="XLA_Args"}
  %__arg3.9.3 = f32[4,12]{1,0} parameter(3), metadata={op_name="XLA_Args"}
  %__dot.9.9 = f32[1,12]{1,0} dot(f32[1,4]{1,0} %__arg0.9.0, f32[4,12]{1,0} %__arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  %__call = f32[1,12]{1,0} call(f32[1,12]{1,0} %__dot.9.9), to_apply=%___pop_op_relu, metadata={op_type="Relu" op_name="Relu"}
  %__transpose.9.35 = f32[12,1]{0,1} transpose(f32[1,12]{1,0} %__call), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %__dot.9.13 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %__call, f32[12,12]{1,0} %__arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  %__constant.9.14 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %__reduce.9.15 = f32[1]{0} reduce(f32[1,12]{1,0} %__dot.9.13, f32[] %__constant.9.14), dimensions={1}, to_apply=%__max7, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %__broadcast.9.16 = f32[1,12]{1,0} broadcast(f32[1]{0} %__reduce.9.15), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %__subtract.9.17 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %__dot.9.13, f32[1,12]{1,0} %__broadcast.9.16), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %__exponential.9.18 = f32[1,12]{1,0} exponential(f32[1,12]{1,0} %__subtract.9.17), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %__constant.9.10 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %__reduce.9.21 = f32[1]{0} reduce(f32[1,12]{1,0} %__exponential.9.18, f32[] %__constant.9.10), dimensions={1}, to_apply=%__add8, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %__broadcast.9.32 = f32[1,12]{1,0} broadcast(f32[1]{0} %__reduce.9.21), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %__divide.9.33 = f32[1,12]{1,0} divide(f32[1,12]{1,0} %__exponential.9.18, f32[1,12]{1,0} %__broadcast.9.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %__arg1.9.1 = f32[1,12]{1,0} parameter(1), metadata={op_name="XLA_Args"}
  %__subtract.9.34 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %__divide.9.33, f32[1,12]{1,0} %__arg1.9.1), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %__dot.9.36 = f32[12,12]{1,0} dot(f32[12,1]{0,1} %__transpose.9.35, f32[1,12]{1,0} %__subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %__multiply.9.40 = f32[12,12]{1,0} multiply(f32[12,12]{1,0} %__call.2, f32[12,12]{1,0} %__dot.9.36), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %__subtract.9.41 = f32[12,12]{1,0} subtract(f32[12,12]{1,0} %__arg2.9.2, f32[12,12]{1,0} %__multiply.9.40), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %__call.3 = f32[4,12]{1,0} call(), to_apply=%___pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %__transpose.9.46 = f32[4,1]{0,1} transpose(f32[1,4]{1,0} %__arg0.9.0), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %__transpose.9.37 = f32[12,12]{0,1} transpose(f32[12,12]{1,0} %__arg2.9.2), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %__dot.9.38 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %__subtract.9.34, f32[12,12]{0,1} %__transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %__call.1 = f32[1,12]{1,0} call(f32[1,12]{1,0} %__call, f32[1,12]{1,0} %__dot.9.38), to_apply=%___pop_op_relugrad, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %__dot.9.47 = f32[4,12]{1,0} dot(f32[4,1]{0,1} %__transpose.9.46, f32[1,12]{1,0} %__call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %__multiply.9.49 = f32[4,12]{1,0} multiply(f32[4,12]{1,0} %__call.3, f32[4,12]{1,0} %__dot.9.47), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  %__subtract.9.50 = f32[4,12]{1,0} subtract(f32[4,12]{1,0} %__arg3.9.3, f32[4,12]{1,0} %__multiply.9.49), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  ROOT %__tuple.9.55 = (f32[12,12]{1,0}, f32[4,12]{1,0}) tuple(f32[12,12]{1,0} %__subtract.9.41, f32[4,12]{1,0} %__subtract.9.50), metadata={op_name="XLA_Retvals"}
}

)";
  auto module0_or_status =
      HloRunner::CreateModuleFromString(hlo_string0, GetDebugOptionsForTest());
  EXPECT_TRUE(module0_or_status.ok());
  auto* module0 = module0_or_status.ValueOrDie().get();

  auto module1_or_status =
      HloRunner::CreateModuleFromString(hlo_string1, GetDebugOptionsForTest());
  EXPECT_TRUE(module1_or_status.ok());
  auto* module1 = module1_or_status.ValueOrDie().get();

  HloHash hash0(module0);
  HloHash hash1(module1);
  EXPECT_EQ(hash0.GetHash(), hash1.GetHash());
  EXPECT_EQ(hash0.GetProtoStr(), hash1.GetProtoStr());
}

TEST_F(HloHashTest, DifferentProgramsDifferentHash) {
  std::string hlo_string0 = R"(
HloModule top

%max7 {
  %x.7.0 = f32[] parameter(0)
  %y.7.1 = f32[] parameter(1)
  ROOT %maximum.7.2 = f32[] maximum(f32[] %x.7.0, f32[] %y.7.1)
}

%add8 {
  %x.8.0 = f32[] parameter(0)
  %y.8.1 = f32[] parameter(1)
  ROOT %add.8.2 = f32[] add(f32[] %x.8.0, f32[] %y.8.1)
}

%_pop_op_relu {
  %constant.9.10.clone = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.9.11.clone = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %arg_0 = f32[1,12]{1,0} parameter(0)
  ROOT %maximum.9.12.clone = f32[1,12]{1,0} maximum(f32[1,12]{1,0} %broadcast.9.11.clone, f32[1,12]{1,0} %arg_0), metadata={op_type="Relu" op_name="Relu"}
}

%_pop_op_relugrad {
  %arg_0.1 = f32[1,12]{1,0} parameter(0)
  %constant.9.10.clone.1 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.9.11.clone.1 = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone.1), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %greater-than.9.44.clone = pred[1,12]{1,0} greater-than(f32[1,12]{1,0} %arg_0.1, f32[1,12]{1,0} %broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %arg_1 = f32[1,12]{1,0} parameter(1)
  ROOT %select.9.45.clone = f32[1,12]{1,0} select(pred[1,12]{1,0} %greater-than.9.44.clone, f32[1,12]{1,0} %arg_1, f32[1,12]{1,0} %broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
}

%_pop_op_wide_const {
  %constant.9.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %broadcast.9.39.clone = f32[12,12]{1,0} broadcast(f32[] %constant.9.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
}

%_pop_op_wide_const.1 {
  %constant.9.6.clone.1 = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %broadcast.9.48.clone = f32[4,12]{1,0} broadcast(f32[] %constant.9.6.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
}

ENTRY %cluster_1 {
  %arg2.9.2 = f32[12,12]{1,0} parameter(2), metadata={op_name="XLA_Args"}
  %call.2 = f32[12,12]{1,0} call(), to_apply=%_pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %arg0.9.0 = f32[1,4]{1,0} parameter(0), metadata={op_name="XLA_Args"}
  %arg3.9.3 = f32[4,12]{1,0} parameter(3), metadata={op_name="XLA_Args"}
  %dot.9.9 = f32[1,12]{1,0} dot(f32[1,4]{1,0} %arg0.9.0, f32[4,12]{1,0} %arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  %call = f32[1,12]{1,0} call(f32[1,12]{1,0} %dot.9.9), to_apply=%_pop_op_relu, metadata={op_type="Relu" op_name="Relu"}
  %transpose.9.35 = f32[12,1]{0,1} transpose(f32[1,12]{1,0} %call), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %dot.9.13 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %call, f32[12,12]{1,0} %arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  %constant.9.14 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %reduce.9.15 = f32[1]{0} reduce(f32[1,12]{1,0} %dot.9.13, f32[] %constant.9.14), dimensions={1}, to_apply=%max7, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.9.16 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.15), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %subtract.9.17 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %dot.9.13, f32[1,12]{1,0} %broadcast.9.16), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %exponential.9.18 = f32[1,12]{1,0} exponential(f32[1,12]{1,0} %subtract.9.17), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %constant.9.10 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %reduce.9.21 = f32[1]{0} reduce(f32[1,12]{1,0} %exponential.9.18, f32[] %constant.9.10), dimensions={1}, to_apply=%add8, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.9.32 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.21), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %divide.9.33 = f32[1,12]{1,0} divide(f32[1,12]{1,0} %exponential.9.18, f32[1,12]{1,0} %broadcast.9.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %arg1.9.1 = f32[1,12]{1,0} parameter(1), metadata={op_name="XLA_Args"}
  %subtract.9.34 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %divide.9.33, f32[1,12]{1,0} %arg1.9.1), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %dot.9.36 = f32[12,12]{1,0} dot(f32[12,1]{0,1} %transpose.9.35, f32[1,12]{1,0} %subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %multiply.9.40 = f32[12,12]{1,0} multiply(f32[12,12]{1,0} %call.2, f32[12,12]{1,0} %dot.9.36), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %subtract.9.41 = f32[12,12]{1,0} subtract(f32[12,12]{1,0} %arg2.9.2, f32[12,12]{1,0} %multiply.9.40), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %call.3 = f32[4,12]{1,0} call(), to_apply=%_pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %transpose.9.46 = f32[4,1]{0,1} transpose(f32[1,4]{1,0} %arg0.9.0), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %transpose.9.37 = f32[12,12]{0,1} transpose(f32[12,12]{1,0} %arg2.9.2), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %dot.9.38 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %subtract.9.34, f32[12,12]{0,1} %transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %call.1 = f32[1,12]{1,0} call(f32[1,12]{1,0} %call, f32[1,12]{1,0} %dot.9.38), to_apply=%_pop_op_relugrad, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %dot.9.47 = f32[4,12]{1,0} dot(f32[4,1]{0,1} %transpose.9.46, f32[1,12]{1,0} %call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %multiply.9.49 = f32[4,12]{1,0} multiply(f32[4,12]{1,0} %call.3, f32[4,12]{1,0} %dot.9.47), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  %subtract.9.50 = f32[4,12]{1,0} subtract(f32[4,12]{1,0} %arg3.9.3, f32[4,12]{1,0} %multiply.9.49), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  ROOT %tuple.9.55 = (f32[12,12]{1,0}, f32[4,12]{1,0}) tuple(f32[12,12]{1,0} %subtract.9.41, f32[4,12]{1,0} %subtract.9.50), metadata={op_name="XLA_Retvals"}
}

)";

  std::string hlo_string1 = R"(
HloModule top

%_pop_op_relu {
  %constant.17.9.clone = f32[] constant(0), metadata={op_type="Relu" op_name="dense/Relu"}
  %broadcast.17.10.clone = f32[32,32]{1,0} broadcast(f32[] %constant.17.9.clone), dimensions={}, metadata={op_type="Relu" op_name="dense/Relu"}
  %arg_0 = f32[32,32]{1,0} parameter(0)
  ROOT %maximum.17.11.clone = f32[32,32]{1,0} maximum(f32[32,32]{1,0} %broadcast.17.10.clone, f32[32,32]{1,0} %arg_0), metadata={op_type="Relu" op_name="dense/Relu"}
}

%_pop_op_sigmoid {
  %constant.17.15.clone = f32[] constant(0.5), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  %broadcast.17.21.clone = f32[32,1]{1,0} broadcast(f32[] %constant.17.15.clone), dimensions={}, metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  %arg_0.1 = f32[32,1]{1,0} parameter(0)
  %multiply.17.17.clone = f32[32,1]{1,0} multiply(f32[32,1]{1,0} %broadcast.17.21.clone, f32[32,1]{1,0} %arg_0.1), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  %tanh.17.18.clone = f32[32,1]{1,0} tanh(f32[32,1]{1,0} %multiply.17.17.clone), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  %multiply.17.20.clone = f32[32,1]{1,0} multiply(f32[32,1]{1,0} %broadcast.17.21.clone, f32[32,1]{1,0} %tanh.17.18.clone), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  ROOT %add.17.22.clone = f32[32,1]{1,0} add(f32[32,1]{1,0} %broadcast.17.21.clone, f32[32,1]{1,0} %multiply.17.20.clone), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
}

ENTRY %cluster_9 {
  %arg0.17.0 = f32[32,100]{1,0} parameter(0), metadata={op_name="XLA_Args"}
  %arg4.17.4 = f32[100,32]{1,0} parameter(4), metadata={op_name="XLA_Args"}
  %dot.17.6 = f32[32,32]{1,0} dot(f32[32,100]{1,0} %arg0.17.0, f32[100,32]{1,0} %arg4.17.4), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="dense/MatMul"}
  %arg3.17.3 = f32[32]{0} parameter(3), metadata={op_name="XLA_Args"}
  %broadcast.17.7 = f32[32,32]{1,0} broadcast(f32[32]{0} %arg3.17.3), dimensions={1}, metadata={op_type="BiasAdd" op_name="dense/BiasAdd"}
  %add.17.8 = f32[32,32]{1,0} add(f32[32,32]{1,0} %dot.17.6, f32[32,32]{1,0} %broadcast.17.7), metadata={op_type="BiasAdd" op_name="dense/BiasAdd"}
  %call = f32[32,32]{1,0} call(f32[32,32]{1,0} %add.17.8), to_apply=%_pop_op_relu, metadata={op_type="Relu" op_name="dense/Relu"}
  %arg2.17.2 = f32[32,1]{1,0} parameter(2), metadata={op_name="XLA_Args"}
  %dot.17.12 = f32[32,1]{1,0} dot(f32[32,32]{1,0} %call, f32[32,1]{1,0} %arg2.17.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="dense_1/MatMul"}
  %arg1.17.1 = f32[1]{0} parameter(1), metadata={op_name="XLA_Args"}
  %broadcast.17.13 = f32[32,1]{1,0} broadcast(f32[1]{0} %arg1.17.1), dimensions={1}, metadata={op_type="BiasAdd" op_name="dense_1/BiasAdd"}
  %add.17.14 = f32[32,1]{1,0} add(f32[32,1]{1,0} %dot.17.12, f32[32,1]{1,0} %broadcast.17.13), metadata={op_type="BiasAdd" op_name="dense_1/BiasAdd"}
  %call.1 = f32[32,1]{1,0} call(f32[32,1]{1,0} %add.17.14), to_apply=%_pop_op_sigmoid, metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  ROOT %tuple.17.24 = (f32[32,1]{1,0}) tuple(f32[32,1]{1,0} %call.1), metadata={op_name="XLA_Retvals"}
}
  )";
  auto module0_or_status =
      HloRunner::CreateModuleFromString(hlo_string0, GetDebugOptionsForTest());
  EXPECT_TRUE(module0_or_status.ok());
  auto* module0 = module0_or_status.ValueOrDie().get();

  auto module1_or_status =
      HloRunner::CreateModuleFromString(hlo_string1, GetDebugOptionsForTest());
  EXPECT_TRUE(module1_or_status.ok());
  auto* module1 = module1_or_status.ValueOrDie().get();

  HloHash hash0(module0);
  HloHash hash1(module1);
  EXPECT_NE(hash0.GetHash(), hash1.GetHash());
  EXPECT_NE(hash0.GetProtoStr(), hash1.GetProtoStr());
}

TEST_F(HloHashTest, DifferentMetadataSameHash) {
  std::string hlo_string0 = R"(
HloModule top

%max7 {
  %x.7.0 = f32[] parameter(0)
  %y.7.1 = f32[] parameter(1)
  ROOT %maximum.7.2 = f32[] maximum(f32[] %x.7.0, f32[] %y.7.1)
}

%add8 {
  %x.8.0 = f32[] parameter(0)
  %y.8.1 = f32[] parameter(1)
  ROOT %add.8.2 = f32[] add(f32[] %x.8.0, f32[] %y.8.1)
}

%_pop_op_relu {
  %constant.9.10.clone = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.9.11.clone = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %arg_0 = f32[1,12]{1,0} parameter(0)
  ROOT %maximum.9.12.clone = f32[1,12]{1,0} maximum(f32[1,12]{1,0} %broadcast.9.11.clone, f32[1,12]{1,0} %arg_0), metadata={op_type="Relu" op_name="Relu"}
}

%_pop_op_relugrad {
  %arg_0.1 = f32[1,12]{1,0} parameter(0)
  %constant.9.10.clone.1 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %broadcast.9.11.clone.1 = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone.1), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  %greater-than.9.44.clone = pred[1,12]{1,0} greater-than(f32[1,12]{1,0} %arg_0.1, f32[1,12]{1,0} %broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %arg_1 = f32[1,12]{1,0} parameter(1)
  ROOT %select.9.45.clone = f32[1,12]{1,0} select(pred[1,12]{1,0} %greater-than.9.44.clone, f32[1,12]{1,0} %arg_1, f32[1,12]{1,0} %broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
}

%_pop_op_wide_const {
  %constant.9.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %broadcast.9.39.clone = f32[12,12]{1,0} broadcast(f32[] %constant.9.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
}

%_pop_op_wide_const.1 {
  %constant.9.6.clone.1 = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT %broadcast.9.48.clone = f32[4,12]{1,0} broadcast(f32[] %constant.9.6.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
}

ENTRY %cluster_1 {
  %arg2.9.2 = f32[12,12]{1,0} parameter(2), metadata={op_name="XLA_Args"}
  %call.2 = f32[12,12]{1,0} call(), to_apply=%_pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %arg0.9.0 = f32[1,4]{1,0} parameter(0), metadata={op_name="XLA_Args"}
  %arg3.9.3 = f32[4,12]{1,0} parameter(3), metadata={op_name="XLA_Args"}
  %dot.9.9 = f32[1,12]{1,0} dot(f32[1,4]{1,0} %arg0.9.0, f32[4,12]{1,0} %arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  %call = f32[1,12]{1,0} call(f32[1,12]{1,0} %dot.9.9), to_apply=%_pop_op_relu, metadata={op_type="Relu" op_name="Relu"}
  %transpose.9.35 = f32[12,1]{0,1} transpose(f32[1,12]{1,0} %call), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %dot.9.13 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %call, f32[12,12]{1,0} %arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  %constant.9.14 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %reduce.9.15 = f32[1]{0} reduce(f32[1,12]{1,0} %dot.9.13, f32[] %constant.9.14), dimensions={1}, to_apply=%max7, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.9.16 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.15), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %subtract.9.17 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %dot.9.13, f32[1,12]{1,0} %broadcast.9.16), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %exponential.9.18 = f32[1,12]{1,0} exponential(f32[1,12]{1,0} %subtract.9.17), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %constant.9.10 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  %reduce.9.21 = f32[1]{0} reduce(f32[1,12]{1,0} %exponential.9.18, f32[] %constant.9.10), dimensions={1}, to_apply=%add8, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %broadcast.9.32 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.21), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %divide.9.33 = f32[1,12]{1,0} divide(f32[1,12]{1,0} %exponential.9.18, f32[1,12]{1,0} %broadcast.9.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %arg1.9.1 = f32[1,12]{1,0} parameter(1), metadata={op_name="XLA_Args"}
  %subtract.9.34 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %divide.9.33, f32[1,12]{1,0} %arg1.9.1), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  %dot.9.36 = f32[12,12]{1,0} dot(f32[12,1]{0,1} %transpose.9.35, f32[1,12]{1,0} %subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  %multiply.9.40 = f32[12,12]{1,0} multiply(f32[12,12]{1,0} %call.2, f32[12,12]{1,0} %dot.9.36), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %subtract.9.41 = f32[12,12]{1,0} subtract(f32[12,12]{1,0} %arg2.9.2, f32[12,12]{1,0} %multiply.9.40), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  %call.3 = f32[4,12]{1,0} call(), to_apply=%_pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  %transpose.9.46 = f32[4,1]{0,1} transpose(f32[1,4]{1,0} %arg0.9.0), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %transpose.9.37 = f32[12,12]{0,1} transpose(f32[12,12]{1,0} %arg2.9.2), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %dot.9.38 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %subtract.9.34, f32[12,12]{0,1} %transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  %call.1 = f32[1,12]{1,0} call(f32[1,12]{1,0} %call, f32[1,12]{1,0} %dot.9.38), to_apply=%_pop_op_relugrad, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  %dot.9.47 = f32[4,12]{1,0} dot(f32[4,1]{0,1} %transpose.9.46, f32[1,12]{1,0} %call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  %multiply.9.49 = f32[4,12]{1,0} multiply(f32[4,12]{1,0} %call.3, f32[4,12]{1,0} %dot.9.47), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  %subtract.9.50 = f32[4,12]{1,0} subtract(f32[4,12]{1,0} %arg3.9.3, f32[4,12]{1,0} %multiply.9.49), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  ROOT %tuple.9.55 = (f32[12,12]{1,0}, f32[4,12]{1,0}) tuple(f32[12,12]{1,0} %subtract.9.41, f32[4,12]{1,0} %subtract.9.50), metadata={op_name="XLA_Retvals"}
}

)";

  std::string hlo_string1 = R"(

HloModule top

%max7 {
  %x.7.0 = f32[] parameter(0)
  %y.7.1 = f32[] parameter(1)
  ROOT %maximum.7.2 = f32[] maximum(f32[] %x.7.0, f32[] %y.7.1)
}

%add8 {
  %x.8.0 = f32[] parameter(0)
  %y.8.1 = f32[] parameter(1)
  ROOT %add.8.2 = f32[] add(f32[] %x.8.0, f32[] %y.8.1)
}

%_pop_op_relu {
  %constant.9.10.clone = f32[] constant(0)
  %broadcast.9.11.clone = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone), dimensions={}
  %arg_0 = f32[1,12]{1,0} parameter(0)
  ROOT %maximum.9.12.clone = f32[1,12]{1,0} maximum(f32[1,12]{1,0} %broadcast.9.11.clone, f32[1,12]{1,0} %arg_0)
}

%_pop_op_relugrad {
  %arg_0.1 = f32[1,12]{1,0} parameter(0)
  %constant.9.10.clone.1 = f32[] constant(0)
  %broadcast.9.11.clone.1 = f32[1,12]{1,0} broadcast(f32[] %constant.9.10.clone.1), dimensions={}
  %greater-than.9.44.clone = pred[1,12]{1,0} greater-than(f32[1,12]{1,0} %arg_0.1, f32[1,12]{1,0} %broadcast.9.11.clone.1)
  %arg_1 = f32[1,12]{1,0} parameter(1)
  ROOT %select.9.45.clone = f32[1,12]{1,0} select(pred[1,12]{1,0} %greater-than.9.44.clone, f32[1,12]{1,0} %arg_1, f32[1,12]{1,0} %broadcast.9.11.clone.1)
}

%_pop_op_wide_const {
  %constant.9.6.clone = f32[] constant(0.01)
  ROOT %broadcast.9.39.clone = f32[12,12]{1,0} broadcast(f32[] %constant.9.6.clone), dimensions={}
}

%_pop_op_wide_const.1 {
  %constant.9.6.clone.1 = f32[] constant(0.01)
  ROOT %broadcast.9.48.clone = f32[4,12]{1,0} broadcast(f32[] %constant.9.6.clone.1), dimensions={}
}

ENTRY %cluster_1 {
  %arg2.9.2 = f32[12,12]{1,0} parameter(2)
  %call.2 = f32[12,12]{1,0} call(), to_apply=%_pop_op_wide_const
  %arg0.9.0 = f32[1,4]{1,0} parameter(0)
  %arg3.9.3 = f32[4,12]{1,0} parameter(3)
  %dot.9.9 = f32[1,12]{1,0} dot(f32[1,4]{1,0} %arg0.9.0, f32[4,12]{1,0} %arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %call = f32[1,12]{1,0} call(f32[1,12]{1,0} %dot.9.9), to_apply=%_pop_op_relu
  %transpose.9.35 = f32[12,1]{0,1} transpose(f32[1,12]{1,0} %call), dimensions={1,0}
  %dot.9.13 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %call, f32[12,12]{1,0} %arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %constant.9.14 = f32[] constant(-inf)
  %reduce.9.15 = f32[1]{0} reduce(f32[1,12]{1,0} %dot.9.13, f32[] %constant.9.14), dimensions={1}, to_apply=%max7
  %broadcast.9.16 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.15), dimensions={0}
  %subtract.9.17 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %dot.9.13, f32[1,12]{1,0} %broadcast.9.16)
  %exponential.9.18 = f32[1,12]{1,0} exponential(f32[1,12]{1,0} %subtract.9.17)
  %constant.9.10 = f32[] constant(0)
  %reduce.9.21 = f32[1]{0} reduce(f32[1,12]{1,0} %exponential.9.18, f32[] %constant.9.10), dimensions={1}, to_apply=%add8
  %broadcast.9.32 = f32[1,12]{1,0} broadcast(f32[1]{0} %reduce.9.21), dimensions={0}
  %divide.9.33 = f32[1,12]{1,0} divide(f32[1,12]{1,0} %exponential.9.18, f32[1,12]{1,0} %broadcast.9.32)
  %arg1.9.1 = f32[1,12]{1,0} parameter(1)
  %subtract.9.34 = f32[1,12]{1,0} subtract(f32[1,12]{1,0} %divide.9.33, f32[1,12]{1,0} %arg1.9.1)
  %dot.9.36 = f32[12,12]{1,0} dot(f32[12,1]{0,1} %transpose.9.35, f32[1,12]{1,0} %subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %multiply.9.40 = f32[12,12]{1,0} multiply(f32[12,12]{1,0} %call.2, f32[12,12]{1,0} %dot.9.36)
  %subtract.9.41 = f32[12,12]{1,0} subtract(f32[12,12]{1,0} %arg2.9.2, f32[12,12]{1,0} %multiply.9.40)
  %call.3 = f32[4,12]{1,0} call(), to_apply=%_pop_op_wide_const.1
  %transpose.9.46 = f32[4,1]{0,1} transpose(f32[1,4]{1,0} %arg0.9.0), dimensions={1,0}
  %transpose.9.37 = f32[12,12]{0,1} transpose(f32[12,12]{1,0} %arg2.9.2), dimensions={1,0}
  %dot.9.38 = f32[1,12]{1,0} dot(f32[1,12]{1,0} %subtract.9.34, f32[12,12]{0,1} %transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %call.1 = f32[1,12]{1,0} call(f32[1,12]{1,0} %call, f32[1,12]{1,0} %dot.9.38), to_apply=%_pop_op_relugrad
  %dot.9.47 = f32[4,12]{1,0} dot(f32[4,1]{0,1} %transpose.9.46, f32[1,12]{1,0} %call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  %multiply.9.49 = f32[4,12]{1,0} multiply(f32[4,12]{1,0} %call.3, f32[4,12]{1,0} %dot.9.47)
  %subtract.9.50 = f32[4,12]{1,0} subtract(f32[4,12]{1,0} %arg3.9.3, f32[4,12]{1,0} %multiply.9.49)
  ROOT %tuple.9.55 = (f32[12,12]{1,0}, f32[4,12]{1,0}) tuple(f32[12,12]{1,0} %subtract.9.41, f32[4,12]{1,0} %subtract.9.50)
}

)";
  auto module0_or_status =
      HloRunner::CreateModuleFromString(hlo_string0, GetDebugOptionsForTest());
  EXPECT_TRUE(module0_or_status.ok());
  auto* module0 = module0_or_status.ValueOrDie().get();

  auto module1_or_status =
      HloRunner::CreateModuleFromString(hlo_string1, GetDebugOptionsForTest());
  EXPECT_TRUE(module1_or_status.ok());
  auto* module1 = module1_or_status.ValueOrDie().get();

  HloHash hash0(module0);
  HloHash hash1(module1);
  EXPECT_EQ(hash0.GetHash(), hash1.GetHash());
  EXPECT_EQ(hash0.GetProtoStr(), hash1.GetProtoStr());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
