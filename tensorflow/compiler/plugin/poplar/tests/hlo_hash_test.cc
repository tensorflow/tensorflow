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

#include "tensorflow/compiler/plugin/poplar/driver/tools/hlo_hash.h"
#include "tensorflow/compiler/plugin/poplar/driver/passes/custom_op_replacer.h"

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
    auto lt0 = builder_cond.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), c0, limit0, ComparisonDirection::kLt));
    auto lt1 = builder_cond.AddInstruction(HloInstruction::CreateCompare(
        ShapeUtil::MakeShape(PRED, {}), c1, limit1, ComparisonDirection::kGt));
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
  auto module0 = CreateNewVerifiedModule();
  CreateWhileLoop(module0.get(), TestName());
  auto module1 = CreateNewVerifiedModule();
  CreateWhileLoop(module1.get(), TestName());
  HloHash hash0(module0.get());
  HloHash hash1(module1.get());
  EXPECT_EQ(hash0.GetHash(), hash1.GetHash());
  EXPECT_EQ(hash0.GetProtoStr(), hash1.GetProtoStr());
}

TEST_F(HloHashTest, DifferentNamesSameHash) {
  std::string hlo_string0 = R"(
HloModule top

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0), metadata={op_type="Relu" op_name="Relu"}
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  greater-than.9.44.clone = pred[1,12] compare(arg_0.1, broadcast.9.11.clone.1), direction=GT, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
}

ENTRY cluster_1 {
  arg2.9.2 = f32[12,12] parameter(2), metadata={op_name="XLA_Args"}
  call.2 = f32[12,12] call(), to_apply=_pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  arg0.9.0 = f32[1,4] parameter(0), metadata={op_name="XLA_Args"}
  arg3.9.3 = f32[4,12] parameter(3), metadata={op_name="XLA_Args"}
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  call = f32[1,12] call(dot.9.9), to_apply=_pop_op_relu, metadata={op_type="Relu" op_name="Relu"}
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  constant.9.14 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  exponential.9.18 = f32[1,12] exponential(subtract.9.17), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  constant.9.10 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  arg1.9.1 = f32[1,12] parameter(1), metadata={op_name="XLA_Args"}
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  call.3 = f32[4,12] call(), to_apply=_pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  call.1 = f32[1,12] call(call, dot.9.38), to_apply=_pop_op_relugrad, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  ROOT tuple.9.55 = (f32[12,12], f32[4,12]) tuple(subtract.9.41, subtract.9.50), metadata={op_name="XLA_Retvals"}
}

)";

  std::string hlo_string1 = R"(
HloModule top

xx_max7 {
  xx_x.7.0 = f32[] parameter(0)
  xx_y.7.1 = f32[] parameter(1)
  ROOT xx_maximum.7.2 = f32[] maximum(xx_x.7.0, xx_y.7.1)
}

xx_add8 {
  xx_x.8.0 = f32[] parameter(0)
  xx_y.8.1 = f32[] parameter(1)
  ROOT xx_add.8.2 = f32[] add(xx_x.8.0, xx_y.8.1)
}

__pop_op_relu {
  xx_constant.9.10.clone = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  xx_broadcast.9.11.clone = f32[1,12] broadcast(xx_constant.9.10.clone), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  xx_arg_0 = f32[1,12] parameter(0)
  ROOT xx_maximum.9.12.clone = f32[1,12] maximum(xx_broadcast.9.11.clone, xx_arg_0), metadata={op_type="Relu" op_name="Relu"}
}

__pop_op_relugrad {
  xx_arg_0.1 = f32[1,12] parameter(0)
  xx_constant.9.10.clone.1 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  xx_broadcast.9.11.clone.1 = f32[1,12] broadcast(xx_constant.9.10.clone.1), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  xx_greater-than.9.44.clone = pred[1,12] compare(xx_arg_0.1, xx_broadcast.9.11.clone.1), direction=GT, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  xx_arg_1 = f32[1,12] parameter(1)
  ROOT xx_select.9.45.clone = f32[1,12] select(xx_greater-than.9.44.clone, xx_arg_1, xx_broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
}

__pop_op_wide_const {
  xx_constant.9.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT xx_broadcast.9.39.clone = f32[12,12] broadcast(xx_constant.9.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
}

__pop_op_wide_const.1 {
  xx_constant.9.6.clone.1 = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT xx_broadcast.9.48.clone = f32[4,12] broadcast(xx_constant.9.6.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
}

ENTRY xx_cluster_1 {
  xx_arg2.9.2 = f32[12,12] parameter(2), metadata={op_name="XLA_Args"}
  xx_call.2 = f32[12,12] call(), to_apply=__pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  xx_arg0.9.0 = f32[1,4] parameter(0), metadata={op_name="XLA_Args"}
  xx_arg3.9.3 = f32[4,12] parameter(3), metadata={op_name="XLA_Args"}
  xx_dot.9.9 = f32[1,12] dot(xx_arg0.9.0, xx_arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  xx_call = f32[1,12] call(xx_dot.9.9), to_apply=__pop_op_relu, metadata={op_type="Relu" op_name="Relu"}
  xx_transpose.9.35 = f32[12,1] transpose(xx_call), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  xx_dot.9.13 = f32[1,12] dot(xx_call, xx_arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  xx_constant.9.14 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  xx_reduce.9.15 = f32[1] reduce(xx_dot.9.13, xx_constant.9.14), dimensions={1}, to_apply=xx_max7, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  xx_broadcast.9.16 = f32[1,12] broadcast(xx_reduce.9.15), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  xx_subtract.9.17 = f32[1,12] subtract(xx_dot.9.13, xx_broadcast.9.16), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  xx_exponential.9.18 = f32[1,12] exponential(xx_subtract.9.17), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  xx_constant.9.10 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  xx_reduce.9.21 = f32[1] reduce(xx_exponential.9.18, xx_constant.9.10), dimensions={1}, to_apply=xx_add8, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  xx_broadcast.9.32 = f32[1,12] broadcast(xx_reduce.9.21), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  xx_divide.9.33 = f32[1,12] divide(xx_exponential.9.18, xx_broadcast.9.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  xx_arg1.9.1 = f32[1,12] parameter(1), metadata={op_name="XLA_Args"}
  xx_subtract.9.34 = f32[1,12] subtract(xx_divide.9.33, xx_arg1.9.1), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  xx_dot.9.36 = f32[12,12] dot(xx_transpose.9.35, xx_subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  xx_multiply.9.40 = f32[12,12] multiply(xx_call.2, xx_dot.9.36), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  xx_subtract.9.41 = f32[12,12] subtract(xx_arg2.9.2, xx_multiply.9.40), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  xx_call.3 = f32[4,12] call(), to_apply=__pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  xx_transpose.9.46 = f32[4,1] transpose(xx_arg0.9.0), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  xx_transpose.9.37 = f32[12,12] transpose(xx_arg2.9.2), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  xx_dot.9.38 = f32[1,12] dot(xx_subtract.9.34, xx_transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  xx_call.1 = f32[1,12] call(xx_call, xx_dot.9.38), to_apply=__pop_op_relugrad, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  xx_dot.9.47 = f32[4,12] dot(xx_transpose.9.46, xx_call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  xx_multiply.9.49 = f32[4,12] multiply(xx_call.3, xx_dot.9.47), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  xx_subtract.9.50 = f32[4,12] subtract(xx_arg3.9.3, xx_multiply.9.49), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  ROOT xx_tuple.9.55 = (f32[12,12], f32[4,12]) tuple(xx_subtract.9.41, xx_subtract.9.50), metadata={op_name="XLA_Retvals"}
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

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0), metadata={op_type="Relu" op_name="Relu"}
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  greater-than.9.44.clone = pred[1,12] compare(arg_0.1, broadcast.9.11.clone.1), direction=GT, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
}

ENTRY cluster_1 {
  arg2.9.2 = f32[12,12] parameter(2), metadata={op_name="XLA_Args"}
  call.2 = f32[12,12] call(), to_apply=_pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  arg0.9.0 = f32[1,4] parameter(0), metadata={op_name="XLA_Args"}
  arg3.9.3 = f32[4,12] parameter(3), metadata={op_name="XLA_Args"}
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  call = f32[1,12] call(dot.9.9), to_apply=_pop_op_relu, metadata={op_type="Relu" op_name="Relu"}
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  constant.9.14 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  exponential.9.18 = f32[1,12] exponential(subtract.9.17), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  constant.9.10 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  arg1.9.1 = f32[1,12] parameter(1), metadata={op_name="XLA_Args"}
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  call.3 = f32[4,12] call(), to_apply=_pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  call.1 = f32[1,12] call(call, dot.9.38), to_apply=_pop_op_relugrad, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  ROOT tuple.9.55 = (f32[12,12], f32[4,12]) tuple(subtract.9.41, subtract.9.50), metadata={op_name="XLA_Retvals"}
}

)";

  std::string hlo_string1 = R"(
HloModule top

_pop_op_relu {
  constant.17.9.clone = f32[] constant(0), metadata={op_type="Relu" op_name="dense/Relu"}
  broadcast.17.10.clone = f32[32,32] broadcast(constant.17.9.clone), dimensions={}, metadata={op_type="Relu" op_name="dense/Relu"}
  arg_0 = f32[32,32] parameter(0)
  ROOT maximum.17.11.clone = f32[32,32] maximum(f32[32,32] broadcast.17.10.clone, f32[32,32] arg_0), metadata={op_type="Relu" op_name="dense/Relu"}
}

_pop_op_sigmoid {
  constant.17.15.clone = f32[] constant(0.5), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  broadcast.17.21.clone = f32[32,1] broadcast(constant.17.15.clone), dimensions={}, metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  arg_0.1 = f32[32,1] parameter(0)
  multiply.17.17.clone = f32[32,1] multiply(f32[32,1] broadcast.17.21.clone, f32[32,1] arg_0.1), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  tanh.17.18.clone = f32[32,1] tanh(f32[32,1] multiply.17.17.clone), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  multiply.17.20.clone = f32[32,1] multiply(f32[32,1] broadcast.17.21.clone, f32[32,1] tanh.17.18.clone), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  ROOT add.17.22.clone = f32[32,1] add(f32[32,1] broadcast.17.21.clone, f32[32,1] multiply.17.20.clone), metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
}

ENTRY cluster_9 {
  arg0.17.0 = f32[32,100] parameter(0), metadata={op_name="XLA_Args"}
  arg4.17.4 = f32[100,32] parameter(4), metadata={op_name="XLA_Args"}
  dot.17.6 = f32[32,32] dot(f32[32,100] arg0.17.0, f32[100,32] arg4.17.4), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="dense/MatMul"}
  arg3.17.3 = f32[32] parameter(3), metadata={op_name="XLA_Args"}
  broadcast.17.7 = f32[32,32] broadcast(f32[32] arg3.17.3), dimensions={1}, metadata={op_type="BiasAdd" op_name="dense/BiasAdd"}
  add.17.8 = f32[32,32] add(f32[32,32] dot.17.6, f32[32,32] broadcast.17.7), metadata={op_type="BiasAdd" op_name="dense/BiasAdd"}
  call = f32[32,32] call(f32[32,32] add.17.8), to_apply=_pop_op_relu, metadata={op_type="Relu" op_name="dense/Relu"}
  arg2.17.2 = f32[32,1] parameter(2), metadata={op_name="XLA_Args"}
  dot.17.12 = f32[32,1] dot(f32[32,32] call, f32[32,1] arg2.17.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="dense_1/MatMul"}
  arg1.17.1 = f32[1] parameter(1), metadata={op_name="XLA_Args"}
  broadcast.17.13 = f32[32,1] broadcast(arg1.17.1), dimensions={1}, metadata={op_type="BiasAdd" op_name="dense_1/BiasAdd"}
  add.17.14 = f32[32,1] add(f32[32,1] dot.17.12, f32[32,1] broadcast.17.13), metadata={op_type="BiasAdd" op_name="dense_1/BiasAdd"}
  call.1 = f32[32,1] call(f32[32,1] add.17.14), to_apply=_pop_op_sigmoid, metadata={op_type="Sigmoid" op_name="dense_1/Sigmoid"}
  ROOT tuple.17.24 = (f32[32,1]) tuple(f32[32,1] call.1), metadata={op_name="XLA_Retvals"}
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

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0), metadata={op_type="Relu" op_name="Relu"}
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}, metadata={op_type="Relu" op_name="Relu"}
  greater-than.9.44.clone = pred[1,12] compare(arg_0.1, broadcast.9.11.clone.1), direction=GT, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1), metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01), metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}, metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
}

ENTRY cluster_1 {
  arg2.9.2 = f32[12,12] parameter(2), metadata={op_name="XLA_Args"}
  call.2 = f32[12,12] call(), to_apply=_pop_op_wide_const, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  arg0.9.0 = f32[1,4] parameter(0), metadata={op_name="XLA_Args"}
  arg3.9.3 = f32[4,12] parameter(3), metadata={op_name="XLA_Args"}
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul"}
  call = f32[1,12] call(dot.9.9), to_apply=_pop_op_relu, metadata={op_type="Relu" op_name="Relu"}
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="MatMul_1"}
  constant.9.14 = f32[] constant(-inf), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  exponential.9.18 = f32[1,12] exponential(subtract.9.17), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  constant.9.10 = f32[] constant(0), metadata={op_type="Relu" op_name="Relu"}
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}, metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  arg1.9.1 = f32[1,12] parameter(1), metadata={op_name="XLA_Args"}
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1), metadata={op_type="SoftmaxCrossEntropyWithLogits" op_name="softmax_cross_entropy_with_logits_sg"}
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul_1"}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w2/ResourceApplyGradientDescent"}
  call.3 = f32[4,12] call(), to_apply=_pop_op_wide_const.1, metadata={op_type="Const" op_name="GradientDescent/learning_rate"}
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_1_grad/MatMul"}
  call.1 = f32[1,12] call(call, dot.9.38), to_apply=_pop_op_relugrad, metadata={op_type="ReluGrad" op_name="gradients/Relu_grad/ReluGrad"}
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}, metadata={op_type="MatMul" op_name="gradients/MatMul_grad/MatMul_1"}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49), metadata={op_type="ResourceApplyGradientDescent" op_name="GradientDescent/update_w1/ResourceApplyGradientDescent"}
  ROOT tuple.9.55 = (f32[12,12], f32[4,12]) tuple(subtract.9.41, subtract.9.50), metadata={op_name="XLA_Retvals"}
}

)";

  std::string hlo_string1 = R"(

HloModule top

max7 {
  x.7.0 = f32[] parameter(0)
  y.7.1 = f32[] parameter(1)
  ROOT maximum.7.2 = f32[] maximum(x.7.0, y.7.1)
}

add8 {
  x.8.0 = f32[] parameter(0)
  y.8.1 = f32[] parameter(1)
  ROOT add.8.2 = f32[] add(x.8.0, y.8.1)
}

_pop_op_relu {
  constant.9.10.clone = f32[] constant(0)
  broadcast.9.11.clone = f32[1,12] broadcast(constant.9.10.clone), dimensions={}
  arg_0 = f32[1,12] parameter(0)
  ROOT maximum.9.12.clone = f32[1,12] maximum(broadcast.9.11.clone, arg_0)
}

_pop_op_relugrad {
  arg_0.1 = f32[1,12] parameter(0)
  constant.9.10.clone.1 = f32[] constant(0)
  broadcast.9.11.clone.1 = f32[1,12] broadcast(constant.9.10.clone.1), dimensions={}
  greater-than.9.44.clone = pred[1,12] compare(arg_0.1, broadcast.9.11.clone.1), direction=GT
  arg_1 = f32[1,12] parameter(1)
  ROOT select.9.45.clone = f32[1,12] select(greater-than.9.44.clone, arg_1, broadcast.9.11.clone.1)
}

_pop_op_wide_const {
  constant.9.6.clone = f32[] constant(0.01)
  ROOT broadcast.9.39.clone = f32[12,12] broadcast(constant.9.6.clone), dimensions={}
}

_pop_op_wide_const.1 {
  constant.9.6.clone.1 = f32[] constant(0.01)
  ROOT broadcast.9.48.clone = f32[4,12] broadcast(constant.9.6.clone.1), dimensions={}
}

ENTRY cluster_1 {
  arg2.9.2 = f32[12,12] parameter(2)
  call.2 = f32[12,12] call(), to_apply=_pop_op_wide_const
  arg0.9.0 = f32[1,4] parameter(0)
  arg3.9.3 = f32[4,12] parameter(3)
  dot.9.9 = f32[1,12] dot(arg0.9.0, arg3.9.3), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call = f32[1,12] call(dot.9.9), to_apply=_pop_op_relu
  transpose.9.35 = f32[12,1] transpose(call), dimensions={1,0}
  dot.9.13 = f32[1,12] dot(call, arg2.9.2), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  constant.9.14 = f32[] constant(-inf)
  reduce.9.15 = f32[1] reduce(dot.9.13, constant.9.14), dimensions={1}, to_apply=max7
  broadcast.9.16 = f32[1,12] broadcast(reduce.9.15), dimensions={0}
  subtract.9.17 = f32[1,12] subtract(dot.9.13, broadcast.9.16)
  exponential.9.18 = f32[1,12] exponential(subtract.9.17)
  constant.9.10 = f32[] constant(0)
  reduce.9.21 = f32[1] reduce(exponential.9.18, constant.9.10), dimensions={1}, to_apply=add8
  broadcast.9.32 = f32[1,12] broadcast(reduce.9.21), dimensions={0}
  divide.9.33 = f32[1,12] divide(exponential.9.18, broadcast.9.32)
  arg1.9.1 = f32[1,12] parameter(1)
  subtract.9.34 = f32[1,12] subtract(divide.9.33, arg1.9.1)
  dot.9.36 = f32[12,12] dot(transpose.9.35, subtract.9.34), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.40 = f32[12,12] multiply(call.2, dot.9.36)
  subtract.9.41 = f32[12,12] subtract(arg2.9.2, multiply.9.40)
  call.3 = f32[4,12] call(), to_apply=_pop_op_wide_const.1
  transpose.9.46 = f32[4,1] transpose(arg0.9.0), dimensions={1,0}
  transpose.9.37 = f32[12,12] transpose(arg2.9.2), dimensions={1,0}
  dot.9.38 = f32[1,12] dot(subtract.9.34, transpose.9.37), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  call.1 = f32[1,12] call(call, dot.9.38), to_apply=_pop_op_relugrad
  dot.9.47 = f32[4,12] dot(transpose.9.46, call.1), lhs_contracting_dims={1}, rhs_contracting_dims={0}
  multiply.9.49 = f32[4,12] multiply(call.3, dot.9.47)
  subtract.9.50 = f32[4,12] subtract(arg3.9.3, multiply.9.49)
  ROOT tuple.9.55 = (f32[12,12], f32[4,12]) tuple(subtract.9.41, subtract.9.50)
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

TEST_F(HloHashTest, DifferentModuleName) {
  std::string hlo_string0 = R"(
HloModule top1

ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  mul0 = f32[2] multiply(arg0, arg1)
  mul1 = f32[2] multiply(arg0, arg2)
  ROOT tuple = (f32[2], f32[2]) tuple(mul0, mul1)
}

)";

  std::string hlo_string1 = R"(
HloModule top2

ENTRY cluster_2 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  mul0 = f32[2] multiply(arg0, arg1)
  mul1 = f32[2] multiply(arg0, arg2)
  ROOT tuple = (f32[2], f32[2]) tuple(mul0, mul1)
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

TEST_F(HloHashTest, DifferentInstructionOrder) {
  std::string hlo_string0 = R"(
HloModule top1

ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  mul0 = f32[2] multiply(arg0, arg1)
  mul1 = f32[2] multiply(arg0, arg2)
  ROOT tuple = (f32[2], f32[2]) tuple(mul0, mul1)
}

)";

  std::string hlo_string1 = R"(
HloModule top1

ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  mul1 = f32[2] multiply(arg0, arg2)
  mul0 = f32[2] multiply(arg0, arg1)
  ROOT tuple = (f32[2], f32[2]) tuple(mul0, mul1)
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

TEST_F(HloHashTest, DifferentParameterOrder) {
  std::string hlo_string0 = R"(
HloModule top1

ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  mul0 = f32[2] multiply(arg0, arg1)
  mul1 = f32[2] multiply(arg0, arg2)
  ROOT tuple = (f32[2], f32[2]) tuple(mul0, mul1)
}

)";

  std::string hlo_string1 = R"(
HloModule top1

ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg2 = f32[2] parameter(2)
  arg1 = f32[2] parameter(1)
  mul0 = f32[2] multiply(arg0, arg1)
  mul1 = f32[2] multiply(arg0, arg2)
  ROOT tuple = (f32[2], f32[2]) tuple(mul0, mul1)
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

TEST_F(HloHashTest, DifferentInstructionOrderWithControlDeps) {
  std::string hlo_string0 = R"(
HloModule top1

ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  mul0 = f32[2] multiply(arg0, arg1)
  mul1 = f32[2] multiply(arg0, arg2), control-predecessors={arg2}
  ROOT tuple = (f32[2], f32[2]) tuple(mul0, mul1)
}

)";

  std::string hlo_string1 = R"(
HloModule top1

ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  mul1 = f32[2] multiply(arg0, arg2), control-predecessors={arg2}
  mul0 = f32[2] multiply(arg0, arg1)
  ROOT tuple = (f32[2], f32[2]) tuple(mul0, mul1)
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

TEST_F(HloHashTest, DifferentComputationOrder) {
  std::string hlo_string0 = R"(
 HloModule top1

_pop_op_relu {
  c0 = f32[] constant(0)
  b0 = f32[2] broadcast(c0), dimensions={}
  a0 = f32[2] parameter(0)
  ROOT m0 = f32[2] maximum(b0, a0)
}

_pop_op_relugrad {
  a1.1 = f32[2] parameter(0)
  c1 = f32[] constant(0)
  b1 = f32[2] broadcast(c1), dimensions={}
  gt1 = pred[2] compare(a1.1, b1), direction=GT
  a1.2 = f32[2] parameter(1)
  ROOT s1 = f32[2] select(gt1, a1.2, b1)
}

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  c0 = f32[2] call(arg0, arg2), to_apply=_pop_op_relugrad
  c1 = f32[2] call(arg1), to_apply=_pop_op_relu
  ROOT tuple = (f32[2], f32[2]) tuple(c0, c1)
}

)";

  std::string hlo_string1 = R"(
 HloModule top1

_pop_op_relugrad {
  a1.1 = f32[2] parameter(0)
  c1 = f32[] constant(0)
  b1 = f32[2] broadcast(c1), dimensions={}
  gt1 = pred[2] compare(a1.1, b1), direction=GT
  a1.2 = f32[2] parameter(1)
  ROOT s1 = f32[2] select(gt1, a1.2, b1)
}

_pop_op_relu {
  c0 = f32[] constant(0)
  b0 = f32[2] broadcast(c0), dimensions={}
  a0 = f32[2] parameter(0)
  ROOT m0 = f32[2] maximum(b0, a0)
}

 ENTRY cluster_1 {
  arg0 = f32[2] parameter(0)
  arg1 = f32[2] parameter(1)
  arg2 = f32[2] parameter(2)
  c0 = f32[2] call(arg0, arg2), to_apply=_pop_op_relugrad
  c1 = f32[2] call(arg1), to_apply=_pop_op_relu
  ROOT tuple = (f32[2], f32[2]) tuple(c0, c1)
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

TEST_F(HloHashTest, HloPoplarInstructionsDifferent) {
  std::string hlo_string0 = R"(
HloModule top

ENTRY c1 {
  p0 = (s32[20], s32[20], s32[20], s32[20]) parameter(0)
  p0_0 = s32[20] get-tuple-element(p0), index=0

  c = s32[20] custom-call(p0), custom_call_target="Popnn::LstmLayerFwd", backend_config="{\"num_channels\":4, \"is_training\":false, \"partials_dtype\":\"DT_FLOAT\"}\n"

  ROOT a = s32[20] add(p0_0, c)
}

)";

  std::string hlo_string1 = R"(
HloModule top

ENTRY c1 {
  p0 = (s32[20], s32[20], s32[20], s32[20]) parameter(0)
  p0_0 = s32[20] get-tuple-element(p0), index=0

  c = s32[20] custom-call(p0), custom_call_target="Popnn::LstmLayerFwd", backend_config="{\"num_channels\":4, \"is_training\":true, \"partials_dtype\":\"DT_FLOAT\"}\n"

  ROOT a = s32[20] add(p0_0, c)
}

)";

  CustomOpReplacer custom_op_replacer;

  auto module0_or_status =
      HloRunner::CreateModuleFromString(hlo_string0, GetDebugOptionsForTest());
  EXPECT_TRUE(module0_or_status.ok());
  auto* module0 = module0_or_status.ValueOrDie().get();
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  auto module1_or_status =
      HloRunner::CreateModuleFromString(hlo_string1, GetDebugOptionsForTest());
  EXPECT_TRUE(module1_or_status.ok());
  auto* module1 = module1_or_status.ValueOrDie().get();
  EXPECT_TRUE(custom_op_replacer.Run(module1).ValueOrDie());

  HloHash hash0(module0);
  HloHash hash1(module1);
  EXPECT_NE(hash0.GetHash(), hash1.GetHash());
  EXPECT_NE(hash0.GetProtoStr(), hash1.GetProtoStr());
}

TEST_F(HloHashTest, HloPoplarInstructionsSame) {
  std::string hlo_string0 = R"(
HloModule top

ENTRY c1 {
  p0 = (s32[20], s32[20], s32[20], s32[20]) parameter(0)
  p0_0 = s32[20] get-tuple-element(p0), index=0

  c = s32[20] custom-call(p0), custom_call_target="Popnn::LstmLayerFwd", backend_config="{\"num_channels\":4, \"is_training\":false, \"partials_dtype\":\"DT_FLOAT\"}\n"

  ROOT a = s32[20] add(p0_0, c)
}

)";

  std::string hlo_string1 = R"(
HloModule top

ENTRY c1 {
  p0 = (s32[20], s32[20], s32[20], s32[20]) parameter(0)
  p0_0 = s32[20] get-tuple-element(p0), index=0

  c = s32[20] custom-call(p0), custom_call_target="Popnn::LstmLayerFwd", backend_config="{\"num_channels\":4, \"is_training\":false, \"partials_dtype\":\"DT_FLOAT\"}\n"

  ROOT a = s32[20] add(p0_0, c)
}

)";

  CustomOpReplacer custom_op_replacer;

  auto module0_or_status =
      HloRunner::CreateModuleFromString(hlo_string0, GetDebugOptionsForTest());
  EXPECT_TRUE(module0_or_status.ok());
  auto* module0 = module0_or_status.ValueOrDie().get();
  EXPECT_TRUE(custom_op_replacer.Run(module0).ValueOrDie());

  auto module1_or_status =
      HloRunner::CreateModuleFromString(hlo_string1, GetDebugOptionsForTest());
  EXPECT_TRUE(module1_or_status.ok());
  auto* module1 = module1_or_status.ValueOrDie().get();
  EXPECT_TRUE(custom_op_replacer.Run(module1).ValueOrDie());

  HloHash hash0(module0);
  HloHash hash1(module1);
  EXPECT_EQ(hash0.GetHash(), hash1.GetHash());
  EXPECT_EQ(hash0.GetProtoStr(), hash1.GetProtoStr());
}

}  // namespace
}  // namespace poplarplugin
}  // namespace xla
