/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"
#include "tensorflow/compiler/xla/client/xla_client/xla_computation.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class ConditionalOpTest : public ClientLibraryTestBase {
 protected:
  XlaComputation CreateR0ConstantComputation(float value) {
    XlaBuilder builder("Constant");
    Parameter(&builder, 0, empty_tuple_, "tuple");
    ConstantR0<float>(&builder, value);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0IdentityComputation() {
    XlaBuilder builder("Identity");
    Parameter(&builder, 0, r0f32_, "x");
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateCeilComputation(const Shape& shape) {
    XlaBuilder builder("Ceil");
    auto param = Parameter(&builder, 0, shape, "param");
    Ceil(param);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0CeilComputation() {
    return CreateCeilComputation(r0f32_);
  }

  XlaComputation CreateR1CeilComputation() {
    return CreateCeilComputation(r1s2f32_);
  }

  XlaComputation CreateFloorComputation(const Shape& shape) {
    XlaBuilder builder("Floor");
    auto param = Parameter(&builder, 0, shape, "param");
    Floor(param);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0FloorComputation() {
    return CreateFloorComputation(r0f32_);
  }

  XlaComputation CreateR1FloorComputation() {
    return CreateFloorComputation(r1s2f32_);
  }

  XlaComputation CreateTupleCeilComputation(const string& computation_name,
                                            const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    auto x_ceil = Ceil(x);
    auto y_ceil = Ceil(y);
    Tuple(&builder, {x_ceil, y_ceil});
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0TupleCeilComputation() {
    return CreateTupleCeilComputation("CeilR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleCeilComputation() {
    return CreateTupleCeilComputation("CeilR1", tuple_2_r1s2f32_);
  }

  XlaComputation CreateTupleFloorComputation(const string& computation_name,
                                             const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    auto x_floor = Floor(x);
    auto y_floor = Floor(y);
    Tuple(&builder, {x_floor, y_floor});
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0TupleFloorComputation() {
    return CreateTupleFloorComputation("FloorR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleFloorComputation() {
    return CreateTupleFloorComputation("FloorR1", tuple_2_r1s2f32_);
  }

  XlaComputation CreateTupleAddComputation(const string& computation_name,
                                           const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0TupleAddComputation() {
    return CreateTupleAddComputation("AddR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleAddComputation() {
    return CreateTupleAddComputation("AddR1", tuple_2_r1s2f32_);
  }

  XlaComputation CreateTupleSubComputation(const string& computation_name,
                                           const Shape& tuple_shape) {
    XlaBuilder builder(computation_name);
    auto tuple = Parameter(&builder, 0, tuple_shape, "tuple");
    auto x = GetTupleElement(tuple, 0);
    auto y = GetTupleElement(tuple, 1);
    Sub(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0TupleSubComputation() {
    return CreateTupleSubComputation("SubR0", tuple_2_r0f32_);
  }

  XlaComputation CreateR1TupleSubComputation() {
    return CreateTupleSubComputation("SubR1", tuple_2_r1s2f32_);
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape r1s2f32_ = ShapeUtil::MakeShape(F32, {2});
  Shape tuple_2_r0f32_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})});
  Shape tuple_2_r1s2f32_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2}), ShapeUtil::MakeShape(F32, {2})});
  Shape empty_tuple_ = ShapeUtil::MakeTupleShape({});
  ErrorSpec error_spec_{0.001};
};

// Test true and false computations that do not take any parameters.
XLA_TEST_F(ConditionalOpTest, Parameters0) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operands = Tuple(&builder, {});
  auto true_computation = CreateR0ConstantComputation(56.0f);
  auto false_computation = CreateR0ConstantComputation(12.0f);
  Conditional(pred, operands, true_computation, operands, false_computation);

  ComputeAndCompareR0<float>(&builder, 56.0f, {pred_arg.get()}, error_spec_);
}

// Test true and false computations that take in 1 parameter.
XLA_TEST_F(ConditionalOpTest, Parameters1) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.0f);
  auto operand2 = ConstantR0<float>(&builder, 12.0f);
  auto identity = CreateR0IdentityComputation();
  Conditional(pred, operand1, identity, operand2, identity);

  ComputeAndCompareR0<float>(&builder, 12.0f, {pred_arg.get()}, error_spec_);
}

// Test conditional with two different computations in the true and false cases
// that take in different arguments.
XLA_TEST_F(ConditionalOpTest, DiffComputationsDiffArgs) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.4f);
  auto operand2 = ConstantR0<float>(&builder, 12.6f);
  Conditional(pred, operand1, CreateR0CeilComputation(), operand2,
              CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {pred_arg.get()}, error_spec_);
}

// Test conditional with two different computations in the true and false cases
// that take in the same arguments.
XLA_TEST_F(ConditionalOpTest, DiffComputationsSameArg) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand = ConstantR0<float>(&builder, 12.6f);
  Conditional(pred, operand, CreateR0CeilComputation(), operand,
              CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {pred_arg.get()}, error_spec_);
}

// Test conditional with the same computation in the true and false cases but
// take in different arguments.
XLA_TEST_F(ConditionalOpTest, SameComputationDiffArgs) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.4f);
  auto operand2 = ConstantR0<float>(&builder, 12.6f);
  auto floor = CreateR0FloorComputation();
  Conditional(pred, operand1, floor, operand2, floor);

  ComputeAndCompareR0<float>(&builder, 12.0f, {pred_arg.get()}, error_spec_);
}

// Test conditional with the same computation in the true and false cases that
// take in the same arguments.
XLA_TEST_F(ConditionalOpTest, SameComputationSameArg) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand = ConstantR0<float>(&builder, 12.6f);
  auto floor = CreateR0FloorComputation();
  Conditional(pred, operand, floor, operand, floor);

  ComputeAndCompareR0<float>(&builder, 12.0f, {pred_arg.get()}, error_spec_);
}

// Test conditional with different instances of the same computation in the true
// and false cases.
XLA_TEST_F(ConditionalOpTest, SameComputationDiffInstances) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.4f);
  auto operand2 = ConstantR0<float>(&builder, 12.6f);
  Conditional(pred, operand1, CreateR0FloorComputation(), operand2,
              CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {pred_arg.get()}, error_spec_);
}

// Test the case when a call invokes a computation that contains a conditional.
XLA_TEST_F(ConditionalOpTest, ConditionalWithCall) {
  Shape r0bool = ShapeUtil::MakeShape(PRED, {});
  XlaBuilder inner_builder(TestName() + ".inner_conditional");
  auto pred_cond = Parameter(&inner_builder, 0, r0bool, "param0");
  auto true_operand = Parameter(&inner_builder, 1, r0f32_, "param1");
  auto false_operand = Parameter(&inner_builder, 2, r0f32_, "param2");
  Conditional(pred_cond, true_operand, CreateR0CeilComputation(), false_operand,
              CreateR0FloorComputation());
  auto inner_builder_result = inner_builder.Build();

  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.4f);
  auto operand2 = ConstantR0<float>(&builder, 12.6f);
  Call(&builder, inner_builder_result.ConsumeValueOrDie(),
       {pred, operand1, operand2});

  ComputeAndCompareR0<float>(&builder, 12.0f, {pred_arg.get()}, error_spec_);
}

// Test true and false computations that take in 2 parameters and predicate is
// true.
XLA_TEST_F(ConditionalOpTest, Parameters2TrueBranch) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.0f);
  auto operand2 = ConstantR0<float>(&builder, 12.0f);
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR0TupleAddComputation(), operands,
              CreateR0TupleSubComputation());

  ComputeAndCompareR0<float>(&builder, 68.0f, {pred_arg.get()}, error_spec_);
}

// Test true and false computations that take in 2 parameters and predicate is
// false.
XLA_TEST_F(ConditionalOpTest, Parameters2FalseBranch) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 56.0f);
  auto operand2 = ConstantR0<float>(&builder, 12.0f);
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR0TupleAddComputation(), operands,
              CreateR0TupleSubComputation());

  ComputeAndCompareR0<float>(&builder, 44.0f, {pred_arg.get()}, error_spec_);
}

// Test true and false computations that take in 2 array parameters and
// predicate is true.
XLA_TEST_F(ConditionalOpTest, Parameters2ArrayTrueBranch) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR1<float>(&builder, {24.0f, 56.0f});
  auto operand2 = ConstantR1<float>(&builder, {10.0f, 11.0f});
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
              CreateR1TupleSubComputation());

  ComputeAndCompareR1<float>(&builder, {34.0f, 67.0f}, {pred_arg.get()},
                             error_spec_);
}

// Test true and false computations that take in 2 array parameters and
// predicate is false.
XLA_TEST_F(ConditionalOpTest, Parameters2ArrayFalseBranch) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR1<float>(&builder, {24.0f, 56.0f});
  auto operand2 = ConstantR1<float>(&builder, {10.0f, 11.0f});
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
              CreateR1TupleSubComputation());

  ComputeAndCompareR1<float>(&builder, {14.0f, 45.0f}, {pred_arg.get()},
                             error_spec_);
}

// Test true and false computations that return a tuple of scalars.
XLA_TEST_F(ConditionalOpTest, ReturnTupleOfScalars) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operands = Tuple(&builder, {ConstantR0<float>(&builder, 12.2f),
                                   ConstantR0<float>(&builder, 25.6f)});
  Conditional(pred, operands, CreateR0TupleCeilComputation(), operands,
              CreateR0TupleFloorComputation());

  ComputeAndCompareTuple(
      &builder,
      *LiteralUtil::MakeTuple({LiteralUtil::CreateR0<float>(12.0f).get(),
                               LiteralUtil::CreateR0<float>(25.0f).get()}),
      {pred_arg.get()}, error_spec_);
}

// Test true and false computations that return a tuple of arrays.
XLA_TEST_F(ConditionalOpTest, ReturnTupleOfArrays) {
  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operands =
      Tuple(&builder, {ConstantR1<float>(&builder, {12.2f, 15.8f}),
                       ConstantR1<float>(&builder, {25.6f, 29.2f})});
  Conditional(pred, operands, CreateR1TupleCeilComputation(), operands,
              CreateR1TupleFloorComputation());

  ComputeAndCompareTuple(
      &builder,
      *LiteralUtil::MakeTuple(
          {LiteralUtil::CreateR1<float>({13.0f, 16.0f}).get(),
           LiteralUtil::CreateR1<float>({26.0f, 30.0f}).get()}),
      {pred_arg.get()}, error_spec_);
}

// Test true and false computations that return a tuple of a predicate, a
// scalar, and an array.
XLA_TEST_F(ConditionalOpTest, ReturnTupleofPredicateScalarArray) {
  XlaBuilder true_builder(TestName() + ".true");
  {
    Parameter(&true_builder, 0, empty_tuple_, "tuple");
    auto true_pred = ConstantR0<bool>(&true_builder, true);
    auto true_scalar = ConstantR0<float>(&true_builder, 12.2f);
    auto true_array = ConstantR1<float>(&true_builder, {12.8f, 14.6f});
    Tuple(&true_builder, {true_pred, true_scalar, true_array});
  }
  auto true_builder_result = true_builder.Build();
  EXPECT_IS_OK(true_builder_result.status());

  XlaBuilder false_builder(TestName() + ".false");
  {
    Parameter(&false_builder, 0, empty_tuple_, "tuple");
    auto false_pred = ConstantR0<bool>(&false_builder, false);
    auto false_scalar = ConstantR0<float>(&false_builder, 25.6f);
    auto false_array = ConstantR1<float>(&false_builder, {26.4f, 32.6f});
    Tuple(&false_builder, {false_pred, false_scalar, false_array});
  }
  auto false_builder_result = false_builder.Build();
  EXPECT_IS_OK(false_builder_result.status());

  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operands = Tuple(&builder, {});
  Conditional(pred, operands, true_builder_result.ConsumeValueOrDie(), operands,
              false_builder_result.ConsumeValueOrDie());

  ComputeAndCompareTuple(
      &builder,
      *LiteralUtil::MakeTuple(
          {LiteralUtil::CreateR0<bool>(true).get(),
           LiteralUtil::CreateR0<float>(12.2f).get(),
           LiteralUtil::CreateR1<float>({12.8f, 14.6f}).get()}),
      {pred_arg.get()}, error_spec_);
}

// Test true and false computations that return a nested tuple.
XLA_TEST_F(ConditionalOpTest, ReturnNestedTuple) {
  XlaBuilder true_builder(TestName() + ".true");
  {
    Parameter(&true_builder, 0, empty_tuple_, "tuple");
    auto true_constant1 = ConstantR0<float>(&true_builder, 12.2f);
    auto true_constant2 = ConstantR1<float>(&true_builder, {12.8f, 14.6f});
    auto true_constant3 = ConstantR1<float>(&true_builder, {25.4f, 29.8f});
    auto true_constant4 = ConstantR0<float>(&true_builder, 35.6f);
    Tuple(&true_builder,
          {Tuple(&true_builder, {true_constant1, true_constant2}),
           Tuple(&true_builder, {true_constant3, true_constant4})});
  }
  auto true_builder_result = true_builder.Build();
  EXPECT_IS_OK(true_builder_result.status());

  XlaBuilder false_builder(TestName() + ".false");
  {
    Parameter(&false_builder, 0, empty_tuple_, "tuple");
    auto false_constant1 = ConstantR0<float>(&false_builder, 46.6f);
    auto false_constant2 = ConstantR1<float>(&false_builder, {54.4f, 58.4f});
    auto false_constant3 = ConstantR1<float>(&false_builder, {62.1f, 67.4f});
    auto false_constant4 = ConstantR0<float>(&false_builder, 9.3f);
    Tuple(&false_builder,
          {Tuple(&false_builder, {false_constant1, false_constant2}),
           Tuple(&false_builder, {false_constant3, false_constant4})});
  }
  auto false_builder_result = false_builder.Build();
  EXPECT_IS_OK(false_builder_result.status());

  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operands = Tuple(&builder, {});
  Conditional(pred, operands, true_builder_result.ConsumeValueOrDie(), operands,
              false_builder_result.ConsumeValueOrDie());

  ComputeAndCompareTuple(
      &builder,
      *LiteralUtil::MakeTuple(
          {LiteralUtil::MakeTuple(
               {LiteralUtil::CreateR0<float>(46.6f).get(),
                LiteralUtil::CreateR1<float>({54.4f, 58.4f}).get()})
               .get(),
           LiteralUtil::MakeTuple(
               {LiteralUtil::CreateR1<float>({62.1f, 67.4f}).get(),
                LiteralUtil::CreateR0<float>(9.3f).get()})
               .get()}),
      {pred_arg.get()}, error_spec_);
}

// Test conditional that takes in scalar operands in the form of external
// params.
XLA_TEST_F(ConditionalOpTest, ScalarOperandsFromExternalParams) {
  Shape r0bool = ShapeUtil::MakeShape(PRED, {});
  XlaBuilder builder(TestName());

  XlaOp pred, operand1, operand2;
  auto pred_arg = CreateR0Parameter<bool>(true, 0, "pred", &builder, &pred);
  auto operand1_param =
      CreateR0Parameter<float>(56.3f, 1, "operand1", &builder, &operand1);
  auto operand2_param =
      CreateR0Parameter<float>(12.7f, 2, "operand2", &builder, &operand2);
  Conditional(pred, operand1, CreateR0CeilComputation(), operand2,
              CreateR0FloorComputation());

  ComputeAndCompareR0<float>(
      &builder, 57.0f,
      {pred_arg.get(), operand1_param.get(), operand2_param.get()},
      error_spec_);
}

// Test conditional that takes in array operands in the form of external params.
XLA_TEST_F(ConditionalOpTest, ArrayOperandsFromExternalParams) {
  Shape r0bool = ShapeUtil::MakeShape(PRED, {});
  XlaBuilder builder(TestName());

  XlaOp pred, operand1, operand2;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1_param = CreateR1Parameter<float>({24.3f, 56.7f}, 1, "operand1",
                                                 &builder, &operand1);
  auto operand2_param = CreateR1Parameter<float>({10.2f, 11.6f}, 2, "operand2",
                                                 &builder, &operand2);
  Conditional(pred, operand1, CreateR1CeilComputation(), operand2,
              CreateR1FloorComputation());

  ComputeAndCompareR1<float>(
      &builder, {10.0f, 11.0f},
      {pred_arg.get(), operand1_param.get(), operand2_param.get()},
      error_spec_);
}

// Test the case where one conditional is nested within another.
XLA_TEST_F(ConditionalOpTest, NestedConditionals) {
  XlaBuilder inner_builder(TestName() + ".inner_conditional");
  {
    Shape r0bool = ShapeUtil::MakeShape(PRED, {});
    Shape tuple_shape = ShapeUtil::MakeTupleShape({r0bool, r0f32_, r0f32_});
    auto param0 = Parameter(&inner_builder, 0, tuple_shape, "param0");
    auto pred_cond = GetTupleElement(param0, 0);
    auto true_operand = GetTupleElement(param0, 1);
    auto false_operand = GetTupleElement(param0, 2);
    Conditional(pred_cond, true_operand, CreateR0CeilComputation(),
                false_operand, CreateR0FloorComputation());
  }
  auto inner_builder_result = inner_builder.Build();
  EXPECT_IS_OK(inner_builder_result.status());

  XlaBuilder builder(TestName());
  XlaOp pred1, pred2;
  auto pred1_arg = CreateR0Parameter<bool>(true, 0, "pred1", &builder, &pred1);
  auto pred2_arg = CreateR0Parameter<bool>(false, 1, "pred2", &builder, &pred2);
  auto operand1 = ConstantR0<float>(&builder, 1.1f);
  auto operand2 = ConstantR0<float>(&builder, 12.2f);
  auto operand3 = ConstantR0<float>(&builder, 43.3f);
  auto tuple_operand = Tuple(&builder, {pred2, operand1, operand2});
  Conditional(pred1, tuple_operand, inner_builder_result.ConsumeValueOrDie(),
              operand3, CreateR0IdentityComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f,
                             {pred1_arg.get(), pred2_arg.get()}, error_spec_);
}

XLA_TEST_F(ConditionalOpTest, ConditionalInNestedComputation) {
  XlaBuilder inner_builder(TestName() + ".inner_conditional");
  {
    Shape r0bool = ShapeUtil::MakeShape(PRED, {});
    Shape tuple_shape = ShapeUtil::MakeTupleShape({r0bool, r0f32_, r0f32_});
    auto param0 = Parameter(&inner_builder, 0, tuple_shape, "param0");
    auto pred_cond = GetTupleElement(param0, 0);
    auto true_operand = GetTupleElement(param0, 1);
    auto false_operand = GetTupleElement(param0, 2);
    Conditional(pred_cond, true_operand, CreateR0CeilComputation(),
                false_operand, CreateR0FloorComputation());
  }
  auto inner_builder_result = inner_builder.Build();
  EXPECT_IS_OK(inner_builder_result.status());

  XlaBuilder builder(TestName());
  XlaOp pred;
  auto pred_arg = CreateR0Parameter<bool>(false, 0, "pred", &builder, &pred);
  auto operand1 = ConstantR0<float>(&builder, 1.1f);
  auto operand2 = ConstantR0<float>(&builder, 12.2f);
  auto tuple_operand = Tuple(&builder, {pred, operand1, operand2});
  Call(&builder, inner_builder_result.ConsumeValueOrDie(), {tuple_operand});

  ComputeAndCompareR0<float>(&builder, 12.0f, {pred_arg.get()}, error_spec_);
}

// Test a mismatch in the shape of the true operand and true computation.
XLA_TEST_F(ConditionalOpTest, ShapeMismatch) {
  XlaBuilder builder(TestName());
  auto pred = ConstantR0<bool>(&builder, true);
  auto operand1 = ConstantR0<float>(&builder, 56.0f);
  auto operand2 = ConstantR0<float>(&builder, 12.0f);
  auto operands = Tuple(&builder, {operand1, operand2});
  Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
              CreateR0TupleSubComputation());

  auto result = builder.Build();
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().error_message(),
              ::testing::HasSubstr("true_operand must match the shape of the "
                                   "only parameter of true_computation"));
}

XLA_TEST_F(ConditionalOpTest, SwappedInputsInSequentialConditionals) {
  Shape tuple_shape = ShapeUtil::MakeTupleShape({r0f32_, r0f32_});
  XlaComputation swapper;
  {
    XlaBuilder builder(TestName() + ".swapper");
    auto param0 = Parameter(&builder, 0, tuple_shape, "sp0");
    auto x = GetTupleElement(param0, 0);
    auto y = GetTupleElement(param0, 1);
    Tuple(&builder, {y, x});
    swapper = builder.Build().ConsumeValueOrDie();
  }
  XlaComputation forwarder;
  {
    XlaBuilder builder(TestName() + ".forwarder");
    auto param0 = Parameter(&builder, 0, tuple_shape, "fp0");
    auto x = GetTupleElement(param0, 0);
    auto y = GetTupleElement(param0, 1);
    Tuple(&builder, {x, y});
    forwarder = builder.Build().ConsumeValueOrDie();
  }
  XlaComputation main;
  {
    XlaBuilder builder(TestName() + ".main");
    auto param0 = Parameter(&builder, 0, tuple_shape, "mp0");
    auto x = GetTupleElement(param0, 0);
    auto y = GetTupleElement(param0, 1);
    auto lt_pred = Lt(x, y);
    auto res = Conditional(lt_pred, param0, forwarder, param0, swapper);
    auto ge_pred = Ge(x, y);
    Conditional(ge_pred, res, swapper, res, forwarder);
    main = builder.Build().ConsumeValueOrDie();
  }

  auto test_swap = [&](float a, float b) {
    XlaBuilder builder(TestName());
    XlaOp x, y;
    auto x_arg = CreateR0Parameter<float>(a, 0, "x", &builder, &x);
    auto y_arg = CreateR0Parameter<float>(b, 1, "y", &builder, &y);
    auto tuple_operand = Tuple(&builder, {x, y});
    Call(&builder, main, {tuple_operand});

    ComputeAndCompareTuple(
        &builder,
        *LiteralUtil::MakeTuple({LiteralUtil::CreateR0<float>(a).get(),
                                 LiteralUtil::CreateR0<float>(b).get()}),
        {x_arg.get(), y_arg.get()}, error_spec_);
  };

  test_swap(3.11f, 9.4f);
  test_swap(11.24f, 5.55f);
}

}  // namespace
}  // namespace xla
