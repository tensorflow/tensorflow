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
    builder.Parameter(0, empty_tuple_, "tuple");
    builder.ConstantR0<float>(value);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateR0IdentityComputation() {
    XlaBuilder builder("Identity");
    builder.Parameter(0, r0f32_, "x");
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  XlaComputation CreateCeilComputation(const Shape& shape) {
    XlaBuilder builder("Ceil");
    auto param = builder.Parameter(0, shape, "param");
    builder.Ceil(param);
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
    auto param = builder.Parameter(0, shape, "param");
    builder.Floor(param);
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
    auto tuple = builder.Parameter(0, tuple_shape, "tuple");
    auto x = builder.GetTupleElement(tuple, 0);
    auto y = builder.GetTupleElement(tuple, 1);
    auto x_ceil = builder.Ceil(x);
    auto y_ceil = builder.Ceil(y);
    builder.Tuple({x_ceil, y_ceil});
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
    auto tuple = builder.Parameter(0, tuple_shape, "tuple");
    auto x = builder.GetTupleElement(tuple, 0);
    auto y = builder.GetTupleElement(tuple, 1);
    auto x_floor = builder.Floor(x);
    auto y_floor = builder.Floor(y);
    builder.Tuple({x_floor, y_floor});
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
    auto tuple = builder.Parameter(0, tuple_shape, "tuple");
    auto x = builder.GetTupleElement(tuple, 0);
    auto y = builder.GetTupleElement(tuple, 1);
    builder.Add(x, y);
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
    auto tuple = builder.Parameter(0, tuple_shape, "tuple");
    auto x = builder.GetTupleElement(tuple, 0);
    auto y = builder.GetTupleElement(tuple, 1);
    builder.Sub(x, y);
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
  auto pred = builder.ConstantR0<bool>(true);
  auto operands = builder.Tuple({});
  auto true_computation = CreateR0ConstantComputation(56.0f);
  auto false_computation = CreateR0ConstantComputation(12.0f);
  builder.Conditional(pred, operands, true_computation, operands,
                      false_computation);

  ComputeAndCompareR0<float>(&builder, 56.0f, {}, error_spec_);
}

// Test true and false computations that take in 1 parameter.
XLA_TEST_F(ConditionalOpTest, Parameters1) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.0f);
  auto operand2 = builder.ConstantR0<float>(12.0f);
  auto identity = CreateR0IdentityComputation();
  builder.Conditional(pred, operand1, identity, operand2, identity);

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with two different computations in the true and false cases
// that take in different arguments.
XLA_TEST_F(ConditionalOpTest, DiffComputationsDiffArgs) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.4f);
  auto operand2 = builder.ConstantR0<float>(12.6f);
  builder.Conditional(pred, operand1, CreateR0CeilComputation(), operand2,
                      CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with two different computations in the true and false cases
// that take in the same arguments.
XLA_TEST_F(ConditionalOpTest, DiffComputationsSameArg) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand = builder.ConstantR0<float>(12.6f);
  builder.Conditional(pred, operand, CreateR0CeilComputation(), operand,
                      CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with the same computation in the true and false cases but
// take in different arguments.
XLA_TEST_F(ConditionalOpTest, SameComputationDiffArgs) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.4f);
  auto operand2 = builder.ConstantR0<float>(12.6f);
  auto floor = CreateR0FloorComputation();
  builder.Conditional(pred, operand1, floor, operand2, floor);

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with the same computation in the true and false cases that
// take in the same arguments.
XLA_TEST_F(ConditionalOpTest, SameComputationSameArg) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand = builder.ConstantR0<float>(12.6f);
  auto floor = CreateR0FloorComputation();
  builder.Conditional(pred, operand, floor, operand, floor);

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with different instances of the same computation in the true
// and false cases.
XLA_TEST_F(ConditionalOpTest, SameComputationDiffInstances) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.4f);
  auto operand2 = builder.ConstantR0<float>(12.6f);
  builder.Conditional(pred, operand1, CreateR0FloorComputation(), operand2,
                      CreateR0FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test the case when a call invokes a computation that contains a conditional.
XLA_TEST_F(ConditionalOpTest, ConditionalWithCall) {
  Shape r0bool = ShapeUtil::MakeShape(PRED, {});
  XlaBuilder inner_builder(TestName() + ".inner_conditional");
  auto pred_cond = inner_builder.Parameter(0, r0bool, "param0");
  auto true_operand = inner_builder.Parameter(1, r0f32_, "param1");
  auto false_operand = inner_builder.Parameter(2, r0f32_, "param2");
  inner_builder.Conditional(pred_cond, true_operand, CreateR0CeilComputation(),
                            false_operand, CreateR0FloorComputation());
  auto inner_builder_result = inner_builder.Build();

  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.4f);
  auto operand2 = builder.ConstantR0<float>(12.6f);
  builder.Call(inner_builder_result.ConsumeValueOrDie(),
               {pred, operand1, operand2});

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test true and false computations that take in 2 parameters and predicate is
// true.
XLA_TEST_F(ConditionalOpTest, Parameters2TrueBranch) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto operand1 = builder.ConstantR0<float>(56.0f);
  auto operand2 = builder.ConstantR0<float>(12.0f);
  auto operands = builder.Tuple({operand1, operand2});
  builder.Conditional(pred, operands, CreateR0TupleAddComputation(), operands,
                      CreateR0TupleSubComputation());

  ComputeAndCompareR0<float>(&builder, 68.0f, {}, error_spec_);
}

// Test true and false computations that take in 2 parameters and predicate is
// false.
XLA_TEST_F(ConditionalOpTest, Parameters2FalseBranch) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.0f);
  auto operand2 = builder.ConstantR0<float>(12.0f);
  auto operands = builder.Tuple({operand1, operand2});
  builder.Conditional(pred, operands, CreateR0TupleAddComputation(), operands,
                      CreateR0TupleSubComputation());

  ComputeAndCompareR0<float>(&builder, 44.0f, {}, error_spec_);
}

// Test true and false computations that take in 2 array parameters and
// predicate is true.
XLA_TEST_F(ConditionalOpTest, Parameters2ArrayTrueBranch) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto operand1 = builder.ConstantR1<float>({24.0f, 56.0f});
  auto operand2 = builder.ConstantR1<float>({10.0f, 11.0f});
  auto operands = builder.Tuple({operand1, operand2});
  builder.Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
                      CreateR1TupleSubComputation());

  ComputeAndCompareR1<float>(&builder, {34.0f, 67.0f}, {}, error_spec_);
}

// Test true and false computations that take in 2 array parameters and
// predicate is false.
XLA_TEST_F(ConditionalOpTest, Parameters2ArrayFalseBranch) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR1<float>({24.0f, 56.0f});
  auto operand2 = builder.ConstantR1<float>({10.0f, 11.0f});
  auto operands = builder.Tuple({operand1, operand2});
  builder.Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
                      CreateR1TupleSubComputation());

  ComputeAndCompareR1<float>(&builder, {14.0f, 45.0f}, {}, error_spec_);
}

// Test true and false computations that return a tuple of scalars.
XLA_TEST_F(ConditionalOpTest, ReturnTupleOfScalars) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operands = builder.Tuple(
      {builder.ConstantR0<float>(12.2f), builder.ConstantR0<float>(25.6f)});
  builder.Conditional(pred, operands, CreateR0TupleCeilComputation(), operands,
                      CreateR0TupleFloorComputation());

  ComputeAndCompareTuple(
      &builder,
      *Literal::MakeTuple({Literal::CreateR0<float>(12.0f).get(),
                           Literal::CreateR0<float>(25.0f).get()}),
      {}, error_spec_);
}

// Test true and false computations that return a tuple of arrays.
XLA_TEST_F(ConditionalOpTest, ReturnTupleOfArrays) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto operands = builder.Tuple({builder.ConstantR1<float>({12.2f, 15.8f}),
                                 builder.ConstantR1<float>({25.6f, 29.2f})});
  builder.Conditional(pred, operands, CreateR1TupleCeilComputation(), operands,
                      CreateR1TupleFloorComputation());

  ComputeAndCompareTuple(
      &builder,
      *Literal::MakeTuple({Literal::CreateR1<float>({13.0f, 16.0f}).get(),
                           Literal::CreateR1<float>({26.0f, 30.0f}).get()}),
      {}, error_spec_);
}

// Test true and false computations that return a tuple of a predicate, a
// scalar, and an array.
XLA_TEST_F(ConditionalOpTest, ReturnTupleofPredicateScalarArray) {
  XlaBuilder true_builder(TestName() + ".true");
  {
    true_builder.Parameter(0, empty_tuple_, "tuple");
    auto true_pred = true_builder.ConstantR0<bool>(true);
    auto true_scalar = true_builder.ConstantR0<float>(12.2f);
    auto true_array = true_builder.ConstantR1<float>({12.8f, 14.6f});
    true_builder.Tuple({true_pred, true_scalar, true_array});
  }
  auto true_builder_result = true_builder.Build();
  EXPECT_IS_OK(true_builder_result.status());

  XlaBuilder false_builder(TestName() + ".false");
  {
    false_builder.Parameter(0, empty_tuple_, "tuple");
    auto false_pred = false_builder.ConstantR0<bool>(false);
    auto false_scalar = false_builder.ConstantR0<float>(25.6f);
    auto false_array = false_builder.ConstantR1<float>({26.4f, 32.6f});
    false_builder.Tuple({false_pred, false_scalar, false_array});
  }
  auto false_builder_result = false_builder.Build();
  EXPECT_IS_OK(false_builder_result.status());

  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto operands = builder.Tuple({});
  builder.Conditional(pred, operands, true_builder_result.ConsumeValueOrDie(),
                      operands, false_builder_result.ConsumeValueOrDie());

  ComputeAndCompareTuple(
      &builder,
      *Literal::MakeTuple({Literal::CreateR0<bool>(true).get(),
                           Literal::CreateR0<float>(12.2f).get(),
                           Literal::CreateR1<float>({12.8f, 14.6f}).get()}),
      {}, error_spec_);
}

// Test true and false computations that return a nested tuple.
XLA_TEST_F(ConditionalOpTest, ReturnNestedTuple) {
  XlaBuilder true_builder(TestName() + ".true");
  {
    true_builder.Parameter(0, empty_tuple_, "tuple");
    auto true_constant1 = true_builder.ConstantR0<float>(12.2f);
    auto true_constant2 = true_builder.ConstantR1<float>({12.8f, 14.6f});
    auto true_constant3 = true_builder.ConstantR1<float>({25.4f, 29.8f});
    auto true_constant4 = true_builder.ConstantR0<float>(35.6f);
    true_builder.Tuple({true_builder.Tuple({true_constant1, true_constant2}),
                        true_builder.Tuple({true_constant3, true_constant4})});
  }
  auto true_builder_result = true_builder.Build();
  EXPECT_IS_OK(true_builder_result.status());

  XlaBuilder false_builder(TestName() + ".false");
  {
    false_builder.Parameter(0, empty_tuple_, "tuple");
    auto false_constant1 = false_builder.ConstantR0<float>(46.6f);
    auto false_constant2 = false_builder.ConstantR1<float>({54.4f, 58.4f});
    auto false_constant3 = false_builder.ConstantR1<float>({62.1f, 67.4f});
    auto false_constant4 = false_builder.ConstantR0<float>(9.3f);
    false_builder.Tuple(
        {false_builder.Tuple({false_constant1, false_constant2}),
         false_builder.Tuple({false_constant3, false_constant4})});
  }
  auto false_builder_result = false_builder.Build();
  EXPECT_IS_OK(false_builder_result.status());

  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operands = builder.Tuple({});
  builder.Conditional(pred, operands, true_builder_result.ConsumeValueOrDie(),
                      operands, false_builder_result.ConsumeValueOrDie());

  ComputeAndCompareTuple(
      &builder,
      *Literal::MakeTuple(
          {Literal::MakeTuple({Literal::CreateR0<float>(46.6f).get(),
                               Literal::CreateR1<float>({54.4f, 58.4f}).get()})
               .get(),
           Literal::MakeTuple({Literal::CreateR1<float>({62.1f, 67.4f}).get(),
                               Literal::CreateR0<float>(9.3f).get()})
               .get()}),
      {}, error_spec_);
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
  builder.Conditional(pred, operand1, CreateR0CeilComputation(), operand2,
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
  builder.Conditional(pred, operand1, CreateR1CeilComputation(), operand2,
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
    auto param0 = inner_builder.Parameter(0, tuple_shape, "param0");
    auto pred_cond = inner_builder.GetTupleElement(param0, 0);
    auto true_operand = inner_builder.GetTupleElement(param0, 1);
    auto false_operand = inner_builder.GetTupleElement(param0, 2);
    inner_builder.Conditional(pred_cond, true_operand,
                              CreateR0CeilComputation(), false_operand,
                              CreateR0FloorComputation());
  }
  auto inner_builder_result = inner_builder.Build();
  EXPECT_IS_OK(inner_builder_result.status());

  XlaBuilder builder(TestName());
  auto pred1 = builder.ConstantR0<bool>(true);
  auto pred2 = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(1.1f);
  auto operand2 = builder.ConstantR0<float>(12.2f);
  auto operand3 = builder.ConstantR0<float>(43.3f);
  auto tuple_operand = builder.Tuple({pred2, operand1, operand2});
  builder.Conditional(pred1, tuple_operand,
                      inner_builder_result.ConsumeValueOrDie(), operand3,
                      CreateR0IdentityComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

XLA_TEST_F(ConditionalOpTest, ConditionalInNestedComputation) {
  XlaBuilder inner_builder(TestName() + ".inner_conditional");
  {
    Shape r0bool = ShapeUtil::MakeShape(PRED, {});
    Shape tuple_shape = ShapeUtil::MakeTupleShape({r0bool, r0f32_, r0f32_});
    auto param0 = inner_builder.Parameter(0, tuple_shape, "param0");
    auto pred_cond = inner_builder.GetTupleElement(param0, 0);
    auto true_operand = inner_builder.GetTupleElement(param0, 1);
    auto false_operand = inner_builder.GetTupleElement(param0, 2);
    inner_builder.Conditional(pred_cond, true_operand,
                              CreateR0CeilComputation(), false_operand,
                              CreateR0FloorComputation());
  }
  auto inner_builder_result = inner_builder.Build();
  EXPECT_IS_OK(inner_builder_result.status());

  XlaBuilder builder(TestName());
  auto pred2 = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(1.1f);
  auto operand2 = builder.ConstantR0<float>(12.2f);
  auto tuple_operand = builder.Tuple({pred2, operand1, operand2});
  builder.Call(inner_builder_result.ConsumeValueOrDie(), {tuple_operand});

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test a mismatch in the shape of the true operand and true computation.
XLA_TEST_F(ConditionalOpTest, ShapeMismatch) {
  XlaBuilder builder(TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto operand1 = builder.ConstantR0<float>(56.0f);
  auto operand2 = builder.ConstantR0<float>(12.0f);
  auto operands = builder.Tuple({operand1, operand2});
  builder.Conditional(pred, operands, CreateR1TupleAddComputation(), operands,
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
    auto param0 = builder.Parameter(0, tuple_shape, "sp0");
    auto x = builder.GetTupleElement(param0, 0);
    auto y = builder.GetTupleElement(param0, 1);
    builder.Tuple({y, x});
    swapper = builder.Build().ConsumeValueOrDie();
  }
  XlaComputation forwarder;
  {
    XlaBuilder builder(TestName() + ".forwarder");
    auto param0 = builder.Parameter(0, tuple_shape, "fp0");
    auto x = builder.GetTupleElement(param0, 0);
    auto y = builder.GetTupleElement(param0, 1);
    builder.Tuple({x, y});
    forwarder = builder.Build().ConsumeValueOrDie();
  }
  XlaComputation main;
  {
    XlaBuilder builder(TestName() + ".main");
    auto param0 = builder.Parameter(0, tuple_shape, "mp0");
    auto x = builder.GetTupleElement(param0, 0);
    auto y = builder.GetTupleElement(param0, 1);
    auto lt_pred = builder.Lt(x, y);
    auto res = builder.Conditional(lt_pred, param0, forwarder, param0, swapper);
    auto ge_pred = builder.Ge(x, y);
    builder.Conditional(ge_pred, res, swapper, res, forwarder);
    main = builder.Build().ConsumeValueOrDie();
  }

  auto test_swap = [&](float a, float b) {
    XlaBuilder builder(TestName());
    auto x = builder.ConstantR0<float>(a);
    auto y = builder.ConstantR0<float>(b);
    auto tuple_operand = builder.Tuple({x, y});
    builder.Call(main, {tuple_operand});

    ComputeAndCompareTuple(
        &builder,
        *Literal::MakeTuple({Literal::CreateR0<float>(a).get(),
                             Literal::CreateR0<float>(b).get()}),
        {}, error_spec_);
  };

  test_swap(3.11f, 9.4f);
  test_swap(11.24f, 5.55f);
}

}  // namespace
}  // namespace xla
