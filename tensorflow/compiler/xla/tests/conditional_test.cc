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

#include "tensorflow/compiler/xla/client/computation_builder.h"
#include "tensorflow/compiler/xla/tests/client_library_test_base.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/compiler/xla/tests/test_macros.h"

namespace xla {
namespace {

class ConditionalOpTest : public ClientLibraryTestBase {
 protected:
  Computation CreateR0F32ConstantComputation(float value) {
    ComputationBuilder builder(client_, "Constant");
    builder.Parameter(0, empty_tuple_, "tuple");
    builder.ConstantR0<float>(value);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Computation CreateR0F32IdentityComputation() {
    ComputationBuilder builder(client_, "Identity");
    builder.Parameter(0, r0f32_, "x");
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Computation CreateR0F32CeilComputation() {
    ComputationBuilder builder(client_, "Ceil");
    auto param = builder.Parameter(0, r0f32_, "param");
    builder.Ceil(param);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Computation CreateR0F32FloorComputation() {
    ComputationBuilder builder(client_, "Ceil");
    auto param = builder.Parameter(0, r0f32_, "param");
    builder.Floor(param);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Computation CreateAddTupleComputation(const string& computation_name,
                                        const Shape& tuple_shape) {
    ComputationBuilder builder(client_, computation_name);
    auto tuple = builder.Parameter(0, tuple_shape, "tuple");
    auto x = builder.GetTupleElement(tuple, 0);
    auto y = builder.GetTupleElement(tuple, 1);
    builder.Add(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Computation CreateAddR0Computation() {
    return CreateAddTupleComputation("AddR0", tuple_2_r0f32_);
  }

  Computation CreateAddR1Computation() {
    return CreateAddTupleComputation("AddR1", tuple_2_r1s2f32_);
  }

  Computation CreateSubTupleComputation(const string& computation_name,
                                        const Shape& tuple_shape) {
    ComputationBuilder builder(client_, computation_name);
    auto tuple = builder.Parameter(0, tuple_shape, "tuple");
    auto x = builder.GetTupleElement(tuple, 0);
    auto y = builder.GetTupleElement(tuple, 1);
    builder.Sub(x, y);
    auto build_status = builder.Build();
    EXPECT_IS_OK(build_status.status());
    return build_status.ConsumeValueOrDie();
  }

  Computation CreateSubR0Computation() {
    return CreateSubTupleComputation("SubR0", tuple_2_r0f32_);
  }

  Computation CreateSubR1Computation() {
    return CreateSubTupleComputation("SubR1", tuple_2_r1s2f32_);
  }

  Shape r0f32_ = ShapeUtil::MakeShape(F32, {});
  Shape tuple_2_r0f32_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {}), ShapeUtil::MakeShape(F32, {})});
  Shape tuple_2_r1s2f32_ = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2}), ShapeUtil::MakeShape(F32, {2})});
  Shape empty_tuple_ = ShapeUtil::MakeTupleShape({});
  ErrorSpec error_spec_{0.001};
};

// Test true and false computations that do not take any parameters.
XLA_TEST_F(ConditionalOpTest, Parameters0) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto operands = builder.Tuple({});
  auto true_computation = CreateR0F32ConstantComputation(56.0f);
  auto false_computation = CreateR0F32ConstantComputation(12.0f);
  auto result = builder.Conditional(pred, operands, true_computation, operands,
                                    false_computation);

  ComputeAndCompareR0<float>(&builder, 56.0f, {}, error_spec_);
}

// Test true and false computations that take in 1 parameter.
XLA_TEST_F(ConditionalOpTest, Parameters1) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.0f);
  auto operand2 = builder.ConstantR0<float>(12.0f);
  auto identity = CreateR0F32IdentityComputation();
  auto result =
      builder.Conditional(pred, operand1, identity, operand2, identity);

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with two different computations in the true and false cases
// that take in different arguments.
XLA_TEST_F(ConditionalOpTest, DiffComputationsDiffArgs) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.4f);
  auto operand2 = builder.ConstantR0<float>(12.6f);
  auto result =
      builder.Conditional(pred, operand1, CreateR0F32CeilComputation(),
                          operand2, CreateR0F32FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with two different computations in the true and false cases
// that take in the same arguments.
XLA_TEST_F(ConditionalOpTest, DiffComputationsSameArg) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand = builder.ConstantR0<float>(12.6f);
  auto result = builder.Conditional(pred, operand, CreateR0F32CeilComputation(),
                                    operand, CreateR0F32FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with the same computation in the true and false cases but
// take in different arguments.
XLA_TEST_F(ConditionalOpTest, SameComputationDiffArgs) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.4f);
  auto operand2 = builder.ConstantR0<float>(12.6f);
  auto floor = CreateR0F32FloorComputation();
  auto result = builder.Conditional(pred, operand1, floor, operand2, floor);

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with the same computation in the true and false cases that
// take in the same arguments.
XLA_TEST_F(ConditionalOpTest, SameComputationSameArg) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand = builder.ConstantR0<float>(12.6f);
  auto floor = CreateR0F32FloorComputation();
  auto result = builder.Conditional(pred, operand, floor, operand, floor);

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test conditional with different instances of the same computation in the true
// and false cases.
XLA_TEST_F(ConditionalOpTest, SameComputationDiffInstances) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.4f);
  auto operand2 = builder.ConstantR0<float>(12.6f);
  auto result =
      builder.Conditional(pred, operand1, CreateR0F32FloorComputation(),
                          operand2, CreateR0F32FloorComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test the case when a call invokes a computation that contains a conditional.
XLA_TEST_F(ConditionalOpTest, ConditionalWithCall) {
  Shape r0bool = ShapeUtil::MakeShape(PRED, {});
  ComputationBuilder inner_builder(client_, TestName() + ".inner_conditional");
  auto pred_cond = inner_builder.Parameter(0, r0bool, "param0");
  auto true_operand = inner_builder.Parameter(1, r0f32_, "param1");
  auto false_operand = inner_builder.Parameter(2, r0f32_, "param2");
  inner_builder.Conditional(pred_cond, true_operand,
                            CreateR0F32CeilComputation(), false_operand,
                            CreateR0F32FloorComputation());
  auto inner_builder_result = inner_builder.Build();

  ComputationBuilder builder(client_, TestName());
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
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto operand1 = builder.ConstantR0<float>(56.0f);
  auto operand2 = builder.ConstantR0<float>(12.0f);
  auto operands = builder.Tuple({operand1, operand2});
  auto result = builder.Conditional(pred, operands, CreateAddR0Computation(),
                                    operands, CreateSubR0Computation());

  ComputeAndCompareR0<float>(&builder, 68.0f, {}, error_spec_);
}

// Test true and false computations that take in 2 parameters and predicate is
// false.
XLA_TEST_F(ConditionalOpTest, Parameters2FalseBranch) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(56.0f);
  auto operand2 = builder.ConstantR0<float>(12.0f);
  auto operands = builder.Tuple({operand1, operand2});
  auto result = builder.Conditional(pred, operands, CreateAddR0Computation(),
                                    operands, CreateSubR0Computation());

  ComputeAndCompareR0<float>(&builder, 44.0f, {}, error_spec_);
}

// Test true and false computations that take in 2 array parameters and
// predicate is true.
XLA_TEST_F(ConditionalOpTest, Parameters2ArrayTrueBranch) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto operand1 = builder.ConstantR1<float>({24.0f, 56.0f});
  auto operand2 = builder.ConstantR1<float>({10.0f, 11.0f});
  auto operands = builder.Tuple({operand1, operand2});
  auto result = builder.Conditional(pred, operands, CreateAddR1Computation(),
                                    operands, CreateSubR1Computation());

  ComputeAndCompareR1<float>(&builder, {34.0f, 67.0f}, {}, error_spec_);
}

// Test true and false computations that take in 2 array parameters and
// predicate is false.
XLA_TEST_F(ConditionalOpTest, Parameters2ArrayFalseBranch) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR1<float>({24.0f, 56.0f});
  auto operand2 = builder.ConstantR1<float>({10.0f, 11.0f});
  auto operands = builder.Tuple({operand1, operand2});
  auto result = builder.Conditional(pred, operands, CreateAddR1Computation(),
                                    operands, CreateSubR1Computation());

  ComputeAndCompareR1<float>(&builder, {14.0f, 45.0f}, {}, error_spec_);
}

// Test the case where one conditional is nested within another.
XLA_TEST_F(ConditionalOpTest, NestedConditionals) {
  Shape r0bool = ShapeUtil::MakeShape(PRED, {});
  Shape tuple_shape = ShapeUtil::MakeTupleShape({r0bool, r0f32_, r0f32_});
  ComputationBuilder inner_builder(client_, TestName() + ".inner_conditional");
  auto param0 = inner_builder.Parameter(0, tuple_shape, "param0");
  auto pred_cond = inner_builder.GetTupleElement(param0, 0);
  auto true_operand = inner_builder.GetTupleElement(param0, 1);
  auto false_operand = inner_builder.GetTupleElement(param0, 2);
  inner_builder.Conditional(pred_cond, true_operand,
                            CreateR0F32CeilComputation(), false_operand,
                            CreateR0F32FloorComputation());
  auto inner_builder_result = inner_builder.Build();

  ComputationBuilder builder(client_, TestName());
  auto pred1 = builder.ConstantR0<bool>(true);
  auto pred2 = builder.ConstantR0<bool>(false);
  auto operand1 = builder.ConstantR0<float>(1.1f);
  auto operand2 = builder.ConstantR0<float>(12.2f);
  auto operand3 = builder.ConstantR0<float>(43.3f);
  auto tuple_operand = builder.Tuple({pred2, operand1, operand2});
  builder.Conditional(pred1, tuple_operand,
                      inner_builder_result.ConsumeValueOrDie(), operand3,
                      CreateR0F32IdentityComputation());

  ComputeAndCompareR0<float>(&builder, 12.0f, {}, error_spec_);
}

// Test a mismatch in the shape of the true operand and true computation.
XLA_TEST_F(ConditionalOpTest, ShapeMismatch) {
  ComputationBuilder builder(client_, TestName());
  auto pred = builder.ConstantR0<bool>(true);
  auto operand1 = builder.ConstantR0<float>(56.0f);
  auto operand2 = builder.ConstantR0<float>(12.0f);
  auto operands = builder.Tuple({operand1, operand2});
  builder.Conditional(pred, operands, CreateAddR1Computation(), operands,
                      CreateSubR0Computation());

  auto result = builder.Build();
  EXPECT_FALSE(result.ok());
  EXPECT_THAT(result.status().error_message(),
              ::testing::HasSubstr("true_operand must match the shape of the "
                                   "only parameter of true_computation"));
}

}  // namespace
}  // namespace xla
