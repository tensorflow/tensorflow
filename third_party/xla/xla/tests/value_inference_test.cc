/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/client/value_inference.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/types/span.h"
#include "xla/client/client_library.h"
#include "xla/client/global_data.h"
#include "xla/client/lib/arithmetic.h"
#include "xla/client/lib/prng.h"
#include "xla/client/xla_builder.h"
#include "xla/client/xla_computation.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/test.h"
#include "xla/tests/literal_test_util.h"
#include "xla/tests/test_macros.h"
#include "xla/tests/test_utils.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace {

class ValueInferenceTest : public ::testing::Test {
 public:
  std::string TestName() const {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }
};

class DynamismInferenceTest : public ValueInferenceTest {
 public:
  explicit DynamismInferenceTest(se::Platform* platform = nullptr)
      : platform_(platform) {}

  absl::StatusOr<Literal> ComputeDynamismLiteral(
      XlaOp operand, XlaBuilder* builder, Layout* output_layout = nullptr) {
    TF_RETURN_IF_ERROR(builder->first_error());
    ValueInference value_inference(builder);
    TF_ASSIGN_OR_RETURN(auto literal_slice,
                        value_inference.AnalyzeIsDynamic(operand));
    return literal_slice.Clone();
  }

  absl::StatusOr<bool> ComputeDynamismScalar(XlaOp operand, XlaBuilder* builder,
                                             ShapeIndex index = {}) {
    TF_ASSIGN_OR_RETURN(auto literal,
                        ComputeDynamismLiteral(operand, builder, nullptr));
    return literal.Get<bool>({}, index);
  }

  se::Platform* platform_;
};

TEST_F(DynamismInferenceTest, ScalarInt32Literal) {
  XlaBuilder b(TestName());
  auto computation = ConstantR0<int32_t>(&b, 42);

  auto value = ComputeDynamismScalar(computation, &b);
  ASSERT_TRUE(value.ok()) << value.status();
  // A constant is not dynamic.
  EXPECT_EQ(value.value(), false);
}

TEST_F(DynamismInferenceTest, Iota) {
  // The output of iota are consistened static.
  XlaBuilder b(TestName());
  auto computation = Iota(&b, S32, 2);
  // Iota is not dynamic.
  EXPECT_FALSE(ComputeDynamismLiteral(computation, &b).value().Get<bool>({0}));
}

TEST_F(DynamismInferenceTest, TupleSimple) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32_t>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto tuple = Tuple(&b, {c, p});
  EXPECT_EQ(ComputeDynamismScalar(tuple, &b, {0}).value(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple, &b, {1}).value(), true);
}

TEST_F(DynamismInferenceTest, TupleGteKeepsDynamism) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32_t>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto tuple = Tuple(&b, {c, p});
  auto gte0 = GetTupleElement(tuple, 0);
  auto gte1 = GetTupleElement(tuple, 1);
  auto tuple_2 = Tuple(&b, {gte0, gte1});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).value(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).value(), true);
}

TEST_F(DynamismInferenceTest, PredValueUsedTwice) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32_t>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");
  auto pred = Eq(c, p);
  auto result = Select(pred, p, c);
  EXPECT_EQ(ComputeDynamismScalar(result, &b, {}).value(), true);
}

TEST_F(DynamismInferenceTest, ReduceUsedTwice) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32_t>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2}), "p0");
  auto zero = ConstantR0<int32_t>(&b, 0);
  XlaComputation add_s32 = CreateScalarAddComputation(S32, &b);
  auto reduce = Reduce(p, zero, add_s32, {0});
  auto pred = Eq(c, reduce);
  auto result = Select(pred, reduce, c);
  EXPECT_EQ(ComputeDynamismScalar(result, &b, {}).value(), true);
}

TEST_F(DynamismInferenceTest, VariadicReduce) {
  XlaBuilder b(TestName());
  auto c = ConstantR2<int32_t>(&b, {{0, 0}});
  auto p = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {1, 2}), "p0");
  // half_dynamic[0] is static, half_dynamic[0] is dynamic.
  auto half_dynamic = ConcatInDim(&b, {c, p}, 0);
  XlaBuilder reduce_add("reduce_add");
  auto p0 = Parameter(&reduce_add, 0, ShapeUtil::MakeScalarShape(S32), "p");
  auto p1 = Parameter(&reduce_add, 1, ShapeUtil::MakeScalarShape(S32), "p");
  auto p2 = Parameter(&reduce_add, 2, ShapeUtil::MakeScalarShape(S32), "p");
  auto p3 = Parameter(&reduce_add, 3, ShapeUtil::MakeScalarShape(S32), "p");
  auto reduce_result = p0;
  reduce_result = Add(reduce_result, p1);
  reduce_result = Add(reduce_result, p2);
  reduce_result = Add(reduce_result, p3);
  Tuple(&reduce_add, {reduce_result, reduce_result});
  auto init = ConstantR0<int32_t>(&b, 0);
  auto variadic_reduce = Reduce(&b, {half_dynamic, half_dynamic}, {init, init},
                                reduce_add.Build().value(), {1});
  auto result = GetTupleElement(variadic_reduce, 0);

  // result[0] should be static; result[1] should be dynamic.
  EXPECT_FALSE(ComputeDynamismLiteral(result, &b).value().Get<bool>({0}));
  EXPECT_TRUE(ComputeDynamismLiteral(result, &b).value().Get<bool>({1}));
}

TEST_F(DynamismInferenceTest, DynamicSelectorWithMixedValues) {
  XlaBuilder b(TestName());
  auto constant_pred = ConstantR1<bool>(&b, {true});
  auto dynamic_pred = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {1}), "p0");
  auto concat = ConcatInDim(&b, {constant_pred, dynamic_pred}, 0);
  auto constant_values = ConstantR1<bool>(&b, {true, true});
  auto result = Select(concat, constant_values, constant_values);
  // First result is static (selector is constant, both values are constant).
  // Iota is not dynamic.
  EXPECT_FALSE(ComputeDynamismLiteral(result, &b).value().Get<bool>({0}));
  // Second result is dynamic (selector is dynamic).
  EXPECT_TRUE(ComputeDynamismLiteral(result, &b).value().Get<bool>({1}));
}

TEST_F(DynamismInferenceTest, ConcatSliceReshapeKeepsDynamism) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32_t>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto concat = ConcatScalars(&b, {c, p});
  auto slice0 = SliceInDim(concat, 0, 1, 1, 0);
  auto reshape0 = Reshape(slice0, {});
  auto slice1 = SliceInDim(concat, 1, 2, 1, 0);
  auto reshape1 = Reshape(slice1, {});
  auto tuple_2 = Tuple(&b, {reshape0, reshape1});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).value(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).value(), true);
}

TEST_F(DynamismInferenceTest, ParameterIsDynamic) {
  XlaBuilder b(TestName());
  auto computation = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto value = ComputeDynamismScalar(computation, &b);
  ASSERT_TRUE(value.ok()) << value.status();
  // A parameter is considered dynamic.
  EXPECT_EQ(value.value(), true);
}

TEST_F(DynamismInferenceTest, UnaryOpKeepsDynamism) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32_t>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  auto neg0 = Neg(c);
  auto neg1 = Neg(p);
  auto tuple_2 = Tuple(&b, {neg0, neg1});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).value(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).value(), true);
}

TEST_F(DynamismInferenceTest, ParameterWithToken) {
  // Test that token shape can be handled in a parameter.
  XlaBuilder b(TestName());
  auto p =
      Parameter(&b, 0,
                ShapeUtil::MakeTupleShape({ShapeUtil::MakeTokenShape(),
                                           ShapeUtil::MakeScalarShape(S32)}),
                "p0");
  EXPECT_EQ(ComputeDynamismScalar(p, &b, {0}).value(), true);
  EXPECT_EQ(ComputeDynamismScalar(p, &b, {1}).value(), true);
}

TEST_F(DynamismInferenceTest, BinaryOpsOrsDynamism) {
  XlaBuilder b(TestName());
  auto c = ConstantR0<int32_t>(&b, 42);
  auto p = Parameter(&b, 0, ShapeUtil::MakeScalarShape(S32), "p0");

  // Static value + static value = static
  auto add1 = Add(c, c);
  // Dynamic value + dynamic value = dynamic
  auto add2 = Add(p, c);
  auto tuple_2 = Tuple(&b, {add1, add2});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).value(), false);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).value(), true);
}

TEST_F(DynamismInferenceTest, GetDimensionSize) {
  XlaBuilder b(TestName());
  // param = Param([<=2, 3])
  // get_dimension_size(param, 0) is dynamic
  // get_dimension_size(param, 1) is static
  auto p =
      Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2, 3}, {true, false}), "p0");

  auto gds0 = GetDimensionSize(p, 0);
  auto gds1 = GetDimensionSize(p, 1);
  auto tuple_2 = Tuple(&b, {gds0, gds1});
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {0}).value(), true);
  EXPECT_EQ(ComputeDynamismScalar(tuple_2, &b, {1}).value(), false);
}

TEST_F(DynamismInferenceTest, DynamicSliceWithConstantOperands) {
  XlaBuilder b(TestName());

  auto constant = ConstantR1<int32_t>(&b, {0, 1, 2, 3});
  auto slice_start = ConstantR0(&b, 1);
  auto dynamic_slice = DynamicSlice(constant, {slice_start}, {1});
  EXPECT_FALSE(
      ComputeDynamismLiteral(dynamic_slice, &b).value().Get<bool>({0}));
}

TEST_F(DynamismInferenceTest, GatherWithCommonParent) {
  XlaBuilder b(TestName());
  // Test the analysis on a gather where first operand and second operand have
  // common parents.
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});

  auto operand1 = Parameter(&b, 0, indices_shape, "p1");
  auto operand2 = Parameter(&b, 1, indices_shape, "p2");
  auto indices = Sub(operand1, operand2);
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_offset_dims(1);
  dim_numbers.add_start_index_map(0);
  dim_numbers.set_index_vector_dim(1);
  auto gather = Gather(operand1, indices, dim_numbers, {1});
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  EXPECT_TRUE(ComputeDynamismLiteral(gather, &b).value().Get<bool>({0, 0}));
}

TEST_F(DynamismInferenceTest, GatherWithConstantParent) {
  XlaBuilder b(TestName());
  // Test the analysis on a gather.
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});
  auto data_operand = ConstantR1<int32_t>(&b, {1, 2});
  auto indices = ConstantR1<int32_t>(&b, {1, 2});
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_offset_dims(1);
  dim_numbers.add_start_index_map(0);
  dim_numbers.set_index_vector_dim(1);
  auto gather = Gather(data_operand, indices, dim_numbers, {1});
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  // Everything is constant, result is also contant.
  EXPECT_FALSE(ComputeDynamismLiteral(gather, &b).value().Get<bool>({0, 0}));
}

TEST_F(DynamismInferenceTest, GatherWithSharedConstantParent) {
  XlaBuilder b(TestName());
  // Test the analysis on a gather.
  Shape indices_shape = ShapeUtil::MakeShape(S32, {2});
  auto operand1 = ConstantR1<int32_t>(&b, {1, 2});
  auto operand2 = ConstantR1<int32_t>(&b, {1, 2});
  auto indices = Sub(operand1, operand2);
  GatherDimensionNumbers dim_numbers;
  dim_numbers.add_offset_dims(1);
  dim_numbers.add_start_index_map(0);
  dim_numbers.set_index_vector_dim(1);
  auto gather = Gather(operand1, indices, dim_numbers, {1});
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  // Everything is constant, result is also contant.
  EXPECT_FALSE(ComputeDynamismLiteral(gather, &b).value().Get<bool>({0, 0}));
}

TEST_F(DynamismInferenceTest, InferThroughPad) {
  XlaBuilder b(TestName());
  // Test the analysis on a gather.
  auto operand1 = ConstantR1<int32_t>(&b, {1, 2});
  auto parameter = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {}), "p0");
  PaddingConfig padding_config;
  padding_config.add_dimensions()->set_edge_padding_high(1);
  // After pad the value is [constant, constant, parameter].
  auto pad = Pad(operand1, parameter, padding_config);
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  // Everything is constant, result is also contant.
  EXPECT_FALSE(ComputeDynamismLiteral(pad, &b).value().Get<bool>({0}));
  EXPECT_FALSE(ComputeDynamismLiteral(pad, &b).value().Get<bool>({1}));
  EXPECT_TRUE(ComputeDynamismLiteral(pad, &b).value().Get<bool>({2}));
}

TEST_F(DynamismInferenceTest, InferThroughConditionalBranchesAreSame) {
  // The result of following conditional is static.
  // pred = .. # a dynamic value
  // if (pred) {
  //  return (1) # both branches return the same value
  // } else {
  //  return (1)
  // }
  //

  auto s32_shape = ShapeUtil::MakeShape(S32, {});
  auto cond_shape = ShapeUtil::MakeTupleShape({s32_shape});
  XlaBuilder true_builder("true");
  Parameter(&true_builder, 0, s32_shape, "cond_param");
  Tuple(&true_builder, {ConstantR0<int32_t>(&true_builder, 1)});
  auto true_computation = true_builder.Build().value();

  XlaBuilder false_builder("false");
  Parameter(&false_builder, 0, s32_shape, "cond_param");
  Tuple(&false_builder, {ConstantR0<int32_t>(&false_builder, 1)});
  auto false_computation = false_builder.Build().value();

  XlaBuilder b(TestName());
  auto parameter = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "p0");
  auto constant = ConstantR0<int32_t>(&b, 0);
  auto cond = Conditional(parameter, constant, true_computation, constant,
                          false_computation);
  auto gte = GetTupleElement(cond, 0);
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  // Result is not dynamic.
  EXPECT_FALSE(ComputeDynamismLiteral(gte, &b).value().Get<bool>({}));
}

TEST_F(DynamismInferenceTest, InferThroughCall) {
  // The result of following call instruction is static.
  //
  // Callee:
  //   p = param
  //   return p
  //
  // Entry:
  //   c = constant(3)
  //   return call(c), callee
  //
  //

  auto s32_shape = ShapeUtil::MakeShape(S32, {});
  XlaBuilder call_builder("call");
  Parameter(&call_builder, 0, s32_shape, "call_param");
  auto call_computation = call_builder.Build().value();

  XlaBuilder b(TestName());
  auto constant = ConstantR0<int32_t>(&b, 3);
  auto call = Call(&b, call_computation, {constant});
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  // Result is static.
  EXPECT_EQ(ComputeDynamismScalar(call, &b, {}).value(), false);
}

TEST_F(DynamismInferenceTest, InferThroughConditionalBranchesAreNotSame) {
  // The result of following conditional is dynamic.
  // pred = .. # a dynamic value
  // if (pred) {
  //  return (1) # These two branches return different values.
  // } else {
  //  return (2)
  // }
  //

  auto s32_shape = ShapeUtil::MakeShape(S32, {});
  auto cond_shape = ShapeUtil::MakeTupleShape({s32_shape});
  XlaBuilder true_builder("true");
  Parameter(&true_builder, 0, s32_shape, "cond_param");
  Tuple(&true_builder, {ConstantR0<int32_t>(&true_builder, 1)});
  auto true_computation = true_builder.Build().value();

  XlaBuilder false_builder("false");
  Parameter(&false_builder, 0, s32_shape, "cond_param");
  Tuple(&false_builder, {ConstantR0<int32_t>(&false_builder, 2)});
  auto false_computation = false_builder.Build().value();

  XlaBuilder b(TestName());
  auto parameter = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "p0");
  auto constant = ConstantR0<int32_t>(&b, 0);
  auto cond = Conditional(parameter, constant, true_computation, constant,
                          false_computation);
  auto gte = GetTupleElement(cond, 0);
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  // Result is dynamic.
  EXPECT_TRUE(ComputeDynamismLiteral(gte, &b).value().Get<bool>({}));
}

TEST_F(DynamismInferenceTest, InferThroughConditionalPredIsConstantTrueBranch) {
  // The result of following conditional is static.
  // pred = true
  // if (pred) {
  //  return (1)
  // } else {
  //  return (..dynamic_value...)
  // }
  //

  auto s32_shape = ShapeUtil::MakeShape(S32, {});
  auto cond_shape = ShapeUtil::MakeTupleShape({s32_shape});
  XlaBuilder true_builder("true");
  Parameter(&true_builder, 0, s32_shape, "cond_param");
  Tuple(&true_builder, {ConstantR0<int32_t>(&true_builder, 0)});
  auto true_computation = true_builder.Build().value();

  XlaBuilder false_builder("false");
  Tuple(&false_builder,
        {Parameter(&false_builder, 0, s32_shape, "cond_param")});
  auto false_computation = false_builder.Build().value();

  XlaBuilder b(TestName());
  auto pred = ConstantR0<bool>(&b, true);
  auto constant = ConstantR0<int32_t>(&b, 0);
  auto cond = Conditional(pred, constant, true_computation, constant,
                          false_computation);
  auto gte = GetTupleElement(cond, 0);
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  // Result is not dynamic.
  EXPECT_FALSE(ComputeDynamismLiteral(gte, &b).value().Get<bool>({}));
}

TEST_F(DynamismInferenceTest,
       InferThroughConditionalPredIsConstantFalseBranch) {
  // The result of following conditional is dynamic.
  // pred = false
  // if (pred) {
  //  return (1)
  // } else {
  //  return (..dynamic_value...)
  // }
  //

  auto s32_shape = ShapeUtil::MakeShape(S32, {});
  auto cond_shape = ShapeUtil::MakeTupleShape({s32_shape});
  XlaBuilder true_builder("true");
  Parameter(&true_builder, 0, s32_shape, "cond_param");
  Tuple(&true_builder, {ConstantR0<int32_t>(&true_builder, 0)});
  auto true_computation = true_builder.Build().value();

  XlaBuilder false_builder("false");
  Tuple(&false_builder,
        {Parameter(&false_builder, 0, s32_shape, "cond_param")});
  auto false_computation = false_builder.Build().value();

  XlaBuilder b(TestName());
  auto param = Parameter(&b, 0, s32_shape, "param");
  auto pred = ConstantR0<bool>(&b, false);
  auto constant = ConstantR0<int32_t>(&b, 0);
  auto cond =
      Conditional(pred, constant, true_computation, param, false_computation);
  auto gte = GetTupleElement(cond, 0);
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  // Result is dynamic.
  EXPECT_TRUE(ComputeDynamismLiteral(gte, &b).value().Get<bool>({}));
}

TEST_F(DynamismInferenceTest, ArgumentForwardingNestedTuple) {
  // The result of following conditional is considered static.
  // pred = .. dynamic value..
  //
  // op = 1
  // if (pred) {
  //   if (pred) {
  //     return op
  //   } else {
  //     return op
  //   }
  // } else {
  //   if (pred) {
  //     return op
  //   } else {
  //     return op
  //   }
  // }
  //
  auto pred_shape = ShapeUtil::MakeShape(PRED, {});
  auto s32_shape = ShapeUtil::MakeShape(S32, {});
  auto tuple_shape = ShapeUtil::MakeTupleShape({pred_shape, s32_shape});
  auto cond_shape = ShapeUtil::MakeTupleShape({s32_shape});
  XlaBuilder inner_true_builder("inner_true");
  Parameter(&inner_true_builder, 0, s32_shape, "cond_param");
  Tuple(&inner_true_builder, {ConstantR0<int32_t>(&inner_true_builder, 0)});
  auto inner_true_computation = inner_true_builder.Build().value();

  XlaBuilder inner_false_builder("inner_false");
  Tuple(&inner_false_builder,
        {Parameter(&inner_false_builder, 0, s32_shape, "cond_param")});
  auto inner_false_computation = inner_false_builder.Build().value();

  XlaBuilder true_builder("true");
  {
    auto param = Parameter(&true_builder, 0, tuple_shape, "param");
    auto op = GetTupleElement(param, 1);
    auto pred = GetTupleElement(param, 0);
    Conditional(pred, op, inner_true_computation, op, inner_false_computation);
  }
  auto true_computation = true_builder.Build().value();
  XlaBuilder false_builder("false");
  {
    auto param = Parameter(&false_builder, 0, tuple_shape, "param");
    auto op = GetTupleElement(param, 1);
    auto pred = GetTupleElement(param, 0);
    Conditional(pred, op, inner_true_computation, op, inner_false_computation);
  }
  auto false_computation = false_builder.Build().value();
  XlaBuilder b(TestName());
  auto constant = ConstantR0<int32_t>(&b, 0);
  auto pred = Parameter(&b, 0, pred_shape, "param");
  auto param = Tuple(&b, {pred, constant});
  auto cond =
      Conditional(pred, param, true_computation, param, false_computation);
  auto gte = GetTupleElement(cond, 0);
  ASSERT_TRUE(b.first_error().ok()) << b.first_error().message();
  // Result is static.
  EXPECT_FALSE(ComputeDynamismLiteral(gte, &b).value().Get<bool>({}));
}

class UpperBoundInferenceTest : public ValueInferenceTest {
 public:
  explicit UpperBoundInferenceTest(se::Platform* platform = nullptr)
      : platform_(platform) {}

  absl::StatusOr<OptionalLiteral> ComputeUpperBoundLiteral(
      XlaOp operand, XlaBuilder* builder, Layout* output_layout = nullptr) {
    ValueInference value_inference(builder);
    TF_ASSIGN_OR_RETURN(auto literal,
                        value_inference.AnalyzeConstant(
                            operand, ValueInferenceMode::kUpperBound));
    return literal;
  }

  se::Platform* platform_;
};

TEST_F(UpperBoundInferenceTest, GetDimensionSize) {
  XlaBuilder b(TestName());
  auto p =
      Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2, 3}, {true, false}), "p0");

  auto gds0 = GetDimensionSize(p, 0);
  auto gds1 = GetDimensionSize(p, 1);
  auto tuple_2 = Tuple(&b, {gds0, gds1});
  EXPECT_EQ(ComputeUpperBoundLiteral(tuple_2, &b).value().Get<int32_t>({}, {0}),
            2);
  EXPECT_EQ(ComputeUpperBoundLiteral(tuple_2, &b).value().Get<int32_t>({}, {1}),
            3);
}

TEST_F(UpperBoundInferenceTest, GetDimensionSizeSub) {
  XlaBuilder b(TestName());
  auto p =
      Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2, 3}, {true, false}), "p0");

  // The range of the first dimension is [0, 2]
  auto gds0 = GetDimensionSize(p, 0);
  // The range of the second dimension is [3, 3]
  auto gds1 = GetDimensionSize(p, 1);
  // Upper bound of `second_dimension - first_dimension` is 3 - 0 = 3
  auto sub = Sub(gds1, gds0);
  EXPECT_EQ(ComputeUpperBoundLiteral(sub, &b).value().Get<int32_t>({}), 3);
}

TEST_F(UpperBoundInferenceTest, GetDimensionSizeDiv) {
  XlaBuilder b(TestName());
  auto p =
      Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2, 3}, {true, false}), "p0");
  // The range of the first dimension is [0, 2]
  auto gds0 = GetDimensionSize(p, 0);
  // The range of the second dimension is [3, 3]
  auto gds1 = GetDimensionSize(p, 1);
  // Upper bound of `second_dimension / first_dimension` is 3 / 1 = 3. Notice we
  // don't use 0 as the lower bound as it would create divide-by-zero error.
  auto div = Div(gds1, gds0);
  EXPECT_EQ(ComputeUpperBoundLiteral(div, &b).value().Get<int32_t>({}), 3);
}

TEST_F(UpperBoundInferenceTest, SumSubtract) {
  // If x = a, y = b - a
  // upperbound(x + y) should be upperbound(b)
  XlaBuilder b(TestName());
  auto p =
      Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2, 3}, {true, true}), "p0");
  // The range of the first dimension is [0, 2]
  auto gds0 = GetDimensionSize(p, 0);
  // The range of the second dimension is [0, 3]
  auto gds1 = GetDimensionSize(p, 1);
  auto sub = Sub(gds1, gds0);
  auto add = Add(sub, gds0);
  EXPECT_EQ(ComputeUpperBoundLiteral(add, &b).value().Get<int32_t>({}), 3);
  auto add2 = Add(gds1, gds0);
  // upperbound(gds1 - gds0 + gds1 + gds0) ==> upperbound(2 * gds1)
  auto add3 = Add(sub, add2);
  EXPECT_EQ(ComputeUpperBoundLiteral(add3, &b).value().Get<int32_t>({}), 6);
}

TEST_F(UpperBoundInferenceTest, SumSubtractWithDataShuffling) {
  // Similar to the test above, but with some data shuffling ops in it
  // (broadcast, slice, reshape, identity convert, etc).
  XlaBuilder b(TestName());
  auto p =
      Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2, 3}, {true, true}), "p0");
  // The range of the first dimension is [0, 2]
  auto gds0 = GetDimensionSize(p, 0);
  // The range of the second dimension is [0, 3]
  auto gds1 = GetDimensionSize(p, 1);
  auto broadcast = Broadcast(gds0, {1, 10});
  auto convert = ConvertElementType(broadcast, S32);  // Identity convert.
  auto slice = SliceInDim(convert, /*start_index=*/0, /*limit_index=*/1,
                          /*stride=*/1, /*dimno=*/1);
  gds0 = Reshape(slice, {});
  auto sub = Sub(gds1, gds0);
  auto add = Add(sub, gds0);
  EXPECT_EQ(ComputeUpperBoundLiteral(add, &b).value().Get<int32_t>({}), 3);
  auto add2 = Add(gds1, gds0);
  // upperbound(gds1 - gds0 + gds1 + gds0) ==> upperbound(2 * gds1)
  auto add3 = Add(sub, add2);
  EXPECT_EQ(ComputeUpperBoundLiteral(add3, &b).value().Get<int32_t>({}), 6);
}

TEST_F(UpperBoundInferenceTest, SumSubtractEquivalentGetDimensionSize) {
  XlaBuilder b(TestName());
  auto p =
      Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2, 3}, {true, true}), "p0");
  // The range of the first dimension is [0, 2]
  auto gds0 = GetDimensionSize(p, 0);
  // The range of the second dimension is [0, 3]
  auto gds1 = GetDimensionSize(p, 1);
  // gds2 is equivalent to gds0
  auto gds2 = GetDimensionSize(p, 0);
  auto sub = Sub(gds1, gds2);
  auto add = Add(sub, gds0);
  // upperbound(gds0 + gds1 - gds2) is equal to upperbound(gds1) if gds0 ==
  // gds2.
  EXPECT_EQ(ComputeUpperBoundLiteral(add, &b).value().Get<int32_t>({}), 3);
}

TEST_F(UpperBoundInferenceTest, ParamCantInferBound) {
  // We can infer a parameter's dimension's bound, but not the parameter value's
  // bound.
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {2}, {true}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}, {}), "p1");
  auto gds = GetDimensionSize(p0, 0);
  auto sub = Div(gds, p1);
  EXPECT_FALSE(
      ComputeUpperBoundLiteral(sub, &b).value().Get<int32_t>({}).has_value());
}

TEST_F(UpperBoundInferenceTest, KeyValueSort) {
  XlaBuilder comparator_b("comparator");
  auto p0 = Parameter(&comparator_b, 0, ShapeUtil::MakeShape(S32, {}), "p0");
  auto p1 = Parameter(&comparator_b, 1, ShapeUtil::MakeShape(S32, {}), "p1");
  Parameter(&comparator_b, 2, ShapeUtil::MakeShape(S32, {}), "p2");
  Parameter(&comparator_b, 3, ShapeUtil::MakeShape(S32, {}), "p3");
  Compare(p0, p1, ComparisonDirection::kGe);
  TF_ASSERT_OK_AND_ASSIGN(auto comparator, comparator_b.Build());

  int64_t elem_count = 17;
  XlaBuilder b(TestName());
  auto param = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {elem_count}), "p0");
  auto iota = Iota(&b, S32, elem_count);
  auto sort = Sort({param, iota}, comparator);
  auto gte = GetTupleElement(sort, 1);

  for (int64_t i = 0; i < elem_count; ++i) {
    auto result_first_elem =
        ComputeUpperBoundLiteral(gte, &b).value().Get<int32_t>({i});
    // We can infer the bound of sort.
    EXPECT_TRUE(result_first_elem.has_value());
    // The bound of the sort result is the max value in the input.
    EXPECT_EQ(result_first_elem.value(), elem_count - 1);
  }
}

class ConstValueInferenceTest : public ValueInferenceTest {
 public:
  explicit ConstValueInferenceTest(se::Platform* platform = nullptr)
      : platform_(platform) {}

  absl::StatusOr<OptionalLiteral> ComputeConstantValueLiteral(
      XlaOp operand, XlaBuilder* builder, Layout* output_layout = nullptr) {
    ValueInference value_inference(builder);
    TF_ASSIGN_OR_RETURN(auto literal, value_inference.AnalyzeConstant(
                                          operand, ValueInferenceMode::kValue));
    return literal;
  }

  se::Platform* platform_;
};

TEST_F(ConstValueInferenceTest, ConstValuePassThroughSetBound) {
  XlaBuilder b(TestName());
  auto p0 = ConstantR0<int32_t>(&b, 32);
  Shape shape = ShapeUtil::MakeShape(S32, {});
  xla::Literal dynamism = xla::LiteralUtil::CreateR0<bool>(false);
  xla::Literal bound = xla::LiteralUtil::CreateR0<int32_t>(32);
  xla::Literal tuple =
      xla::LiteralUtil::MakeTupleOwned(std::move(bound), std::move(dynamism));
  auto set_bound =
      CustomCall(&b, "SetBound", {p0}, shape, "", false, {}, &tuple);
  auto result =
      ComputeConstantValueLiteral(set_bound, &b).value().Get<int32_t>({});
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 32);
}

// Parameters are always dynamic unless there is a SetBound wrapping it.
TEST_F(ConstValueInferenceTest, ParamaterValuePassThroughSetBound) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {}), "p0");
  Shape shape = ShapeUtil::MakeShape(S32, {});
  xla::Literal dynamism = xla::LiteralUtil::CreateR0<bool>(false);
  xla::Literal bound = xla::LiteralUtil::CreateR0<int32_t>(32);
  xla::Literal tuple =
      xla::LiteralUtil::MakeTupleOwned(std::move(bound), std::move(dynamism));
  auto set_bound =
      CustomCall(&b, "SetBound", {p0}, shape, "", false, {}, &tuple);
  auto result =
      ComputeConstantValueLiteral(set_bound, &b).value().Get<int32_t>({});
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(result.value(), 32);
}

}  // namespace
}  // namespace xla
