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

#include "xla/hlo/builder/xla_builder.h"

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/comparison_util.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/padding.h"
#include "xla/hlo/builder/sharding_builder.h"
#include "xla/hlo/builder/value_inference.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_input_output_alias_config.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/hlo/testlib/pattern_matcher_gmock.h"
#include "xla/hlo/testlib/test.h"
#include "xla/hlo/testlib/test_helpers.h"
#include "xla/layout_util.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/pattern_matcher.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

namespace m = ::xla::match;

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::Test;
using ::tsl::testing::StatusIs;

HloInstruction* GetRoot(HloModule& module) {
  return module.entry_computation()->root_instruction();
}

// TODO(b/74197823): Move the tests to service/.
absl::StatusOr<std::unique_ptr<HloModule>> BuildHloModule(XlaBuilder& b) {
  TF_ASSIGN_OR_RETURN(XlaComputation computation,
                      b.Build(/*remove_dynamic_dimensions=*/false));
  const HloModuleProto& proto = computation.proto();
  TF_ASSIGN_OR_RETURN(const auto& config,
                      HloModule::CreateModuleConfigFromProto(
                          proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(proto, config);
}

// Overload which explicitly specifies the root instruction.
absl::StatusOr<std::unique_ptr<HloModule>> BuildHloModule(XlaBuilder& b,
                                                          XlaOp root) {
  TF_ASSIGN_OR_RETURN(XlaComputation computation,
                      b.Build(root, /*remove_dynamic_dimensions=*/false));
  const HloModuleProto& proto = computation.proto();
  TF_ASSIGN_OR_RETURN(const auto& config,
                      HloModule::CreateModuleConfigFromProto(
                          proto, GetDebugOptionsFromFlags()));
  return HloModule::CreateFromProto(proto, config);
}

// Returns the name of the test currently being run.
std::string TestName() {
  return ::testing::UnitTest::GetInstance()->current_test_info()->name();
}

TEST(XlaBuilderTest, IsConstant) {
  {
    // cst -> tuple -> get_tuple_element
    XlaBuilder b(TestName());
    auto cst = ConstantR0<float>(&b, 1.0);
    auto tuple = Tuple(&b, {cst, cst});
    auto get_tuple_element = GetTupleElement(tuple, 0);
    TF_ASSERT_OK_AND_ASSIGN(bool is_constant, b.IsConstant(get_tuple_element));
    EXPECT_TRUE(is_constant);
  }
  {
    // param -> tuple -> get_tuple_element
    XlaBuilder b(TestName());
    auto param = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "p0");
    auto tuple = Tuple(&b, {param, param});
    auto get_tuple_element = GetTupleElement(tuple, 0);
    TF_ASSERT_OK_AND_ASSIGN(bool is_constant, b.IsConstant(get_tuple_element));
    EXPECT_FALSE(is_constant);
  }
  {
    // cst -> add -> tuple -> get_tuple_element
    XlaBuilder b(TestName());
    auto cst = ConstantR0<float>(&b, 1.0);
    auto add = Add(cst, cst);
    auto tuple = Tuple(&b, {add, add});
    auto get_tuple_element = GetTupleElement(tuple, 0);
    TF_ASSERT_OK_AND_ASSIGN(bool is_constant, b.IsConstant(get_tuple_element));
    EXPECT_TRUE(is_constant);
  }
  {
    // cst,param -> add -> tuple -> get_tuple_element
    XlaBuilder b(TestName());
    auto cst = ConstantR0<float>(&b, 1.0);
    auto param = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "p0");
    auto add = Add(cst, param);
    auto tuple = Tuple(&b, {add, add});
    auto get_tuple_element = GetTupleElement(tuple, 0);
    TF_ASSERT_OK_AND_ASSIGN(bool is_constant, b.IsConstant(get_tuple_element));
    EXPECT_FALSE(is_constant);
  }
}

TEST(XlaBuilderTest, ConstantSubgraph) {
  // cst -> tuple -> get_tuple_element
  XlaBuilder b(TestName());
  auto cst = ConstantR1<float>(&b, {1.0});
  auto tuple = Tuple(&b, {cst, cst});
  auto get_tuple_element = GetTupleElement(tuple, 0);
  // Returns ok if subgraph's root is a constant.
  EXPECT_TRUE(b.BuildConstantSubGraph(get_tuple_element).ok());
}

TEST(XlaBuilderTest, ConstantSubgraphGetDimSize) {
  // Needs a little hackery to update the param from dynamic to static since
  // xla::GetDimensionSize will fold to a constant if the dimension is static.

  // param->get_dim_size
  XlaBuilder b(TestName());
  internal::XlaBuilderFriend builder_friend;

  auto param = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {10}, {true}), "p0");
  auto get_dim_size = GetDimensionSize(param, 0);

  // Make input param static
  HloInstructionProto* param_proto = builder_friend.GetInstruction(param);
  param_proto->mutable_shape()->set_is_dynamic_dimension(0, false);

  EXPECT_TRUE(b.BuildConstantSubGraph(get_dim_size).ok());
}

TEST(XlaBuilderTest, OnePlusTwo) {
  XlaBuilder b(TestName());
  Add(ConstantR0<float>(&b, 1.0), ConstantR0<float>(&b, 2.0));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Constant(), m::Constant())));
}

TEST(XlaBuilderTest, UnaryOperatorsBuildExpectedHLO) {
  auto test_unary_operator = [&](std::function<XlaOp(XlaOp)> op,
                                 auto matches_pattern) {
    XlaBuilder b(TestName());
    op(ConstantR0<int32_t>(&b, 1));
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_THAT(GetRoot(*module), matches_pattern);
  };
  test_unary_operator([](XlaOp x) { return -x; },
                      GmockMatch(m::Negate(m::Constant())));
  test_unary_operator([](XlaOp x) { return ~x; },
                      GmockMatch(m::Not(m::Constant())));
}

TEST(XlaBuilderTest, BinaryOperatorsBuildExpectedHLO) {
  auto test_binary_operator = [&](std::function<XlaOp(XlaOp, XlaOp)> op,
                                  auto matches_pattern) {
    XlaBuilder b(TestName());
    op(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_THAT(GetRoot(*module), matches_pattern);
  };

  test_binary_operator([](XlaOp x, XlaOp y) { return x + y; },
                       GmockMatch(m::Add(m::Constant(), m::Constant())));
  test_binary_operator([](XlaOp x, XlaOp y) { return x - y; },
                       GmockMatch(m::Subtract(m::Constant(), m::Constant())));
  test_binary_operator([](XlaOp x, XlaOp y) { return x * y; },
                       GmockMatch(m::Multiply(m::Constant(), m::Constant())));
  test_binary_operator([](XlaOp x, XlaOp y) { return x / y; },
                       GmockMatch(m::Divide(m::Constant(), m::Constant())));

  test_binary_operator([](XlaOp x, XlaOp y) { return x & y; },
                       GmockMatch(m::And(m::Constant(), m::Constant())));
  test_binary_operator([](XlaOp x, XlaOp y) { return x | y; },
                       GmockMatch(m::Or(m::Constant(), m::Constant())));
  test_binary_operator([](XlaOp x, XlaOp y) { return x ^ y; },
                       GmockMatch(m::Xor(m::Constant(), m::Constant())));
  test_binary_operator([](XlaOp x, XlaOp y) { return x << y; },
                       GmockMatch(m::ShiftLeft(m::Constant(), m::Constant())));
  test_binary_operator(
      [](XlaOp x, XlaOp y) { return x >> y; },
      GmockMatch(m::ShiftRightArithmetic(m::Constant(), m::Constant())));

  auto test_unsigned_binary_operator =
      [&](std::function<XlaOp(XlaOp, XlaOp)> op, auto matches_pattern) {
        XlaBuilder b(TestName());
        op(ConstantR0<uint32_t>(&b, 1), ConstantR0<uint32_t>(&b, 2));
        TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
        EXPECT_THAT(GetRoot(*module), matches_pattern);
      };
  test_unsigned_binary_operator(
      [](XlaOp x, XlaOp y) { return x >> y; },
      GmockMatch(m::ShiftRightLogical(m::Constant(), m::Constant())));
}

TEST(XlaBuilderTest, VariadicAnd) {
  XlaBuilder b(TestName());
  const Shape s = ShapeUtil::MakeShape(PRED, {});
  And(Parameter(&b, 0, s, "p0"), Parameter(&b, 1, s, "p1"),
      Parameter(&b, 2, s, "p2"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  // Don't specify in the test whether And(x, y, z) is right- or
  // left-associative; accept either one.
  EXPECT_THAT(GetRoot(*module),
              ::testing::AnyOf(
                  GmockMatch(m::And(m::Parameter(0),
                                    m::And(m::Parameter(1), m::Parameter(2)))),
                  GmockMatch(m::And(m::And(m::Parameter(0), m::Parameter(1)),
                                    m::Parameter(2)))));
}

TEST(XlaBuilderTest, VariadicOr) {
  XlaBuilder b(TestName());
  const Shape s = ShapeUtil::MakeShape(PRED, {});
  Or(Parameter(&b, 0, s, "p0"), Parameter(&b, 1, s, "p1"),
     Parameter(&b, 2, s, "p2"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  // Don't specify in the test whether Or(x, y, z) is right- or
  // left-associative; accept either one.
  EXPECT_THAT(GetRoot(*module),
              ::testing::AnyOf(
                  GmockMatch(m::Or(m::Parameter(0),
                                   m::Or(m::Parameter(1), m::Parameter(2)))),
                  GmockMatch(m::Or(m::Or(m::Parameter(0), m::Parameter(1)),
                                   m::Parameter(2)))));
}

TEST(XlaBuilderTest, ShiftRightOperatorOnNonIntegerProducesError) {
  XlaBuilder b(TestName());
  ConstantR0<float>(&b, 1) >> ConstantR0<float>(&b, 2);
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr("Argument to >> operator does not have an integral type"));
}

TEST(XlaBuilderTest, ParamPlusConstantHasScalarBroadcast) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {3, 5}), "x");
  Add(x, ConstantR0<float>(&b, 1.0));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Parameter(), m::Broadcast(m::Constant()))));
}

TEST(XlaBuilderTest, ParamPlusConstantHasScalarBroadcastReversed) {
  XlaBuilder b(TestName());
  const XlaOp x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {3, 5}), "x");
  Add(ConstantR0<float>(&b, 1.0), x);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Broadcast(m::Constant()), m::Parameter())));
}

TEST(XlaBuilderTest, ParamPlusParamHasBroadcast) {
  XlaBuilder b(TestName());
  const auto& x_shape = ShapeUtil::MakeShape(S32, {2, 4, 6});
  const auto& y_shape = ShapeUtil::MakeShape(S32, {2, 4});
  auto x = Parameter(&b, 0, x_shape, "x");
  auto y = Parameter(&b, 1, y_shape, "y");
  auto add = Add(x, y, /*broadcast_dimensions=*/{0, 1});

  TF_ASSERT_OK_AND_ASSIGN(const auto add_shape, b.GetShape(add));
  EXPECT_TRUE(ShapeUtil::Equal(add_shape, x_shape));

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(
      GetRoot(*module),
      GmockMatch(m::Add(m::Parameter(0), m::Broadcast(m::Parameter(1)))));
}

TEST(XlaBuilderTest, XPlusX) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {1, 3, 5, 7}), "x");
  Add(x, x);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Parameter(0), m::Parameter(0))));
}

TEST(XlaBuilderTest, TestBinaryOpImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[2, 2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[2,2]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/{1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, TestBinaryOpImplicitBroadcastBounded) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[<=2, <=2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[<=2, <=2]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/{1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, ShapeInferenceError) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 4, 6}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {2, 4}), "y");
  Add(x, y);
  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Shapes must be equal rank"));
}

TEST(XlaBuilderTest, DynamicDimensionReshapeToR0) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {1}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "dyn_dim");
  auto dx = SetDimensionSize(x, y, 0);
  Reshape(dx, {});
  auto statusor = BuildHloModule(b);
  ASSERT_TRUE(statusor.ok());
}

TEST(XlaBuilderTest, ParameterAlreadyRegistered) {
  XlaBuilder b_call("add");
  Parameter(&b_call, 0, ShapeUtil::MakeShape(PRED, {}), "x");

  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "x");
  auto y = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "y");
  Add(x, y);
  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("parameter 0 already registered"));
}

TEST(XlaBuilderTest, Call) {
  XlaBuilder b_call("the_only_to_apply");
  auto p0 = Parameter(&b_call, 0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto p1 = Parameter(&b_call, 1, ShapeUtil::MakeShape(F32, {}), "p1");
  Add(p0, p1);
  TF_ASSERT_OK_AND_ASSIGN(const auto call, b_call.Build());
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "y");
  auto one = ConstantR0<float>(&b, 1);
  auto two = ConstantR0<float>(&b, 2);
  Add(Call(&b, call, {x, y}), Call(&b, call, {one, two}));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Call(m::Parameter(), m::Parameter()),
                                m::Call(m::Constant(), m::Constant()))));
}

TEST(XlaBuilderTest, CallSharedSubcomputation) {
  XlaBuilder b(TestName());

  auto b_call = b.CreateSubBuilder("the_only_to_apply");
  auto p0 = Parameter(b_call.get(), 0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto p1 = Parameter(b_call.get(), 1, ShapeUtil::MakeShape(F32, {}), "p1");
  Add(p0, p1);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputationId call, b_call->BuildSubComputation());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "y");
  auto one = ConstantR0<float>(&b, 1);
  auto two = ConstantR0<float>(&b, 2);
  Add(Call(&b, call, {x, y}), Call(&b, call, {one, two}));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  // The callee should not be duplicated.
  EXPECT_EQ(module->computation_count(), 2);
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Call(m::Parameter(), m::Parameter()),
                                m::Call(m::Constant(), m::Constant()))));
}

TEST(XlaBuilderTest, BuildFromSubcomputation) {
  XlaBuilder b_root(TestName());

  auto b_call = b_root.CreateSubBuilder("the_only_to_apply");
  auto p0 = Parameter(b_call.get(), 0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto p1 = Parameter(b_call.get(), 1, ShapeUtil::MakeShape(F32, {}), "p1");
  Add(p0, p1);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputationId call, b_call->BuildSubComputation());

  auto b = b_root.CreateSubBuilder("main");

  auto x = Parameter(b.get(), 0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = Parameter(b.get(), 1, ShapeUtil::MakeShape(F32, {}), "y");
  auto one = ConstantR0<float>(b.get(), 1);
  auto two = ConstantR0<float>(b.get(), 2);
  Add(Call(b.get(), call, {x, y}), Call(b.get(), call, {one, two}));
  TF_ASSERT_OK_AND_ASSIGN(XlaComputationId main, b->BuildSubComputation());

  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation, b_root.Build(main));
  const HloModuleProto& proto = computation.proto();
  TF_ASSERT_OK_AND_ASSIGN(const auto& config,
                          HloModule::CreateModuleConfigFromProto(
                              proto, GetDebugOptionsFromFlags()));
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          HloModule::CreateFromProto(proto, config));

  // The callee should not be duplicated.
  EXPECT_EQ(module->computation_count(), 2);
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Call(m::Parameter(), m::Parameter()),
                                m::Call(m::Constant(), m::Constant()))));
}

TEST(XlaBuilderTest, CompositeCall) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});

  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const XlaComputation computation, bsum.Build());

  std::vector<XlaOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands),
                /*name=*/"foo.bar",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/1);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Call(m::Parameter(), m::Parameter())));
}

TEST(XlaBuilderTest, CompositeCallFrontendAttributesStayLocal) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});

  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const XlaComputation computation, bsum.Build());

  std::vector<XlaOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands),
                /*name=*/"foo.bar",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/1);
  Add(operands[0], operands[1]);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_TRUE(GetRoot(*module)->frontend_attributes().map().empty());
}

TEST(XlaBuilderTest, CompositeCallMissingName) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});

  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const XlaComputation computation, bsum.Build());

  std::vector<XlaOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands), /*name=*/"",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/1);

  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("A composite call op must have frontend attributes "
                        "with key composite.name whose value is non-empty"));
}

TEST(XlaBuilderTest, CompositeCallMissingAttribute) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});

  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const XlaComputation computation, bsum.Build());

  std::vector<XlaOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands), /*name=*/"foo.bar",
                /*attributes=*/"", /*version=*/1);

  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "A composite call op must have frontend attributes with key "
          "composite.attributes whose value is default: {} or non-empty"));
}

TEST(XlaBuilderTest, CompositeCallNonNegativeVersion) {
  XlaBuilder b(TestName());

  FrontendAttributes frontend_attributes = b.frontend_attributes();
  frontend_attributes.mutable_map()->insert({"foo", "bar"});
  b.SetFrontendAttributes(frontend_attributes);

  const Shape shape = ShapeUtil::MakeShape(F32, {});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});

  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const XlaComputation computation, bsum.Build());

  std::vector<XlaOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands),
                /*name=*/"foo.bar",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/-1);

  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("A composite call op must have frontend attributes "
                        "with a composite.version whose value is a "
                        "non-negative integer but got: -1"));
}

TEST(XlaBuilderTest, CompositeCallOptionalVersionAndAttribute) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});

  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const XlaComputation computation, bsum.Build());

  std::vector<XlaOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands), /*name=*/"foo.bar");

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  ASSERT_THAT(GetRoot(*module),
              GmockMatch(m::Call(m::Parameter(), m::Parameter())));
  ASSERT_TRUE(GetRoot(*module)->frontend_attributes().map().contains(
      "composite.attributes"));
  EXPECT_EQ(
      GetRoot(*module)->frontend_attributes().map().at("composite.attributes"),
      "{}");
  EXPECT_EQ(
      GetRoot(*module)->frontend_attributes().map().at("composite.version"),
      "0");
}

TEST(XlaBuilderTest, CompositeCallWithExtraFrontendAttributes) {
  XlaBuilder b(TestName());

  FrontendAttributes frontend_attributes = b.frontend_attributes();
  frontend_attributes.mutable_map()->insert({"foo", "bar"});
  b.SetFrontendAttributes(frontend_attributes);

  const Shape shape = ShapeUtil::MakeShape(F32, {});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});

  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, shape, "arg0"), Parameter(&bsum, 1, shape, "arg1"));
  TF_ASSERT_OK_AND_ASSIGN(const XlaComputation computation, bsum.Build());

  std::vector<XlaOp> operands = {Parameter(&b, 0, shape, "arg0"),
                                 Parameter(&b, 1, shape, "arg1")};
  CompositeCall(&b, computation, absl::MakeSpan(operands),
                /*name=*/"foo.bar",
                /*attributes=*/"{n = 1 : i32, tensor = dense<1> : tensor<i32>}",
                /*version=*/1);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Call(m::Parameter(), m::Parameter())));
  ASSERT_TRUE(GetRoot(*module)->frontend_attributes().map().contains("foo"));
  EXPECT_EQ(GetRoot(*module)->frontend_attributes().map().at("foo"), "bar");
}

TEST(XlaBuilderTest, BinopHasDegenerateBroadcast) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {1, 2, 3}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {1, 2, 1}), "y");
  Add(x, y);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  // Expected:
  //
  //  x: f32[1,2,3]  y: f32[1,2,1]
  //      |               |
  //      |          reshape: f32[1,2]
  //      |               |
  //      |          broadcast: f32[1,2,3]
  //       \             /
  //            add
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Parameter(0),
                                m::Broadcast(m::Reshape(m::Parameter(1))))));
}

TEST(XlaBuilderTest, BinopHasInDimAndDegenerateBroadcast) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 3}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {2, 1, 4}), "y");
  Add(x, y, /*broadcast_dimensions=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  // The binary operation has in-dim broadcast and degenerate broadcast, should
  // first do the in-dim broadcast then convert the degenerate broadcast into a
  // reshape and a broadcast.
  //
  // Expected:
  //
  //  x: f32[2,3]            y: f32[2,1,4]
  //      |                        |
  //  broadcast: f32[2,3,4]  reshape: f32[2,4]
  //      |                        |
  //      |                  broadcast: f32[2,3,4]
  //       \                      /
  //                 add
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Broadcast(m::Parameter(0)),
                                m::Broadcast(m::Reshape(m::Parameter(1))))));
}

TEST(XlaBuilderTest, BroadcastInDim) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 3}), "x");
  BroadcastInDim(x, {2, 4, 3},
                 /*broadcast_dimensions=*/{0, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Broadcast()));
}

TEST(XlaBuilderTest, BroadcastInDimWithDegeneratedDim) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 1, 4}), "x");
  BroadcastInDim(x, {2, 3, 4},
                 /*broadcast_dimensions=*/{0, 1, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Broadcast(m::Reshape(m::Broadcast()))));
}

TEST(XlaBuilderTest, BroadcastInDimWithBoundedDim) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[2, <=3]"));
  auto x = Parameter(&b, 0, shape, "x");
  BroadcastInDim(x, {1, 2, 3},
                 /*broadcast_dimensions=*/{1, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Broadcast()));
}

TEST(XlaBuilderTest, BroadcastInDimWithNegativeSize) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 1, 4}), "x");
  BroadcastInDim(x, {-3, 3, 4},
                 /*broadcast_dimensions=*/{0, 1, 2});
  auto statusor = BuildHloModule(b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("Invalid dimension size"));
}

TEST(XlaBuilderTest, OperandFromWrongBuilder) {
  XlaBuilder b1("b1");
  auto p0 = Parameter(&b1, 0, ShapeUtil::MakeShape(F32, {}), "p0");
  XlaBuilder builder("main");
  auto p = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "p");
  Add(p, p0);
  auto statusor = builder.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().message(),
      HasSubstr(
          "built by builder 'b1', but is trying to use it in builder 'main'"));
}

TEST(XlaBuilderTest, ReshapeDefaultOrder) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 3, 5, 7}), "x");
  Reshape(x, /*dimensions=*/{6, 35});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Reshape(m::Parameter())));
}

TEST(XlaBuilderTest, Transpose) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  Transpose(x, /*permutation=*/{1, 0});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Transpose(m::Parameter())));
}

TEST(XlaBuilderTest, AllGatherR1) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4}), "x");
  AllGather(x, /*all_gather_dimension=*/0, /*shard_count=*/4);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  const HloInstruction* root = GetRoot(*module);
  EXPECT_EQ(root->opcode(), HloOpcode::kAllGather);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), ShapeUtil::MakeShape(F32, {16})));
}

TEST(XlaBuilderTest, AllGatherR2) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4, 16}), "x");
  AllGather(x, /*all_gather_dimension=*/1, /*shard_count=*/4);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  const HloInstruction* root = GetRoot(*module);
  EXPECT_EQ(root->opcode(), HloOpcode::kAllGather);
  EXPECT_TRUE(
      ShapeUtil::Equal(root->shape(), ShapeUtil::MakeShape(F32, {4, 64})));
}

TEST(XlaBuilderTest, AllGatherWithTuple) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4}), "x");
  auto x2 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {16, 4}), "x2");
  AllGather(Tuple(&b, {x, x2}), /*all_gather_dimension=*/0,
            /*shard_count=*/4);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  const HloInstruction* root = GetRoot(*module);
  EXPECT_EQ(root->opcode(), HloOpcode::kAllGather);
  EXPECT_TRUE(ShapeUtil::Equal(
      root->shape(),
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {16}),
                                 ShapeUtil::MakeShape(F32, {64, 4})})));
}

TEST(XlaBuilderTest, AllGatherTuple) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {128, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {128, 8}), "p1");
  AllGatherTuple({p0, p1}, /*all_gather_dimension=*/1, /*shard_count=*/4);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  auto tuple_shape =
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {128, 16}),
                                 ShapeUtil::MakeShape(F32, {128, 32})});
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op()
                             .WithOpcode(HloOpcode::kAllGather)
                             .WithShapeEqualTo(&tuple_shape)));
}

TEST(XlaBuilderTest, ReduceScatter) {
  XlaBuilder b(TestName());
  XlaComputation to_apply;
  {
    auto sub_builder = b.CreateSubBuilder("add");
    auto arg0 =
        Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(F32), "x");
    auto arg1 =
        Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(F32), "y");
    Add(arg0, arg1);
    TF_ASSERT_OK_AND_ASSIGN(to_apply, sub_builder->Build());
  }
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4, 16}), "x");
  ReplicaGroup group;
  group.add_replica_ids(0);
  group.add_replica_ids(1);
  ReduceScatter(x, to_apply, /*scatter_dimension=*/1, /*shard_count=*/2,
                /*replica_groups=*/{group});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  const HloInstruction* root = GetRoot(*module);
  EXPECT_EQ(root->opcode(), HloOpcode::kReduceScatter);
  EXPECT_TRUE(
      ShapeUtil::Equal(root->shape(), ShapeUtil::MakeShape(F32, {4, 8})));
}

TEST(XlaBuilderTest, ReduceScatterWithTuple) {
  XlaBuilder b(TestName());
  XlaComputation to_apply;
  {
    auto sub_builder = b.CreateSubBuilder("add");
    auto arg0 =
        Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(F32), "x");
    auto arg1 =
        Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(F32), "y");
    Add(arg0, arg1);
    TF_ASSERT_OK_AND_ASSIGN(to_apply, sub_builder->Build());
  }
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4, 16}), "x");
  auto x2 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {16, 4}), "x2");
  ReplicaGroup group;
  group.add_replica_ids(0);
  group.add_replica_ids(1);
  ReduceScatter(Tuple(&b, {x, x2}), to_apply, /*scatter_dimension=*/1,
                /*shard_count=*/2,
                /*replica_groups=*/{group});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  const HloInstruction* root = GetRoot(*module);
  EXPECT_EQ(root->opcode(), HloOpcode::kReduceScatter);
  EXPECT_TRUE(ShapeUtil::Equal(
      root->shape(),
      ShapeUtil::MakeTupleShape({ShapeUtil::MakeShape(F32, {4, 8}),
                                 ShapeUtil::MakeShape(F32, {16, 2})})));
}

TEST(XlaBuilderTest, AllToAll) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4, 16}), "x");
  AllToAll(x, /*split_dimension=*/1, /*concat_dimension=*/0,
           /*split_count=*/2);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  // AllToAll is decomposed into slices -> all-to-all -> gte -> concat.
  const HloInstruction* root = GetRoot(*module);
  EXPECT_EQ(root->opcode(), HloOpcode::kReshape);
  EXPECT_EQ(root->operand(0)->operand(0)->operand(0)->opcode(),
            HloOpcode::kAllToAll);
  EXPECT_TRUE(
      ShapeUtil::Equal(root->shape(), ShapeUtil::MakeShape(F32, {8, 8})));
}

// Test the special case where split_dimension is the same as concat_dimension.
TEST(XlaBuilderTest, AllToAllSpecial) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4, 16, 8}), "x");
  AllToAll(x, /*split_dimension=*/0, /*concat_dimension=*/0,
           /*split_count=*/2);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  // AllToAll is converted into a single all-to-all HloInstruction.
  const HloInstruction* root = GetRoot(*module);
  EXPECT_EQ(root->opcode(), HloOpcode::kAllToAll);
  EXPECT_TRUE(
      ShapeUtil::Equal(root->shape(), ShapeUtil::MakeShape(F32, {4, 16, 8})));
}

TEST(XlaBuilderTest, AllToAllTuple) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {2, 4}), "p1");
  ReplicaGroup replica_group;
  replica_group.add_replica_ids(0);
  replica_group.add_replica_ids(1);

  AllToAllTuple({p0, p1}, {replica_group}, LayoutUtil::MakeAscendingLayout(2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  // Check shape and replica groups.
  auto expected_shape =
      ShapeUtil::MakeShapeWithDenseLayout(F32, /* dimensions= */ {2, 4},
                                          /* minor_to_major= */ {0, 1});
  auto tuple_shape =
      ShapeUtil::MakeTupleShape({expected_shape, expected_shape});
  auto is_replica_group_pred = [](const HloInstruction* instr) {
    return instr->replica_groups().size() == 1 &&
           absl::c_equal(instr->replica_groups()[0].replica_ids(),
                         std::vector<int64_t>{0, 1});
  };

  // AllToAll is converted into a single all-to-all HloInstruction.
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op()
                             .WithOpcode(HloOpcode::kAllToAll)
                             .WithShapeEqualTo(&tuple_shape)
                             .WithPredicate(is_replica_group_pred)));
}

TEST(XlaBuilderTest, AllToAllTupleWithLayout) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[3, 15]{1,0}"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[1, 45]{1,0}"));
  AllToAllTuple(/*operand=*/Parameter(&b, 0, operand, "operand"),
                /*split_dimension=*/0,
                /*concat_dimension=*/1,
                /*split_count=*/3,
                /*replica_groups=*/{});

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, AllReduceTuple) {
  XlaBuilder b(TestName());
  auto shape0 = ShapeUtil::MakeShape(F32, {});
  auto shape1 = ShapeUtil::MakeShape(F32, {1, 2});
  auto p0 = Parameter(&b, 0, shape0, "p0");
  auto p1 = Parameter(&b, 1, shape1, "p1");

  XlaBuilder bsum(TestName());
  auto f32Scalar = ShapeUtil::MakeShape(F32, {});
  Add(Parameter(&bsum, 0, f32Scalar, "x"), Parameter(&bsum, 1, f32Scalar, "y"));
  TF_ASSERT_OK_AND_ASSIGN(const auto sum, bsum.Build());

  AllReduceTuple({p0, p1}, sum);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  // Check shape and replica groups.
  auto tuple_shape = ShapeUtil::MakeTupleShape({shape0, shape1});

  // AllToAll is converted into a single all-to-all HloInstruction.
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op()
                             .WithOpcode(HloOpcode::kAllReduce)
                             .WithShapeEqualTo(&tuple_shape)));
}

TEST(XlaBuilderTest, CollectiveBroadcast) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  ReplicaGroup replica_group;
  replica_group.add_replica_ids(0);
  replica_group.add_replica_ids(1);
  CollectiveBroadcast(x, {replica_group});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kCollectiveBroadcast);
}

TEST(XlaBuilderTest, CollectivePermute) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  CollectivePermute(x, {{0, 1}, {1, 2}, {2, 3}});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kCollectivePermute);
}

TEST(XlaBuilderTest, CombinedCollectivePermute) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {5, 7}), "y");
  MultiCollectivePermute({x, y}, {{0, 1}, {1, 2}, {2, 3}});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kCollectivePermute);
}

TEST(XlaBuilderTest, GetDimensionSize) {
  XlaBuilder b(TestName());
  auto x =
      Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}, {false, true}), "x");
  GetDimensionSize(x, 1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kGetDimensionSize);
}

TEST(XlaBuilderTest, GetDimensionSizeConstant) {
  XlaBuilder b(TestName());
  auto x =
      Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}, {false, true}), "x");
  // Get dimension size from a constant dimension gives us a constant.
  GetDimensionSize(x, 0);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kConstant);
}

TEST(XlaBuilderTest, OpToString) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  auto y = Add(x, x);
  EXPECT_EQ(b.OpToString(y),
            "add, shape=[5, 7], metadata={:0}\n"
            "  parameter, shape=[5, 7], metadata={:0}\n"
            "  parameter, shape=[5, 7], metadata={:0}");
}

TEST(XlaBuilderTest, ReplicaId) {
  XlaBuilder b(TestName());
  ReplicaId(&b);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kReplicaId);
}

TEST(XlaBuilderTest, ReportError) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  Add(b.ReportError(InvalidArgument("a test error")), x);
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("a test error"));
}

TEST(XlaBuilderTest, ReportErrorOrReturnHandlesNonErrors) {
  XlaBuilder b(TestName());
  absl::StatusOr<XlaOp> op(ConstantR0<float>(&b, 1.0));
  Add(b.ReportErrorOrReturn(op), ConstantR0<float>(&b, 2.0));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Add(m::Constant(), m::Constant())));
}

TEST(XlaBuilderTest, ReportErrorOrReturnHandlesErrors) {
  XlaBuilder b(TestName());
  absl::StatusOr<XlaOp> op(InvalidArgument("a test error"));
  Add(b.ReportErrorOrReturn(op), ConstantR0<float>(&b, 2.0));
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(), HasSubstr("a test error"));
}

TEST(XlaBuilderTest, BuildWithSpecificRoot) {
  XlaBuilder b(TestName());
  const XlaOp constant = ConstantR0<float>(&b, 1.0);
  Add(constant, ConstantR0<float>(&b, 2.0));
  TF_ASSERT_OK_AND_ASSIGN(const auto module,
                          BuildHloModule(b, /*root=*/constant));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Constant()));
}

TEST(XlaBuilderTest, BuildWithSpecificRootAndMultipleParameters) {
  // Specifying a particular root in Build should still include all entry
  // parameters.
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {42, 123});
  const XlaOp x = Parameter(&b, 0, shape, "x");
  const XlaOp y = Parameter(&b, 1, shape, "y");
  const XlaOp z = Parameter(&b, 2, shape, "z");
  Add(x, Sub(y, z));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, /*root=*/x));
  EXPECT_THAT(GetRoot(*module), GmockMatch(m::Parameter()));
  EXPECT_EQ(module->entry_computation()->num_parameters(), 3);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 5);
}

TEST(XlaBuilderTest, BuildWithSpecificRootWithWrongBuilder) {
  XlaBuilder b(TestName());
  XlaBuilder other_b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {42, 123});

  Parameter(&b, 0, shape, "param");
  const XlaOp other_param = Parameter(&other_b, 0, shape, "other_param");

  absl::Status status = b.Build(other_param).status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(
      status.message(),
      ::testing::HasSubstr("root operation is not in this computation"));
}

TEST(XlaBuilderTest, ProtoMatches) {
  std::vector<XlaComputation> computations;
  const int n = 2;
  computations.reserve(n);
  for (int i = 0; i < n; ++i) {
    XlaBuilder b_call("the_only_to_apply");
    auto p0 = Parameter(&b_call, 0, ShapeUtil::MakeShape(F32, {}), "p0");
    auto p1 = Parameter(&b_call, 1, ShapeUtil::MakeShape(F32, {}), "p1");
    Add(p0, Add(p1, p0));
    TF_ASSERT_OK_AND_ASSIGN(const auto call, b_call.Build());
    XlaBuilder b(TestName());
    auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "y");
    auto one = ConstantR0<float>(&b, 1);
    auto two = ConstantR0<float>(&b, 2);
    Add(Call(&b, call, {x, y}), Call(&b, call, {one, two}));
    computations.push_back(b.Build().value());
  }
  auto c0_string = computations[0].proto().SerializeAsString();
  auto c1_string = computations[1].proto().SerializeAsString();
  EXPECT_EQ(c0_string, c1_string);
}

TEST(XlaBuilderTest, DynamicParameter) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {6}, {true})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  Parameter(&b, 1, ShapeUtil::MakeShape(U32, {}), "p1");
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, /*root=*/p0));
  const Shape& param_shape = module->entry_computation()
                                 ->parameter_instruction(0)
                                 ->shape()
                                 .tuple_shapes(1);
  EXPECT_TRUE(param_shape.is_dynamic_dimension(0));
}

TEST(XlaBuilderTest, SetDimensionSize) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {10}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "p1");
  auto set_dim_size = SetDimensionSize(p0, p1, 0);
  TF_ASSERT_OK_AND_ASSIGN(const auto module,
                          BuildHloModule(b, /*root=*/set_dim_size));
  const Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(root_shape.is_dynamic_dimension(0));
}

TEST(XlaBuilderTest, RemoveDynamicDimension) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {10}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "p1");
  auto set_dim_size = SetDimensionSize(p0, p1, 0);
  auto remove_dim_size = RemoveDynamicDimension(set_dim_size, 0);
  TF_ASSERT_OK_AND_ASSIGN(const auto module,
                          BuildHloModule(b, /*root=*/remove_dim_size));
  const Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  // Dynamic dimension has been removed.
  EXPECT_FALSE(root_shape.is_dynamic_dimension(0));
}

TEST(XlaBuilderTest, RemoveDynamicDimensionMultiDims) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {10, 10}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "p1");
  auto set_dim_size = SetDimensionSize(p0, p1, 0);
  set_dim_size = SetDimensionSize(set_dim_size, p1, 1);
  auto remove_dim_size = RemoveDynamicDimension(set_dim_size, 0);
  remove_dim_size = RemoveDynamicDimension(remove_dim_size, 1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module,
                          BuildHloModule(b, /*root=*/remove_dim_size));
  const Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  // Dynamic dimensions are removed.
  EXPECT_FALSE(root_shape.is_dynamic_dimension(0));
  EXPECT_FALSE(root_shape.is_dynamic_dimension(1));
}

TEST(XlaBuilderTest, RemoveDynamicDimensionInBuild) {
  XlaBuilder b(TestName());
  auto dyn_shape = ShapeUtil::MakeShape(F32, {10, 10}, {true, true});
  auto static_shape = ShapeUtil::MakeShape(F32, {10, 10});
  auto param = Parameter(&b, 0, dyn_shape, "p0");
  EXPECT_EQ(b.GetProgramShape().value().result(), dyn_shape);
  EXPECT_EQ(b.GetProgramShape(param).value().result(), dyn_shape);
  TF_ASSERT_OK_AND_ASSIGN(XlaComputation computation,
                          b.Build(/*remove_dynamic_dimensions=*/true));
  EXPECT_EQ(computation.GetProgramShape().value().result(), static_shape);
}

TEST(XlaBuilderTest, DynamicUnary) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);
  Neg(gte);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(0));
}

TEST(XlaBuilderTest, DynamicBinary) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}, {true}),
       ShapeUtil::MakeShape(F32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(0));
}

TEST(XlaBuilderTest, DynamicBinaryHasBroadcast) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(F32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1, {0});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicBroadcast) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);
  BroadcastInDim(gte, /*out_dim_size=*/{3, 5, 4},
                 /*broadcast_dimensions=*/{1, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {false, true, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicBinaryHasDegenerateBroadcast) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {10}, {true}),
       ShapeUtil::MakeShape(F32, {1, 15}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1, /*broadcast_dimensions=*/{0});  // f32[<=10, 15]
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicSelectOnlyPredDynamic) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {10}, {true}),
       ShapeUtil::MakeShape(F32, {10}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);

  Select(gte0, gte1, gte1);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true}))
      << result_shape;
}

TEST(XlaBuilderTest, SelectIntoConditional) {
  XlaBuilder b(TestName());
  const Shape selector_shape = ShapeUtil::MakeShape(PRED, {});
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(S32, {}), ShapeUtil::MakeShape(F32, {})});
  const XlaOp p0 = Parameter(&b, 0, selector_shape, "p0");
  const XlaOp p1 = Parameter(&b, 1, tuple_param_shape, "p1");
  const XlaOp p2 = Parameter(&b, 2, tuple_param_shape, "p2");

  Select(p0, p1, p2);

  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              GmockMatch(m::Conditional(m::Parameter(0), m::Parameter(1),
                                        m::Parameter(2))));
  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->branch_computation(0)
                  ->root_instruction(),
              GmockMatch(m::Parameter(0)));
  EXPECT_THAT(module->entry_computation()
                  ->root_instruction()
                  ->branch_computation(1)
                  ->root_instruction(),
              GmockMatch(m::Parameter(0)));
}

TEST(XlaBuilderTest, DynamicPad) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pad_val = ConstantR0<float>(&b, -1);
  auto gte = GetTupleElement(p0, 0);
  PaddingConfig padding_config;
  for (int i = 0; i < 2; i++) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(0);
    dimension->set_interior_padding(0);
  }
  Pad(gte, pad_val, padding_config);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicConvolution) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {1, 2, 2, 128}, {true, false, false, false}),
       ShapeUtil::MakeShape(F32, {2, 2, 128, 8}, {false, false, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto input = GetTupleElement(p0, 0);
  auto filter = GetTupleElement(p0, 1);
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);
  ConvWithGeneralDimensions(input, filter, {1, 1}, Padding::kValid, dnums,
                            /*feature_group_count=*/1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(),
                              {true, false, false, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicConvolutions) {
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {1, 2, 2, 128}, {true, false, false, false}),
       ShapeUtil::MakeShape(F32, {2, 2, 128, 8}, {false, false, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);
  ASSERT_TRUE(XlaBuilder::Validate(dnums).ok());

  {
    // DynamicConvInputGrad
    XlaBuilder b(TestName());
    auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
    auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {4}), "p0");
    auto input = GetTupleElement(p0, 0);
    auto filter = GetTupleElement(p0, 1);
    DynamicConvInputGrad(
        p1, input, filter, {1, 1}, {{1, 1}, {1, 1}}, {1, 1}, {1, 1}, dnums,
        /*feature_group_count=*/1, /*batch_group_count=*/1,
        /*precision_config=*/nullptr, PaddingType::PADDING_VALID);
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_TRUE(GetRoot(*module)->IsCustomCall("DynamicConvolutionInputGrad"));
  }
  {
    // DynamicConvKernelGrad
    XlaBuilder b(TestName());
    auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
    auto input = GetTupleElement(p0, 0);
    auto filter = GetTupleElement(p0, 1);
    DynamicConvKernelGrad(
        input, filter, {1, 1}, {{1, 1}, {1, 1}}, {1, 1}, {1, 1}, dnums,
        /*feature_group_count=*/1, /*batch_group_count=*/1,
        /*precision_config=*/nullptr, PaddingType::PADDING_VALID);
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_TRUE(GetRoot(*module)->IsCustomCall("DynamicConvolutionKernelGrad"));
  }
  {
    // DynamicConvForward
    XlaBuilder b(TestName());
    auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
    auto input = GetTupleElement(p0, 0);
    auto filter = GetTupleElement(p0, 1);
    DynamicConvForward(
        input, filter, {1, 1}, {{1, 1}, {1, 1}}, {1, 1}, {1, 1}, dnums,
        /*feature_group_count=*/1, /*batch_group_count=*/1,
        /*precision_config=*/nullptr, PaddingType::PADDING_VALID);
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_TRUE(GetRoot(*module)->IsCustomCall("DynamicConvolutionForward"));
  }
  {
    // ConvWithGeneralPadding
    XlaBuilder b(TestName());
    auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
    auto input = GetTupleElement(p0, 0);
    auto filter = GetTupleElement(p0, 1);
    ConvWithGeneralPadding(input, filter, {1, 1}, {{1, 1}, {1, 1}},
                           /*feature_group_count=*/1, /*batch_group_count=*/1,
                           /*precision_config=*/nullptr, std::nullopt);
    TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
    EXPECT_EQ(GetRoot(*module)->opcode(), HloOpcode::kConvolution);
  }
}

TEST(XlaBuilderTest, DynamicDot) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 3, 4}, {true, true, false}),
       ShapeUtil::MakeShape(F32, {2, 4, 5}, {true, false, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");

  auto lhs = GetTupleElement(p0, 0);
  auto rhs = GetTupleElement(p0, 1);
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);
  DotGeneral(lhs, rhs, dnums);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {true, true, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicReduce) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5, 4, 3}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto init = ConstantR0<float>(&b, 0);
  auto gte = GetTupleElement(p0, 0);
  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, ShapeUtil::MakeShape(F32, {}), "x"),
      Parameter(&bsum, 1, ShapeUtil::MakeShape(F32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(const auto sum, bsum.Build());
  Reduce(gte, init, sum, {0});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicReduceWindow) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 4, 8}, {true, false, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto init = ConstantR0<float>(&b, 0.f);
  auto gte = GetTupleElement(p0, 0);
  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, ShapeUtil::MakeShape(F32, {}), "x"),
      Parameter(&bsum, 1, ShapeUtil::MakeShape(F32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(const auto sum, bsum.Build());
  ReduceWindow(gte, init, sum, /*window_dimensions=*/{1, 2, 4},
               /*window_strides=*/{1, 1, 1}, Padding::kValid);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  VLOG(2) << module->entry_computation()->root_instruction()->ToString()
          << "\n";
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {true, false, false}))
      << result_shape;
}

TEST(XlaBuilderTest, VariadicDynamicReduceWindow) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 4, 8}, {true, false, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto p1 = Parameter(&b, 1, tuple_param_shape, "p1");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p1, 0);
  std::vector<XlaOp> input_operands = {gte0, gte1};
  XlaBuilder bsum(TestName());
  auto p2 = Parameter(&bsum, 0, ShapeUtil::MakeShape(F32, {}), "x0");
  auto p3 = Parameter(&bsum, 1, ShapeUtil::MakeShape(F32, {}), "x1");
  auto p4 = Parameter(&bsum, 2, ShapeUtil::MakeShape(F32, {}), "y0");
  auto p5 = Parameter(&bsum, 3, ShapeUtil::MakeShape(F32, {}), "y1");
  std::vector<XlaOp> output_operands = {Add(p2, p4), Add(p3, p5)};
  Tuple(&bsum, absl::MakeSpan(output_operands));
  TF_ASSERT_OK_AND_ASSIGN(const auto sum, bsum.Build());
  auto init = ConstantR0<float>(&b, 0.f);
  ReduceWindow(input_operands, {init, init}, sum,
               /*window_dimensions=*/{1, 2, 4},
               /*window_strides=*/{1, 1, 1}, Padding::kValid);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  VLOG(2) << module->entry_computation()->root_instruction()->ToString()
          << "\n";
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.tuple_shapes(0).dynamic_dimensions(),
                              {true, false, false}))
      << result_shape.tuple_shapes(0);
  EXPECT_TRUE(ContainersEqual(result_shape.tuple_shapes(1).dynamic_dimensions(),
                              {true, false, false}))
      << result_shape.tuple_shapes(1);
}

TEST(XlaBuilderTest, DynamicSelectAndScatter) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 4, 8}, {true, false, false}),
       ShapeUtil::MakeShape(F32, {2, 2, 2}, {true, false, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto init = ConstantR0<float>(&b, 0.f);
  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, ShapeUtil::MakeShape(F32, {}), "x"),
      Parameter(&bsum, 1, ShapeUtil::MakeShape(F32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(const auto sum, bsum.Build());
  XlaBuilder bge(TestName());
  Ge(Parameter(&bge, 0, ShapeUtil::MakeShape(F32, {}), "x"),
     Parameter(&bge, 1, ShapeUtil::MakeShape(F32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(const auto ge, bge.Build());

  auto gte0 = GetTupleElement(p0, 0);
  auto source = GetTupleElement(p0, 1);
  SelectAndScatter(gte0, ge, {1, 2, 4}, {1, 2, 4}, Padding::kValid, source,
                   init, sum);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {true, false, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicReshape) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 3, 4, 5, 6},
                            {false, false, true, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);  // f32[2, 3, <=4, <=5, 6]
  Reshape(gte, /*new_sizes=*/{6, 4, 5, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(1));
  EXPECT_TRUE(result_shape.is_dynamic_dimension(2));
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(),
                              {false, true, true, false, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicSelect) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(F32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pred = Parameter(&b, 1, ShapeUtil::MakeShape(PRED, {}), "pred");
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Select(pred, gte0, gte1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(1));
  EXPECT_FALSE(result_shape.is_dynamic_dimension(2));
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {false, true, false}))
      << result_shape;
}

TEST(XlaBuilderTest, DynamicSelectNotCompatible) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(F32, {4, 5, 6}, {false, false, true}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pred = Parameter(&b, 1, ShapeUtil::MakeShape(PRED, {}), "pred");
  auto gte0 = GetTupleElement(p0, 0);  // f32[4,<=5,6]
  auto gte1 = GetTupleElement(p0, 1);  // f32[4,5,<=6]
  Select(pred, gte0, gte1);
  absl::Status status = BuildHloModule(b).status();
  ASSERT_IS_OK(status);
}

TEST(XlaBuilderTest, DynamicTranspose) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 5}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto gte = GetTupleElement(p0, 0);
  Transpose(gte, /*permutation=*/{1, 0});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {false, true}))
      << result_shape;
}

TEST(XlaBuilderTest, DotWithPreferredElementType) {
  XlaBuilder b(TestName());
  const Shape p0_shape = ShapeUtil::MakeShape(U8, {2, 3});
  const Shape p1_shape = ShapeUtil::MakeShape(U16, {3, 2});
  auto p0 = Parameter(&b, 0, p0_shape, "p0");
  auto p1 = Parameter(&b, 1, p1_shape, "p1");

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  DotGeneral(p0, p1, dnums, /*precision_config=*/nullptr,
             /*preferred_element_type=*/U32);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(U32, {2, 2}), result_shape));
}

TEST(XlaBuilderTest, FftWithFFT) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("c64[5, <=10]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("c64[5, <=10]"));
  Fft(Parameter(&b, 0, operand, "operand"), /*fft_type=*/FftType::FFT,
      fft_length);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, FftWithIFFT) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("c64[5, <=10]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("c64[5, <=10]"));
  Fft(Parameter(&b, 0, operand, "operand"), /*fft_type=*/FftType::IFFT,
      fft_length);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, FftWithRFFT) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f64[10, <=5]"));
  const std::vector<int64_t> fft_length = {5};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("c128[10, <=3]"));
  Fft(Parameter(&b, 0, operand, "operand"), /*fft_type=*/FftType::RFFT,
      fft_length);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, FftWithIRFFT) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("c128[10, <=3]"));
  const std::vector<int64_t> fft_length = {5};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f64[10, <=5]"));
  Fft(Parameter(&b, 0, operand, "operand"), /*fft_type=*/FftType::IRFFT,
      fft_length);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, SparseDot) {
  XlaBuilder b(TestName());
  auto lhs = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {10, 16}), "lhs");
  auto rhs = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {32, 20}), "rhs");
  auto meta = Parameter(&b, 2, ShapeUtil::MakeShape(U16, {10, 2}), "meta");

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(1);
  dnums.add_rhs_contracting_dimensions(0);
  SparsityDescriptor sparsity_descriptor;
  sparsity_descriptor.set_type(SparsityType::SPARSITY_STRUCTURED_N_M);
  sparsity_descriptor.set_n(2);
  sparsity_descriptor.set_m(4);
  sparsity_descriptor.set_index(0);
  sparsity_descriptor.set_dimension(1);
  std::vector<SparsityDescriptor> sparsity = {sparsity_descriptor};
  std::vector<XlaOp> sparse_meta = {meta};

  SparseDot(lhs, rhs, sparse_meta, sparsity, dnums);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[10, 20]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, RaggedDotNonContractingWithPreferredElementType) {
  XlaBuilder b(TestName());
  auto lhs = Parameter(&b, 0, ShapeUtil::MakeShape(S8, {11, 5}), "lhs");
  auto rhs = Parameter(&b, 1, ShapeUtil::MakeShape(S8, {3, 5, 7}), "rhs");
  auto group_sizes =
      Parameter(&b, 2, ShapeUtil::MakeShape(U32, {3}), "group_sizes");

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(1);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(0);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  RaggedDot(lhs, rhs, group_sizes, ragged_dot_dnums,
            /*precision_config=*/nullptr, /*preferred_element_type=*/S32);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("s32[11, 7]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, RaggedDotContractingWithPreferredElementType) {
  XlaBuilder b(TestName());
  auto lhs = Parameter(&b, 0, ShapeUtil::MakeShape(BF16, {11, 5}), "lhs");
  auto rhs = Parameter(&b, 1, ShapeUtil::MakeShape(BF16, {5, 7}), "rhs");
  auto group_sizes =
      Parameter(&b, 2, ShapeUtil::MakeShape(U32, {3}), "group_sizes");

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_contracting_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(0);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(1);

  RaggedDot(lhs, rhs, group_sizes, ragged_dot_dnums,
            /*precision_config=*/nullptr, /*preferred_element_type=*/F32);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[3, 11, 7]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, BatchedRaggedDotNonContractingWithPreferredElementType) {
  XlaBuilder b(TestName());
  auto lhs = Parameter(&b, 0, ShapeUtil::MakeShape(S8, {19, 11, 5}), "lhs");
  auto rhs = Parameter(&b, 1, ShapeUtil::MakeShape(S8, {3, 19, 5, 7}), "rhs");
  auto group_sizes =
      Parameter(&b, 2, ShapeUtil::MakeShape(U32, {19, 3}), "group_sizes");

  DotDimensionNumbers dot_dnums;
  dot_dnums.add_lhs_batch_dimensions(0);
  dot_dnums.add_lhs_contracting_dimensions(2);
  dot_dnums.add_rhs_batch_dimensions(1);
  dot_dnums.add_rhs_contracting_dimensions(2);
  RaggedDotDimensionNumbers ragged_dot_dnums;
  *ragged_dot_dnums.mutable_dot_dimension_numbers() = dot_dnums;
  ragged_dot_dnums.add_lhs_ragged_dimensions(1);
  ragged_dot_dnums.add_rhs_group_dimensions(0);

  RaggedDot(lhs, rhs, group_sizes, ragged_dot_dnums,
            /*precision_config=*/nullptr, /*preferred_element_type=*/S32);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("s32[19, 11, 7]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, ConvolutionWithPreferredElementType) {
  XlaBuilder b(TestName());
  const Shape p0_shape = ShapeUtil::MakeShape(S16, {1, 2, 2, 128});
  const Shape p1_shape = ShapeUtil::MakeShape(S8, {2, 2, 128, 8});
  auto p0 = Parameter(&b, 0, p0_shape, "p0");
  auto p1 = Parameter(&b, 1, p1_shape, "p1");

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);
  ConvWithGeneralDimensions(p0, p1, {1, 1}, Padding::kValid, dnums,
                            /*feature_group_count=*/1, /*batch_group_count=*/1,
                            /*precision_config=*/nullptr,
                            /*preferred_element_type=*/S32);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  ASSERT_TRUE(
      ShapeUtil::Equal(ShapeUtil::MakeShape(S32, {1, 1, 1, 8}), result_shape));
}

TEST(XlaBuilderTest, AfterAllWithNonTokenOperands) {
  XlaBuilder b(TestName());
  AfterAll(&b, {CreateToken(&b), ConstantR0<float>(&b, 1.0)});
  absl::Status status = b.Build().status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(status.message(),
              ::testing::HasSubstr("All operands to AfterAll must be tokens"));
}

TEST(XlaBuilderTest, AfterAllWithNoInputs) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("token[]"));
  AfterAll(&b, {});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, CheckInputOutputAlias) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.SetUpAlias({0}, 1, {});

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, root));

  const HloInputOutputAliasConfig& config = module->input_output_alias_config();
  EXPECT_TRUE(config.ParameterHasAlias(0, {}));
  EXPECT_TRUE(config.ParameterHasAlias(1, {}));

  auto alias_p0 = config.GetAliasedOutput(0, {});
  ASSERT_TRUE(alias_p0.has_value());
  EXPECT_EQ(*alias_p0, ShapeIndex({1}));

  auto alias_p1 = config.GetAliasedOutput(1, {});
  ASSERT_TRUE(alias_p1.has_value());
  EXPECT_EQ(*alias_p1, ShapeIndex({0}));
}

TEST(XlaBuilderTest, CheckBufferDonor) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.AddBufferDonor(0, {});

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, root));

  const HloBufferDonorConfig& config = module->buffer_donor_config();
  EXPECT_TRUE(config.ParameterIsBufferDonor(0, {}));
  EXPECT_FALSE(config.ParameterIsBufferDonor(1, {}));
}

TEST(XlaBuilderTest, ConstantLiteral) {
  XlaBuilder b(TestName());
#if defined(__x86_64__) && defined(_MM_DENORMALS_ZERO_ON)
  int old_csr = _mm_getcsr();
  // Treat denormals as zero. This will make the small number below equal to
  // 0.0, as far as the FP unit is concerned.
  _mm_setcsr(old_csr | _MM_DENORMALS_ZERO_ON);
#endif
  ConstantR1<float>(&b, {0.0f, 1.401298e-45f});
#if defined(__x86_64__) && defined(_MM_DENORMALS_ZERO_ON)
  _mm_setcsr(old_csr);
#endif
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);
  ASSERT_THAT(root, GmockMatch(m::Constant()));
}

TEST(XlaBuilderTest, InvalidInputOutputAliasBufferDonor) {
  XlaBuilder b(TestName());

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.AddBufferDonor(0, {});

  auto statusor = BuildHloModule(b, root);
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("is already aliased with one output, thus it cannot be "
                        "added as a buffer donor for any output."));
}

TEST(XlaBuilderTest, ValidInputOutputAliasBufferDonor) {
  XlaBuilder b(TestName());

  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.AddBufferDonor(1, {});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b, root));

  const HloInputOutputAliasConfig& io_alias_config =
      module->input_output_alias_config();
  const HloBufferDonorConfig& buffer_donor_config =
      module->buffer_donor_config();

  EXPECT_TRUE(io_alias_config.ParameterHasAlias(0, {}));
  EXPECT_FALSE(io_alias_config.ParameterHasAlias(1, {}));
  EXPECT_FALSE(buffer_donor_config.ParameterIsBufferDonor(0, {}));
  EXPECT_TRUE(buffer_donor_config.ParameterIsBufferDonor(1, {}));

  auto alias_p0 = io_alias_config.GetAliasedOutput(0, {});
  ASSERT_TRUE(alias_p0.has_value());
  EXPECT_EQ(*alias_p0, ShapeIndex({1}));
}

void ExpectAttributesMatch(const FrontendAttributes& attr,
                           const FrontendAttributes& ref) {
  EXPECT_EQ(ref.map_size(), attr.map_size());
  for (auto reference : ref.map()) {
    auto other = attr.map().find(reference.first);
    EXPECT_NE(other, attr.map().end());
    EXPECT_EQ(other->second, reference.second);
  }
}

void ExpectInstructionsAttributesMatch(
    const HloModule& module, const std::vector<FrontendAttributes>& expected) {
  ASSERT_EQ(module.computation_count(), 1);
  auto expected_it = expected.begin();
  for (auto inst : module.entry_computation()->instructions()) {
    ASSERT_NE(expected_it, expected.end());
    ExpectAttributesMatch(inst->frontend_attributes(), *expected_it);
    expected_it++;
  }
  EXPECT_EQ(expected_it, expected.end());
}

TEST(XlaBuilderTest, SimpleSetFrontendAttributes) {
  XlaBuilder b(TestName());
  FrontendAttributes attributes;

  ConstantR0(&b, 0);  // No attribute set

  (*attributes.mutable_map())["attr_a"] = "a";
  b.SetFrontendAttributes(attributes);
  ConstantR0(&b, 0);  // One attribute: { "attr_a": "a" }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));

  std::vector<FrontendAttributes> expected{FrontendAttributes(), attributes,
                                           FrontendAttributes()};
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST(XlaBuilderTest, ComplexSetFrontendAttributes) {
  XlaBuilder b(TestName());

  ConstantR0(&b, 0);  // No attribute set.
  std::vector<FrontendAttributes> expected{FrontendAttributes()};

  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);  // One attribute: { "attr_a": "a" }
    expected.push_back(attributes);
  }

  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_b"] = "b";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);  // One attribute: { "attr_b": "b" }
    expected.push_back(attributes);
  }

  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_b"] = "b";
    (*attributes.mutable_map())["attr_c"] = "c";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);  // Two attributes: { "attr_b": "b", "attr_c": "c" }
    expected.push_back(attributes);
  }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set
  expected.push_back(FrontendAttributes());

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST(XlaBuilderTest, AddFrontendAttribute) {
  XlaBuilder b(TestName());

  ConstantR0(&b, 0);
  std::vector<FrontendAttributes> expected{FrontendAttributes()};

  // One attribute: { "attr_a": "a" }
  {
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    b.SetFrontendAttributes(attributes);
    ConstantR0(&b, 0);
    expected.push_back(attributes);
  }

  // Two attributes: {"attra": "a", "attr_c": "c"}
  {
    auto op = ConstantR0(&b, 0);
    EXPECT_IS_OK(b.SetInstructionFrontendAttribute(op, "attr_c", "c"));

    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    (*attributes.mutable_map())["attr_c"] = "c";
    expected.push_back(attributes);
  }

  // Override value of existing "attr_a"
  // One attribute: { "attr_a", "a2"}
  {
    auto op = ConstantR0(&b, 0);
    EXPECT_IS_OK(b.SetInstructionFrontendAttribute(op, "attr_a", "a2"));
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a2";
    expected.push_back(attributes);
  }

  // Check "attr_a" is back to its original value
  // One attribute: { "attr_a", "a"}
  {
    auto op = ConstantR0(&b, 0);
    (void)op;
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_a"] = "a";
    expected.push_back(attributes);
  }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set
  expected.push_back(FrontendAttributes());

  // One attribute: { "attr_d", "d"}
  {
    auto op = ConstantR0(&b, 0);
    EXPECT_IS_OK(b.SetInstructionFrontendAttribute(op, "attr_d", "d"));
    FrontendAttributes attributes;
    (*attributes.mutable_map())["attr_d"] = "d";
    expected.push_back(attributes);
  }

  ConstantR0(&b, 0);  // No attribute set
  expected.push_back(FrontendAttributes());

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST(XlaBuilderTest, SetAndGetSharding) {
  XlaBuilder b(TestName());

  const Shape shape = ShapeUtil::MakeShape(F32, {1024});
  OpSharding op_sharding_1 = sharding_builder::Replicate();
  OpSharding op_sharding_2 = sharding_builder::Tile1D(shape, 4);
  TF_ASSERT_OK_AND_ASSIGN(HloSharding hlo_sharding_1,
                          HloSharding::FromProto(op_sharding_1));
  TF_ASSERT_OK_AND_ASSIGN(HloSharding hlo_sharding_2,
                          HloSharding::FromProto(op_sharding_2));

  b.SetSharding(op_sharding_1);
  XlaOp p0 = Parameter(&b, 0, shape, "p0");
  TF_ASSERT_OK_AND_ASSIGN(auto p0_sharding, b.GetOpSharding(p0));
  EXPECT_TRUE(p0_sharding.has_value());
  EXPECT_EQ(HloSharding::FromProto(p0_sharding.value()).value(),
            hlo_sharding_1);

  EXPECT_TRUE(b.SetInstructionSharding(p0, std::nullopt).ok());
  TF_ASSERT_OK_AND_ASSIGN(p0_sharding, b.GetOpSharding(p0));
  EXPECT_FALSE(p0_sharding.has_value());

  EXPECT_TRUE(b.SetInstructionSharding(p0, op_sharding_2).ok());
  TF_ASSERT_OK_AND_ASSIGN(p0_sharding, b.GetOpSharding(p0));
  EXPECT_TRUE(p0_sharding.has_value());
  EXPECT_EQ(HloSharding::FromProto(p0_sharding.value()).value(),
            hlo_sharding_2);

  EXPECT_EQ(HloSharding::FromProto(b.sharding().value()).value(),
            hlo_sharding_1);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_TRUE(
      module->entry_computation()->parameter_instruction(0)->has_sharding());
  EXPECT_EQ(module->entry_computation()->parameter_instruction(0)->sharding(),
            hlo_sharding_2);
}

TEST(XlaBuilderTest, ComparisonType) {
  XlaBuilder b(TestName());
  (void)Le(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);
  ASSERT_THAT(root, GmockMatch(m::Compare(m::Constant(), m::Constant())));
  EXPECT_EQ(Comparison::Type::kSigned,
            DynCast<HloCompareInstruction>(root)->type());
}

TEST(XlaBuilderTest, StableLookUpInstructionByHandle) {
  XlaBuilder b(TestName());
  internal::XlaBuilderFriend builder_friend;
  const XlaOp le = Le(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
  HloInstructionProto* first_op = builder_friend.GetInstruction(le);
  // Create some more instructions.
  for (int i = 0; i < 100; ++i) {
    (void)Le(ConstantR0<int32_t>(&b, 1), ConstantR0<int32_t>(&b, 2));
  }
  // Make sure first_op hasn't changed.
  HloInstructionProto* first_op_now = builder_friend.GetInstruction(le);
  EXPECT_EQ(first_op, first_op_now);
}

TEST(XlaBuilderTest, ComplexAbsConstant) {
  XlaBuilder b(TestName());
  const XlaOp out =
      Abs(ConstantR0<std::complex<float>>(&b, std::complex<float>{-1, -1}));
  ValueInference value_inference(&b);
  absl::StatusOr<OptionalLiteral> analyzed =
      value_inference.AnalyzeConstant(out, kUpperBound);
  EXPECT_IS_OK(analyzed.status());
  EXPECT_EQ(analyzed->GetValue().value().shape().element_type(),
            PrimitiveType::F32);
}

TEST(XlaBuilderTest, OutfeedDummyTupleSharding) {
  XlaBuilder b(TestName());
  const XlaOp value = ConstantR1<int32_t>(&b, {0});
  const Shape shape =
      ShapeUtil::MakeShapeWithDenseLayout(S32, /* dimensions= */ {1},
                                          /* minor_to_major= */ {0});
  Outfeed(value, shape, "");
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_FALSE(module->entry_computation()->root_instruction()->has_sharding());
}

TEST(XlaBuilderTest, OutfeedTokenSharding) {
  XlaBuilder b(TestName());
  const XlaOp value = ConstantR1<int32_t>(&b, {0});
  const Shape shape =
      ShapeUtil::MakeShapeWithDenseLayout(S32, /* dimensions= */ {1},
                                          /* minor_to_major= */ {0});
  b.SetSharding(sharding_builder::Replicate());
  Outfeed(value, shape, "");
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  auto it = std::find_if(module->entry_computation()->instructions().begin(),
                         module->entry_computation()->instructions().end(),
                         HloPredicateIsOp<HloOpcode::kOutfeed>);
  EXPECT_NE(it, module->entry_computation()->instructions().end());
  auto* outfeed = *it;
  EXPECT_TRUE(outfeed->has_sharding());
  EXPECT_TRUE(outfeed->sharding().IsTuple());
  EXPECT_EQ(outfeed->sharding().tuple_elements().size(), 2);
  EXPECT_TRUE(outfeed->operand(1)->has_sharding());
  EXPECT_EQ(outfeed->sharding().tuple_elements().back(),
            HloSharding::FromProto(sharding_builder::AssignDevice(0)).value());
  EXPECT_EQ(outfeed->operand(1)->sharding(),
            HloSharding::FromProto(sharding_builder::AssignDevice(0)).value());
}

TEST(XlaBuilderTest, NormalizeTupleSharding) {
  XlaBuilder b(TestName());
  const Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {6})});
  b.SetSharding(sharding_builder::Replicate());
  Parameter(&b, 0, tuple_param_shape, "p0");
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);
  EXPECT_TRUE(root->has_sharding());
  EXPECT_TRUE(root->sharding().IsTuple());
  EXPECT_EQ(GetRoot(*module)->sharding().tuple_elements().size(), 2);
}

TEST(XlaBuilderTest, InvalidSharding) {
  XlaBuilder b(TestName());
  const Shape shape2d = ShapeUtil::MakeShape(F32, {6, 8});
  const Shape shape1d = ShapeUtil::MakeShape(F32, {5});
  b.SetSharding(sharding_builder::Tile1D(shape1d, 4));
  Parameter(&b, 0, shape2d, "p0");
  auto statusor = b.Build();
  EXPECT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().message(),
              HasSubstr("Number of tile assignment dimensions (excluding "
                        "subgroups) is different than the input rank"));
}

TEST(XlaBuilderTest, TopKDimensions) {
  XlaBuilder b(TestName());
  int64_t k = 1;
  int64_t largest = true;
  TopK(Parameter(&b, 0, ShapeUtil::MakeShape(F32, {6, 8}), "p0"), k, largest);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);
  EXPECT_TRUE(root->opcode() == HloOpcode::kTopK);
  EXPECT_TRUE(root->shape().IsTuple());
  EXPECT_EQ(root->shape().tuple_shapes().size(), 2);
  EXPECT_EQ(root->shape().tuple_shapes(0).dimensions().size(), 2);
  EXPECT_EQ(root->shape().tuple_shapes(1).dimensions().size(), 2);
  EXPECT_EQ(root->shape().tuple_shapes(0).dimensions(0), 6);
  EXPECT_EQ(root->shape().tuple_shapes(0).dimensions(1), k);
  EXPECT_EQ(root->shape().tuple_shapes(1).dimensions(0), 6);
  EXPECT_EQ(root->shape().tuple_shapes(1).dimensions(1), k);
}

TEST(XlaBuilderTest, ExpWithResultAccuracy) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {1, 1});
  ResultAccuracy result_accuracy;
  ResultAccuracy::Tolerance tolerance;
  tolerance.set_ulps(120.0f);
  *result_accuracy.mutable_tolerance() = tolerance;
  Exp(Parameter(&b, 0, shape, "p0"), result_accuracy);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);

  EXPECT_EQ(root->result_accuracy().tolerance().ulps(), 120.0f);
  EXPECT_EQ(root->result_accuracy().mode(), ResultAccuracy::DEFAULT);
}

TEST(XlaBuilderTest, RsqrtWithResultAccuracy) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {1, 1});
  ResultAccuracy result_accuracy;
  ResultAccuracy::Tolerance tolerance;
  tolerance.set_ulps(120.0f);
  *result_accuracy.mutable_tolerance() = tolerance;
  Rsqrt(Parameter(&b, 0, shape, "p0"), result_accuracy);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);

  EXPECT_EQ(root->result_accuracy().tolerance().ulps(), 120.0f);
  EXPECT_EQ(root->result_accuracy().mode(), ResultAccuracy::DEFAULT);
}

TEST(XlaBuilderTest, CosineWithResultAccuracy) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {1, 1});
  ResultAccuracy result_accuracy;
  ResultAccuracy::Tolerance tolerance;
  tolerance.set_ulps(120.0f);
  *result_accuracy.mutable_tolerance() = tolerance;
  Cos(Parameter(&b, 0, shape, "p0"), result_accuracy);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  const HloInstruction* root = GetRoot(*module);

  EXPECT_EQ(root->result_accuracy().tolerance().ulps(), 120.0f);
  EXPECT_EQ(root->result_accuracy().mode(), ResultAccuracy::DEFAULT);
}

//============================================================================//
// Experimental Test
//============================================================================//

TEST(XlaBuilderTest, MhloDynamicBroadcastInDimExportSuccess) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[1, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("s32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("f32[1, 2, 3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[1, 2, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(module->ToString(), HasSubstr("mhlo.dynamic_broadcast_in_dim"));
  EXPECT_THAT(module->ToString(), HasSubstr("broadcast_dimensions=[1,2]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest,
     MhloDynamicBroadcastInDimNonBroadcastDimSizeGreaterThanOne) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("s32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("f32[2, 2, 3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[2, 2, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(module->ToString(), HasSubstr("mhlo.dynamic_broadcast_in_dim"));
  EXPECT_THAT(module->ToString(), HasSubstr("broadcast_dimensions=[1,2]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, MhloDynamicBroadcastInDimDynamicResultSize) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[1, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("s32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("f32[1, 2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[1, 2, ?]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(module->ToString(), HasSubstr("mhlo.dynamic_broadcast_in_dim"));
  EXPECT_THAT(module->ToString(), HasSubstr("broadcast_dimensions=[1,2]"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest,
     MhloDynamicBroadcastInDimInvalidOutputDimensionsElementType) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("f32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("f32[2, 3, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr("output_dimensions must be an integer type f32[3]")));
}

TEST(XlaBuilderTest, MhloDynamicBroadcastInDimInvalidOutputDimensionsRank) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions,
                          ParseShape("s32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("f32[2, 3, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr("output_dimensions must be rank 1 but got rank 2")));
}

TEST(XlaBuilderTest, MhloDynamicBroadcastInDimIncompatibleBroadcastSize) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_dimensions, ParseShape("s32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("f32[2, 3, 3]"));
  MhloDynamicBroadcastInDim(
      Parameter(&b, 0, operand, "operand"),
      Parameter(&b, 1, output_dimensions, "output_dimensions"),
      /*broadcast_dimensions=*/{1, 2}, output_shape);
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_, HasSubstr("size of operand dimension 0 (2) is not compatible "
                            "with size of result dimension 1 (3)")));
}

TEST(XlaBuilderTest, MhloDynamicReshapeExportSuccess) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 15]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("s32[2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[?, 15]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 15]"));
  MhloDynamicReshape(
      /*operand=*/Parameter(&b, 0, operand, "operand"),
      /*output_shape=*/Parameter(&b, 1, output_shape, "output_shape"),
      /*shape=*/shape);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(module->ToString(), HasSubstr("mhlo.dynamic_reshape"));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, MhloDynamicReshapeIncompatibleElementType) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 15]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("s32[2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("s32[?, 15]"));
  MhloDynamicReshape(
      /*operand=*/Parameter(&b, 0, operand, "operand"),
      /*output_shape=*/Parameter(&b, 1, output_shape, "output_shape"),
      /*shape=*/shape);
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("Element type of operand f32[?,15] and "
                                    "output s32[?,15] must match")));
}

TEST(XlaBuilderTest, MhloDynamicReshapeElementCountMismatch) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[3, 15]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("s32[2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[4, 15]"));
  MhloDynamicReshape(
      /*operand=*/Parameter(&b, 0, operand, "operand"),
      /*output_shape=*/Parameter(&b, 1, output_shape, "output_shape"),
      /*shape=*/shape);
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("MhloDynamicReshape has mismatched "
                                    "element counts: from=45 (f32[3,15]) "
                                    "to=60 (f32[4,15])")));
}

TEST(XlaBuilderTest, MhloDynamicReshapeRankMismatch) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 15]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output_shape, ParseShape("s32[3]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[?, 15]"));
  MhloDynamicReshape(
      /*operand=*/Parameter(&b, 0, operand, "operand"),
      /*output_shape=*/Parameter(&b, 1, output_shape, "output_shape"),
      /*shape=*/shape);
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_, HasSubstr("output_shape dimension size=3 (s32[3]) and rank "
                            "of shape=2 (f32[?,15]) must match")));
}

TEST(XlaBuilderTest, ConvertSpmdShardToFullShape) {
  XlaBuilder b(TestName());
  auto shape = ShapeUtil::MakeShape(F32, {8, 8});
  auto param = Parameter(&b, 0, shape, "operand");
  OpSharding sharding;
  sharding.ParseFromString("{devices=[4,8]<=[8,4]T(1,0)}");

  ASSERT_TRUE(
      ConvertSpmdShardToFullShape(&b, param, shape, 0, sharding, {}).ok());
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(module->ToString(), HasSubstr("custom_call_target=\"Sharding\", "
                                            "sharding={manual}"));
  EXPECT_THAT(module->ToString(),
              HasSubstr("custom_call_target=\"SPMDShardToFullShape\", "
                        "sharding={replicated}"));
}

TEST(XlaBuilderTest, ConvertSpmdFullToShardShape) {
  XlaBuilder b(TestName());
  auto shape = ShapeUtil::MakeShape(F32, {8, 8});
  auto param = Parameter(&b, 0, shape, "operand");
  OpSharding sharding;
  sharding.ParseFromString("{devices=[4,8]<=[8,4]T(1,0)}");

  ASSERT_TRUE(ConvertSpmdFullToShardShape(&b, param, 0, sharding, {}).ok());
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(module->ToString(), HasSubstr("custom_call_target=\"Sharding\", "
                                            "sharding={replicated}"));
  EXPECT_THAT(module->ToString(),
              HasSubstr("custom_call_target=\"SPMDFullToShardShape\", "
                        "sharding={manual}"));
}

//============================================================================//
// Unbounded Dynamism Test
//============================================================================//

struct UnaryOpTestCase {
  std::string operand;
  std::string expected;
  std::function<XlaOp(XlaOp)> unary_op;
};

struct BinaryOpTestCase {
  std::string lhs;
  std::string rhs;
  absl::Span<const int64_t> broadcast_dimensions;
  std::string expected;
  std::function<XlaOp(XlaOp, XlaOp, absl::Span<const int64_t>)> binary_op;
  std::optional<absl::string_view> error_message;
};

constexpr absl::string_view kBroadcastDimensionMismatch =
    "Broadcast dimension 0 mismatch: 2 != -9223372036854775808; f32[2] and "
    "f32[?,10].";
std::array<const int64_t, 0> empty_array = {};
std::array<const int64_t, 1> zero_array = {0};

class XlaBuilderUnboundedUnaryOpTest
    : public ::testing::TestWithParam<UnaryOpTestCase> {};

class XlaBuilderUnboundedBinaryOpTest
    : public ::testing::TestWithParam<BinaryOpTestCase> {};

TEST_P(XlaBuilderUnboundedUnaryOpTest, UnboundedUnaryOpTest) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape(GetParam().operand));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape(GetParam().expected));
  GetParam().unary_op(Parameter(&b, 0, operand, "operand"));
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST_P(XlaBuilderUnboundedBinaryOpTest, UnboundedBinaryOpTest) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape(GetParam().lhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape(GetParam().rhs));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape(GetParam().expected));
  GetParam().binary_op(Parameter(&b, 0, lhs, "lhs"),
                       Parameter(&b, 1, rhs, "rhs"),
                       GetParam().broadcast_dimensions);
  if (const auto result = BuildHloModule(b); result.ok()) {
    ASSERT_NE(*result, nullptr);
    EXPECT_THAT(GetRoot(**result),
                GmockMatch(m::Op().WithShapeEqualTo(&expected)));
  } else {
    ASSERT_TRUE(GetParam().error_message.has_value());
    EXPECT_THAT(result, StatusIs(_, HasSubstr(*GetParam().error_message)));
  }
}

TEST(XlaBuilderTest, UnboundedAddScalarBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/empty_array);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedAddDegenerateBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[1, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedAddUnsupportedImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Add(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/zero_array);
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr(kBroadcastDimensionMismatch)));
}

TEST(XlaBuilderTest, UnboundedAllGather) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  AllGather(Parameter(&b, 0, operand, "operand"), /*all_gather_dimension=*/0,
            /*shard_count=*/2,
            /*replica_groups=*/{});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedAllReduce) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));

  XlaComputation computation;
  {
    const std::unique_ptr<XlaBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, operand, "arg0"),
        Parameter(sub_builder.get(), 1, operand, "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(computation, sub_builder->Build());
  }

  AllReduce(Parameter(&b, 0, operand, "operand"), computation,
            /*replica_groups=*/{});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedAllToAllDynamicSplitDimension) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 15]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 45]"));
  AllToAll(/*operand=*/Parameter(&b, 0, operand, "operand"),
           /*split_dimension=*/0,
           /*concat_dimension=*/1,
           /*split_count=*/3,
           /*replica_groups=*/{});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedAllToAllDynamicConcatDimension) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 15]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 5]"));
  AllToAll(/*operand=*/Parameter(&b, 0, operand, "operand"),
           /*split_dimension=*/1,
           /*concat_dimension=*/0,
           /*split_count=*/3,
           /*replica_groups=*/{});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedAllToAllDynamicSplitAndConcatDimensionEqual) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 15]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 15]"));
  AllToAll(/*operand=*/Parameter(&b, 0, operand, "operand"),
           /*split_dimension=*/0,
           /*concat_dimension=*/0,
           /*split_count=*/3,
           /*replica_groups=*/{});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedAllToAllFullyDynamic) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, ?]"));
  AllToAll(/*operand=*/Parameter(&b, 0, operand, "operand"),
           /*split_dimension=*/0,
           /*concat_dimension=*/1,
           /*split_count=*/3,
           /*replica_groups=*/{});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedAllToAllTupleVariadicUnsupported) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 15]{1,0}"));
  b.ReportErrorOrReturn(
      AllToAllTuple(/*operands=*/{Parameter(&b, 0, operand, "operand0"),
                                  Parameter(&b, 1, operand, "operand1")},
                    /*replica_groups=*/{}));
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr(
                   "AllToAllTuple does not support unbounded dynamic shapes")));
}

TEST(XlaBuilderTest, UnboundedAllToAllTupleUnsupported) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 15]{1,0}"));
  b.ReportErrorOrReturn(
      AllToAllTuple(/*operand=*/Parameter(&b, 0, operand, "operand"),
                    /*split_dimension=*/0,
                    /*concat_dimension=*/1,
                    /*split_count=*/3,
                    /*replica_groups=*/{}));
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr(
                   "AllToAllTuple does not support unbounded dynamic shapes")));
}

TEST(XlaBuilderTest, BoundedAllToAllTupleUnsupported) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[3, <=15]{1,0}"));
  b.ReportErrorOrReturn(
      AllToAllTuple(/*operand=*/Parameter(&b, 0, operand, "operand"),
                    /*split_dimension=*/0,
                    /*concat_dimension=*/1,
                    /*split_count=*/3,
                    /*replica_groups=*/{}));
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr("AllToAll does not support bounded dynamic shapes")));
}

TEST(XlaBuilderTest, BoundedAllToAllUnsupported) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[3, <=15]{1,0}"));
  b.ReportErrorOrReturn(
      AllToAllTuple(/*operand=*/Parameter(&b, 0, operand, "operand"),
                    /*split_dimension=*/0,
                    /*concat_dimension=*/1,
                    /*split_count=*/3,
                    /*replica_groups=*/{}));
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr("AllToAll does not support bounded dynamic shapes")));
}

TEST(XlaBuilderTest, UnboundedAnd) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("s32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("s32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("s32[?, ?, 2, 2, <=2, <=2, ?]"));
  And(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/empty_array);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedBatchNormGrad) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape grad_operand, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape scale, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape mean, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape variance, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape grad_scale, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape grad_offset, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape grad_output, ParseShape("f32[5, ?, 7]"));
  const Shape expected =
      ShapeUtil::MakeTupleShape({grad_operand, grad_scale, grad_offset});
  BatchNormGrad(
      Parameter(&b, 0, operand, "operand"), Parameter(&b, 1, scale, "scale"),
      Parameter(&b, 2, mean, "mean"), Parameter(&b, 3, variance, "variance"),
      Parameter(&b, 4, grad_output, "grad_output"), 1.0, 1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedBatchNormInference) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape scale, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape offset, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape mean, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape variance, ParseShape("f32[5]"));
  BatchNormInference(
      Parameter(&b, 0, operand, "operand"), Parameter(&b, 1, scale, "scale"),
      Parameter(&b, 2, offset, "offset"), Parameter(&b, 3, mean, "mean"),
      Parameter(&b, 4, variance, "variance"), 1.0, 1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedBatchNormTraining) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape output, ParseShape("f32[?, ?, 7]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape scale, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape offset, ParseShape("f32[5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape batch_mean, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape batch_var, ParseShape("f32[?]"));
  const Shape expected =
      ShapeUtil::MakeTupleShape({output, batch_mean, batch_var});
  BatchNormTraining(Parameter(&b, 0, operand, "operand"),
                    Parameter(&b, 1, scale, "scale"),
                    Parameter(&b, 2, offset, "offset"), 1.0, 1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedBitcastConvert) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f16[?, 10, 2]"));
  BitcastConvertType(Parameter(&b, 0, operand, "operand"), PrimitiveType::F16);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedBroadcastUnsupportedOperand) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[<=3, ?]"));
  Broadcast(Parameter(&b, 0, operand, "operand"), /*broadcast_sizes=*/{1});
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("is_unbounded_dynamic")));
}

TEST(XlaBuilderTest, UnboundedBroadcastUnsupportedBroadcastSize) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[1]"));
  Broadcast(Parameter(&b, 0, operand, "operand"),
            /*broadcast_sizes=*/{Shape::kUnboundedSize});
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_, HasSubstr("Non-broadcast dimensions must not be dynamic.")));
}

TEST(XlaBuilderTest, UnboundedBroadcastInDim) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[<=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[<=2, 3, 4]"));
  BroadcastInDim(Parameter(&b, 0, operand, "operand"),
                 /*out_dim_size=*/{2, 3, 4},
                 /*broadcast_dimensions=*/{0, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedBroadcastInDimUnsupported) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[<=3, ?]"));
  BroadcastInDim(Parameter(&b, 0, operand, "operand"),
                 /*out_dim_size=*/{2, 3, Shape::kUnboundedSize},
                 /*broadcast_dimensions=*/{0, 2});
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("BroadcastInDim output must shape be "
                                    "static or bounded dynamic")));
}

TEST(XlaBuilderTest, UnboundedCall) {
  // This also serves as a test for StableHLO's FuncOp since there is no direct
  // 1 to 1 mapping of StableHLO FuncOp in HLO. Instead, we create
  // XlaComputation with unbounded shapes to test "UnboundedFunc".
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));

  XlaComputation computation;
  {
    const std::unique_ptr<XlaBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, operand, "arg0"),
        Parameter(sub_builder.get(), 1, operand, "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(computation, sub_builder->Build());
  }

  Call(/*builder=*/&b, /*computation=*/computation, /*operands=*/
       {Parameter(&b, 0, operand, "arg0"), Parameter(&b, 1, operand, "arg1")});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedCholesky) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape a, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Cholesky(Parameter(&b, 0, a, "a"), /*lower=*/true);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedClamp) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("f32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("f32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs,
                          ParseShape("f32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("f32[?, 1, ?, 2, ?, <=2, ?]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedClampScalarMinImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedClampScalarMinMaxImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedClampScalarOperandMaxImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedClampScalarMinOperandImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest,
     UnboundedClampUnsupportedDegenerateOperandImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[?, 10]"));
  Clamp(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
        Parameter(&b, 2, ehs, "ehs"));
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("Unimplemented implicit broadcast.")));
}

TEST(XlaBuilderTest, UnboundedCollectiveBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  CollectiveBroadcast(/*operand=*/Parameter(&b, 0, operand, "operand"),
                      /*replica_groups=*/{});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedCollectivePermute) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  CollectivePermute(/*operand=*/Parameter(&b, 0, operand, "operand"),
                    /*source_target_pairs=*/{std::make_pair(0, 1)});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedCompare) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("f32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("f32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("pred[?, ?, 2, 2, <=2, <=2, ?]"));
  Compare(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
          /*direction=*/{});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedConcatenate) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1,
                          ParseShape("f32[3, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand2,
                          ParseShape("f32[?, 4, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand3,
                          ParseShape("f32[?, ?, 2, 2, <=2, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("f32[3, 4, ?, 2, <=2, <=2, ?]"));
  ConcatInDim(&b,
              {Parameter(&b, 0, operand1, "operand1"),
               Parameter(&b, 1, operand2, "operand2"),
               Parameter(&b, 2, operand3, "operand3")},
              2);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedConvert) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("s32[?]"));
  ConvertElementType(Parameter(&b, 0, operand, "operand"), PrimitiveType::S32);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedConvolution) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, 2, ?, 128]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[2, 2, <=128, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 1, ?, 8]"));

  ConvolutionDimensionNumbers dnums;
  dnums.set_input_batch_dimension(0);
  dnums.set_output_batch_dimension(0);
  dnums.add_input_spatial_dimensions(1);
  dnums.add_output_spatial_dimensions(1);
  dnums.add_input_spatial_dimensions(2);
  dnums.add_output_spatial_dimensions(2);
  dnums.set_input_feature_dimension(3);
  dnums.set_output_feature_dimension(3);
  dnums.add_kernel_spatial_dimensions(0);
  dnums.add_kernel_spatial_dimensions(1);
  dnums.set_kernel_input_feature_dimension(2);
  dnums.set_kernel_output_feature_dimension(3);
  ConvWithGeneralDimensions(Parameter(&b, 0, lhs, "lhs"),
                            Parameter(&b, 1, rhs, "rhs"),
                            /*window_strides=*/{1, 1}, Padding::kValid, dnums);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedDot) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Dot(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedDotGeneral) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("f32[?, <=3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[2, 4, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, <=3, 5]"));

  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);

  DotGeneral(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"), dnums);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedDynamicSlice) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_indices, ParseShape("s32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[2, 2]"));
  DynamicSlice(Parameter(&b, 0, operand, "operand"),
               /*start_indices=*/
               {
                   Parameter(&b, 1, start_indices, "start_indices0"),
                   Parameter(&b, 2, start_indices, "start_indices1"),
               },
               /*slice_sizes=*/{2, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedDynamicUpdateSlice) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape update, ParseShape("f32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_indices, ParseShape("s32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  DynamicUpdateSlice(Parameter(&b, 0, operand, "operand"),
                     Parameter(&b, 1, update, "update"),
                     /*start_indices=*/
                     {Parameter(&b, 2, start_indices, "start_indices0"),
                      Parameter(&b, 3, start_indices, "start_indices1")});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedFftWithFFT) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("c64[2, <=5, ?]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("c64[2, <=5, ?]"));
  Fft(Parameter(&b, 0, operand, "operand"), /*fft_type=*/FftType::FFT,
      fft_length);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedFftWithIFFT) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("c64[2, <=5, ?]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("c64[2, <=5, ?]"));
  Fft(Parameter(&b, 0, operand, "operand"), /*fft_type=*/FftType::IFFT,
      fft_length);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedFftWithRFFT) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f64[2, <=5, ?]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("c128[2, <=5, 6]"));
  Fft(Parameter(&b, 0, operand, "operand"), /*fft_type=*/FftType::RFFT,
      fft_length);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedFftWithIRFFT) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("c128[2, <=5, ?]"));
  const std::vector<int64_t> fft_length = {5, 10};
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f64[2, <=5, 10]"));
  Fft(Parameter(&b, 0, operand, "operand"), /*fft_type=*/FftType::IRFFT,
      fft_length);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedGather) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[3, 4, 2]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape start_indices,
                          ParseShape("s32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, ?, 2, 2]"));

  GatherDimensionNumbers dimension_numbers;
  dimension_numbers.add_offset_dims(2);
  dimension_numbers.add_offset_dims(3);
  dimension_numbers.add_collapsed_slice_dims(0);
  dimension_numbers.add_start_index_map(1);
  dimension_numbers.add_start_index_map(0);
  dimension_numbers.set_index_vector_dim(2);

  Gather(Parameter(&b, 0, operand, "operand"),
         Parameter(&b, 1, start_indices, "start_indices"), dimension_numbers,
         /*slice_sizes=*/{1, 2, 2});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedGetTupleElement) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  GetTupleElement(Tuple(&b, {Parameter(&b, 0, operand, "operand")}), 0);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedInfeed) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Infeed(/*builder=*/&b, /*shape=*/shape, /*config=*/"");
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedInfeedWithToken) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(f32[?, 10], token[])"));
  InfeedWithToken(/*token=*/CreateToken(&b), /*shape=*/shape, /*config=*/"");
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<xla::HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedMap) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand0, ParseShape("f32[2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand1, ParseShape("f32[?, 3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[2, ?, ?]"));

  XlaComputation computation;
  {
    const std::unique_ptr<XlaBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(F32),
                  "arg0"),
        Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(F32),
                  "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(computation, sub_builder->Build());
  }

  Map(&b, /*operands=*/
      {Parameter(&b, 0, operand0, "operand0"),
       Parameter(&b, 1, operand1, "operand1")},
      computation, /*dimensions=*/{0, 1, 2},
      /*static_operands=*/{});

  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedOptimizationBarrier) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  OptimizationBarrier(Parameter(&b, 0, operand, "operand"));
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedOr) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("s32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("s32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("s32[?, ?, 2, 2, <=2, <=2, ?]"));
  Or(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
     /*broadcast_dimensions=*/empty_array);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedOutfeed) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape_with_layout,
                          ParseShape("f32[?, 10]"));
  Outfeed(/*operand=*/Parameter(&b, 0, operand, "operand"),
          /*shape_with_layout=*/shape_with_layout, /*outfeed_config=*/"");
  EXPECT_IS_OK(BuildHloModule(b));
}

TEST(XlaBuilderTest, UnboundedOutfeedWithToken) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape_with_layout,
                          ParseShape("f32[?, 10]"));
  OutfeedWithToken(/*operand=*/Parameter(&b, 0, operand, "operand"),
                   /*token=*/CreateToken(&b),
                   /*shape_with_layout=*/shape_with_layout,
                   /*outfeed_config=*/"");
  EXPECT_IS_OK(BuildHloModule(b));
}

TEST(XlaBuilderTest, UnboundedPad) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 21]"));
  PaddingConfig padding_config;
  for (int i = 0; i < 2; i++) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(1);
    dimension->set_edge_padding_high(1);
    dimension->set_interior_padding(1);
  }
  Pad(Parameter(&b, 0, operand, "operand"),
      /*padding_value=*/ConstantR0<float>(&b, 0), padding_config);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedRecv) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[?, 10]"));
  ChannelHandle handle;
  handle.set_handle(1);
  handle.set_type(ChannelHandle::DEVICE_TO_DEVICE);
  Recv(/*builder=*/&b, /*shape=*/shape, /*handle=*/handle);
  EXPECT_IS_OK(BuildHloModule(b));
}

TEST(XlaBuilderTest, UnboundedRecvFromHost) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[?, 10]"));
  ChannelHandle handle;
  handle.set_handle(1);
  handle.set_type(ChannelHandle::HOST_TO_DEVICE);
  RecvFromHost(/*token=*/CreateToken(&b), /*shape=*/shape, /*handle=*/handle);
  EXPECT_IS_OK(BuildHloModule(b));
}

TEST(XlaBuilderTest, UnboundedRecvWithToken) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[?, 10]"));
  ChannelHandle handle;
  handle.set_handle(1);
  handle.set_type(ChannelHandle::DEVICE_TO_DEVICE);
  RecvWithToken(/*token=*/CreateToken(&b), /*shape=*/shape, /*handle=*/handle);
  EXPECT_IS_OK(BuildHloModule(b));
}

TEST(XlaBuilderTest, UnboundedReduce) {
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {7}, {false});
  const Shape expected = ShapeUtil::MakeTupleShape({shape, shape, shape});

  TF_ASSERT_OK_AND_ASSIGN(const Shape input0, ParseShape("f32[7, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input1, ParseShape("f32[?, 5]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape input2, ParseShape("f32[7, ?]"));
  const Shape scalar_f32 = ShapeUtil::MakeShape(F32, {});
  const XlaOp init = Parameter(&b, 3, scalar_f32, "init");

  XlaBuilder bsum(TestName());
  std::vector<XlaOp> output_operands = {
      Add(Parameter(&bsum, 0, scalar_f32, "arg0"),
          Parameter(&bsum, 1, scalar_f32, "arg1")),
      Add(Parameter(&bsum, 2, scalar_f32, "arg2"),
          Parameter(&bsum, 3, scalar_f32, "arg3")),
      Add(Parameter(&bsum, 4, scalar_f32, "arg4"),
          Parameter(&bsum, 5, scalar_f32, "arg5"))};
  Tuple(&bsum, absl::MakeSpan(output_operands));
  TF_ASSERT_OK_AND_ASSIGN(const XlaComputation sum, bsum.Build());
  Reduce(
      &b,
      {Parameter(&b, 0, input0, "input0"), Parameter(&b, 1, input1, "input1"),
       Parameter(&b, 2, input2, "input2")},
      {init, init, init}, sum, {1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedReducePrecision) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  ReducePrecision(Parameter(&b, 0, operand, "operand"), /*exponent_bits=*/2,
                  /*mantissa_bits=*/2);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedReduceScatter) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));

  XlaComputation computation;
  {
    const std::unique_ptr<XlaBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(/*lhs=*/Parameter(sub_builder.get(), 0, operand, "arg0"),
        /*rhs=*/Parameter(sub_builder.get(), 1, operand, "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(computation, sub_builder->Build());
  }

  ReplicaGroup replica_group;
  replica_group.add_replica_ids(0);
  replica_group.add_replica_ids(1);

  ReduceScatter(
      /*operand=*/Parameter(&b, 0, operand, "operand"),
      /*computation=*/computation,
      /*scatter_dimension=*/0,
      /*shard_count=*/2,
      /*replica_groups=*/{replica_group});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedReduceWindow) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape input, ParseShape("f32[?, 4, 8]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 3, 5]"));

  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, ShapeUtil::MakeShape(F32, {}), "x"),
      Parameter(&bsum, 1, ShapeUtil::MakeShape(F32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(const XlaComputation sum, bsum.Build());

  ReduceWindow(Parameter(&b, 0, input, "input"), ConstantR0<float>(&b, 0.f),
               sum,
               /*window_dimensions=*/{1, 2, 4},
               /*window_strides=*/{1, 1, 1}, Padding::kValid);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedReshape) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[2,3]"));
  Reshape(Parameter(&b, 0, operand, "operand"), /*dimensions=*/{2, 3});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedReshapeUnsupportedOutputShape) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[6]"));
  Reshape(Parameter(&b, 0, operand, "operand"),
          /*dimensions=*/{Shape::kUnboundedSize, Shape::kUnboundedSize});
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr(
                   "Reshaping with unbounded result shape is not supported.")));
}

TEST(XlaBuilderTest, UnboundedReshapeUnsupportedInferredShape) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?]"));
  Reshape(operand, Parameter(&b, 0, operand, "operand"));
  EXPECT_THAT(
      BuildHloModule(b),
      StatusIs(_,
               HasSubstr(
                   "Reshaping with unbounded result shape is not supported.")));
}

TEST(XlaBuilderTest, UnboundedReverse) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Rev(Parameter(&b, 0, operand, "operand"), /*dimensions=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedRngBitGenerator) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape initial_state, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("u32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("(u32[?, 10], u32[?, 10])"));
  RngBitGenerator(RandomAlgorithm::RNG_DEFAULT,
                  Parameter(&b, 0, initial_state, "initial_state"), shape);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedRngNormal) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  RngNormal(Parameter(&b, 0, ShapeUtil::MakeScalarShape(F32), "mu"),
            Parameter(&b, 1, ShapeUtil::MakeScalarShape(F32), "sigma"), shape);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedRngUniform) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  RngUniform(Parameter(&b, 0, ShapeUtil::MakeScalarShape(F32), "a"),
             Parameter(&b, 1, ShapeUtil::MakeScalarShape(F32), "b"), shape);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedScatter) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape input, ParseShape("f32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape scatter_indices,
                          ParseShape("s32[?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape updates, ParseShape("f32[?, ?, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, ?, ?]"));

  XlaComputation update_computation;
  {
    const std::unique_ptr<XlaBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(F32),
                  "arg0"),
        Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(F32),
                  "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(update_computation, sub_builder->Build());
  }

  ScatterDimensionNumbers dimension_numbers;
  dimension_numbers.add_update_window_dims(2);
  dimension_numbers.add_update_window_dims(3);
  dimension_numbers.add_inserted_window_dims(0);
  dimension_numbers.add_scatter_dims_to_operand_dims(1);
  dimension_numbers.add_scatter_dims_to_operand_dims(0);
  dimension_numbers.set_index_vector_dim(2);

  Scatter(Parameter(&b, 0, input, "input"),
          Parameter(&b, 1, scatter_indices, "scatter_indices"),
          Parameter(&b, 2, updates, "updates"), update_computation,
          dimension_numbers, /*indices_are_sorted=*/false,
          /*unique_indices=*/false);

  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedSelect) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("pred[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("f32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs,
                          ParseShape("f32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("f32[1, 1, 2, 2, <=2, <=2, ?]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedSelectScalarPred) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedSelectScalarOnTrueOnFalseImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedSelectScalarPredOnFalseImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedSelectScalarPredOnTrueImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest,
     UnboundedSelectUnsupportedDegenerateOperandImplicitBroadcast) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs, ParseShape("pred[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs, ParseShape("f32[1]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape ehs, ParseShape("f32[?, 10]"));
  Select(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
         Parameter(&b, 2, ehs, "ehs"));
  EXPECT_THAT(BuildHloModule(b),
              StatusIs(_, HasSubstr("Unimplemented implicit broadcast.")));
}

TEST(XlaBuilderTest, UnboundedSelectAndScatter) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape source, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape init_value, ParseShape("f32[]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));

  XlaComputation select;
  {
    const std::unique_ptr<XlaBuilder> sub_builder =
        b.CreateSubBuilder("compare");
    Compare(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(F32),
                      "arg0"),
            Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(F32),
                      "arg1"),
            ComparisonDirection::kGe);
    TF_ASSERT_OK_AND_ASSIGN(select, sub_builder->Build());
  }

  XlaComputation scatter;
  {
    const std::unique_ptr<XlaBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(F32),
                  "arg0"),
        Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(F32),
                  "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(scatter, sub_builder->Build());
  }

  SelectAndScatter(Parameter(&b, 0, operand, "operand"), select,
                   /*window_dimensions=*/
                   std::array<int64_t, 2>({3, 1}),
                   /*window_strides=*/std::array<int64_t, 2>({2, 1}),
                   Padding::kValid, Parameter(&b, 1, source, "source"),
                   Parameter(&b, 2, init_value, "init_value"), scatter);

  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedSend) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  ChannelHandle handle;
  handle.set_handle(1);
  handle.set_type(ChannelHandle::DEVICE_TO_DEVICE);
  Send(/*operand=*/Parameter(&b, 0, operand, "operand"), /*handle=*/handle);
  EXPECT_IS_OK(BuildHloModule(b));
}

TEST(XlaBuilderTest, UnboundedSendToHost) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape shape_with_layout,
                          ParseShape("f32[?, 10]"));
  ChannelHandle handle;
  handle.set_handle(1);
  handle.set_type(ChannelHandle::DEVICE_TO_HOST);
  SendToHost(/*operand=*/Parameter(&b, 0, operand, "operand"),
             /*token=*/CreateToken(&b), /*shape_with_layout=*/shape_with_layout,
             /*handle=*/handle);
  EXPECT_IS_OK(BuildHloModule(b));
}

TEST(XlaBuilderTest, UnboundedSendWithToken) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  ChannelHandle handle;
  handle.set_handle(1);
  handle.set_type(ChannelHandle::DEVICE_TO_DEVICE);
  SendWithToken(/*operand=*/Parameter(&b, 0, operand, "operand"),
                /*token=*/CreateToken(&b), /*handle=*/handle);
  EXPECT_IS_OK(BuildHloModule(b));
}

TEST(XlaBuilderTest, UnboundedSlice) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[1, <=3, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[1, <=2, 3]"));
  Slice(Parameter(&b, 0, operand, "operand"),
        /*start_indices=*/{0, 1, 2},
        /*limit_indices=*/{1, 3, 5},
        /*strides=*/{1, 1, 1});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedSort) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?, 10]"));

  XlaComputation comparator;
  {
    const std::unique_ptr<XlaBuilder> sub_builder =
        b.CreateSubBuilder("compare");
    Compare(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(F32),
                      "arg0"),
            Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(F32),
                      "arg1"),
            ComparisonDirection::kLt);
    TF_ASSERT_OK_AND_ASSIGN(comparator, sub_builder->Build());
  }

  Sort({Parameter(&b, 0, operand, "operand")}, comparator,
       /*dimension=*/0, /*is_stable=*/true);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedTranspose) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand,
                          ParseShape("f32[1, ?, 2, ?, <=2]{4,3,2,1,0}"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("f32[<=2, 1, ?, 2, ?]{0,2,3,4,1}"));
  Transpose(Parameter(&b, 0, operand, "operand"),
            /*permutation=*/{4, 0, 3, 2, 1});
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedTriangularSolve) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape a_shape, ParseShape("f32[?, 10]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape b_shape, ParseShape("f32[10, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[10, ?]"));
  TriangularSolveOptions options;
  TriangularSolve(Parameter(&b, 0, a_shape, "a"),
                  Parameter(&b, 1, b_shape, "b"),
                  /*left_side=*/true, /*lower*/ true, /*unit_diagonal=*/false,
                  TriangularSolveOptions::TRANSPOSE);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedTuple) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape operand, ParseShape("f32[?, 10]"));
  const Shape expected = ShapeUtil::MakeTupleShape({operand});
  Tuple(&b, {Parameter(&b, 0, operand, "operand")});
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedWhile) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape init, ParseShape("f32[?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected, ParseShape("f32[?]"));

  XlaComputation add;
  {
    const std::unique_ptr<XlaBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(Parameter(sub_builder.get(), 0, ShapeUtil::MakeScalarShape(F32),
                  "arg0"),
        Parameter(sub_builder.get(), 1, ShapeUtil::MakeScalarShape(F32),
                  "arg1"));
    TF_ASSERT_OK_AND_ASSIGN(add, sub_builder->Build());
  }

  XlaComputation condition;
  {
    const std::unique_ptr<XlaBuilder> sub_builder =
        b.CreateSubBuilder("compare");
    Ge(/*lhs=*/ConstantR0<float>(sub_builder.get(), 10.0f),
       /*rhs=*/Reduce(/*operand=*/Parameter(sub_builder.get(), 0, init, "prev"),
                      ConstantR0<float>(sub_builder.get(), 0.0f), add,
                      /*dimensions_to_reduce=*/{0}));
    TF_ASSERT_OK_AND_ASSIGN(condition, sub_builder->Build());
  }

  XlaComputation body;
  {
    const std::unique_ptr<XlaBuilder> sub_builder = b.CreateSubBuilder("add");
    Add(ConstantR1<float>(sub_builder.get(), {1.0f}),
        Parameter(sub_builder.get(), 0, init, "prev"),
        /*broadcast_dimensions=*/{0});
    TF_ASSERT_OK_AND_ASSIGN(body, sub_builder->Build());
  }

  While(condition, body, Parameter(&b, 0, init, "init"));
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

TEST(XlaBuilderTest, UnboundedXor) {
  XlaBuilder b(TestName());
  TF_ASSERT_OK_AND_ASSIGN(const Shape lhs,
                          ParseShape("s32[1, ?, 2, ?, <=2, ?, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape rhs,
                          ParseShape("s32[?, 1, ?, 2, ?, <=2, ?]"));
  TF_ASSERT_OK_AND_ASSIGN(const Shape expected,
                          ParseShape("s32[?, ?, 2, 2, <=2, <=2, ?]"));
  Xor(Parameter(&b, 0, lhs, "lhs"), Parameter(&b, 1, rhs, "rhs"),
      /*broadcast_dimensions=*/empty_array);
  TF_ASSERT_OK_AND_ASSIGN(const std::unique_ptr<HloModule> module,
                          BuildHloModule(b));
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::Op().WithShapeEqualTo(&expected)));
}

INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, XlaBuilderUnboundedUnaryOpTest,
    ::testing::ValuesIn<UnaryOpTestCase>(
        {{"f32[?]", "f32[?]", &Abs},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Cbrt(x); }},
         {"f32[?]", "f32[?]", &Ceil},
         {"u32[?]", "u32[?]", &Clz},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Cos(x); }},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Erf(x); }},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Exp(x); }},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Expm1(x); }},
         {"f32[?]", "f32[?]", &Floor},
         {"f32[?]", "f32[?]", &Imag},
         {"f32[?]", "pred[?]", &IsFinite},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Log(x); }},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Log1p(x); }},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Logistic(x); }},
         {"f32[?]", "f32[?]", &Neg},
         {"s32[?]", "s32[?]", &Not},
         {"u32[?]", "u32[?]", &PopulationCount},
         {"f32[?]", "f32[?]", &Real},
         {"f32[?]", "f32[?]", &Round},
         {"f32[?]", "f32[?]", &RoundNearestEven},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Rsqrt(x); }},
         {"f32[?]", "f32[?]", &Sign},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Sin(x); }},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Sqrt(x); }},
         {"f32[?]", "f32[?]", [](XlaOp x) { return Tanh(x); }}}));

INSTANTIATE_TEST_SUITE_P(
    UnboundedDynamism, XlaBuilderUnboundedBinaryOpTest,
    ::testing::ValuesIn<BinaryOpTestCase>({
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &Add},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &Add},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &Atan2},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "c64[?, ?, 2, 2, <=2, <=2, ?]",
         &Complex},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "c64[?, 10]", &Complex},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &Div},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &Div},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &Max},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &Max},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &Min},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &Min},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &Mul},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &Mul},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "pred[?, 10]", &Ne},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &Pow},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &Pow},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &Rem},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &Rem},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &ShiftLeft},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &ShiftLeft},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &ShiftRightArithmetic},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &ShiftRightArithmetic},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &ShiftRightLogical},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &ShiftRightLogical},
        {"f32[1, ?, 2, ?, <=2, ?, ?]", "f32[?, 1, ?, 2, ?, <=2, ?]",
         /*broadcast_dimensions=*/empty_array, "f32[?, ?, 2, 2, <=2, <=2, ?]",
         &Sub},
        {"f32[?, 10]", "f32[1]", /*broadcast_dimensions=*/zero_array,
         "f32[?, 10]", &Sub},
    }));

TEST(XlaBuilderTest, UnorderedInfeed) {
  XlaBuilder b(TestName());
  const Shape s = ShapeUtil::MakeShape(PRED, {});
  XlaOp infeed0 = Infeed(&b, s);
  XlaOp infeed1 = Infeed(&b, s);
  And(infeed0, infeed1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  // infeed0 should use an arbitrary token.
  // infeed1 should use the token produced by infeed0.
  EXPECT_THAT(GetRoot(*module),
              GmockMatch(m::And(
                  m::GetTupleElement(m::Infeed(m::AfterAll()), 0),
                  m::GetTupleElement(
                      m::Infeed(m::GetTupleElement(m::Infeed(), 1)), 0))));
}

TEST(XlaBuilderTest, UnorderedOutfeed) {
  XlaBuilder b(TestName());
  const Shape s = ShapeUtil::MakeShape(S32, {});
  XlaOp param0 = Parameter(&b, 0, s, "p0");
  XlaOp param1 = Parameter(&b, 1, s, "p1");
  Outfeed(param0, s, "");
  Outfeed(param1, s, "");
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  // outfeed0 should use an arbitrary token.
  // outfeed1 should use the token produced by infeed0.
  HloInstruction* p1 = module->entry_computation()->parameter_instruction(1);
  EXPECT_EQ(p1->user_count(), 1);
  EXPECT_THAT(p1->users()[0], GmockMatch(m::Outfeed(
                                  m::Parameter(1),
                                  m::Outfeed(m::Parameter(0), m::AfterAll()))));
}

TEST(XlaBuilderTest, InfeedTokenSharding) {
  XlaBuilder b(TestName());
  b.SetSharding(sharding_builder::Replicate());
  const Shape s = ShapeUtil::MakeShape(PRED, {});
  XlaOp infeed0 = Infeed(&b, s);
  XlaOp infeed1 = Infeed(&b, s);
  And(infeed0, infeed1);
  TF_ASSERT_OK_AND_ASSIGN(const auto module, BuildHloModule(b));
  TF_ASSERT_OK_AND_ASSIGN(
      const auto token_sharding,
      HloSharding::FromProto(sharding_builder::AssignDevice(0)));
  for (const HloInstruction* instruction :
       module->entry_computation()->instructions()) {
    if (instruction->shape().IsToken()) {
      ASSERT_TRUE(instruction->has_sharding());
      EXPECT_EQ(instruction->sharding(), token_sharding);
    }
  }
}

}  // namespace
}  // namespace xla
