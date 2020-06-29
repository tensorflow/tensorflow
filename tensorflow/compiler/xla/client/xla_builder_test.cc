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

#include "tensorflow/compiler/xla/client/xla_builder.h"

#include <string>

#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
#include "tensorflow/compiler/xla/test_helpers.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::HasSubstr;

// TODO(b/74197823): Move the tests to service/.
class XlaBuilderTest : public ::testing::Test {
 protected:
  StatusOr<std::unique_ptr<HloModule>> BuildHloModule(XlaBuilder* b) {
    TF_ASSIGN_OR_RETURN(XlaComputation computation,
                        b->Build(/*remove_dynamic_dimensions=*/false));
    const HloModuleProto& proto = computation.proto();
    TF_ASSIGN_OR_RETURN(const auto& config,
                        HloModule::CreateModuleConfigFromProto(
                            proto, GetDebugOptionsFromFlags()));
    return HloModule::CreateFromProto(proto, config);
  }

  // Overload which explicitly specifies the root instruction.
  StatusOr<std::unique_ptr<HloModule>> BuildHloModule(XlaBuilder* b,
                                                      XlaOp root) {
    TF_ASSIGN_OR_RETURN(XlaComputation computation,
                        b->Build(root, /*remove_dynamic_dimensions=*/false));
    const HloModuleProto& proto = computation.proto();
    TF_ASSIGN_OR_RETURN(const auto& config,
                        HloModule::CreateModuleConfigFromProto(
                            proto, GetDebugOptionsFromFlags()));
    return HloModule::CreateFromProto(proto, config);
  }

  // Returns the name of the test currently being run.
  string TestName() const {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }
};

TEST_F(XlaBuilderTest, OnePlusTwo) {
  XlaBuilder b(TestName());
  Add(ConstantR0<float>(&b, 1.0), ConstantR0<float>(&b, 2.0));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Constant(), op::Constant()));
}

TEST_F(XlaBuilderTest, UnaryOperatorsBuildExpectedHLO) {
  auto test_unary_operator =
      [&](std::function<XlaOp(XlaOp)> op,
          ::testing::Matcher<const ::xla::HloInstruction*> matches_pattern) {
        XlaBuilder b(TestName());
        op(ConstantR0<int32>(&b, 1));
        TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
        auto root = module->entry_computation()->root_instruction();
        EXPECT_THAT(root, matches_pattern);
      };
  test_unary_operator([](XlaOp x) { return -x; }, op::Negate(op::Constant()));
  test_unary_operator([](XlaOp x) { return ~x; }, op::Not(op::Constant()));
}

TEST_F(XlaBuilderTest, BinaryOperatorsBuildExpectedHLO) {
  auto test_binary_operator =
      [&](std::function<XlaOp(XlaOp, XlaOp)> op,
          ::testing::Matcher<const ::xla::HloInstruction*> matches_pattern) {
        XlaBuilder b(TestName());
        op(ConstantR0<int32>(&b, 1), ConstantR0<int32>(&b, 2));
        TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
        auto root = module->entry_computation()->root_instruction();
        EXPECT_THAT(root, matches_pattern);
      };

  test_binary_operator([](XlaOp x, XlaOp y) { return x + y; },
                       op::Add(op::Constant(), op::Constant()));
  test_binary_operator([](XlaOp x, XlaOp y) { return x - y; },
                       op::Subtract(op::Constant(), op::Constant()));
  test_binary_operator([](XlaOp x, XlaOp y) { return x * y; },
                       op::Multiply(op::Constant(), op::Constant()));
  test_binary_operator([](XlaOp x, XlaOp y) { return x / y; },
                       op::Divide(op::Constant(), op::Constant()));

  test_binary_operator([](XlaOp x, XlaOp y) { return x & y; },
                       op::And(op::Constant(), op::Constant()));
  test_binary_operator([](XlaOp x, XlaOp y) { return x | y; },
                       op::Or(op::Constant(), op::Constant()));
  test_binary_operator([](XlaOp x, XlaOp y) { return x ^ y; },
                       op::Xor(op::Constant(), op::Constant()));
  test_binary_operator([](XlaOp x, XlaOp y) { return x << y; },
                       op::ShiftLeft(op::Constant(), op::Constant()));
  test_binary_operator(
      [](XlaOp x, XlaOp y) { return x >> y; },
      op::ShiftRightArithmetic(op::Constant(), op::Constant()));

  auto test_unsigned_binary_operator =
      [&](std::function<XlaOp(XlaOp, XlaOp)> op,
          ::testing::Matcher<const ::xla::HloInstruction*> matches_pattern) {
        XlaBuilder b(TestName());
        op(ConstantR0<uint32>(&b, 1), ConstantR0<uint32>(&b, 2));
        TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
        auto root = module->entry_computation()->root_instruction();
        EXPECT_THAT(root, matches_pattern);
      };
  test_unsigned_binary_operator(
      [](XlaOp x, XlaOp y) { return x >> y; },
      op::ShiftRightLogical(op::Constant(), op::Constant()));
}

TEST_F(XlaBuilderTest, VariadicAnd) {
  XlaBuilder b(TestName());
  Shape s = ShapeUtil::MakeShape(PRED, {});
  And(Parameter(&b, 0, s, "p0"), Parameter(&b, 1, s, "p1"),
      Parameter(&b, 2, s, "p2"));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  // Don't specify in the test whether And(x, y, z) is right- or
  // left-associative; accept either one.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      ::testing::AnyOf(op::And(op::Parameter(0),
                               op::And(op::Parameter(1), op::Parameter(2))),
                       op::And(op::And(op::Parameter(0), op::Parameter(1)),
                               op::Parameter(2))));
}

TEST_F(XlaBuilderTest, VariadicOr) {
  XlaBuilder b(TestName());
  Shape s = ShapeUtil::MakeShape(PRED, {});
  Or(Parameter(&b, 0, s, "p0"), Parameter(&b, 1, s, "p1"),
     Parameter(&b, 2, s, "p2"));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  // Don't specify in the test whether Or(x, y, z) is right- or
  // left-associative; accept either one.
  EXPECT_THAT(
      module->entry_computation()->root_instruction(),
      ::testing::AnyOf(
          op::Or(op::Parameter(0), op::Or(op::Parameter(1), op::Parameter(2))),
          op::Or(op::Or(op::Parameter(0), op::Parameter(1)),
                 op::Parameter(2))));
}

TEST_F(XlaBuilderTest, ShiftRightOperatorOnNonIntegerProducesError) {
  XlaBuilder b(TestName());
  ConstantR0<float>(&b, 1) >> ConstantR0<float>(&b, 2);
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr("Argument to >> operator does not have an integral type"));
}

TEST_F(XlaBuilderTest, ParamPlusConstantHasScalarBroadcast) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {3, 5}), "x");
  Add(x, ConstantR0<float>(&b, 1.0));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Parameter(), op::Broadcast(op::Constant())));
}

TEST_F(XlaBuilderTest, ParamPlusParamHasBroadcast) {
  XlaBuilder b(TestName());
  const auto& x_shape = ShapeUtil::MakeShape(S32, {2, 4, 6});
  const auto& y_shape = ShapeUtil::MakeShape(S32, {2, 4});
  auto x = Parameter(&b, 0, x_shape, "x");
  auto y = Parameter(&b, 1, y_shape, "y");
  auto add = Add(x, y, /*broadcast_dimensions=*/{0, 1});

  TF_ASSERT_OK_AND_ASSIGN(auto add_shape, b.GetShape(add));
  EXPECT_TRUE(ShapeUtil::Equal(add_shape, x_shape));

  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Parameter(0), op::Broadcast(op::Parameter(1))));
}

TEST_F(XlaBuilderTest, XPlusX) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(S32, {1, 3, 5, 7}), "x");
  Add(x, x);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Parameter(0), op::Parameter(0)));
}

TEST_F(XlaBuilderTest, ShapeInferenceError) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(U32, {2, 4, 6}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(U32, {2, 4}), "y");
  Add(x, y);
  auto statusor = BuildHloModule(&b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(), HasSubstr("shape inference"));
}

TEST_F(XlaBuilderTest, DynamicDimensionReshapeToR0) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {1}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "dyn_dim");
  auto dx = SetDimensionSize(x, y, 0);
  Reshape(dx, {});
  auto statusor = BuildHloModule(&b);
  ASSERT_TRUE(statusor.ok());
}

TEST_F(XlaBuilderTest, ParameterAlreadyRegistered) {
  XlaBuilder b_call("add");
  Parameter(&b_call, 0, ShapeUtil::MakeShape(PRED, {}), "x");

  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "x");
  auto y = Parameter(&b, 0, ShapeUtil::MakeShape(PRED, {}), "y");
  Add(x, y);
  auto statusor = BuildHloModule(&b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("parameter 0 already registered"));
}

TEST_F(XlaBuilderTest, Call) {
  XlaBuilder b_call("the_only_to_apply");
  auto p0 = Parameter(&b_call, 0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto p1 = Parameter(&b_call, 1, ShapeUtil::MakeShape(F32, {}), "p1");
  Add(p0, p1);
  TF_ASSERT_OK_AND_ASSIGN(auto call, b_call.Build());
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "y");
  auto one = ConstantR0<float>(&b, 1);
  auto two = ConstantR0<float>(&b, 2);
  Add(Call(&b, call, {x, y}), Call(&b, call, {one, two}));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Call(op::Parameter(), op::Parameter()),
                            op::Call(op::Constant(), op::Constant())));
}

TEST_F(XlaBuilderTest, BinopHasDegenerateBroadcast) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {1, 2, 3}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {1, 2, 1}), "y");
  Add(x, y);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));

  // Expected:
  //
  //  x: f32[1,2,3]  y: f32[1,2,1]
  //      |               |
  //      |          reshape: f32[1,2]
  //      |               |
  //      |          broadcast: f32[1,2,3]
  //       \             /
  //            add
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Parameter(0),
                            op::Broadcast(op::Reshape(op::Parameter(1)))));
}

TEST_F(XlaBuilderTest, BinopHasInDimAndDegenerateBroadcast) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 3}), "x");
  auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {2, 1, 4}), "y");
  Add(x, y, /*broadcast_dimensions=*/{0, 1});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));

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
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Broadcast(op::Parameter(0)),
                            op::Broadcast(op::Reshape(op::Parameter(1)))));
}

TEST_F(XlaBuilderTest, BroadcastInDim) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 3}), "x");
  BroadcastInDim(x, {2, 4, 3},
                 /*broadcast_dimensions=*/{0, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Broadcast());
}

TEST_F(XlaBuilderTest, BroadcastInDimWithDegeneratedDim) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 1, 4}), "x");
  BroadcastInDim(x, {2, 3, 4},
                 /*broadcast_dimensions=*/{0, 1, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  EXPECT_THAT(module->entry_computation()->root_instruction(),
              op::Broadcast(op::Reshape(op::Broadcast())));
}

TEST_F(XlaBuilderTest, BroadcastInDimWithNegativeSize) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 1, 4}), "x");
  BroadcastInDim(x, {-3, 3, 4},
                 /*broadcast_dimensions=*/{0, 1, 2});
  auto statusor = BuildHloModule(&b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("shape's dimensions must not be < 0"));
}

TEST_F(XlaBuilderTest, OperandFromWrongBuilder) {
  XlaBuilder b1("b1");
  auto p0 = Parameter(&b1, 0, ShapeUtil::MakeShape(F32, {}), "p0");
  XlaBuilder builder("main");
  auto p = Parameter(&builder, 0, ShapeUtil::MakeShape(F32, {}), "p");
  Add(p, p0);
  auto statusor = builder.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(
      statusor.status().error_message(),
      HasSubstr(
          "built by builder 'b1', but is trying to use it in builder 'main'"));
}

TEST_F(XlaBuilderTest, ReshapeDefaultOrder) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 3, 5, 7}), "x");
  Reshape(x, /*new_sizes=*/{6, 35});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Reshape(op::Parameter()));
}

TEST_F(XlaBuilderTest, ReshapeHasTranspose) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {2, 3, 5, 7}), "x");
  Reshape(x, /*dimensions=*/{3, 2, 1, 0}, /*new_sizes=*/{6, 35});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Reshape(op::Transpose(op::Parameter())));
}

TEST_F(XlaBuilderTest, Transpose) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  Transpose(x, /*permutation=*/{1, 0});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Transpose(op::Parameter()));
}

TEST_F(XlaBuilderTest, AllGatherR1) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4}), "x");
  AllGather(x, /*all_gather_dimension=*/0, /*shard_count=*/4);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();

  EXPECT_EQ(root->opcode(), HloOpcode::kAllGather);
  EXPECT_TRUE(ShapeUtil::Equal(root->shape(), ShapeUtil::MakeShape(F32, {16})));
}

TEST_F(XlaBuilderTest, AllGatherR2) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4, 16}), "x");
  AllGather(x, /*all_gather_dimension=*/1, /*shard_count=*/4);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();

  EXPECT_EQ(root->opcode(), HloOpcode::kAllGather);
  EXPECT_TRUE(
      ShapeUtil::Equal(root->shape(), ShapeUtil::MakeShape(F32, {4, 64})));
}

TEST_F(XlaBuilderTest, AllToAll) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {4, 16}), "x");
  AllToAll(x, /*split_dimension=*/1, /*concat_dimension=*/0,
           /*split_count=*/2);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();

  // AllToAll is decomposed into slices -> all-to-all -> gte -> concat.
  EXPECT_EQ(root->opcode(), HloOpcode::kConcatenate);
  EXPECT_EQ(root->operand(0)->operand(0)->opcode(), HloOpcode::kAllToAll);
  EXPECT_TRUE(
      ShapeUtil::Equal(root->shape(), ShapeUtil::MakeShape(F32, {8, 8})));
}

TEST_F(XlaBuilderTest, CollectivePermute) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  CollectivePermute(x, {{0, 1}, {1, 2}, {2, 3}});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kCollectivePermute);
}

TEST_F(XlaBuilderTest, GetDimensionSize) {
  XlaBuilder b(TestName());
  auto x =
      Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}, {false, true}), "x");
  GetDimensionSize(x, 1);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kGetDimensionSize);
}

TEST_F(XlaBuilderTest, GetDimensionSizeConstant) {
  XlaBuilder b(TestName());
  auto x =
      Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}, {false, true}), "x");
  // Get dimension size from a contant dimension gives us a constant.
  GetDimensionSize(x, 0);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kConstant);
}

TEST_F(XlaBuilderTest, ReportError) {
  XlaBuilder b(TestName());
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  Add(b.ReportError(InvalidArgument("a test error")), x);
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(), HasSubstr("a test error"));
}

TEST_F(XlaBuilderTest, ReportErrorOrReturnHandlesNonErrors) {
  XlaBuilder b(TestName());
  StatusOr<XlaOp> op(ConstantR0<float>(&b, 1.0));
  Add(b.ReportErrorOrReturn(op), ConstantR0<float>(&b, 2.0));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Constant(), op::Constant()));
}

TEST_F(XlaBuilderTest, ReportErrorOrReturnHandlesErrors) {
  XlaBuilder b(TestName());
  StatusOr<XlaOp> op(InvalidArgument("a test error"));
  Add(b.ReportErrorOrReturn(op), ConstantR0<float>(&b, 2.0));
  auto statusor = b.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(), HasSubstr("a test error"));
}

TEST_F(XlaBuilderTest, BuildWithSpecificRoot) {
  XlaBuilder b(TestName());
  XlaOp constant = ConstantR0<float>(&b, 1.0);
  Add(constant, ConstantR0<float>(&b, 2.0));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b, /*root=*/constant));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Constant());
}

TEST_F(XlaBuilderTest, BuildWithSpecificRootAndMultipleParameters) {
  // Specifying a particular root in Build should still include all entry
  // parameters.
  XlaBuilder b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {42, 123});
  XlaOp x = Parameter(&b, 0, shape, "x");
  XlaOp y = Parameter(&b, 1, shape, "y");
  XlaOp z = Parameter(&b, 2, shape, "z");
  Add(x, Sub(y, z));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b, /*root=*/x));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Parameter());
  EXPECT_EQ(module->entry_computation()->num_parameters(), 3);
  EXPECT_EQ(module->entry_computation()->instruction_count(), 5);
}

TEST_F(XlaBuilderTest, BuildWithSpecificRootWithWrongBuilder) {
  XlaBuilder b(TestName());
  XlaBuilder other_b(TestName());
  const Shape shape = ShapeUtil::MakeShape(F32, {42, 123});

  Parameter(&b, 0, shape, "param");
  XlaOp other_param = Parameter(&other_b, 0, shape, "other_param");

  Status status = b.Build(other_param).status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(
      status.error_message(),
      ::testing::HasSubstr("root operation is not in this computation"));
}

TEST_F(XlaBuilderTest, ProtoMatches) {
  std::vector<XlaComputation> computations;
  for (int i = 0; i < 2; ++i) {
    XlaBuilder b_call("the_only_to_apply");
    auto p0 = Parameter(&b_call, 0, ShapeUtil::MakeShape(F32, {}), "p0");
    auto p1 = Parameter(&b_call, 1, ShapeUtil::MakeShape(F32, {}), "p1");
    Add(p0, Add(p1, p0));
    TF_ASSERT_OK_AND_ASSIGN(auto call, b_call.Build());
    XlaBuilder b(TestName());
    auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {}), "x");
    auto y = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {}), "y");
    auto one = ConstantR0<float>(&b, 1);
    auto two = ConstantR0<float>(&b, 2);
    Add(Call(&b, call, {x, y}), Call(&b, call, {one, two}));
    computations.push_back(b.Build().ValueOrDie());
  }
  auto c0_string = computations[0].proto().SerializeAsString();
  auto c1_string = computations[1].proto().SerializeAsString();
  EXPECT_EQ(c0_string, c1_string);
}

TEST_F(XlaBuilderTest, DynamicParameter) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}), ShapeUtil::MakeShape(F32, {6}, {true})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  Parameter(&b, 1, ShapeUtil::MakeShape(U32, {}), "p1");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/1,
                                   /*dynamic_size_param_index=*/{},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{1},
                                   /*target_dim_num=*/0));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b, /*root=*/p0));
  const Shape& param_shape = module->entry_computation()
                                 ->parameter_instruction(0)
                                 ->shape()
                                 .tuple_shapes(1);
  EXPECT_TRUE(param_shape.is_dynamic_dimension(0));
}

TEST_F(XlaBuilderTest, SetDimensionSize) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {10}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "p1");
  auto set_dim_size = SetDimensionSize(p0, p1, 0);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          BuildHloModule(&b, /*root=*/set_dim_size));
  const Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(root_shape.is_dynamic_dimension(0));
}

TEST_F(XlaBuilderTest, RemoveDimensionSize) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {10}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(S32, {}), "p1");
  auto set_dim_size = SetDimensionSize(p0, p1, 0);
  auto remove_dim_size = RemoveDynamicDimension(set_dim_size, 0);
  TF_ASSERT_OK_AND_ASSIGN(auto module,
                          BuildHloModule(&b, /*root=*/remove_dim_size));
  const Shape& root_shape =
      module->entry_computation()->root_instruction()->shape();
  // Dynamic dimension has been removed.
  EXPECT_FALSE(root_shape.is_dynamic_dimension(0));
}

TEST_F(XlaBuilderTest, DynamicUnary) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{1},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  auto gte = GetTupleElement(p0, 0);
  Neg(gte);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(0));
}

TEST_F(XlaBuilderTest, DynamicBinary) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5}, {true}),
       ShapeUtil::MakeShape(F32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{1},
                                   /*target_dim_num=*/0));
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(0));
}

TEST_F(XlaBuilderTest, DynamicBinaryHasBroadcast) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(F32, {5}, {true}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{1},
                                   /*target_dim_num=*/0));
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1, {0});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicBroadcast) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{1},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  auto gte = GetTupleElement(p0, 0);
  BroadcastInDim(gte, /*out_dim_size=*/{3, 5, 4},
                 /*broadcast_dimensions=*/{1, 2});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {false, true, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicBinaryHasDegenerateBroadcast) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {10}, {true}),
       ShapeUtil::MakeShape(F32, {1, 15}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{1},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Add(gte0, gte1, /*broadcast_dimensions=*/{0});  // f32[<=10, 15]
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicSelectOnlyPredDynamic) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(PRED, {10}, {true}),
       ShapeUtil::MakeShape(F32, {10}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{1},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);

  Select(gte0, gte1, gte1);

  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicPad) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5, 4}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pad_val = ConstantR0<float>(&b, -1);
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{1},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  auto gte = GetTupleElement(p0, 0);
  PaddingConfig padding_config;
  for (int i = 0; i < 2; i++) {
    auto dimension = padding_config.add_dimensions();
    dimension->set_edge_padding_low(0);
    dimension->set_edge_padding_high(0);
    dimension->set_interior_padding(0);
  }
  Pad(gte, pad_val, padding_config);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicConvolution) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {1, 2, 2, 128}, {true, false, false, false}),
       ShapeUtil::MakeShape(F32, {2, 2, 128, 8}, {false, false, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{3},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{1},
                                   /*target_dim_num=*/2));
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
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(),
                              {true, false, false, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicDot) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 3, 4}, {true, true, false}),
       ShapeUtil::MakeShape(F32, {2, 4, 5}, {true, false, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{1},
                                   /*target_dim_num=*/0));
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{3},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/1));

  auto lhs = GetTupleElement(p0, 0);
  auto rhs = GetTupleElement(p0, 1);
  DotDimensionNumbers dnums;
  dnums.add_lhs_contracting_dimensions(2);
  dnums.add_rhs_contracting_dimensions(1);
  dnums.add_lhs_batch_dimensions(0);
  dnums.add_rhs_batch_dimensions(0);
  DotGeneral(lhs, rhs, dnums);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {true, true, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicReduce) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {5, 4, 3}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto init = ConstantR0<float>(&b, 0);
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{1},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/1));
  auto gte = GetTupleElement(p0, 0);
  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, ShapeUtil::MakeShape(F32, {}), "x"),
      Parameter(&bsum, 1, ShapeUtil::MakeShape(F32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(auto sum, bsum.Build());
  Reduce(gte, init, sum, {0});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {true, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicReduceWindow) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 4, 8}, {true, false, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto init = ConstantR0<float>(&b, 0.f);
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{1},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  auto gte = GetTupleElement(p0, 0);
  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, ShapeUtil::MakeShape(F32, {}), "x"),
      Parameter(&bsum, 1, ShapeUtil::MakeShape(F32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(auto sum, bsum.Build());
  ReduceWindow(gte, init, sum, /*window_dimensions=*/{1, 2, 4},
               /*window_strides=*/{1, 1, 1}, Padding::kValid);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {true, false, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicSelectAndScatter) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 4, 8}, {true, false, false}),
       ShapeUtil::MakeShape(F32, {2, 2, 2}, {true, false, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto init = ConstantR0<float>(&b, 0.f);
  XlaBuilder bsum(TestName());
  Add(Parameter(&bsum, 0, ShapeUtil::MakeShape(F32, {}), "x"),
      Parameter(&bsum, 1, ShapeUtil::MakeShape(F32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(auto sum, bsum.Build());
  XlaBuilder bge(TestName());
  Ge(Parameter(&bge, 0, ShapeUtil::MakeShape(F32, {}), "x"),
     Parameter(&bge, 1, ShapeUtil::MakeShape(F32, {}), "y"));
  TF_ASSERT_OK_AND_ASSIGN(auto ge, bge.Build());

  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{1},
                                   /*target_dim_num=*/0));
  auto gte0 = GetTupleElement(p0, 0);
  auto source = GetTupleElement(p0, 1);
  SelectAndScatter(gte0, ge, {1, 2, 4}, {1, 2, 4}, Padding::kValid, source,
                   init, sum);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {true, false, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicReshape) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {2, 3, 4, 5, 6},
                            {false, false, true, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{1},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/2));
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/3));
  auto gte = GetTupleElement(p0, 0);  // f32[2, 3, <=4, <=5, 6]
  Reshape(gte, /*new_sizes=*/{6, 4, 5, 2, 3});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(1));
  EXPECT_TRUE(result_shape.is_dynamic_dimension(2));
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(),
                              {false, true, true, false, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicSelect) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(F32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pred = Parameter(&b, 1, ShapeUtil::MakeShape(PRED, {}), "pred");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/1));
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{3},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{1},
                                   /*target_dim_num=*/1));
  auto gte0 = GetTupleElement(p0, 0);
  auto gte1 = GetTupleElement(p0, 1);
  Select(pred, gte0, gte1);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(result_shape.is_dynamic_dimension(1));
  EXPECT_FALSE(result_shape.is_dynamic_dimension(2));
  EXPECT_TRUE(
      ContainersEqual(result_shape.dynamic_dimensions(), {false, true, false}))
      << result_shape;
}

TEST_F(XlaBuilderTest, DynamicSelectNotCompatible) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {4, 5, 6}, {false, true, false}),
       ShapeUtil::MakeShape(F32, {4, 5, 6}, {false, false, true}),
       ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  auto pred = Parameter(&b, 1, ShapeUtil::MakeShape(PRED, {}), "pred");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{2},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/1));
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{3},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{1},
                                   /*target_dim_num=*/2));
  auto gte0 = GetTupleElement(p0, 0);  // f32[4,<=5,6]
  auto gte1 = GetTupleElement(p0, 1);  // f32[4,5,<=6]
  Select(pred, gte0, gte1);
  Status status = BuildHloModule(&b).status();
  ASSERT_IS_OK(status);
}

TEST_F(XlaBuilderTest, DynamicTranspose) {
  XlaBuilder b(TestName());
  Shape tuple_param_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeShape(F32, {3, 5}, {true, false}),
       ShapeUtil::MakeShape(U32, {})});
  auto p0 = Parameter(&b, 0, tuple_param_shape, "p0");
  ASSERT_IS_OK(b.SetDynamicBinding(/*dynamic_size_param_num=*/0,
                                   /*dynamic_size_param_index=*/{1},
                                   /*target_param_num=*/0,
                                   /*target_param_index=*/{0},
                                   /*target_dim_num=*/0));
  auto gte = GetTupleElement(p0, 0);
  Transpose(gte, /*permutation=*/{1, 0});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  const Shape& result_shape =
      module->entry_computation()->root_instruction()->shape();
  EXPECT_TRUE(ContainersEqual(result_shape.dynamic_dimensions(), {false, true}))
      << result_shape;
}

TEST_F(XlaBuilderTest, AfterAllWithNonTokenOperands) {
  XlaBuilder b(TestName());
  AfterAll(&b, {CreateToken(&b), ConstantR0<float>(&b, 1.0)});
  Status status = b.Build().status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("All operands to AfterAll must be tokens"));
}

TEST_F(XlaBuilderTest, CheckInputOutputAlias) {
  XlaBuilder b(TestName());
  auto p0 = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {8, 4}), "p0");
  auto p1 = Parameter(&b, 1, ShapeUtil::MakeShape(F32, {8, 4}), "p1");
  auto add = Add(p0, p1);
  auto sub = Sub(p0, p1);
  auto root = Tuple(&b, {add, sub});

  b.SetUpAlias({1}, 0, {});
  b.SetUpAlias({0}, 1, {});

  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b, root));

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

TEST_F(XlaBuilderTest, SimpleSetFrontendAttributes) {
  XlaBuilder b(TestName());
  FrontendAttributes attributes;

  ConstantR0(&b, 0);  // No attribute set

  (*attributes.mutable_map())["attr_a"] = "a";
  b.SetFrontendAttributes(attributes);
  ConstantR0(&b, 0);  // One attribute: { "attr_a": "a" }

  b.ClearFrontendAttributes();
  ConstantR0(&b, 0);  // No attribute set

  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));

  std::vector<FrontendAttributes> expected{FrontendAttributes(), attributes,
                                           FrontendAttributes()};
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST_F(XlaBuilderTest, ComplexSetFrontendAttributes) {
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

  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  ExpectInstructionsAttributesMatch(*module, expected);
}

TEST_F(XlaBuilderTest, AddFrontendAttribute) {
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

  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  ExpectInstructionsAttributesMatch(*module, expected);
}
}  // namespace
}  // namespace xla
