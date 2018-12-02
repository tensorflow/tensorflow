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
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

namespace {

namespace op = xla::testing::opcode_matchers;

using ::testing::HasSubstr;

// TODO(b/74197823): Move the tests to service/.
class XlaBuilderTest : public ::testing::Test {
 protected:
  StatusOr<std::unique_ptr<HloModule>> BuildHloModule(XlaBuilder* b) {
    TF_ASSIGN_OR_RETURN(XlaComputation computation, b->Build());
    const HloModuleProto& proto = computation.proto();
    TF_ASSIGN_OR_RETURN(const auto& config,
                        HloModule::CreateModuleConfigFromProto(
                            proto, GetDebugOptionsFromFlags()));
    return HloModule::CreateFromProto(proto, config);
  }

  // Overload which explicitly specifies the root instruction.
  StatusOr<std::unique_ptr<HloModule>> BuildHloModule(XlaBuilder* b,
                                                      XlaOp root) {
    TF_ASSIGN_OR_RETURN(XlaComputation computation, b->Build(root));
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
  // first do the in-dim broadcast then convert the degnerate broadcast into a
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
  auto x = Parameter(&b, 0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  GetDimensionSize(x, 1);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_EQ(root->opcode(), HloOpcode::kGetDimensionSize);
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

TEST_F(XlaBuilderTest, AfterAllWithNonTokenOperands) {
  XlaBuilder b(TestName());
  AfterAll(&b, {CreateToken(&b), ConstantR0<float>(&b, 1.0)});
  Status status = b.Build().status();
  ASSERT_IS_NOT_OK(status);
  EXPECT_THAT(status.error_message(),
              ::testing::HasSubstr("All operands to AfterAll must be tokens"));
}

}  // namespace
}  // namespace xla
