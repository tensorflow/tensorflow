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

#include "tensorflow/compiler/xla/client/xla_client/xla_builder.h"

#include <string>

#include "tensorflow/compiler/xla/legacy_flags/debug_options_flags.h"
#include "tensorflow/compiler/xla/service/hlo_matchers.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/test.h"
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
                            proto, legacy_flags::GetDebugOptionsFromFlags()));
    return HloModule::CreateFromProto(proto, config);
  }

  // Returns the name of the test currently being run.
  string TestName() const {
    return ::testing::UnitTest::GetInstance()->current_test_info()->name();
  }
};

TEST_F(XlaBuilderTest, OnePlusTwo) {
  XlaBuilder b(TestName());
  b.Add(b.ConstantR0<float>(1.0), b.ConstantR0<float>(2.0));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Constant(), op::Constant()));
}

TEST_F(XlaBuilderTest, ParamPlusConstantHasScalarBroadcast) {
  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {3, 5}), "x");
  b.Add(x, b.ConstantR0<float>(1.0));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Parameter(), op::Broadcast(op::Constant())));
}

TEST_F(XlaBuilderTest, ParamPlusParamHasBroadcast) {
  XlaBuilder b(TestName());
  const auto& x_shape = ShapeUtil::MakeShape(S32, {2, 4, 6});
  const auto& y_shape = ShapeUtil::MakeShape(S32, {2, 4});
  auto x = b.Parameter(0, x_shape, "x");
  auto y = b.Parameter(1, y_shape, "y");
  auto add = b.Add(x, y, /*broadcast_dimensions=*/{0, 1});

  TF_ASSERT_OK_AND_ASSIGN(auto add_shape, add.GetShape());
  EXPECT_TRUE(ShapeUtil::Equal(add_shape, x_shape));

  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Parameter(0), op::Broadcast(op::Parameter(1))));
}

TEST_F(XlaBuilderTest, XPlusX) {
  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(S32, {1, 3, 5, 7}), "x");
  b.Add(x, x);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Parameter(0), op::Parameter(0)));
}

TEST_F(XlaBuilderTest, ShapeInferenceError) {
  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(U32, {2, 4, 6}), "x");
  auto y = b.Parameter(1, ShapeUtil::MakeShape(U32, {2, 4}), "y");
  b.Add(x, y);
  auto statusor = BuildHloModule(&b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(), HasSubstr("shape inference"));
}

TEST_F(XlaBuilderTest, ParameterAlreadyRegistered) {
  XlaBuilder b_call("add");
  b_call.Parameter(0, ShapeUtil::MakeShape(PRED, {}), "x");

  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(PRED, {}), "x");
  auto y = b.Parameter(0, ShapeUtil::MakeShape(PRED, {}), "y");
  b.Add(x, y);
  auto statusor = BuildHloModule(&b);
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("parameter 0 already registered"));
}

TEST_F(XlaBuilderTest, Call) {
  XlaBuilder b_call("the_only_to_apply");
  auto p0 = b_call.Parameter(0, ShapeUtil::MakeShape(F32, {}), "p0");
  auto p1 = b_call.Parameter(1, ShapeUtil::MakeShape(F32, {}), "p1");
  b_call.Add(p0, p1);
  TF_ASSERT_OK_AND_ASSIGN(auto call, b_call.Build());
  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {}), "x");
  auto y = b.Parameter(1, ShapeUtil::MakeShape(F32, {}), "y");
  auto one = b.ConstantR0<float>(1);
  auto two = b.ConstantR0<float>(2);
  b.Add(b.Call(call, {x, y}), b.Call(call, {one, two}));
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Add(op::Call(op::Parameter(), op::Parameter()),
                            op::Call(op::Constant(), op::Constant())));
}

TEST_F(XlaBuilderTest, BinopHasDegenerateBroadcast) {
  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {1, 2, 3}), "x");
  auto y = b.Parameter(1, ShapeUtil::MakeShape(F32, {1, 2, 1}), "y");
  b.Add(x, y);
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
  auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {2, 3}), "x");
  auto y = b.Parameter(1, ShapeUtil::MakeShape(F32, {2, 1, 4}), "y");
  b.Add(x, y, /*broadcast_dimensions=*/{0, 1});
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

TEST_F(XlaBuilderTest, OperandFromWrongBuilder) {
  XlaBuilder b1("b1");
  auto p0 = b1.Parameter(0, ShapeUtil::MakeShape(F32, {}), "p0");
  XlaBuilder builder("main");
  builder.Add(p0, p0);
  auto statusor = builder.Build();
  ASSERT_FALSE(statusor.ok());
  EXPECT_THAT(statusor.status().error_message(),
              HasSubstr("Do not add XlaOp from builder b1 to builder main"));
}

TEST_F(XlaBuilderTest, ReshapeDefaultOrder) {
  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {2, 3, 5, 7}), "x");
  b.Reshape(x, /*new_sizes=*/{6, 35});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Reshape(op::Parameter()));
}

TEST_F(XlaBuilderTest, ReshapeHasTranspose) {
  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {2, 3, 5, 7}), "x");
  b.Reshape(x, /*dimensions=*/{3, 2, 1, 0}, /*new_sizes=*/{6, 35});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Reshape(op::Transpose(op::Parameter())));
}

TEST_F(XlaBuilderTest, Transpose) {
  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(F32, {5, 7}), "x");
  b.Transpose(x, /*permutation=*/{1, 0});
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  EXPECT_THAT(root, op::Transpose(op::Parameter()));
}

// TODO(b/65209188): Create a dedicated lowering for Xor.
TEST_F(XlaBuilderTest, Xor) {
  XlaBuilder b(TestName());
  auto x = b.Parameter(0, ShapeUtil::MakeShape(PRED, {}), "x");
  auto y = b.Parameter(1, ShapeUtil::MakeShape(PRED, {}), "y");
  b.Xor(x, y);
  TF_ASSERT_OK_AND_ASSIGN(auto module, BuildHloModule(&b));
  auto root = module->entry_computation()->root_instruction();
  LOG(ERROR) << module->ToString();
  EXPECT_THAT(root,
              op::Or(op::And(op::Not(op::Parameter(0)), op::Parameter(1)),
                     op::And(op::Parameter(0), op::Not(op::Parameter(1)))));
}

}  // namespace
}  // namespace xla
