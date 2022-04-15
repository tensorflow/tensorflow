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

#include <memory>

#include "absl/memory/memory.h"
#include "tensorflow/compiler/tf2xla/xla_expression.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/tests/literal_test_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

class XlaExpressionTest : public ::testing::Test {
 protected:
  void SetUp() override {
    client_ = xla::ClientLibrary::LocalClientOrDie();
    builder_ = absl::make_unique<xla::XlaBuilder>("acomputation");
    constant_ = test::AsScalar<int32>(42);
    op_ = xla::ConstantR0<int32>(builder_.get(), 7);
    non_constant_op_ = xla::Parameter(
        builder_.get(), 0, xla::ShapeUtil::MakeShape(xla::F32, {}), "x");
    resource_ = absl::make_unique<XlaResource>(
        XlaResource::kVariable, /*arg_num=*/0, /*name=*/string("avariable"),
        DT_INT32, TensorShape({17, 3}), op_, /*tensor_array_size=*/-1,
        /*tensor_array_gradients=*/std::set<string>(),
        /*tensor_array_multiple_writes_aggregate=*/false);
  }

  xla::Client* client_;
  std::unique_ptr<xla::XlaBuilder> builder_;
  Tensor constant_;
  xla::XlaOp op_;
  xla::XlaOp non_constant_op_;
  std::unique_ptr<XlaResource> resource_;
};

TEST_F(XlaExpressionTest, Kind) {
  EXPECT_TRUE(XlaExpression::Kind::kInvalid == XlaExpression().kind());
  EXPECT_TRUE(XlaExpression::Kind::kInvalid == XlaExpression::Invalid().kind());
  EXPECT_TRUE(XlaExpression::Kind::kConstant ==
              XlaExpression::Constant(constant_).kind());
  EXPECT_TRUE(XlaExpression::Kind::kXlaOp ==
              XlaExpression::XlaOp(op_, DT_INT32).kind());
  EXPECT_TRUE(XlaExpression::Kind::kResource ==
              XlaExpression::Resource(resource_.get()).kind());
}

TEST_F(XlaExpressionTest, HumanString) {
  EXPECT_EQ("invalid", XlaExpression().HumanString());
  EXPECT_EQ("invalid", XlaExpression::Invalid().HumanString());
  EXPECT_EQ("constant", XlaExpression::Constant(constant_).HumanString());
  EXPECT_EQ("xla_op", XlaExpression::XlaOp(op_, DT_INT32).HumanString());
  EXPECT_EQ("resource", XlaExpression::Resource(resource_.get()).HumanString());
}

TEST_F(XlaExpressionTest, AsXlaOp) {
  xla::XlaOp op_as_op =
      XlaExpression::XlaOp(op_, DT_INT32).AsXlaOp(builder_.get());
  EXPECT_TRUE(op_.IsIdenticalTo(op_as_op));

  xla::XlaOp const_as_op =
      XlaExpression::Constant(constant_).AsXlaOp(builder_.get());
  TF_ASSERT_OK_AND_ASSIGN(xla::XlaComputation computation,
                          builder_->BuildConstantSubGraph(const_as_op));
  TF_ASSERT_OK_AND_ASSIGN(xla::Literal value,
                          client_->ComputeConstant(computation));
  EXPECT_TRUE(xla::LiteralTestUtil::Equal(xla::LiteralUtil::CreateR0<int32>(42),
                                          value));
}

TEST_F(XlaExpressionTest, GetShape) {
  EXPECT_FALSE(XlaExpression().GetShape().ok());
  EXPECT_FALSE(XlaExpression::Invalid().GetShape().ok());

  TF_ASSERT_OK_AND_ASSIGN(TensorShape resource_shape,
                          XlaExpression::Resource(resource_.get()).GetShape());
  EXPECT_EQ(TensorShape({}), resource_shape);

  TF_ASSERT_OK_AND_ASSIGN(TensorShape op_shape,
                          XlaExpression::XlaOp(op_, DT_INT32).GetShape());
  EXPECT_EQ(TensorShape({}), op_shape);

  TF_ASSERT_OK_AND_ASSIGN(TensorShape constant_shape,
                          XlaExpression::Constant(constant_).GetShape());
  EXPECT_EQ(TensorShape({}), constant_shape);
}

TEST_F(XlaExpressionTest, ResolveConstant) {
  EXPECT_FALSE(XlaExpression().ResolveConstant(client_).ok());
  EXPECT_FALSE(XlaExpression::Invalid().ResolveConstant(client_).ok());

  EXPECT_FALSE(XlaExpression::Resource(resource_.get())
                   .ResolveConstant(client_)
                   ->has_value());

  TF_ASSERT_OK_AND_ASSIGN(
      absl::optional<Tensor> op_constant,
      XlaExpression::XlaOp(op_, DT_INT32).ResolveConstant(client_));
  ASSERT_TRUE(op_constant.has_value());
  test::ExpectTensorEqual<int32>(test::AsScalar<int32>(7), *op_constant);

  TF_ASSERT_OK_AND_ASSIGN(absl::optional<Tensor> op_nonconstant,
                          XlaExpression::XlaOp(non_constant_op_, DT_FLOAT)
                              .ResolveConstant(client_));
  EXPECT_FALSE(op_nonconstant.has_value());

  TF_ASSERT_OK_AND_ASSIGN(
      absl::optional<Tensor> constant_constant,
      XlaExpression::Constant(constant_).ResolveConstant(client_));
  ASSERT_TRUE(constant_constant.has_value());
  test::ExpectTensorEqual<int32>(constant_, *constant_constant);
}

TEST_F(XlaExpressionTest, ResolveConstantOnResource) {
  XlaExpression constant_resource =
      XlaExpression::ConstantResource(constant_, resource_.get());
  EXPECT_TRUE(constant_resource.ResolveConstant(client_).ok());
  EXPECT_TRUE(resource_->SetZeroValue(builder_.get()).ok());
  LOG(ERROR) << "Resource is overwritten: " << resource_->IsOverwritten();
  StatusOr<absl::optional<Tensor>> resolved_constant =
      constant_resource.ResolveConstant(client_);
  EXPECT_TRUE(resolved_constant.ok());
  EXPECT_FALSE(resolved_constant->has_value());
}

}  // namespace
}  // namespace tensorflow
