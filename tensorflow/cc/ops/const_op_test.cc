/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

namespace {

template <typename T>
void ExpectNodeEqual(const Node* n, gtl::ArraySlice<T> values,
                     TensorShape shape) {
  EXPECT_TRUE(n->IsConstant());
  Tensor tensor;
  TF_EXPECT_OK(GetNodeAttr(n->attrs(), "value", &tensor));
  DataType dtype;
  TF_EXPECT_OK(GetNodeAttr(n->attrs(), "dtype", &dtype));
  EXPECT_EQ(tensor.dtype(), dtype);
  test::ExpectTensorEqual<T>(tensor, test::AsTensor(values, shape));
}

void ExpectTypeAndShape(const Node* n, DataType expected_dtype,
                        TensorShape expected_shape) {
  EXPECT_TRUE(n->IsConstant());
  Tensor tensor;
  TF_EXPECT_OK(GetNodeAttr(n->attrs(), "value", &tensor));
  DataType dtype;
  TF_EXPECT_OK(GetNodeAttr(n->attrs(), "dtype", &dtype));
  EXPECT_EQ(dtype, expected_dtype);
  EXPECT_EQ(expected_shape, TensorShape(tensor.shape()));
}

}  // namespace

TEST(ConstOpTest, Basic) {
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, 42.0f);
  TF_EXPECT_OK(root.status());
  EXPECT_EQ(c.op().output_type(0), DT_FLOAT);
  ExpectNodeEqual<float>(c.node(), {42.0f}, {});
}

TEST(ConstOpTest, MultiDim) {
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, {{2.0}, {3.0}});
  TF_CHECK_OK(root.status());
  EXPECT_EQ(c.op().output_type(0), DT_DOUBLE);
  ExpectNodeEqual<double>(c.node(), {2.0, 3.0}, {2, 1});
}

TEST(ConstOpTest, Empty) {
  Scope root = Scope::NewRootScope();

  auto c1 = ops::Const(root, {});
  TF_CHECK_OK(root.status());
  ExpectTypeAndShape(c1.node(), DT_FLOAT, {0});

  auto c2 = ops::Const(root, {{}});
  TF_CHECK_OK(root.status());
  ExpectTypeAndShape(c2.node(), DT_FLOAT, {1, 0});

  auto c3 = ops::Const(root, {{{}, {}}});
  TF_CHECK_OK(root.status());
  ExpectTypeAndShape(c3.node(), DT_FLOAT, {1, 2, 0});

  auto c4 = ops::Const<int>(root, {{{}}});
  TF_CHECK_OK(root.status());
  ExpectTypeAndShape(c4.node(), DT_INT32, {1, 1, 0});

  ops::Const(root, {{}, {{}}});
  EXPECT_FALSE(root.status().ok());
}

TEST(ConstOpTest, WithExplicitShape) {
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, 42.0, {2, 2});
  TF_CHECK_OK(root.status());
  EXPECT_EQ(c.op().output_type(0), DT_DOUBLE);
  ExpectNodeEqual<double>(c.node(), {42.0, 42.0, 42.0, 42.0}, {2, 2});

  auto d = ops::Const(root, {"1", "2", "3", "4", "5", "6"}, {2, 3});
  TF_CHECK_OK(root.status());
  EXPECT_EQ(d.op().output_type(0), DT_STRING);
  ExpectNodeEqual<tstring>(d.node(), {"1", "2", "3", "4", "5", "6"}, {2, 3});
}

TEST(ConstOpTest, FromProto) {
  Scope root = Scope::NewRootScope();
  TensorProto proto;
  proto.set_dtype(DT_DOUBLE);
  TensorShape({2, 2}).AsProto(proto.mutable_tensor_shape());
  for (int i = 0; i < 4; ++i) {
    proto.add_double_val(static_cast<double>(i));
  }
  auto c = ops::ConstFromProto(root, proto);
  TF_CHECK_OK(root.status());
  EXPECT_EQ(c.op().output_type(0), DT_DOUBLE);
  ExpectNodeEqual<double>(c.node(), {0.0, 1.0, 2.0, 3.0}, {2, 2});
}

TEST(ConstOpTest, InvalidInitializer) {
  Scope root = Scope::NewRootScope();
  ops::Const(root, {{2.0}, {"df"}});
  EXPECT_FALSE(root.status().ok());
}

TEST(ConstOpTest, Names) {
  Scope root = Scope::NewRootScope();
  auto c = ops::Const(root, {{2.0}, {3.0}});
  EXPECT_EQ(c.node()->name(), "Const");
  auto c_1 = ops::Const(root, {{2.0}, {3.0}});
  EXPECT_EQ(c_1.node()->name(), "Const_1");

  auto x = ops::Const(root.WithOpName("x"), 1);
  EXPECT_EQ(x.node()->name(), "x");
  auto x_1 = ops::Const(root.WithOpName("x"), 1);
  EXPECT_EQ(x_1.node()->name(), "x_1");

  Scope child = root.NewSubScope("c");
  auto c_y = ops::Const(child.WithOpName("y"), 1);
  EXPECT_EQ(c_y.node()->name(), "c/y");
  auto c_y_1 = ops::Const(child.WithOpName("y"), 1);
  EXPECT_EQ(c_y_1.node()->name(), "c/y_1");
}

TEST(ConstOpTest, TemplatedConst) {
  Scope root = Scope::NewRootScope();
  auto c1 = ops::Const<int>(root, {1, 2});
  ExpectTypeAndShape(c1.node(), DT_INT32, {2});

  auto c2 = ops::Const<tstring>(root, {{"this"}, {"is"}, {"a"}, {"constant"}});
  ExpectTypeAndShape(c2.node(), DT_STRING, {4, 1});
}

}  // namespace tensorflow
