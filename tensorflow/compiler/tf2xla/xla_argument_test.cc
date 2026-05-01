/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/xla_argument.h"

#include <iterator>
#include <optional>
#include <string>
#include <utility>
#include <variant>

#include "tensorflow/compiler/tf2xla/xla_argument.pb.h"
#include "tensorflow/compiler/tf2xla/xla_resource.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(XlaArgumentToProto, ShapeIsTensorShape) {
  XlaArgument arg;
  arg.shape = TensorShape({1, 2, 3});
  tf2xla::XlaArgumentProto proto = arg.ToProto();

  EXPECT_TRUE(proto.has_shape());
  EXPECT_TRUE(proto.shape().has_tensor_shape());
  EXPECT_EQ(proto.shape().tensor_shape().dim_size(), 3);
  EXPECT_EQ(proto.shape().tensor_shape().dim(0).size(), 1);
  EXPECT_EQ(proto.shape().tensor_shape().dim(1).size(), 2);
  EXPECT_EQ(proto.shape().tensor_shape().dim(2).size(), 3);
  EXPECT_FALSE(proto.shape().has_xla_shape());
}

TEST(XlaArgumentToProto, ShapeIsXlaShape) {
  XlaArgument arg;
  arg.shape = xla::Shape(xla::F32, {1, 2, 3});
  tf2xla::XlaArgumentProto proto = arg.ToProto();

  EXPECT_TRUE(proto.has_shape());
  EXPECT_FALSE(proto.shape().has_tensor_shape());
  EXPECT_TRUE(proto.shape().has_xla_shape());
  EXPECT_EQ(proto.shape().xla_shape().element_type(), xla::F32);
  EXPECT_EQ(proto.shape().xla_shape().dimensions_size(), 3);
  EXPECT_EQ(proto.shape().xla_shape().dimensions(0), 1);
  EXPECT_EQ(proto.shape().xla_shape().dimensions(1), 2);
  EXPECT_EQ(proto.shape().xla_shape().dimensions(2), 3);
}

TEST(XlaArgumentToProto, ShapeIsMonostate) {
  XlaArgument arg;
  tf2xla::XlaArgumentProto proto = arg.ToProto();
  EXPECT_FALSE(proto.has_shape());
}

TEST(XlaArgumentToProto, AllFields) {
  XlaArgument arg;
  arg.kind = XlaArgument::kConstant;
  arg.type = DT_FLOAT;
  arg.shape = TensorShape({1, 2, 3});
  Tensor constant_value(DT_FLOAT, TensorShape({1, 2, 3}));
  auto constant_value_map = constant_value.tensor<float, 3>();
  constant_value_map.setZero();
  constant_value_map(0, 0, 0) = 1.0f;
  constant_value_map(0, 0, 1) = 2.0f;
  arg.constant_value = std::move(constant_value);
  Tensor value_bound(DT_FLOAT, TensorShape({1, 2, 3}));
  auto value_bound_map = value_bound.tensor<float, 3>();
  value_bound_map.setZero();
  value_bound_map(0, 0, 0) = 3.0f;
  value_bound_map(0, 0, 1) = 4.0f;
  arg.value_bound = std::move(value_bound);
  // value_dynamism is intentionally not set to test how std::optional is
  // handled.
  arg.name = "name";
  arg.node_name = "node_name";
  arg.resource_kind = XlaResource::kVariable;
  arg.initialized = true;
  arg.fast_mem = false;
  arg.max_array_size = 10;
  arg.tensor_array_gradients = {"gradient_0", "gradient_1"};
  arg.is_same_data_across_replicas = true;
  arg.requires_broadcast = false;

  tf2xla::XlaArgumentProto proto = arg.ToProto();
  EXPECT_EQ(proto.kind(), static_cast<int>(XlaArgument::kConstant));
  EXPECT_EQ(proto.type(), DT_FLOAT);
  EXPECT_TRUE(proto.has_shape());
  EXPECT_TRUE(proto.shape().has_tensor_shape());
  EXPECT_EQ(proto.shape().tensor_shape().dim_size(), 3);
  EXPECT_EQ(proto.shape().tensor_shape().dim(0).size(), 1);
  EXPECT_EQ(proto.shape().tensor_shape().dim(1).size(), 2);
  EXPECT_EQ(proto.shape().tensor_shape().dim(2).size(), 3);
  EXPECT_TRUE(proto.has_constant_value());
  EXPECT_TRUE(proto.has_value_bound());
  EXPECT_FALSE(proto.has_value_dynamism());
  EXPECT_EQ(proto.name(), "name");
  EXPECT_EQ(proto.node_name(), "node_name");
  EXPECT_EQ(proto.resource_kind(), static_cast<int>(XlaResource::kVariable));
  EXPECT_TRUE(proto.initialized());
  EXPECT_FALSE(proto.fast_mem());
  EXPECT_EQ(proto.max_array_size(), 10);
  EXPECT_EQ(proto.tensor_array_gradients_size(), 2);
  EXPECT_EQ(proto.tensor_array_gradients(0), "gradient_0");
  EXPECT_EQ(proto.tensor_array_gradients(1), "gradient_1");
  EXPECT_TRUE(proto.is_same_data_across_replicas());
  EXPECT_FALSE(proto.requires_broadcast());

  TF_ASSERT_OK_AND_ASSIGN(XlaArgument arg_from_proto,
                          XlaArgument::FromProto(proto));
  EXPECT_EQ(arg, arg_from_proto);
}

TEST(XlaArgumentFromProto, ShapeIsTensorShape) {
  tf2xla::XlaArgumentProto proto;
  proto.mutable_shape()->mutable_tensor_shape()->add_dim()->set_size(1);
  proto.mutable_shape()->mutable_tensor_shape()->add_dim()->set_size(2);
  proto.mutable_shape()->mutable_tensor_shape()->add_dim()->set_size(3);

  TF_ASSERT_OK_AND_ASSIGN(XlaArgument arg, XlaArgument::FromProto(proto));
  EXPECT_TRUE(std::holds_alternative<TensorShape>(arg.shape));
  EXPECT_EQ(std::get<TensorShape>(arg.shape), TensorShape({1, 2, 3}));
}

TEST(XlaArgumentFromProto, ShapeIsXlaShape) {
  tf2xla::XlaArgumentProto proto;
  proto.mutable_shape()->mutable_xla_shape()->set_element_type(xla::F32);
  proto.mutable_shape()->mutable_xla_shape()->add_dimensions(1);
  proto.mutable_shape()->mutable_xla_shape()->add_dimensions(2);
  proto.mutable_shape()->mutable_xla_shape()->add_dimensions(3);

  TF_ASSERT_OK_AND_ASSIGN(XlaArgument arg, XlaArgument::FromProto(proto));
  EXPECT_TRUE(std::holds_alternative<xla::Shape>(arg.shape));
  EXPECT_EQ(std::get<xla::Shape>(arg.shape), xla::Shape(xla::F32, {1, 2, 3}));
}

TEST(XlaArgumentFromProto, ShapeIsMonostate) {
  tf2xla::XlaArgumentProto proto;
  TF_ASSERT_OK_AND_ASSIGN(XlaArgument arg, XlaArgument::FromProto(proto));
  EXPECT_TRUE(std::holds_alternative<std::monostate>(arg.shape));
}

TEST(XlaArgumentFromProto, AllFields) {
  tf2xla::XlaArgumentProto proto;
  proto.set_kind(static_cast<int>(XlaArgument::kConstant));
  proto.set_type(DT_FLOAT);
  proto.mutable_shape()->mutable_tensor_shape()->add_dim()->set_size(1);
  proto.mutable_shape()->mutable_tensor_shape()->add_dim()->set_size(2);
  proto.mutable_shape()->mutable_tensor_shape()->add_dim()->set_size(3);
  Tensor constant_value(DT_FLOAT, TensorShape({1, 2, 3}));
  auto constant_value_map = constant_value.tensor<float, 3>();
  constant_value_map.setZero();
  constant_value_map(0, 0, 0) = 1.0f;
  constant_value_map(0, 0, 1) = 2.0f;
  constant_value.AsProtoTensorContent(proto.mutable_constant_value());
  Tensor value_bound(DT_FLOAT, TensorShape({1, 2, 3}));
  auto value_bound_map = value_bound.tensor<float, 3>();
  value_bound_map.setZero();
  value_bound_map(0, 0, 0) = 3.0f;
  value_bound_map(0, 0, 1) = 4.0f;
  value_bound.AsProtoTensorContent(proto.mutable_value_bound());
  proto.set_name("name");
  proto.set_node_name("node_name");
  proto.set_resource_kind(static_cast<int>(XlaResource::kVariable));
  proto.set_initialized(true);
  proto.set_fast_mem(false);
  proto.set_max_array_size(10);
  proto.add_tensor_array_gradients("gradient_0");
  proto.add_tensor_array_gradients("gradient_1");
  proto.set_is_same_data_across_replicas(true);
  proto.set_requires_broadcast(false);

  TF_ASSERT_OK_AND_ASSIGN(XlaArgument arg, XlaArgument::FromProto(proto));
  EXPECT_EQ(arg.kind, XlaArgument::kConstant);
  EXPECT_EQ(arg.type, DT_FLOAT);
  EXPECT_TRUE(std::holds_alternative<TensorShape>(arg.shape));
  EXPECT_EQ(std::get<TensorShape>(arg.shape), TensorShape({1, 2, 3}));
  test::ExpectEqual(arg.constant_value, constant_value);
  ASSERT_TRUE(arg.value_bound.has_value());
  test::ExpectEqual(arg.value_bound.value(), value_bound);
  EXPECT_FALSE(arg.value_dynamism.has_value());
  EXPECT_EQ(arg.name, "name");
  EXPECT_EQ(arg.node_name, "node_name");
  EXPECT_EQ(arg.resource_kind, XlaResource::kVariable);
  EXPECT_TRUE(arg.initialized);
  EXPECT_FALSE(arg.fast_mem);
  EXPECT_EQ(arg.max_array_size, 10);
  EXPECT_EQ(arg.tensor_array_gradients.size(), 2);
  EXPECT_EQ(*arg.tensor_array_gradients.begin(), "gradient_0");
  EXPECT_EQ(*std::next(arg.tensor_array_gradients.begin()), "gradient_1");
  EXPECT_TRUE(arg.is_same_data_across_replicas);
  EXPECT_FALSE(arg.requires_broadcast);
}

}  // namespace
}  // namespace tensorflow
