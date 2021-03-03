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

#include "tensorflow/compiler/tf2xla/tf2xla.h"

#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/client/client_library.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

AttrValue TypeAttrValue(DataType type) {
  AttrValue attr_value;
  SetAttrValue(type, &attr_value);
  return attr_value;
}

GraphDef SumGraph() {
  GraphDef graph_def;
  NodeDef* x = graph_def.add_node();
  x->set_name("x");
  x->set_op("Placeholder");
  (*x->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  NodeDef* y = graph_def.add_node();
  y->set_name("y");
  y->set_op("Placeholder");
  (*y->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  NodeDef* sum = graph_def.add_node();
  sum->set_name("sum");
  sum->set_op("Add");
  sum->add_input("x");
  sum->add_input("y");
  (*sum->mutable_attr())["T"] = TypeAttrValue(DT_INT32);
  return graph_def;
}

tf2xla::Config SumConfig() {
  tf2xla::Config config;
  config.add_feed()->mutable_id()->set_node_name("x");
  config.add_feed()->mutable_id()->set_node_name("y");
  config.add_fetch()->mutable_id()->set_node_name("sum");
  return config;
}

TEST(ConvertGraphDefToXla, Sum) {
  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, client, &computation));

  // Set up arguments.
  auto x_literal = xla::LiteralUtil::CreateR0<int32>(10);
  auto y_literal = xla::LiteralUtil::CreateR0<int32>(32);
  auto x_global_or = client->TransferToServer(x_literal);
  auto y_global_or = client->TransferToServer(y_literal);
  TF_EXPECT_OK(x_global_or.status());
  TF_EXPECT_OK(y_global_or.status());
  std::unique_ptr<xla::GlobalData> x_global =
      std::move(x_global_or.ValueOrDie());
  std::unique_ptr<xla::GlobalData> y_global =
      std::move(y_global_or.ValueOrDie());

  // Execute and check result.
  auto result_or =
      client->ExecuteAndTransfer(computation, {x_global.get(), y_global.get()});
  TF_EXPECT_OK(result_or.status());
  xla::Literal result = std::move(result_or.ValueOrDie());
  EXPECT_EQ("(\ns32[] 42\n)", result.ToString());

  config.mutable_feed(0)->mutable_id()->set_output_index(
      123); /* invalid output_index */
  EXPECT_TRUE(errors::IsInvalidArgument(
      ConvertGraphDefToXla(graph_def, config, client, &computation)));
}

TEST(ConvertGraphDefToXla, SumWithUnusedArgument) {
  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();
  NodeDef* unused = graph_def.add_node();
  unused->set_name("unused");
  unused->set_op("Placeholder");
  (*unused->mutable_attr())["dtype"] = TypeAttrValue(DT_INT32);
  config.add_feed()->mutable_id()->set_node_name("unused");

  xla::LocalClient* client = xla::ClientLibrary::LocalClientOrDie();
  xla::XlaComputation computation;
  TF_EXPECT_OK(ConvertGraphDefToXla(graph_def, config, client, &computation));

  // Set up arguments.
  auto x_literal = xla::LiteralUtil::CreateR0<int32>(10);
  auto y_literal = xla::LiteralUtil::CreateR0<int32>(32);
  auto x_global_or = client->TransferToServer(x_literal);
  auto y_global_or = client->TransferToServer(y_literal);
  auto unused_global_or = client->TransferToServer(y_literal);
  TF_EXPECT_OK(x_global_or.status());
  TF_EXPECT_OK(y_global_or.status());
  TF_EXPECT_OK(unused_global_or.status());
  std::unique_ptr<xla::GlobalData> x_global =
      std::move(x_global_or.ValueOrDie());
  std::unique_ptr<xla::GlobalData> y_global =
      std::move(y_global_or.ValueOrDie());
  std::unique_ptr<xla::GlobalData> unused_global =
      std::move(unused_global_or.ValueOrDie());

  // Execute and check result.
  auto result_or = client->ExecuteAndTransfer(
      computation, {x_global.get(), y_global.get(), unused_global.get()});
  TF_EXPECT_OK(result_or.status());
  xla::Literal result = std::move(result_or.ValueOrDie());
  EXPECT_EQ("(\ns32[] 42\n)", result.ToString());
}

}  // namespace
}  // namespace tensorflow
