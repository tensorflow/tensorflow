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

#include "tensorflow/compiler/tf2xla/xla_jit_compiled_cpu_function.h"

#include "tensorflow/compiler/tf2xla/tf2xla.pb.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/framework/attr_value.pb.h"
#include "tensorflow/core/framework/attr_value_util.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
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
  tf2xla::Feed* x = config.add_feed();
  x->mutable_id()->set_node_name("x");
  x->set_name("x_name");
  tf2xla::Feed* y = config.add_feed();
  y->mutable_id()->set_node_name("y");
  y->set_name("y_name");
  tf2xla::Fetch* sum = config.add_fetch();
  sum->mutable_id()->set_node_name("sum");
  sum->set_name("sum_name");
  return config;
}

TEST(XlaJitCompiledCpuFunction, Sum) {
  GraphDef graph_def = SumGraph();
  tf2xla::Config config = SumConfig();

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<XlaJitCompiledCpuFunction> jit,
      XlaJitCompiledCpuFunction::Compile(graph_def, config,
                                         xla::ExecutableBuildOptions()));
  XlaCompiledCpuFunction function(jit->StaticData());

  // Run the function and check results.
  *static_cast<int32*>(function.arg_data(0)) = 10;
  *static_cast<int32*>(function.arg_data(1)) = 32;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 42);

  // Run the function again.
  *static_cast<int32*>(function.arg_data(0)) = 100;
  *static_cast<int32*>(function.arg_data(1)) = 320;
  EXPECT_TRUE(function.Run());
  EXPECT_EQ(function.error_msg(), "");
  EXPECT_EQ(*static_cast<int32*>(function.result_data(0)), 420);

  // Check name to index lookups.
  EXPECT_TRUE(function.HasNameIndices());

  EXPECT_EQ(function.LookupArgIndex("x_name"), 0);
  EXPECT_EQ(function.LookupArgIndex("y_name"), 1);
  EXPECT_EQ(function.LookupArgIndex(""), -1);
  EXPECT_EQ(function.LookupArgIndex("x"), -1);
  EXPECT_EQ(function.LookupArgIndex("y"), -1);
  EXPECT_EQ(function.LookupArgIndex("sum"), -1);
  EXPECT_EQ(function.LookupArgIndex("sum_name"), -1);

  EXPECT_EQ(function.LookupResultIndex("sum_name"), 0);
  EXPECT_EQ(function.LookupResultIndex(""), -1);
  EXPECT_EQ(function.LookupResultIndex("x"), -1);
  EXPECT_EQ(function.LookupResultIndex("y"), -1);
  EXPECT_EQ(function.LookupResultIndex("sum"), -1);
  EXPECT_EQ(function.LookupResultIndex("x_name"), -1);
  EXPECT_EQ(function.LookupResultIndex("y_name"), -1);

  // Check program shape.
  using xla::ShapeUtil;
  const xla::Shape s32 = ShapeUtil::MakeShape(xla::S32, {});
  ASSERT_TRUE(function.ProgramShape() != nullptr);
  const xla::ProgramShape program_shape(*function.ProgramShape());
  ASSERT_EQ(program_shape.parameters_size(), 2);
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(0), s32));
  EXPECT_TRUE(ShapeUtil::Compatible(program_shape.parameters(1), s32));

  const xla::Shape& result = program_shape.result();
  ASSERT_EQ(result.element_type(), xla::TUPLE);
  ASSERT_EQ(ShapeUtil::TupleElementCount(result), 1);
  const xla::Shape& result0 = ShapeUtil::GetTupleElementShape(result, 0);
  EXPECT_TRUE(ShapeUtil::Compatible(result0, s32));
}

// Test when a graph compilation terminates early, resources are properly
// reclaimed.
TEST(XlaJitCompiledCpuFunction, SumWithJunkAttr) {
  GraphDef graph_def = SumGraph();

  (*graph_def.mutable_node(2)->mutable_attr())["junk"] =
      TypeAttrValue(DT_INT32);

  tf2xla::Config config = SumConfig();
  EXPECT_FALSE(XlaJitCompiledCpuFunction::Compile(graph_def, config,
                                                  xla::ExecutableBuildOptions())
                   .ok());
}

}  // namespace
}  // namespace tensorflow
