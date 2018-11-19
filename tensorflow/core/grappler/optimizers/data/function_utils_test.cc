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

#include "tensorflow/core/grappler/optimizers/data/function_utils.h"

#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/tools/graph_transforms/transform_utils.h"

namespace tensorflow {
namespace grappler {
namespace function_utils {
namespace {

TEST(FunctionDefTensorDesc, Parsing) {
  FunctionDefTensorDesc f("Cast:y:0");
  EXPECT_EQ(f.full_str, "Cast:y:0");
  EXPECT_EQ(f.node_name, "Cast");
  EXPECT_EQ(f.node_output, "y");
  EXPECT_EQ(f.position, 0);

  FunctionDefTensorDesc f2("Arg0");
  EXPECT_EQ(f2.full_str, "Arg0");
  EXPECT_EQ(f2.node_name, "Arg0");
  EXPECT_EQ(f2.node_output, "");
  EXPECT_EQ(f2.position, -1);
}

TEST(ReplaceReferencesTest, ReplaceReferencesTest) {
  FunctionDef outer = FunctionDefHelper::Create(
      "outer", {"arg0: int32"}, {"out: int32", "out2: int64"}, {}, {},
      {{"out", "MapDefun:output:0"}, {"out2", "Cast:y:0"}});
  NodeDef* derive_node =
      AddNode("X", "Some_Op", {"MapDefun:output:0"}, {}, &outer);
  // Check that both the input to "X" and retval of "outer" are replaced.
  ReplaceReferences("MapDefun:output:0", "arg0", &outer);
  EXPECT_EQ(outer.ret().at("out"), "arg0");
  EXPECT_EQ(derive_node->input(0), "arg0");
}

TEST(FunctionUtilsTest, AddFunctionOutputWithUniqueName) {
  FunctionDef function = test::function::XTimesTwo();
  AddFunctionOutputWithUniqueName("y", "two", &function, DT_INT64);
  EXPECT_TRUE(ContainsFunctionOutputWithName("y/_1", function));
  EXPECT_EQ(function.ret().at("y/_1"), "two");
}

TEST(FunctionUtilsTest, ContainsFunctionNodeWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_FALSE(ContainsFunctionNodeWithName(
      "weird_name_that_should_not_be_there", function));
  EXPECT_TRUE(ContainsFunctionNodeWithName("two", function));
}

TEST(FunctionUtilsTest, ContainsFunctionNodeWithOp) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_FALSE(ContainsFunctionNodeWithOp("weird_op_that_should_not_be_there",
                                          function));
  EXPECT_TRUE(ContainsFunctionNodeWithOp("Mul", function));
}

TEST(FunctionUtilsTest, ContainsFunctionOutputWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_TRUE(ContainsFunctionOutputWithName("y", function));
  EXPECT_FALSE(ContainsFunctionOutputWithName("Add:z:0", function));
}

TEST(FunctionUtilsTest, FindFunctionNodeWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_EQ(
      FindFunctionNodeWithName("weird_name_that_should_not_be_there", function),
      -1);
  EXPECT_NE(FindFunctionNodeWithName("two", function), -1);
}

TEST(FunctionUtilsTest, FindFunctionNodeWithOp) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_EQ(
      FindFunctionNodeWithOp("weird_op_that_should_not_be_there", function),
      -1);
  EXPECT_NE(FindFunctionNodeWithOp("Mul", function), -1);
}

TEST(FunctionUtilsTest, FindFunctionInputWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_EQ(FindFunctionInputWithName("x", function), 0);
  EXPECT_EQ(FindFunctionInputWithName("not_a_name", function), -1);
}

TEST(FunctionUtilsTest, FindFunctionOutputWithName) {
  FunctionDef function = test::function::XTimesTwo();
  EXPECT_EQ(FindFunctionOutputWithName("y", function), 0);
  EXPECT_EQ(FindFunctionOutputWithName("Add:z:0", function), -1);
}

TEST(FunctionUtilsTest, SetUniqueFunctionNodeName) {
  FunctionDef function = test::function::XTimesTwo();
  NodeDef node;
  SetUniqueFunctionNodeName("abc", &function, &node);
  for (const NodeDef& function_node : function.node_def()) {
    EXPECT_NE(node.name(), function_node.name());
  }
  auto* new_node = function.add_node_def();
  *new_node = node;

  NodeDef other;
  SetUniqueFunctionNodeName("abc", &function, &other);
  EXPECT_NE(other.name(), new_node->name());
}

TEST(FunctionUtilsTest, AddNodeToFunctionDef) {
  FunctionDef func;
  const char* op_name = "xxx";
  AddNode(op_name, op_name, {}, {}, &func);

  const NodeDef& node1 = func.node_def(FindFunctionNodeWithName("xxx", func));
  EXPECT_EQ(node1.op(), op_name);
  EXPECT_EQ(node1.input_size(), 0);
  EXPECT_EQ(node1.attr_size(), 0);

  const std::vector<string> inputs({"input1", "input2"});
  AddNode("", op_name, inputs, {}, &func);
  const NodeDef& node2 =
      func.node_def(FindFunctionNodeWithName("xxx/_2", func));
  EXPECT_EQ(node2.op(), op_name);
  EXPECT_EQ(node2.attr_size(), 0);
  EXPECT_EQ(node2.input_size(), inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    EXPECT_EQ(node2.input(i), inputs[i]);
  }

  AttrValue a1, a2;
  a1.set_type(DT_INT32);
  a2.set_type(DT_INT64);
  const std::vector<std::pair<string, AttrValue>> attrs(
      {{"attr1", a1}, {"attr2", a2}});
  AddNode("", op_name, {}, attrs, &func);
  const NodeDef& node3 =
      func.node_def(FindFunctionNodeWithName("xxx/_3", func));
  EXPECT_EQ(node3.op(), op_name);
  EXPECT_EQ(node3.input_size(), 0);
  EXPECT_EQ(node3.attr_size(), attrs.size());
  for (size_t i = 0; i < attrs.size(); ++i) {
    EXPECT_EQ(attrs[i].second.type(), node3.attr().at(attrs[i].first).type());
  }
}

}  // namespace
}  // namespace function_utils
}  // namespace grappler
}  // namespace tensorflow
