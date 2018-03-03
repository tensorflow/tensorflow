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

#include "tensorflow/core/grappler/utils/functions.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class FunctionsTest : public ::testing::Test {};

TEST_F(FunctionsTest, FromSimpleFunctionDef) {
  const Tensor kTwo = test::AsScalar<int64>(2);
  FunctionDef func = FunctionDefHelper::Define(
      // Name
      "XTimesTwo",
      // Args
      {"x: T"},
      // Return values
      {"y: T"},
      // Attr def
      {"T: {float, double, int32, int64}"},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", "$T"}}},
      });

  std::unordered_map<string, AttrValue> func_attr;
  func_attr["T"].set_type(DT_FLOAT);
  FunctionDefLibrary library;
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, library);
  CHECK(item);
  EXPECT_EQ("XTimesTwo", item->id);
  EXPECT_EQ(4, item->graph.node_size());
  EXPECT_EQ(std::vector<string>({"y:0"}), item->fetch);
  EXPECT_EQ(1, item->feed.size());
  EXPECT_EQ("x", item->feed[0].first);

  for (const NodeDef &node : item->graph.node()) {
    if (node.name() == "x") {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "two") {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "scale") {
      EXPECT_EQ("Cast", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("DstT").type());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("two:0", node.input(0));
    } else if (node.name() == "y") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("scale:0", node.input(1));
    }
  }
}

TEST_F(FunctionsTest, FromFunctionDefWithMultiOutputNodes) {
  // Gradient graph for the Subtract operation
  std::vector<FunctionDefHelper::Node> nodes = {
      {{"sx"}, "Shape", {"x"}},
      {{"sy"}, "Shape", {"y"}},
      {{"gx"}, "Identity", {"dz"}},
      {{"gy"}, "Neg", {"dz"}},
      {{"rx", "ry"}, "BroadcastGradientArgs", {"sx", "sy"}},
      {{"sum_gx"}, "Sum", {"gx", "rx"}},
      {{"dx"}, "Reshape", {"sum_gx", "sx"}},
      {{"sum_gy"}, "Sum", {"gy", "ry"}},
      {{"dy"}, "Reshape", {"sum_gy", "sy"}},
  };

  for (auto &n : nodes) {
    // "BroadcastGradientArgs" doesn't need any attrs.
    if (n.attr.empty() && n.op != "BroadcastGradientArgs") {
      n.attr = {{"T", "$T"}};
    }
  }
  FunctionDef func = FunctionDefHelper::Define(
      // Name
      "SubGrad",
      // Arg defs
      {"x: T", "y: T", "dz: T"},
      // Ret val defs
      {"dx: T", "dy: T"},
      // Attr defs
      {{"T: {half, float, double}"}},
      // Nodes
      nodes);

  std::unordered_map<string, AttrValue> func_attr;
  func_attr["T"].set_type(DT_FLOAT);
  FunctionDefLibrary library;
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, library);
  CHECK(item);
  EXPECT_EQ("SubGrad", item->id);
  EXPECT_EQ(12, item->graph.node_size());
  EXPECT_EQ(std::vector<string>({"dx:0", "dy:0"}), item->fetch);
  EXPECT_EQ(3, item->feed.size());
  EXPECT_EQ("x", item->feed[0].first);
  EXPECT_EQ("y", item->feed[1].first);
  EXPECT_EQ("dz", item->feed[2].first);

  for (const NodeDef &node : item->graph.node()) {
    if (node.name() == "x" || node.name() == "y" || node.name() == "dz") {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "rx") {
      EXPECT_EQ("BroadcastGradientArgs", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("sx:0", node.input(0));
      EXPECT_EQ("sy:0", node.input(1));
    } else if (node.name() == "sum_gx") {
      EXPECT_EQ("Sum", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("gx:0", node.input(0));
      EXPECT_EQ("rx:0", node.input(1));
    } else if (node.name() == "sum_gy") {
      EXPECT_EQ("Sum", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("gy:0", node.input(0));
      EXPECT_EQ("rx:1", node.input(1));
    }
  }
}

TEST_F(FunctionsTest, FromFunctionDefWithNestedFuncs) {
  FunctionDefLibrary library;
  *library.add_function() = FunctionDefHelper::Define(
      // Name
      "Swap",
      // Args
      {"i0: T", "i1: T"},
      // Return values
      {"o0: T", "o1: T"},
      // Attr def
      {"T: {float, double}"},
      // Nodes
      {{{"o0"}, "Identity", {"i1"}, {{"T", "$T"}}},
       {{"o1"}, "Identity", {"i0"}, {{"T", "$T"}}}});

  FunctionDef func = FunctionDefHelper::Create(
      // Name
      "ManySwapsFirst",
      // Args
      {"x: float", "y: float"},
      // Return values
      {"o: float"},
      // attr def
      {},
      // Nodes
      // o = x*x + y*y.  Furthermore, The 1st swap depends on x2, and
      // y2 depends on the 2nd swap.  The 2nd swap has data dependency
      // on the 1st swap.
      {{{"a0"}, "Swap", {"x", "y"}, {{"T", DT_FLOAT}}, {"x2"}},
       {{"a1"}, "Swap", {"a0:o0:0", "a0:o1:0"}, {{"T", DT_FLOAT}}},
       {{"x2"}, "Mul", {"x", "x"}, {{"T", DT_FLOAT}}},
       {{"y2"}, "Mul", {"y", "y"}, {{"T", DT_FLOAT}}, {"a1"}},
       {{"o"}, "Add", {"x2:z:0", "y2:z:0"}, {{"T", DT_FLOAT}}}},
      // Output Mapping
      {{"o", "o:z:0"}});

  std::unordered_map<string, AttrValue> func_attr;
  func_attr["T"].set_type(DT_FLOAT);
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, library);

  for (const NodeDef &node : item->graph.node()) {
    if (node.name() == "x" || node.name() == "y") {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "a0") {
      EXPECT_EQ("Swap", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^x2", node.input(2));
    } else if (node.name() == "a1") {
      EXPECT_EQ("Swap", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("a0:0", node.input(0));
      EXPECT_EQ("a0:1", node.input(1));
    } else if (node.name() == "x2") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("x", node.input(1));
    } else if (node.name() == "y2") {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^a1", node.input(2));
    } else if (node.name() == "o") {
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x2:0", node.input(0));
      EXPECT_EQ("y2:0", node.input(1));
    }
  }
}

TEST_F(FunctionsTest, FromFunctionDefWithOutputMappings) {
  FunctionDef func = FunctionDefHelper::Create(
      // Name
      "Exp_func",
      // Args
      {"in: float"},
      // Return values
      {"out: float"},
      // Attr def
      {},
      // Nodes
      {{{"Linear_func"}, "Identity", {"in"}, {{"T", DT_FLOAT}}},
       {{"Exp"}, "Exp", {"Linear_func:output:0"}, {{"T", DT_FLOAT}}}},
      // Mapping
      {{"out", "Exp:y:0"}});

  std::unordered_map<string, AttrValue> func_attr;
  FunctionDefLibrary library;
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, library);

  EXPECT_EQ(1, item->fetch.size());
  EXPECT_EQ("Exp:0", item->fetch[0]);

  for (const NodeDef &node : item->graph.node()) {
    if (node.name() == "in") {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "Linear_func") {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("in", node.input(0));
    } else if (node.name() == "Exp") {
      EXPECT_EQ("Exp", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("Linear_func:0", node.input(0));
    }
  }
}

TEST_F(FunctionsTest, FromFunctionDefWithInputForwarding) {
  FunctionDef func = FunctionDefHelper::Create(
      // Name
      "ForwardInputs",
      // Args
      {"in0: float", "in1: float", "arg2: float", "arg3: int32", "arg4: float"},
      // Return values
      {"out0: float", "arg2: float", "arg3: int32"},
      // Attr def
      {},
      // Nodes
      {},
      // Mapping
      {{"out0", "in0"}});

  std::unordered_map<string, AttrValue> func_attr;
  FunctionDefLibrary library;
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, library);

  EXPECT_EQ(3, item->fetch.size());
  EXPECT_EQ("in0", item->fetch[0]);
  EXPECT_EQ("arg2", item->fetch[1]);
  EXPECT_EQ("arg3", item->fetch[2]);

  EXPECT_EQ(5, item->graph.node_size());
  for (const NodeDef &node : item->graph.node()) {
    EXPECT_TRUE(node.name() == "in0" || node.name() == "in1" ||
                node.name() == "arg2" || node.name() == "arg3" ||
                node.name() == "arg4");
    EXPECT_EQ("Placeholder", node.op());
    if (node.name() == "arg3") {
      EXPECT_EQ(DT_INT32, node.attr().at("T").type());
    } else {
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
    }
  }
}

TEST_F(FunctionsTest, FromFunctionDefWithoutInput) {
  const Tensor kTwo = test::AsScalar<int64>(2);
  FunctionDef func = FunctionDefHelper::Define(
      // Name
      "GenerateTwo",
      // Args
      {},
      // Return value
      {"o: T"},
      // Attr def
      {"T: {float, double}"},
      // Nodes
      {{{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
       {{"o"}, "Cast", {"two"}, {{"SrcT", DT_INT64}, {"DstT", "$T"}}}});

  std::unordered_map<string, AttrValue> func_attr;
  func_attr["T"].set_type(DT_FLOAT);
  FunctionDefLibrary library;
  std::unique_ptr<GrapplerItem> item =
      GrapplerItemFromFunctionDef(func, func_attr, library);

  EXPECT_EQ(0, item->feed.size());
  EXPECT_EQ(1, item->fetch.size());
  EXPECT_EQ("o:0", item->fetch[0]);

  EXPECT_EQ(2, item->graph.node_size());
  const NodeDef &two = item->graph.node(0);
  EXPECT_EQ("two", two.name());
  EXPECT_EQ(0, two.input_size());
  const NodeDef &cast = item->graph.node(1);
  EXPECT_EQ("o", cast.name());
  EXPECT_EQ(1, cast.input_size());
  EXPECT_EQ("two:0", cast.input(0));

  std::cout << item->graph.DebugString() << std::endl;
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
