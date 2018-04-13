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
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"

namespace tensorflow {
namespace grappler {
namespace {

class FunctionsTest : public ::testing::Test {};

TEST_F(FunctionsTest, GrapplerFunctionConnectivity_ExpandFunctionDefInput) {
  GrapplerFunctionConnectivity connectivity;

  connectivity.RegisterInputArgExpansion({"inputA", DT_FLOAT, {"inputA"}});
  connectivity.RegisterInputArgExpansion(
      {"inputB", DT_FLOAT, {"inputB_0", "inputB_1"}});

  connectivity.RegisterFunctionBodyOutputs("Add", {{"z", {0, 1}}});
  connectivity.RegisterFunctionBodyOutputs("Func",
                                           {{"o1", {0, 2}}, {"o2", {2, 4}}});

  std::vector<string> inputs;
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("inputA", &inputs));
  ASSERT_EQ(1, inputs.size());
  EXPECT_EQ("inputA", inputs[0]);

  inputs.clear();
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("inputB", &inputs));
  ASSERT_EQ(2, inputs.size());
  EXPECT_EQ("inputB_0", inputs[0]);
  EXPECT_EQ("inputB_1", inputs[1]);

  inputs.clear();
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("inputB:1", &inputs));
  ASSERT_EQ(1, inputs.size());
  EXPECT_EQ("inputB_1", inputs[0]);

  inputs.clear();
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("Add:z", &inputs));
  ASSERT_EQ(1, inputs.size());
  EXPECT_EQ("Add", inputs[0]);

  inputs.clear();
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("Func:o1", &inputs));
  ASSERT_EQ(2, inputs.size());
  EXPECT_EQ("Func", inputs[0]);
  EXPECT_EQ("Func:1", inputs[1]);

  inputs.clear();
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("Func:o2", &inputs));
  ASSERT_EQ(2, inputs.size());
  EXPECT_EQ("Func:2", inputs[0]);
  EXPECT_EQ("Func:3", inputs[1]);

  inputs.clear();
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("Func:o1:0", &inputs));
  ASSERT_EQ(1, inputs.size());
  EXPECT_EQ("Func", inputs[0]);

  inputs.clear();
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("Func:o1:1", &inputs));
  ASSERT_EQ(1, inputs.size());
  EXPECT_EQ("Func:1", inputs[0]);

  inputs.clear();
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("Func:o2:0", &inputs));
  ASSERT_EQ(1, inputs.size());
  EXPECT_EQ("Func:2", inputs[0]);

  inputs.clear();
  TF_EXPECT_OK(connectivity.ExpandFunctionDefInput("Func:o2:1", &inputs));
  ASSERT_EQ(1, inputs.size());
  EXPECT_EQ("Func:3", inputs[0]);
}

TEST_F(FunctionsTest, GrapplerFunctionConnectivity_AsFunctionDefInput) {
  GrapplerFunctionConnectivity connectivity;

  connectivity.RegisterInputArgExpansion({"inputA", DT_FLOAT, {"inputA"}});
  connectivity.RegisterInputArgExpansion(
      {"inputB", DT_FLOAT, {"inputB_0", "inputB_1"}});

  connectivity.RegisterFunctionBodyOutputs("Add", {{"z", {0, 1}}});
  connectivity.RegisterFunctionBodyOutputs("Func",
                                           {{"o1", {0, 2}}, {"o2", {2, 4}}});

  string input;

  TF_EXPECT_OK(connectivity.AsFunctionDefInput("inputA", &input));
  EXPECT_EQ("inputA:0", input);

  TF_EXPECT_OK(connectivity.AsFunctionDefInput("inputB_0", &input));
  EXPECT_EQ("inputB:0", input);

  TF_EXPECT_OK(connectivity.AsFunctionDefInput("inputB_1", &input));
  EXPECT_EQ("inputB:1", input);

  TF_EXPECT_OK(connectivity.AsFunctionDefInput("Add", &input));
  EXPECT_EQ("Add:z:0", input);

  TF_EXPECT_OK(connectivity.AsFunctionDefInput("Func", &input));
  EXPECT_EQ("Func:o1:0", input);

  TF_EXPECT_OK(connectivity.AsFunctionDefInput("Func:1", &input));
  EXPECT_EQ("Func:o1:1", input);

  TF_EXPECT_OK(connectivity.AsFunctionDefInput("Func:2", &input));
  EXPECT_EQ("Func:o2:0", input);

  TF_EXPECT_OK(connectivity.AsFunctionDefInput("Func:3", &input));
  EXPECT_EQ("Func:o2:1", input);
}

TEST_F(FunctionsTest, GrapplerFunctionConnectivity_ExpandNodeInputs) {
  GrapplerFunctionConnectivity connectivity;

  connectivity.RegisterInputArgExpansion({"inputA", DT_FLOAT, {"inputA"}});
  connectivity.RegisterInputArgExpansion(
      {"inputB", DT_FLOAT, {"inputB_0", "inputB_1"}});

  NodeDef node;
  node.add_input("inputA:0");
  node.add_input("inputB");

  TF_EXPECT_OK(connectivity.ExpandNodeInputs(&node));

  EXPECT_EQ(3, node.input_size());
  EXPECT_EQ("inputA", node.input(0));
  EXPECT_EQ("inputB_0", node.input(1));
  EXPECT_EQ("inputB_1", node.input(2));
}

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
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func, func_attr, flib, &item));

  EXPECT_EQ("XTimesTwo", item.id);
  EXPECT_EQ(4, item.function_body().node_size());

  EXPECT_EQ(1, item.input_size());
  EXPECT_EQ("x", item.input(0).input_name);
  EXPECT_EQ(std::vector<string>{"x"}, item.input(0).placeholders);

  EXPECT_EQ(1, item.output_size());
  EXPECT_EQ("y", item.output(0).output_name);
  EXPECT_EQ("y", item.output(0).output_tensors[0]);

  int count = 0;
  for (const NodeDef &node : item.function_body().node()) {
    if (node.name() == "x" && count++) {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "two" && count++) {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "scale" && count++) {
      EXPECT_EQ("Cast", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("DstT").type());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("two", node.input(0));
    } else if (node.name() == "y" && count++) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("scale", node.input(1));
    }
  }
  EXPECT_EQ(4, count);
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
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func, func_attr, flib, &item));

  EXPECT_EQ("SubGrad", item.id);
  EXPECT_EQ(12, item.function_body().node_size());

  ASSERT_EQ(3, item.input_size());
  EXPECT_EQ("x", item.input(0).input_name);
  EXPECT_EQ("y", item.input(1).input_name);
  EXPECT_EQ("dz", item.input(2).input_name);

  ASSERT_EQ(2, item.output_size());
  EXPECT_EQ("dx", item.output(0).output_tensors[0]);
  EXPECT_EQ("dy", item.output(1).output_tensors[0]);

  int count = 0;
  for (const NodeDef &node : item.function_body().node()) {
    if (node.name() == "x" || node.name() == "y" || node.name() == "dz") {
      count++;
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "rx" && count++) {
      EXPECT_EQ("BroadcastGradientArgs", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("sx", node.input(0));
      EXPECT_EQ("sy", node.input(1));
    } else if (node.name() == "sum_gx" && count++) {
      EXPECT_EQ("Sum", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("gx", node.input(0));
      EXPECT_EQ("rx", node.input(1));
    } else if (node.name() == "sum_gy" && count++) {
      EXPECT_EQ("Sum", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("gy", node.input(0));
      EXPECT_EQ("rx:1", node.input(1));
    }
  }
  EXPECT_EQ(6, count);
}

TEST_F(FunctionsTest, FromFunctionDefWithNestedFuncs) {
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());
  TF_ASSERT_OK(flib.AddFunctionDef(FunctionDefHelper::Define(
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
       {{"o1"}, "Identity", {"i0"}, {{"T", "$T"}}}})));

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

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func, func_attr, flib, &item));

  int count = 0;
  for (const NodeDef &node : item.function_body().node()) {
    if (node.name() == "x" || node.name() == "y") {
      count++;
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "a0" && count++) {
      EXPECT_EQ("Swap", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^x2", node.input(2));
    } else if (node.name() == "a1" && count++) {
      EXPECT_EQ("Swap", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("a0", node.input(0));
      EXPECT_EQ("a0:1", node.input(1));
    } else if (node.name() == "x2" && count++) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("x", node.input(1));
    } else if (node.name() == "y2" && count++) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^a1", node.input(2));
    } else if (node.name() == "o" && count++) {
      EXPECT_EQ("Add", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x2", node.input(0));
      EXPECT_EQ("y2", node.input(1));
    }
  }
  EXPECT_EQ(7, count);
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
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func, func_attr, flib, &item));

  EXPECT_EQ(1, item.output_size());
  EXPECT_EQ("Exp", item.output(0).output_tensors[0]);

  int count = 0;
  for (const NodeDef &node : item.function_body().node()) {
    if (node.name() == "in" && count++) {
      EXPECT_EQ("Placeholder", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "Linear_func" && count++) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("in", node.input(0));
    } else if (node.name() == "Exp" && count++) {
      EXPECT_EQ("Exp", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("Linear_func", node.input(0));
    }
  }
  EXPECT_EQ(3, count);
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
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func, func_attr, flib, &item));

  EXPECT_EQ("ForwardInputs", item.id);
  EXPECT_EQ(5, item.function_body().node_size());

  EXPECT_EQ(3, item.output_size());
  EXPECT_EQ("in0", item.output(0).output_tensors[0]);
  EXPECT_EQ("arg2", item.output(1).output_tensors[0]);
  EXPECT_EQ("arg3", item.output(2).output_tensors[0]);

  int count = 0;
  for (const NodeDef &node : item.function_body().node()) {
    EXPECT_TRUE(node.name() == "in0" || node.name() == "in1" ||
                node.name() == "arg2" || node.name() == "arg3" ||
                node.name() == "arg4");
    count++;
    EXPECT_EQ("Placeholder", node.op());
    if (node.name() == "arg3") {
      EXPECT_EQ(DT_INT32, node.attr().at("T").type());
    } else {
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
    }
  }
  EXPECT_EQ(5, count);
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
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func, func_attr, flib, &item));

  EXPECT_EQ(0, item.input_size());
  EXPECT_EQ(1, item.output_size());
  EXPECT_EQ("o", item.output(0).output_tensors[0]);

  EXPECT_EQ(2, item.function_body().node_size());
  const NodeDef &two = item.function_body().node(0);
  EXPECT_EQ("two", two.name());
  EXPECT_EQ(0, two.input_size());
  const NodeDef &cast = item.function_body().node(1);
  EXPECT_EQ("o", cast.name());
  EXPECT_EQ(1, cast.input_size());
  EXPECT_EQ("two", cast.input(0));
}

TEST_F(FunctionsTest, MakeSpecializedFunctionDef) {
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
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func, func_attr, flib, &item));

  FunctionDef specialized;
  TF_EXPECT_OK(MakeSpecializedFunctionDef(item, flib, &specialized));

  // Input and output types are resolved based on instantiation attributes.
  EXPECT_EQ("x", specialized.signature().input_arg(0).name());
  EXPECT_EQ(DT_FLOAT, specialized.signature().input_arg(0).type());
  EXPECT_EQ("y", specialized.signature().output_arg(0).name());
  EXPECT_EQ(DT_FLOAT, specialized.signature().output_arg(0).type());

  // Function body specialized for instantiation types
  int count = 0;
  for (const NodeDef &node : specialized.node_def()) {
    if (node.name() == "scale" && count++) {
      EXPECT_EQ(DT_FLOAT, node.attr().at("DstT").type());
    } else if (node.name() == "y" && count++) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ("x:0", node.input(0));
      EXPECT_EQ("scale:y:0", node.input(1));
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
    }
  }
  EXPECT_EQ(2, count);
}

TEST_F(FunctionsTest, SwapFunctionBodyAndMakeSpecializedFunctionDef) {
  using test::function::NDef;

  FunctionDef mul_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"output"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});

  FunctionDef func = FunctionDefHelper::Create(
      "MySquare", {"x:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"output"}, "MyMul", {"x", "x"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});

  GraphDef id_func_body = test::function::GDef(
      {/* pass input to output through identity */
       NDef("output", "Identity", {"x"}, {{"T", "float"}})});

  std::unordered_map<string, AttrValue> func_attr;
  func_attr["T"].set_type(DT_FLOAT);

  FunctionDefLibrary lib_def;
  *lib_def.add_function() = func;
  *lib_def.add_function() = mul_func;
  FunctionLibraryDefinition flib(OpRegistry::Global(), lib_def);

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func, func_attr, flib, &item));

  // Replace function body with identity function
  item.SwapFunctionBody(std::move(id_func_body));
  FunctionDef specialized;
  TF_EXPECT_OK(MakeSpecializedFunctionDef(item, flib, &specialized));

  // Check that graph body was updated.
  int count = 0;
  for (const NodeDef &node : specialized.node_def()) {
    if (node.name() == "output" && count++) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ("x:0", node.input(0));
    }
  }
  EXPECT_EQ(1, count);

  // And return tensor mapping was updated with a new output name (z->output).
  EXPECT_EQ("output:output:0", (*specialized.mutable_ret())["z"]);
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
