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

#include "absl/container/flat_hash_map.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

namespace tensorflow {
namespace grappler {
namespace {

constexpr char kDevice[] = "/device:CPU:0";

class FunctionsTest : public ::testing::Test {};

TEST_F(FunctionsTest, IsParametrized) {
  // Function is defined for multiple input types.
  FunctionDef parametrized_func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"output"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});

  // Function is defined just for float inputs.
  FunctionDef non_parametrized_func = FunctionDefHelper::Create(
      "MyMul", {"x:float", "y:float"}, {"z:float"}, {},
      {{{"output"}, "Mul", {"x", "y"}, {{"T", DT_FLOAT}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});

  EXPECT_TRUE(HasParametrizedType(parametrized_func));
  EXPECT_TRUE(HasParametrizedBody(parametrized_func));
  EXPECT_TRUE(IsParametrized(parametrized_func));

  EXPECT_FALSE(HasParametrizedType(non_parametrized_func));
  EXPECT_FALSE(HasParametrizedBody(non_parametrized_func));
  EXPECT_FALSE(IsParametrized(non_parametrized_func));
}

TEST_F(FunctionsTest, InstantiationParameters) {
  // Function definition is invalid, only type/body parameters are important.
  FunctionDef func = FunctionDefHelper::Create(
      "ParametrizedFunc",
      /* inputs */
      {"input1:A", "input2:B", "input3:float", "input4: C"},
      /* outputs */
      {"output1: A", "output2:D"},
      /* type parameters */
      {
          "A: {float, double}",
          "B: {float, int32}",
          "C: list(type)",
          "D: {float, double}",
      },
      /* function body*/
      {{{"output"}, "FakeOp", {"input1", "input2"}, {{"key", "$key"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"x", "cx:output:0"}, {"y", "cy:output:0"}});

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  func_instantiation_attr["key"].set_s("key-value");
  func_instantiation_attr["A"].set_type(DT_FLOAT);
  func_instantiation_attr["B"].set_type(DT_INT32);
  func_instantiation_attr["C"].mutable_list()->add_type(DT_FLOAT);
  func_instantiation_attr["C"].mutable_list()->add_type(DT_INT32);
  func_instantiation_attr["D"].set_type(DT_DOUBLE);

  absl::flat_hash_map<string, DataType> type_parameters;
  TF_EXPECT_OK(InstantiationTypeParameters(
      func, AttrSlice(&func_instantiation_attr), &type_parameters));

  ASSERT_EQ(5, type_parameters.size());
  EXPECT_EQ(DT_FLOAT, type_parameters["A"]);
  EXPECT_EQ(DT_INT32, type_parameters["B"]);
  EXPECT_EQ(DT_FLOAT, type_parameters["C:0"]);
  EXPECT_EQ(DT_INT32, type_parameters["C:1"]);
  EXPECT_EQ(DT_DOUBLE, type_parameters["D"]);

  absl::flat_hash_map<string, AttrValue> body_parameters;
  TF_EXPECT_OK(InstantiationBodyParameters(
      func, AttrSlice(&func_instantiation_attr), &body_parameters));

  ASSERT_EQ(1, body_parameters.size());
  EXPECT_EQ("key-value", body_parameters["key"].s());
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

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  func_instantiation_attr["T"].set_type(DT_FLOAT);
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  EXPECT_EQ("XTimesTwo", item.id);
  EXPECT_EQ(5, item.function_body().node_size());

  EXPECT_EQ(1, item.input_size());
  EXPECT_EQ("x", item.input(0).node_name);

  EXPECT_EQ(1, item.output_size());
  EXPECT_EQ("y_RetVal", item.output(0).node_name);

  int count = 0;
  for (const NodeDef &node : item.function_body().node()) {
    if (node.name() == "x" && ++count) {
      EXPECT_EQ("_Arg", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.attr().at("index").i());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "two" && ++count) {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "scale" && ++count) {
      EXPECT_EQ("Cast", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("DstT").type());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("two", node.input(0));
    } else if (node.name() == "y" && ++count) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("scale", node.input(1));
    } else if (node.name() == "y_RetVal" && ++count) {
      EXPECT_EQ("_Retval", node.op());
      ASSERT_EQ(1, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ(0, node.attr().at("index").i());
    }
  }
  EXPECT_EQ(5, count);
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

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  func_instantiation_attr["T"].set_type(DT_FLOAT);
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  EXPECT_EQ("SubGrad", item.id);
  EXPECT_EQ(14, item.function_body().node_size());

  ASSERT_EQ(3, item.input_size());
  EXPECT_EQ("x", item.input(0).node_name);
  EXPECT_EQ("y", item.input(1).node_name);
  EXPECT_EQ("dz", item.input(2).node_name);

  ASSERT_EQ(2, item.output_size());
  EXPECT_EQ("dx_RetVal", item.output(0).node_name);
  EXPECT_EQ("dy_RetVal", item.output(1).node_name);

  int count = 0;
  for (const NodeDef &node : item.function_body().node()) {
    if (node.name() == "x" || node.name() == "y" || node.name() == "dz") {
      count++;
      EXPECT_EQ("_Arg", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      int expected_index = node.name() == "x" ? 0 : node.name() == "y" ? 1 : 2;
      EXPECT_EQ(expected_index, node.attr().at("index").i());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "rx" && ++count) {
      EXPECT_EQ("BroadcastGradientArgs", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("sx", node.input(0));
      EXPECT_EQ("sy", node.input(1));
    } else if (node.name() == "sum_gx" && ++count) {
      EXPECT_EQ("Sum", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("gx", node.input(0));
      EXPECT_EQ("rx", node.input(1));
    } else if (node.name() == "sum_gy" && ++count) {
      EXPECT_EQ("Sum", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("gy", node.input(0));
      EXPECT_EQ("rx:1", node.input(1));
    } else if (node.name() == "dx_RetVal" && ++count) {
      EXPECT_EQ("_Retval", node.op());
      EXPECT_EQ(0, node.attr().at("index").i());
      ASSERT_EQ(1, node.input_size());
      EXPECT_EQ("dx", node.input(0));
    } else if (node.name() == "dy_RetVal" && ++count) {
      EXPECT_EQ("_Retval", node.op());
      EXPECT_EQ(1, node.attr().at("index").i());
      ASSERT_EQ(1, node.input_size());
      EXPECT_EQ("dy", node.input(0));
    }
  }
  EXPECT_EQ(8, count);
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

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  func_instantiation_attr["T"].set_type(DT_FLOAT);

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  int count = 0;
  for (const NodeDef &node : item.function_body().node()) {
    if (node.name() == "x" || node.name() == "y") {
      count++;
      EXPECT_EQ("_Arg", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      int expected_index = node.name() == "x" ? 0 : 1;
      EXPECT_EQ(expected_index, node.attr().at("index").i());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "a0" && ++count) {
      EXPECT_EQ("Swap", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^x2", node.input(2));
    } else if (node.name() == "a1" && ++count) {
      EXPECT_EQ("Swap", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("a0", node.input(0));
      EXPECT_EQ("a0:1", node.input(1));
    } else if (node.name() == "x2" && ++count) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(2, node.input_size());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("x", node.input(1));
    } else if (node.name() == "y2" && ++count) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ(3, node.input_size());
      EXPECT_EQ("y", node.input(0));
      EXPECT_EQ("y", node.input(1));
      EXPECT_EQ("^a1", node.input(2));
    } else if (node.name() == "o" && ++count) {
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

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  EXPECT_EQ(1, item.output_size());
  EXPECT_EQ("out_RetVal", item.output(0).node_name);

  int count = 0;
  for (const NodeDef &node : item.function_body().node()) {
    if (node.name() == "in" && ++count) {
      EXPECT_EQ("_Arg", node.op());
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
      EXPECT_EQ(0, node.attr().at("index").i());
      EXPECT_EQ(0, node.input_size());
    } else if (node.name() == "Linear_func" && ++count) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("in", node.input(0));
    } else if (node.name() == "Exp" && ++count) {
      EXPECT_EQ("Exp", node.op());
      EXPECT_EQ(1, node.input_size());
      EXPECT_EQ("Linear_func", node.input(0));
    } else if (node.name() == "out_RetVal" && ++count) {
      EXPECT_EQ("_Retval", node.op());
      EXPECT_EQ(0, node.attr().at("index").i());
      ASSERT_EQ(1, node.input_size());
      EXPECT_EQ("Exp", node.input(0));
    }
  }
  EXPECT_EQ(4, count);
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

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  func_instantiation_attr["T"].set_type(DT_FLOAT);
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  EXPECT_EQ(0, item.input_size());
  EXPECT_EQ(1, item.output_size());
  EXPECT_EQ("o_RetVal", item.output(0).node_name);
  EXPECT_EQ(3, item.function_body().node_size());

  const NodeDef &two = item.function_body().node(0);
  EXPECT_EQ("two", two.name());
  EXPECT_EQ(0, two.input_size());

  const NodeDef &cast = item.function_body().node(1);
  EXPECT_EQ("o", cast.name());
  EXPECT_EQ(1, cast.input_size());
  EXPECT_EQ("two", cast.input(0));

  const NodeDef &retval = item.function_body().node(2);
  EXPECT_EQ("o_RetVal", retval.name());
  EXPECT_EQ(1, retval.input_size());
  EXPECT_EQ("o", retval.input(0));
}

TEST_F(FunctionsTest, FromFunctionDefWithSideEffectfulOps) {
  const Tensor kOne = test::AsScalar<float>(1.0);
  FunctionDef func = FunctionDefHelper::Define(
      /* Name */ "SideEffects",
      /* Args */ {"x: Ref(float)"},
      /* Return values */ {},
      /* Attr def */ {},
      /* Nodes */
      {{{"one"}, "Const", {}, {{"value", kOne}, {"dtype", DT_FLOAT}}},
       {{"update"}, "AssignAdd", {"x", "one"}, {{"T", DT_FLOAT}}}});

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  EXPECT_EQ("SideEffects", item.id);
  EXPECT_EQ(3, item.function_body().node_size());
  EXPECT_EQ(1, item.input_size());
  EXPECT_EQ(0, item.output_size());

  const auto &opts = item.optimization_options();
  EXPECT_FALSE(opts.allow_pruning_stateful_and_dataset_ops);
}

TEST_F(FunctionsTest, FromFunctionDefWithControlOutputs) {
  const Tensor kOne = test::AsScalar<float>(1.0);
  FunctionDef func = FunctionDefHelper::Create(
      "WithControlOutputs", /*in_def=*/{"x: Ref(float)"}, /*out_def=*/{}, {},
      {
          {{"one"}, "Const", {}, {{"value", kOne}, {"dtype", DT_FLOAT}}},
          {{"update"}, "AssignAdd", {"x", "one:output:0"}, {{"T", DT_FLOAT}}},
      },
      {}, {{"side_effects", "update"}});

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  EXPECT_EQ("WithControlOutputs", item.id);
  EXPECT_EQ(3, item.function_body().node_size());
  EXPECT_EQ(1, item.input_size());
  EXPECT_EQ(0, item.output_size());

  ASSERT_EQ(1, item.keep_ops.size());
  EXPECT_EQ("update", item.keep_ops[0]);

  ASSERT_EQ(1, item.control_output_size());
  const ControlOutput &ctrl = item.control_outputs()[0];
  EXPECT_EQ("side_effects", ctrl.output_name);
  EXPECT_EQ("update", ctrl.node_name);
}

TEST_F(FunctionsTest, MakeFunctionDef) {
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

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  func_instantiation_attr["T"].set_type(DT_FLOAT);
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  FunctionDef specialized;
  TF_EXPECT_OK(MakeFunctionDef(item, flib, &specialized));

  // Input and output types are resolved based on instantiation attributes.
  EXPECT_EQ("x", specialized.signature().input_arg(0).name());
  EXPECT_EQ(DT_FLOAT, specialized.signature().input_arg(0).type());
  EXPECT_EQ("y", specialized.signature().output_arg(0).name());
  EXPECT_EQ(DT_FLOAT, specialized.signature().output_arg(0).type());

  // Function body specialized for instantiation types.
  int count = 0;
  for (const NodeDef &node : specialized.node_def()) {
    if (node.name() == "scale" && ++count) {
      EXPECT_EQ(DT_FLOAT, node.attr().at("DstT").type());
    } else if (node.name() == "y" && ++count) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ("x", node.input(0));
      EXPECT_EQ("scale:y:0", node.input(1));
      EXPECT_EQ(DT_FLOAT, node.attr().at("T").type());
    }
  }
  EXPECT_EQ(2, count);
}

TEST_F(FunctionsTest, ReplaceInputWithConst) {
  FunctionDef func = FunctionDefHelper::Create(
      "MyMul", {"x:T", "y:T"}, {"z:T"}, {"T: {float, double}"},
      {{{"output"}, "Mul", {"x", "y"}, {{"T", "$T"}}}},
      /* Mapping between function returns and function node outputs. */
      {{"z", "output:z:0"}});

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  func_instantiation_attr["T"].set_type(DT_FLOAT);
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  EXPECT_EQ(2, item.input_size());
  EXPECT_EQ(1, item.output_size());

  ASSERT_EQ(4, item.function_body().node_size());

  const NodeDef &input_x = item.function_body().node(0);
  const NodeDef &input_y = item.function_body().node(1);

  // Initially inputs added to the graph as _Arg nodes.
  EXPECT_EQ("_Arg", input_x.op());
  EXPECT_EQ("_Arg", input_y.op());

  // Replace inputs x and y with constants.
  NodeDef const_input_x;
  const_input_x.set_op("Const");
  AddNodeAttr("Tag", "const_input_x", &const_input_x);

  NodeDef const_input_y;
  const_input_y.set_op("Const");
  AddNodeAttr("Tag", "const_input_y", &const_input_y);

  // Replace input x.
  TF_EXPECT_OK(ReplaceInputWithConst(const_input_x, 0, &item));

  EXPECT_EQ(1, item.input_size());
  EXPECT_EQ("Const", input_x.op());
  EXPECT_EQ("const_input_x", input_x.attr().at("Tag").s());

  // Replace input y.
  TF_EXPECT_OK(ReplaceInputWithConst(const_input_y, 0, &item));

  EXPECT_EQ(0, item.input_size());
  EXPECT_EQ("Const", input_y.op());
  EXPECT_EQ("const_input_y", input_y.attr().at("Tag").s());

  // Make a function from const-specialized function item.
  FunctionDef specialized;
  TF_EXPECT_OK(MakeFunctionDef(item, flib, &specialized));

  EXPECT_EQ(0, specialized.signature().input_arg_size());
  EXPECT_EQ(1, specialized.signature().output_arg_size());
  EXPECT_EQ(3, specialized.node_def_size());

  // Check that graph has const nodes pushed into function body.
  int count = 0;
  for (const NodeDef &node : specialized.node_def()) {
    if (node.name() == "x" && ++count) {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ("const_input_x", node.attr().at("Tag").s());
    } else if (node.name() == "y" && ++count) {
      EXPECT_EQ("Const", node.op());
      EXPECT_EQ("const_input_y", node.attr().at("Tag").s());
    } else if (node.name() == "output" && ++count) {
      EXPECT_EQ("Mul", node.op());
      EXPECT_EQ("x:output:0", node.input(0));
      EXPECT_EQ("y:output:0", node.input(1));
    }
  }
  EXPECT_EQ(3, count);
}

TEST_F(FunctionsTest, SwapFunctionBodyAndMakeFunctionDef) {
  using ::tensorflow::test::function::NDef;

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
      {/* Read and return input argument through Identity node. */
       NDef("read_x", "Identity", {"x"}, {{"T", "float"}}),
       NDef("z_RetVal", "_Retval", {"read_x"}, {{"T", "float"}})});

  protobuf::Map<string, AttrValue> func_instantiation_attr;
  func_instantiation_attr["T"].set_type(DT_FLOAT);

  FunctionDefLibrary lib_def;
  *lib_def.add_function() = func;
  *lib_def.add_function() = mul_func;
  FunctionLibraryDefinition flib(OpRegistry::Global(), lib_def);

  GrapplerFunctionItem item;
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  // Replace function body with identity function
  item.SwapFunctionBody(std::move(id_func_body));
  FunctionDef specialized;
  TF_EXPECT_OK(MakeFunctionDef(item, flib, &specialized));

  // Check that graph body was updated.
  int count = 0;
  for (const NodeDef &node : specialized.node_def()) {
    if (node.name() == "read_x" && ++count) {
      EXPECT_EQ("Identity", node.op());
      EXPECT_EQ("x", node.input(0));
    }
  }
  EXPECT_EQ(1, count);

  // And return tensor mapping was updated with a new output name (z->read_x).
  EXPECT_EQ("read_x:output:0", (*specialized.mutable_ret())["z"]);
}

TEST_F(FunctionsTest, FunctionDefGrapplerFunctionItemRoundTrip) {
  FunctionDef func = FunctionDefHelper::Create(
      "DoNothing", /*in_def=*/{"i: int32"}, /*out_def*/ {"o: int32"},
      /*attr_def*/ {},
      {
          {{"id"}, "Identity", {"i"}, {{"T", DT_INT32}}},
      },
      /*ret_def=*/{{"o", "id:output:0"}},
      /*control_ret_def=*/{{"must_execute", "id"}});

  constexpr char description[] = "This is a helpful description.";
  func.mutable_signature()->set_description(description);
  FunctionLibraryDefinition flib(OpRegistry::Global(), FunctionDefLibrary());

  GrapplerFunctionItem item;
  protobuf::Map<string, AttrValue> func_instantiation_attr;
  func_instantiation_attr["T"].set_type(DT_INT32);
  TF_EXPECT_OK(MakeGrapplerFunctionItem(func,
                                        AttrSlice(&func_instantiation_attr),
                                        flib, TF_GRAPH_DEF_VERSION, &item));

  FunctionDef func2;
  TF_EXPECT_OK(MakeFunctionDef(item, flib, &func2));
  EXPECT_TRUE(FunctionDefsEqual(func, func2));
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
