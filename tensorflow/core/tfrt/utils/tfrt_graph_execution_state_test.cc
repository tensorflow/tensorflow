/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/core/tfrt/utils/tfrt_graph_execution_state.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/cc/ops/state_ops.h"
#include "tensorflow/cc/ops/while_loop.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/compiler/mlir/tensorflow/utils/attribute_utils.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/device_factory.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/utils/grappler_test.h"
#include "tensorflow/core/lib/monitoring/cell_reader.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"
#include "tensorflow/core/tfrt/graph_executor/config.h"
#include "tsl/platform/statusor.h"

namespace tensorflow {
namespace tfrt_stub {
namespace {

using ::testing::_;
using ::testing::ElementsAre;
using ::testing::EqualsProto;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::Pair;
using ::testing::proto::IgnoringFieldPaths;
using ::testing::proto::IgnoringRepeatedFieldOrdering;

class PruneGraphDefTest : public grappler::GrapplerTest {};

TEST_F(PruneGraphDefTest, ConstFeedWithInput) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 0.0f, {10, 10});

    Output b = ops::Const(scope.WithControlDependencies(a).WithOpName("b"),
                          0.0f, {10, 10});
    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  CallableOptions callable_options;
  callable_options.add_feed("b");
  callable_options.add_fetch("c");

  TF_ASSERT_OK(PruneGraphDef(graphdef, callable_options));

  GraphDef expected;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output b = ops::Const(scope.WithOpName("b"), 0.0f, {10, 10});
    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  CompareGraphs(expected, graphdef);
}

absl::Status LessThanTenCond(const Scope& scope,
                             const std::vector<Output>& inputs,
                             Output* output) {
  *output = ops::Less(scope, inputs[0], 10);
  return scope.status();
}

absl::Status AddOneBody(const Scope& scope, const std::vector<Output>& inputs,
                        std::vector<Output>* outputs) {
  outputs->push_back(ops::AddN(scope, {inputs[0], 1}));
  return scope.status();
}

TEST_F(PruneGraphDefTest, InsertIdentityForLoopExitFeed) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    std::vector<Output> inputs;
    inputs.push_back(ops::Placeholder(scope.WithOpName("input"), DT_INT32));
    std::vector<Output> outputs;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop",
                                     &outputs));

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  CallableOptions callable_options;
  callable_options.add_feed("input");
  callable_options.add_fetch("while/Exit");

  TF_ASSERT_OK(PruneGraphDef(graphdef, callable_options));

  for (const auto& node : graphdef.node()) {
    if (node.op() == "Exit") {
      EXPECT_EQ(node.name(), "while/Exit/tfrt_renamed");
    }
    if (node.name() == "while/Exit") {
      EXPECT_EQ(node.op(), "Identity");
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "while/Exit/tfrt_renamed");
    }
  }
}

TEST_F(PruneGraphDefTest, EliminateRefEntersFromControlFlow) {
  GraphDef graphdef;
  absl::flat_hash_map<std::string, NodeDef> name_to_node;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    std::vector<Output> inputs;
    inputs.push_back(ops::Placeholder(scope.WithOpName("input"), DT_INT32));
    std::vector<Output> outputs1;
    std::vector<Output> outputs2;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop",
                                     &outputs1));
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop2",
                                     &outputs2));

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));

    // Simply replace Enter with RefEnter. Note this is not valid graph though.
    for (auto& node : *graphdef.mutable_node()) {
      if (node.op() == "Enter") {
        node.set_op("RefEnter");
      }
      name_to_node.insert({node.name(), node});
    }
  }

  TF_ASSERT_OK(EliminateRefVariablesFromV1ControlFlow(graphdef));

  int num_identity_op = 0;
  int num_enter_op = 0;
  int num_ref_enter_op = 0;
  for (const auto& node : graphdef.node()) {
    if (node.op() == "Identity") {
      num_identity_op++;
      EXPECT_EQ(node.name(), "input/identity");
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_THAT(node.attr(), ElementsAre(Pair("T", _)));
    } else if (node.op() == "RefEnter") {
      num_ref_enter_op++;
    } else if (node.op() == "Enter") {
      // Identity op should be placed before Enter.
      EXPECT_EQ(num_identity_op, 1);
      num_enter_op++;
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "input/identity");
      EXPECT_THAT(
          node, IgnoringFieldPaths({"input", "op"},
                                   EqualsProto(name_to_node.at(node.name()))));
    } else {
      EXPECT_THAT(node, EqualsProto(name_to_node.at(node.name())));
    }
    name_to_node.erase(node.name());
  }
  EXPECT_EQ(num_identity_op, 1);
  EXPECT_EQ(num_enter_op, 2);
  EXPECT_EQ(num_ref_enter_op, 0);
  EXPECT_THAT(name_to_node, IsEmpty());
}

TEST_F(PruneGraphDefTest, EliminateRefSwitchesFromControlFlow) {
  GraphDef graphdef;
  absl::flat_hash_map<std::string, NodeDef> name_to_node;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output cond_a = ops::Placeholder(scope.WithOpName("cond_a"), DT_BOOL);
    Output cond_b = ops::Placeholder(scope.WithOpName("cond_b"), DT_BOOL);
    Output input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);

    ops::Switch switch_a(scope.WithOpName("switch_a"), input, cond_a);
    ops::Switch switch_b(scope.WithOpName("switch_b"), input, cond_b);

    Output switch_a_true =
        ops::Identity(scope.WithOpName("switch_a_true"), switch_a.output_true);
    Output switch_b_true =
        ops::Identity(scope.WithOpName("switch_b_true"), switch_b.output_true);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));

    // Simply replace Switch with RefSwitch. Note this is not valid graph
    // though.
    for (auto& node : *graphdef.mutable_node()) {
      if (node.op() == "Switch") {
        node.set_op("RefSwitch");
      }
      name_to_node.insert({node.name(), node});
    }
  }

  TF_ASSERT_OK(EliminateRefVariablesFromV1ControlFlow(graphdef));

  int num_identity_op = 0;
  int num_switch_op = 0;
  int num_ref_switch_op = 0;
  for (const auto& node : graphdef.node()) {
    if (node.name() == "switch_a_true" || node.name() == "switch_b_true") {
      EXPECT_THAT(node, EqualsProto(name_to_node.at(node.name())));
    } else if (node.op() == "Identity") {
      num_identity_op++;
      EXPECT_EQ(node.name(), "input/identity");
      ASSERT_EQ(node.input().size(), 1);
      EXPECT_EQ(node.input(0), "input");
      EXPECT_THAT(node.attr(), ElementsAre(Pair("T", _)));
    } else if (node.op() == "RefSwitch") {
      num_ref_switch_op++;
    } else if (node.op() == "Switch") {
      // Identity op should be placed before Switch.
      EXPECT_EQ(num_identity_op, 1);
      num_switch_op++;
      ASSERT_EQ(node.input().size(), 2);
      EXPECT_TRUE(node.input(0) == "input/identity" ||
                  node.input(1) == "input/identity");
      EXPECT_THAT(
          node, IgnoringFieldPaths({"input", "op"},
                                   EqualsProto(name_to_node.at(node.name()))));
    } else {
      EXPECT_THAT(node, EqualsProto(name_to_node.at(node.name())));
    }
    name_to_node.erase(node.name());
  }
  EXPECT_EQ(num_identity_op, 1);
  EXPECT_EQ(num_switch_op, 2);
  EXPECT_EQ(num_ref_switch_op, 0);
  EXPECT_THAT(name_to_node, IsEmpty());
}

TEST_F(PruneGraphDefTest, EliminateRefVariablesFromV1ControlFlowFailed) {
  GraphDef graphdef;
  absl::flat_hash_map<std::string, NodeDef> name_to_node;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output cond = ops::Placeholder(scope.WithOpName("cond"), DT_BOOL);
    Output input = ops::Placeholder(scope.WithOpName("input"), DT_FLOAT);

    ops::Switch switch_op(scope.WithOpName("switch"), input, cond);
    Output var = ops::Variable(scope.WithOpName("var"), {}, DataType::DT_FLOAT);
    Output assign =
        ops::Assign(scope.WithOpName("assign"), var, switch_op.output_true);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));

    // Simply replace Switch with RefSwitch. Note this is not valid graph
    // though.
    for (auto& node : *graphdef.mutable_node()) {
      if (node.op() == "Switch") {
        node.set_op("RefSwitch");
      }
      name_to_node.insert({node.name(), node});
    }
  }

  const auto status = EliminateRefVariablesFromV1ControlFlow(graphdef);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(status.ToString(), HasSubstr("requires its input to be refs"));
}

TEST_F(PruneGraphDefTest, KeepLoopStructureComplete) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    std::vector<Output> inputs;
    inputs.push_back(ops::Placeholder(scope.WithOpName("input"), DT_INT32));
    std::vector<Output> outputs;
    TF_ASSERT_OK(ops::BuildWhileLoop(scope.NewSubScope("while"), inputs,
                                     LessThanTenCond, AddOneBody, "test_loop",
                                     &outputs));

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  CallableOptions callable_options;
  callable_options.add_feed("input");
  // Sets the fetch node such that traversing from there will miss part of the
  // while loop structure.
  callable_options.add_fetch("while/LoopCond");

  GraphDef original_graphdef = graphdef;
  TF_ASSERT_OK(PruneGraphDef(graphdef, callable_options));
  EXPECT_THAT(graphdef,
              IgnoringRepeatedFieldOrdering(EqualsProto(original_graphdef)));
}

class OptimizeGraphTest : public grappler::GrapplerTest {};

TEST_F(OptimizeGraphTest, OptimizeFunctions) {
  GraphDef graphdef;
  tensorflow::FunctionDefLibrary fdef_lib;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    auto fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y: float"}, {},
        {{{"three"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kThree}}},
         {{"pow3"}, "Pow", {"x", "three:output:0"}, {{"T", DT_FLOAT}}}},
        {{"y", "pow3:z:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(fdef.signature().name());
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output b = pcall.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create({}, fdef_lib));

  TfrtGraphExecutionState::Options options;
  options.run_placer_grappler_on_functions = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(options, graphdef, *fallback_state));

  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  tensorflow::ArrayInfo array_info;
  array_info.imported_dtype = DT_FLOAT;
  array_info.shape.set_unknown_rank(true);
  graph_import_config.inputs["a"] = array_info;
  graph_import_config.outputs = {"c"};

  TF_ASSERT_OK_AND_ASSIGN(
      auto optimized_graph,
      graph_execution_state->CreateOptimizedGraph(graph_import_config));
  GraphDef optimized_graph_def;
  optimized_graph.graph->ToGraphDef(&optimized_graph_def);

  GraphDef expected;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    // After optimization, "x^3" will be transformed to "(x^2)*x".
    auto fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y_retval: float"}, {},
        {{{"ArithmeticOptimizer/ConvertPow__inner_pow3"},
          "Square",
          {"x"},
          {{"dtype", DT_FLOAT}},
          /*dep=*/{},
          "/job:localhost/replica:0/task:0/device:CPU:0"},
         {{"pow3"},
          "Mul",
          {"ArithmeticOptimizer/ConvertPow__inner_pow3:y:0", "x"},
          {{"T", DT_FLOAT}},
          /*dep=*/{},
          "/job:localhost/replica:0/task:0/device:CPU:0"}},
        {{"y_retval", "pow3:z:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(fdef.signature().name());
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output b = pcall.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  CompareGraphs(expected, optimized_graph_def);
  CompareFunctions(expected.library().function(0),
                   optimized_graph_def.library().function(0));
}

TEST_F(OptimizeGraphTest, OptimizeFunctionsUsedByFunctionNodes) {
  GraphDef graphdef;
  tensorflow::FunctionDefLibrary fdef_lib;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    auto pow3_fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y: float"}, {},
        {{{"three"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kThree}}},
         {{"pow3"}, "Pow", {"x", "three:output:0"}, {{"T", DT_FLOAT}}}},
        {{"y", "pow3:z:0"}});

    const Tensor kOne = test::AsScalar<float>(1.0);
    auto base2pow3_fdef = tensorflow::FunctionDefHelper::Create(
        "Add1Pow3", {"x: float"}, {"y: float"}, {},
        {{{"one"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kOne}}},
         {{"add"}, "Add", {"x", "one:output:0"}, {{"T", DT_FLOAT}}},
         {{"pcall"},
          "PartitionedCall",
          {"add:z:0"},
          {{"Tin", DataTypeSlice({DT_FLOAT})},
           {"Tout", DataTypeSlice({DT_FLOAT})},
           {"f", tensorflow::FunctionDefHelper::FunctionRef(
                     "Pow3", {{"T", DT_FLOAT}})}}}},
        {{"y", "pcall:output:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = pow3_fdef;
    *fdef_lib.add_function() = base2pow3_fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 1.0, {1, 1});

    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        base2pow3_fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(base2pow3_fdef.signature().name());
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output b = pcall.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create({}, fdef_lib));

  TfrtGraphExecutionState::Options options;
  options.run_placer_grappler_on_functions = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(options, graphdef, *fallback_state));

  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  tensorflow::ArrayInfo array_info;
  array_info.imported_dtype = DT_FLOAT;
  array_info.shape.set_unknown_rank(true);
  graph_import_config.inputs["a"] = array_info;
  graph_import_config.outputs = {"c"};

  TF_ASSERT_OK_AND_ASSIGN(
      auto optimized_graph,
      graph_execution_state->CreateOptimizedGraph(graph_import_config));
  GraphDef optimized_graph_def;
  optimized_graph.graph->ToGraphDef(&optimized_graph_def);

  GraphDef expected;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    // After optimization, "x^3" will be transformed to "(x^2)*x".
    auto pow3_fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y_retval: float"}, {},
        {{{"ArithmeticOptimizer/ConvertPow__inner_pow3"},
          "Square",
          {"x"},
          {{"dtype", DT_FLOAT}},
          /*dep=*/{},
          "/job:localhost/replica:0/task:0/device:CPU:0"},
         {{"pow3"},
          "Mul",
          {"ArithmeticOptimizer/ConvertPow__inner_pow3:y:0", "x"},
          {{"T", DT_FLOAT}},
          /*dep=*/{},
          "/job:localhost/replica:0/task:0/device:CPU:0"}},
        {{"y_retval", "pow3:z:0"}});

    const Tensor kOne = test::AsScalar<float>(1.0);
    auto base2pow3_fdef = tensorflow::FunctionDefHelper::Create(
        "Add1Pow3", {"x: float"}, {"y: float"}, {},
        {{{"one"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kOne}}},
         {{"add"}, "Add", {"x", "one:output:0"}, {{"T", DT_FLOAT}}},
         {{"pcall"},
          "PartitionedCall",
          {"add:z:0"},
          {{"Tin", DataTypeSlice({DT_FLOAT})},
           {"Tout", DataTypeSlice({DT_FLOAT})},
           {"f", tensorflow::FunctionDefHelper::FunctionRef(
                     "Pow3", {{"T", DT_FLOAT}})}}}},
        {{"y", "pcall:output:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = pow3_fdef;
    *fdef_lib.add_function() = base2pow3_fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 1.0, {1, 1});

    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        base2pow3_fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(base2pow3_fdef.signature().name());
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output b = pcall.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  // Since `Pow3` is called by `Add1Pow3`, it is optimized.
  CompareFunctions(expected.library().function(1),
                   optimized_graph_def.library().function(1));
  ASSERT_EQ("Pow3",
            optimized_graph_def.library().function(1).signature().name());
}

TEST_F(OptimizeGraphTest, DontOptimizeUnsafeFunction) {
  GraphDef graphdef;
  tensorflow::FunctionDefLibrary fdef_lib;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    auto fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y: float"}, {},
        {{{"three"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kThree}}},
         {{"pow3"}, "Pow", {"x", "three:output:0"}, {{"T", DT_FLOAT}}}},
        {{"y", "pow3:z:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    Output cond = ops::Const(scope.WithOpName("cond"), true, {1, 1});
    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(fdef.signature().name());
    auto if_op =
        ops::If(scope, cond, inputs, output_dtypes, func_attr, func_attr);
    Output b = if_op.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create({}, fdef_lib));

  TfrtGraphExecutionState::Options options;
  options.run_placer_grappler_on_functions = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(options, graphdef, *fallback_state));

  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  tensorflow::ArrayInfo array_info;
  array_info.imported_dtype = DT_FLOAT;
  array_info.shape.set_unknown_rank(true);
  graph_import_config.inputs["a"] = array_info;
  graph_import_config.outputs = {"c"};

  TF_ASSERT_OK_AND_ASSIGN(
      auto optimized_graph,
      graph_execution_state->CreateOptimizedGraph(graph_import_config));
  GraphDef optimized_graph_def;
  optimized_graph.graph->ToGraphDef(&optimized_graph_def);

  // The optimized graph remains the same as the original one, because the
  // function used by `If` op is not optimized.
  CompareGraphs(graphdef, optimized_graph_def);
  CompareFunctions(graphdef.library().function(0),
                   optimized_graph_def.library().function(0));
}

TEST_F(OptimizeGraphTest, FunctionBecomeUnsafeIfAnyOpIsUnsafe) {
  GraphDef graphdef;
  tensorflow::FunctionDefLibrary fdef_lib;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    auto fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y: float"}, {},
        {{{"three"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kThree}}},
         {{"pow3"}, "Pow", {"x", "three:output:0"}, {{"T", DT_FLOAT}}}},
        {{"y", "pow3:z:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    Output cond = ops::Const(scope.WithOpName("cond"), true, {1, 1});
    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(fdef.signature().name());
    auto if_op =
        ops::If(scope, cond, inputs, output_dtypes, func_attr, func_attr);
    Output b = if_op.output.front();

    inputs = {b};
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output c = pcall.output.front();

    Output d = ops::Identity(scope.WithOpName("d"), c);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create({}, fdef_lib));

  TfrtGraphExecutionState::Options options;
  options.run_placer_grappler_on_functions = true;
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(options, graphdef, *fallback_state));

  tensorflow::GraphImportConfig graph_import_config;
  graph_import_config.prune_unused_nodes = true;
  graph_import_config.enable_shape_inference = false;
  tensorflow::ArrayInfo array_info;
  array_info.imported_dtype = DT_FLOAT;
  array_info.shape.set_unknown_rank(true);
  graph_import_config.inputs["a"] = array_info;
  graph_import_config.outputs = {"d"};

  TF_ASSERT_OK_AND_ASSIGN(
      auto optimized_graph,
      graph_execution_state->CreateOptimizedGraph(graph_import_config));
  GraphDef optimized_graph_def;
  optimized_graph.graph->ToGraphDef(&optimized_graph_def);

  // Both `If` and `PartitionedCall` ops use the function, so the function
  // remains unoptimized.
  CompareFunctions(graphdef.library().function(0),
                   optimized_graph_def.library().function(0));
}

TEST_F(OptimizeGraphTest, MLIRBridgeCanBeDisabled) {
  monitoring::testing::CellReader<int64_t> check_mlir_bridge_disabled_reader(
      "/tensorflow/core/tf_mlir_bridge_first_phase_v2_count");
  GraphDef graphdef;
  tensorflow::FunctionDefLibrary fdef_lib;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice(
        "/job:localhost/replica:0/task:0/device:CPU:0");

    const Tensor kThree = test::AsScalar<float>(3.0);
    auto fdef = tensorflow::FunctionDefHelper::Create(
        "Pow3", {"x: float"}, {"y: float"}, {},
        {{{"three"}, "Const", {}, {{"dtype", DT_FLOAT}, {"value", kThree}}},
         {{"pow3"}, "Pow", {"x", "three:output:0"}, {{"T", DT_FLOAT}}}},
        {{"y", "pow3:z:0"}});

    tensorflow::FunctionDefLibrary fdef_lib;
    *fdef_lib.add_function() = fdef;
    TF_ASSERT_OK(scope.graph()->AddFunctionLibrary(fdef_lib));

    Output a = ops::Const(scope.WithOpName("a"), 2.0, {1, 1});

    std::vector<tensorflow::Output> inputs = {a};
    std::vector<tensorflow::DataType> output_dtypes = {
        fdef.signature().output_arg(0).type()};
    tensorflow::NameAttrList func_attr;
    func_attr.set_name(fdef.signature().name());
    auto pcall = ops::PartitionedCall(scope, inputs, output_dtypes, func_attr);
    Output b = pcall.output.front();

    Output c = ops::Identity(scope.WithOpName("c"), b);

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create({}, fdef_lib));

  TfrtGraphExecutionState::Options options;
  options.run_placer_grappler_on_functions = true;

  tensorflow::tfrt_stub::RuntimeConfig runtime_config;
  tensorflow::tf2xla::v1::MlirBridgeConfig mlir_bridge_config;
  mlir_bridge_config.set_enable_tf2xla_mlir_bridge(false);
  TF_ASSERT_OK(runtime_config.Add(mlir_bridge_config));

  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(options, graphdef, *fallback_state,
                                      &runtime_config));

  EXPECT_EQ(
      check_mlir_bridge_disabled_reader.Delta(
          mlir::TF::kMlirPh1BridgeCounterReplicated,
          mlir::TF::kMlirPh1BridgeCounterV1, mlir::TF::kMlirPh1BridgeCounterTpu,
          "fallback_enabled", "disabled_by_user"),
      1);
}

class ExtendGraphTest : public grappler::GrapplerTest {};

TEST_F(ExtendGraphTest, ExtendGraph) {
  GraphDef graphdef;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 0.0f, {10, 10});

    TF_ASSERT_OK(scope.ToGraphDef(&graphdef));
  }

  SessionOptions session_options;
  // Disable optimizations for static graph to allow calls to Session::Extend.
  session_options.config.mutable_experimental()
      ->set_disable_optimize_for_static_graph(true);
  TF_ASSERT_OK_AND_ASSIGN(
      auto fallback_state,
      tensorflow::tfrt_stub::FallbackState::Create(session_options, {}));

  TfrtGraphExecutionState::Options options;
  options.run_placer_grappler_on_functions = false;
  TF_ASSERT_OK_AND_ASSIGN(
      auto graph_execution_state,
      TfrtGraphExecutionState::Create(options, graphdef, *fallback_state));

  GraphDef extension;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output b = ops::Const(scope.WithOpName("b"), 0.0f, {10, 10});

    TF_ASSERT_OK(scope.ToGraphDef(&extension));
  }

  TF_ASSERT_OK(graph_execution_state->Extend(extension));

  GraphDef expected;
  {
    auto scope = tensorflow::Scope::NewRootScope().WithDevice("/device:CPU:0");

    Output a = ops::Const(scope.WithOpName("a"), 0.0f, {10, 10});

    Output b = ops::Const(scope.WithOpName("b"), 0.0f, {10, 10});

    TF_ASSERT_OK(scope.ToGraphDef(&expected));
  }

  ASSERT_NE(graph_execution_state->original_graph_def(), nullptr);
  CompareGraphs(expected, *graph_execution_state->original_graph_def());
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
