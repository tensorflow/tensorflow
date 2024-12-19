/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/graph_to_tf_executor_util.h"

#include <initializer_list>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/array_ops.h"
#include "tensorflow/cc/ops/control_flow_ops.h"
#include "tensorflow/cc/ops/functional_ops.h"
#include "tensorflow/cc/ops/tpu_functional_ops.h"
#include "tensorflow/cc/ops/tpu_replication_ops.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/enable_tf2_utils.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/platform/status.h"

namespace tensorflow {

namespace {

REGISTER_OP("OneRefOutput").Output("y: Ref(float)");

FunctionDef XTimesTwo() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  return FunctionDefHelper::Define(
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
}

FunctionDef XTimesTwoFloat() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  return FunctionDefHelper::Define(
      // Name
      "XTimesTwoFloat",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64}}},
          {{"scale"},
           "Cast",
           {"two"},
           {{"SrcT", DT_INT64}, {"DstT", DT_FLOAT}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", DT_FLOAT}}},
      });
}

FunctionDef XTimesTwoFloatRef() {
  const Tensor kTwo = test::AsScalar<int64>(2);
  return FunctionDefHelper::Define(
      // Name
      "XTimesTwoFloatRef",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      // Nodes
      {
          {{"two"}, "Const", {}, {{"value", kTwo}, {"dtype", DT_INT64_REF}}},
          {{"scale"},
           "Cast",
           {"two"},
           {{"SrcT", DT_INT64_REF}, {"DstT", DT_FLOAT}}},
          {{"y"}, "Mul", {"x", "scale"}, {{"T", DT_FLOAT}}},
      });
}

Node* FromNodeDef(absl::string_view name, absl::string_view node_type,
                  int num_inputs, DataType dt, Graph& graph) {
  auto builder = NodeDefBuilder(name, node_type);
  for (int i = 0; i < num_inputs; ++i) {
    builder = builder.Input(absl::StrCat("node_", i), i, dt);
  }

  NodeDef node_def;
  TF_CHECK_OK(builder.Finalize(&node_def));

  absl::Status s;
  Node* node = graph.AddNode(node_def, &s);
  TF_CHECK_OK(s);
  return node;
}

TEST(SupportedGraphTest, SupportedGraphReturnsFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = tensorflow::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = tensorflow::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = tensorflow::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = tensorflow::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  tensorflow::set_tf2_execution(true);
  (void)tensorflow::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InvalidGraphTest, InvalidFuncBodyReturnsTrue) {
  tensorflow::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwo();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwo");
  ops::PartitionedCall f(root.WithOpName("f"), {x}, {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  // The call to XTimesTwo is invalid (missing an attribute), so we expect the
  // graph to be unsupported.
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(RefVarTest, RefVariablesReturnsTrue) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  // Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);
  tensorflow::set_tf2_execution(true);
  const std::vector<int32> shape_array{2, 2};
  auto shape = TensorShape();
  TF_ASSERT_OK(TensorShapeUtils::MakeShape(shape_array, &shape));
  Output value = Output(
      FromNodeDef("value", "OneRefOutput", 0, DT_FLOAT_REF, *root.graph()));

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(RefVarTest, NoRefVariablesCalleeFuncReturnsFalse) {
  tensorflow::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::PartitionedCall f(root.WithOpName("f"), {x}, {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(RefVarTest, RefVariablesInCalleeFunctionReturnsTrue) {
  tensorflow::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloatRef();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloatRef");
  ops::PartitionedCall f(root.WithOpName("f"), {x}, {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(RefVarTest, RefVariablesInExternalCalleeFunctionReturnsTrue) {
  tensorflow::set_tf2_execution(true);
  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloatRef();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloatRef");
  ops::PartitionedCall f(root.WithOpName("f"), {x}, {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/&flib_def, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, ContainsInferenceNodeEagerRuntimeReturnsTrue) {
  tensorflow::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, ContainsInferenceNodeTFRTBridgeReturnsTrue) {
  tensorflow::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kTFRTNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, ContainsInferenceNodeDirectSessionReturnsFalse) {
  tensorflow::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(ControlFlowTest, ContainsV1ControlFlowReturnsTrue) {
  tensorflow::set_tf2_execution(true);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);

  ops::Switch switch_a(root.WithOpName("switch_a"), value, cond_a);
  ops::Switch switch_b(root.WithOpName("switch_b"), value, cond_b);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(ControlFlowTest, TFRTContainsV1ControlFlowReturnsTrue) {
  tensorflow::set_tf2_execution(true);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  Output value = ops::Placeholder(root.WithOpName("value"), DT_FLOAT);

  ops::Switch switch_a(root.WithOpName("switch_a"), value, cond_a);
  ops::Switch switch_b(root.WithOpName("switch_b"), value, cond_b);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kTFRTNominal,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF1ReturnsTrue) {
  tensorflow::set_tf2_execution(false);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = tensorflow::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = tensorflow::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = tensorflow::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = tensorflow::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)tensorflow::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionFalseV1CompatBridgeReturnTrue) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = tensorflow::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = tensorflow::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = tensorflow::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = tensorflow::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)tensorflow::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  tensorflow::set_tf2_execution(false);

  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionTrueV1CompatBridgeReturnFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = tensorflow::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = tensorflow::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = tensorflow::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = tensorflow::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)tensorflow::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  tensorflow::set_tf2_execution(true);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionFalseTfrtNominalBridgeReturnFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = tensorflow::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = tensorflow::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = tensorflow::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = tensorflow::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)tensorflow::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  tensorflow::set_tf2_execution(false);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kTFRTNominal,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionTrueTfrtNominalBridgeReturnFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = tensorflow::ops::Placeholder(root.WithOpName("input"), DT_UINT8);
  auto depth = tensorflow::ops::Placeholder(root.WithOpName("depth"), DT_INT32);
  auto on = tensorflow::ops::Placeholder(root.WithOpName("on"), DT_UINT8);
  auto off = tensorflow::ops::Placeholder(root.WithOpName("off"), DT_UINT8);
  (void)tensorflow::ops::OneHot(root.WithOpName("output"), input, depth, on,
                                off);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  tensorflow::set_tf2_execution(true);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kTFRTNominal,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionFalseNominalBridgeReturnsFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = tensorflow::ops::Placeholder(root.WithOpName("input"), DT_UINT8);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  tensorflow::set_tf2_execution(false);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(TFVersionTest, TF2ExecutionTrueNominalBridgeReturnsFalse) {
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input = tensorflow::ops::Placeholder(root.WithOpName("input"), DT_UINT8);

  Graph graph(OpRegistry::Global());
  TF_ASSERT_OK(root.ToGraph(&graph));
  tensorflow::set_tf2_execution(true);

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(UnsupportedOpTest,
     InfeedDequeueTupleWithTPUReplicatedCoreAttrNotSupported) {
  tensorflow::set_tf2_execution(true);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      tensorflow::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);

  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  // Build InfeedDequeueTuple node with TPU_REPLICATED_CORE Attr
  auto builder = NodeDefBuilder("InfeedDequeueTuple", "InfeedDequeueTuple");
  builder.Attr("dtypes", DT_FLOAT);
  builder.Attr("shapes", 1);
  NodeDef node_def;
  TF_CHECK_OK(builder.Finalize(&node_def));
  absl::Status s;
  Node* node_InfeedDequeueTuple = (*root.graph()).AddNode(node_def, &s);
  node_InfeedDequeueTuple->set_requested_device(
      "/device:TPU_REPLICATED_CORE:0");
  TF_CHECK_OK(s);
  ASSERT_NE(node_InfeedDequeueTuple, nullptr);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/true));
}

TEST(ManualControlDependencyTest,
     TPUReplicatedCoreWithManualControlDependencyReturnsFalse) {
  tensorflow::set_tf2_execution(true);
  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      tensorflow::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);

  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  auto metadata = tensorflow::ops::TPUReplicateMetadata(root, 2);
  metadata.operation.node()->AddAttr("_has_manual_control_dependencies", true);

  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  TF_ASSERT_OK(root.ToGraph(&graph));

  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/true));
}

TEST(InferenceTest,
     ContainsInferenceNodeTPUReplicatedCoreDirectSessionReturnsFalse) {
  tensorflow::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      tensorflow::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);
  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest,
     ContainsInferenceNodeTPUReplicatedCoreEagerRuntimeReturnsTrue) {
  tensorflow::set_tf2_execution(true);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      tensorflow::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);
  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kNominal,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, TF2ExecutionFalseV1CompatBridgeReturnFalse) {
  tensorflow::set_tf2_execution(false);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      tensorflow::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);
  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_FALSE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

TEST(InferenceTest, V1CompatBridgeVariableRefReturnTrue) {
  tensorflow::set_tf2_execution(false);
  FunctionDefLibrary flib;
  *flib.add_function() = XTimesTwoFloat();
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kDirectSession);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  auto input =
      tensorflow::ops::Placeholder(root.WithOpName("node_0"), DT_FLOAT);
  auto node = FromNodeDef("Identity", "Identity", 1, DT_FLOAT, *root.graph());
  ASSERT_NE(node, nullptr);
  node->set_requested_device("/device:TPU_REPLICATED_CORE:0");

  Output x = ops::Placeholder(root.WithOpName("x"), DT_FLOAT);
  NameAttrList f_name_attr;
  f_name_attr.set_name("XTimesTwoFloat");
  ops::TPUPartitionedCall f(root.WithOpName("f"), {x}, /*device_ordinal=*/0,
                            {DT_FLOAT}, f_name_attr);

  Output cond_a = ops::Placeholder(root.WithOpName("cond_a"), DT_BOOL);
  Output cond_b = ops::Placeholder(root.WithOpName("cond_b"), DT_BOOL);

  tensorflow::set_tf2_execution(true);
  const std::vector<int32> shape_array{2, 2};
  auto shape = TensorShape();
  TF_ASSERT_OK(TensorShapeUtils::MakeShape(shape_array, &shape));
  Output value = Output(
      FromNodeDef("value", "OneRefOutput", 0, DT_FLOAT_REF, *root.graph()));

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(GraphHasUnsupportedFeaturesInMlirBridge(
      graph, /*function_library=*/nullptr, config,
      /*bridge_version=*/tensorflow::TF2XLABridgeVersion::kV1Compat,
      /*single_core_inference_mode=*/false));
}

}  // namespace

}  // namespace tensorflow
