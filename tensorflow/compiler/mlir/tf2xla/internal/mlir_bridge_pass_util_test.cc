/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/mlir/tf2xla/internal/mlir_bridge_pass_util.h"

#include <vector>

#include <gtest/gtest.h>
#include "tensorflow/cc/framework/ops.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/function_ops.h"
#include "tensorflow/compiler/tf2xla/tf2xla_defs.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/enable_tf2_utils.h"
#include "tsl/lib/core/status_test_util.h"

namespace tensorflow {

namespace {

FunctionDef OuterXTimesTwo() {
  return FunctionDefHelper::Define(
      // Name
      "OuterXTimesTwo",
      // Args
      {"x: float"},
      // Return values
      {"y: float"},
      // Attr def
      {},
      {{{"y"},
        "StatefulPartitionedCall",
        {"x"},
        {{"Tin", DataTypeSlice{DT_FLOAT}},
         {"Tout", DataTypeSlice{DT_FLOAT}},
         {"f",
          FunctionDefHelper::FunctionRef("XTimesTwoFloat", {{"T", DT_FLOAT}})},
         {std::string(kMustCompileAttr), true}}}});
}

TEST(HasCompileDeviceTypeAttr, GraphWithXlaClusters) {
  const FunctionDef& fd = test::function::XTimesTwo();
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  tensorflow::set_tf2_execution(true);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  Node* call;
  NameAttrList f_name_attr;
  f_name_attr.set_name(fd.signature().name());
  TF_ASSERT_OK(
      NodeBuilder("B", "StatefulPartitionedCall", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("Tin", {DT_FLOAT})
          .Attr("Tout", {DT_FLOAT})
          .Attr("f", f_name_attr)
          .Finalize(root.graph(), &call));
  call->AddAttr(std::string(kCompileDeviceTypeAttr), kGpuDevice);

  TF_ASSERT_OK(root.ToGraph(&graph));

  FunctionLibraryDefinition empty_flib_def(OpRegistry::Global());
  EXPECT_TRUE(
      HasCompileDeviceTypeAttr(graph, /*function_library=*/empty_flib_def));
}

TEST(HasTpuReplicateAttr, GraphWithXlaClusters) {
  const FunctionDef& fd = test::function::XTimesTwo();
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  tensorflow::set_tf2_execution(true);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  Node* call;
  NameAttrList f_name_attr;
  f_name_attr.set_name(fd.signature().name());
  TF_ASSERT_OK(
      NodeBuilder("B", "StatefulPartitionedCall", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("Tin", {DT_FLOAT})
          .Attr("Tout", {DT_FLOAT})
          .Attr("f", f_name_attr)
          .Finalize(root.graph(), &call));
  call->AddAttr(std::string(kTpuReplicateAttr), "cluster");

  TF_ASSERT_OK(root.ToGraph(&graph));

  FunctionLibraryDefinition empty_flib_def(OpRegistry::Global());
  EXPECT_TRUE(HasTpuReplicateAttr(graph, /*function_library=*/empty_flib_def));
}

TEST(IsNonReplicatedGraph, GraphWithXlaClusters) {
  const FunctionDef& fd = test::function::XTimesTwo();
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  tensorflow::set_tf2_execution(true);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  Node* call;
  NameAttrList f_name_attr;
  f_name_attr.set_name(fd.signature().name());
  TF_ASSERT_OK(
      NodeBuilder("B", "StatefulPartitionedCall", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("Tin", {DT_FLOAT})
          .Attr("Tout", {DT_FLOAT})
          .Attr("f", f_name_attr)
          .Finalize(root.graph(), &call));
  call->AddAttr(std::string(kMustCompileAttr), true);

  TF_ASSERT_OK(root.ToGraph(&graph));

  FunctionLibraryDefinition empty_flib_def(OpRegistry::Global());
  EXPECT_TRUE(IsNonReplicatedGraph(graph, /*function_library=*/empty_flib_def));
}

// Checks that HasAttr actually goes through function library.
TEST(IsNonReplicatedGraph, FunctionLibraryWithXlaClusters) {
  const FunctionDef& fd = OuterXTimesTwo();
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(OpRegistry::Global());
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  tensorflow::set_tf2_execution(true);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  // Builds a call without compilation markers that calls a function with Xla
  // clusters.
  Node* call;
  NameAttrList f_name_attr;
  f_name_attr.set_name(fd.signature().name());
  TF_ASSERT_OK(
      NodeBuilder("B", "StatefulPartitionedCall", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("Tin", {DT_FLOAT})
          .Attr("Tout", {DT_FLOAT})
          .Attr("f", f_name_attr)
          .Finalize(root.graph(), &call));

  TF_ASSERT_OK(root.ToGraph(&graph));
  EXPECT_TRUE(IsNonReplicatedGraph(graph, /*function_library=*/flib_def));
}

TEST(IsSingleCoreTpuGraph, GraphWithXlaClusters) {
  const FunctionDef& fd = test::function::XTimesTwo();
  FunctionDefLibrary flib;
  *flib.add_function() = fd;
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), flib);
  Graph graph(flib_def);
  graph.SetConstructionContext(ConstructionContext::kEagerRuntime);
  tensorflow::set_tf2_execution(true);

  ConfigProto config = ConfigProto();
  Scope root = Scope::NewRootScope().ExitOnError();

  Output a = ops::_Arg(root.WithOpName("A"), DT_FLOAT, 0);
  std::vector<NodeBuilder::NodeOut> inputs({NodeBuilder::NodeOut(a.node())});

  Node* call;
  NameAttrList f_name_attr;
  f_name_attr.set_name(fd.signature().name());
  TF_ASSERT_OK(
      NodeBuilder("B", "StatefulPartitionedCall", &root.graph()->flib_def())
          .Input(inputs)
          .Attr("Tin", {DT_FLOAT})
          .Attr("Tout", {DT_FLOAT})
          .Attr("f", f_name_attr)
          .Finalize(root.graph(), &call));
  call->AddAttr(std::string(kCompileDeviceTypeAttr), kTpuDevice);

  TF_ASSERT_OK(root.ToGraph(&graph));

  FunctionLibraryDefinition empty_flib_def(OpRegistry::Global());
  EXPECT_TRUE(IsSingleCoreTpuGraph(graph, /*function_library=*/empty_flib_def));
}

}  // namespace

}  // namespace tensorflow
