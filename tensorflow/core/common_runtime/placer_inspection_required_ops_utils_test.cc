/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/placer_inspection_required_ops_utils.h"

#include <map>

#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/graph_def_builder_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using FDH = ::tensorflow::FunctionDefHelper;

// Returns void so that we can call TF_ASSERT_OK inside it.
void VerifyPlacerInspectionRequiredOps(const GraphDef& graph_def,
                                       std::map<string, bool> deep_nodes) {
  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));
  PlacerInspectionRequiredOpChecker checker(&graph);
  std::unordered_map<string, Node*> node_map = graph.BuildNodeNameIndex();
  for (const auto& entry : deep_nodes) {
    const Node* node = node_map[entry.first];
    ASSERT_NE(node, nullptr) << "Failed to find node " << entry.first
                             << " in the graph " << graph_def.DebugString();
    const bool expected_is_deep = entry.second;
    bool actual_is_deep;
    TF_EXPECT_OK(checker.IsPlacerInspectionRequired(*node, &actual_is_deep));
    EXPECT_EQ(expected_is_deep, actual_is_deep)
        << " Expected is_deep to be " << expected_is_deep << " for node "
        << entry.first;
  }
}

TEST(PlacerInspectionRequiredOpCheckerTest, Basic) {
  /*
   *                x (_Arg, DT_RESOURCE)
   *                   |
   *                   v
   *                f (PartitionedCallOp: ResourceIdentity)
   *                   |
   *                   v
   *                y (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::ResourceIdentity();
  GraphDef graph_def = GDef(
      {
          NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"x"},
               {{"Tin", DataTypeSlice{DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE}},
                {"f", FDH::FunctionRef("ResourceIdentity", {})}}),
          NDef("y", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  VerifyPlacerInspectionRequiredOps(graph_def,
                                    {{"x", false}, {"f", true}, {"y", false}});
}

TEST(PlacerInspectionRequiredOpCheckerTest, DirectCallsAreNotDeep) {
  /*
   *                x (_Arg, DT_RESOURCE)
   *                   |
   *                   v
   *                f (direct function call to ResourceIdentity)
   *                   |
   *                   v
   *                y (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::ResourceIdentity();
  GraphDef graph_def = GDef(
      {
          NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "ResourceIdentity", {"x"}),
          NDef("y", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  VerifyPlacerInspectionRequiredOps(graph_def,
                                    {{"x", false}, {"f", false}, {"y", false}});
}

TEST(PlacerInspectionRequiredOpCheckerTest,
     FunctionsNotReturningResourcesAreNotDeep) {
  /*
   *                x (_Arg, DT_RESOURCE)
   *                   |
   *                   v
   *                f (PartitionedCallOp: ReadResourceVariable))
   *                   |
   *                   v
   *                y (_Retval, DT_FLOAT)
   */
  FunctionDef func = test::function::ReadResourceVariable();
  GraphDef graph_def = GDef(
      {
          NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"x"},
               {{"Tin", DataTypeSlice{DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_FLOAT}},
                {"f", FDH::FunctionRef("ReadResourceVariable", {})}}),
          NDef("y", "_Retval", {"f:0"}, {{"T", DT_FLOAT}}),
      },
      // FunctionLib
      {func});

  VerifyPlacerInspectionRequiredOps(graph_def,
                                    {{"x", false}, {"f", false}, {"y", false}});
}

}  // namespace tensorflow
