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

#include "tensorflow/core/common_runtime/isolate_placer_inspection_required_ops_pass.h"

#include <map>
#include <unordered_map>

#include "absl/memory/memory.h"
#include "absl/strings/str_join.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/equal_graph_def.h"

namespace tensorflow {

using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using FDH = ::tensorflow::FunctionDefHelper;

// Returns void so that we can call TF_ASSERT_OK inside it.
static void RunPass(const GraphDef& original, GraphDef* rewritten,
                    FunctionLibraryDefinition* flib_def) {
  std::unique_ptr<Graph> graph = absl::make_unique<Graph>(OpRegistry::Global());
  GraphConstructorOptions opts;
  opts.add_default_attributes = false;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, original, graph.get()));
  GraphOptimizationPassOptions options;
  options.graph = &graph;
  options.flib_def = flib_def;
  IsolatePlacerInspectionRequiredOpsPass pass;
  TF_ASSERT_OK(pass.Run(options));
  graph->ToGraphDef(rewritten);
}
static void RunPass(const GraphDef& original, GraphDef* rewritten) {
  FunctionLibraryDefinition flib_def(OpRegistry::Global(), original.library());
  RunPass(original, rewritten, &flib_def);
}

void RunPassAndCompare(const GraphDef& original, const GraphDef& expected) {
  GraphDef rewritten;
  RunPass(original, &rewritten);
  TF_EXPECT_GRAPH_EQ(expected, rewritten);
}

void RunPassAndCompare(const GraphDef& original,
                       const std::vector<GraphDef>& expected_alternatives) {
  GraphDef rewritten;
  RunPass(original, &rewritten);

  std::vector<string> errors;
  errors.push_back(absl::StrCat("Graphs did not match.\n  Rewritten graph:\n",
                                SummarizeGraphDef(rewritten)));
  for (const GraphDef& alternative : expected_alternatives) {
    string diff;
    bool graphs_equal = EqualGraphDef(rewritten, alternative, &diff);
    if (graphs_equal) {
      return;
    }
    errors.push_back(absl::StrCat("  Expected alternative:\n",
                                  SummarizeGraphDef(alternative)));
  }
  EXPECT_TRUE(false) << absl::StrJoin(errors, "\n");
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, Basic) {
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
  GraphDef original = GDef(
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

  GraphDef expected = GDef(
      {
          NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("x_f", "Identity", {"x"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"x_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE}},
                {"f", FDH::FunctionRef("ResourceIdentity", {})}}),
          NDef("f_y", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("y", "_Retval", {"f_y:0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, expected);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, FunctionDefinitionNotInGraph) {
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
  GraphDef original = GDef({
      NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
      NDef("f", "PartitionedCall", {"x"},
           {{"Tin", DataTypeSlice{DT_RESOURCE}},
            {"Tout", DataTypeSlice{DT_RESOURCE}},
            {"f", FDH::FunctionRef("ResourceIdentity", {})}}),
      NDef("y", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
  });

  GraphDef expected = GDef({
      NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
      NDef("x_f", "Identity", {"x"}, {{"T", DT_RESOURCE}}),
      NDef("f", "PartitionedCall", {"x_f"},
           {{"Tin", DataTypeSlice{DT_RESOURCE}},
            {"Tout", DataTypeSlice{DT_RESOURCE}},
            {"f", FDH::FunctionRef("ResourceIdentity", {})}}),
      NDef("f_y", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
      NDef("y", "_Retval", {"f_y:0"}, {{"T", DT_RESOURCE}}),
  });

  FunctionLibraryDefinition flib_def(OpRegistry::Global(), {});
  TF_ASSERT_OK(flib_def.AddFunctionDef(func));
  GraphDef rewritten;
  RunPass(original, &rewritten, &flib_def);
  TF_EXPECT_GRAPH_EQ(expected, rewritten);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, MultipleInputsAndOutputs) {
  /*
   *                a (_Arg, DT_RESOURCE)
   *                   |   b (_Arg, DT_RESOURCE)
   *                   |      |
   *                   v      v
   *                f (PartitionedCallOp: Swap)
   *                   |      |
   *                   |      v
   *                   v    r2 (_Retval, DT_RESOURCE)
   *                r1 (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::Swap();
  GraphDef original = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a", "b"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("r1", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f:1"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  GraphDef expected = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b_f", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "b_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_r1", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r1", "_Retval", {"f_r1"}, {{"T", DT_RESOURCE}}),
          NDef("f_r2", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f_r2"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, expected);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, UnusedOutput) {
  /*
   *                a (_Arg, DT_RESOURCE)
   *                   |   b (_Arg, DT_RESOURCE)
   *                   |      |
   *                   v      v
   *                f (PartitionedCallOp: Swap)
   *                   |      |
   *                   |      v
   *                   v    <unused>
   *                r1 (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::Swap();
  GraphDef original = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a", "b"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("r1", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  GraphDef expected = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b_f", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "b_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_r1", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r1", "_Retval", {"f_r1"}, {{"T", DT_RESOURCE}}),
          // Identity is created for output that was not used.
          NDef("f_0", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, expected);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, OutputsConsumedBySameOp) {
  /*
   *                a (_Arg, DT_RESOURCE)
   *                   |   b (_Arg, DT_RESOURCE)
   *                   |      |
   *                   v      v
   *                f (PartitionedCallOp: Swap)
   *                   |     |
   *                   |     |
   *                   v     v
   *                add (Add, DT_RESOURCE)
   */
  FunctionDef func = test::function::Swap();
  GraphDef original = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a", "b"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("add", "Add", {"f:0", "f:1"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  // There are two possible namings for outputs depending on map
  // iteration order.
  GraphDef expected1 = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b_f", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "b_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_add", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("f_add_0", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("add", "Add", {"f_add", "f_add_0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  GraphDef expected2 = GDef(
      {
          // Same as above
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("b", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("b_f", "Identity", {"b"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "b_f"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          // Different from above
          NDef("f_add", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("f_add_0", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("add", "Add", {"f_add_0", "f_add"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, {expected1, expected2});
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, IdenticalInputs) {
  /*
   *                a (_Arg, DT_RESOURCE)
   *                   |      |
   *                   |      |
   *                   v      v
   *                f (PartitionedCallOp: Swap)
   *                   |      |
   *                   |      v
   *                   v    r2 (_Retval, DT_RESOURCE)
   *                r1 (_Retval, DT_RESOURCE)
   */
  FunctionDef func = test::function::Swap();
  GraphDef original = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a", "a"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("r1", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f:1"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  // There are two possible namings for outputs depending on map
  // iteration order.
  GraphDef expected1 = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("a_f_0", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall", {"a_f", "a_f_0"},
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_r1", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r1", "_Retval", {"f_r1"}, {{"T", DT_RESOURCE}}),
          NDef("f_r2", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f_r2"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  GraphDef expected2 = GDef(
      {
          NDef("a", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("a_f", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("a_f_0", "Identity", {"a"}, {{"T", DT_RESOURCE}}),
          NDef("f", "PartitionedCall",
               {"a_f_0", "a_f"},  // the only different line from above
               {{"Tin", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"Tout", DataTypeSlice{DT_RESOURCE, DT_RESOURCE}},
                {"f", FDH::FunctionRef("Swap", {{"T", DT_RESOURCE}})}}),
          NDef("f_r1", "Identity", {"f:0"}, {{"T", DT_RESOURCE}}),
          NDef("r1", "_Retval", {"f_r1"}, {{"T", DT_RESOURCE}}),
          NDef("f_r2", "Identity", {"f:1"}, {{"T", DT_RESOURCE}}),
          NDef("r2", "_Retval", {"f_r2"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, {expected1, expected2});
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest, DirectCallsAreNotIsolated) {
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
  GraphDef original = GDef(
      {
          NDef("x", "_Arg", {}, {{"T", DT_RESOURCE}}),
          NDef("f", "ResourceIdentity", {"x"}),
          NDef("y", "_Retval", {"f:0"}, {{"T", DT_RESOURCE}}),
      },
      // FunctionLib
      {func});

  RunPassAndCompare(original, original);
}

TEST(IsolatePlacerInspectionRequiredOpsPassTest,
     FunctionsNotReturningResourcesAreNotIsolated) {
  /*
   *                x (_Arg, DT_RESOURCE)
   *                   |
   *                   v
   *                f (PartitionedCallOp, ReadResourceVariable)
   *                   |
   *                   v
   *                y (_Retval, DT_FLOAT)
   */
  FunctionDef func = test::function::ReadResourceVariable();
  GraphDef original = GDef(
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

  RunPassAndCompare(original, original);
}

}  // namespace tensorflow
