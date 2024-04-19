/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/validate.h"

#include <string>

#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

REGISTER_OP("FloatInput").Output("o: float");
REGISTER_OP("Int32Input").Output("o: int32");

TEST(ValidateGraphDefTest, TestValidGraph) {
  const string graph_def_str =
      "node { name: 'A' op: 'FloatInput' }"
      "node { name: 'B' op: 'FloatInput' }"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  TF_ASSERT_OK(graph::ValidateGraphDef(graph_def, *OpRegistry::Global()));
}

TEST(ValidateGraphDefTest, GraphWithUnspecifiedDefaultAttr) {
  const string graph_def_str =
      "node { name: 'A' op: 'FloatInput' }"
      "node { name: 'B' op: 'Int32Input' }"
      "node { "
      "       name: 'C' op: 'Sum' "
      "       attr { key: 'T' value { type: DT_FLOAT } }"
      "       input: ['A', 'B'] "
      "}";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  Status s = graph::ValidateGraphDef(graph_def, *OpRegistry::Global());
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "NodeDef missing attr"));

  // Add the defaults.
  TF_ASSERT_OK(AddDefaultAttrsToGraphDef(&graph_def, *OpRegistry::Global(), 0));

  // Validation should succeed.
  TF_ASSERT_OK(graph::ValidateGraphDef(graph_def, *OpRegistry::Global()));
}

TEST(ValidateGraphDefTest, GraphWithUnspecifiedRequiredAttr) {
  // "DstT" attribute is missing.
  const string graph_def_str =
      "node { name: 'A' op: 'FloatInput' }"
      "node { "
      "       name: 'B' op: 'Cast' "
      "       attr { key: 'SrcT' value { type: DT_FLOAT } }"
      "       input: ['A'] "
      "}";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  Status s = graph::ValidateGraphDef(graph_def, *OpRegistry::Global());
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "NodeDef missing attr"));

  // Add the defaults.
  TF_ASSERT_OK(AddDefaultAttrsToGraphDef(&graph_def, *OpRegistry::Global(), 0));

  // Validation should still fail.
  s = graph::ValidateGraphDef(graph_def, *OpRegistry::Global());
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(absl::StrContains(s.ToString(), "NodeDef missing attr"));
}

TEST(ValidateGraphDefAgainstOpListTest, GraphWithOpOnlyInOpList) {
  OpRegistrationData op_reg_data;
  TF_ASSERT_OK(OpDefBuilder("UniqueSnowflake").Finalize(&op_reg_data));
  OpList op_list;
  *op_list.add_op() = op_reg_data.op_def;
  const string graph_def_str = "node { name: 'A' op: 'UniqueSnowflake' }";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  TF_ASSERT_OK(graph::ValidateGraphDefAgainstOpList(graph_def, op_list));
}

TEST(ValidateGraphDefAgainstOpListTest, GraphWithGlobalOpNotInOpList) {
  OpRegistrationData op_reg_data;
  TF_ASSERT_OK(OpDefBuilder("NotAnywhere").Finalize(&op_reg_data));
  OpList op_list;
  *op_list.add_op() = op_reg_data.op_def;
  const string graph_def_str = "node { name: 'A' op: 'FloatInput' }";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  ASSERT_FALSE(graph::ValidateGraphDefAgainstOpList(graph_def, op_list).ok());
}

REGISTER_OP("HasDocs").Doc("This is in the summary.");

TEST(GetOpListForValidationTest, ShouldStripDocs) {
  bool found_float = false;
  bool found_int32 = false;
  bool found_has_docs = false;
  OpList op_list;
  graph::GetOpListForValidation(&op_list);
  for (const OpDef& op_def : op_list.op()) {
    if (op_def.name() == "FloatInput") {
      EXPECT_FALSE(found_float);
      found_float = true;
    }
    if (op_def.name() == "Int32Input") {
      EXPECT_FALSE(found_int32);
      found_int32 = true;
    }
    if (op_def.name() == "HasDocs") {
      EXPECT_FALSE(found_has_docs);
      found_has_docs = true;
      EXPECT_TRUE(op_def.summary().empty());
    }
  }
  EXPECT_TRUE(found_float);
  EXPECT_TRUE(found_int32);
  EXPECT_TRUE(found_has_docs);
}

TEST(VerifyNoDuplicateNodeNames, NoDuplicateNodeNames) {
  const string graph_def_str =
      "node { name: 'A' op: 'FloatInput' }"
      "node { name: 'B' op: 'Int32Input' }"
      "node { "
      "       name: 'C' op: 'Sum' "
      "       attr { key: 'T' value { type: DT_FLOAT } }"
      "       input: ['A', 'B'] "
      "}";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  TF_ASSERT_OK(graph::VerifyNoDuplicateNodeNames(graph_def));
}

TEST(VerifyNoDuplicateNodeNames, DuplicateNodeNames) {
  const string graph_def_str =
      "node { name: 'A' op: 'FloatInput' }"
      "node { name: 'A' op: 'Int32Input' }"
      "node { "
      "       name: 'C' op: 'Sum' "
      "       attr { key: 'T' value { type: DT_FLOAT } }"
      "       input: ['A', 'A'] "
      "}";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  EXPECT_EQ(graph::VerifyNoDuplicateNodeNames(graph_def).code(),
            tensorflow::error::ALREADY_EXISTS);
}

TEST(ValidateGraphHasNoCycleTest, NoCyclePasses) {
  const string graph_def_str =
      "node { name: 'A' op: 'FloatInput' }"
      "node { name: 'B' op: 'FloatInput' }"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;

  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));

  TF_EXPECT_OK(graph::ValidateGraphHasNoCycle(graph));
}

TEST(ValidateGraphHasNoCycleTest, NoCycleWithMergePasses) {
  const string graph_def_str =
      R"EOF(
      node { name: 'A' op: 'FloatInput' }
      node { name: 'merge' op: 'Merge' input: [ 'A:0', 'next:0' ]
             attr { key: "N" value: { i: 2 } }
             attr { key: "T" value: { type: DT_FLOAT } } }
      node { name: 'B' op: 'Mul'
             attr { key: 'T' value { type: DT_FLOAT } }
             input: [ 'merge:0', 'merge:0' ] }
      node { name: 'next' op: 'NextIteration' input: ['B:0']
             attr { key: "T" value: { type: DT_FLOAT } } }
      )EOF";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;

  Graph graph(OpRegistry::Global());
  GraphConstructorOptions opts;
  TF_ASSERT_OK(ConvertGraphDefToGraph(opts, graph_def, &graph));

  TF_EXPECT_OK(graph::ValidateGraphHasNoCycle(graph));
}

Node* AddNodeFromNodeDef(Graph& graph, const string& name,
                         const string& node_type, int num_inputs) {
  auto builder = NodeDefBuilder(name, node_type);
  for (int i = 0; i < num_inputs; ++i) {
    builder = builder.Input(strings::StrCat("node_", i), i, DT_FLOAT);
  }

  NodeDef node_def;
  TF_CHECK_OK(builder.Finalize(&node_def));

  Status s;
  Node* node = graph.AddNode(node_def, &s);
  TF_CHECK_OK(s);
  return node;
}

TEST(ValidateGraphHasNoCycleTest, CycleFails) {
  // Need to construct graph explicitly, since GraphDefToGraph has its own
  // cycle validation routine.
  Graph graph(OpRegistry::Global());

  Node* a = AddNodeFromNodeDef(graph, "A", "FloatInput", 0);
  Node* c = AddNodeFromNodeDef(graph, "B", "Mul", 2);
  graph.AddEdge(a, 0, c, 0);
  graph.AddEdge(c, 0, c, 1);  // Loop from C->C.

  EXPECT_THAT(
      graph::ValidateGraphHasNoCycle(graph),
      tsl::testing::StatusIs(
          tsl::error::Code::INVALID_ARGUMENT,
          ::testing::ContainsRegex("Graph is invalid, contains a cycle")));
}

}  // namespace
}  // namespace tensorflow
