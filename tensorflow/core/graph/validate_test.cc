/* Copyright 2015 Google Inc. All Rights Reserved.

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
#include <vector>

#include "testing/base/public/gunit.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/graph/subgraph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/public/status.h"

namespace tensorflow {
namespace {

REGISTER_OP("FloatInput").Output("o: float");
REGISTER_OP("Int32Input").Output("o: int32");

TEST(ValidateGraphDefTest, TestValidGraph) {
  string graph_def_str =
      "node { name: 'A' op: 'FloatInput'}"
      "node { name: 'B' op: 'FloatInput'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  ASSERT_OK(graph::ValidateGraphDef(graph_def, OpRegistry::Global()));
}

TEST(ValidateGraphDefTest, GraphWithUnspecifiedDefaultAttr) {
  string graph_def_str =
      "node { name: 'A' op: 'FloatInput'}"
      "node { name: 'B' op: 'Int32Input'}"
      "node { "
      "       name: 'C' op: 'Sum' "
      "       attr { key: 'T' value { type: DT_FLOAT } }"
      "       input: ['A', 'B'] "
      "}";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  Status s = graph::ValidateGraphDef(graph_def, OpRegistry::Global());
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(StringPiece(s.ToString()).contains("NodeDef missing attr"));

  // Add the defaults.
  ASSERT_OK(AddDefaultAttrsToGraphDef(&graph_def, OpRegistry::Global(), 0));

  // Validation should succeed.
  ASSERT_OK(graph::ValidateGraphDef(graph_def, OpRegistry::Global()));
}

TEST(ValidateGraphDefTest, GraphWithUnspecifiedRequiredAttr) {
  // "DstT" attribute is missing.
  string graph_def_str =
      "node { name: 'A' op: 'FloatInput'}"
      "node { "
      "       name: 'B' op: 'Cast' "
      "       attr { key: 'SrcT' value { type: DT_FLOAT } }"
      "       input: ['A'] "
      "}";
  GraphDef graph_def;
  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(graph_def_str, &graph_def)) << graph_def_str;
  Status s = graph::ValidateGraphDef(graph_def, OpRegistry::Global());
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(StringPiece(s.ToString()).contains("NodeDef missing attr"));

  // Add the defaults.
  ASSERT_OK(AddDefaultAttrsToGraphDef(&graph_def, OpRegistry::Global(), 0));

  // Validation should still fail.
  s = graph::ValidateGraphDef(graph_def, OpRegistry::Global());
  EXPECT_FALSE(s.ok());
  EXPECT_TRUE(StringPiece(s.ToString()).contains("NodeDef missing attr"));
}

}  // namespace
}  // namespace tensorflow
