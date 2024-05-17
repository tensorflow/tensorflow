/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/util/dump_graph.h"

#include "tensorflow/core/framework/function.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/path.h"
#include "tensorflow/core/platform/str_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/status.h"

namespace tensorflow {
namespace {

TEST(DumpGraph, DumpGraphToFileSuccess) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  string ret = DumpGraphToFile("graph", graph);
  EXPECT_EQ(ret, io::JoinPath(testing::TmpDir(), "graph.pbtxt"));
  ret = DumpGraphToFile("graph", graph);
  EXPECT_EQ(ret, io::JoinPath(testing::TmpDir(), "graph_1.pbtxt"));

  GraphDef gdef;
  TF_ASSERT_OK(ReadTextProto(
      Env::Default(), io::JoinPath(testing::TmpDir(), "graph.pbtxt"), &gdef));
  string read, written;
  gdef.AppendToString(&read);
  graph.ToGraphDefDebug().AppendToString(&written);
  EXPECT_EQ(read, written);
}

TEST(DumpGraph, DumpGraphToFileNoEnvPrefix) {
  Graph graph(OpRegistry::Global());
  unsetenv("TF_DUMP_GRAPH_PREFIX");
  string ret = DumpGraphToFile("graph", graph);
  EXPECT_TRUE(str_util::StrContains(ret, "TF_DUMP_GRAPH_PREFIX not specified"));
}

TEST(DumpGraph, DumpFunctionDefToFileSuccess) {
  FunctionDef fdef;
  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  string ret = DumpFunctionDefToFile("function", fdef);
  EXPECT_EQ(ret, io::JoinPath(testing::TmpDir(), "function.pbtxt"));
}

TEST(DumpGraph, DumpProtoToFileSuccess) {
  NodeDef ndef_in;
  ndef_in.set_name("foo");

  setenv("TF_DUMP_GRAPH_PREFIX", testing::TmpDir().c_str(), 1);
  string expected_filepath = io::JoinPath(testing::TmpDir(), "node_def.pbtxt");
  string actual_filepath = DumpProtoToFile("node_def", ndef_in);
  EXPECT_EQ(expected_filepath, actual_filepath);

  NodeDef ndef_out;
  TF_ASSERT_OK(ReadTextProto(Env::Default(), expected_filepath, &ndef_out));
  EXPECT_EQ(ndef_in.DebugString(), ndef_out.DebugString());
}

}  // namespace
}  // namespace tensorflow
