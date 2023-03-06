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

#include "tensorflow/core/util/debug_data_dumper.h"

#include <string>

#include "absl/strings/str_format.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/io/path.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

TEST(DebugDataDumper, DumpGraphToFileTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);

  DebugDataDumper::Global()->DumpGraph("DumpGraphToFileTest", &graph, "tag");

  std::string dumpFilename =
      io::JoinPath(dir, "DumpGraphToFileTest.0.tag.pbtxt");
  EXPECT_EQ(OkStatus(), Env::Default()->FileExists(dumpFilename));
}

TEST(DebugDataDumper, NoNameFilterTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);

  DebugDataDumper::Global()->DumpGraph("NoNameFilterTest", &graph, "tag");

  std::string dumpFilename = io::JoinPath(dir, "NoNameFilterTest.0.tag1.pbtxt");
  EXPECT_EQ(absl::StatusCode::kNotFound,
            Env::Default()->FileExists(dumpFilename).code());
}

TEST(DebugDataDumper, DumpOrderIdTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);

  // Dump two graphs with the same name.
  DebugDataDumper::Global()->DumpGraph("DumpOrderIdTest", &graph, "tag1");
  DebugDataDumper::Global()->DumpGraph("DumpOrderIdTest", &graph, "tag2");

  // We should have two files with order 0 and 1.
  std::string dumpFilename1 = io::JoinPath(dir, "DumpOrderIdTest.0.tag1.pbtxt");
  EXPECT_EQ(OkStatus(), Env::Default()->FileExists(dumpFilename1));

  std::string dumpFilename2 = io::JoinPath(dir, "DumpOrderIdTest.1.tag2.pbtxt");
  EXPECT_EQ(OkStatus(), Env::Default()->FileExists(dumpFilename2));
}

TEST(DebugDataDumper, NameFilterTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "NameFilterTest1", 1);

  // Dump two graphs with the same name.
  DebugDataDumper::Global()->DumpGraph("NameFilterTest1", &graph, "tag");
  DebugDataDumper::Global()->DumpGraph("NameFilterTest2", &graph, "tag");

  // We should only have the dump for the first graph.
  std::string dumpFilename1 = io::JoinPath(dir, "NameFilterTest1.0.tag.pbtxt");
  EXPECT_EQ(OkStatus(), Env::Default()->FileExists(dumpFilename1));

  std::string dumpFilename2 = io::JoinPath(dir, "NameFilterTest2.0.tag.pbtxt");
  EXPECT_EQ(absl::StatusCode::kNotFound,
            Env::Default()->FileExists(dumpFilename2).code());
}

TEST(DebugDataDumper, LongFileNameCrashTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);

  // Make sure long file name does not crash.
  std::string name = std::string(256, 'x');
  DebugDataDumper::Global()->DumpGraph(name, &graph, "tag");

  std::string dumpFilename =
      io::JoinPath(dir, absl::StrFormat("%s.0.tag.pbtxt", name.c_str()));
  EXPECT_EQ(absl::StatusCode::kNotFound,
            Env::Default()->FileExists(dumpFilename).code());
}

}  // namespace
}  // namespace tensorflow
