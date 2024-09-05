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

TEST(DebugDataDumper, NoPrefixTest) {
  EXPECT_EQ(false, DEBUG_DATA_DUMPER()->ShouldDump("DumpGraphToFileTest",
                                                   kDebugGroupMain));
}

TEST(DebugDataDumper, NoNameFilterTest) {
  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();

  EXPECT_EQ(false, DEBUG_DATA_DUMPER()->ShouldDump("DumpGraphToFileTest",
                                                   kDebugGroupMain));
}

TEST(DebugDataDumper, ShouldDumpTest) {
  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();
  EXPECT_EQ(true, DEBUG_DATA_DUMPER()->ShouldDump("DumpGraphToFileTest",
                                                  kDebugGroupMain));

  setenv("TF_DUMP_GRAPH_NAME_FILTER", "DumpGraph", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();
  EXPECT_EQ(true, DEBUG_DATA_DUMPER()->ShouldDump("DumpGraphToFileTest",
                                                  kDebugGroupMain));

  setenv("TF_DUMP_GRAPH_NAME_FILTER", "DoNotDumpGraph", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();
  EXPECT_EQ(false, DEBUG_DATA_DUMPER()->ShouldDump("DumpGraphToFileTest",
                                                   kDebugGroupMain));

  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();
  EXPECT_EQ(false,
            DEBUG_DATA_DUMPER()->ShouldDump("DumpGraphToFileTest",
                                            kDebugGroupBridgePhase1Clustering));

  setenv("TF_DUMP_GRAPH_GROUPS", "main,bridge_phase1_clustering", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();
  EXPECT_EQ(true,
            DEBUG_DATA_DUMPER()->ShouldDump("DumpGraphToFileTest",
                                            kDebugGroupBridgePhase1Clustering));

  DEBUG_DATA_DUMPER()->LoadEnvvars();
  EXPECT_EQ(false, DEBUG_DATA_DUMPER()->ShouldDump(
                       "__wrapped__DumpGraphToFileTest", kDebugGroupMain));

  setenv("TF_DUMP_GRAPH_WRAPPED", "true", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();
  EXPECT_EQ(true, DEBUG_DATA_DUMPER()->ShouldDump(
                      "__wrapped__DumpGraphToFileTest", kDebugGroupMain));
}

TEST(DebugDataDumper, DumpFileBasenameTest) {
  // For the same name, the order id should increment for each new dump file
  // name.
  EXPECT_EQ("DumpFileBasenameTest1.0000.main.tag1",
            DEBUG_DATA_DUMPER()->GetDumpFilename("DumpFileBasenameTest1",
                                                 kDebugGroupMain, "tag1"));
  EXPECT_EQ("DumpFileBasenameTest1.0001.main.tag2",
            DEBUG_DATA_DUMPER()->GetDumpFilename("DumpFileBasenameTest1",
                                                 kDebugGroupMain, "tag2"));

  // For other names, the order id should restart from 0.
  EXPECT_EQ("DumpFileBasenameTest2.0000.main.tag1",
            DEBUG_DATA_DUMPER()->GetDumpFilename("DumpFileBasenameTest2",
                                                 kDebugGroupMain, "tag1"));
}

TEST(DebugDataDumper, DumpGraphToFileTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();

  DEBUG_DATA_DUMPER()->DumpGraph("DumpGraphToFileTest", kDebugGroupMain, "tag",
                                 &graph, nullptr, false);

  std::string dumpFilename =
      io::JoinPath(dir, "DumpGraphToFileTest.0000.main.tag.pbtxt");
  EXPECT_EQ(absl::OkStatus(), Env::Default()->FileExists(dumpFilename));
}

TEST(DebugDataDumper, DumpGraphLongFileNameCrashTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();

  // Make sure long file name does not crash.
  std::string name = std::string(256, 'x');
  DEBUG_DATA_DUMPER()->DumpGraph(name, kDebugGroupMain, "tag", &graph, nullptr,
                                 false);

  std::string dumpFilename = io::JoinPath(
      dir, absl::StrFormat("%s.0000.main.tag.pbtxt", name.c_str()));
  EXPECT_EQ(absl::StatusCode::kNotFound,
            Env::Default()->FileExists(dumpFilename).code());
}

TEST(DebugDataDumper, DumpOpCreationStacktracesTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);
  setenv("TF_DUMP_OP_CREATION_STACKTRACES", "1", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();

  DEBUG_DATA_DUMPER()->DumpOpCreationStackTraces(
      "DumpOpCreationStacktracesTest", kDebugGroupMain, "test", &graph);

  std::string dumpFilename =
      io::JoinPath(dir, "DumpOpCreationStacktracesTest.0000.main.test.csv");
  EXPECT_EQ(absl::OkStatus(), Env::Default()->FileExists(dumpFilename));
}

TEST(DebugDataDumper, NoDumpOpCreationStacktracesTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);
  DEBUG_DATA_DUMPER()->LoadEnvvars();

  DEBUG_DATA_DUMPER()->DumpOpCreationStackTraces(
      "DumpOpCreationStacktracesTest", kDebugGroupMain, "test", &graph);

  std::string dumpFilename =
      io::JoinPath(dir, "DumpOpCreationStacktracesTest.0000.main.test.json");
  EXPECT_EQ(absl::StatusCode::kNotFound,
            Env::Default()->FileExists(dumpFilename).code());
}

}  // namespace
}  // namespace tensorflow
