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
  EXPECT_EQ(false,
            DebugDataDumper::Global()->ShouldDump("DumpGraphToFileTest"));
}

TEST(DebugDataDumper, NoNameFilterTest) {
  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  EXPECT_EQ(false,
            DebugDataDumper::Global()->ShouldDump("DumpGraphToFileTest"));
}

TEST(DebugDataDumper, ShouldDumpTest) {
  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);

  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);
  EXPECT_EQ(true, DebugDataDumper::Global()->ShouldDump("DumpGraphToFileTest"));

  setenv("TF_DUMP_GRAPH_NAME_FILTER", "DumpGraph", 1);
  EXPECT_EQ(true, DebugDataDumper::Global()->ShouldDump("DumpGraphToFileTest"));

  setenv("TF_DUMP_GRAPH_NAME_FILTER", "DoNotDumpGraph", 1);
  EXPECT_EQ(false,
            DebugDataDumper::Global()->ShouldDump("DumpGraphToFileTest"));
}

TEST(DebugDataDumper, DumpFileBasenameTest) {
  // For the same name, the order id should increment for each new dump file
  // name.
  EXPECT_EQ("DumpFileBasenameTest1.0000.tag1",
            DebugDataDumper::Global()->GetDumpFileBasename(
                "DumpFileBasenameTest1", "tag1"));
  EXPECT_EQ("DumpFileBasenameTest1.0001.tag2",
            DebugDataDumper::Global()->GetDumpFileBasename(
                "DumpFileBasenameTest1", "tag2"));

  // For other names, the order id should restart from 0.
  EXPECT_EQ("DumpFileBasenameTest2.0000.tag1",
            DebugDataDumper::Global()->GetDumpFileBasename(
                "DumpFileBasenameTest2", "tag1"));
}

TEST(DebugDataDumper, DumpGraphToFileTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);

  DUMP_GRAPH("DumpGraphToFileTest", "tag", &graph, nullptr, false);

  std::string dumpFilename =
      io::JoinPath(dir, "DumpGraphToFileTest.0000.tag.pbtxt");
  EXPECT_EQ(OkStatus(), Env::Default()->FileExists(dumpFilename));
}

TEST(DebugDataDumper, DumpGraphLongFileNameCrashTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);

  // Make sure long file name does not crash.
  std::string name = std::string(256, 'x');
  DUMP_GRAPH(name, "tag", &graph, nullptr, false);

  std::string dumpFilename =
      io::JoinPath(dir, absl::StrFormat("%s.0000.tag.pbtxt", name.c_str()));
  EXPECT_EQ(absl::StatusCode::kNotFound,
            Env::Default()->FileExists(dumpFilename).code());
}

TEST(DebugDataDumper, DumpMLIRModuleTest) {
  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);

  DUMP_MLIR_MODULE("DumpMLIRModuleTest", "test", "fake_mlir_txt", false);

  std::string dumpFilepath =
      io::JoinPath(dir, "DumpMLIRModuleTest.0000.test.mlir");
  EXPECT_EQ(OkStatus(), Env::Default()->FileExists(dumpFilepath));
}

TEST(DebugDataDumper, DumpMLIRModuleLongFileNameCrashTest) {
  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);

  // Make sure long file name does not crash.
  std::string name = std::string(256, 'x');
  DUMP_MLIR_MODULE(name, "tag", "fake_mlir_txt", false);

  std::string dumpFilename =
      io::JoinPath(dir, absl::StrFormat("%s.0000.tag.pbtxt", name.c_str()));
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

  DUMP_OP_CREATION_STACKTRACES("DumpOpCreationStacktracesTest", "test", &graph);

  std::string dumpFilename =
      io::JoinPath(dir, "DumpOpCreationStacktracesTest.0000.test.csv");
  EXPECT_EQ(OkStatus(), Env::Default()->FileExists(dumpFilename));
}

TEST(DebugDataDumper, NoDumpOpCreationStacktracesTest) {
  Graph graph(OpRegistry::Global());
  Node* node;
  TF_CHECK_OK(NodeBuilder("A", "NoOp").Finalize(&graph, &node));

  std::string dir = testing::TmpDir();
  setenv("TF_DUMP_GRAPH_PREFIX", dir.c_str(), 1);
  setenv("TF_DUMP_GRAPH_NAME_FILTER", "*", 1);

  DUMP_OP_CREATION_STACKTRACES("DumpOpCreationStacktracesTest", "test", &graph);

  std::string dumpFilename =
      io::JoinPath(dir, "DumpOpCreationStacktracesTest.0000.test.json");
  EXPECT_EQ(absl::StatusCode::kNotFound,
            Env::Default()->FileExists(dumpFilename).code());
}

}  // namespace
}  // namespace tensorflow
