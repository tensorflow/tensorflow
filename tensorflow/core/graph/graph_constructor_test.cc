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

#include "tensorflow/core/graph/graph_constructor.h"

#include <vector>
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

// TODO(josh11b): Test InitCostModel().
// TODO(josh11b): Test setting the "device" field of a NodeDef.
// TODO(josh11b): Test that feeding won't prune targets.

namespace tensorflow {
namespace {

class GraphConstructorTest : public ::testing::Test {
 protected:
  GraphConstructorTest() : g_(new Graph(OpRegistry::Global())) {
  }
  ~GraphConstructorTest() override {}

  void Convert(const string& gdef_ascii) {
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &gdef_));
  }

  void ExpectError(const string& gdef_ascii, const string& expected_error_re) {
    Convert(gdef_ascii);
    GraphConstructorOptions opts;
    Status status = ConvertGraphDefToGraph(opts, gdef_, g_.get());
    EXPECT_FALSE(status.ok());
    EXPECT_TRUE(RE2::PartialMatch(status.error_message(), expected_error_re))
        << status;
  }

  void ExpectOK(const string& gdef_ascii) {
    Convert(gdef_ascii);
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, gdef_, g_.get()));
  }

  void ExpectVersions(int min_consumer, int producer) {
    EXPECT_NE(nullptr, g_);
    EXPECT_EQ(min_consumer, g_->versions().min_consumer())
        << "Expected min consumer " << min_consumer << ", got "
        << g_->versions().min_consumer();
    EXPECT_EQ(producer, g_->versions().producer()) << "Expected producer "
                                                   << producer << ", got "
                                                   << g_->versions().producer();
  }

  Node* FindNode(const string& name) {
    for (Node* n : g_->nodes()) {
      if (n->name() == name) return n;
    }
    return nullptr;
  }

  bool HasNode(const string& name) { return FindNode(name) != nullptr; }

  void ExpectNodes(const string& nodes) {
    int count = 0;
    std::vector<string> actual_nodes;
    for (Node* n : g_->nodes()) {
      if (n->IsOp()) {
        count++;
        actual_nodes.push_back(n->name());
      }
    }
    std::sort(actual_nodes.begin(), actual_nodes.end());

    LOG(INFO) << "Nodes present: " << str_util::Join(actual_nodes, " ");

    std::vector<string> expected_nodes = str_util::Split(nodes, ',');
    std::sort(expected_nodes.begin(), expected_nodes.end());
    for (const string& s : expected_nodes) {
      Node* n = FindNode(s);
      EXPECT_TRUE(n != nullptr) << s;
    }

    EXPECT_TRUE(actual_nodes.size() == expected_nodes.size())
        << "\nActual:   " << str_util::Join(actual_nodes, ",")
        << "\nExpected: " << str_util::Join(expected_nodes, ",");
  }

  bool HasEdge(const string& src, int src_out, const string& dst, int dst_in) {
    for (const Edge* e : g_->edges()) {
      if (e->src()->name() == src && e->src_output() == src_out &&
          e->dst()->name() == dst && e->dst_input() == src_out)
        return true;
    }
    return false;
  }
  bool HasControlEdge(const string& src, const string& dst) {
    return HasEdge(src, Graph::kControlSlot, dst, Graph::kControlSlot);
  }

 private:
  GraphDef gdef_;
  std::unique_ptr<Graph> g_;
};

REGISTER_OP("ABC");
REGISTER_OP("TestParams").Output("o: float");
REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_OP("TestMul").Input("a: float").Input("b: float").Output("o: float");
REGISTER_OP("TestInt").Input("a: int32");

TEST_F(GraphConstructorTest, InvalidNodeName) {
  auto expect_invalid_name = [this](const char* name) {
    ExpectError(strings::StrCat("node { name: '", name, "' op: 'ABC' }"),
                strings::StrCat("Node '", name,
                                "': Node name contains invalid characters"));
  };

  expect_invalid_name("a:b");
  expect_invalid_name("_abc");  // Can't start with '_'
  // Name is a\b, but proto text format escapes slashes so we use a\\b here.
  // This works for ExpectError too, since re2 also treats \\ as one slash.
  expect_invalid_name(R"(a\\b)");
  expect_invalid_name("/a");
  expect_invalid_name("-a");

  ExpectOK("node { name: 'a-bc_' op: 'ABC' }");
  ExpectOK("node { name: 'a-B.0/.c_' op: 'ABC' }");
  ExpectOK("node { name: '0123' op: 'ABC' }");
  ExpectOK("node { name: '.0123' op: 'ABC' }");
}

TEST_F(GraphConstructorTest, InvalidSourceNodeName) {
  ExpectError(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: 'W999' input: 'input' }",

      "Unknown input node.*W999");
}

TEST_F(GraphConstructorTest, InvalidSourceNodeIndex) {
  ExpectError(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1:1', 'input:1' ] }",

      "Connecting to invalid output 1 of source node W1");
}

TEST_F(GraphConstructorTest, GraphWithCycle) {
  ExpectError(
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'input:0', 't2' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'input:1', 't1' ] }",

      "cycle");
}

TEST_F(GraphConstructorTest, TypeMismatch) {
  ExpectError(
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 'int' op: 'TestInt' input: [ 'input' ] }",

      "Input 0 of node int was passed float from input:0 incompatible with "
      "expected int32.");
}

TEST_F(GraphConstructorTest, EmptyGraph) {
  ExpectOK("");
  ExpectVersions(0, 0);  // The default GraphDef versions are 0
}

TEST_F(GraphConstructorTest, VersionGraph) {
  ExpectOK(strings::StrCat("versions { producer: ", TF_GRAPH_DEF_VERSION,
                           " min_consumer: ", TF_GRAPH_DEF_VERSION_MIN_CONSUMER,
                           "}"));
  ExpectVersions(TF_GRAPH_DEF_VERSION_MIN_CONSUMER, TF_GRAPH_DEF_VERSION);
}

TEST_F(GraphConstructorTest, LowVersion) {
  ExpectError(strings::StrCat("versions { producer: ", -1, " }"),
              strings::StrCat(R"(^GraphDef producer version -1 below min )"
                              "producer ",
                              TF_GRAPH_DEF_VERSION_MIN_PRODUCER,
                              " supported by TensorFlow ", TF_VERSION_STRING,
                              R"(\.  Please regenerate your graph\.$)"));
}

TEST_F(GraphConstructorTest, HighVersion) {
  const int version = TF_GRAPH_DEF_VERSION + 1;
  ExpectError(strings::StrCat("versions { min_consumer: ", version, " }"),
              strings::StrCat(R"(^GraphDef min consumer version )", version,
                              " above current version ", TF_GRAPH_DEF_VERSION,
                              " for TensorFlow ", TF_VERSION_STRING,
                              R"(\.  Please upgrade TensorFlow\.$)"));
}

TEST_F(GraphConstructorTest, BadVersion) {
  const int version = TF_GRAPH_DEF_VERSION + 1;
  const int bad = TF_GRAPH_DEF_VERSION;
  ExpectError(
      strings::StrCat("versions { producer: ", version, " bad_consumers: ", bad,
                      " }"),
      strings::StrCat(
          R"(^GraphDef disallows consumer version )", bad,
          R"(\.  Please upgrade TensorFlow: this version is likely buggy\.$)"));
}

TEST_F(GraphConstructorTest, SimpleModel) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }");
  EXPECT_TRUE(HasNode("W1"));
  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasEdge("W1", 0, "t1", 0));
  EXPECT_TRUE(HasEdge("input", 1, "t1", 0));
}

TEST_F(GraphConstructorTest, SimpleModelWithControlEdges) {
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' input: [ '^W1' ] }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W1', 'input:1', '^t1' ] }");
  EXPECT_TRUE(HasNode("W1"));
  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasNode("t2"));
  EXPECT_TRUE(HasEdge("W1", 0, "t1", 0));
  EXPECT_TRUE(HasEdge("input", 1, "t1", 0));
  EXPECT_TRUE(HasEdge("W1", 0, "t2", 0));
  EXPECT_TRUE(HasEdge("input", 1, "t2", 0));
  EXPECT_TRUE(HasControlEdge("W1", "input"));
  EXPECT_TRUE(HasControlEdge("t1", "t2"));
}

TEST_F(GraphConstructorTest, Error_ControlEdgeBeforeRealInput) {
  ExpectError(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' input: [ '^W1' ] }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W1', '^t1', 'input:1' ] }",
      "Node 't2': Control dependencies must come after regular dependencies");
}

TEST_F(GraphConstructorTest, CopyGraph) {
  const int v = TF_GRAPH_DEF_VERSION;
  const int bad = v + 17;
  VersionDef versions;
  versions.set_producer(v - 1);
  versions.set_min_consumer(v - 2);
  versions.add_bad_consumers(bad);

  Graph src(OpRegistry::Global());
  src.set_versions(versions);

  Graph dst(OpRegistry::Global());
  CopyGraph(src, &dst);
  EXPECT_EQ(dst.versions().producer(), versions.producer());
  EXPECT_EQ(dst.versions().min_consumer(), versions.min_consumer());
  EXPECT_EQ(dst.versions().bad_consumers_size(), 1);
  EXPECT_EQ(dst.versions().bad_consumers(0), bad);
}

}  // namespace
}  // namespace tensorflow
