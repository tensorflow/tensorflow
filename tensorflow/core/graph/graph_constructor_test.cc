#include "tensorflow/core/graph/graph_constructor.h"

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/regexp.h"
#include "tensorflow/core/public/status.h"
#include <gtest/gtest.h>

// TODO(josh11b): Test InitCostModel().
// TODO(josh11b): Test setting the "device" field of a NodeDef.
// TODO(josh11b): Test that feeding won't prune targets.

namespace tensorflow {
namespace {

class GraphConstructorTest : public ::testing::Test {
 protected:
  GraphConstructorTest() : g_(new Graph(OpRegistry::Global())) {
    RequireDefaultOps();
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
  ExpectError("node { name: 'a:b' op: 'ABC' }",
              "Node 'a:b': Node name contains invalid characters");
  ExpectError("node { name: '_abc' op: 'ABC' }",
              // Can't start with '_'
              "Node '_abc': Node name contains invalid characters");
  ExpectOK("node { name: 'a-bc_' op: 'ABC' }");
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

TEST_F(GraphConstructorTest, EmptyGraph) { ExpectOK(""); }

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

}  // namespace
}  // namespace tensorflow
