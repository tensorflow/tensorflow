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

#include "tensorflow/core/graph/graph_constructor.h"

#include <vector>
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/version.h"

// TODO(josh11b): Test InitCostModel().
// TODO(josh11b): Test setting the "device" field of a NodeDef.
// TODO(josh11b): Test that feeding won't prune targets.

namespace tensorflow {
namespace {

class GraphConstructorTest : public ::testing::Test {
 protected:
  GraphConstructorTest() : graph_(OpRegistry::Global()) {}

  void Convert(const string& gdef_ascii) {
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &gdef_));
  }

  void ExpectError(const string& gdef_ascii,
                   const std::vector<string>& expected_error_strs) {
    Convert(gdef_ascii);
    GraphConstructorOptions opts;
    Status status = ConvertGraphDefToGraph(opts, gdef_, &graph_);
    EXPECT_FALSE(status.ok());

    for (const string& error : expected_error_strs) {
      EXPECT_TRUE(status.error_message().find(error) != string::npos)
          << "Expected to find '" << error << "' in " << status;
    }
  }

  void ExpectOK(const string& gdef_ascii) {
    Convert(gdef_ascii);
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, gdef_, &graph_));
  }

  void ExpectVersions(int min_consumer, int producer) {
    EXPECT_EQ(min_consumer, graph_.versions().min_consumer())
        << "Expected min consumer " << min_consumer << ", got "
        << graph_.versions().min_consumer();
    EXPECT_EQ(producer, graph_.versions().producer())
        << "Expected producer " << producer << ", got "
        << graph_.versions().producer();
  }

  Node* FindNode(const string& name) {
    for (Node* n : graph_.nodes()) {
      if (n->name() == name) return n;
    }
    return nullptr;
  }

  bool HasNode(const string& name) { return FindNode(name) != nullptr; }

  bool HasEdge(const string& src, int src_out, const string& dst, int dst_in) {
    for (const Edge* e : graph_.edges()) {
      if (e->src()->name() == src && e->src_output() == src_out &&
          e->dst()->name() == dst && e->dst_input() == dst_in)
        return true;
    }
    return false;
  }

  bool HasControlEdge(const string& src, const string& dst) {
    return HasEdge(src, Graph::kControlSlot, dst, Graph::kControlSlot);
  }

  string ColocationGroup(const string& node) {
    Node* n = nullptr;
    for (Node* ni : graph_.nodes()) {
      if (ni->name() == node) {
        n = ni;
        break;
      }
    }
    if (n == nullptr) {
      return "";
    }
    std::vector<string> value;
    Status s = GetNodeAttr(n->def(), kColocationAttrName, &value);
    if (!s.ok()) {
      return "";
    }
    if (value.size() != 1) {
      ADD_FAILURE()
          << "ColocationGroup was written with the assumption of at most 1 "
             "value for the _class attribute. Update it and its callers";
      return "";
    }
    StringPiece loc(value[0]);
    return loc.Consume(kColocationGroupPrefix) ? loc.ToString() : "";
  }

  string GraphDebugString() const {
    GraphDef def;
    graph_.ToGraphDef(&def);
    return def.DebugString();
  }

  Graph graph_;

 private:
  GraphDef gdef_;
};

Status Scalars(shape_inference::InferenceContext* c) {
  for (int i = 0; i < c->num_outputs(); ++i) {
    c->set_output(i, c->Scalar());
  }
  return Status::OK();
}

REGISTER_OP("ABC");
REGISTER_OP("TestParams").Output("o: float").SetShapeFn(Scalars);
REGISTER_OP("TestInput").Output("a: float").Output("b: float");
REGISTER_OP("TestMul")
    .Input("a: float")
    .Input("b: float")
    .Output("o: float")
    .SetShapeFn(Scalars);
REGISTER_OP("TestInt").Input("a: int32");
REGISTER_OP("TestOneInputTwoOutputs")
    .Input("x: float")
    .Output("y: float")
    .Output("z: float")
    .SetShapeFn(Scalars);
REGISTER_OP("TestOneInputOneOutput")
    .Input("x: T")
    .Output("y: T")
    .Attr("T: {float, int64}")
    .SetShapeFn(shape_inference::UnchangedShape);
REGISTER_OP("TestDefaultAttr")
    .Attr("default_int: int=31415")
    .SetShapeFn(shape_inference::NoOutputs);

TEST_F(GraphConstructorTest, InvalidNodeName) {
  auto expect_invalid_name = [this](const char* name) {
    ExpectError(strings::StrCat("node { name: '", name, "' op: 'ABC' }"),
                {"Node name contains invalid characters"});
  };

  expect_invalid_name("a:b");
  expect_invalid_name("_abc");  // Can't start with '_'
  // Name is a\b, but proto text format escapes slashes so we use a\\b here.
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

      {"Unknown input node", "W999"});
}

TEST_F(GraphConstructorTest, InvalidSourceNodeIndex) {
  ExpectError(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1:1', 'input:1' ] }",

      {"Connecting to invalid output 1 of source node W1"});
}

TEST_F(GraphConstructorTest, GraphWithCycle) {
  ExpectError(
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: [ 'input:0', 't2' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'input:1', 't1' ] }",

      {"cycle"});
}

TEST_F(GraphConstructorTest, GraphWithOKCycle) {
  // Test graph produced in python using:
  /*
     with tf.Graph().as_default():
       i = tf.constant(0)
       c = lambda i: tf.less(i, 10)
       b = lambda i: tf.add(i, 1)
       r = tf.while_loop(c, b, [i])
       with open('/tmp/graph.txt', 'w') as f:
         f.write(str(tf.get_default_graph().as_graph_def()))
  */
  ExpectOK(R"EOF(
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "while/Enter"
  op: "Enter"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "frame_name"
    value {
      s: "while/while/"
    }
  }
  attr {
    key: "is_constant"
    value {
      b: false
    }
  }
  attr {
    key: "parallel_iterations"
    value {
      i: 10
    }
  }
}
node {
  name: "while/Merge"
  op: "Merge"
  input: "while/Enter"
  input: "while/NextIteration"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Less/y"
  op: "Const"
  input: "^while/Merge"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "while/Less"
  op: "Less"
  input: "while/Merge"
  input: "while/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/LoopCond"
  op: "LoopCond"
  input: "while/Less"
}
node {
  name: "while/Switch"
  op: "Switch"
  input: "while/Merge"
  input: "while/LoopCond"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@while/Merge"
      }
    }
  }
}
node {
  name: "while/Identity"
  op: "Identity"
  input: "while/Switch:1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Add/y"
  op: "Const"
  input: "^while/Identity"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "while/Add"
  op: "Add"
  input: "while/Identity"
  input: "while/Add/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/NextIteration"
  op: "NextIteration"
  input: "while/Add"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Exit"
  op: "Exit"
  input: "while/Switch"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
versions {
  producer: 11
}
  )EOF");
}

TEST_F(GraphConstructorTest, TypeMismatch) {
  ExpectError(
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 'int' op: 'TestInt' input: [ 'input' ] }",

      {"Input 0 of node int was passed float from input:0 incompatible with "
       "expected int32."});
}

TEST_F(GraphConstructorTest, EmptyGraph) {
  ExpectOK("");
  ExpectVersions(0, 0);
}

TEST_F(GraphConstructorTest, VersionGraph) {
  ExpectOK(strings::StrCat("versions { producer: ", TF_GRAPH_DEF_VERSION,
                           " min_consumer: ", TF_GRAPH_DEF_VERSION_MIN_CONSUMER,
                           "}"));
  ExpectVersions(TF_GRAPH_DEF_VERSION_MIN_CONSUMER, TF_GRAPH_DEF_VERSION);
}

TEST_F(GraphConstructorTest, LowVersion) {
  ExpectError(strings::StrCat("versions { producer: ", -1, " }"),
              {strings::StrCat("GraphDef producer version -1 below min "
                               "producer ",
                               TF_GRAPH_DEF_VERSION_MIN_PRODUCER,
                               " supported by TensorFlow ", TF_VERSION_STRING,
                               ".  Please regenerate your graph.")});
}

TEST_F(GraphConstructorTest, HighVersion) {
  const int version = TF_GRAPH_DEF_VERSION + 1;
  ExpectError(strings::StrCat("versions { min_consumer: ", version, " }"),
              {strings::StrCat("GraphDef min consumer version ", version,
                               " above current version ", TF_GRAPH_DEF_VERSION,
                               " for TensorFlow ", TF_VERSION_STRING,
                               ".  Please upgrade TensorFlow.")});
}

TEST_F(GraphConstructorTest, BadVersion) {
  const int version = TF_GRAPH_DEF_VERSION + 1;
  const int bad = TF_GRAPH_DEF_VERSION;
  ExpectError(
      strings::StrCat("versions { producer: ", version, " bad_consumers: ", bad,
                      " }"),
      {strings::StrCat(
          "GraphDef disallows consumer version ", bad,
          ".  Please upgrade TensorFlow: this version is likely buggy.")});
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
  EXPECT_TRUE(HasEdge("input", 1, "t1", 1));
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
  EXPECT_TRUE(HasEdge("input", 1, "t1", 1));
  EXPECT_TRUE(HasEdge("W1", 0, "t2", 0));
  EXPECT_TRUE(HasEdge("input", 1, "t2", 1));
  EXPECT_TRUE(HasControlEdge("W1", "input"));
  EXPECT_TRUE(HasControlEdge("t1", "t2"));
}

TEST_F(GraphConstructorTest, Error_ControlEdgeBeforeRealInput) {
  ExpectError(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' input: [ '^W1' ] }"
      "node { name: 't1' op: 'TestMul' input: [ 'W1', 'input:1' ] }"
      "node { name: 't2' op: 'TestMul' input: [ 'W1', '^t1', 'input:1' ] }",
      {"Node 't2': Control dependencies must come after regular dependencies"});
}

TEST_F(GraphConstructorTest, ImportGraphDef) {
  GraphDef def;
  ImportGraphDefOptions opts;
  const string& source = graph_.FindNodeId(Graph::kSourceId)->name();
  const string& sink = graph_.FindNodeId(Graph::kSinkId)->name();

  // Importing an empty graph is fine.
  Status s = ImportGraphDef(opts, def, &graph_, nullptr);
  ASSERT_EQ(Status::OK(), s) << s;
  EXPECT_EQ(2, graph_.num_nodes());
  EXPECT_TRUE(HasControlEdge(source, sink));
  EXPECT_EQ(1, graph_.num_edges());

  bool parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
        node { name: "A" op: "TestParams" }
        node { name: "X" op: "TestParams" }
        node {
          name: "B"
          op: "TestOneInputTwoOutputs"
          input: "A"
          attr {
            key: "_class"
            value { list { s: "loc:@A" } }
          }
        }
        node {
          name: "C"
          op: "TestOneInputTwoOutputs"
          input: "B:1"
          input: "^X"
        }
        node {
          name: "D"
          op: "TestMul"
          input: "B:0"
          input: "C:0"
        })EOF",
      &def);
  ASSERT_TRUE(parsed);

  // First import should work out fine.
  s = ImportGraphDef(opts, def, &graph_, nullptr);
  ASSERT_EQ(Status::OK(), s) << s;
  EXPECT_EQ(5 + 2, graph_.num_nodes());  // Added nodes + source and sink
  EXPECT_EQ("A", ColocationGroup("B"));
  EXPECT_TRUE(HasEdge("A", 0, "B", 0));
  EXPECT_TRUE(HasEdge("B", 1, "C", 0));
  EXPECT_TRUE(HasEdge("B", 0, "D", 0));
  EXPECT_TRUE(HasEdge("C", 0, "D", 1));
  EXPECT_TRUE(HasControlEdge("X", "C"));
  EXPECT_TRUE(HasControlEdge(source, sink));
  EXPECT_TRUE(HasControlEdge(source, "A"));
  EXPECT_TRUE(HasControlEdge(source, "X"));
  EXPECT_TRUE(HasControlEdge("D", sink));
  EXPECT_EQ(9, graph_.num_edges());

  // Importing again should fail because of node name collissions.
  s = ImportGraphDef(opts, def, &graph_, nullptr);
  EXPECT_TRUE(errors::IsInvalidArgument(s)) << s;

  // But succeed if a unique prefix is provided.
  opts.prefix = "import";
  s = ImportGraphDef(opts, def, &graph_, nullptr);
  ASSERT_EQ(Status::OK(), s) << s;
  EXPECT_EQ(
      10 + 2,
      graph_.num_nodes());  // Added nodes + original nodes + source and sink
  EXPECT_EQ("A", ColocationGroup("B"));
  EXPECT_EQ("import/A", ColocationGroup("import/B"));
  EXPECT_TRUE(HasEdge("A", 0, "B", 0));
  EXPECT_TRUE(HasEdge("B", 1, "C", 0));
  EXPECT_TRUE(HasEdge("B", 0, "D", 0));
  EXPECT_TRUE(HasEdge("C", 0, "D", 1));
  EXPECT_TRUE(HasControlEdge("X", "C"));
  EXPECT_TRUE(HasEdge("import/A", 0, "import/B", 0));
  EXPECT_TRUE(HasEdge("import/B", 1, "import/C", 0));
  EXPECT_TRUE(HasEdge("import/B", 0, "import/D", 0));
  EXPECT_TRUE(HasEdge("import/C", 0, "import/D", 1));
  EXPECT_TRUE(HasControlEdge("import/X", "import/C"));
  EXPECT_TRUE(HasControlEdge(source, sink));
  EXPECT_TRUE(HasControlEdge(source, "A"));
  EXPECT_TRUE(HasControlEdge(source, "X"));
  EXPECT_TRUE(HasControlEdge("D", sink));
  EXPECT_TRUE(HasControlEdge(source, "import/A"));
  EXPECT_TRUE(HasControlEdge(source, "import/X"));
  EXPECT_TRUE(HasControlEdge("import/D", sink));
  EXPECT_EQ(17, graph_.num_edges());
}

TEST_F(GraphConstructorTest, ImportGraphDef_DefaultAttrs) {
  GraphDef def;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "node{ name:'A' op:'TestDefaultAttr'}", &def));
  Status s = ImportGraphDef(ImportGraphDefOptions(), def, &graph_, nullptr);
  ASSERT_EQ(Status::OK(), s) << s;
  Node* a = nullptr;
  for (Node* n : graph_.nodes()) {
    if (n->name() == "A") {
      a = n;
      break;
    }
  }
  ASSERT_TRUE(a != nullptr);
  int value = 0;
  s = GetNodeAttr(a->def(), "default_int", &value);
  ASSERT_EQ(Status::OK(), s) << s << " -- " << a->def().DebugString();
  EXPECT_EQ(31415, value);
}

TEST_F(GraphConstructorTest, ImportGraphDef_Versioning) {
  GraphDef def;
  const ImportGraphDefOptions opts;

  def.mutable_versions()->set_producer(TF_GRAPH_DEF_VERSION_MIN_PRODUCER - 1);
  Status s = ImportGraphDef(opts, def, &graph_, nullptr);
  EXPECT_TRUE(errors::IsInvalidArgument(s)) << s;

  def.mutable_versions()->Clear();
  def.mutable_versions()->set_min_consumer(TF_GRAPH_DEF_VERSION + 1);
  s = ImportGraphDef(opts, def, &graph_, nullptr);
  EXPECT_TRUE(errors::IsInvalidArgument(s)) << s;

  def.mutable_versions()->Clear();
  def.mutable_versions()->add_bad_consumers(TF_GRAPH_DEF_VERSION);
  s = ImportGraphDef(opts, def, &graph_, nullptr);
  EXPECT_TRUE(errors::IsInvalidArgument(s)) << s;

  def.mutable_versions()->Clear();
  graph_.ToGraphDef(&def);
  s = ImportGraphDef(opts, def, &graph_, nullptr);
  EXPECT_EQ(Status::OK(), s) << s;

  def.Clear();
  const int original_min_consumer = graph_.versions().min_consumer();
  def.mutable_versions()->set_min_consumer(original_min_consumer + 2);
  def.mutable_versions()->add_bad_consumers(TF_GRAPH_DEF_VERSION - 1);
  s = ImportGraphDef(opts, def, &graph_, nullptr);
  EXPECT_EQ(Status::OK(), s) << s;
  EXPECT_EQ(original_min_consumer + 2, graph_.versions().min_consumer());
  ASSERT_EQ(1, graph_.versions().bad_consumers_size());
  EXPECT_EQ(TF_GRAPH_DEF_VERSION - 1, graph_.versions().bad_consumers(0));
}

TEST_F(GraphConstructorTest, ImportGraphDef_ShapeWhitelist) {
  // Barrier's shape is an output vector of 2, but the graph says it's a vector
  // of 1. This is currently whitelisted.
  GraphDef def;
  bool parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
      node {
        name: "A"
        op: "Barrier"
        attr {
          key: "_output_shapes"
          value { list { shape {} } }
        }
        attr {
          key: "component_types"
          value { list { type: DT_FLOAT } }
        }
      }
      )EOF",
      &def);
  ASSERT_TRUE(parsed);
  Status s = ImportGraphDef(ImportGraphDefOptions(), def, &graph_, nullptr);
  EXPECT_EQ(Status::OK(), s) << s;
}

TEST_F(GraphConstructorTest, ImportGraphDef_WithCycle) {
  // Test graph produced in python using:
  /*
     with tf.Graph().as_default():
       i = tf.constant(0)
       c = lambda i: tf.less(i, 10)
       b = lambda i: tf.add(i, 1)
       r = tf.while_loop(c, b, [i])
       with open('/tmp/graph.txt', 'w') as f:
         f.write(str(tf.get_default_graph().as_graph_def()))
  */
  GraphDef def;
  bool parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
node {
  name: "Const"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 0
      }
    }
  }
}
node {
  name: "while/Enter"
  op: "Enter"
  input: "Const"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "frame_name"
    value {
      s: "while/while/"
    }
  }
  attr {
    key: "is_constant"
    value {
      b: false
    }
  }
  attr {
    key: "parallel_iterations"
    value {
      i: 10
    }
  }
}
node {
  name: "while/Merge"
  op: "Merge"
  input: "while/Enter"
  input: "while/NextIteration"
  attr {
    key: "N"
    value {
      i: 2
    }
  }
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Less/y"
  op: "Const"
  input: "^while/Merge"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 10
      }
    }
  }
}
node {
  name: "while/Less"
  op: "Less"
  input: "while/Merge"
  input: "while/Less/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/LoopCond"
  op: "LoopCond"
  input: "while/Less"
}
node {
  name: "while/Switch"
  op: "Switch"
  input: "while/Merge"
  input: "while/LoopCond"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "_class"
    value {
      list {
        s: "loc:@while/Merge"
      }
    }
  }
}
node {
  name: "while/Identity"
  op: "Identity"
  input: "while/Switch:1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Add/y"
  op: "Const"
  input: "^while/Identity"
  attr {
    key: "dtype"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_INT32
        tensor_shape {
        }
        int_val: 1
      }
    }
  }
}
node {
  name: "while/Add"
  op: "Add"
  input: "while/Identity"
  input: "while/Add/y"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/NextIteration"
  op: "NextIteration"
  input: "while/Add"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/Exit"
  op: "Exit"
  input: "while/Switch"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
versions {
  producer: 11
}
  )EOF",
      &def);
  ASSERT_TRUE(parsed);
  Status s = ImportGraphDef(ImportGraphDefOptions(), def, &graph_, nullptr);
  EXPECT_EQ(Status::OK(), s) << s;
}

TEST_F(GraphConstructorTest, ImportGraphDef_ErrorsDoNoChangeTheGraph) {
  GraphDef def;
  NodeDefBuilder("scope/A", "TestParams").Finalize(def.add_node());
  ImportGraphDefOptions opts;
  const string& source = graph_.FindNodeId(Graph::kSourceId)->name();
  const string& sink = graph_.FindNodeId(Graph::kSinkId)->name();

  Status s = ImportGraphDef(opts, def, &graph_, nullptr);
  ASSERT_EQ(Status::OK(), s) << s;
  EXPECT_EQ(3, graph_.num_nodes());  // 'scope/A', source and sink
  EXPECT_TRUE(HasControlEdge(source, sink));
  EXPECT_TRUE(HasControlEdge(source, "scope/A"));
  EXPECT_TRUE(HasControlEdge("scope/A", sink));
  EXPECT_EQ(3, graph_.num_edges());
  const string original_graph_description = GraphDebugString();

#define EXPECT_IMPORT_FAILURE(graph_def, options, expected_err)             \
  do {                                                                      \
    Status s = ImportGraphDef(options, graph_def, &graph_, nullptr);        \
    EXPECT_NE(Status::OK(), s) << s;                                        \
    EXPECT_TRUE(s.error_message().find(expected_err) != string::npos) << s; \
    const string graph_description = GraphDebugString();                    \
    EXPECT_EQ(original_graph_description, graph_description);               \
    EXPECT_EQ(3, graph_.num_nodes());                                       \
    EXPECT_TRUE(HasControlEdge(source, sink));                              \
    EXPECT_TRUE(HasControlEdge(source, "scope/A"));                         \
    EXPECT_TRUE(HasControlEdge("scope/A", sink));                           \
    EXPECT_EQ(3, graph_.num_edges());                                       \
  } while (0)

  EXPECT_IMPORT_FAILURE(def, opts,
                        "Node 'scope/A' already exists in the Graph");

  GraphDef bad_def;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "node{name:'!B' op:'TestParams'}", &bad_def));
  EXPECT_IMPORT_FAILURE(bad_def, opts,
                        "Node '!B': Node name contains invalid characters");

  opts.prefix = "!bad_prefix";
  EXPECT_IMPORT_FAILURE(def, opts,
                        "Imported node name prefix '!bad_prefix/' would lead "
                        "to invalid node names");

  opts.prefix = "import";
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "node{name:'B' op:'SomeUnknownOp'}", &bad_def));
  EXPECT_IMPORT_FAILURE(bad_def, opts,
                        "Op type not registered 'SomeUnknownOp'");

  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "node{name:'B' op:'TestOneInputTwoOutputs' input:'C'}", &bad_def));
  EXPECT_IMPORT_FAILURE(bad_def, opts, "Node 'B': Unknown input node 'C'");

  bool parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
      node{ name:"Root" op:"TestParams" } # TestParams produces a float
      node{
        name:"Integer"
        op:"TestOneInputOneOutput"
        attr{ key:"T" value{ type:DT_INT64 } }
        input: "Root"
      }
      )EOF",
      &bad_def);
  ASSERT_TRUE(parsed);
  EXPECT_IMPORT_FAILURE(bad_def, opts,
                        "Input 0 of node import/Integer was passed float from "
                        "import/Root:0 incompatible with expected int64");

  parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
      node{ name:"A" op:"TestParams" }
      node{ name:"B" op:"TestOneInputTwoOutputs" input:"A:1" }
      )EOF",
      &bad_def);
  ASSERT_TRUE(parsed);
  EXPECT_IMPORT_FAILURE(bad_def, opts,
                        "Node 'B': Connecting to invalid output 1 of source "
                        "node A which has 1 outputs");

  parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
      node{ name:"A" op:"TestParams" }
      node{ name:"B" op:"TestParams" }
      node{ name:"C" op:"TestOneInputTwoOutputs" input:"A" input:"B" }
      )EOF",
      &bad_def);
  ASSERT_TRUE(parsed);
  EXPECT_IMPORT_FAILURE(bad_def, opts, "do not match 2 inputs specified");

  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "node{ name:'A' op:'TestOneInputTwoOutputs' }", &bad_def));
  EXPECT_IMPORT_FAILURE(bad_def, opts, "do not match 0 inputs specified");

  parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
      node{
        name:"A"
        op:"TestParams"
        attr{
          key:"_class"
          value{ list{ s:"loc:@B" } }
        }
      })EOF",
      &bad_def);
  ASSERT_TRUE(parsed);
  EXPECT_IMPORT_FAILURE(
      bad_def, opts, "Node 'A' expects to be colocated with unknown node 'B'");

  opts.prefix = "";
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "node{name:'scope/A' op:'TestParams'}", &bad_def));
  EXPECT_IMPORT_FAILURE(bad_def, opts,
                        "Node 'scope/A' already exists in the Graph");

  parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
      node { name: "A" op: "TestParams" }
      node { name: "B" op: "L2Loss"
             input: "A:0"
             attr { key: "T" value { type: DT_FLOAT } }
             attr { key: "_output_shapes"
                    value { list { shape { dim { size: 43 } } } } } }
      )EOF",
      &bad_def);
  ASSERT_TRUE(parsed);
  EXPECT_IMPORT_FAILURE(bad_def, opts,
                        "Node 'B' has an _output_shapes attribute inconsistent "
                        "with the GraphDef for output #0");
#undef EXPECT_IMPORT_FAILURE
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
