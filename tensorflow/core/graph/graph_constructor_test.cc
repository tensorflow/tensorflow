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
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/versions.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/public/session.h"
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
    // Used to verify that errors don't change graph
    const string original_graph_description = GraphDebugString();

    Convert(gdef_ascii);
    GraphConstructorOptions opts;
    Status status = ConvertGraphDefToGraph(opts, gdef_, &graph_);
    EXPECT_FALSE(status.ok());

    for (const string& error : expected_error_strs) {
      EXPECT_TRUE(status.error_message().find(error) != string::npos)
          << "Expected to find '" << error << "' in " << status;
    }

    EXPECT_EQ(original_graph_description, GraphDebugString());
  }

  void ExpectError(const string& gdef_ascii, const ImportGraphDefOptions& opts,
                   const std::vector<string>& expected_error_strs,
                   ShapeRefiner* refiner = nullptr,
                   ImportGraphDefResults* results = nullptr) {
    // Used to verify that errors don't change graph
    const string original_graph_description = GraphDebugString();

    Convert(gdef_ascii);
    Status status = ImportGraphDef(opts, gdef_, &graph_, refiner, results);
    EXPECT_FALSE(status.ok());

    for (const string& error : expected_error_strs) {
      EXPECT_TRUE(status.error_message().find(error) != string::npos)
          << "Expected to find '" << error << "' in " << status;
    }

    EXPECT_EQ(original_graph_description, GraphDebugString());
  }

  void ExpectOK(const string& gdef_ascii) {
    Convert(gdef_ascii);
    GraphConstructorOptions opts;
    TF_CHECK_OK(ConvertGraphDefToGraph(opts, gdef_, &graph_));
  }

  void ExpectOK(const string& gdef_ascii, const ImportGraphDefOptions& opts,
                ShapeRefiner* refiner = nullptr,
                ImportGraphDefResults* results = nullptr) {
    Convert(gdef_ascii);
    Status s = ImportGraphDef(opts, gdef_, &graph_, refiner, results);
    EXPECT_EQ(Status::OK(), s) << s;
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
          e->dst()->name() == dst && e->dst_input() == dst_in) {
        return true;
      }
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
    Status s = GetNodeAttr(n->attrs(), kColocationAttrName, &value);
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
    return str_util::ConsumePrefix(&loc, kColocationGroupPrefix) ? string(loc)
                                                                 : "";
  }

  string GraphDebugString() const {
    return graph_.ToGraphDefDebug().DebugString();
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
REGISTER_OP("TestInput")
    .Output("a: float")
    .Output("b: float")
    .SetShapeFn(Scalars);
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
REGISTER_OP("TestVariadicOutput")
    .Output("outputs: N * int32")
    .Attr("N: int >= 0")
    .SetShapeFn(shape_inference::UnknownShape);
REGISTER_OP("TestDefaultAttr")
    .Attr("default_int: int=31415")
    .SetShapeFn(shape_inference::NoOutputs);
REGISTER_OP("RequiresCurrentGraphVersion")
    .Output("version: int32")
    .SetIsStateful()
    .SetShapeFn([](shape_inference::InferenceContext* c) {
      if (c->graph_def_version() != TF_GRAPH_DEF_VERSION) {
        return errors::InvalidArgument("Wrong graph version for shape");
      }
      return shape_inference::ScalarShape(c);
    });

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

TEST_F(GraphConstructorTest, ImportGraphThatUsesConstantValueFromInsideLoop) {
  // Test graph produced in python using:
  /*
    with tf.Graph().as_default():
      i = tf.constant(0)
      j = tf.constant([0])
      def s(t):
        t.set_shape(tf.vector(1))
        return t
      c = lambda i, _: tf.less(i, 10)
      b = lambda i, j: [i, s(tf.transpose(j, j))]
      r1, r2 = tf.while_loop(c, b, [i, j])
      with open('/tmp/graph.txt', 'w') as f:
        f.write(str(tf.get_default_graph().as_graph_def()))

  */
  const string pb_ascii = R"EOF(
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
  name: "Const_1"
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
          dim {
            size: 1
          }
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
  name: "while/Enter_1"
  op: "Enter"
  input: "Const_1"
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
  name: "while/Merge_1"
  op: "Merge"
  input: "while/Enter_1"
  input: "while/NextIteration_1"
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
  name: "while/Switch_1"
  op: "Switch"
  input: "while/Merge_1"
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
        s: "loc:@while/Merge_1"
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
  name: "while/Identity_1"
  op: "Identity"
  input: "while/Switch_1:1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/transpose"
  op: "Transpose"
  input: "while/Identity_1"
  input: "while/Identity_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tperm"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/NextIteration"
  op: "NextIteration"
  input: "while/Identity"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
node {
  name: "while/NextIteration_1"
  op: "NextIteration"
  input: "while/transpose"
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
node {
  name: "while/Exit_1"
  op: "Exit"
  input: "while/Switch_1"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
}
versions {
  producer: 21
}
  )EOF";
  GraphDef def;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(pb_ascii, &def));

  ImportGraphDefOptions opts;
  auto s = ImportGraphDef(opts, def, &graph_, nullptr);
  ASSERT_EQ(Status::OK(), s) << s;
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
  s = GetNodeAttr(a->attrs(), "default_int", &value);
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

TEST_F(GraphConstructorTest, ImportGraphDef_DeprecatedOps) {
  // BatchNormWithGlobalNormalization was deprecated in GraphDef version 9
  GraphDef def;
  bool parsed = protobuf::TextFormat::ParseFromString(
      R"EOF(
node {
  name: "zeros"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 1
          }
          dim {
            size: 149
          }
          dim {
            size: 149
          }
          dim {
            size: 32
          }
        }
        float_val: 0.0
      }
    }
  }
}
node {
  name: "m_v_beta_gamma"
  op: "Const"
  attr {
    key: "dtype"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "value"
    value {
      tensor {
        dtype: DT_FLOAT
        tensor_shape {
          dim {
            size: 32
          }
        }
        tensor_content: "\265\374\010=S\250\t\276\206\371>;Z\306y>\217]@\276\347\206\202\275\3747\241\275+1\227=J1\352\275\353?H;`\253\000>\023Y\014\276\341\310L;\301\030\314;\032Kw\275\273fQ;\036\252\200=\257o/\273\377\241\247\275\307,\332\274L\255\247\274\023\331R=r\271\225<\016/\204<\364\340\375\272t\030J=\220\306}\276\276x\003\275\231\013}\276\212\034\224\276\257\020\216>A\223\217\276"
      }
    }
  }
}
node {
  name: "batchnorm"
  op: "BatchNormWithGlobalNormalization"
  input: "zeros"
  input: "m_v_beta_gamma"
  input: "m_v_beta_gamma"
  input: "m_v_beta_gamma"
  input: "m_v_beta_gamma"
  attr {
    key: "T"
    value {
      type: DT_FLOAT
    }
  }
  attr {
    key: "scale_after_normalization"
    value {
      b: false
    }
  }
  attr {
    key: "variance_epsilon"
    value {
      f: 0.0010000000475
    }
  }
}
  )EOF",
      &def);
  ASSERT_TRUE(parsed);
  Status s = ImportGraphDef(ImportGraphDefOptions(), def, &graph_, nullptr);
  EXPECT_EQ(Status::OK(), s) << s;

  Graph g2(OpRegistry::Global());
  def.mutable_versions()->set_producer(10);
  s = ImportGraphDef(ImportGraphDefOptions(), def, &g2, nullptr);
  EXPECT_EQ(error::UNIMPLEMENTED, s.code());
  EXPECT_TRUE(s.error_message().find("BatchNormWithGlobalNormalization is not "
                                     "available in GraphDef version 10") !=
              string::npos)
      << s;
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

TEST_F(GraphConstructorTest, ImportGraphDef_InputMap) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK("node { name: 'input' op: 'TestInput' }", ImportGraphDefOptions(),
           &refiner);

  // Create input_map and use it to import more nodes
  ImportGraphDefOptions opts;
  opts.input_map[TensorId("new_input", 0)] = TensorId("input", 1);
  opts.input_map[TensorId("new_input", 1)] = TensorId("input", 0);

  ExpectOK(
      R"EOF(
      node { name: 'new_input' op: 'TestInput' }
      node { name: 't1' op: 'TestMul' input: [ 'new_input:0', 'new_input:1' ] }
      node { name: 't2' op: 'TestMul' input: [ 't1:0', 't1:0' ] }
      )EOF",
      opts, &refiner);

  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasNode("t2"));
  // `new_input` node is imported even though it's outputs aren't used
  EXPECT_TRUE(HasNode("new_input"));

  EXPECT_TRUE(HasEdge("input", 1, "t1", 0));
  EXPECT_TRUE(HasEdge("input", 0, "t1", 1));
  EXPECT_FALSE(HasEdge("new_input", 0, "t1", 0));
  EXPECT_FALSE(HasEdge("new_input", 0, "t1", 1));
  // Test that t2 is unaffected
  EXPECT_TRUE(HasEdge("t1", 0, "t2", 0));

  // Check that t1's NodeDef is consistent with graph
  Node* t1 = FindNode("t1");
  ASSERT_EQ(t1->requested_inputs().size(), 2);
  ASSERT_EQ(t1->requested_inputs()[0], "input:1");
  ASSERT_EQ(t1->requested_inputs()[1], "input:0");
}

TEST_F(GraphConstructorTest, ImportGraphDef_InputMapWithPrefix) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK(
      "node { name: 'input' op: 'TestInput' } "
      "node { name: 'unmapped_input' op: 'TestInput'}",
      ImportGraphDefOptions(), &refiner);

  // Map multiple inputs to the same existing input for more coverage
  ImportGraphDefOptions opts;
  opts.input_map[TensorId("input", 0)] = TensorId("input", 0);
  opts.input_map[TensorId("input", 1)] = TensorId("input", 0);

  opts.prefix = "import";

  // Import nodes with the same names as those already in the graph (the prefix
  // makes them unique)
  ExpectOK(
      R"EOF(
      node { name: 'input' op: 'TestInput' }
      node { name: 'unmapped_input' op: 'TestInput' }
      node { name: 't1' op: 'TestMul' input: [ 'input:0', 'input:1' ] }
      node { name: 't2' op: 'TestMul' input: [ 't1:0', 't1:0' ] }
      node { name: 't3' op: 'TestMul' input: [ 'unmapped_input:0',
                                               'unmapped_input:1' ] }
      )EOF",
      opts, &refiner);

  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("unmapped_input"));
  EXPECT_TRUE(HasNode("import/unmapped_input"));
  EXPECT_TRUE(HasNode("import/t1"));
  EXPECT_TRUE(HasNode("import/t2"));
  // `input` node is imported even though it's outputs aren't used
  EXPECT_TRUE(HasNode("import/input"));

  EXPECT_TRUE(HasEdge("input", 0, "import/t1", 0));
  EXPECT_TRUE(HasEdge("input", 0, "import/t1", 1));
  EXPECT_FALSE(HasEdge("import/input", 0, "import/t1", 0));
  EXPECT_FALSE(HasEdge("import/input", 0, "import/t1", 1));
  // Test that t2 and t3 are unaffected
  EXPECT_TRUE(HasEdge("import/t1", 0, "import/t2", 0));
  EXPECT_TRUE(HasEdge("import/unmapped_input", 0, "import/t3", 0));
  EXPECT_TRUE(HasEdge("import/unmapped_input", 1, "import/t3", 1));

  // Check that NodeDefs are consistent with graph
  Node* t1 = FindNode("import/t1");
  ASSERT_EQ(t1->requested_inputs().size(), 2);
  EXPECT_EQ(t1->requested_inputs()[0], "input:0");
  EXPECT_EQ(t1->requested_inputs()[1], "input:0");

  Node* t2 = FindNode("import/t2");
  ASSERT_EQ(t2->requested_inputs().size(), 2);
  EXPECT_EQ(t2->requested_inputs()[0], "import/t1:0");
  EXPECT_EQ(t2->requested_inputs()[1], "import/t1:0");

  Node* t3 = FindNode("import/t3");
  ASSERT_EQ(t3->requested_inputs().size(), 2);
  EXPECT_EQ(t3->requested_inputs()[0], "import/unmapped_input:0");
  EXPECT_EQ(t3->requested_inputs()[1], "import/unmapped_input:1");
}

TEST_F(GraphConstructorTest, ImportGraphDef_InputMapWithControlEdges) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK("node { name: 'W1' op: 'TestParams' }", ImportGraphDefOptions(),
           &refiner);

  // Create input_map containing control edges and use it to import more nodes
  ImportGraphDefOptions opts;
  const int kControlSlot = Graph::kControlSlot;
  opts.input_map[TensorId("W2", kControlSlot)] = TensorId("W1", kControlSlot);
  opts.input_map[TensorId("W3", kControlSlot)] = TensorId("W1", kControlSlot);
  ExpectOK(
      R"EOF(
      node { name: 'W2' op: 'TestParams' }
      node { name: 'W3' op: 'TestParams' }
      node { name: 'input' op: 'TestInput' input: [ '^W2' ] }
      node { name: 't1' op: 'TestOneInputTwoOutputs' input: [ 'W2' ] }
      node { name: 't2' op: 'TestOneInputTwoOutputs'
             input: [ 'input', '^W2', '^W3' ] }
      )EOF",
      opts, &refiner);

  EXPECT_TRUE(HasNode("W1"));
  EXPECT_TRUE(HasNode("W2"));
  EXPECT_TRUE(HasNode("W3"));
  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasNode("t2"));

  EXPECT_TRUE(HasControlEdge("W1", "input"));
  EXPECT_FALSE(HasControlEdge("W2", "input"));

  // Test that non-control edge is unaffected
  EXPECT_TRUE(HasEdge("W2", 0, "t1", 0));

  EXPECT_TRUE(HasControlEdge("W1", "t2"));
  EXPECT_FALSE(HasControlEdge("W2", "t2"));
  EXPECT_TRUE(HasEdge("input", 0, "t2", 0));
  // Test that t2's control inputs have been merged to single W1 edge
  Node* t2 = FindNode("t2");
  EXPECT_EQ(t2->in_edges().size(), 2);

  // Test remapping a control edge from a node with the same name as an existing
  // node
  opts.prefix = "import";
  opts.input_map.clear();
  opts.input_map[TensorId("W1", kControlSlot)] = TensorId("W1", kControlSlot);
  ExpectOK(
      R"EOF(
      node { name: 'W1' op: 'TestParams' }
      node { name: 'input' op: 'TestInput' input: [ '^W1' ] }
      node { name: 't1' op: 'TestOneInputTwoOutputs' input: [ 'W1' ] }
      )EOF",
      opts, &refiner);

  EXPECT_TRUE(HasNode("import/W1"));
  EXPECT_TRUE(HasNode("import/input"));
  EXPECT_TRUE(HasNode("import/t1"));

  EXPECT_TRUE(HasControlEdge("W1", "import/input"));
  EXPECT_FALSE(HasControlEdge("import/W1", "import/input"));
  EXPECT_TRUE(HasEdge("import/W1", 0, "import/t1", 0));
}

TEST_F(GraphConstructorTest, ImportGraphDef_InputMapWithBadControlEdge) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK("node { name: 'W1' op: 'TestParams' }", ImportGraphDefOptions(),
           &refiner);

  // Create input_map with bad control edge mapping
  ImportGraphDefOptions opts;
  opts.input_map[TensorId("W2", Graph::kControlSlot)] = TensorId("W1", 0);
  ExpectError(
      R"EOF(
      node { name: 'W2' op: 'TestParams' }
      node { name: 'input' op: 'TestInput' input: [ '^W2' ] }
      )EOF",
      opts,
      {"input_map entry ^W2->W1:0 between control edge and non-control edge"},
      &refiner);

  opts.input_map.clear();
  // "W2:0" isn't used in the imported graph but still causes an error
  opts.input_map[TensorId("W2", 0)] = TensorId("W1", Graph::kControlSlot);
  ExpectError(
      R"EOF(
      node { name: 'W2' op: 'TestParams' }
      node { name: 'input' op: 'TestInput' input: [ '^W2' ] }
      )EOF",
      opts,
      {"input_map entry W2:0->^W1 between control edge and non-control edge"},
      &refiner);
}

TEST_F(GraphConstructorTest, ImportGraphDef_InputMapWithInvalidNodeIndex) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK("node { name: 'input1' op: 'TestInput' }", ImportGraphDefOptions(),
           &refiner);

  // Create input_map with invalid source node index
  ImportGraphDefOptions opts;
  opts.input_map[TensorId("input2", 0)] = TensorId("input1", 3);
  ExpectError(
      R"EOF(
      node { name: 'input2' op: 'TestInput' }
      node { name: 't1' op: 'TestMul' input: [ 'input2:0', 'input2:1' ] }
      )EOF",
      opts,
      {"Node 't1': Connecting to invalid output 3 of source node input1 which "
       "has 2 outputs"},
      &refiner);
}

TEST_F(GraphConstructorTest, ImportGraphDef_InputMapWithMissingEntries) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK("node { name: 'W1' op: 'TestParams' }", ImportGraphDefOptions(),
           &refiner);

  // Create input_map referencing node that doesn't exist in graph
  ImportGraphDefOptions opts;
  const int kControlSlot = Graph::kControlSlot;
  opts.input_map[TensorId("W2", kControlSlot)] = TensorId("DNE", kControlSlot);
  ExpectError(
      R"EOF(
      node { name: 'W2' op: 'TestParams' }
      node { name: 'input' op: 'TestInput' input: [ '^W2' ] }
      )EOF",
      opts,
      {"node 'DNE' in input_map does not exist in graph (input_map entry: "
       "^W2->^DNE)"},
      &refiner);
}

TEST_F(GraphConstructorTest, ImportGraphDef_InputMapDuplicateNodeNames) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Add two nodes with the same name to graph
  Node* node;
  TF_CHECK_OK(NodeBuilder("dup", "Placeholder")
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(&graph_, &node));
  TF_CHECK_OK(NodeBuilder("dup", "Placeholder")
                  .Attr("dtype", DT_FLOAT)
                  .Finalize(&graph_, &node));

  // Create input_map referencing duplicate node
  ImportGraphDefOptions opts;
  opts.input_map[TensorId("new_input", 0)] = TensorId("dup", 0);
  ExpectError(
      R"EOF(
      node { name: 'new_input' op: 'TestInput' }
      node { name: 't1' op: 'TestMul' input: [ 'new_input:0', 'new_input:1' ] }
      )EOF",
      opts,
      {"cannot resolve input_map because multiple nodes exist with name 'dup'"},
      &refiner);
}

TEST_F(GraphConstructorTest, ImportGraphDef_InputMapMissingUnusedKeys) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // No input map
  ImportGraphDefOptions opts;
  ImportGraphDefResults results;
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }",
      opts, &refiner, &results);
  EXPECT_TRUE(results.missing_unused_input_map_keys.empty());

  // Non-empty missing_unused_input_map_keys
  results.missing_unused_input_map_keys.push_back(TensorId());
  ExpectError(
      "node { name: 'W2' op: 'TestParams' }", opts,
      {"All fields in results argument to ImportGraphDef() must be empty."},
      &refiner, &results);

  // Input map with some used, some unused keys
  const int kControlSlot = Graph::kControlSlot;
  results.missing_unused_input_map_keys.clear();
  opts.input_map[TensorId("W2", kControlSlot)] = TensorId("W1", kControlSlot);
  opts.input_map[TensorId("new_input", 0)] = TensorId("input", 0);
  opts.input_map[TensorId("new_input", 1)] = TensorId("input", 0);
  // Unused and missing (nonexistent index)
  opts.input_map[TensorId("new_input", 3)] = TensorId("input", 0);
  // Unused and missing (nonexistent node)
  opts.input_map[TensorId("DNE", 0)] = TensorId("input", 0);
  // Unused but not missing
  opts.input_map[TensorId("t1", 0)] = TensorId("W1", 0);
  // Unused but not missing
  opts.input_map[TensorId("variadic", 4)] = TensorId("input", 0);
  ExpectOK(
      R"EOF(
      node { name: 'W2' op: 'TestParams' }
      node { name: 'new_input' op: 'TestInput' input: [ '^W2' ] }
      node { name: 't1' op: 'TestMul' input: [ 'new_input:0', 'new_input:1' ] }
      node { name: 'variadic' op: 'TestVariadicOutput'
             attr { key: "N" value { i: 5 } } }
      )EOF",
      opts, &refiner, &results);

  std::set<TensorId> expected_unused_keys = {TensorId("new_input", 3),
                                             TensorId("DNE", 0)};
  ASSERT_EQ(results.missing_unused_input_map_keys.size(),
            expected_unused_keys.size());

  std::set<TensorId> actual_unused_keys(
      results.missing_unused_input_map_keys.begin(),
      results.missing_unused_input_map_keys.end());
  EXPECT_EQ(actual_unused_keys, expected_unused_keys);

  // Test edge case: node isn't imported due to skip_mapped_nodes, but we still
  // have a bad input_map key involving it.
  opts = ImportGraphDefOptions();
  opts.input_map[TensorId("new_input", 0)] = TensorId("input", 0);
  opts.input_map[TensorId("new_input", 1)] = TensorId("input", 1);
  // Index out of bounds
  opts.input_map[TensorId("new_input", 2)] = TensorId("input", 1);
  opts.skip_mapped_nodes = true;
  opts.prefix = "import";
  results = ImportGraphDefResults();
  ExpectOK(
      R"EOF(
      node { name: 'W2' op: 'TestParams' }
      node { name: 'new_input' op: 'TestInput' input: [ '^W2' ] }
      node { name: 't1' op: 'TestMul' input: [ 'new_input:0', 'new_input:1' ] }
      )EOF",
      opts, &refiner, &results);

  ASSERT_EQ(results.missing_unused_input_map_keys.size(), 1);
  EXPECT_EQ(results.missing_unused_input_map_keys[0],
            SafeTensorId("new_input", 2));
}

TEST_F(GraphConstructorTest, ImportGraphDef_InputMapWithUnboundInput) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK("node { name: 'input' op: 'TestInput' }", ImportGraphDefOptions(),
           &refiner);

  // Create input_map and use it to import more nodes
  ImportGraphDefOptions opts;
  opts.input_map[TensorId("new_input", 0)] = TensorId("input", 1);
  opts.input_map[TensorId("new_input", 1)] = TensorId("input", 0);

  // new_input exists in input_map but not in the graph being imported.
  ExpectOK(
      R"EOF(
      node { name: 't1' op: 'TestMul' input: [ 'new_input:0', 'new_input:1' ] }
      node { name: 't2' op: 'TestMul' input: [ 't1:0', 't1:0' ] }
      )EOF",
      opts, &refiner);

  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasNode("t2"));
  EXPECT_FALSE(HasNode("new_input"));

  EXPECT_TRUE(HasEdge("input", 1, "t1", 0));
  EXPECT_TRUE(HasEdge("input", 0, "t1", 1));
  // Test that t2 is unaffected
  EXPECT_TRUE(HasEdge("t1", 0, "t2", 0));

  // Check that t1's NodeDef is consistent with graph
  Node* t1 = FindNode("t1");
  ASSERT_EQ(t1->requested_inputs().size(), 2);
  ASSERT_EQ(t1->requested_inputs()[0], "input:1");
  ASSERT_EQ(t1->requested_inputs()[1], "input:0");
}

TEST_F(GraphConstructorTest, ImportGraphDef_SkipMappedNodes_FullyMapped) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK("node { name: 'input' op: 'TestInput' }", ImportGraphDefOptions(),
           &refiner);

  // Create input_map and use it to import more nodes
  ImportGraphDefOptions opts;
  opts.skip_mapped_nodes = true;
  opts.input_map[TensorId("new_input", 0)] = TensorId("input", 1);
  opts.input_map[TensorId("new_input", 1)] = TensorId("input", 0);

  ExpectOK(
      R"EOF(
      node { name: 'new_input' op: 'TestInput' }
      node { name: 't1' op: 'TestMul' input: [ 'new_input:0', 'new_input:1' ] }
      node { name: 't2' op: 'TestMul' input: [ 't1:0', 't1:0' ] }
      )EOF",
      opts, &refiner);

  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasNode("t2"));
  // `new_input` node is not imported because we set skip_mapped_nodes = true
  // and all of its inputs are mapped
  EXPECT_FALSE(HasNode("new_input"));

  EXPECT_TRUE(HasEdge("input", 1, "t1", 0));
  EXPECT_TRUE(HasEdge("input", 0, "t1", 1));
  // Test that t2 is unaffected
  EXPECT_TRUE(HasEdge("t1", 0, "t2", 0));

  // Check that t1's NodeDef is consistent with graph
  Node* t1 = FindNode("t1");
  ASSERT_EQ(t1->requested_inputs().size(), 2);
  ASSERT_EQ(t1->requested_inputs()[0], "input:1");
  ASSERT_EQ(t1->requested_inputs()[1], "input:0");
}

TEST_F(GraphConstructorTest, ImportGraphDef_SkipMappedNodes_NotFullyMapped) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK("node { name: 'input' op: 'TestInput' }", ImportGraphDefOptions(),
           &refiner);

  // Create input_map and use it to import more nodes
  ImportGraphDefOptions opts;
  opts.skip_mapped_nodes = true;
  opts.input_map[TensorId("new_input", 1)] = TensorId("input", 0);

  ExpectOK(
      R"EOF(
      node { name: 'new_input' op: 'TestInput' }
      node { name: 't1' op: 'TestMul' input: [ 'new_input:0', 'new_input:1' ] }
      node { name: 't2' op: 'TestMul' input: [ 't1:0', 't1:0' ] }
      )EOF",
      opts, &refiner);

  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasNode("t2"));
  // `new_input` node is imported because not all of its inputs are mapped
  EXPECT_TRUE(HasNode("new_input"));

  EXPECT_FALSE(HasEdge("input", 1, "t1", 0));
  EXPECT_TRUE(HasEdge("input", 0, "t1", 1));
  EXPECT_TRUE(HasEdge("new_input", 0, "t1", 0));
  EXPECT_FALSE(HasEdge("new_input", 1, "t1", 1));
  // Test that t2 is unaffected
  EXPECT_TRUE(HasEdge("t1", 0, "t2", 0));

  // Check that t1's NodeDef is consistent with graph
  Node* t1 = FindNode("t1");
  ASSERT_EQ(t1->requested_inputs().size(), 2);
  ASSERT_EQ(t1->requested_inputs()[0], "new_input:0");
  ASSERT_EQ(t1->requested_inputs()[1], "input:0");
}

TEST_F(GraphConstructorTest, ImportGraphDef_ReturnTensors) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  ImportGraphDefOptions opts;
  opts.return_tensors.push_back({"input", 1});
  opts.return_tensors.push_back({"t1", 0});
  opts.return_tensors.push_back({"input", 0});
  ImportGraphDefResults results;
  ExpectOK(
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: ['input:0', 'input:1'] }",
      opts, &refiner, &results);

  // Sanity checks
  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasEdge("input", 0, "t1", 0));
  EXPECT_TRUE(HasEdge("input", 1, "t1", 1));

  // Check return tensors
  ASSERT_EQ(results.return_tensors.size(), 3);
  EXPECT_EQ(results.return_tensors[0].first->name(), "input");
  EXPECT_EQ(results.return_tensors[0].second, 1);
  EXPECT_EQ(results.return_tensors[1].first->name(), "t1");
  EXPECT_EQ(results.return_tensors[1].second, 0);
  EXPECT_EQ(results.return_tensors[2].first->name(), "input");
  EXPECT_EQ(results.return_tensors[2].second, 0);

  // Test using prefix and returning element from input_map
  opts.return_tensors.clear();
  results = ImportGraphDefResults();
  opts.prefix = "import";
  opts.input_map[{"new_input", 1}] = {"input", 0};
  opts.return_tensors.push_back({"new_input", 0});
  opts.return_tensors.push_back({"new_input", 1});
  ExpectOK("node { name: 'new_input' op: 'TestInput' }", opts, &refiner,
           &results);

  EXPECT_TRUE(HasNode("import/new_input"));

  ASSERT_EQ(results.return_tensors.size(), 2);
  EXPECT_EQ(results.return_tensors[0].first->name(), "import/new_input");
  EXPECT_EQ(results.return_tensors[0].second, 0);
  EXPECT_EQ(results.return_tensors[1].first->name(), "input");
  EXPECT_EQ(results.return_tensors[1].second, 0);

  // Test returning node remapped to source node
  opts.prefix.clear();
  opts.input_map.clear();
  opts.return_tensors.clear();
  results = ImportGraphDefResults();
  opts.input_map[{"new_input", 0}] = {"_SOURCE", 0};
  opts.return_tensors.push_back({"new_input", 0});
  ExpectOK("node { name: 'new_input' op: 'TestInput' }", opts, &refiner,
           &results);

  EXPECT_TRUE(HasNode("new_input"));

  ASSERT_EQ(results.return_tensors.size(), 1);
  EXPECT_EQ(results.return_tensors[0].first->name(), "_SOURCE");
  EXPECT_EQ(results.return_tensors[0].second, 0);
}

TEST_F(GraphConstructorTest, ImportGraphDef_ReturnTensorsErrors) {
  // Null results with non-empty opts.return_tensors
  ImportGraphDefOptions opts;
  opts.return_tensors.push_back({"new_input", 0});
  ExpectError("node { name: 'new_input' op: 'TestInput' }", opts,
              {"results argument to ImportGraphDef() must be non-null if "
               "opts.return_tensors is non-empty"});

  // Non-empty results.return_tensors
  ImportGraphDefResults results;
  results.return_tensors.push_back({nullptr, 0});
  ExpectError(
      "node { name: 'new_input' op: 'TestInput' }", opts,
      {"All fields in results argument to ImportGraphDef() must be empty."},
      nullptr, &results);

  // Requesting tensor that isn't in graph def
  results.return_tensors.clear();
  ExpectError("node { name: 'W1' op: 'TestParams' }", opts,
              {"Requested return tensor 'new_input:0' not found in graph def"},
              nullptr, &results);

  // Requesting invalid node index
  opts.return_tensors.clear();
  opts.return_tensors.push_back({"new_input", 2});
  ExpectError("node { name: 'new_input' op: 'TestInput' }", opts,
              {"Invalid return output 2 of node 'new_input', which has 2 "
               "output(s)"},
              nullptr, &results);
}

TEST_F(GraphConstructorTest, ImportGraphDef_ReturnNodes) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  ImportGraphDefOptions opts;
  opts.return_nodes.push_back("input");
  opts.return_nodes.push_back("t1");
  ImportGraphDefResults results;
  ExpectOK(
      "node { name: 'input' op: 'TestInput' }"
      "node { name: 'input2' op: 'TestInput' }"
      "node { name: 't1' op: 'TestMul' input: ['input:0', 'input2:1'] }",
      opts, &refiner, &results);

  // Sanity checks
  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("input2"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasEdge("input", 0, "t1", 0));
  EXPECT_TRUE(HasEdge("input2", 1, "t1", 1));

  // Check return tensors
  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_tensors.size(), 0);
  EXPECT_EQ(results.missing_unused_input_map_keys.size(), 0);
  EXPECT_EQ(results.return_nodes[0]->name(), "input");
  EXPECT_EQ(results.return_nodes[1]->name(), "t1");

  // Test using prefix
  opts = ImportGraphDefOptions();
  results = ImportGraphDefResults();
  opts.prefix = "import";
  opts.return_nodes.push_back("input");
  ExpectOK("node { name: 'input' op: 'TestInput' }", opts, &refiner, &results);

  EXPECT_TRUE(HasNode("import/input"));

  ASSERT_EQ(results.return_nodes.size(), 1);
  EXPECT_EQ(results.return_nodes[0]->name(), "import/input");

  // Test that input_map has no effect
  opts = ImportGraphDefOptions();
  results = ImportGraphDefResults();
  opts.input_map[{"new_input", 0}] = {"input", 0};
  opts.return_nodes.push_back("new_input");
  ExpectOK("node { name: 'new_input' op: 'TestInput' }", opts, &refiner,
           &results);

  EXPECT_TRUE(HasNode("new_input"));

  ASSERT_EQ(results.return_nodes.size(), 1);
  EXPECT_EQ(results.return_nodes[0]->name(), "new_input");
}

TEST_F(GraphConstructorTest, ImportGraphDef_ReturnNodesErrors) {
  // Null results with non-empty opts.return_nodes
  ImportGraphDefOptions opts;
  opts.return_nodes.push_back("new_input");
  ExpectError("node { name: 'new_input' op: 'TestInput' }", opts,
              {"results argument to ImportGraphDef() must be non-null if "
               "opts.return_nodes is non-empty"});

  // Non-empty results.return_nodes
  ImportGraphDefResults results;
  results.return_nodes.push_back(nullptr);
  ExpectError(
      "node { name: 'new_input' op: 'TestInput' }", opts,
      {"All fields in results argument to ImportGraphDef() must be empty."},
      nullptr, &results);

  // Requesting node that isn't in graph def
  results.return_nodes.clear();
  ExpectError("node { name: 'W1' op: 'TestParams' }", opts,
              {"Requested return node 'new_input' not found in graph def"},
              nullptr, &results);

  // Requesting return_nodes with skip_mapped_nodes not yet implemented
  opts.skip_mapped_nodes = true;
  ExpectError("node { name: 'new_input' op: 'TestInput' }", opts,
              {"Requesting return_nodes with skip_mapped_nodes set is not "
               "currently supported"});
}

TEST_F(GraphConstructorTest, ImportGraphDef_UniquifyNames) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  const char* graph_def_str =
      "node { name: 'A' op: 'TestInput' }"
      "node { name: 'B' op: 'TestOneInputTwoOutputs' input: ['A'] }";

  // Initial import
  ImportGraphDefOptions opts;
  opts.uniquify_names = true;
  opts.return_nodes.push_back("A");
  opts.return_nodes.push_back("B");
  ImportGraphDefResults results;
  ExpectOK(graph_def_str, opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A");
  EXPECT_EQ(results.return_nodes[1]->name(), "B");
  EXPECT_EQ(results.return_nodes[1]->def().input(0), "A");

  // Repeat the same import
  results = ImportGraphDefResults();
  ExpectOK(graph_def_str, opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A_1");
  EXPECT_EQ(results.return_nodes[1]->name(), "B_1");
  EXPECT_EQ(results.return_nodes[1]->def().input(0), "A_1:0");

  // Repeat the same import again
  results = ImportGraphDefResults();
  ExpectOK(graph_def_str, opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A_2");
  EXPECT_EQ(results.return_nodes[1]->name(), "B_2");
  EXPECT_EQ(results.return_nodes[1]->def().input(0), "A_2:0");

  // Import with an already-used prefix and uniquify_prefix = true
  opts.prefix = "A";
  opts.uniquify_prefix = true;
  results = ImportGraphDefResults();
  ExpectOK(graph_def_str, opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A_3/A");
  EXPECT_EQ(results.return_nodes[1]->name(), "A_3/B");
  EXPECT_EQ(results.return_nodes[1]->def().input(0), "A_3/A");

  // Create B_3 node to keep the A/B numbering in sync
  ExpectOK("node { name: 'B_3' op: 'TestInput' }");

  // Import with an already-used prefix and uniquify_prefix = false
  opts.uniquify_prefix = false;
  results = ImportGraphDefResults();
  ExpectOK(graph_def_str, opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A/A");
  EXPECT_EQ(results.return_nodes[1]->name(), "A/B");
  EXPECT_EQ(results.return_nodes[1]->def().input(0), "A/A");

  // Repeat the same import
  results = ImportGraphDefResults();
  ExpectOK(graph_def_str, opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A/A_1");
  EXPECT_EQ(results.return_nodes[1]->name(), "A/B_1");
  EXPECT_EQ(results.return_nodes[1]->def().input(0), "A/A_1:0");

  // Import with existing de-duped node names
  opts = ImportGraphDefOptions();
  opts.uniquify_names = true;
  opts.return_nodes.push_back("A_1");
  opts.return_nodes.push_back("B_1");
  results = ImportGraphDefResults();
  ExpectOK(
      "node { name: 'A_1' op: 'TestInput' }"
      "node { name: 'B_1' op: 'TestOneInputTwoOutputs' input: ['A_1:0'] }",
      opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A_1_1");
  EXPECT_EQ(results.return_nodes[1]->name(), "B_1_1");
  EXPECT_EQ(results.return_nodes[1]->def().input(0), "A_1_1:0");

  // Import with node names that must be de-duped from names and prefixes that
  // exist in both the existing graph and the GraphDef being imported.
  opts = ImportGraphDefOptions();
  opts.uniquify_names = true;
  opts.return_nodes.push_back("A");
  opts.return_nodes.push_back("A_4");
  opts.return_nodes.push_back("B");
  opts.return_nodes.push_back("B_4/B");
  results = ImportGraphDefResults();
  ExpectOK(
      "node { name: 'A' op: 'TestInput' }"
      "node { name: 'A_4' op: 'TestInput' }"
      "node { name: 'B' op: 'TestOneInputTwoOutputs' input: ['A'] }"
      "node { name: 'B_4/B' op: 'TestOneInputTwoOutputs' input: ['A_4'] }",
      opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 4);
  EXPECT_EQ(results.return_nodes[0]->name(), "A_5");
  EXPECT_EQ(results.return_nodes[1]->name(), "A_4");
  EXPECT_EQ(results.return_nodes[2]->name(), "B_5");
  EXPECT_EQ(results.return_nodes[2]->def().input(0), "A_5:0");
  EXPECT_EQ(results.return_nodes[3]->name(), "B_4/B");
  EXPECT_EQ(results.return_nodes[3]->def().input(0), "A_4");

  // Create node with prefix and then import node with same name
  ExpectOK("node { name: 'foo/abc' op: 'ABC' }");
  opts = ImportGraphDefOptions();
  opts.uniquify_names = true;
  opts.return_nodes.push_back("foo");
  results = ImportGraphDefResults();
  ExpectOK("node { name: 'foo' op: 'TestInput' }", opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 1);
  EXPECT_EQ(results.return_nodes[0]->name(), "foo_1");

  // Imported nodes can't conflict with intermediate name (but can conflict with
  // outer name)
  ExpectOK("node { name: 'outer/inner/abc' op: 'ABC' }");

  opts = ImportGraphDefOptions();
  opts.uniquify_names = true;
  opts.return_nodes.push_back("outer");
  opts.return_nodes.push_back("inner");
  opts.return_nodes.push_back("abc");
  opts.return_nodes.push_back("outer/inner");
  opts.return_nodes.push_back("outer/inner/abc");
  results = ImportGraphDefResults();
  ExpectOK(
      "node { name: 'outer' op: 'TestInput' }"
      "node { name: 'inner' op: 'TestInput' }"
      "node { name: 'abc' op: 'TestInput' }"
      "node { name: 'outer/inner' op: 'TestInput' }"
      "node { name: 'outer/inner/abc' op: 'TestInput' }",
      opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 5);
  EXPECT_EQ(results.return_nodes[0]->name(), "outer_1");
  EXPECT_EQ(results.return_nodes[1]->name(), "inner");
  EXPECT_EQ(results.return_nodes[2]->name(), "abc");
  EXPECT_EQ(results.return_nodes[3]->name(), "outer/inner_1");
  EXPECT_EQ(results.return_nodes[4]->name(), "outer/inner/abc_1");

  // Import with input map containing conflicting names
  opts = ImportGraphDefOptions();
  opts.uniquify_names = true;
  opts.input_map[TensorId("A", 0)] = TensorId("A", 0);
  opts.input_map[TensorId("B", 0)] = TensorId("B", 0);
  opts.return_nodes.push_back("A");
  opts.return_nodes.push_back("B");
  results = ImportGraphDefResults();
  ExpectOK(graph_def_str, opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A_6");
  EXPECT_EQ(results.return_nodes[1]->name(), "B_6");
  EXPECT_EQ(results.return_nodes[1]->def().input(0), "A:0");
}

TEST_F(GraphConstructorTest, ImportGraphDef_UniquifyNames_ColocationGroups) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Create nodes 'A' and 'b"
  ExpectOK(
      "node { name: 'A' op: 'TestInput' }"
      "node { name: 'B' op: 'TestOneInputTwoOutputs' input: ['A'] }");

  // Check that colocation groups are updated
  ImportGraphDefOptions opts;
  opts.uniquify_names = true;
  opts.return_nodes.push_back("A");
  opts.return_nodes.push_back("B");
  ImportGraphDefResults results;
  ExpectOK(
      "node { name: 'A' op: 'TestInput' }"
      "node { name: 'B' op: 'TestOneInputTwoOutputs' input: ['A:0'] "
      "       attr { key: '_class' value { list { s:'loc:@A' } } } }",
      opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A_1");
  EXPECT_EQ(results.return_nodes[1]->name(), "B_1");
  const AttrValue* class_attr =
      results.return_nodes[1]->attrs().Find(kColocationAttrName);
  ASSERT_TRUE(class_attr != nullptr);
  ASSERT_EQ(class_attr->list().s_size(), 1);
  EXPECT_EQ(class_attr->list().s(0), "loc:@A_1");

  results = ImportGraphDefResults();
  ExpectOK(
      "node { name: 'A' op: 'TestInput' "
      "       attr { key: '_class' value { list { s:'loc:@B' } } } }"
      "node { name: 'B' op: 'TestOneInputTwoOutputs' input: ['A:0'] }",
      opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A_2");
  EXPECT_EQ(results.return_nodes[1]->name(), "B_2");
  class_attr = results.return_nodes[0]->attrs().Find(kColocationAttrName);
  ASSERT_TRUE(class_attr != nullptr);
  ASSERT_EQ(class_attr->list().s_size(), 1);
  EXPECT_EQ(class_attr->list().s(0), "loc:@B_2");

  results = ImportGraphDefResults();
  ExpectOK(
      "node { name: 'A' op: 'TestInput' "
      "       attr { key: '_class' value { list { s:'loc:@B' } } } }"
      "node { name: 'B' op: 'TestOneInputTwoOutputs' input: ['A:0'] "
      "       attr { key: '_class' value { list { s:'loc:@B' } } } }",
      opts, &refiner, &results);

  ASSERT_EQ(results.return_nodes.size(), 2);
  EXPECT_EQ(results.return_nodes[0]->name(), "A_3");
  EXPECT_EQ(results.return_nodes[1]->name(), "B_3");
  class_attr = results.return_nodes[0]->attrs().Find(kColocationAttrName);
  ASSERT_TRUE(class_attr != nullptr);
  ASSERT_EQ(class_attr->list().s_size(), 1);
  EXPECT_EQ(class_attr->list().s(0), "loc:@B_3");
  class_attr = results.return_nodes[1]->attrs().Find(kColocationAttrName);
  ASSERT_TRUE(class_attr != nullptr);
  ASSERT_EQ(class_attr->list().s_size(), 1);
  EXPECT_EQ(class_attr->list().s(0), "loc:@B_3");
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

TEST_F(GraphConstructorTest, ImportGraphDef_ControlDeps) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with nodes we'll use in control deps and input map
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'W2' op: 'TestParams' }",
      ImportGraphDefOptions(), &refiner);

  ImportGraphDefOptions opts;
  opts.control_dependencies = {"W1", "W2"};
  opts.prefix = "import";
  // Create two input mappings to the same control dep so we can test adding and
  // consolidating control deps from the same node
  opts.input_map[TensorId("W2", -1)] = TensorId("W2", -1);
  opts.input_map[TensorId("W3", -1)] = TensorId("W2", -1);
  ExpectOK(
      R"EOF(
      node { name: 'W2' op: 'TestParams' }
      node { name: 'W3' op: 'TestParams' }
      node { name: 'input' op: 'TestInput' }
      node { name: 'input2' op: 'TestInput' input: [ '^W2' ] }
      node { name: 'input3' op: 'TestInput' input: [ '^W2', '^W3' ] }
      node { name: 't1' op: 'TestMul' input: [ 'input:0', 'input:1' ] }
      node { name: 't2' op: 'TestMul'
             input: [ 'input:0', 'input:1', '^W2', '^W3' ] }
      )EOF",
      opts, &refiner);

  // Sanity checks
  EXPECT_TRUE(HasNode("import/W2"));
  EXPECT_TRUE(HasNode("import/W3"));
  EXPECT_TRUE(HasNode("import/input"));
  EXPECT_TRUE(HasNode("import/input2"));
  EXPECT_TRUE(HasNode("import/input3"));
  EXPECT_TRUE(HasNode("import/t1"));
  EXPECT_TRUE(HasNode("import/t2"));

  EXPECT_TRUE(HasControlEdge("W1", "import/W2"));
  EXPECT_TRUE(HasControlEdge("W2", "import/W2"));

  EXPECT_TRUE(HasControlEdge("W1", "import/W3"));
  EXPECT_TRUE(HasControlEdge("W2", "import/W3"));

  EXPECT_TRUE(HasControlEdge("W1", "import/input"));
  EXPECT_TRUE(HasControlEdge("W2", "import/input"));

  // Test that t1 doesn't have redundant control edges
  EXPECT_FALSE(HasControlEdge("W1", "import/t1"));
  EXPECT_FALSE(HasControlEdge("W2", "import/t1"));
  EXPECT_TRUE(HasEdge("import/input", 0, "import/t1", 0));
  EXPECT_TRUE(HasEdge("import/input", 1, "import/t1", 1));

  // Test that t2 has consolidated remapped control edge and not redundant
  // control edge
  EXPECT_TRUE(HasControlEdge("W2", "import/t2"));
  EXPECT_FALSE(HasControlEdge("W1", "import/t2"));
  EXPECT_TRUE(HasEdge("import/input", 0, "import/t1", 0));
  EXPECT_TRUE(HasEdge("import/input", 1, "import/t1", 1));

  // Test that input2 has control edges since its only input was remapped
  EXPECT_TRUE(HasControlEdge("W1", "import/input2"));
  EXPECT_TRUE(HasControlEdge("W2", "import/input2"));
  EXPECT_FALSE(HasControlEdge("import/W2", "import/input2"));

  // Test that input3 has consolidated remapped control edge and added control
  // edge
  EXPECT_TRUE(HasControlEdge("W1", "import/input3"));
  EXPECT_TRUE(HasControlEdge("W2", "import/input3"));

  // Test that node defs are consistent with graph
  Node* w2 = FindNode("import/W2");
  ASSERT_EQ(w2->requested_inputs().size(), 2);
  EXPECT_EQ(w2->requested_inputs()[0], "^W1");
  EXPECT_EQ(w2->requested_inputs()[1], "^W2");

  Node* w3 = FindNode("import/W3");
  ASSERT_EQ(w3->requested_inputs().size(), 2);
  EXPECT_EQ(w3->requested_inputs()[0], "^W1");
  EXPECT_EQ(w3->requested_inputs()[1], "^W2");

  Node* input = FindNode("import/input");
  ASSERT_EQ(input->requested_inputs().size(), 2);
  EXPECT_EQ(input->requested_inputs()[0], "^W1");
  EXPECT_EQ(input->requested_inputs()[1], "^W2");

  Node* input2 = FindNode("import/input2");
  ASSERT_EQ(input2->requested_inputs().size(), 2);
  EXPECT_EQ(input2->requested_inputs()[0], "^W2");
  EXPECT_EQ(input2->requested_inputs()[1], "^W1");

  Node* input3 = FindNode("import/input3");
  ASSERT_EQ(input3->requested_inputs().size(), 2);
  EXPECT_EQ(input3->requested_inputs()[0], "^W2");
  EXPECT_EQ(input3->requested_inputs()[1], "^W1");

  Node* t1 = FindNode("import/t1");
  ASSERT_EQ(t1->requested_inputs().size(), 2);
  EXPECT_EQ(t1->requested_inputs()[0], "import/input:0");
  EXPECT_EQ(t1->requested_inputs()[1], "import/input:1");

  Node* t2 = FindNode("import/t2");
  ASSERT_EQ(t2->requested_inputs().size(), 3);
  EXPECT_EQ(t2->requested_inputs()[0], "import/input:0");
  EXPECT_EQ(t2->requested_inputs()[1], "import/input:1");
  EXPECT_EQ(t2->requested_inputs()[2], "^W2");
}

TEST_F(GraphConstructorTest, ImportGraphDef_ControlDepsWithCycle) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with nodes we'll use in control deps and input map
  ExpectOK(
      "node { name: 'W1' op: 'TestParams' }"
      "node { name: 'input' op: 'TestInput' }",
      ImportGraphDefOptions(), &refiner);

  ImportGraphDefOptions opts;
  opts.control_dependencies.push_back("W1");
  // Use input_map to ensure the cycle doesn't inherit the control deps from
  // new_input
  opts.input_map[TensorId("new_input", 0)] = TensorId("input", 0);

  // ImportGraphDef only allows backedges into merge nodes that are part of
  // while loops (since backedges are only expected in while loops)
  ExpectOK(
      R"EOF(
      node { name: 'new_input' op: 'TestInput' }
      node { name: 'merge' op: 'Merge' input: [ 'new_input:0', 'next:0' ]
             attr { key: "N" value: { i: 2 } }
             attr { key: "T" value: { type: DT_FLOAT } } }
      node { name: 't1' op: 'TestMul' input: [ 'merge:0', 'merge:0' ] }
      node { name: 'next' op: 'NextIteration' input: ['t1:0']
             attr { key: "T" value: { type: DT_FLOAT } } }
      )EOF",
      opts, &refiner);

  EXPECT_TRUE(HasNode("new_input"));
  EXPECT_TRUE(HasNode("merge"));
  EXPECT_TRUE(HasNode("t1"));
  EXPECT_TRUE(HasNode("next"));

  // Sanity check we created cycle
  EXPECT_TRUE(HasEdge("merge", 0, "t1", 0));
  EXPECT_TRUE(HasEdge("t1", 0, "next", 0));
  EXPECT_TRUE(HasEdge("next", 0, "merge", 1));

  // Test that control dep was added to exactly one node of cycle
  EXPECT_TRUE(HasControlEdge("W1", "merge"));
  EXPECT_FALSE(HasControlEdge("W1", "t1"));

  // Test that node defs are consistent with graph
  Node* merge = FindNode("merge");
  ASSERT_EQ(merge->requested_inputs().size(), 3);
  EXPECT_EQ(merge->requested_inputs()[0], "input:0");
  EXPECT_EQ(merge->requested_inputs()[1], "next:0");
  EXPECT_EQ(merge->requested_inputs()[2], "^W1");

  Node* t1 = FindNode("t1");
  ASSERT_EQ(t1->requested_inputs().size(), 2);
  EXPECT_EQ(t1->requested_inputs()[0], "merge:0");
  EXPECT_EQ(t1->requested_inputs()[1], "merge:0");

  Node* next = FindNode("next");
  ASSERT_EQ(next->requested_inputs().size(), 1);
  EXPECT_EQ(next->requested_inputs()[0], "t1:0");
}

TEST_F(GraphConstructorTest, ImportGraphDef_ControlDepsErrors) {
  // Control dep that isn't in graph def
  ImportGraphDefOptions opts;
  opts.control_dependencies.push_back("W1");
  ExpectError("node { name: 'W1' op: 'TestParams' }", opts,
              {"node 'W1' in control_dependencies does not exist in graph"});
}

TEST_F(GraphConstructorTest, ImportGraphDef_ErrorsDoNoChangeTheGraph) {
  GraphDef def;
  TF_EXPECT_OK(
      NodeDefBuilder("scope/A", "TestParams").Finalize(def.add_node()));
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
                        "Node name 'scope/A' already exists in the Graph");

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
                        "Node name 'scope/A' already exists in the Graph");

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

TEST_F(GraphConstructorTest, ImportGraphDef_FunctionDefs) {
  // Import a graph def containing a function. The graph def was generated using
  // this python code:
  // @function.Defun(tf.float32, tf.float32, tf.float32)
  // def FooGrad(x, y, dz): return dz, dz
  //
  // @function.Defun(tf.float32, tf.float32, grad_func=FooGrad)
  // def Foo(x, y): return x + y
  //
  // p1 = tf.placeholder(tf.float32)
  // p2 = tf.placeholder(tf.float32)
  // foo = Foo(p1, p2)
  ImportGraphDefOptions opts;
  ExpectOK(
      R"EOF(
      node {
        name: "Placeholder" op: "Placeholder"
        attr { key: "dtype" value { type: DT_FLOAT } }
        attr { key: "shape" value { shape { } } }
      }
      node {
        name: "Placeholder_1" op: "Placeholder"
        attr { key: "dtype" value { type: DT_FLOAT } }
        attr { key: "shape" value { shape { } } }
      }
      node {
        name: "Foo_d03c39a3" op: "Foo_d03c39a3"
        input: "Placeholder" input: "Placeholder_1"
      }
      library {
        function {
          signature {
            name: "Foo_d03c39a3"
            input_arg { name: "x" type: DT_FLOAT }
            input_arg { name: "y" type: DT_FLOAT }
            output_arg { name: "add" type: DT_FLOAT }
          }
          node_def {
            name: "add" op: "Add" input: "x" input: "y"
            attr { key: "T" value { type: DT_FLOAT } }
          }
          ret { key: "add" value: "add:z:0" }
        }
        function {
          signature {
            name: "FooGrad_dc60abc8"
            input_arg { name: "x" type: DT_FLOAT }
            input_arg { name: "y" type: DT_FLOAT }
            input_arg { name: "dz" type: DT_FLOAT }
            output_arg { name: "dz" type: DT_FLOAT }
            output_arg { name: "dz_U0" type: DT_FLOAT }
          }
          ret { key: "dz" value: "dz:0" }
          ret { key: "dz_U0" value: "dz:0" }
        }
        gradient {
          function_name: "Foo_d03c39a3" gradient_func: "FooGrad_dc60abc8"
        }
      }
      versions { producer: 21 min_consumer: 12 }
      )EOF",
      opts);

  EXPECT_TRUE(HasNode("Placeholder"));
  EXPECT_TRUE(HasNode("Placeholder_1"));
  EXPECT_TRUE(HasNode("Foo_d03c39a3"));
  // Check that Foo and FooGrad have been imported
  const OpDef* op_def;
  TF_ASSERT_OK(graph_.op_registry()->LookUpOpDef("Foo_d03c39a3", &op_def));
  TF_ASSERT_OK(graph_.op_registry()->LookUpOpDef("FooGrad_dc60abc8", &op_def));

  // Re-serialize and run the graph. This tests that re-serialized functions can
  // be imported again and that imported functions can be run.
  GraphDef gdef;
  graph_.ToGraphDef(&gdef);
  EXPECT_EQ(gdef.library().function_size(), 2);
  EXPECT_EQ(gdef.library().gradient_size(), 1);
  EXPECT_EQ(gdef.library().gradient()[0].function_name(), "Foo_d03c39a3");
  EXPECT_EQ(gdef.library().gradient()[0].gradient_func(), "FooGrad_dc60abc8");

  std::unique_ptr<Session> sess(NewSession(SessionOptions()));
  TF_ASSERT_OK(sess->Create(gdef));

  Tensor p1(DT_FLOAT, TensorShape({1}));
  p1.scalar<float>()() = 1.0;
  Tensor p2(DT_FLOAT, TensorShape({1}));
  p2.scalar<float>()() = 2.0;
  std::vector<std::pair<string, Tensor>> inputs = {{"Placeholder", p1},
                                                   {"Placeholder_1", p2}};
  std::vector<string> output_names = {"Foo_d03c39a3"};
  std::vector<string> target_names;
  std::vector<Tensor> outputs;
  TF_ASSERT_OK(sess->Run(inputs, output_names, target_names, &outputs));

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].scalar<float>()(), 3.0);
}

TEST_F(GraphConstructorTest, ImportGraphDef_NestedFunctionDefs) {
  // Import a graph def containing a function. The graph def was generated using
  // this python code:
  //   @function.Defun(tf.float32, tf.float32)
  //   def Inner(x, y): return x + y
  //
  //   @function.Defun(tf.float32, tf.float32)
  //   def Outer(x, y): return Inner(x, y)
  //
  //   p1 = tf.placeholder(tf.float32)
  //   p2 = tf.placeholder(tf.float32)
  //   Outer(p1, p2)
  ImportGraphDefOptions opts;
  ExpectOK(
      R"EOF(
      node {
        name: "Placeholder" op: "Placeholder"
        attr { key: "dtype" value { type: DT_FLOAT } }
        attr { key: "shape" value { shape { } } }
      }
      node {
        name: "Placeholder_1" op: "Placeholder"
        attr { key: "dtype" value { type: DT_FLOAT } }
        attr { key: "shape" value { shape { } } }
      }
      node {
        name: "Outer_966fa13d" op: "Outer_966fa13d"
        input: "Placeholder" input: "Placeholder_1"
      }
      library {
        function {
          signature {
            name: "Outer_966fa13d"
            input_arg { name: "x" type: DT_FLOAT }
            input_arg { name: "y" type: DT_FLOAT }
            output_arg { name: "Inner_d03c39a3" type: DT_FLOAT }
          }
          node_def {
            name: "Inner_d03c39a3" op: "Inner_d03c39a3" input: "x" input: "y"
          }
          ret { key: "Inner_d03c39a3" value: "Inner_d03c39a3:add:0" }
        }
        function {
          signature {
            name: "Inner_d03c39a3"
            input_arg { name: "x" type: DT_FLOAT }
            input_arg { name: "y" type: DT_FLOAT }
            output_arg { name: "add" type: DT_FLOAT }
          }
          node_def {
            name: "add" op: "Add" input: "x" input: "y"
            attr { key: "T" value { type: DT_FLOAT } }
          }
          ret { key: "add" value: "add:z:0" }
        }
      }
      versions { producer: 21 min_consumer: 12 }
      )EOF",
      opts);

  EXPECT_TRUE(HasNode("Placeholder"));
  EXPECT_TRUE(HasNode("Placeholder_1"));
  EXPECT_TRUE(HasNode("Outer_966fa13d"));
  // Check that Inner and Outer have been imported
  const OpDef* op_def;
  Status s = graph_.op_registry()->LookUpOpDef("Inner_d03c39a3", &op_def);
  ASSERT_TRUE(s.ok()) << s.error_message();
  s = graph_.op_registry()->LookUpOpDef("Outer_966fa13d", &op_def);
  ASSERT_TRUE(s.ok()) << s.error_message();

  // Re-serialize and run the graph. This tests that re-serialized functions can
  // be imported again and that imported functions can be run.
  GraphDef gdef;
  graph_.ToGraphDef(&gdef);
  std::unique_ptr<Session> sess(NewSession(SessionOptions()));
  s = sess->Create(gdef);
  ASSERT_TRUE(s.ok()) << s.error_message();

  Tensor p1(DT_FLOAT, TensorShape({1}));
  p1.scalar<float>()() = 1.0;
  Tensor p2(DT_FLOAT, TensorShape({1}));
  p2.scalar<float>()() = 2.0;
  std::vector<std::pair<string, Tensor>> inputs = {{"Placeholder", p1},
                                                   {"Placeholder_1", p2}};
  std::vector<string> output_names = {"Outer_966fa13d"};
  std::vector<string> target_names;
  std::vector<Tensor> outputs;
  s = sess->Run(inputs, output_names, target_names, &outputs);
  ASSERT_TRUE(s.ok()) << s.error_message();

  ASSERT_EQ(outputs.size(), 1);
  EXPECT_EQ(outputs[0].scalar<float>()(), 3.0);
}

// NOTE(skyewm): the C API depends on this behavior, but it's easier to write
// the test here.
TEST_F(GraphConstructorTest, ImportGraphDef_OptionsMemMgmt) {
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());

  // Populate graph with node we'll use in input map
  ExpectOK("node { name: 'input' op: 'TestInput' }", ImportGraphDefOptions(),
           &refiner);

  // Add some strings to ImportGraphDefOptions and then rewrite the buffers.
  char buf1[100];
  char buf2[100];
  char buf3[100];
  snprintf(buf1, sizeof(buf1), "input");
  snprintf(buf2, sizeof(buf2), "new_input");
  snprintf(buf3, sizeof(buf3), "t1");

  ImportGraphDefOptions opts;
  opts.input_map[TensorId(buf2, 0)] = TensorId(buf1, 0);
  opts.return_tensors.push_back(TensorId(buf3, 0));

  snprintf(buf1, sizeof(buf1), "xxxxxxxxxxxxxxxxxxxx");
  snprintf(buf2, sizeof(buf2), "xxxxxxxxxxxxxxxxxxxx");
  snprintf(buf3, sizeof(buf3), "xxxxxxxxxxxxxxxxxxxx");

  // Import some new nodes using opts.
  ImportGraphDefResults results;
  ExpectOK(
      R"EOF(
      node { name: 'new_input' op: 'TestInput' }
      node { name: 't1' op: 'TestMul' input: [ 'new_input:0', 'new_input:1' ] }
      )EOF",
      opts, &refiner, &results);

  EXPECT_TRUE(HasNode("input"));
  EXPECT_TRUE(HasNode("new_input"));
  EXPECT_TRUE(HasNode("t1"));

  EXPECT_TRUE(HasEdge("input", 0, "t1", 0));
  EXPECT_TRUE(HasEdge("new_input", 1, "t1", 1));

  ASSERT_EQ(results.return_tensors.size(), 1);
  EXPECT_EQ(results.return_tensors[0].first->name(), "t1");
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

// Confirms that graph def version in the graph reaches the shape inference
// function.
TEST_F(GraphConstructorTest, GraphDefVersionUsedForShapeInference) {
  string gdef_ascii = strings::StrCat(R"EOF(
      node{ name:"A" op:"RequiresCurrentGraphVersion" }
      versions { producer: )EOF",
                                      TF_GRAPH_DEF_VERSION - 1, "}");
  ImportGraphDefOptions opts;
  ExpectError(gdef_ascii, opts, {"Wrong graph version for shape"});
  gdef_ascii = strings::StrCat(R"EOF(
      node{ name:"A" op:"RequiresCurrentGraphVersion" }
      versions { producer: )EOF",
                               TF_GRAPH_DEF_VERSION, "}");
  ExpectOK(gdef_ascii, opts);
}

TEST_F(GraphConstructorTest, GraphDefVersionMergingDuringImport) {
  ImportGraphDefOptions opts;
  ExpectOK(
      "versions { producer: 15 min_consumer: 5 bad_consumers: 2 bad_consumers: "
      "3 "
      "}",
      opts);
  EXPECT_EQ(15, graph_.versions().producer());
  EXPECT_EQ(5, graph_.versions().min_consumer());
  ASSERT_EQ(2, graph_.versions().bad_consumers_size());
  EXPECT_EQ(2, graph_.versions().bad_consumers(0));
  EXPECT_EQ(3, graph_.versions().bad_consumers(1));

  ExpectOK(
      "versions { producer: 10 min_consumer: 8 bad_consumers: 1 bad_consumers: "
      "3 "
      "}",
      opts);
  EXPECT_EQ(10, graph_.versions().producer());
  EXPECT_EQ(8, graph_.versions().min_consumer());
  ASSERT_EQ(3, graph_.versions().bad_consumers_size());
  EXPECT_EQ(1, graph_.versions().bad_consumers(0));
  EXPECT_EQ(2, graph_.versions().bad_consumers(1));
  EXPECT_EQ(3, graph_.versions().bad_consumers(2));

  // This one is a no-op.
  ExpectOK("versions { producer: 20 min_consumer: 7 }", opts);
  EXPECT_EQ(10, graph_.versions().producer());
  EXPECT_EQ(8, graph_.versions().min_consumer());
  ASSERT_EQ(3, graph_.versions().bad_consumers_size());
  EXPECT_EQ(1, graph_.versions().bad_consumers(0));
  EXPECT_EQ(2, graph_.versions().bad_consumers(1));
  EXPECT_EQ(3, graph_.versions().bad_consumers(2));
}

TEST_F(GraphConstructorTest, ImportGraphDefProvidedShapeRefinerVersions) {
  ImportGraphDefOptions opts;
  // A valid graph at producer version 20, but one
  // that would not import if the graph_def_version were 21.
  string gdef_ascii;
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  gdef_ascii = strings::StrCat(R"EOF(
node {
  name: "Sum/input"
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
          dim {
            size: 2
          }
          dim {
            size: 1
          }
        }
        tensor_content: "\000\000\000\001\000\000\000\002"
      }
    }
  }
}
node {
  name: "Sum/reduction_indices"
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
          dim {
            size: 2
          }
          dim {
            size: 1
          }
        }
        tensor_content: "\000\000\000\000\000\000\000\001"
      }
    }
  }
}
node {
  name: "Sum"
  op: "Sum"
  input: "Sum/input"
  input: "Sum/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
versions {
  producer: 20
})EOF");

#else
  gdef_ascii = strings::StrCat(R"EOF(
node {
  name: "Sum/input"
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
          dim {
            size: 2
          }
          dim {
            size: 1
          }
        }
        tensor_content: "\001\000\000\000\002\000\000\000"
      }
    }
  }
}
node {
  name: "Sum/reduction_indices"
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
          dim {
            size: 2
          }
          dim {
            size: 1
          }
        }
        tensor_content: "\000\000\000\000\001\000\000\000"
      }
    }
  }
}
node {
  name: "Sum"
  op: "Sum"
  input: "Sum/input"
  input: "Sum/reduction_indices"
  attr {
    key: "T"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "Tidx"
    value {
      type: DT_INT32
    }
  }
  attr {
    key: "keep_dims"
    value {
      b: false
    }
  }
}
versions {
  producer: 20
})EOF");
#endif
  // Create a shape refiner with the latest TF_GRAPH_DEF_VERSION.
  // Importing the graphdef with an existing refiner should
  // make the refiner inherit the graphdef version from the
  // passed in graphdef since it has a lower producer.
  ShapeRefiner refiner(TF_GRAPH_DEF_VERSION, graph_.op_registry());
  ExpectOK(gdef_ascii, opts, &refiner);

  // Add another node with a higher producer
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  gdef_ascii = strings::StrCat(R"EOF(
node {
  name: "RandomConst"
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
          dim {
            size: 2
          }
          dim {
            size: 1
          }
        }
        tensor_content: "\000\000\000\001\000\000\000\002"
      }
    }
  }
}
versions {
  producer: 21
})EOF");

#else
  gdef_ascii = strings::StrCat(R"EOF(
node {
  name: "RandomConst"
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
          dim {
            size: 2
          }
          dim {
            size: 1
          }
        }
        tensor_content: "\001\000\000\000\002\000\000\000"
      }
    }
  }
}
versions {
  producer: 21
})EOF");
#endif

  ExpectOK(gdef_ascii, opts, &refiner);
  // Check that the refiner's graph def version is the lowest of
  // the graph defs we have seen so far.
  EXPECT_EQ(20, refiner.graph_def_version());

  // Add another node with a lower producer
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
  gdef_ascii = strings::StrCat(R"EOF(
node {
  name: "RandomConst2"
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
          dim {
            size: 2
          }
          dim {
            size: 1
          }
        }
        tensor_content: "\000\000\000\001\000\000\000\002"
      }
    }
  }
}
versions {
  producer: 17
})EOF");

#else
  gdef_ascii = strings::StrCat(R"EOF(
node {
  name: "RandomConst2"
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
          dim {
            size: 2
          }
          dim {
            size: 1
          }
        }
        tensor_content: "\001\000\000\000\002\000\000\000"
      }
    }
  }
}
versions {
  producer: 17
})EOF");
#endif
  ExpectOK(gdef_ascii, opts, &refiner);

  // Check that the refiner's graph def version is the lowest of
  // the graph defs we have seen so far.
  EXPECT_EQ(17, refiner.graph_def_version());
}

TEST_F(GraphConstructorTest, ImportGraphDef_ValidateColationConstraints) {
  GraphDef def;
  ASSERT_TRUE(protobuf::TextFormat::ParseFromString(
      "node { name: 'A' op: 'TestInput' attr { key: '_class' value { list { "
      "s:'loc:@missing' } } } }",
      &def));
  ImportGraphDefOptions options;
  // TODO(yaozhang): Extend ExpectError to check error type and use ExpectError
  // and ExpectOK to replace the code below.
  Status s = ImportGraphDef(options, def, &graph_, nullptr);
  EXPECT_TRUE(errors::IsInvalidArgument(s)) << s;
  options.validate_colocation_constraints = false;
  TF_EXPECT_OK(ImportGraphDef(options, def, &graph_, nullptr));
}

TEST_F(GraphConstructorTest, ImportGraphDef_UnknownOps) {
  const string pb_ascii = "node { name: 'op_from_contrib' op: 'OpFromContrib'}";
  // Try load twice to check for two parts of the error message. We cannot check
  // for the whole thing in one go because the message includes the hostname.
  ExpectError(pb_ascii, {"Op type not registered 'OpFromContrib'"});
  ExpectError(
      pb_ascii,
      {"Make sure the Op and Kernel are registered in the "
       "binary running in this process. Note that if you "
       "are loading a saved graph which used ops from "
       "tf.contrib, accessing (e.g.) `tf.contrib.resampler` should be done "
       "before importing the graph, as contrib ops are lazily registered "
       "when the module is first accessed."});
}

}  // namespace
}  // namespace tensorflow
