/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

class UtilsTest : public ::testing::Test {
 protected:
  NodeDef CreateConcatOffsetNode() const {
    const string gdef_ascii =
        " name: 'gradients/InceptionV3/Mixed_7c/Branch_1/concat_v2_grad/"
        "ConcatOffset'"
        " op: 'ConcatOffset'"
        " input: 'InceptionV3/Mixed_7c/Branch_1/concat_v2/axis'"
        " input: 'gradients/InceptionV3/Mixed_7c/Branch_1/concat_v2_grad/Shape'"
        " input: "
        " 'gradients/InceptionV3/Mixed_7c/Branch_1/concat_v2_grad/Shape_1'"
        " attr {"
        "  key: 'N'"
        "  value {"
        "    i: 2"
        "  }"
        " }";
    NodeDef node;
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &node));
    return node;
  }

  NodeDef CreateDequeueNode() const {
    const string gdef_ascii =
        " name: 'Train/TrainInput/input_producer_Dequeue'"
        " op: 'QueueDequeueV2'"
        " input: 'Train/TrainInput/input_producer'"
        " attr {"
        "  key: 'component_types'"
        "   value {"
        "     list {"
        "       type: DT_INT32"
        "     }"
        "   }"
        " }"
        " attr {"
        "   key: 'timeout_ms'"
        "   value {"
        "     i: -1"
        "   }"
        " }";

    NodeDef node;
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &node));
    return node;
  }

  NodeDef CreateFusedBatchNormNode() const {
    const string gdef_ascii =
        " name: 'InceptionV3/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm'"
        " op: 'FusedBatchNorm'"
        " input: 'InceptionV3/Conv2d_1a_3x3/BatchNorm/FusedBatchNorm'"
        " input: 'InceptionV3/Conv2d_1a_3x3/BatchNorm/gamma/read'"
        " input: 'InceptionV3/Conv2d_1a_3x3/BatchNorm/beta/read'"
        " input: 'InceptionV3/Conv2d_1a_3x3/BatchNorm/Const'"
        " input: 'InceptionV3/Conv2d_1a_3x3/BatchNorm/Const_1'"
        " attr {"
        "   key: 'T'"
        "   value {"
        "     type: DT_FLOAT"
        "   }"
        " }"
        " attr {"
        "   key: 'data_format'"
        "   value {"
        "     s: 'NHWC'"
        "   }"
        " }"
        " attr {"
        "   key: 'epsilon'"
        "   value {"
        "     f: 0.001"
        "   }"
        " }"
        " attr {"
        "   key: 'is_training'"
        "   value {"
        "     b: true"
        "   }"
        " }";

    NodeDef node;
    CHECK(protobuf::TextFormat::ParseFromString(gdef_ascii, &node));
    return node;
  }
};

TEST_F(UtilsTest, NodeName) {
  EXPECT_EQ("abc", NodeName("abc"));
  EXPECT_EQ("abc", NodeName("^abc"));
  EXPECT_EQ("abc", NodeName("abc:0"));
  EXPECT_EQ("abc", NodeName("^abc:0"));

  EXPECT_EQ("abc/def", NodeName("abc/def"));
  EXPECT_EQ("abc/def", NodeName("^abc/def"));
  EXPECT_EQ("abc/def", NodeName("abc/def:1"));
  EXPECT_EQ("abc/def", NodeName("^abc/def:1"));

  EXPECT_EQ("abc/def0", NodeName("abc/def0"));
  EXPECT_EQ("abc/def0", NodeName("^abc/def0"));
  EXPECT_EQ("abc/def0", NodeName("abc/def0:0"));
  EXPECT_EQ("abc/def0", NodeName("^abc/def0:0"));

  EXPECT_EQ("abc/def_0", NodeName("abc/def_0"));
  EXPECT_EQ("abc/def_0", NodeName("^abc/def_0"));
  EXPECT_EQ("abc/def_0", NodeName("abc/def_0:3"));
  EXPECT_EQ("abc/def_0", NodeName("^abc/def_0:3"));

  EXPECT_EQ("abc/def_0", NodeName("^abc/def_0:3214"));
}

TEST_F(UtilsTest, NodePosition) {
  EXPECT_EQ(2, NodePosition("abc:2"));
  EXPECT_EQ(123, NodePosition("abc:123"));
  EXPECT_EQ(-1, NodePosition("^abc:123"));
  EXPECT_EQ(-1, NodePosition("^abc"));
  EXPECT_EQ(0, NodePosition(""));
}

TEST_F(UtilsTest, AddNodeNamePrefix) {
  EXPECT_EQ("OPTIMIZED/abc", AddPrefixToNodeName("abc", "OPTIMIZED"));
  EXPECT_EQ("^OPTIMIZED/abc", AddPrefixToNodeName("^abc", "OPTIMIZED"));
  EXPECT_EQ("OPTIMIZED/", AddPrefixToNodeName("", "OPTIMIZED"));
}

TEST_F(UtilsTest, ExecuteWithTimeout) {
  std::unique_ptr<thread::ThreadPool> thread_pool(
      new thread::ThreadPool(Env::Default(), "ExecuteWithTimeout", 2));

  // This should run till the end.
  ASSERT_TRUE(ExecuteWithTimeout(
      []() {  // Do nothing.
      },
      1000 /* timeout_in_ms */, thread_pool.get()));

  // This should time out.
  Notification notification;
  ASSERT_FALSE(ExecuteWithTimeout(
      [&notification]() { notification.WaitForNotification(); },
      1 /* timeout_in_ms */, thread_pool.get()));
  // Make sure to unblock the thread.
  notification.Notify();

  // This should run till the end.
  ASSERT_TRUE(ExecuteWithTimeout([]() { sleep(1); }, 0 /* timeout_in_ms */,
                                 thread_pool.get()));

  // Deleting before local variables go off the stack.
  thread_pool.reset();
}

TEST_F(UtilsTest, NumOutputs) {
  GraphDef graph;
  EXPECT_EQ(2, NumOutputs(CreateConcatOffsetNode(), &graph));
  EXPECT_EQ(5, NumOutputs(CreateFusedBatchNormNode(), &graph));
  EXPECT_EQ(1, NumOutputs(CreateDequeueNode(), &graph));
}

TEST_F(UtilsTest, AsControlDependency) {
  NodeDef node;
  node.set_name("foo");
  EXPECT_EQ("^foo", AsControlDependency(node));
  EXPECT_EQ("^foo", AsControlDependency(node.name()));
  EXPECT_EQ("^foo", AsControlDependency("^foo"));
}

TEST_F(UtilsTest, GetTailOfChain) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();
  Output c0 = ops::Const(s.WithOpName("c0"), {1.0f, 2.0f}, {1, 2});
  Output c1 = ops::Const(s.WithOpName("c1"), {3.0f, 4.0f}, {1, 2});
  // Add a node with only connected by control output.
  Output neg0 = ops::Neg(s.WithOpName("neg0"), c1);
  // Add a node with two outputs.
  Output neg1 =
      ops::Neg(s.WithControlDependencies(neg0).WithOpName("neg1"), c0);
  Output neg2 = ops::Neg(s.WithOpName("neg2"), neg1);
  Output id1 = ops::Identity(s.WithOpName("id1"), neg2);
  Output id2 = ops::Identity(s.WithOpName("id2"), neg1);
  auto noop = ops::NoOp(s.WithControlDependencies(neg0).WithOpName("noop"));
  GraphDef graph;
  TF_CHECK_OK(s.ToGraphDef(&graph));
  LOG(INFO) << graph.DebugString();

  ASSERT_EQ("c0", graph.node(0).name());
  ASSERT_EQ("c1", graph.node(1).name());
  ASSERT_EQ("neg0", graph.node(2).name());
  ASSERT_EQ("neg1", graph.node(3).name());
  ASSERT_EQ("neg2", graph.node(4).name());
  ASSERT_EQ("id1", graph.node(5).name());
  ASSERT_EQ("id2", graph.node(6).name());
  ASSERT_EQ("noop", graph.node(7).name());

  NodeMap node_map(&graph);
  auto is_neg = [&](const NodeDef& node) { return node.op() == "Neg"; };
  // We walk backwards, starting as "id1", so tail should be "neg1".
  NodeDef* tail = GetTailOfChain(graph.node(5), node_map,
                                 /*follow_control_input=*/false, is_neg);
  EXPECT_NE(tail, nullptr);
  EXPECT_EQ("neg1", tail->name());

  // We stop at branching nodes, so tail should be "neg2".
  auto is_neg_and_non_branching = [&](const NodeDef& node) {
    return node.op() == "Neg" && NumNonControlOutputs(node, node_map) == 1;
  };
  tail =
      GetTailOfChain(graph.node(5), node_map,
                     /*follow_control_input=*/false, is_neg_and_non_branching);
  EXPECT_NE(tail, nullptr);
  EXPECT_EQ("neg2", tail->name());

  // We walk backwards, starting from "noop", also following control inputs,
  // so tail should be "neg0".
  tail = GetTailOfChain(graph.node(7), node_map,
                        /*follow_control_input=*/true, is_neg);
  EXPECT_NE(tail, nullptr);
  EXPECT_EQ("neg0", tail->name());

  // We walk backwards, starting from "noop", not following control inputs,
  // so tail should be "noop" itself.
  tail = GetTailOfChain(graph.node(7), node_map,
                        /*follow_control_input=*/false, is_neg);
  EXPECT_NE(tail, nullptr);
  EXPECT_EQ("noop", tail->name());
}

TEST_F(UtilsTest, DedupControlInputs) {
  NodeDef foo;
  foo.set_name("foo");
  foo.add_input("bar");
  DedupControlInputs(&foo);
  EXPECT_EQ(1, foo.input_size());
  EXPECT_EQ("bar", foo.input(0));

  foo.set_input(0, "^bar");
  DedupControlInputs(&foo);
  EXPECT_EQ(1, foo.input_size());
  EXPECT_EQ("^bar", foo.input(0));

  foo.set_input(0, "bar");
  foo.add_input("bar");
  DedupControlInputs(&foo);
  EXPECT_EQ(2, foo.input_size());
  EXPECT_EQ("bar", foo.input(0));
  EXPECT_EQ("bar", foo.input(1));

  foo.set_input(1, "^bar");
  DedupControlInputs(&foo);
  EXPECT_EQ(1, foo.input_size());
  EXPECT_EQ("bar", foo.input(0));

  foo.set_input(0, "^bar");
  foo.add_input("^bar");
  DedupControlInputs(&foo);
  EXPECT_EQ(1, foo.input_size());
  EXPECT_EQ("^bar", foo.input(0));

  foo.set_input(0, "bar");
  foo.add_input("gnu");
  foo.add_input("^bar");
  foo.add_input("^gnu");
  DedupControlInputs(&foo);
  EXPECT_EQ(2, foo.input_size());
  EXPECT_EQ("bar", foo.input(0));
  EXPECT_EQ("gnu", foo.input(1));
}

TEST_F(UtilsTest, NumNonControlOutputs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  //  *) Round node has control dependency edge from Add, which
  //     is not on this scheme (ASCII graphics limitation).
  //
  //   *Round    [Sqrt, Shape]
  //      |           |
  //      |   ctrl    |
  //     Mul ------> Add
  //     / \         / \
  //    x   y       a   b
  auto x = ops::Variable(s.WithOpName("x"), {1, 2}, DT_FLOAT);
  auto y = ops::Variable(s.WithOpName("y"), {1, 2}, DT_FLOAT);
  auto a = ops::Variable(s.WithOpName("a"), {1, 2}, DT_FLOAT);
  auto b = ops::Variable(s.WithOpName("b"), {1, 2}, DT_FLOAT);

  auto mul = ops::Multiply(s.WithOpName("mul"), x, y);
  auto add = ops::Add(s.WithOpName("add").WithControlDependencies(mul), a, b);

  auto shape = ops::Shape(s.WithOpName("shape"), add);
  auto sqrt = ops::Sqrt(s.WithOpName("sqrt"), add);

  auto round =
      ops::Round(s.WithOpName("round").WithControlDependencies(add), mul);

  GraphDef graph;
  TF_CHECK_OK(s.ToGraphDef(&graph));
  NodeMap node_map(&graph);

  const NodeDef* add_node = node_map.GetNode("add");
  ASSERT_TRUE(add_node != nullptr);

  // [a, b] are only non-control inputs
  EXPECT_EQ(2, NumNonControlInputs(*add_node));
  // [sqrt, shape] are non control outputs
  EXPECT_EQ(2, NumNonControlOutputs(*add_node, node_map));
  // sqrt is the only data output
  EXPECT_EQ(1, NumNonControlDataOutputs(*add_node, node_map));
}

TEST_F(UtilsTest, DeleteNodes) {
  // TODO(rmlarsen): write forgtten test.
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
