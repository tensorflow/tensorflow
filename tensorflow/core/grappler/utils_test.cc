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

#include <unistd.h>

#include <limits>
#include <memory>

#include "absl/strings/substitute.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/graph/benchmark_testlib.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/bfloat16.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/platform/types.h"

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
  EXPECT_EQ(NodeName("abc"), "abc");
  EXPECT_EQ(NodeName("^abc"), "abc");
  EXPECT_EQ(NodeName("abc:0"), "abc");
  EXPECT_EQ(NodeName("^abc:0"), "abc");

  EXPECT_EQ(NodeName("abc/def"), "abc/def");
  EXPECT_EQ(NodeName("^abc/def"), "abc/def");
  EXPECT_EQ(NodeName("abc/def:1"), "abc/def");
  EXPECT_EQ(NodeName("^abc/def:1"), "abc/def");

  EXPECT_EQ(NodeName("abc/def0"), "abc/def0");
  EXPECT_EQ(NodeName("^abc/def0"), "abc/def0");
  EXPECT_EQ(NodeName("abc/def0:0"), "abc/def0");
  EXPECT_EQ(NodeName("^abc/def0:0"), "abc/def0");

  EXPECT_EQ(NodeName("abc/def_0"), "abc/def_0");
  EXPECT_EQ(NodeName("^abc/def_0"), "abc/def_0");
  EXPECT_EQ(NodeName("abc/def_0:3"), "abc/def_0");
  EXPECT_EQ(NodeName("^abc/def_0:3"), "abc/def_0");

  EXPECT_EQ(NodeName("^abc/def_0:3214"), "abc/def_0");
}

TEST_F(UtilsTest, NodePosition) {
  EXPECT_EQ(NodePosition("abc:2"), 2);
  EXPECT_EQ(NodePosition("abc:123"), 123);
  EXPECT_EQ(NodePosition("^abc:123"), -1);
  EXPECT_EQ(NodePosition("^abc"), -1);
  EXPECT_EQ(NodePosition(""), 0);
}

TEST_F(UtilsTest, NodePositionIfSameNode) {
  EXPECT_EQ(NodePositionIfSameNode(":123", ""), -2);
  EXPECT_EQ(NodePositionIfSameNode(":", ""), -2);
  EXPECT_EQ(NodePositionIfSameNode("", ""), -2);
  EXPECT_EQ(NodePositionIfSameNode("abc:123", "abc"), 123);
  EXPECT_EQ(NodePositionIfSameNode("^abc", "abc"), -1);
  EXPECT_EQ(NodePositionIfSameNode("^abc:123", "abc"), -1);
  EXPECT_EQ(NodePositionIfSameNode("abc", "xyz"), -2);
  EXPECT_EQ(NodePositionIfSameNode("abc", "abc/xyz"), -2);
  EXPECT_EQ(NodePositionIfSameNode("abc/xyz", "abc"), -2);
  EXPECT_EQ(NodePositionIfSameNode("abc:123", "xyz"), -2);
  EXPECT_EQ(NodePositionIfSameNode("^abc", "xyz"), -2);
  EXPECT_EQ(NodePositionIfSameNode("^abc:123", "xyz"), -2);
}

TEST_F(UtilsTest, AddNodeNamePrefix) {
  EXPECT_EQ(AddPrefixToNodeName("abc", "OPTIMIZED"), "OPTIMIZED/abc");
  EXPECT_EQ(AddPrefixToNodeName("^abc", "OPTIMIZED"), "^OPTIMIZED/abc");
  EXPECT_EQ(AddPrefixToNodeName("", "OPTIMIZED"), "OPTIMIZED/");
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
  ASSERT_TRUE(ExecuteWithTimeout(
      []() { Env::Default()->SleepForMicroseconds(1000000); },
      0 /* timeout_in_ms */, thread_pool.get()));

  // Deleting before local variables go off the stack.
  thread_pool.reset();
}

TEST_F(UtilsTest, NumOutputs) {
  GraphDef graph;
  EXPECT_EQ(NumOutputs(CreateConcatOffsetNode(), &graph), 2);
  EXPECT_EQ(NumOutputs(CreateFusedBatchNormNode(), &graph), 5);
  EXPECT_EQ(NumOutputs(CreateDequeueNode(), &graph), 1);
}

TEST_F(UtilsTest, AsControlDependency) {
  NodeDef node;
  node.set_name("foo");
  EXPECT_EQ(AsControlDependency(node), "^foo");
  EXPECT_EQ(AsControlDependency(node.name()), "^foo");
  EXPECT_EQ(AsControlDependency("^foo"), "^foo");
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

  ASSERT_EQ(graph.node_size(), 8);
  ASSERT_EQ(graph.node(0).name(), "c0");
  ASSERT_EQ(graph.node(1).name(), "c1");
  ASSERT_EQ(graph.node(2).name(), "neg0");
  ASSERT_EQ(graph.node(3).name(), "neg1");
  ASSERT_EQ(graph.node(4).name(), "neg2");
  ASSERT_EQ(graph.node(5).name(), "id1");
  ASSERT_EQ(graph.node(6).name(), "id2");
  ASSERT_EQ(graph.node(7).name(), "noop");

  NodeMap node_map(&graph);
  auto is_neg = [&](const NodeDef& node) { return node.op() == "Neg"; };
  // We walk backwards, starting as "id1", so tail should be "neg1".
  NodeDef* tail = GetTailOfChain(graph.node(5), node_map,
                                 /*follow_control_input=*/false, is_neg);
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(tail->name(), "neg1");

  // We stop at branching nodes, so tail should be "neg2".
  auto is_neg_and_non_branching = [&](const NodeDef& node) {
    return node.op() == "Neg" && NumNonControlOutputs(node, node_map) == 1;
  };
  tail =
      GetTailOfChain(graph.node(5), node_map,
                     /*follow_control_input=*/false, is_neg_and_non_branching);
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(tail->name(), "neg2");

  // We walk backwards, starting from "noop", also following control inputs,
  // so tail should be "neg0".
  tail = GetTailOfChain(graph.node(7), node_map,
                        /*follow_control_input=*/true, is_neg);
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(tail->name(), "neg0");

  // We walk backwards, starting from "noop", not following control inputs,
  // so tail should be "noop" itself.
  tail = GetTailOfChain(graph.node(7), node_map,
                        /*follow_control_input=*/false, is_neg);
  ASSERT_NE(tail, nullptr);
  EXPECT_EQ(tail->name(), "noop");
}

TEST_F(UtilsTest, DedupControlInputs) {
  NodeDef foo;
  foo.set_name("foo");
  foo.add_input("bar");
  DedupControlInputs(&foo);
  ASSERT_EQ(foo.input_size(), 1);
  EXPECT_EQ(foo.input(0), "bar");

  foo.set_input(0, "^bar");
  DedupControlInputs(&foo);
  ASSERT_EQ(foo.input_size(), 1);
  EXPECT_EQ(foo.input(0), "^bar");

  foo.set_input(0, "bar");
  foo.add_input("bar");
  DedupControlInputs(&foo);
  ASSERT_EQ(foo.input_size(), 2);
  EXPECT_EQ(foo.input(0), "bar");
  EXPECT_EQ(foo.input(1), "bar");

  foo.set_input(1, "^bar");
  DedupControlInputs(&foo);
  ASSERT_EQ(foo.input_size(), 1);
  EXPECT_EQ(foo.input(0), "bar");

  foo.set_input(0, "^bar");
  foo.add_input("^bar");
  DedupControlInputs(&foo);
  ASSERT_EQ(foo.input_size(), 1);
  EXPECT_EQ(foo.input(0), "^bar");

  foo.set_input(0, "bar");
  foo.add_input("gnu");
  foo.add_input("^bar");
  foo.add_input("^gnu");
  DedupControlInputs(&foo);
  ASSERT_EQ(foo.input_size(), 2);
  EXPECT_EQ(foo.input(0), "bar");
  EXPECT_EQ(foo.input(1), "gnu");
}

TEST_F(UtilsTest, NumNonControlOutputs) {
  tensorflow::Scope s = tensorflow::Scope::NewRootScope();

  //   *Round    [Sqrt, Shape]
  //      |           |
  //      |   ctrl    |
  //     Mul ------> Add
  //     / \         / \
  //    x   y       a   b
  //
  //  *) Round node has control dependency edge from Add, which
  //     is not on this scheme (ASCII graphics limitation).
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

  {
    NodeMap node_map(&graph);

    const NodeDef* add_node = node_map.GetNode("add");
    const NodeDef* mul_node = node_map.GetNode("mul");
    ASSERT_NE(add_node, nullptr);

    // [a, b] are only non-control inputs
    EXPECT_EQ(NumNonControlInputs(*add_node), 2);
    EXPECT_EQ(NumControlInputs(*add_node), 1);
    // [sqrt, shape] are non control outputs
    EXPECT_EQ(NumNonControlOutputs(*add_node, node_map), 2);
    // sqrt is the only data output
    EXPECT_EQ(NumNonControlDataOutputs(*add_node, node_map), 1);
    EXPECT_EQ(NumControlInputs(*mul_node), 0);

    EXPECT_TRUE(HasControlInputs(*add_node));
    EXPECT_TRUE(HasRegularInputs(*add_node));
    EXPECT_TRUE(HasControlOutputs(*add_node, node_map));
    EXPECT_TRUE(HasRegularOutputs(*add_node, node_map));

    const NodeDef* x_node = node_map.GetNode("x");
    ASSERT_NE(x_node, nullptr);
    EXPECT_FALSE(HasControlInputs(*x_node));
    EXPECT_FALSE(HasRegularInputs(*x_node));
    EXPECT_FALSE(HasControlOutputs(*x_node, node_map));
    EXPECT_TRUE(HasRegularOutputs(*x_node, node_map));

    const NodeDef* round_node = node_map.GetNode("round");
    ASSERT_NE(round_node, nullptr);
    EXPECT_TRUE(HasControlInputs(*round_node));
    EXPECT_TRUE(HasRegularInputs(*round_node));
    EXPECT_FALSE(HasControlOutputs(*round_node, node_map));
    EXPECT_FALSE(HasRegularOutputs(*round_node, node_map));
  }

  {
    // Similar test for ImmutableNodeMap.
    ImmutableNodeMap node_map(&graph);

    const NodeDef* add_node = node_map.GetNode("add");
    const NodeDef* mul_node = node_map.GetNode("mul");
    ASSERT_NE(add_node, nullptr);

    // [a, b] are only non-control inputs
    EXPECT_EQ(NumNonControlInputs(*add_node), 2);
    EXPECT_EQ(NumControlInputs(*add_node), 1);
    EXPECT_EQ(NumControlInputs(*mul_node), 0);

    EXPECT_TRUE(HasControlInputs(*add_node));
    EXPECT_TRUE(HasRegularInputs(*add_node));

    const NodeDef* x_node = node_map.GetNode("x");
    ASSERT_NE(x_node, nullptr);
    EXPECT_FALSE(HasControlInputs(*x_node));
    EXPECT_FALSE(HasRegularInputs(*x_node));

    const NodeDef* round_node = node_map.GetNode("round");
    ASSERT_NE(round_node, nullptr);
    EXPECT_TRUE(HasControlInputs(*round_node));
    EXPECT_TRUE(HasRegularInputs(*round_node));
  }
}

TEST(CheckAttrExists, All) {
  NodeDef node;
  node.set_name("node");
  (*node.mutable_attr())["apple"].set_i(7);
  (*node.mutable_attr())["pear"].set_b(true);

  TF_EXPECT_OK(CheckAttrExists(node, "apple"));
  TF_EXPECT_OK(CheckAttrExists(node, "pear"));

  TF_EXPECT_OK(CheckAttrsExist(node, {}));
  TF_EXPECT_OK(CheckAttrsExist(node, {"apple"}));
  TF_EXPECT_OK(CheckAttrsExist(node, {"pear"}));
  TF_EXPECT_OK(CheckAttrsExist(node, {"apple", "pear"}));
  TF_EXPECT_OK(CheckAttrsExist(node, {"pear", "apple"}));

  Status status = CheckAttrExists(node, "banana");
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.ToString(),
            "Invalid argument: Node 'node' lacks 'banana' attr: name: \"node\" "
            "attr { key: \"apple\" value { i: 7 } } attr { key: \"pear\" value "
            "{ b: true } }");
  EXPECT_FALSE(CheckAttrsExist(node, {""}).ok());
  EXPECT_FALSE(CheckAttrsExist(node, {"pear", "cherry"}).ok());
  EXPECT_FALSE(CheckAttrsExist(node, {"banana", "apple"}).ok());
}

TEST_F(UtilsTest, DeleteNodes) {
  // TODO(rmlarsen): write forgotten test.
}

TEST(IsKernelRegisteredForNode, All) {
  NodeDef node;
  node.set_name("foo");
  node.set_op("MatMul");
  node.set_device("/cpu:0");
  AttrValue v;
  v.set_type(DataType::DT_FLOAT);
  (*node.mutable_attr())["T"] = v;
  TF_EXPECT_OK(IsKernelRegisteredForNode(node));
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  node.set_device("/gpu:0");
  TF_EXPECT_OK(IsKernelRegisteredForNode(node));
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

  // Bad device name.
  node.set_device("");
  EXPECT_FALSE(IsKernelRegisteredForNode(node).ok());

  // Check an op that is only defined on CPU.
  node.set_op("MatchingFiles");
  node.set_device("/cpu:0");
  TF_EXPECT_OK(IsKernelRegisteredForNode(node));
  node.set_device("/gpu:0");
  EXPECT_FALSE(IsKernelRegisteredForNode(node).ok());
}

#define BM_NodePositionIfSameNode(I, N, NAME)              \
  static void BM_NodePositionIfSameNode_##NAME(            \
      ::testing::benchmark::State& state) {                \
    string input = I;                                      \
    string node = N;                                       \
    for (auto s : state) {                                 \
      const int pos = NodePositionIfSameNode(input, node); \
      CHECK_GT(pos, -3);                                   \
    }                                                      \
  }                                                        \
  BENCHMARK(BM_NodePositionIfSameNode_##NAME)

BM_NodePositionIfSameNode("foo/bar/baz:7", "foo/bar/baz", Match_7);
BM_NodePositionIfSameNode("foo/bar/baz", "foo/bar/baz", Match_0);
BM_NodePositionIfSameNode("^foo/bar/baz", "foo/bar/baz", Match_Ctrl);
BM_NodePositionIfSameNode("blah", "foo/bar/baz", NoMatch_0);
BM_NodePositionIfSameNode("foo/bar/baz/gnu", "foo/bar/baz", NoMatch_end);

void BM_NodeNameAsStringPiece(::testing::benchmark::State& state) {
  const int size = state.range(0);

  string input(size + 3, 'x');
  input[size] = ':';
  for (auto s : state) {
    StringPiece node_name = NodeNameAsStringPiece(input);
    CHECK_GT(node_name.size(), 0);
  }
}
BENCHMARK(BM_NodeNameAsStringPiece)->Range(1, 1024);

#define BM_ParseNodeNameAsStringPiece(I, NAME)                               \
  static void BM_ParseNodeNameAsStringPiece_##NAME(                          \
      ::testing::benchmark::State& state) {                                  \
    string input = I;                                                        \
    for (auto s : state) {                                                   \
      int position;                                                          \
      const StringPiece name = ParseNodeNameAsStringPiece(input, &position); \
      CHECK_GE(position, -1);                                                \
      CHECK(!name.empty());                                                  \
    }                                                                        \
  }                                                                          \
  BENCHMARK(BM_ParseNodeNameAsStringPiece_##NAME)

BM_ParseNodeNameAsStringPiece("foo", foo);
BM_ParseNodeNameAsStringPiece("foo/bar/baz", foo_bar_baz);
BM_ParseNodeNameAsStringPiece("^foo/bar/baz", foo_bar_baz_ctrl);
BM_ParseNodeNameAsStringPiece("foo:123", foo123);
BM_ParseNodeNameAsStringPiece("foo/bar/baz:123", foo_bar_baz_123);
BM_ParseNodeNameAsStringPiece("^foo/bar/baz:123", foo_bar_baz_123_ctrl);

TEST_F(UtilsTest, SetTensorValueBFloat16) {
  Tensor t(DT_BFLOAT16, TensorShape({}));
  TF_ASSERT_OK(SetTensorValue(t.dtype(), 2, &t));
  test::ExpectTensorEqual<bfloat16>(Tensor(bfloat16(2)), t);
}

TEST_F(UtilsTest, SetTensorValueBFloat16IntMax) {
  Tensor t(DT_BFLOAT16, TensorShape({}));
  TF_ASSERT_OK(SetTensorValue(t.dtype(), std::numeric_limits<int>::max(), &t));
  test::ExpectTensorEqual<bfloat16>(
      Tensor(bfloat16(std::numeric_limits<int>::max())), t);
}

TEST_F(UtilsTest, SetTensorValueBFloat16IntMin) {
  Tensor t(DT_BFLOAT16, TensorShape({}));
  TF_ASSERT_OK(SetTensorValue(t.dtype(), std::numeric_limits<int>::min(), &t));
  test::ExpectTensorEqual<bfloat16>(
      Tensor(bfloat16(std::numeric_limits<int>::min())), t);
}

TEST_F(UtilsTest, TensorIdToString) {
  EXPECT_EQ(TensorIdToString({"foo", -1}), "^foo");
  EXPECT_EQ(TensorIdToString({"foo", 0}), "foo");
  EXPECT_EQ(TensorIdToString({"foo", 1}), "foo:1");
  EXPECT_EQ(TensorIdToString({"foo", 2}), "foo:2");
}

TEST_F(UtilsTest, SafeTensorIdToString) {
  EXPECT_EQ(SafeTensorIdToString({"foo", -1}), "^foo");
  EXPECT_EQ(SafeTensorIdToString({"foo", 0}), "foo");
  EXPECT_EQ(SafeTensorIdToString({"foo", 1}), "foo:1");
  EXPECT_EQ(SafeTensorIdToString({"foo", 2}), "foo:2");
}

TEST_F(UtilsTest, EraseRegularNodeAttributes) {
  NodeDef node;
  AttrValue dummy;
  node.set_name("foo");
  node.set_op("MatMul");
  (*node.mutable_attr())["baz"] = dummy;
  EXPECT_EQ(EraseRegularNodeAttributes(&node), 1);
  EXPECT_EQ(node.attr_size(), 0);
  EXPECT_EQ(EraseRegularNodeAttributes(&node), 0);

  (*node.mutable_attr())["baz"] = dummy;
  (*node.mutable_attr())["_bar"] = dummy;
  EXPECT_EQ(EraseRegularNodeAttributes(&node), 1);
  EXPECT_EQ(node.attr_size(), 1);
  EXPECT_EQ(node.attr().begin()->first, "_bar");
  EXPECT_EQ(EraseRegularNodeAttributes(&node), 0);
}

TEST_F(UtilsTest, EraseNodeOutputAttributes) {
  NodeDef node;
  AttrValue dummy;
  node.set_name("foo");
  node.set_op("MatMul");
  EXPECT_EQ(EraseNodeOutputAttributes(&node), 0);
  (*node.mutable_attr())["_xla_inferred_shapes"] = dummy;
  EXPECT_EQ(EraseNodeOutputAttributes(&node), 1);
  EXPECT_EQ(node.attr_size(), 0);
  EXPECT_EQ(EraseNodeOutputAttributes(&node), 0);

  (*node.mutable_attr())["baz"] = dummy;
  (*node.mutable_attr())["_output_shapes"] = dummy;
  (*node.mutable_attr())["_xla_inferred_shapes"] = dummy;
  (*node.mutable_attr())["_output_gnu"] = dummy;
  EXPECT_EQ(EraseNodeOutputAttributes(&node), 3);
  EXPECT_EQ(node.attr_size(), 1);
  EXPECT_EQ(node.attr().begin()->first, "baz");
  EXPECT_EQ(EraseNodeOutputAttributes(&node), 0);
}

template <typename T>
void TestSetTensorValue(DataType type, int val, bool success,
                        absl::string_view error_msg) {
  Tensor t(type, TensorShape({}));
  Status s = SetTensorValue(t.dtype(), val, &t);
  EXPECT_EQ(s.ok(), success);
  if (s.ok()) {
    test::ExpectTensorEqual<T>(Tensor(static_cast<T>(val)), t);
  } else {
    EXPECT_EQ(s.error_message(), error_msg);
  }
}

TEST(SetTensorValueTest, Quantized) {
  auto int_min_error = [](DataType type) {
    return absl::Substitute(
        "Cannot store value -2147483648 in tensor of type $0",
        DataType_Name(type));
  };
  auto int_max_error = [](DataType type) {
    return absl::Substitute(
        "Cannot store value 2147483647 in tensor of type $0",
        DataType_Name(type));
  };
  const int kMinInt = std::numeric_limits<int>::min();
  const int kMaxInt = std::numeric_limits<int>::max();

  TestSetTensorValue<qint8>(DT_QINT8, -8, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint8>(DT_QINT8, 0, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint8>(DT_QINT8, 8, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint8>(DT_QINT8, std::numeric_limits<qint8>::min(),
                            /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint8>(DT_QINT8, std::numeric_limits<qint8>::max(),
                            /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint8>(DT_QINT8, kMinInt, /*success=*/false,
                            int_min_error(DT_QINT8));
  TestSetTensorValue<qint8>(DT_QINT8, kMaxInt, /*success=*/false,
                            int_max_error(DT_QINT8));

  TestSetTensorValue<quint8>(
      DT_QUINT8, -8, /*success=*/false,
      /*error_msg=*/"Cannot store value -8 in tensor of type DT_QUINT8");
  TestSetTensorValue<quint8>(DT_QUINT8, 0, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<quint8>(DT_QUINT8, 8, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<quint8>(DT_QUINT8, std::numeric_limits<quint8>::min(),
                             /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<quint8>(DT_QUINT8, std::numeric_limits<quint8>::max(),
                             /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<quint8>(DT_QUINT8, kMinInt, /*success=*/false,
                             int_min_error(DT_QUINT8));
  TestSetTensorValue<quint8>(DT_QUINT8, kMaxInt, /*success=*/false,
                             int_max_error(DT_QUINT8));

  TestSetTensorValue<qint16>(DT_QINT16, -8, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint16>(DT_QINT16, 0, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint16>(DT_QINT16, 8, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint16>(DT_QINT16, std::numeric_limits<qint16>::min(),
                             /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint16>(DT_QINT16, std::numeric_limits<qint16>::max(),
                             /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint16>(DT_QINT16, kMinInt, /*success=*/false,
                             int_min_error(DT_QINT16));
  TestSetTensorValue<qint16>(DT_QINT16, kMaxInt, /*success=*/false,
                             int_max_error(DT_QINT16));

  TestSetTensorValue<quint16>(
      DT_QUINT16, -8, /*success=*/false,
      /*error_msg=*/"Cannot store value -8 in tensor of type DT_QUINT16");
  TestSetTensorValue<quint16>(DT_QUINT16, 0, /*success=*/true,
                              /*error_msg=*/"");
  TestSetTensorValue<quint16>(DT_QUINT16, 8, /*success=*/true,
                              /*error_msg=*/"");
  TestSetTensorValue<quint16>(DT_QUINT16, std::numeric_limits<quint16>::min(),
                              /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<quint16>(DT_QUINT16, std::numeric_limits<quint16>::max(),
                              /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<quint16>(DT_QUINT16, kMinInt, /*success=*/false,
                              int_min_error(DT_QUINT16));
  TestSetTensorValue<quint16>(DT_QUINT16, kMaxInt, /*success=*/false,
                              int_max_error(DT_QUINT16));

  TestSetTensorValue<qint32>(DT_QINT32, -8, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint32>(DT_QINT32, 0, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint32>(DT_QINT32, 8, /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint32>(DT_QINT32, std::numeric_limits<qint32>::min(),
                             /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint32>(DT_QINT32, std::numeric_limits<qint32>::max(),
                             /*success=*/true, /*error_msg=*/"");
  TestSetTensorValue<qint32>(DT_QINT32, kMinInt, /*success=*/true,
                             /*error_msg=*/"");
  TestSetTensorValue<qint32>(DT_QINT32, kMaxInt, /*success=*/true,
                             /*error_msg=*/"");
}

void BM_NodeMapConstruct(::testing::benchmark::State& state) {
  const int size = state.range(0);

  GraphDef graph = test::CreateRandomGraph(size);
  for (auto s : state) {
    NodeMap node_map(&graph);
  }
}
BENCHMARK(BM_NodeMapConstruct)->Range(1, 1 << 20);

void BM_ImmutableNodeMapConstruct(::testing::benchmark::State& state) {
  const int size = state.range(0);

  GraphDef graph = test::CreateRandomGraph(size);
  for (auto s : state) {
    ImmutableNodeMap node_map(&graph);
  }
}
BENCHMARK(BM_ImmutableNodeMapConstruct)->Range(1, 1 << 20);

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
