#include "tensorflow/core/graph/equal_graph_def.h"

#include <gtest/gtest.h>
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph_def_builder.h"
#include "tensorflow/core/kernels/ops_util.h"

namespace tensorflow {
namespace {

REGISTER_OP("Input").Output("o: float");
REGISTER_OP("Alternate").Output("o: float");
REGISTER_OP("Cross").Input("a: float").Input("b: float").Output("o: float");

Node* Input(const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("Input", opts);
}

Node* Alternate(const GraphDefBuilder::Options& opts) {
  return ops::SourceOp("Alternate", opts);
}

Node* Cross(ops::NodeOut a, ops::NodeOut b,
            const GraphDefBuilder::Options& opts) {
  return ops::BinaryOp("Cross", a, b, opts);
}

class EqualGraphDefTest : public ::testing::Test {
 protected:
  EqualGraphDefTest()
      : e_(GraphDefBuilder::kFailImmediately),
        a_(GraphDefBuilder::kFailImmediately) {
    RequireDefaultOps();
  }

  bool Match() {
    GraphDef expected;
    e_.ToGraphDef(&expected);
    GraphDef actual;
    a_.ToGraphDef(&actual);
    return EqualGraphDef(actual, expected, &diff_);
  }

  GraphDefBuilder e_;
  GraphDefBuilder a_;
  string diff_;
};

TEST_F(EqualGraphDefTest, Match) {
  Input(e_.opts().WithName("A"));
  Input(a_.opts().WithName("A"));
  EXPECT_TRUE(Match()) << diff_;
}

TEST_F(EqualGraphDefTest, NoMatch) {
  Input(e_.opts().WithName("A"));
  Input(a_.opts().WithName("B"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Did not find expected node 'A = Input[]()'", diff_);
}

TEST_F(EqualGraphDefTest, MissingNode) {
  Input(e_.opts().WithName("A"));
  Input(e_.opts().WithName("B"));
  Input(a_.opts().WithName("A"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Did not find expected node 'B = Input[]()'", diff_);
}

TEST_F(EqualGraphDefTest, ExtraNode) {
  Input(e_.opts().WithName("A"));
  Input(a_.opts().WithName("A"));
  Input(a_.opts().WithName("B"));
  EXPECT_FALSE(Match());
  EXPECT_EQ(
      "Found unexpected node 'B = Input[]()' not in expected graph:\n"
      "A = Input[]();\n",
      diff_);
}

TEST_F(EqualGraphDefTest, NodeOrder) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Cross(a, b, e_.opts().WithName("C"));

  b = Input(a_.opts().WithName("B"));
  a = Input(a_.opts().WithName("A"));
  Cross(a, b, a_.opts().WithName("C"));
  EXPECT_TRUE(Match()) << diff_;
}

TEST_F(EqualGraphDefTest, NameMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  // Have to call EqualNodeDef() directly here, since EqualGraphDef()
  // only calls EqualNodeDef() with nodes that have matching names.
  EXPECT_FALSE(EqualNodeDef(a->def(), b->def(), &diff_));
  EXPECT_EQ("Actual node name 'A' is not expected 'B'", diff_);
}

TEST_F(EqualGraphDefTest, OpMismatch) {
  Input(e_.opts().WithName("A"));
  Alternate(a_.opts().WithName("A"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'A' has op 'Alternate' that is not expected 'Input'",
            diff_);
}

TEST_F(EqualGraphDefTest, DeviceMatch) {
  Input(e_.opts().WithName("A").WithDevice("/cpu:0"));
  Input(a_.opts().WithName("A").WithDevice("/cpu:0"));
  EXPECT_TRUE(Match()) << diff_;
}

TEST_F(EqualGraphDefTest, DeviceMismatch) {
  Input(e_.opts().WithName("A").WithDevice("/cpu:0"));
  Input(a_.opts().WithName("A").WithDevice("/cpu:1"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'A' has device '/cpu:1' that is not expected '/cpu:0'",
            diff_);
}

TEST_F(EqualGraphDefTest, InputMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Cross(a, a, e_.opts().WithName("C"));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  Cross(b, b, a_.opts().WithName("C"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'C' has input 0 'B' that doesn't match expected 'A'",
            diff_);
}

TEST_F(EqualGraphDefTest, InputOrderMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Cross(a, b, e_.opts().WithName("C"));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  Cross(b, a, a_.opts().WithName("C"));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'C' has input 0 'B' that doesn't match expected 'A'",
            diff_);
}

TEST_F(EqualGraphDefTest, ControlInputOrder) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Node* c = Input(e_.opts().WithName("C"));
  Node* d = Input(e_.opts().WithName("D"));
  Cross(a, a, e_.opts()
                  .WithName("E")
                  .WithControlInput(b)
                  .WithControlInput(c)
                  .WithControlInput(d));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  c = Input(a_.opts().WithName("C"));
  d = Input(a_.opts().WithName("D"));
  Cross(a, a, a_.opts()
                  .WithName("E")
                  .WithControlInput(c)
                  .WithControlInput(d)
                  .WithControlInput(b));
  EXPECT_TRUE(Match()) << diff_;
}

TEST_F(EqualGraphDefTest, ControlInputMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Node* c = Input(e_.opts().WithName("C"));
  Node* d = Input(e_.opts().WithName("D"));
  Cross(a, a, e_.opts().WithName("E").WithControlInput(b).WithControlInput(c));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  c = Input(a_.opts().WithName("C"));
  d = Input(a_.opts().WithName("D"));
  Cross(a, a, a_.opts().WithName("E").WithControlInput(b).WithControlInput(d));
  EXPECT_FALSE(Match());
  EXPECT_EQ("Node named 'E' missing expected control input '^C'", diff_);
}

TEST_F(EqualGraphDefTest, ControlInputAdded) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Node* c = Input(e_.opts().WithName("C"));
  Cross(a, a, e_.opts().WithName("D").WithControlInput(b));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  c = Input(a_.opts().WithName("C"));
  Cross(a, a, a_.opts().WithName("D").WithControlInput(b).WithControlInput(c));
  EXPECT_FALSE(Match());
  EXPECT_EQ(
      "Node named 'D' has inputs 'A, A, ^B, ^C' that don't match "
      "expected 'A, A, ^B'",
      diff_);
}

TEST_F(EqualGraphDefTest, ControlInputRemoved) {
  Node* a = Input(e_.opts().WithName("A"));
  Node* b = Input(e_.opts().WithName("B"));
  Node* c = Input(e_.opts().WithName("C"));
  Cross(a, a, e_.opts().WithName("D").WithControlInput(b).WithControlInput(c));

  a = Input(a_.opts().WithName("A"));
  b = Input(a_.opts().WithName("B"));
  c = Input(a_.opts().WithName("C"));
  Cross(a, a, a_.opts().WithName("D").WithControlInput(b));
  EXPECT_FALSE(Match());
  EXPECT_EQ(
      "Node named 'D' has inputs 'A, A, ^B' that don't match "
      "expected 'A, A, ^B, ^C'",
      diff_);
}

TEST_F(EqualGraphDefTest, Attr) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef same(a->def());
  AddNodeAttr("foo", "bar", &same);
  EXPECT_TRUE(EqualNodeDef(same, same, &diff_)) << diff_;
}

TEST_F(EqualGraphDefTest, AttrAdded) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef actual(a->def());
  AddNodeAttr("foo", "bar", &actual);
  EXPECT_FALSE(EqualNodeDef(actual, a->def(), &diff_));
  EXPECT_EQ("Node named 'A' has unexpected attr 'foo' with value: \"bar\"",
            diff_);
}

TEST_F(EqualGraphDefTest, AttrRemoved) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef expected(a->def());
  AddNodeAttr("foo", "bar", &expected);
  EXPECT_FALSE(EqualNodeDef(a->def(), expected, &diff_));
  EXPECT_EQ("Node named 'A' missing expected attr 'foo' with value: \"bar\"",
            diff_);
}

TEST_F(EqualGraphDefTest, AttrOrder) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef actual(a->def());
  AddNodeAttr("foo", "bar", &actual);
  AddNodeAttr("baz", 42, &actual);

  NodeDef expected(a->def());
  AddNodeAttr("baz", 42, &expected);
  AddNodeAttr("foo", "bar", &expected);

  EXPECT_TRUE(EqualNodeDef(actual, expected, &diff_)) << diff_;
}

TEST_F(EqualGraphDefTest, AttrMismatch) {
  Node* a = Input(e_.opts().WithName("A"));
  NodeDef actual(a->def());
  AddNodeAttr("foo", "bar", &actual);
  AddNodeAttr("baz", 5, &actual);

  NodeDef expected(a->def());
  AddNodeAttr("baz", 42, &expected);
  AddNodeAttr("foo", "bar", &expected);

  EXPECT_FALSE(EqualNodeDef(actual, expected, &diff_));
  EXPECT_EQ(
      "Node named 'A' has attr 'baz' with value: 5 that does not match "
      "expected: 42",
      diff_);
}

}  // namespace
}  // namespace tensorflow
