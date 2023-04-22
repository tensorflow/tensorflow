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

#include "tensorflow/core/grappler/graph_analyzer/graph_analyzer.h"

#include <algorithm>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/memory/memory.h"
#include "tensorflow/core/grappler/graph_analyzer/test_tools.h"

namespace tensorflow {
namespace grappler {
namespace graph_analyzer {
namespace test {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Ne;
using ::testing::SizeIs;
using ::testing::UnorderedElementsAre;

class GraphAnalyzerTest : public ::testing::Test, protected TestGraphs {
 protected:
  Status BuildMap() { return gran_->BuildMap(); }

  void FindSubgraphs() { gran_->FindSubgraphs(); }

  void DropInvalidSubgraphs() { gran_->DropInvalidSubgraphs(); }

  Status CollateResult() { return gran_->CollateResult(); }

  void ExtendSubgraph(Subgraph* parent) { gran_->ExtendSubgraph(parent); }

  void ExtendSubgraphPortAllOrNone(Subgraph* parent, GenNode* node,
                                   GenNode::Port port) {
    gran_->ExtendSubgraphPortAllOrNone(parent, node, port);
  }

  void ExtendSubgraphAllOrNone(Subgraph* parent, GenNode* node) {
    gran_->ExtendSubgraphAllOrNone(parent, node);
  }

  std::vector<string> DumpRawSubgraphs() { return gran_->DumpRawSubgraphs(); }

  std::vector<string> DumpPartials() {
    std::vector<string> result;
    for (const auto& it : gran_->partial_) {
      result.emplace_back(it->Dump());
    }
    return result;
  }

  const GenNodeMap& GetNodes() { return gran_->nodes_; }

  GenNode* GetNode(const string& name) { return gran_->nodes_.at(name).get(); }

  SubgraphPtrSet& GetResult() { return gran_->result_; }
  SubgraphPtrSet& GetPartial() { return gran_->partial_; }
  std::deque<Subgraph*>& GetTodo() { return gran_->todo_; }

  // Gets initialized by a particular test from a suitable GraphDef.
  std::unique_ptr<GraphAnalyzer> gran_;
};

TEST_F(GraphAnalyzerTest, BuildMap) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 1);
  Status st = BuildMap();
  EXPECT_THAT(st, Eq(Status::OK()));

  auto& map = GetNodes();
  EXPECT_THAT(map.find("node1"), Ne(map.end()));
  EXPECT_THAT(map.find("node2"), Ne(map.end()));
  EXPECT_THAT(map.find("node3"), Ne(map.end()));
}

TEST_F(GraphAnalyzerTest, BuildMapError) {
  // A duplicate node.
  (*graph_3n_self_control_.add_node()) = MakeNodeConst("node1");
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 1);
  Status st = BuildMap();
  ASSERT_THAT(
      st, Eq(Status(error::INVALID_ARGUMENT, "Duplicate node name 'node1'.")));
}

TEST_F(GraphAnalyzerTest, FindSubgraphs0) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 0);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  FindSubgraphs();
  auto& subgraphs = GetResult();
  EXPECT_THAT(subgraphs, SizeIs(0));
  EXPECT_THAT(DumpRawSubgraphs(), ElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

TEST_F(GraphAnalyzerTest, FindSubgraphs1) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 1);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  FindSubgraphs();
  auto& subgraphs = GetResult();
  EXPECT_THAT(subgraphs, SizeIs(3));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: BroadcastGradientArgs(node3)",
      "1: Const(node1)",
      "1: Sub(node2)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// The required subgraphs are larger than the graph.
TEST_F(GraphAnalyzerTest, FindSubgraphsTooLarge) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_3n_self_control_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  FindSubgraphs();
  EXPECT_THAT(DumpRawSubgraphs(), ElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

//===

// Successfully propagate backwards through a multi-input link,
// with the base (currently-extending) node already in the graph.
TEST_F(GraphAnalyzerTest, MultiInputSuccessBackwardsBaseIn) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards through a multi-input link,
// with the base (currently-extending) node not in the graph yet.
TEST_F(GraphAnalyzerTest, MultiInputSuccessBackwardsBaseOut) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto parent = absl::make_unique<Subgraph>(Subgraph::Identity());
  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraphPortAllOrNone(parent.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards through a multi-input link,
// where the target subgraph size is larger.
TEST_F(GraphAnalyzerTest, MultiInputSuccessBackwardsIncomplete) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 5);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  // clang-format off
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// Propagate backwards through a multi-input link, finding that the
// resulting subgraph would be too large.
TEST_F(GraphAnalyzerTest, MultiInputTooLargeBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 3);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Propagate backwards through a multi-input link, finding that nothing
// would be added to the parent subgraph.
TEST_F(GraphAnalyzerTest, MultiInputNothingAddedBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root = absl::make_unique<Subgraph>(
      Subgraph::Identity({GetNode("add2"), GetNode("const2_1"),
                          GetNode("const2_2"), GetNode("const2_3")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate forwards through a multi-input link,
// with the base (currently-extending) node not in the subgraph yet.
TEST_F(GraphAnalyzerTest, MultiInputSuccessForwardsBaseOut) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("const2_1")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("add2"),
                              GenNode::Port(true, 0));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards through a multi-input link.
TEST_F(GraphAnalyzerTest, MultiInputSuccessBackwardsFull) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("add2")}));

  ExtendSubgraph(root.get());

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: AddN(add2), Sub(sub)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// Successfully propagate forwards through a multi-input link.
TEST_F(GraphAnalyzerTest, MultiInputSuccessForwardsFull) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("const2_1")}));

  ExtendSubgraph(root.get());

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add2), Const(const2_1), Const(const2_2), Const(const2_3)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

TEST_F(GraphAnalyzerTest, DropInvalidSubgraphsMulti) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_multi_input_, 3);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  // A good one, multi-input is all-in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("const1_1"),
      GetNode("const1_2"),
      GetNode("add1"),
  })));
  // A good one, multi-input is all-out
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("add1"),
      GetNode("add2"),
      GetNode("sub"),
  })));
  // A bad one, multi-input is partially in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("const1_1"),
      GetNode("add1"),
      GetNode("sub"),
  })));
  // A bad one, multi-input is partially in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("add2"),
      GetNode("const2_1"),
      GetNode("const2_2"),
  })));

  DropInvalidSubgraphs();

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: AddN(add1), AddN(add2), Sub(sub)",
      "1: AddN(add1), Const(const1_1), Const(const1_2)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

//===

// Successfully propagate backwards through a multi-input link,
// with the base (currently-extending) node already in the graph.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass2")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass2"));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass2)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards through a multi-input link,
// but no control links propagate. It also tests the situation
// where the target subgraph size is larger.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessBackwardsNoControl) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 5);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass1")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass1"));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: Const(const1_1), Const(const1_2), IdentityN(pass1)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// The control links propagate separately as all-or-none, even on the nodes
// that are all-or-none for the normal inputs.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSeparateControl) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 5);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass1")}));

  ExtendSubgraphPortAllOrNone(root.get(), GetNode("pass1"),
                              GenNode::Port(true, -1));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass1)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// Propagate backwards from all-or-none-input node, finding that the
// resulting subgraph would be too large.
TEST_F(GraphAnalyzerTest, AllOrNoneInputTooLargeBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 3);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass2")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass2"));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Propagate backwards from all-or-none-input node, finding that nothing
// would be added to the parent subgraph.
TEST_F(GraphAnalyzerTest, AllOrNoneInputNothingAddedBackwards) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root = absl::make_unique<Subgraph>(
      Subgraph::Identity({GetNode("pass2"), GetNode("const2_1"),
                          GetNode("const2_2"), GetNode("const2_3")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass2"));

  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre());
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate forwards to all-or-none-input node,
// with the base (currently-extending) node not in the subgraph yet.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessForwardsBaseOut) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("const2_1")}));

  ExtendSubgraphAllOrNone(root.get(), GetNode("pass2"));

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass2)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

// Successfully propagate backwards from all-or-none-input node.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessBackwardsFull) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("pass2")}));

  ExtendSubgraph(root.get());

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass2)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre(
      "1: IdentityN(pass2), Sub(sub)"
      ));
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(1));
}

// Successfully propagate forwards to all-or-none-input node. This includes
// both all-or-none-input for the normal inputs, and multi-input by the
// control path.
TEST_F(GraphAnalyzerTest, AllOrNoneInputSuccessForwardsFull) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 4);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  auto root =
      absl::make_unique<Subgraph>(Subgraph::Identity({GetNode("const2_1")}));

  ExtendSubgraph(root.get());

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass2)",
      "1: Const(const2_1), Const(const2_2), Const(const2_3), IdentityN(pass1)"
      ));
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  // clang-format on
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

TEST_F(GraphAnalyzerTest, DropInvalidSubgraphsAllOrNone) {
  gran_ = absl::make_unique<GraphAnalyzer>(graph_all_or_none_, 3);
  Status st = BuildMap();
  ASSERT_THAT(st, Eq(Status::OK()));

  // A good one, all-or-none is all-in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("const1_1"),
      GetNode("const1_2"),
      GetNode("pass1"),
  })));
  // A good one, all-or-none is all-out
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("pass1"),
      GetNode("pass2"),
      GetNode("sub"),
  })));
  // A bad one, all-or-none is partially in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("const1_1"),
      GetNode("pass1"),
      GetNode("sub"),
  })));
  // A bad one, all-or-none is partially in.
  GetResult().insert(absl::make_unique<Subgraph>(Subgraph::Identity({
      GetNode("pass2"),
      GetNode("const2_1"),
      GetNode("const2_2"),
  })));

  DropInvalidSubgraphs();

  // clang-format off
  EXPECT_THAT(DumpRawSubgraphs(), UnorderedElementsAre(
      "1: IdentityN(pass1), IdentityN(pass2), Sub(sub)",
      "1: Const(const1_1), Const(const1_2), IdentityN(pass1)"
      ));
  // clang-format on
  EXPECT_THAT(DumpPartials(), UnorderedElementsAre());
  EXPECT_THAT(GetTodo(), SizeIs(0));
}

}  // end namespace test
}  // end namespace graph_analyzer
}  // end namespace grappler
}  // end namespace tensorflow
