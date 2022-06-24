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

#include "tensorflow/compiler/tf2tensorrt/segment/segment.h"

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/graph/testlib.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace segment {
namespace test {

class SegmentTest : public ::testing::Test {
 protected:
  std::function<Status(const Node*)> MakeCandidateFn(
      const std::set<string>& node_names) {
    return [node_names](const Node* node) -> Status {
      if (node_names.find(node->name()) != node_names.end()) {
        return OkStatus();
      }
      return errors::NotFound("Not a user specified candidate");
    };
  }

  std::function<bool(const Edge*)> MakeInputEdgeCandidateFn(
      const std::set<string>& node_names) {
    return [node_names](const Edge* in_edge) -> bool {
      return node_names.find(in_edge->dst()->name()) != node_names.end();
    };
  }

  std::function<bool(const Edge*)> MakeOutputEdgeCandidateFn(
      const std::set<string>& node_names) {
    return [node_names](const Edge* out_edge) -> bool {
      return node_names.find(out_edge->src()->name()) != node_names.end();
    };
  }

  void RunTest(const Graph* graph,
               const grappler::GraphProperties* graph_properties,
               const std::set<string>& candidates,
               const std::set<string>& input_candidates,
               const std::set<string>& output_candidates,
               const std::vector<std::set<string>>& expected_segments) {
    SegmentVector segments;
    TF_EXPECT_OK(SegmentGraph(graph, graph_properties,
                              MakeCandidateFn(candidates),
                              MakeInputEdgeCandidateFn(input_candidates),
                              MakeOutputEdgeCandidateFn(output_candidates),
                              segment_options_, &segments));
    ValidateSegment(segments, expected_segments);
  }

  void RunTest(const Graph* graph, const std::set<string>& candidates,
               const std::set<string>& input_candidates,
               const std::set<string>& output_candidates,
               const std::vector<std::set<string>>& expected_segments) {
    RunTest(graph, nullptr, candidates, input_candidates, output_candidates,
            expected_segments);
  }

  void ValidateSegment(const SegmentVector& segments,
                       const std::vector<std::set<string>>& expected_segments) {
    EXPECT_EQ(expected_segments.size(), segments.size());
    for (int i = 0; i < segments.size(); ++i) {
      std::set<string> segment_node_names;
      for (const Node* node : segments[i].nodes) {
        segment_node_names.insert(node->name());
      }
      const auto& expected = expected_segments[i];
      for (const auto& name : expected) {
        EXPECT_TRUE(segment_node_names.count(name))
            << "Segment " << i << " is missing expected node: " << name;
      }
      if (segment_node_names.size() == expected.size()) continue;
      for (const auto& name : segment_node_names) {
        EXPECT_TRUE(expected.count(name))
            << "Unexpected node found in segment " << i << ": " << name;
      }
    }
  }

  void DisableImplicitBatchMode() {
    segment_options_.use_implicit_batch = false;
    segment_options_.allow_dynamic_non_batch_dim = true;
  }

  void EnableImplicitBatchModeForStaticEngine(int maximum_batch_size = 1000) {
    segment_options_.use_implicit_batch = true;
    segment_options_.maximum_batch_size = maximum_batch_size;
    segment_options_.allow_dynamic_non_batch_dim = false;
  }

  SegmentOptions segment_options_;
};

std::set<string> operator-(const std::set<string>& lhs, const string& rhs) {
  std::set<string> result = lhs;
  CHECK(result.erase(rhs));
  return result;
}

TEST_F(SegmentTest, Empty) {
  Scope s = Scope::NewRootScope();
  Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));
  // Expect no segments/subgraphs.
  DisableImplicitBatchMode();
  RunTest(&g, {}, {}, {}, {});
}

TEST_F(SegmentTest, Simple) {
  //           feed
  //          //  \\
  //       add0    add1
  //        | \    /
  //        |  add2
  //        | /   \\
  //       add3    add4
  //          \    /
  //          <sink>
  Scope s = Scope::NewRootScope();
  auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT);
  auto add0 = ops::Add(s.WithOpName("add0"), feed, feed);
  auto add1 = ops::Add(s.WithOpName("add1"), feed, feed);
  auto add2 = ops::Add(s.WithOpName("add2"), add0, add1);
  auto add3 = ops::Add(s.WithOpName("add3"), add0, add2);
  auto add4 = ops::Add(s.WithOpName("add4"), add2, add2);
  Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));

  // All Add operations are candidates, and we expect all of them to be
  // collapsed into a single segment
  const std::set<string> all_adds = {"add0", "add1", "add2", "add3", "add4"};
  DisableImplicitBatchMode();
  RunTest(&g, all_adds, all_adds, all_adds, {all_adds});

  // Make add1 not a candidate, and we expect all other Add operations to be
  // collapsed into a single segment
  auto without_add1 = all_adds - "add1";
  RunTest(&g, without_add1, without_add1, without_add1, {without_add1});

  // Make add1 not a candidate and add2 not an input candidate, and we expect
  // add0 and add2 are removed from the segment.
  auto without_add2 = all_adds - "add2";
  RunTest(&g, without_add1, without_add2, without_add1, {{"add3", "add4"}});

  // Making add2 not an input candidate itself won't affect anything.
  RunTest(&g, all_adds, without_add2, all_adds, {all_adds});

  // Making add1 not an input candidate.
  RunTest(&g, all_adds, without_add1, all_adds, {without_add1});

  // Making add3 not an output candidate doesn't affect anything, since it's
  // output is sink.
  auto without_add3 = all_adds - "add3";
  RunTest(&g, all_adds, all_adds, without_add3, {all_adds});
}

TEST_F(SegmentTest, WithDeviceAssignments) {
  //           feed
  //          //  \\
  //       add0    add1
  //        | \    /
  //        |  add2
  //        | /   \\
  //       add3    add4
  //          \    /
  //          <sink>
  Scope s = Scope::NewRootScope();
  auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT);
  auto add0 = ops::Add(s.WithOpName("add0"), feed, feed);
  auto add1 = ops::Add(s.WithOpName("add1"), feed, feed);
  auto add2 = ops::Add(s.WithOpName("add2"), add0, add1);
  auto add3 = ops::Add(s.WithOpName("add3"), add0, add2);
  auto add4 = ops::Add(s.WithOpName("add4"), add2, add2);

  const std::set<string> all_adds = {"add0", "add1", "add2", "add3", "add4"};
  DisableImplicitBatchMode();

  {
    Graph g(OpRegistry::Global());
    TF_EXPECT_OK(s.ToGraph(&g));
    RunTest(&g, all_adds, all_adds, all_adds, {all_adds});
  }

  {
    // Assigning add1 to CPU to exclude it from the cluster.
    add1.node()->set_assigned_device_name("/device:CPU:0");
    Graph g(OpRegistry::Global());
    TF_EXPECT_OK(s.ToGraph(&g));
    RunTest(&g, all_adds, all_adds, all_adds, {all_adds - "add1"});
    add1.node()->set_assigned_device_name("");
  }

  {
    // Assigning operations add3 and add4 to another GPU to exclude the
    // operation from the cluster.
    constexpr char kGpu0[] = "/device:GPU:0";
    add0.node()->set_assigned_device_name(kGpu0);
    add1.node()->set_assigned_device_name(kGpu0);
    add2.node()->set_assigned_device_name(kGpu0);
    constexpr char kGpu1[] = "/device:GPU:1";
    add3.node()->set_assigned_device_name(kGpu1);
    add4.node()->set_assigned_device_name(kGpu1);
    Graph g(OpRegistry::Global());
    TF_EXPECT_OK(s.ToGraph(&g));
    RunTest(&g, all_adds, all_adds, all_adds, {{"add0", "add1", "add2"}});
  }

  {
    // Assigning the operations to two compatibile GPU devices resulting in
    // one cluster with all operations.
    constexpr char kGpuAny[] = "/device:GPU:*";
    add3.node()->set_assigned_device_name(kGpuAny);
    add4.node()->set_assigned_device_name(kGpuAny);
    Graph g(OpRegistry::Global());
    TF_EXPECT_OK(s.ToGraph(&g));
    RunTest(&g, all_adds, all_adds, all_adds, {all_adds});
  }
}

TEST_F(SegmentTest, AvoidCycle) {
  //           feed
  //          //  \\
  //       add0    add1
  //        | \    /
  //        |  add2
  //        |  /  \\
  //       add3    add4
  //          \    /
  //          <sink>
  Scope s = Scope::NewRootScope();
  auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT);
  auto add0 = ops::Add(s.WithOpName("add0"), feed, feed);
  auto add1 = ops::Add(s.WithOpName("add1"), feed, feed);
  auto add2 = ops::Add(s.WithOpName("add2"), add0, add1);
  auto add3 = ops::Add(s.WithOpName("add3"), add0, add2);
  auto add4 = ops::Add(s.WithOpName("add4"), add2, add2);
  Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));

  // add2 is not a TRT candidate so there should be no segments generated.
  const std::set<string> without_add2 = {"add0", "add1", "add3", "add4"};
  DisableImplicitBatchMode();
  RunTest(&g, without_add2, without_add2, without_add2, {});
}

TEST_F(SegmentTest, Multiple) {
  //              feed
  //           //  ||  \\
  //        add0  add1  add7
  //        |  \  /     / \\
  //        |  add2    /   \\
  //        |   || \   |   ||
  //        |   ||  add5  add8
  //        |  /  \ /  \   /
  //        add3  add4  add6
  //           \   |   /
  //             <sink>
  Scope s = Scope::NewRootScope();
  auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT);
  auto add0 = ops::Add(s.WithOpName("add0"), feed, feed);
  auto add1 = ops::Add(s.WithOpName("add1"), feed, feed);
  auto add7 = ops::Add(s.WithOpName("add7"), feed, feed);
  auto add2 = ops::Add(s.WithOpName("add2"), add0, add1);
  auto add5 = ops::Add(s.WithOpName("add5"), add2, add7);
  auto add8 = ops::Add(s.WithOpName("add8"), add7, add7);
  auto add3 = ops::Add(s.WithOpName("add3"), add0, add2);
  auto add4 = ops::Add(s.WithOpName("add4"), add2, add5);
  auto add6 = ops::Add(s.WithOpName("add6"), add5, add8);
  Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));

  const std::set<string> all_adds = {"add0", "add1", "add2", "add3", "add4",
                                     "add5", "add6", "add7", "add8"};
  // Make add5 not a TRT candidate, and we expect two segments.
  auto without_add5 = all_adds - "add5";
  DisableImplicitBatchMode();
  RunTest(&g, without_add5, without_add5, without_add5,
          {{"add0", "add1", "add2", "add3"}, {"add6", "add8"}});

  // Make add8 not a candidate and add6 not an input candidate, then all direct
  // and indirect inputs of add6 will be removed from the segment.
  auto without_add8 = all_adds - "add8";
  auto without_add6 = all_adds - "add6";
  RunTest(&g, without_add8, without_add6, all_adds, {{"add3", "add4"}});

  // Make add3 not a candidate and add0 not an output candidate, then all
  // direct and indirect outputs of add0 will be removed from the segment.
  auto without_add3 = all_adds - "add3";
  auto without_add0 = all_adds - "add0";
  RunTest(&g, without_add3, all_adds, without_add0, {{"add1", "add7", "add8"}});
}

TEST_F(SegmentTest, BigIfElse) {
  //           feed
  //            ||
  //           add0
  //         //    \\
  //       add1    add4
  //        ||      ||
  //       add2    add5
  //        ||      ||
  //       add3    add6
  //         \\    //
  //           add7
  //            ||
  //          <sink>
  Scope s = Scope::NewRootScope();
  auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT);
  auto add0 = ops::Add(s.WithOpName("add0"), feed, feed);
  auto add1 = ops::Add(s.WithOpName("add1"), add0, add0);
  auto add2 = ops::Add(s.WithOpName("add2"), add1, add1);
  auto add3 = ops::Add(s.WithOpName("add3"), add2, add2);
  auto add4 = ops::Add(s.WithOpName("add4"), add0, add0);
  auto add5 = ops::Add(s.WithOpName("add5"), add4, add4);
  auto add6 = ops::Add(s.WithOpName("add6"), add5, add5);
  auto add7 = ops::Add(s.WithOpName("add7"), add3, add6);
  Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));

  // Make add2 not a TRT candidate, and we expect 2 segments.
  const std::set<string> all_adds = {"add0", "add1", "add2", "add3",
                                     "add4", "add5", "add6", "add7"};
  DisableImplicitBatchMode();
  RunTest(&g, all_adds - "add2", all_adds, all_adds,
          {{"add0", "add1"}, {"add3", "add4", "add5", "add6", "add7"}});
}

TEST_F(SegmentTest, IdentityOps) {
  Scope s = Scope::NewRootScope();
  auto feed = ops::Placeholder(s.WithOpName("feed"), DT_FLOAT);
  auto identity0 = ops::Identity(s.WithOpName("identity0"), feed);
  auto identity1 = ops::Identity(s.WithOpName("identity1"), identity0);
  auto identity2 = ops::Identity(s.WithOpName("identity2"), identity1);
  auto identity3 = ops::Identity(s.WithOpName("identity3"), identity2);
  Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));

  const std::set<string> all_identities = {"identity0", "identity1",
                                           "identity2", "identity3"};
  // Identity ops are not counted as effective ops in the segment, so no segment
  // will be formed in this case.
  DisableImplicitBatchMode();
  RunTest(&g, all_identities, all_identities, all_identities, {});
}

// Testing implicit batch mode segmentation: it excludes the add-2 operation
// with a dynamic non-batch dimension.
TEST_F(SegmentTest, ExcludeAddWithDynamicNonBatchDimension) {
  Scope s = Scope::NewRootScope();
  auto feed_0_shape = ops::Placeholder::Shape(PartialTensorShape({-1, 2, 3}));
  auto feed_1_shape = ops::Placeholder::Shape(PartialTensorShape({-1, -1, 3}));
  auto const_val = ops::Const<float>(s, {1.0}, {});
  auto feed_0 =
      ops::Placeholder(s.WithOpName("feed-1"), DT_FLOAT, feed_0_shape);
  auto feed_1 =
      ops::Placeholder(s.WithOpName("feed-2"), DT_FLOAT, feed_1_shape);
  auto add_0 = ops::Add(s.WithOpName("add-0"), feed_0, const_val);
  auto add_1 = ops::Add(s.WithOpName("add-1"), add_0, feed_0);
  auto add_2 = ops::Add(s.WithOpName("add-2"), const_val, feed_1);

  grappler::GrapplerItem item;
  item.fetch.push_back("add-2");
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));

  grappler::GraphProperties static_graph_properties(item);
  TF_EXPECT_OK(static_graph_properties.InferStatically(true));

  Graph g(OpRegistry::Global());
  TF_CHECK_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph, &g));

  const std::set<string> all_nodes = {"add-0", "add-1", "add-2"};
  EnableImplicitBatchModeForStaticEngine();
  RunTest(&g, &static_graph_properties, all_nodes, all_nodes, all_nodes,
          {all_nodes - "add-2"});
}

// Testing implicit batch mode segmentation: It excludes the reshape operation
// with a dynamic non-batch output dimension.
// TODO(bixia): hoist the check for reshape should not change batch size from
// the converter to the segmenter and add another test case for excluding
// a reshape without dynamic dimensions involved.
TEST_F(SegmentTest, ExcludeReshapeWithDynamicNonBatchDimensionInOutput) {
  Scope s = Scope::NewRootScope();
  auto feed_0_shape = ops::Placeholder::Shape(PartialTensorShape({-1, 2, 3}));
  auto const_val = ops::Const<float>(s, {1.0}, {});
  auto feed_0 =
      ops::Placeholder(s.WithOpName("feed-1"), DT_FLOAT, feed_0_shape);
  auto add_0 = ops::Add(s.WithOpName("add-0"), feed_0, const_val);
  auto reshape = ops::Reshape(s.WithOpName("reshape"), add_0, Input({6, -1}));
  auto add_1 = ops::Add(s.WithOpName("add-1"), reshape, const_val);

  grappler::GrapplerItem item;
  item.fetch.push_back("add-1");
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));

  grappler::GraphProperties static_graph_properties(item);
  TF_EXPECT_OK(static_graph_properties.InferStatically(true));

  Graph g(OpRegistry::Global());
  TF_CHECK_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph, &g));

  const std::set<string> all_nodes = {"add-0", "reshape", "add-1"};
  EnableImplicitBatchModeForStaticEngine();
  RunTest(&g, &static_graph_properties, all_nodes, all_nodes, all_nodes, {});
}

TEST_F(SegmentTest, RankOneCannotUseImplicitBatch) {
  Scope s = Scope::NewRootScope();
  auto input_0_shape = ops::Placeholder::Shape(TensorShape({3}));
  auto input_1_shape = ops::Placeholder::Shape(TensorShape({3}));
  auto input_0 =
      ops::Placeholder(s.WithOpName("input-0"), DT_FLOAT, input_0_shape);
  auto input_1 =
      ops::Placeholder(s.WithOpName("input-1"), DT_FLOAT, input_1_shape);
  auto const_val = ops::Const(s.WithOpName("const-scalar"), 1.0f, {});
  auto output_0 = ops::Add(s.WithOpName("output-0"), input_0, const_val);
  auto output_1 = ops::Add(s.WithOpName("output-1"), input_1, const_val);

  grappler::GrapplerItem item;
  item.fetch.push_back("output-0");
  item.fetch.push_back("output-1");
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));

  grappler::GraphProperties static_graph_properties(item);
  TF_EXPECT_OK(static_graph_properties.InferStatically(true));

  Graph g(OpRegistry::Global());
  TF_CHECK_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph, &g));

  const std::set<string> all_nodes = {"const-scalar", "output-0", "output-1"};
  EnableImplicitBatchModeForStaticEngine();
  RunTest(&g, &static_graph_properties, all_nodes, all_nodes, all_nodes, {});
}

TEST_F(SegmentTest, TwoChainsDiffBatchSizes) {
  Scope s = Scope::NewRootScope();
  auto input_0_shape = ops::Placeholder::Shape(TensorShape({2, 3}));
  auto input_1_shape = ops::Placeholder::Shape(TensorShape({5, 3}));
  auto input_0 =
      ops::Placeholder(s.WithOpName("input-0"), DT_FLOAT, input_0_shape);
  auto input_1 =
      ops::Placeholder(s.WithOpName("input-1"), DT_FLOAT, input_1_shape);
  auto const_val = ops::Const(s.WithOpName("const-scalar"), 1.0f, {});
  auto output_0 = ops::Add(s.WithOpName("output-0"), input_0, const_val);
  auto output_1 = ops::Add(s.WithOpName("output-1"), input_1, const_val);

  grappler::GrapplerItem item;
  item.fetch.push_back("output-0");
  item.fetch.push_back("output-1");
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));

  grappler::GraphProperties static_graph_properties(item);
  TF_EXPECT_OK(static_graph_properties.InferStatically(true));

  Graph g(OpRegistry::Global());
  TF_CHECK_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph, &g));

  const std::set<string> all_nodes = {"const-scalar", "output-0", "output-1"};
  EnableImplicitBatchModeForStaticEngine();
  RunTest(&g, &static_graph_properties, all_nodes, all_nodes, all_nodes,
          /*expected_segments=*/{{"output-0", "const-scalar"}});

  // Converter will create engines based on the static batch size
  EnableImplicitBatchModeForStaticEngine(1);
  RunTest(&g, &static_graph_properties, all_nodes, all_nodes, all_nodes,
          /*expected_segments=*/{{"output-0", "const-scalar"}});
}

TEST_F(SegmentTest, SameRankImplicitBroadcastingStaticBatchSize) {
  Scope s = Scope::NewRootScope();
  auto input_0_shape = ops::Placeholder::Shape(TensorShape({2, 3, 1}));
  auto input_1_shape = ops::Placeholder::Shape(TensorShape({1, 3, 4}));
  auto input_2_shape = ops::Placeholder::Shape(TensorShape({2, 3, 4}));
  auto input_0 =
      ops::Placeholder(s.WithOpName("input-0"), DT_FLOAT, input_0_shape);
  auto input_1 =
      ops::Placeholder(s.WithOpName("input-1"), DT_FLOAT, input_1_shape);
  auto input_2 =
      ops::Placeholder(s.WithOpName("input-2"), DT_FLOAT, input_2_shape);
  auto multiple = ops::Mul(s.WithOpName("multiple"), input_2, input_2);
  auto output_0 = ops::Add(s.WithOpName("output-0"), input_0, multiple);
  auto output_1 = ops::Add(s.WithOpName("output-1"), input_1, multiple);

  grappler::GrapplerItem item;
  item.fetch.push_back("output-0");
  item.fetch.push_back("output-1");
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));

  grappler::GraphProperties static_graph_properties(item);
  TF_EXPECT_OK(static_graph_properties.InferStatically(true));

  Graph g(OpRegistry::Global());
  TF_CHECK_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph, &g));

  const std::set<string> all_nodes = {"multiple", "output-0", "output-1"};
  EnableImplicitBatchModeForStaticEngine();
  RunTest(&g, &static_graph_properties, all_nodes, all_nodes, all_nodes,
          {all_nodes});
}

TEST_F(SegmentTest, SameRankImplicitBroadcastingDynamicBatchSize) {
  Scope s = Scope::NewRootScope();
  auto input_0_shape = ops::Placeholder::Shape(PartialTensorShape({-1, 2}));
  auto input_1_shape = ops::Placeholder::Shape(TensorShape({1, 2}));
  auto input_0 =
      ops::Placeholder(s.WithOpName("input-0"), DT_FLOAT, input_0_shape);
  auto input_1 =
      ops::Placeholder(s.WithOpName("input-1"), DT_FLOAT, input_1_shape);
  auto const_val = ops::Const(s.WithOpName("const-val"), 1.0f, {1, 1});
  auto add_0 = ops::Add(s.WithOpName("add-0"), input_0, const_val);
  auto output_0 = ops::Add(s.WithOpName("output-0"), input_0, add_0);

  grappler::GrapplerItem item;
  item.fetch.push_back("output-0");
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));

  grappler::GraphProperties static_graph_properties(item);
  TF_EXPECT_OK(static_graph_properties.InferStatically(true));

  Graph g(OpRegistry::Global());
  TF_CHECK_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph, &g));

  const std::set<string> all_nodes = {"const-val", "add-0", "output-0"};
  EnableImplicitBatchModeForStaticEngine();
  RunTest(&g, &static_graph_properties, all_nodes, all_nodes, all_nodes,
          {{"const-val", "add-0", "output-0"}});
}

TEST_F(SegmentTest, IncompatibleBatchSizes) {
  Scope s = Scope::NewRootScope();
  auto input_0_shape = ops::Placeholder::Shape(PartialTensorShape({-1, 2}));
  auto input_1_shape = ops::Placeholder::Shape(TensorShape({2, 2}));
  auto input_0 =
      ops::Placeholder(s.WithOpName("input-0"), DT_FLOAT, input_0_shape);
  auto input_1 =
      ops::Placeholder(s.WithOpName("input-1"), DT_FLOAT, input_1_shape);
  auto const_val = ops::Const(s.WithOpName("const-val"), 1.0f, {2, 2});
  auto add_0 = ops::Add(s.WithOpName("add-0"), input_0, const_val);
  auto output_0 = ops::Add(s.WithOpName("output-0"), input_0, add_0);

  grappler::GrapplerItem item;
  item.fetch.push_back("output-0");
  TF_EXPECT_OK(s.ToGraphDef(&item.graph));

  grappler::GraphProperties static_graph_properties(item);
  TF_EXPECT_OK(static_graph_properties.InferStatically(true));

  Graph g(OpRegistry::Global());
  TF_CHECK_OK(
      ConvertGraphDefToGraph(GraphConstructorOptions(), item.graph, &g));

  const std::set<string> all_nodes = {"const-val", "add-0", "output-0"};
  EnableImplicitBatchModeForStaticEngine();
  RunTest(&g, &static_graph_properties, all_nodes, all_nodes, all_nodes, {});
}
}  // namespace test
}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT
