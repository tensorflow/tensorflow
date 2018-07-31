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

#include "tensorflow/contrib/tensorrt/segment/segment.h"

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

namespace tensorflow {
namespace tensorrt {
namespace segment {
namespace test {
namespace ops = ::tensorflow::ops;

class SegmentTest : public ::testing::Test {
 protected:
  std::function<bool(const tensorflow::Node*)> MakeCandidateFn(
      const std::set<string>& node_names) {
    return [node_names](const tensorflow::Node* node) -> bool {
      return node_names.find(node->name()) != node_names.end();
    };
  }

  std::function<bool(const tensorflow::Edge*)> MakeInputEdgeCandidateFn(
      const std::set<string>& node_names) {
    return [node_names](const tensorflow::Edge* in_edge) -> bool {
      return node_names.find(in_edge->dst()->name()) != node_names.end();
    };
  }

  std::function<bool(const tensorflow::Edge*)> MakeOutputEdgeCandidateFn(
      const std::set<string>& node_names) {
    return [node_names](const tensorflow::Edge* out_edge) -> bool {
      return node_names.find(out_edge->src()->name()) != node_names.end();
    };
  }

  void RunTest(const tensorflow::Graph* graph,
               const std::set<string>& candidates,
               const std::set<string>& input_candidates,
               const std::set<string>& output_candidates,
               const std::vector<std::set<string>>& expected_segments) {
    SegmentNodesVector segments;
    TF_EXPECT_OK(SegmentGraph(graph, MakeCandidateFn(candidates),
                              MakeInputEdgeCandidateFn(input_candidates),
                              MakeOutputEdgeCandidateFn(output_candidates),
                              default_options_, &segments));
    ValidateSegment(segments, expected_segments);
  }

  void ValidateSegment(const SegmentNodesVector& segments,
                       const std::vector<std::set<string>>& expected_segments) {
    EXPECT_EQ(expected_segments.size(), segments.size());
    for (int i = 0; i < segments.size(); ++i) {
      const auto& segment_node_names = segments[i].first;
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

  SegmentOptions default_options_;
};

std::set<string> operator-(const std::set<string>& lhs, const string& rhs) {
  std::set<string> result = lhs;
  CHECK(result.erase(rhs));
  return result;
}

TEST_F(SegmentTest, Empty) {
  Scope s = Scope::NewRootScope();
  tensorflow::Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));
  // Expect no segments/subgraphs.
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
  tensorflow::Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));

  // All Add operations are candidates, and we expect all of them to be
  // collapsed into a single segment
  const std::set<string> all_adds = {"add0", "add1", "add2", "add3", "add4"};
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
  tensorflow::Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));

  // add2 is not a TRT candidate so there should be no segments generated.
  const std::set<string> without_add2 = {"add0", "add1", "add3", "add4"};
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
  tensorflow::Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));

  const std::set<string> all_adds = {"add0", "add1", "add2", "add3", "add4",
                                     "add5", "add6", "add7", "add8"};
  // Make add5 not a TRT candidate, and we expect two segments.
  auto without_add5 = all_adds - "add5";
  RunTest(&g, without_add5, without_add5, without_add5,
          {{"add6", "add8"}, {"add0", "add1", "add2", "add3"}});

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
  tensorflow::Graph g(OpRegistry::Global());
  TF_EXPECT_OK(s.ToGraph(&g));

  // Make add2 not a TRT candidate, and we expect 2 segments.
  const std::set<string> all_adds = {"add0", "add1", "add2", "add3",
                                     "add4", "add5", "add6", "add7"};
  RunTest(&g, all_adds - "add2", all_adds, all_adds,
          {{"add3", "add4", "add5", "add6", "add7"}, {"add0", "add1"}});
}

}  // namespace test
}  // namespace segment
}  // namespace tensorrt
}  // namespace tensorflow
