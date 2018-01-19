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

#include "tensorflow/contrib/tensorrt/segment/segment.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/test.h"

//------------------------------------------------------------------------------
using namespace tensorflow;

namespace tensorrt {
namespace segment {
namespace test {

class SegmentTest : public ::testing::Test {
 public:
  bool GetGraphDef(TF_Graph* graph, tensorflow::GraphDef* graph_def);

  TF_Operation* Placeholder(TF_Graph* graph, TF_Status* s, const char* name);
  TF_Operation* Add(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                    TF_Status* s, const char* name);

  std::function<bool(const NodeDef&)> MakeCandidateFn(
      const std::set<std::string>& node_names);

 protected:
  void PlaceholderHelper(TF_Graph* graph, TF_Status* s, const char* name,
                         TF_Operation** op);
  void AddHelper(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                 TF_Status* s, const char* name, TF_Operation** op, bool check);

  SegmentOptions default_options_;
};

bool SegmentTest::GetGraphDef(TF_Graph* graph,
                              tensorflow::GraphDef* graph_def) {
  TF_Status* s = TF_NewStatus();
  TF_Buffer* buffer = TF_NewBuffer();
  TF_GraphToGraphDef(graph, buffer, s);
  bool ret = TF_GetCode(s) == TF_OK;
  EXPECT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  if (ret) ret = graph_def->ParseFromArray(buffer->data, buffer->length);
  TF_DeleteBuffer(buffer);
  TF_DeleteStatus(s);
  return ret;
}

std::function<bool(const NodeDef&)> SegmentTest::MakeCandidateFn(
    const std::set<std::string>& node_names) {
  return [node_names](const NodeDef& node) -> bool {
    return node_names.find(node.name()) != node_names.end();
  };
}

void SegmentTest::PlaceholderHelper(TF_Graph* graph, TF_Status* s,
                                    const char* name, TF_Operation** op) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "Placeholder", name);
  TF_SetAttrType(desc, "dtype", TF_INT32);
  *op = TF_FinishOperation(desc, s);
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  ASSERT_NE(*op, nullptr);
}

TF_Operation* SegmentTest::Placeholder(TF_Graph* graph, TF_Status* s,
                                       const char* name) {
  TF_Operation* op;
  PlaceholderHelper(graph, s, name, &op);
  return op;
}

void SegmentTest::AddHelper(TF_Operation* l, TF_Operation* r, TF_Graph* graph,
                            TF_Status* s, const char* name, TF_Operation** op,
                            bool check) {
  TF_OperationDescription* desc = TF_NewOperation(graph, "AddN", name);
  TF_Output add_inputs[2] = {{l, 0}, {r, 0}};
  TF_AddInputList(desc, add_inputs, 2);
  *op = TF_FinishOperation(desc, s);
  if (check) {
    ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
    ASSERT_NE(*op, nullptr);
  }
}

TF_Operation* SegmentTest::Add(TF_Operation* l, TF_Operation* r,
                               TF_Graph* graph, TF_Status* s,
                               const char* name) {
  TF_Operation* op;
  AddHelper(l, r, graph, s, name, &op, true);
  return op;
}

//------------------------------------------------------------------------------
TEST_F(SegmentTest, Empty) {
  TF_Graph* graph = TF_NewGraph();

  GraphDef graph_def;
  ASSERT_TRUE(GetGraphDef(graph, &graph_def));

  SegmentNodesVector segments;
  ASSERT_EQ(
      SegmentGraph(graph_def, MakeCandidateFn({}), default_options_, &segments),
      tensorflow::Status::OK());

  // Expect no segments/subgraphs.
  EXPECT_TRUE(segments.empty());
}

//------------------------------------------------------------------------------
TEST_F(SegmentTest, Simple) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  //           feed
  //         //    ||
  //       add0    add1
  //        | |    /
  //        |  add2
  //        |  /  ||
  //       add3    add4
  //           |  /
  //          <sink>
  //
  TF_Operation* feed = Placeholder(graph, s, "feed");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("feed"), string(TF_OperationName(feed)));

  TF_Operation* add0 = Add(feed, feed, graph, s, "add0");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add1 = Add(feed, feed, graph, s, "add1");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add2 = Add(add0, add1, graph, s, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add3 = Add(add0, add2, graph, s, "add3");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("add3"), string(TF_OperationName(add3)));
  TF_Operation* add4 = Add(add2, add2, graph, s, "add4");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("add4"), string(TF_OperationName(add4)));

  GraphDef graph_def;
  ASSERT_TRUE(GetGraphDef(graph, &graph_def));

  SegmentNodesVector segments;
  ASSERT_EQ(
      SegmentGraph(graph_def,
                   MakeCandidateFn({"add0", "add1", "add2", "add3", "add4"}),
                   default_options_, &segments),
      tensorflow::Status::OK());

  // Expect all Add operations to be collapsed into a single segment
  ASSERT_EQ(segments.size(), 1);
  std::vector<std::string> expected{"add0", "add1", "add2", "add3", "add4"};
  for (const auto& ex : expected) {
    EXPECT_TRUE(segments[0].find(ex) != segments[0].end())
        << "Missing expected node " << ex;
  }
}

//------------------------------------------------------------------------------
TEST_F(SegmentTest, AvoidCycle) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  // add2 is not a TRT candidate so add0/add3 cannot be formed as a
  // subgraph
  //
  //           feed
  //         //    ||
  //       add0    add1
  //        | |    /
  //        |  add2
  //        |  /  ||
  //       add3    add4
  //           |  /
  //          <sink>
  //
  TF_Operation* feed = Placeholder(graph, s, "feed");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("feed"), string(TF_OperationName(feed)));

  TF_Operation* add0 = Add(feed, feed, graph, s, "add0");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add1 = Add(feed, feed, graph, s, "add1");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add2 = Add(add0, add1, graph, s, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add3 = Add(add0, add2, graph, s, "add3");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("add3"), string(TF_OperationName(add3)));
  TF_Operation* add4 = Add(add2, add2, graph, s, "add4");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("add4"), string(TF_OperationName(add4)));

  GraphDef graph_def;
  ASSERT_TRUE(GetGraphDef(graph, &graph_def));

  SegmentNodesVector segments;
  ASSERT_EQ(
      SegmentGraph(graph_def, MakeCandidateFn({"add0", "add1", "add3", "add4"}),
                   default_options_, &segments),
      tensorflow::Status::OK());

  // Expect no subgraphs
  EXPECT_EQ(segments.size(), 0);
}

//------------------------------------------------------------------------------
TEST_F(SegmentTest, Multiple) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  // add5 is not a TRT candidate so two subgraphs should be formed
  //
  //                feed
  //         //      ||     ||
  //       add0    add1      add7
  //        | |    /        /   ||
  //        |  add2-----add5    add8
  //        |  /  |    |  |    |
  //       add3   add4     add6
  //           |     |     /
  //               <sink>
  //
  TF_Operation* feed = Placeholder(graph, s, "feed");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("feed"), string(TF_OperationName(feed)));

  TF_Operation* add0 = Add(feed, feed, graph, s, "add0");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add1 = Add(feed, feed, graph, s, "add1");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add7 = Add(feed, feed, graph, s, "add7");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add2 = Add(add0, add1, graph, s, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add5 = Add(add2, add7, graph, s, "add5");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add8 = Add(add7, add7, graph, s, "add8");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add3 = Add(add0, add2, graph, s, "add3");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("add3"), string(TF_OperationName(add3)));
  TF_Operation* add4 = Add(add2, add5, graph, s, "add4");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("add4"), string(TF_OperationName(add4)));
  TF_Operation* add6 = Add(add5, add8, graph, s, "add6");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("add6"), string(TF_OperationName(add6)));

  GraphDef graph_def;
  ASSERT_TRUE(GetGraphDef(graph, &graph_def));

  SegmentNodesVector segments;
  ASSERT_EQ(SegmentGraph(graph_def,
                         MakeCandidateFn({"add0", "add1", "add2", "add3",
                                          "add4", "add6", "add7", "add8"}),
                         default_options_, &segments),
            tensorflow::Status::OK());

  // Expect two subgraphs
  EXPECT_EQ(segments.size(), 2);

  std::vector<std::string> expected0{"add0", "add1", "add2", "add3"};
  for (const auto& ex : expected0) {
    EXPECT_TRUE(segments[0].find(ex) != segments[0].end())
        << "Missing expected node " << ex;
  }

  std::vector<std::string> expected1{"add6", "add8"};
  for (const auto& ex : expected1) {
    EXPECT_TRUE(segments[1].find(ex) != segments[1].end())
        << "Missing expected node " << ex;
  }
}

//------------------------------------------------------------------------------
TEST_F(SegmentTest, BigIfElse) {
  TF_Status* s = TF_NewStatus();
  TF_Graph* graph = TF_NewGraph();

  // add2 is not a TRT candidate
  //
  //           feed
  //            ||
  //           add0
  //         //    ||
  //       add1    add4
  //        ||      ||
  //       add2    add5
  //        ||      ||
  //       add3    add6
  //         ||    //
  //           add7
  //            ||
  //          <sink>
  //
  TF_Operation* feed = Placeholder(graph, s, "feed");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("feed"), string(TF_OperationName(feed)));

  TF_Operation* add0 = Add(feed, feed, graph, s, "add0");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add1 = Add(add0, add0, graph, s, "add1");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add2 = Add(add1, add1, graph, s, "add2");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add3 = Add(add2, add2, graph, s, "add3");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add4 = Add(add0, add0, graph, s, "add4");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add5 = Add(add4, add4, graph, s, "add5");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add6 = Add(add5, add5, graph, s, "add6");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  TF_Operation* add7 = Add(add3, add6, graph, s, "add7");
  ASSERT_EQ(TF_OK, TF_GetCode(s)) << TF_Message(s);
  EXPECT_EQ(string("add7"), string(TF_OperationName(add7)));

  GraphDef graph_def;
  ASSERT_TRUE(GetGraphDef(graph, &graph_def));

  SegmentNodesVector segments;
  ASSERT_EQ(SegmentGraph(graph_def,
                         MakeCandidateFn({"add0", "add1", "add3", "add4",
                                          "add5", "add6", "add7"}),
                         default_options_, &segments),
            tensorflow::Status::OK());

  // Expect 2 subgraphs
  EXPECT_EQ(segments.size(), 2);

  std::vector<std::string> expected0{"add3", "add4", "add5", "add6", "add7"};
  for (const auto& ex : expected0) {
    EXPECT_TRUE(segments[0].find(ex) != segments[0].end())
        << "Missing expected node " << ex;
  }

  std::vector<std::string> expected1{"add0", "add1"};
  for (const auto& ex : expected1) {
    EXPECT_TRUE(segments[1].find(ex) != segments[1].end())
        << "Missing expected node " << ex;
  }
}

}  // namespace test
}  // namespace segment
}  // namespace tensorrt
