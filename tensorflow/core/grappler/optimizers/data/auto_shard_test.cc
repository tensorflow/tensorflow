/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/data/auto_shard.h"

#include <string>
#include <vector>

#include "absl/strings/str_join.h"
#include "tensorflow/core/framework/dataset.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/optimizers/data/graph_test_utils.h"
#include "tensorflow/core/grappler/optimizers/data/graph_utils.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace grappler {
namespace {

using ::tensorflow::grappler::graph_tests_utils::MakeBatchV2Node;
using ::tensorflow::grappler::graph_tests_utils::MakeMapAndBatchNode;
using ::tensorflow::grappler::graph_tests_utils::MakeParallelBatchNode;
using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using ::testing::UnorderedElementsAre;

// Adds a MapDataset, a RebatchDataset, a PrefetchDataset and a fake sink that
// are common to all graphs; and sets the fetch node to the fake sink.
void FinishItem(GrapplerItem* item, const string& input_node_name) {
  *item->graph.add_node() =
      NDef("map_before_rebatch", "MapDataset", {input_node_name},
           {{"f", "__inference_Dataset_map_normalize_8232"},
            {"output_shapes", absl::Span<const TensorShape>{}},
            {"output_types", absl::Span<const DataType>{}}});
  *item->graph.add_node() =
      NDef("num_replicas", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}});
  *item->graph.add_node() =
      NDef("rebatch", "RebatchDataset", {"map_before_rebatch", "num_replicas"},
           {{"output_shapes", absl::Span<const TensorShape>{}},
            {"output_types", absl::Span<const DataType>{}}});
  *item->graph.add_node() =
      NDef("prefetch_count", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}});
  *item->graph.add_node() =
      NDef("prefetch", "PrefetchDataset", {"rebatch", "prefetch_count"},
           {{"output_shapes", absl::Span<const TensorShape>{}},
            {"output_types", absl::Span<const DataType>{}}});
  *item->graph.add_node() = NDef("Sink", "Identity", {"prefetch"}, {});
  item->fetch.push_back("Sink");
}

NodeDef AddCardinalityAttr(NodeDef node, int64_t cardinality) {
  (*node.mutable_attr())[data::kCardinalityAttrForRewrite].set_i(cardinality);
  return node;
}

TEST(RewriteBatchTest, InfiniteSource) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("repeat_count", "Const", {}, {{"value", -1}, {"dtype", DT_INT32}}),
      NDef("repeat", "RepeatDataset", {"tf_record", "repeat_count"},
           {{"output_shapes", absl::Span<const TensorShape>{}},
            {"output_types", absl::Span<const DataType>{}}}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "repeat", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kInfiniteCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, InfiniteSourceMapAndBatch) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("repeat_count", "Const", {}, {{"value", -1}, {"dtype", DT_INT32}}),
      NDef("repeat", "RepeatDataset", {"tf_record", "repeat_count"},
           {{"output_shapes", absl::Span<const TensorShape>{}},
            {"output_types", absl::Span<const DataType>{}}}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("num_parallel_calls", "Const", {},
           {{"value", 2}, {"dtype", DT_INT64}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeMapAndBatchNode("batch", "repeat", "batch_size",
                              "num_parallel_calls", "drop_remainder"),
          data::kInfiniteCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, InfiniteSourceParallelBatch) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("repeat_count", "Const", {}, {{"value", -1}, {"dtype", DT_INT32}}),
      NDef("repeat", "RepeatDataset", {"tf_record", "repeat_count"},
           {{"output_shapes", absl::Span<const TensorShape>{}},
            {"output_types", absl::Span<const DataType>{}}}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("num_parallel_calls", "Const", {},
           {{"value", 2}, {"dtype", DT_INT64}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeParallelBatchNode("batch", "repeat", "batch_size",
                                "num_parallel_calls", "drop_remainder",
                                /*deterministic=*/"true"),
          data::kInfiniteCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, FiniteSourceNoDropRemainder) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", false}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kUnknownCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

TEST(RewriteBatchTest, FiniteSourceDropRemainder) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          /*cardinality=*/1337),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_DROP_REMAINDER_NOT_INFINITE"));
}

TEST(RewriteBatchTest, UnknownCardinalitySourceDropRemainder) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kUnknownCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_DROP_REMAINDER_NOT_INFINITE"));
}

TEST(RewriteBatchTest, FiniteSourceDropRemainderUnknown) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "RandomBool", {}, {}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kUnknownCardinality),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_DROP_REMAINDER_UNKNOWN"));
}

TEST(RewriteBatchTest, DropRemainderCardinalityNotAvailable) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {}, {{"value", true}}),
      MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                      /*parallel_copy=*/false),
  });
  FinishItem(&item, "batch");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("BATCH_CARDINALITY_NOT_AVAILABLE"));
}

TEST(RewriteBatchTest, OpNotSupported) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "tf_record", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kUnknownCardinality),
      NDef("take_count", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      graph_tests_utils::MakeTakeNode("take", "batch", "take_count"),
  });
  FinishItem(&item, "take");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason,
              UnorderedElementsAre("OP_NOT_SUPPORTED_TakeDataset",
                                   "BATCH_DROP_REMAINDER_NOT_INFINITE"));
}

TEST(RewriteBatchTest, BatchNotFound) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      graph_tests_utils::MakeTakeNode("take", "tf_record", "take_count"),
      NDef("take_count", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
  });
  FinishItem(&item, "take");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_FALSE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                    &ineligible_reason));
  EXPECT_THAT(ineligible_reason, UnorderedElementsAre("BATCH_NOT_FOUND"));
}

// This is a very rare case (OneDeviceStrategy).
TEST(RewriteBatchTest, InfiniteSourceNoRebatch) {
  GrapplerItem item;
  item.graph = GDef({
      NDef("files", "Const", {},
           {{"values", std::vector<std::string>{"file1", "file2"}},
            {"dtype", DT_STRING}}),
      NDef("tf_record", "TFRecordDataset", {"file"}, {}),
      NDef("repeat_count", "Const", {}, {{"value", -1}, {"dtype", DT_INT32}}),
      NDef("repeat", "RepeatDataset", {"tf_record", "repeat_count"},
           {{"output_shapes", absl::Span<const TensorShape>{}},
            {"output_types", absl::Span<const DataType>{}}}),
      NDef("batch_size", "Const", {}, {{"value", 2}, {"dtype", DT_INT32}}),
      NDef("drop_remainder", "Const", {},
           {{"value", true}, {"dtype", DT_BOOL}}),
      AddCardinalityAttr(
          MakeBatchV2Node("batch", "repeat", "batch_size", "drop_remainder",
                          /*parallel_copy=*/false),
          data::kInfiniteCardinality),
      NDef("Sink", "Identity", {"batch"}, {}),
  });
  item.fetch.push_back("Sink");

  MutableGraphView graph(&item.graph);
  NodeDef* sink_node = nullptr;
  TF_ASSERT_OK(graph_utils::GetFetchNode(graph, item, &sink_node));
  std::vector<std::string> ineligible_reason;
  EXPECT_TRUE(internal::IsEligibleRewriteBatchSize(*sink_node, graph,
                                                   &ineligible_reason))
      << absl::StrJoin(ineligible_reason, ",");
}

}  // namespace
}  // namespace grappler
}  // namespace tensorflow
