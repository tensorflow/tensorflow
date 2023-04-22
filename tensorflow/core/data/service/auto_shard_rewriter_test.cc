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
#include "tensorflow/core/data/service/auto_shard_rewriter.h"

#include <string>

#include "absl/strings/string_view.h"
#include "absl/strings/substitute.h"
#include "tensorflow/core/data/service/common.pb.h"
#include "tensorflow/core/data/service/test_util.h"
#include "tensorflow/core/framework/dataset_options.pb.h"
#include "tensorflow/core/framework/function_testlib.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_testutil.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::testing::RangeSquareDataset;
using ::tensorflow::test::AsScalar;
using ::tensorflow::test::function::GDef;
using ::tensorflow::test::function::NDef;
using ::tensorflow::testing::IsOkAndHolds;
using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::MakePolymorphicMatcher;
using ::testing::MatchResultListener;
using ::testing::PolymorphicMatcher;
using ::testing::SizeIs;

constexpr int64 kShardHint = -1;

DatasetDef RangeDatasetWithShardHint(const int64 range) {
  DatasetDef dataset_def;
  *dataset_def.mutable_graph() = GDef(
      {NDef("start", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64>(0)}, {"dtype", DT_INT64}}),
       NDef("stop", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64>(range)}, {"dtype", DT_INT64}}),
       NDef("step", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64>(1)}, {"dtype", DT_INT64}}),
       NDef("range", "RangeDataset", /*inputs=*/{"start", "stop", "step"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
             {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}}),
       NDef("num_shards", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64>(kShardHint)}, {"dtype", DT_INT64}}),
       NDef("index", "Const", /*inputs=*/{},
            {{"value", AsScalar<int64>(kShardHint)}, {"dtype", DT_INT64}}),
       NDef("ShardDataset", "ShardDataset",
            /*inputs=*/{"range", "num_shards", "index"},
            {{"output_shapes", gtl::ArraySlice<TensorShape>{TensorShape()}},
             {"output_types", gtl::ArraySlice<DataType>{DT_INT64}}}),
       NDef("dataset", "_Retval", /*inputs=*/{"ShardDataset"},
            {{"T", DT_VARIANT}, {"index", 0}})},
      /*funcs=*/{});
  return dataset_def;
}

StatusOr<NodeDef> GetNode(const GraphDef& graph_def, absl::string_view name) {
  for (const NodeDef& node : graph_def.node()) {
    if (node.name() == name) {
      return node;
    }
  }
  return errors::NotFound(absl::Substitute("Node $0 not found in graph $1.",
                                           name, graph_def.ShortDebugString()));
}

StatusOr<int64> GetValue(const GraphDef& graph_def, absl::string_view name) {
  for (const NodeDef& node : graph_def.node()) {
    if (node.name() == name) {
      return node.attr().at("value").tensor().int64_val()[0];
    }
  }
  return errors::NotFound(absl::Substitute("Node $0 not found in graph $1.",
                                           name, graph_def.ShortDebugString()));
}

// TODO(yangchen): Make EqualsProto available in Googletest
// (https://github.com/google/googletest/issues/1761).
class ProtoStringMatcher {
 public:
  explicit ProtoStringMatcher(const tensorflow::protobuf::Message& expected)
      : expected_(expected.ShortDebugString()) {}

  template <typename Message>
  bool MatchAndExplain(const Message& p, MatchResultListener*) const {
    return p.ShortDebugString() == expected_;
  }

  void DescribeTo(::std::ostream* os) const { *os << expected_; }
  void DescribeNegationTo(::std::ostream* os) const {
    *os << "not equal to expected message: " << expected_;
  }

 private:
  const std::string expected_;
};

inline ::testing::PolymorphicMatcher<ProtoStringMatcher> EqualsProto(
    const tensorflow::protobuf::Message& x) {
  return ::testing::MakePolymorphicMatcher(ProtoStringMatcher(x));
}

TEST(AutoShardRewriterTest, AutoShard) {
  DatasetDef dataset = RangeSquareDataset(10);
  TF_ASSERT_OK_AND_ASSIGN(
      AutoShardRewriter rewriter,
      AutoShardRewriter::Create(
          AutoShardPolicy::AUTO,
          /*worker_addresses=*/{"worker0", "worker1", "worker2"},
          /*worker_address=*/"worker1"));
  TF_ASSERT_OK_AND_ASSIGN(GraphDef rewritten_graph,
                          rewriter.ApplyAutoShardRewrite(dataset.graph()));
  TF_ASSERT_OK_AND_ASSIGN(NodeDef shard_node,
                          GetNode(rewritten_graph, "ShardDataset"));
  ASSERT_THAT(shard_node.input(), SizeIs(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(1)), IsOkAndHolds(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(2)), IsOkAndHolds(1));
}

TEST(AutoShardRewriterTest, ShardByData) {
  DatasetDef dataset = RangeSquareDataset(10);
  TF_ASSERT_OK_AND_ASSIGN(
      AutoShardRewriter rewriter,
      AutoShardRewriter::Create(
          AutoShardPolicy::DATA,
          /*worker_addresses=*/{"worker0", "worker1", "worker2"},
          /*worker_address=*/"worker1"));
  TF_ASSERT_OK_AND_ASSIGN(GraphDef rewritten_graph,
                          rewriter.ApplyAutoShardRewrite(dataset.graph()));
  TF_ASSERT_OK_AND_ASSIGN(NodeDef shard_node,
                          GetNode(rewritten_graph, "ShardDataset"));
  ASSERT_THAT(shard_node.input(), SizeIs(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(1)), IsOkAndHolds(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(2)), IsOkAndHolds(1));
}

TEST(AutoShardRewriterTest, ShardByFile) {
  DatasetDef dataset = RangeSquareDataset(10);
  TF_ASSERT_OK_AND_ASSIGN(
      AutoShardRewriter rewriter,
      AutoShardRewriter::Create(
          AutoShardPolicy::FILE,
          /*worker_addresses=*/{"worker0", "worker1", "worker2"},
          /*worker_address=*/"worker1"));
  EXPECT_THAT(rewriter.ApplyAutoShardRewrite(dataset.graph()),
              StatusIs(error::NOT_FOUND,
                       HasSubstr("Found an unshardable source dataset")));
}

TEST(AutoShardRewriterTest, ShardByHint) {
  DatasetDef dataset = RangeDatasetWithShardHint(10);
  TF_ASSERT_OK_AND_ASSIGN(
      AutoShardRewriter rewriter,
      AutoShardRewriter::Create(
          AutoShardPolicy::HINT,
          /*worker_addresses=*/{"worker0", "worker1", "worker2"},
          /*worker_address=*/"worker1"));
  TF_ASSERT_OK_AND_ASSIGN(GraphDef rewritten_graph,
                          rewriter.ApplyAutoShardRewrite(dataset.graph()));
  TF_ASSERT_OK_AND_ASSIGN(NodeDef shard_node,
                          GetNode(rewritten_graph, "ShardDataset"));
  ASSERT_THAT(shard_node.input(), SizeIs(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(1)), IsOkAndHolds(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(2)), IsOkAndHolds(1));
}

TEST(AutoShardRewriterTest, NoShard) {
  DatasetDef dataset = RangeSquareDataset(10);
  TF_ASSERT_OK_AND_ASSIGN(
      AutoShardRewriter rewriter,
      AutoShardRewriter::Create(
          AutoShardPolicy::OFF,
          /*worker_addresses=*/{"worker0", "worker1", "worker2"},
          /*worker_address=*/"worker1"));
  EXPECT_THAT(rewriter.ApplyAutoShardRewrite(dataset.graph()),
              IsOkAndHolds(EqualsProto(dataset.graph())));
}

TEST(AutoShardRewriterTest, EmptyDataset) {
  DatasetDef dataset = RangeSquareDataset(0);
  TF_ASSERT_OK_AND_ASSIGN(
      AutoShardRewriter rewriter,
      AutoShardRewriter::Create(
          AutoShardPolicy::AUTO,
          /*worker_addresses=*/{"worker0", "worker1", "worker2"},
          /*worker_address=*/"worker1"));
  TF_ASSERT_OK_AND_ASSIGN(GraphDef rewritten_graph,
                          rewriter.ApplyAutoShardRewrite(dataset.graph()));
  TF_ASSERT_OK_AND_ASSIGN(NodeDef shard_node,
                          GetNode(rewritten_graph, "ShardDataset"));
  ASSERT_THAT(shard_node.input(), SizeIs(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(1)), IsOkAndHolds(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(2)), IsOkAndHolds(1));
}

TEST(AutoShardRewriterTest, WorkerNotFound) {
  EXPECT_THAT(AutoShardRewriter::Create(AutoShardPolicy::AUTO,
                                        /*worker_addresses=*/{},
                                        /*worker_address=*/"worker1"),
              StatusIs(error::NOT_FOUND,
                       HasSubstr("Worker worker1 is not in the auto-shard "
                                 "workers list.")));
}

TEST(AutoShardRewriterTest, WorkerListIsEmptyWhenShardIsOff) {
  DatasetDef dataset = RangeSquareDataset(10);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(AutoShardPolicy::OFF,
                                                    /*worker_addresses=*/{},
                                                    /*worker_address=*/""));
  EXPECT_THAT(rewriter.ApplyAutoShardRewrite(dataset.graph()),
              IsOkAndHolds(EqualsProto(dataset.graph())));
}

}  // namespace
}  // namespace data
}  // namespace tensorflow
