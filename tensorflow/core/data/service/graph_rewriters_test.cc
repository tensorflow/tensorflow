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
#include "tensorflow/core/data/service/graph_rewriters.h"

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
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/tstring.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/protobuf/data_service.pb.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace data {
namespace {

using ::tensorflow::data::testing::EqualsProto;
using ::tensorflow::data::testing::RangeDatasetWithShardHint;
using ::tensorflow::data::testing::RangeSquareDataset;
using ::tensorflow::testing::IsOkAndHolds;
using ::tensorflow::testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::SizeIs;

absl::StatusOr<NodeDef> GetNode(const GraphDef& graph_def,
                                absl::string_view name) {
  for (const NodeDef& node : graph_def.node()) {
    if (node.name() == name) {
      return node;
    }
  }
  return errors::NotFound(absl::Substitute("Node $0 not found in graph $1.",
                                           name, graph_def.ShortDebugString()));
}

absl::StatusOr<int64_t> GetValue(const GraphDef& graph_def,
                                 absl::string_view name) {
  for (const NodeDef& node : graph_def.node()) {
    if (node.name() == name) {
      return node.attr().at("value").tensor().int64_val()[0];
    }
  }
  return errors::NotFound(absl::Substitute("Node $0 not found in graph $1.",
                                           name, graph_def.ShortDebugString()));
}

TaskDef GetTaskDef(const ProcessingModeDef::ShardingPolicy sharding_policy,
                   const int64_t num_workers, const int64_t worker_index) {
  TaskDef task_def;
  task_def.mutable_processing_mode_def()->set_sharding_policy(sharding_policy);
  task_def.set_num_workers(num_workers);
  task_def.set_worker_index(worker_index);
  return task_def;
}

TEST(AutoShardRewriterTest, AutoShard) {
  TaskDef task_def = GetTaskDef(ProcessingModeDef::FILE_OR_DATA,
                                /*num_workers=*/3, /*worker_index=*/1);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(task_def));

  DatasetDef dataset = RangeSquareDataset(10);
  TF_ASSERT_OK_AND_ASSIGN(GraphDef rewritten_graph,
                          rewriter.ApplyAutoShardRewrite(dataset.graph()));
  TF_ASSERT_OK_AND_ASSIGN(NodeDef shard_node,
                          GetNode(rewritten_graph, "ShardDataset"));
  ASSERT_THAT(shard_node.input(), SizeIs(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(1)), IsOkAndHolds(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(2)), IsOkAndHolds(1));
}

TEST(AutoShardRewriterTest, ShardByData) {
  TaskDef task_def = GetTaskDef(ProcessingModeDef::DATA, /*num_workers=*/3,
                                /*worker_index=*/1);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(task_def));

  DatasetDef dataset = RangeSquareDataset(10);
  TF_ASSERT_OK_AND_ASSIGN(GraphDef rewritten_graph,
                          rewriter.ApplyAutoShardRewrite(dataset.graph()));
  TF_ASSERT_OK_AND_ASSIGN(NodeDef shard_node,
                          GetNode(rewritten_graph, "ShardDataset"));
  ASSERT_THAT(shard_node.input(), SizeIs(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(1)), IsOkAndHolds(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(2)), IsOkAndHolds(1));
}

TEST(AutoShardRewriterTest, ShardByFile) {
  TaskDef task_def = GetTaskDef(ProcessingModeDef::FILE, /*num_workers=*/3,
                                /*worker_index=*/1);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(task_def));

  DatasetDef dataset = RangeSquareDataset(10);
  EXPECT_THAT(rewriter.ApplyAutoShardRewrite(dataset.graph()),
              StatusIs(error::NOT_FOUND,
                       HasSubstr("Found an unshardable source dataset")));
}

TEST(AutoShardRewriterTest, ShardByHint) {
  TaskDef task_def = GetTaskDef(ProcessingModeDef::HINT, /*num_workers=*/3,
                                /*worker_index=*/1);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(task_def));

  DatasetDef dataset = RangeDatasetWithShardHint(10);
  TF_ASSERT_OK_AND_ASSIGN(GraphDef rewritten_graph,
                          rewriter.ApplyAutoShardRewrite(dataset.graph()));
  TF_ASSERT_OK_AND_ASSIGN(NodeDef shard_node,
                          GetNode(rewritten_graph, "ShardDataset"));
  ASSERT_THAT(shard_node.input(), SizeIs(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(1)), IsOkAndHolds(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(2)), IsOkAndHolds(1));
}

TEST(AutoShardRewriterTest, NoShard) {
  TaskDef task_def =
      GetTaskDef(ProcessingModeDef::OFF, /*num_workers=*/3, /*worker_index=*/1);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(task_def));

  DatasetDef dataset = RangeSquareDataset(10);
  EXPECT_THAT(rewriter.ApplyAutoShardRewrite(dataset.graph()),
              IsOkAndHolds(EqualsProto(dataset.graph())));
}

TEST(AutoShardRewriterTest, EmptyDataset) {
  TaskDef task_def =
      GetTaskDef(ProcessingModeDef::FILE_OR_DATA, /*num_workers=*/3,
                 /*worker_index=*/1);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(task_def));

  DatasetDef dataset = RangeSquareDataset(0);
  TF_ASSERT_OK_AND_ASSIGN(GraphDef rewritten_graph,
                          rewriter.ApplyAutoShardRewrite(dataset.graph()));
  TF_ASSERT_OK_AND_ASSIGN(NodeDef shard_node,
                          GetNode(rewritten_graph, "ShardDataset"));
  ASSERT_THAT(shard_node.input(), SizeIs(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(1)), IsOkAndHolds(3));
  EXPECT_THAT(GetValue(rewritten_graph, shard_node.input(2)), IsOkAndHolds(1));
}

TEST(AutoShardRewriterTest, NoWorkers) {
  TaskDef task_def =
      GetTaskDef(ProcessingModeDef::FILE_OR_DATA, /*num_workers=*/0,
                 /*worker_index=*/0);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(task_def));

  DatasetDef dataset = RangeSquareDataset(10);
  EXPECT_THAT(rewriter.ApplyAutoShardRewrite(dataset.graph()),
              StatusIs(error::INVALID_ARGUMENT,
                       "num_workers should be >= 1, currently 0"));
}

TEST(AutoShardRewriterTest, NoWorkersWhenShardIsOff) {
  TaskDef task_def =
      GetTaskDef(ProcessingModeDef::OFF, /*num_workers=*/0, /*worker_index=*/0);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(task_def));

  DatasetDef dataset = RangeSquareDataset(10);
  EXPECT_THAT(rewriter.ApplyAutoShardRewrite(dataset.graph()),
              IsOkAndHolds(EqualsProto(dataset.graph())));
}

TEST(AutoShardRewriterTest, WorkerIndexOutOfRange) {
  TaskDef task_def =
      GetTaskDef(ProcessingModeDef::FILE_OR_DATA, /*num_workers=*/2,
                 /*worker_index=*/5);
  TF_ASSERT_OK_AND_ASSIGN(AutoShardRewriter rewriter,
                          AutoShardRewriter::Create(task_def));

  DatasetDef dataset = RangeSquareDataset(10);
  EXPECT_THAT(rewriter.ApplyAutoShardRewrite(dataset.graph()),
              StatusIs(error::INVALID_ARGUMENT,
                       "index should be >= 0 and < 2, currently 5"));
}

TEST(WorkerIndexResolverTest, AddOneWorker) {
  WorkerIndexResolver resolver(std::vector<std::string>{"localhost"});
  EXPECT_THAT(resolver.GetWorkerIndex("localhost:12345"),
              StatusIs(error::NOT_FOUND));

  TF_EXPECT_OK(resolver.ValidateWorker("localhost:12345"));
  resolver.AddWorker("localhost:12345");
  EXPECT_THAT(resolver.GetWorkerIndex("localhost:12345"), IsOkAndHolds(0));
}

TEST(WorkerIndexResolverTest, AddMultipleWorkers) {
  WorkerIndexResolver resolver(std::vector<std::string>{
      "/worker/task/0", "/worker/task/1", "/worker/task/2"});
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/2:12345"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/1:23456"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/0:34567"));
  resolver.AddWorker("/worker/task/2:12345");
  resolver.AddWorker("/worker/task/1:23456");
  resolver.AddWorker("/worker/task/0:34567");
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/0:34567"), IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/1:23456"), IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/2:12345"), IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, NamedPorts) {
  WorkerIndexResolver resolver(
      std::vector<std::string>{"/worker/task/0:worker", "/worker/task/1:worker",
                               "/worker/task/2:worker"});
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/2:worker"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/1:worker"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/0:worker"));
  resolver.AddWorker("/worker/task/2:worker");
  resolver.AddWorker("/worker/task/1:worker");
  resolver.AddWorker("/worker/task/0:worker");
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/0:worker"),
              IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/1:worker"),
              IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/2:worker"),
              IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, DynamicPorts) {
  WorkerIndexResolver resolver(std::vector<std::string>{
      "/worker/task/0:%port_worker%", "/worker/task/1:%port_worker%",
      "/worker/task/2:%port_worker%"});
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/2:worker"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/1:worker"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/0:worker"));
  resolver.AddWorker("/worker/task/2:worker");
  resolver.AddWorker("/worker/task/1:worker");
  resolver.AddWorker("/worker/task/0:worker");
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/0:worker"),
              IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/1:worker"),
              IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/2:worker"),
              IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, AnonymousPorts) {
  WorkerIndexResolver resolver(
      std::vector<std::string>{"/worker/task/0:%port%", "/worker/task/1:%port%",
                               "/worker/task/2:%port%"});
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/2:10000"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/1:10001"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/0:10002"));
  resolver.AddWorker("/worker/task/2:10000");
  resolver.AddWorker("/worker/task/1:10001");
  resolver.AddWorker("/worker/task/0:10002");
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/0:10002"), IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/1:10001"), IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/2:10000"), IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, NumericPorts) {
  WorkerIndexResolver resolver(std::vector<std::string>{
      "/worker/task/0:12345", "/worker/task/1:23456", "/worker/task/2:34567"});
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/0:12345"), IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/1:23456"), IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/2:34567"), IsOkAndHolds(2));

  // Adding duplicate workers is a no-op.
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/2:34567"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/1:23456"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/0:12345"));
  resolver.AddWorker("/worker/task/2:34567");
  resolver.AddWorker("/worker/task/1:23456");
  resolver.AddWorker("/worker/task/0:12345");
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/0:12345"), IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/1:23456"), IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/2:34567"), IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, IPv6Addresses) {
  WorkerIndexResolver resolver(std::vector<std::string>{
      "[1080:0:0:0:8:800:200C:417A]", "[1080:0:0:0:8:800:200C:417B]",
      "[1080:0:0:0:8:800:200C:417C]"});
  TF_EXPECT_OK(resolver.ValidateWorker("[1080:0:0:0:8:800:200C:417A]:12345"));
  TF_EXPECT_OK(resolver.ValidateWorker("[1080:0:0:0:8:800:200C:417B]:23456"));
  TF_EXPECT_OK(resolver.ValidateWorker("[1080:0:0:0:8:800:200C:417C]:34567"));
  resolver.AddWorker("[1080:0:0:0:8:800:200C:417A]:12345");
  resolver.AddWorker("[1080:0:0:0:8:800:200C:417B]:23456");
  resolver.AddWorker("[1080:0:0:0:8:800:200C:417C]:34567");
  EXPECT_THAT(resolver.GetWorkerIndex("[1080:0:0:0:8:800:200C:417A]:12345"),
              IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("[1080:0:0:0:8:800:200C:417B]:23456"),
              IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("[1080:0:0:0:8:800:200C:417C]:34567"),
              IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, IPv6AddressesWithDynamicPort) {
  WorkerIndexResolver resolver(
      std::vector<std::string>{"[1080:0:0:0:8:800:200C:417A]:%port%",
                               "[1080:0:0:0:8:800:200C:417B]:%port%",
                               "[1080:0:0:0:8:800:200C:417C]:%port%"});
  TF_EXPECT_OK(resolver.ValidateWorker("[1080:0:0:0:8:800:200C:417A]:12345"));
  TF_EXPECT_OK(resolver.ValidateWorker("[1080:0:0:0:8:800:200C:417B]:23456"));
  TF_EXPECT_OK(resolver.ValidateWorker("[1080:0:0:0:8:800:200C:417C]:34567"));
  resolver.AddWorker("[1080:0:0:0:8:800:200C:417A]:12345");
  resolver.AddWorker("[1080:0:0:0:8:800:200C:417B]:23456");
  resolver.AddWorker("[1080:0:0:0:8:800:200C:417C]:34567");
  EXPECT_THAT(resolver.GetWorkerIndex("[1080:0:0:0:8:800:200C:417A]:12345"),
              IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("[1080:0:0:0:8:800:200C:417B]:23456"),
              IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("[1080:0:0:0:8:800:200C:417C]:34567"),
              IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, AddressesWithProtocols) {
  WorkerIndexResolver resolver(std::vector<std::string>{
      "http://127.0.0.1", "http://127.0.0.1", "http://127.0.0.1"});
  TF_EXPECT_OK(resolver.ValidateWorker("http://127.0.0.1:12345"));
  TF_EXPECT_OK(resolver.ValidateWorker("http://127.0.0.1:23456"));
  TF_EXPECT_OK(resolver.ValidateWorker("http://127.0.0.1:34567"));
  resolver.AddWorker("http://127.0.0.1:12345");
  resolver.AddWorker("http://127.0.0.1:23456");
  resolver.AddWorker("http://127.0.0.1:34567");
  EXPECT_THAT(resolver.GetWorkerIndex("http://127.0.0.1:12345"),
              IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("http://127.0.0.1:23456"),
              IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("http://127.0.0.1:34567"),
              IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, AddressesWithProtocolsAndDynamicPorts) {
  WorkerIndexResolver resolver(std::vector<std::string>{
      "http://127.0.0.1:%port_name%", "http://127.0.0.1:%port_name%",
      "http://127.0.0.1:%port_name%"});
  TF_EXPECT_OK(resolver.ValidateWorker("http://127.0.0.1:12345"));
  TF_EXPECT_OK(resolver.ValidateWorker("http://127.0.0.1:23456"));
  TF_EXPECT_OK(resolver.ValidateWorker("http://127.0.0.1:34567"));
  resolver.AddWorker("http://127.0.0.1:12345");
  resolver.AddWorker("http://127.0.0.1:23456");
  resolver.AddWorker("http://127.0.0.1:34567");
  EXPECT_THAT(resolver.GetWorkerIndex("http://127.0.0.1:12345"),
              IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("http://127.0.0.1:23456"),
              IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("http://127.0.0.1:34567"),
              IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, HostNameHasColons) {
  WorkerIndexResolver resolver(
      std::vector<std::string>{":worker:task:0:%port%", ":worker:task:1:%port%",
                               ":worker:task:2:34567"});
  TF_EXPECT_OK(resolver.ValidateWorker(":worker:task:0:12345"));
  TF_EXPECT_OK(resolver.ValidateWorker(":worker:task:1:23456"));
  TF_EXPECT_OK(resolver.ValidateWorker(":worker:task:2:34567"));
  resolver.AddWorker(":worker:task:0:12345");
  resolver.AddWorker(":worker:task:1:23456");
  resolver.AddWorker(":worker:task:2:34567");
  EXPECT_THAT(resolver.GetWorkerIndex(":worker:task:0:12345"), IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex(":worker:task:1:23456"), IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex(":worker:task:2:34567"), IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, ChangeWorkerPort) {
  WorkerIndexResolver resolver(std::vector<std::string>{
      "/worker/task/0", "/worker/task/1", "/worker/task/2"});
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/2:12345"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/1:23456"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/0:34567"));
  resolver.AddWorker("/worker/task/2:12345");
  resolver.AddWorker("/worker/task/1:23456");
  resolver.AddWorker("/worker/task/0:34567");
  EXPECT_THAT(resolver.ValidateWorker("/worker/task/0:99999"),
              StatusIs(error::FAILED_PRECONDITION,
                       HasSubstr("already running at the configured host")));
  EXPECT_THAT(resolver.ValidateWorker("/worker/task/1:99999"),
              StatusIs(error::FAILED_PRECONDITION,
                       HasSubstr("already running at the configured host")));
  EXPECT_THAT(resolver.ValidateWorker("/worker/task/2:99999"),
              StatusIs(error::FAILED_PRECONDITION,
                       HasSubstr("already running at the configured host")));
}

TEST(WorkerIndexResolverTest, WorkerNotFound) {
  WorkerIndexResolver resolver(std::vector<std::string>{
      "/worker/task/0", "/worker/task/1", "/worker/task/2"});
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/0:34567"),
              StatusIs(error::NOT_FOUND));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/1:23456"),
              StatusIs(error::NOT_FOUND));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/2:12345"),
              StatusIs(error::NOT_FOUND));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/3:45678"),
              StatusIs(error::NOT_FOUND));

  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/2:12345"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/1:23456"));
  TF_EXPECT_OK(resolver.ValidateWorker("/worker/task/0:34567"));
  EXPECT_THAT(resolver.ValidateWorker("/worker/task/3:45678"),
              StatusIs(error::FAILED_PRECONDITION,
                       HasSubstr("The worker's address is not configured")));
  resolver.AddWorker("/worker/task/3:45678");
  resolver.AddWorker("/worker/task/2:12345");
  resolver.AddWorker("/worker/task/1:23456");
  resolver.AddWorker("/worker/task/0:34567");

  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/0:34567"), IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/1:23456"), IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("/worker/task/2:12345"), IsOkAndHolds(2));
  EXPECT_THAT(
      resolver.GetWorkerIndex("/worker/task/3:45678"),
      StatusIs(error::NOT_FOUND,
               HasSubstr(
                   "Worker /worker/task/3:45678 is not in the workers list.")));
}

TEST(WorkerIndexResolverTest, MultipleWorkersInOneHost) {
  WorkerIndexResolver resolver(
      std::vector<std::string>{"localhost", "localhost", "localhost"});
  TF_EXPECT_OK(resolver.ValidateWorker("localhost:12345"));
  resolver.AddWorker("localhost:12345");
  TF_EXPECT_OK(resolver.ValidateWorker("localhost:23456"));
  resolver.AddWorker("localhost:23456");
  TF_EXPECT_OK(resolver.ValidateWorker("localhost:34567"));
  resolver.AddWorker("localhost:34567");
  EXPECT_THAT(resolver.GetWorkerIndex("localhost:12345"), IsOkAndHolds(0));
  EXPECT_THAT(resolver.GetWorkerIndex("localhost:23456"), IsOkAndHolds(1));
  EXPECT_THAT(resolver.GetWorkerIndex("localhost:34567"), IsOkAndHolds(2));
}

TEST(WorkerIndexResolverTest, MoreWorkersThanConfigured) {
  WorkerIndexResolver resolver(std::vector<std::string>{
      "localhost:%port%", "localhost:%port%", "localhost:%port%"});
  TF_EXPECT_OK(resolver.ValidateWorker("localhost:12345"));
  resolver.AddWorker("localhost:12345");
  TF_EXPECT_OK(resolver.ValidateWorker("localhost:23456"));
  resolver.AddWorker("localhost:23456");
  TF_EXPECT_OK(resolver.ValidateWorker("localhost:34567"));
  resolver.AddWorker("localhost:34567");
  TF_EXPECT_OK(resolver.ValidateWorker("localhost:12345"));
  resolver.AddWorker("localhost:12345");
  TF_EXPECT_OK(resolver.ValidateWorker("localhost:23456"));
  resolver.AddWorker("localhost:23456");
  TF_EXPECT_OK(resolver.ValidateWorker("localhost:34567"));
  resolver.AddWorker("localhost:34567");
  EXPECT_THAT(resolver.ValidateWorker("localhost:45678"),
              StatusIs(error::FAILED_PRECONDITION,
                       HasSubstr("already running at the configured host")));
  EXPECT_THAT(resolver.ValidateWorker("localhost:56789"),
              StatusIs(error::FAILED_PRECONDITION,
                       HasSubstr("already running at the configured host")));
}

TEST(WorkerIndexResolverTest, WorkerNotConfigured) {
  WorkerIndexResolver resolver(std::vector<std::string>{""});
  EXPECT_THAT(resolver.GetWorkerIndex("localhost:12345"),
              StatusIs(error::NOT_FOUND));
  EXPECT_THAT(resolver.ValidateWorker("localhost:12345"),
              StatusIs(error::FAILED_PRECONDITION,
                       HasSubstr("The worker's address is not configured")));
  resolver.AddWorker("localhost:12345");
  EXPECT_THAT(resolver.GetWorkerIndex("localhost:12345"),
              StatusIs(error::NOT_FOUND));
}
}  // namespace
}  // namespace data
}  // namespace tensorflow
