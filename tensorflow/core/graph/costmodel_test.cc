/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/graph/costmodel.h"

#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/common_runtime/costmodel_manager.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/common_runtime/step_stats_collector.h"
#include "tensorflow/core/framework/allocation_description.pb.h"
#include "tensorflow/core/framework/cost_graph.pb.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/framework/tensor_description.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/types.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {
namespace {

using ::testing::Not;

// Work-around for lack of ShapeProtoEquals in OSS.
MATCHER_P(ShapeProtoEquals, other, "") {
  if (arg.unknown_rank()) {
    return other.unknown_rank();
  }
  if (arg.dim_size() != other.dim_size()) {
    return false;
  }
  for (int i = 0; i < arg.dim_size(); ++i) {
    if (arg.dim(i).size() != other.dim(i).size()) {
      return false;
    }
  }
  return true;
}

static void InitGraph(const string& s, Graph* graph) {
  GraphDef graph_def;

  auto parser = protobuf::TextFormat::Parser();
  CHECK(parser.MergeFromString(s, &graph_def)) << s;
  GraphConstructorOptions opts;
  TF_CHECK_OK(ConvertGraphDefToGraph(opts, graph_def, graph));
}

static void InitModelFromGraph(const Graph& graph, CostModel& cm) {
  // This adjusts the model to include all of the graph's nodes.
  // Unlike CostModel::InitFromGraph(), this method does not add
  // default estimates for sizes or times.
  for (const auto& node : graph.nodes()) {
    cm.SetNumOutputs(node, node->num_outputs());
  }
}

// Creates a graph with two multiply nodes.
static std::unique_ptr<Graph> CreateBasicTestGraph() {
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  InitGraph(
      "node { name: 'A' op: 'Input'}"
      "node { name: 'B' op: 'Input'}"
      "node { name: 'C' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }"
      "node { name: 'D' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A', 'B'] }",
      graph.get());
  return graph;
}

Node* FindNode(const Graph& graph, std::string name) {
  for (const auto& node : graph.nodes()) {
    if (node->name() == name) {
      return node;
    }
  }
  return nullptr;
}

Node* AddNode(Graph& graph, const string& name, const string& node_type,
              int num_inputs) {
  auto builder = NodeDefBuilder(name, node_type);
  for (int i = 0; i < num_inputs; ++i) {
    builder = builder.Input(absl::StrCat("node_", i), i, DT_FLOAT);
  }

  NodeDef node_def;
  TF_CHECK_OK(builder.Finalize(&node_def));

  absl::Status s;
  Node* node = graph.AddNode(node_def, &s);
  TF_CHECK_OK(s);
  return node;
}

static void GenerateStepStats(Graph* graph, StepStats* step_stats,
                              const string& device_name) {
  // Fill RunMetadata's step_stats and partition_graphs fields.
  DeviceStepStats* device_stepstats = step_stats->add_dev_stats();
  device_stepstats->set_device(device_name);
  for (const auto& node_def : graph->nodes()) {
    NodeExecStats* node_stats = device_stepstats->add_node_stats();
    node_stats->set_node_name(node_def->name());
  }
}

REGISTER_OP("Input").Output("o: float").SetIsStateful();

TEST(CostModelTest, WorksWithManager) {
  Scope scope = Scope::NewRootScope().ExitOnError();
  auto graph1 = std::make_unique<Graph>(OpRegistry::Global());
  auto graph2 = std::make_unique<Graph>(OpRegistry::Global());
  InitGraph(
      "node { name: 'A1' op: 'Input'}"
      "node { name: 'B1' op: 'Input'}"
      "node { name: 'C1' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A1', 'B1'] }"
      "node { name: 'D1' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A1', 'B1'] }",
      graph1.get());
  InitGraph(
      "node { name: 'A2' op: 'Input'}"
      "node { name: 'B2' op: 'Input'}"
      "node { name: 'C2' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A2', 'B2'] }"
      "node { name: 'D2' op: 'Mul' attr { key: 'T' value { type: DT_FLOAT } }"
      " input: ['A2', 'B2'] }",
      graph2.get());
  StepStats step_stats;
  GenerateStepStats(graph1.get(), &step_stats, "DummyDevice1");
  GenerateStepStats(graph2.get(), &step_stats, "DummyDevice2");
  StepStatsCollector collector(&step_stats);
  std::unordered_map<string, const Graph*> device_map;
  device_map["DummyDevice1"] = graph1.get();
  device_map["DummyDevice2"] = graph2.get();
  CostModelManager cost_model_manager;
  collector.BuildCostModel(&cost_model_manager, device_map);
  CostGraphDef cost_graph_def;
  TF_ASSERT_OK(
      cost_model_manager.AddToCostGraphDef(graph1.get(), &cost_graph_def));
  TF_ASSERT_OK(
      cost_model_manager.AddToCostGraphDef(graph2.get(), &cost_graph_def));
  ASSERT_EQ(cost_graph_def.node_size(), 12);
  absl::flat_hash_map<int32, const CostGraphDef::Node> ids;
  for (auto node : cost_graph_def.node()) {
    int32_t index = node.id();
    auto result = ids.insert({index, node});
    EXPECT_TRUE(result.second);
  }
}

TEST(CostModelTest, GlobalId) {
  auto graph = CreateBasicTestGraph();
  CostModel cm_local(/*is_global=*/false);
  CostModel cm_global(/*is_global=*/true);

  constexpr int kOffset = 7;
  for (const auto& node : graph->nodes()) {
    // Local cost models use the local id and offset.
    EXPECT_EQ(cm_local.GlobalId(node, kOffset), node->id() + kOffset);
    // Global cost modesl use the cost id.
    EXPECT_EQ(cm_global.GlobalId(node, kOffset), node->cost_id());
  }
}

TEST(CostModelTest, RecordTime) {
  auto graph = CreateBasicTestGraph();
  CostModel cm(/*is_global=*/false);
  InitModelFromGraph(*graph, cm);

  constexpr int kIters = 100;
  constexpr int kMicrosPerIter = 1000;
  for (int i = 0; i < kIters; ++i) {
    for (const auto& node : graph->op_nodes()) {
      cm.RecordTime(node, node->id() * Microseconds(kMicrosPerIter));
    }
  }

  for (const auto& node : graph->op_nodes()) {
    EXPECT_EQ(cm.TotalTime(node),
              Microseconds(node->id() * kIters * kMicrosPerIter));
  }

  // Total time for unrecorded node is 0.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  EXPECT_EQ(cm.TotalTime(E), Microseconds(0));
}

TEST(CostModelTest, RecordCount) {
  auto graph = CreateBasicTestGraph();
  CostModel cm(/*is_global=*/false);
  InitModelFromGraph(*graph, cm);

  constexpr int kIters = 100;
  constexpr int kCountPerIter = 4;
  for (int i = 0; i < kIters; ++i) {
    for (const auto& node : graph->op_nodes()) {
      cm.RecordCount(node, node->id() * kCountPerIter);
    }
  }

  for (const auto& node : graph->op_nodes()) {
    EXPECT_EQ(cm.TotalCount(node), node->id() * kIters * kCountPerIter);
  }

  // Total count for unrecorded node is 0.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  EXPECT_EQ(cm.TotalCount(E), 0);
}

TEST(CostModelTest, RecordSize) {
  auto graph = CreateBasicTestGraph();
  CostModel cm(/*is_global=*/false);
  InitModelFromGraph(*graph, cm);

  constexpr int kIters = 100;
  constexpr int kBytesPerIter = 4;
  for (int i = 0; i < kIters; ++i) {
    for (const auto& node : graph->op_nodes()) {
      for (int slot = 0; slot < node->num_outputs(); ++slot) {
        cm.RecordSize(node, slot, Bytes((node->id() + slot) * kBytesPerIter));
      }
    }
  }

  for (const auto& node : graph->op_nodes()) {
    for (int slot = 0; slot < node->num_outputs(); ++slot) {
      EXPECT_EQ(cm.TotalBytes(node, slot),
                Bytes((node->id() + slot) * kIters * kBytesPerIter));
    }
  }

  // Total size for unrecorded node is 0.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  EXPECT_EQ(cm.TotalBytes(E, 0), Bytes(0));
}

TEST(CostModelTest, SizeEstimate) {
  auto graph = CreateBasicTestGraph();
  CostModel cm(/*is_global=*/false);
  InitModelFromGraph(*graph, cm);
  Node* C = FindNode(*graph, "C");

  // Size estimate should be total bytes / total count.
  constexpr int kBytesPerCount = 31;
  constexpr int kCount = 17;
  cm.RecordCount(C, kCount);
  cm.RecordSize(C, 0, Bytes(kCount * kBytesPerCount));
  EXPECT_EQ(cm.SizeEstimate(C, 0), Bytes(kBytesPerCount));
}

TEST(CostModelTest, TimeEstimate) {
  auto graph = CreateBasicTestGraph();
  CostModel cm(/*is_global=*/false);
  InitModelFromGraph(*graph, cm);
  Node* C = FindNode(*graph, "C");

  // Time estimate should be total time / total count.
  constexpr int kMicrosPerCount = 31;
  constexpr int kCount = 17;
  cm.RecordCount(C, kCount);
  cm.RecordTime(C, Microseconds(kCount * kMicrosPerCount));
  EXPECT_EQ(cm.TimeEstimate(C), Microseconds(kMicrosPerCount));
}

TensorShapeProto CreateTensorShapeProto(absl::Span<const int64_t> dims) {
  TensorShapeProto shape;
  for (int i = 0; i < dims.size(); ++i) {
    shape.add_dim()->set_size(dims[i]);
  }
  return shape;
}

int64_t Count(const TensorShapeProto& shape) {
  int64_t count = 1;
  for (int i = 0; i < shape.dim_size(); ++i) {
    count *= shape.dim(i).size();
  }
  return count;
}

TEST(CostModelTest, RecordMaxMemorySize) {
  auto graph = CreateBasicTestGraph();
  CostModel cm(/*is_global=*/false);
  Node* C = FindNode(*graph, "C");
  InitModelFromGraph(*graph, cm);

  EXPECT_EQ(cm.MaxMemorySize(C, 0), Bytes(-1));

  {
    const TensorShapeProto shape = CreateTensorShapeProto({2, 5, 10});
    const DataType dtype = DataType::DT_FLOAT;
    const Bytes bytes = Bytes(Count(shape) * sizeof(float));
    cm.RecordMaxMemorySize(C, 0, bytes, shape, dtype);
    EXPECT_EQ(cm.MaxMemorySize(C, 0), bytes);
    EXPECT_EQ(cm.MaxMemoryType(C, 0), dtype);
    EXPECT_THAT(cm.MaxMemoryShape(C, 0), ShapeProtoEquals(shape));
  }

  // Records higher memory value.
  {
    const TensorShapeProto shape = CreateTensorShapeProto({3, 6, 11});
    const DataType dtype = DataType::DT_DOUBLE;
    const Bytes bytes = Bytes(Count(shape) * sizeof(double));
    cm.RecordMaxMemorySize(C, 0, bytes, shape, dtype);
    EXPECT_EQ(cm.MaxMemorySize(C, 0), bytes);
    EXPECT_EQ(cm.MaxMemoryType(C, 0), dtype);
    EXPECT_THAT(cm.MaxMemoryShape(C, 0), ShapeProtoEquals(shape));
  }

  // Lower memory value ignored.
  {
    const TensorShapeProto shape = CreateTensorShapeProto({1, 1, 1});
    const DataType dtype = DataType::DT_BFLOAT16;
    const Bytes bytes = Bytes(Count(shape) * sizeof(double));
    cm.RecordMaxMemorySize(C, 0, bytes, shape, dtype);
    EXPECT_GT(cm.MaxMemorySize(C, 0), bytes);
    EXPECT_NE(cm.MaxMemoryType(C, 0), dtype);
    EXPECT_THAT(cm.MaxMemoryShape(C, 0), Not(ShapeProtoEquals(shape)));
  }

  // Bytes computed from shape/dtype.
  {
    const TensorShapeProto shape = CreateTensorShapeProto({100, 100, 100});
    const DataType dtype = DataType::DT_BFLOAT16;
    cm.RecordMaxMemorySize(C, 0, Bytes(-1), shape, dtype);
    EXPECT_EQ(cm.MaxMemorySize(C, 0), Bytes(Count(shape) * sizeof(bfloat16)));
    EXPECT_EQ(cm.MaxMemoryType(C, 0), dtype);
    EXPECT_THAT(cm.MaxMemoryShape(C, 0), ShapeProtoEquals(shape));
  }

  // Max memory size for unrecorded node is 0.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  EXPECT_EQ(cm.MaxMemorySize(E, 0), Bytes(0));
  EXPECT_THAT(cm.MaxMemoryType(E, 0), DataType::DT_INVALID);
  TensorShapeProto unknown;
  unknown.set_unknown_rank(true);
  EXPECT_THAT(cm.MaxMemoryShape(E, 0), ShapeProtoEquals(unknown));
}

TEST(CostModelTest, RecordMaxExecutionTime) {
  auto graph = CreateBasicTestGraph();
  CostModel cm(/*is_global=*/false);
  InitModelFromGraph(*graph, cm);
  Node* C = FindNode(*graph, "C");

  EXPECT_EQ(cm.MaxExecutionTime(C), Microseconds(0));

  cm.RecordMaxExecutionTime(C, Microseconds(13));
  EXPECT_EQ(cm.MaxExecutionTime(C), Microseconds(13));
  cm.RecordMaxExecutionTime(C, Microseconds(27));
  EXPECT_EQ(cm.MaxExecutionTime(C), Microseconds(27));
  cm.RecordMaxExecutionTime(C, Microseconds(9));
  EXPECT_EQ(cm.MaxExecutionTime(C), Microseconds(27));

  // Max execution time for unrecorded node is 0.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  EXPECT_EQ(cm.MaxExecutionTime(E), Microseconds(0));
}

TEST(CostModelTest, RecordMemoryStats) {
  auto graph = CreateBasicTestGraph();
  CostModel cm(/*is_global=*/false);
  InitModelFromGraph(*graph, cm);
  Node* C = FindNode(*graph, "C");

  MemoryStats stats;
  stats.set_temp_memory_size(256);
  stats.set_persistent_memory_size(16);
  stats.add_persistent_tensor_alloc_ids(1);
  stats.add_persistent_tensor_alloc_ids(3);
  stats.add_persistent_tensor_alloc_ids(5);
  stats.add_persistent_tensor_alloc_ids(5);  // Intentional duplicate.

  cm.RecordMemoryStats(C, stats);
  EXPECT_EQ(cm.TempMemorySize(C), stats.temp_memory_size());
  EXPECT_EQ(cm.PersistentMemorySize(C), stats.persistent_memory_size());
  EXPECT_TRUE(cm.IsPersistentTensor(C, 1));
  EXPECT_TRUE(cm.IsPersistentTensor(C, 3));
  EXPECT_TRUE(cm.IsPersistentTensor(C, 5));
  EXPECT_FALSE(cm.IsPersistentTensor(C, 31));

  // Info for unrecorded node is 0.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  EXPECT_EQ(cm.TempMemorySize(E), Bytes(0));
  EXPECT_EQ(cm.PersistentMemorySize(E), Bytes(0));
}

TEST(CostModelTest, RecordAllocationId) {
  auto graph = CreateBasicTestGraph();
  CostModel cm(/*is_global=*/false);
  InitModelFromGraph(*graph, cm);
  Node* C = FindNode(*graph, "C");

  cm.RecordAllocationId(C, /*output_slot=*/0, /*alloc_id=*/13);
  EXPECT_EQ(cm.AllocationId(C, /*output_slot=*/0), 13);

  // Invalid slot returns -1.
  EXPECT_EQ(cm.AllocationId(C, /*output_slot=*/7), -1);
  // Unrecorded node returns -1.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  EXPECT_EQ(cm.AllocationId(E, /*output_slot=*/0), -1);
}

TEST(CostModelTest, CopyTimeEstimate) {
  // Current estimate is a linear model bytes / rate + latency.
  int64_t bytes = 32568;
  double latency_ms = 10.2;
  double gbps = 2.2;
  double bytes_per_usec = gbps * 1000 / 8;
  double cost_usecs = (bytes / bytes_per_usec + latency_ms * 1000);

  EXPECT_EQ(CostModel::CopyTimeEstimate(Bytes(bytes), latency_ms, gbps),
            Microseconds(static_cast<uint64_t>(cost_usecs)));
}

TEST(CostModelTest, ComputationTimeEstimate) {
  // Current estimate is 1000 math ops per microsecond.
  constexpr int64_t kNumMathOps = 32150;
  EXPECT_EQ(CostModel::ComputationTimeEstimate(kNumMathOps),
            Microseconds(kNumMathOps / 1000));
}

TEST(CostModel, UpdateTimes) {
  CostModel cm(/*is_global=*/false);
  EXPECT_EQ(cm.GetUpdateTimes(), 0);

  constexpr int kNumUpdates = 111;
  for (int i = 0; i < kNumUpdates; ++i) {
    cm.IncrementUpdateTimes();
  }
  EXPECT_EQ(cm.GetUpdateTimes(), kNumUpdates);
}

TEST(CostModel, SuppressInfrequent) {
  // Infrequent count is used in the size and time estimates.
  CostModel cm(/*is_global=*/false);
  auto graph = std::make_unique<Graph>(OpRegistry::Global());
  Node* A = AddNode(*graph, "A", "Mul", 2);
  Node* B = AddNode(*graph, "B", "Mul", 2);
  Node* C = AddNode(*graph, "B", "Mul", 2);
  InitModelFromGraph(*graph, cm);

  // A and B are frequent, C is not.
  cm.RecordCount(A, 1000);
  cm.RecordSize(A, 0, Bytes(8 * 1000));
  cm.RecordTime(A, Microseconds(8 * 1000));
  cm.RecordCount(B, 2000);
  cm.RecordSize(B, 0, Bytes(2000 * 10));
  cm.RecordTime(B, Microseconds(2000 * 10));
  cm.RecordCount(C, 17);
  cm.RecordSize(C, 0, Bytes(32 * 17));
  cm.RecordTime(C, Microseconds(32 * 17));

  // Estimate size and time without suppression.
  EXPECT_EQ(cm.SizeEstimate(A, 0), Bytes(8));
  EXPECT_EQ(cm.TimeEstimate(A), Microseconds(8));
  EXPECT_EQ(cm.SizeEstimate(B, 0), Bytes(10));
  EXPECT_EQ(cm.TimeEstimate(B), Microseconds(10));
  EXPECT_EQ(cm.SizeEstimate(C, 0), Bytes(32));
  EXPECT_EQ(cm.TimeEstimate(C), Microseconds(32));

  cm.SuppressInfrequent();
  // Sizes and times suppressed for C but not A, B.
  EXPECT_EQ(cm.SizeEstimate(A, 0), Bytes(8));
  EXPECT_EQ(cm.TimeEstimate(A), Microseconds(8));
  EXPECT_EQ(cm.SizeEstimate(B, 0), Bytes(10));
  EXPECT_EQ(cm.TimeEstimate(B), Microseconds(10));
  EXPECT_EQ(cm.SizeEstimate(C, 0), Bytes(0));
  EXPECT_EQ(cm.TimeEstimate(C), Microseconds(1));  // kMinTimeEstimate.
}

TEST(CostModelTest, MergeFromLocal) {
  CostModel cm_global(/*is_global=*/true);
  CostModel cm_local(/*is_global=*/false);

  auto graph = CreateBasicTestGraph();
  InitModelFromGraph(*graph, cm_global);

  // Populate global model.
  Node* C = FindNode(*graph, "C");
  Node* D = FindNode(*graph, "D");
  cm_global.RecordCount(C, 23);
  cm_global.RecordSize(C, 0, Bytes(23));
  cm_global.RecordTime(C, Microseconds(123));
  cm_global.RecordCount(D, 17);
  cm_global.RecordSize(D, 0, Bytes(17));
  cm_global.RecordTime(D, Microseconds(117));

  // Add new nodes and add cost to a local model.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  graph->AddEdge(C, 0, E, 0);
  graph->AddEdge(D, 0, E, 1);
  Node* F = AddNode(*graph, "F", "Mul", 2);
  graph->AddEdge(E, 0, F, 0);
  graph->AddEdge(D, 0, F, 1);
  InitModelFromGraph(*graph, cm_local);

  cm_local.RecordCount(E, 37);
  cm_local.RecordSize(E, 0, Bytes(37));
  cm_local.RecordTime(E, Microseconds(137));
  cm_local.RecordCount(F, 41);
  cm_local.RecordSize(F, 0, Bytes(41));
  cm_local.RecordTime(F, Microseconds(141));
  // Add existing node to check stats are added.
  cm_local.RecordCount(C, 1);
  cm_local.RecordSize(C, 0, Bytes(1));
  cm_local.RecordTime(C, Microseconds(100));

  // Merge and check that stats from local are now in global.
  cm_global.MergeFromLocal(*graph, cm_local);
  EXPECT_EQ(cm_global.TotalCount(E), cm_local.TotalCount(E));
  EXPECT_EQ(cm_global.TotalBytes(E, 0), cm_local.TotalBytes(E, 0));
  EXPECT_EQ(cm_global.TotalTime(E), cm_local.TotalTime(E));
  EXPECT_EQ(cm_global.TotalCount(F), cm_local.TotalCount(F));
  EXPECT_EQ(cm_global.TotalBytes(F, 0), cm_local.TotalBytes(F, 0));
  EXPECT_EQ(cm_global.TotalTime(F), cm_local.TotalTime(F));
  // Stats for C are added.
  EXPECT_EQ(cm_global.TotalCount(C), Microseconds(24));
  EXPECT_EQ(cm_global.TotalBytes(C, 0), Bytes(24));
  EXPECT_EQ(cm_global.TotalTime(C), Microseconds(223));
}

TEST(CostModelTest, MergeFromGlobal) {
  CostModel cm1(/*is_global=*/true);
  CostModel cm2(/*is_global=*/true);

  auto graph = CreateBasicTestGraph();
  InitModelFromGraph(*graph, cm1);

  // Populate global model.
  Node* C = FindNode(*graph, "C");
  Node* D = FindNode(*graph, "D");
  cm1.RecordCount(C, 23);
  cm1.RecordSize(C, 0, Bytes(23));
  cm1.RecordTime(C, Microseconds(123));
  cm1.RecordCount(D, 17);
  cm1.RecordSize(D, 0, Bytes(17));
  cm1.RecordTime(D, Microseconds(117));

  // Add new nodes and add cost to a local model.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  graph->AddEdge(C, 0, E, 0);
  graph->AddEdge(D, 0, E, 1);
  Node* F = AddNode(*graph, "F", "Mul", 2);
  graph->AddEdge(E, 0, F, 0);
  graph->AddEdge(D, 0, F, 1);
  InitModelFromGraph(*graph, cm2);

  cm2.RecordCount(E, 37);
  cm2.RecordSize(E, 0, Bytes(37));
  cm2.RecordTime(E, Microseconds(137));
  cm2.RecordCount(F, 41);
  cm2.RecordSize(F, 0, Bytes(41));
  cm2.RecordTime(F, Microseconds(141));
  // Add existing node to check stats are added.
  cm2.RecordCount(C, 1);
  cm2.RecordSize(C, 0, Bytes(1));
  cm2.RecordTime(C, Microseconds(100));

  // Merge and check that stats are merged.
  cm1.MergeFromGlobal(cm2);
  EXPECT_EQ(cm1.TotalCount(E), cm2.TotalCount(E));
  EXPECT_EQ(cm1.TotalBytes(E, 0), cm2.TotalBytes(E, 0));
  EXPECT_EQ(cm1.TotalTime(E), cm2.TotalTime(E));
  EXPECT_EQ(cm1.TotalCount(F), cm2.TotalCount(F));
  EXPECT_EQ(cm1.TotalBytes(F, 0), cm2.TotalBytes(F, 0));
  EXPECT_EQ(cm1.TotalTime(F), cm2.TotalTime(F));
  // Stats for C are added.
  EXPECT_EQ(cm1.TotalCount(C), Microseconds(24));
  EXPECT_EQ(cm1.TotalBytes(C, 0), Bytes(24));
  EXPECT_EQ(cm1.TotalTime(C), Microseconds(223));
}

NodeExecStats CreateNodeExecStats(const Node* node, int64_t time,
                                  int64_t bytes) {
  NodeExecStats stats;
  stats.set_node_name(node->name());
  stats.set_op_start_rel_micros(10);
  stats.set_op_end_rel_micros(10 + time);
  for (int i = 0; i < node->num_outputs(); ++i) {
    NodeOutput* no = stats.add_output();
    no->set_slot(i);
    no->mutable_tensor_description()
        ->mutable_allocation_description()
        ->set_requested_bytes(bytes);
  }
  return stats;
}

TEST(CostModelTest, MergeFromStats) {
  CostModel cm(/*is_global=*/true);
  auto graph = CreateBasicTestGraph();
  InitModelFromGraph(*graph, cm);

  // Populate global model.
  Node* C = FindNode(*graph, "C");
  Node* D = FindNode(*graph, "D");
  cm.RecordCount(C, 23);
  cm.RecordTime(C, Microseconds(123));
  cm.RecordCount(D, 17);
  cm.RecordTime(D, Microseconds(117));

  // Add new nodes and create stats.
  Node* E = AddNode(*graph, "E", "Mul", 2);
  graph->AddEdge(C, 0, E, 0);
  graph->AddEdge(D, 0, E, 1);
  Node* F = AddNode(*graph, "F", "Mul", 2);
  graph->AddEdge(E, 0, F, 0);
  graph->AddEdge(D, 0, F, 1);

  StepStats stats;
  DeviceStepStats* dstats = stats.add_dev_stats();
  *(dstats->add_node_stats()) = CreateNodeExecStats(C, 10, 10);
  *(dstats->add_node_stats()) = CreateNodeExecStats(D, 10, 10);
  *(dstats->add_node_stats()) = CreateNodeExecStats(E, 20, 20);
  *(dstats->add_node_stats()) = CreateNodeExecStats(E, 20, 20);
  *(dstats->add_node_stats()) = CreateNodeExecStats(F, 30, 30);
  *(dstats->add_node_stats()) = CreateNodeExecStats(F, 30, 30);

  NodeNameToCostIdMap id_map;
  for (const auto& node : graph->nodes()) {
    id_map.emplace(node->name(), node->cost_id());
  }
  cm.MergeFromStats(id_map, stats);

  // Stats for C/D are added to existing.
  EXPECT_EQ(cm.TotalCount(C), 24);
  EXPECT_EQ(cm.TotalTime(C), Microseconds(133));
  EXPECT_EQ(cm.TotalBytes(C, 0), Bytes(10));
  EXPECT_EQ(cm.TotalCount(D), 18);
  EXPECT_EQ(cm.TotalTime(D), Microseconds(127));
  EXPECT_EQ(cm.TotalBytes(D, 0), Bytes(10));

  // Stats for E/F are accumulated.
  EXPECT_EQ(cm.TotalCount(E), 2);
  EXPECT_EQ(cm.TotalTime(E), Microseconds(40));
  EXPECT_EQ(cm.TotalBytes(E, 0), Bytes(40));
  EXPECT_EQ(cm.TotalCount(F), 2);
  EXPECT_EQ(cm.TotalTime(F), Microseconds(60));
  EXPECT_EQ(cm.TotalBytes(F, 0), Bytes(60));
}

}  // namespace
}  // namespace tensorflow
