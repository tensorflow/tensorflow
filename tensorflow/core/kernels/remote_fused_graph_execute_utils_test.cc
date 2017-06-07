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

#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"
#include "tensorflow/cc/framework/scope.h"
#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_op_test_utils.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace {

using ClusterInfo = RemoteFusedGraphExecuteUtils::ClusterInfo;

constexpr const char* const NAME_A = "A";
constexpr const char* const NAME_B = "B";
constexpr const char* const NAME_A_PLUS_B = "A_PLUS_B";
constexpr float NODE_A_VAL = 2.0f;
constexpr float NODE_B_VAL = 3.0f;
constexpr float VALUE_TOLERANCE_FLOAT = 1e-8f;

static NodeDef* GetNodeDef(const string& name, GraphDef* def) {
  CHECK_NE(def, nullptr);
  for (NodeDef& node_def : *def->mutable_node()) {
    if (node_def.name() == name) {
      return &node_def;
    }
  }
  return nullptr;
}

class FuseRemoteGraphMultipleAddOpsTest : public ::testing::Test {
 protected:
  void SetUp() final {
    TF_ASSERT_OK(
        RemoteFusedGraphExecuteOpTestUtils::BuildMultipleAddGraph(&graph_def_));
    RemoteFusedGraphExecuteUtils::ExecutorBuildRegistrar
        k_hexagon_remote_fused_graph_executor_build(
            "remote_graph_executor_name",
            [](std::unique_ptr<IRemoteFusedGraphExecutor>* executor) -> Status {
              return Status::OK();
            });
  }

  void TearDown() final {}

  Status FuseByInOut() {
    // Feed output shapes and types
    RemoteFusedGraphExecuteUtils::TensorShapeMap tensor_shape_map;
    GraphDef graph_def_with_shapetype = graph_def_;
    TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::BuildAndAddTensorShapes(
        input_tensors_, /*dry_run_inference*/ true, &graph_def_with_shapetype));

    return RemoteFusedGraphExecuteUtils::FuseRemoteGraphByBorder(
        graph_def_with_shapetype, inputs_, outputs_,
        "remote_fused_graph_node_names", subgraph_input_names_,
        subgraph_output_names_, "remote_graph_executor_name",
        /*require_shape_type=*/true, &result_graph_def_);
  }

  Status FuseByNodes() {
    return RemoteFusedGraphExecuteUtils::FuseRemoteGraphByNodeNames(
        graph_def_, inputs_, outputs_, "remote_fused_graph_node_names",
        subgraph_node_names_, "remote_graph_executor_name",
        /*require_shape_type=*/false, &result_graph_def_);
  }

  Status BuildAndAddTensorShape() {
    return RemoteFusedGraphExecuteUtils::BuildAndAddTensorShapes(
        input_tensors_, /*dry_run_inference=*/true, &graph_def_);
  }

  Status PlaceRemoteGraphArguments() {
    return RemoteFusedGraphExecuteUtils::PlaceRemoteGraphArguments(
        inputs_, outputs_, subgraph_node_names_, subgraph_input_names_,
        subgraph_output_names_, "remote_fused_graph_node_names",
        "remote_graph_executor_name", &graph_def_);
  }

  Status FuseByPlacedArguments() {
    const Status status =
        RemoteFusedGraphExecuteUtils::FuseRemoteGraphByPlacedArguments(
            graph_def_, input_tensors_, &graph_def_);
    result_graph_def_ = graph_def_;
    return status;
  }

  bool IsFuseReady() {
    return RemoteFusedGraphExecuteUtils::IsFuseReady(graph_def_,
                                                     input_tensors_);
  }

 public:
  const std::vector<std::pair<string, Tensor>> input_tensors_{
      {"A", {DT_FLOAT, {1, 1, 1, 1}}}};
  const std::vector<string> inputs_{"A"};
  const std::vector<string> outputs_{"K"};
  GraphDef graph_def_;
  GraphDef result_graph_def_;
  std::vector<string> subgraph_input_names_;
  std::vector<string> subgraph_output_names_;
  std::unordered_set<string> subgraph_node_names_;
};

void SetSubgraphArguments(const std::vector<string>& input_names,
                          const std::vector<string>& output_names,
                          FuseRemoteGraphMultipleAddOpsTest* fixture) {
  for (const string& input_name : input_names) {
    fixture->subgraph_input_names_.emplace_back(input_name);
  }

  fixture->subgraph_output_names_ = output_names;
}

template <typename T>
static string IterToString(const T& set) {
  string out;
  for (const string& val : set) {
    if (!out.empty()) {
      out += ", ";
    }
    out += val;
  }
  return out;
}

static string SummarizeGraphDef(const GraphDef& graph_def) {
  string out;
  for (const NodeDef& node : graph_def.node()) {
    out += strings::StrCat("node: ", node.name(), "\n    input: ");
    for (const string& input : node.input()) {
      out += strings::StrCat(input, ", ");
    }
    out += "\n";
  }
  return out;
}

static string DumpInOutNames(const std::vector<ClusterInfo>& ci_vec) {
  for (int i = 0; i < ci_vec.size(); ++i) {
    LOG(INFO) << "Cluster(" << i << ")";
    LOG(INFO) << "input: " << IterToString(std::get<1>(ci_vec.at(i)));
    LOG(INFO) << "output: " << IterToString(std::get<2>(ci_vec.at(i)));
  }
  return "";
}

static void ClearCluster(ClusterInfo* cluster) {
  std::get<0>(*cluster).clear();
  std::get<1>(*cluster).clear();
  std::get<2>(*cluster).clear();
}

TEST(RemoteFusedGraphExecuteUtils, DryRunAddGraphA) {
  GraphDef def;
  TF_ASSERT_OK(RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B, &def));
  std::pair<string, Tensor> input_node_info;
  input_node_info.first = NAME_A;
  input_node_info.second = Tensor(DT_FLOAT, {});
  input_node_info.second.scalar<float>()() = 1.0f;
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info};
  std::vector<string> outputs = {NAME_B, NAME_A_PLUS_B};
  std::vector<tensorflow::Tensor> output_tensors;
  Status status = RemoteFusedGraphExecuteUtils::DryRunInference(
      def, inputs, outputs, false /* initialize_by_zero */, &output_tensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(outputs.size(), output_tensors.size());
  EXPECT_NEAR(NODE_B_VAL, output_tensors.at(0).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
  EXPECT_NEAR(1.0f + NODE_B_VAL, output_tensors.at(1).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
}

TEST(RemoteFusedGraphExecuteUtils, DryRunAddGraphAUninitialized) {
  GraphDef def;
  TF_ASSERT_OK(RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B, &def));
  std::pair<string, Tensor> input_node_info;
  input_node_info.first = NAME_A;
  input_node_info.second = Tensor(DT_FLOAT, {});
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info};
  std::vector<string> outputs = {NAME_B, NAME_A_PLUS_B};
  std::vector<tensorflow::Tensor> output_tensors;
  Status status = RemoteFusedGraphExecuteUtils::DryRunInference(
      def, inputs, outputs, true /* initialize_by_zero */, &output_tensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(outputs.size(), output_tensors.size());
  EXPECT_NEAR(NODE_B_VAL, output_tensors.at(0).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
  EXPECT_NEAR(NODE_B_VAL, output_tensors.at(1).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
}

TEST(RemoteFusedGraphExecuteUtils, DryRunAddGraphAB) {
  GraphDef def;
  TF_ASSERT_OK(RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B, &def));
  std::pair<string, Tensor> input_node_info_a;
  input_node_info_a.first = NAME_A;
  input_node_info_a.second = Tensor(DT_FLOAT, {});
  input_node_info_a.second.scalar<float>()() = NODE_A_VAL;
  std::pair<string, Tensor> input_node_info_b;
  input_node_info_b.first = NAME_B;
  input_node_info_b.second = Tensor(DT_FLOAT, {});
  input_node_info_b.second.scalar<float>()() = NODE_B_VAL;
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info_a,
                                                      input_node_info_b};
  std::vector<string> outputs = {NAME_A_PLUS_B};
  std::vector<tensorflow::Tensor> output_tensors;
  Status status = RemoteFusedGraphExecuteUtils::DryRunInference(
      def, inputs, outputs, false /* initialize_by_zero */, &output_tensors);
  ASSERT_TRUE(status.ok()) << status;
  EXPECT_EQ(outputs.size(), output_tensors.size());
  EXPECT_NEAR(NODE_A_VAL + NODE_B_VAL, output_tensors.at(0).scalar<float>()(),
              VALUE_TOLERANCE_FLOAT);
}

TEST(RemoteFusedGraphExecuteUtils, DryRunAddGraphForAllNodes) {
  // Set Node "A" as an input with value (= 1.0f)
  std::pair<string, Tensor> input_node_info_a;
  input_node_info_a.first = NAME_A;
  input_node_info_a.second = Tensor(DT_FLOAT, {});
  input_node_info_a.second.scalar<float>()() = 1.0f;

  // Setup dryrun arguments
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info_a};
  RemoteFusedGraphExecuteUtils::TensorShapeMap tensor_shape_map;

  GraphDef def;
  TF_ASSERT_OK(RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B, &def));

  // dryrun
  const Status status = RemoteFusedGraphExecuteUtils::DryRunInferenceForAllNode(
      def, inputs, false /* initialize_by_zero */, &tensor_shape_map);

  ASSERT_TRUE(status.ok()) << status;

  // Assert output node count
  ASSERT_EQ(3, tensor_shape_map.size());
  ASSERT_EQ(1, tensor_shape_map.count(NAME_A));
  ASSERT_EQ(1, tensor_shape_map.count(NAME_B));
  ASSERT_EQ(1, tensor_shape_map.count(NAME_A_PLUS_B));

  const RemoteFusedGraphExecuteUtils::TensorShapeType* tst =
      RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                       NAME_B);
  EXPECT_NE(tst, nullptr);
  EXPECT_EQ(DT_FLOAT, tst->first);
  EXPECT_EQ(0, tst->second.dims());

  tst = RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                         NAME_A_PLUS_B);
  EXPECT_NE(tst, nullptr);
  EXPECT_EQ(DT_FLOAT, tst->first);
  EXPECT_EQ(0, tst->second.dims());
}

TEST(RemoteFusedGraphExecuteUtils, PropagateAndBuildTensorShapeMap) {
  std::pair<string, Tensor> input_node_info_a;
  input_node_info_a.first = NAME_A;
  input_node_info_a.second = Tensor(DT_FLOAT, {});
  input_node_info_a.second.scalar<float>()() = NODE_A_VAL;
  std::pair<string, Tensor> input_node_info_b;
  input_node_info_b.first = NAME_B;
  input_node_info_b.second = Tensor(DT_FLOAT, {});
  input_node_info_b.second.scalar<float>()() = NODE_B_VAL;
  const std::vector<std::pair<string, Tensor>> inputs{input_node_info_a,
                                                      input_node_info_b};

  RemoteFusedGraphExecuteUtils::TensorShapeMap tensor_shape_map;
  GraphDef def;
  TF_ASSERT_OK(RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B, &def));
  ImportGraphDefOptions opts;
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions().producer(), graph.op_registry());
  Status status = ImportGraphDef(opts, def, &graph, &shape_refiner);
  ASSERT_TRUE(RemoteFusedGraphExecuteUtils::PropagateShapeInference(
                  def, inputs, &graph, &shape_refiner)
                  .ok());
  ASSERT_TRUE(RemoteFusedGraphExecuteUtils::BuildTensorShapeMapFromGraph(
                  graph, shape_refiner, &tensor_shape_map)
                  .ok());

  ASSERT_EQ(3, tensor_shape_map.size());
  ASSERT_EQ(1, tensor_shape_map.count(NAME_A));
  ASSERT_EQ(1, tensor_shape_map.count(NAME_B));
  ASSERT_EQ(1, tensor_shape_map.count(NAME_A_PLUS_B));

  const RemoteFusedGraphExecuteUtils::TensorShapeType* tst =
      RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                       NAME_B);
  EXPECT_NE(tst, nullptr);
  EXPECT_EQ(DT_FLOAT, tst->first);
  EXPECT_EQ(0, tst->second.dims());

  tst = RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map,
                                                         NAME_A_PLUS_B);
  EXPECT_NE(tst, nullptr);
  EXPECT_EQ(DT_FLOAT, tst->first);
  EXPECT_EQ(0, tst->second.dims());

  {
    NodeDef* node_def = GetNodeDef(NAME_B, &def);
    TF_ASSERT_OK(
        RemoteFusedGraphExecuteUtils::AddOutputTensorShapeTypeByTensorShapeMap(
            tensor_shape_map, node_def));
    std::vector<DataType> data_types;
    TF_ASSERT_OK(GetNodeAttr(
        *node_def, RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_DATA_TYPES,
        &data_types));
    ASSERT_EQ(1, data_types.size());
    EXPECT_EQ(DT_FLOAT, data_types.at(0));

    std::vector<TensorShape> shapes;
    TF_ASSERT_OK(GetNodeAttr(
        *node_def, RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_SHAPES, &shapes));
    ASSERT_EQ(1, shapes.size());
    EXPECT_EQ(0, shapes.at(0).dims());
  }

  {
    NodeDef* node_def = GetNodeDef(NAME_A_PLUS_B, &def);
    TF_ASSERT_OK(
        RemoteFusedGraphExecuteUtils::AddOutputTensorShapeTypeByTensorShapeMap(
            tensor_shape_map, node_def));
    std::vector<DataType> data_types;
    TF_ASSERT_OK(GetNodeAttr(
        *node_def, RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_DATA_TYPES,
        &data_types));
    ASSERT_EQ(1, data_types.size());
    EXPECT_EQ(DT_FLOAT, data_types.at(0));

    std::vector<TensorShape> shapes;
    TF_ASSERT_OK(GetNodeAttr(
        *node_def, RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_SHAPES, &shapes));
    ASSERT_EQ(1, shapes.size());
    EXPECT_EQ(0, shapes.at(0).dims());
  }
}

TEST(RemoteFusedGraphExecuteUtils,
     BuildRemoteFusedGraphExecuteInfoWithShapeInference) {
  // Build inputs
  std::pair<string, Tensor> input_node_info_a;
  input_node_info_a.first = NAME_A;
  input_node_info_a.second = Tensor(DT_FLOAT, {});
  input_node_info_a.second.scalar<float>()() = NODE_A_VAL;
  std::pair<string, Tensor> input_node_info_b;
  input_node_info_b.first = NAME_B;
  input_node_info_b.second = Tensor(DT_FLOAT, {});
  input_node_info_b.second.scalar<float>()() = NODE_B_VAL;
  const std::vector<std::pair<string, Tensor>> input_tensors{input_node_info_a,
                                                             input_node_info_b};
  const std::vector<string> inputs{NAME_A, NAME_B};

  // Build outputs
  const std::vector<string> outputs = {NAME_A_PLUS_B};

  GraphDef def;
  TF_ASSERT_OK(RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B, &def));
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildAndAddTensorShapes(
      input_tensors, /*dry_run_inference*/ true, &def));

  RemoteFusedGraphExecuteInfo execute_info0;
  DataTypeVector input_types0;
  DataTypeVector output_types0;

  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildRemoteFusedGraphExecuteInfo(
      "executor", def, inputs, outputs, /*require_shape_type=*/true,
      &execute_info0, &input_types0, &output_types0));

  EXPECT_EQ(inputs.size(),
            execute_info0.default_graph_input_tensor_shape_size());
  EXPECT_EQ(outputs.size(),
            execute_info0.default_graph_output_tensor_shape_size());
  EXPECT_EQ(inputs.size(), input_types0.size());
  EXPECT_EQ(outputs.size(), output_types0.size());

  EXPECT_EQ(def.node_size(), execute_info0.remote_graph().node_size());
}

TEST(RemoteFusedGraphExecuteUtils, BuildRemoteFusedGraphExecuteOpNode) {
  const std::vector<string> inputs{NAME_A, NAME_B};

  // Build outputs
  const std::vector<string> outputs = {NAME_A_PLUS_B};

  GraphDef def;
  TF_ASSERT_OK(RemoteFusedGraphExecuteOpTestUtils::BuildAddGraph(
      NAME_A, NODE_A_VAL, NAME_B, NODE_B_VAL, NAME_A_PLUS_B, &def));

  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions().producer(), graph.op_registry());
  TF_ASSERT_OK(ImportGraphDef({}, def, &graph, &shape_refiner));

  Node* node;
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildRemoteFusedGraphExecuteOpNode(
      "fused_name", "executor", def, inputs, outputs,
      /*require_shape_type=*/false, &graph, &node));
}

TEST(RemoteFusedGraphExecuteUtils, ExtractSubgraphNodes) {
  GraphDef graph_def;
  TF_ASSERT_OK(
      RemoteFusedGraphExecuteOpTestUtils::BuildMultipleAddGraph(&graph_def));
  ClusterInfo cluster;
  const std::unordered_set<string>& node_names = std::get<0>(cluster);
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      {"H", "I"}, {"J"}, graph_def, &cluster));
  EXPECT_EQ(1, node_names.size()) << IterToString(node_names);

  ClearCluster(&cluster);
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      {"F", "C", "G"}, {"J"}, graph_def, &cluster));
  EXPECT_EQ(3, node_names.size()) << IterToString(node_names);

  ClearCluster(&cluster);
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      {"A", "B", "C", "D", "E"}, {"J"}, graph_def, &cluster));
  EXPECT_EQ(5, node_names.size()) << IterToString(node_names);

  ClearCluster(&cluster);
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      {"A", "B", "C", "D", "E"}, {"K"}, graph_def, &cluster));
  EXPECT_EQ(6, node_names.size()) << IterToString(node_names);

  ClearCluster(&cluster);
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      {"F"}, {"H"}, graph_def, &cluster));
  EXPECT_EQ(2, node_names.size()) << IterToString(node_names);
}

TEST(RemoteFusedGraphExecuteUtils, ClusterizeNodes) {
  GraphDef graph_def;
  TF_ASSERT_OK(
      RemoteFusedGraphExecuteOpTestUtils::BuildMultipleAddGraph(&graph_def));

  std::vector<ClusterInfo> ci_vec;
  TF_ASSERT_OK(
      RemoteFusedGraphExecuteUtils::ClusterizeNodes({"J"}, graph_def, &ci_vec));
  ASSERT_EQ(1, ci_vec.size());
  EXPECT_EQ(2, std::get<1>(ci_vec.at(0)).size()) << DumpInOutNames(ci_vec);
  EXPECT_EQ(1, std::get<2>(ci_vec.at(0)).size()) << DumpInOutNames(ci_vec);

  ci_vec.clear();
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::ClusterizeNodes(
      {"H", "I", "J"}, graph_def, &ci_vec));
  ASSERT_EQ(1, ci_vec.size());
  EXPECT_EQ(3, std::get<1>(ci_vec.at(0)).size()) << DumpInOutNames(ci_vec);
  EXPECT_EQ(1, std::get<2>(ci_vec.at(0)).size()) << DumpInOutNames(ci_vec);

  ci_vec.clear();
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::ClusterizeNodes(
      {"F", "C", "G", "H", "I", "J"}, graph_def, &ci_vec));
  ASSERT_EQ(1, ci_vec.size());
  EXPECT_EQ(4, std::get<1>(ci_vec.at(0)).size()) << DumpInOutNames(ci_vec);
  EXPECT_EQ(2, std::get<2>(ci_vec.at(0)).size()) << DumpInOutNames(ci_vec);

  ci_vec.clear();
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::ClusterizeNodes(
      {"A", "B", "C", "D", "E"}, graph_def, &ci_vec));
  ASSERT_EQ(5, ci_vec.size());

  ci_vec.clear();
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::ClusterizeNodes(
      {"A", "B", "D", "E", "F", "G"}, graph_def, &ci_vec));
  ASSERT_EQ(2, ci_vec.size());
}

TEST(RemoteFusedGraphExecuteUtils, BuildSubgraphDefByInOut) {
  GraphDef graph_def;
  TF_ASSERT_OK(
      RemoteFusedGraphExecuteOpTestUtils::BuildMultipleAddGraph(&graph_def));

  ClusterInfo cluster;
  GraphDef subgraph_def;
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      std::vector<string>{"H", "I"}, std::vector<string>{"J"}, graph_def,
      &cluster));
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterSubgraphDef(
      cluster, graph_def, &subgraph_def));
  EXPECT_EQ(3, subgraph_def.node_size());

  ClearCluster(&cluster);
  subgraph_def.Clear();
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      std::vector<string>{"F", "C", "G"}, std::vector<string>{"J"}, graph_def,
      &cluster));
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterSubgraphDef(
      cluster, graph_def, &subgraph_def));
  EXPECT_EQ(6, subgraph_def.node_size());

  ClearCluster(&cluster);
  subgraph_def.Clear();
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      std::vector<string>{"A", "B", "C", "D", "E"}, std::vector<string>{"J"},
      graph_def, &cluster));
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterSubgraphDef(
      cluster, graph_def, &subgraph_def));
  EXPECT_EQ(10, subgraph_def.node_size());

  ClearCluster(&cluster);
  subgraph_def.Clear();

  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      std::vector<string>{"A", "B", "C", "D", "E"}, std::vector<string>{"K"},
      graph_def, &cluster));
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterSubgraphDef(
      cluster, graph_def, &subgraph_def));
  EXPECT_EQ(11, subgraph_def.node_size());

  ClearCluster(&cluster);
  subgraph_def.Clear();
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      std::vector<string>{"F"}, std::vector<string>{"H"}, graph_def, &cluster));
  TF_ASSERT_OK(RemoteFusedGraphExecuteUtils::BuildClusterSubgraphDef(
      cluster, graph_def, &subgraph_def));
  EXPECT_EQ(3, subgraph_def.node_size());
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, FuseSubgraphByInOut_HI_J) {
  SetSubgraphArguments(std::vector<string>{"H", "I"}, std::vector<string>{"J"},
                       this);

  TF_ASSERT_OK(FuseByInOut());

  EXPECT_EQ(11, graph_def_.node_size());
  EXPECT_EQ(11, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, FuseSubgraphByInOut_FCG_J) {
  SetSubgraphArguments(std::vector<string>{"F", "C", "G"},
                       std::vector<string>{"J"}, this);

  TF_ASSERT_OK(FuseByInOut());

  EXPECT_EQ(11, graph_def_.node_size());
  EXPECT_EQ(9, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, FuseSubgraphByInOut_ABCDE_J) {
  SetSubgraphArguments(std::vector<string>{"A", "B", "C", "D", "E"},
                       std::vector<string>{"J"}, this);

  TF_ASSERT_OK(FuseByInOut());

  EXPECT_EQ(11, graph_def_.node_size());
  EXPECT_EQ(8, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, FuseSubgraphByInOut_ABCDE_K) {
  SetSubgraphArguments(std::vector<string>{"A", "B", "C", "D", "E"},
                       std::vector<string>{"K"}, this);

  TF_ASSERT_OK(FuseByInOut());

  EXPECT_EQ(11, graph_def_.node_size());
  EXPECT_EQ(7, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, FuseSubgraphByNodes_H) {
  subgraph_node_names_ = {"H"};

  TF_ASSERT_OK(FuseByNodes());

  EXPECT_EQ(11, graph_def_.node_size());
  EXPECT_EQ(11, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, FuseSubgraphByNodes_HIJ) {
  subgraph_node_names_ = {"H", "I", "J"};

  TF_ASSERT_OK(FuseByNodes());

  EXPECT_EQ(11, graph_def_.node_size());
  EXPECT_EQ(9, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, FuseSubgraphByNodes_CFGHIJ) {
  subgraph_node_names_ = {"C", "F", "G", "H", "I", "J"};

  TF_ASSERT_OK(FuseByNodes());

  EXPECT_EQ(11, graph_def_.node_size());
  EXPECT_EQ(6, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, FuseSubgraphByNodes_ABCDEFGHIJ) {
  subgraph_node_names_ = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J"};

  TF_ASSERT_OK(FuseByNodes());

  EXPECT_EQ(11, graph_def_.node_size());
  EXPECT_EQ(3, result_graph_def_.node_size())  // "A", "RFG", "K"
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, FuseSubgraphByNodes_ABCDEFGHIJK) {
  subgraph_node_names_ = {"A", "B", "C", "D", "E", "F",
                          "G", "H", "I", "J", "K"};

  TF_ASSERT_OK(FuseByNodes());

  EXPECT_EQ(11, graph_def_.node_size());
  EXPECT_EQ(3, result_graph_def_.node_size())  // "A", "RFG", "K"
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, PlaceAndFuse_H) {
  subgraph_node_names_ = {"H"};

  TF_ASSERT_OK(PlaceRemoteGraphArguments());
  ASSERT_TRUE(IsFuseReady());
  TF_ASSERT_OK(BuildAndAddTensorShape());

  EXPECT_EQ(11, graph_def_.node_size());

  TF_ASSERT_OK(FuseByPlacedArguments());

  EXPECT_EQ(11, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, PlaceAndFuse_CFGHIJ) {
  subgraph_node_names_ = {"C", "F", "G", "H", "I", "J"};

  TF_ASSERT_OK(PlaceRemoteGraphArguments());
  ASSERT_TRUE(IsFuseReady());
  TF_ASSERT_OK(BuildAndAddTensorShape());

  EXPECT_EQ(11, graph_def_.node_size());

  TF_ASSERT_OK(FuseByPlacedArguments());

  EXPECT_EQ(6, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, PlaceAndFuse_ABCDEFGHIJK) {
  subgraph_node_names_ = {"A", "B", "C", "D", "E", "F",
                          "G", "H", "I", "J", "K"};

  TF_ASSERT_OK(PlaceRemoteGraphArguments());
  ASSERT_TRUE(IsFuseReady());
  TF_ASSERT_OK(BuildAndAddTensorShape());

  EXPECT_EQ(11, graph_def_.node_size());

  TF_ASSERT_OK(FuseByPlacedArguments());

  EXPECT_EQ(3, result_graph_def_.node_size())  // "A", "RFG", "K"
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, PlaceAndFuse_HI_J) {
  SetSubgraphArguments(std::vector<string>{"H", "I"}, std::vector<string>{"J"},
                       this);

  TF_ASSERT_OK(PlaceRemoteGraphArguments());
  ASSERT_TRUE(IsFuseReady());
  TF_ASSERT_OK(BuildAndAddTensorShape());

  EXPECT_EQ(11, graph_def_.node_size());

  TF_ASSERT_OK(FuseByPlacedArguments());

  EXPECT_EQ(11, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, PlaceAndFuse_FCG_J) {
  SetSubgraphArguments(std::vector<string>{"F", "C", "G"},
                       std::vector<string>{"J"}, this);

  TF_ASSERT_OK(PlaceRemoteGraphArguments());
  ASSERT_TRUE(IsFuseReady());
  TF_ASSERT_OK(BuildAndAddTensorShape());

  EXPECT_EQ(11, graph_def_.node_size());

  TF_ASSERT_OK(FuseByPlacedArguments());

  EXPECT_EQ(9, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

TEST_F(FuseRemoteGraphMultipleAddOpsTest, PlaceAndFuse_ABCDE_K) {
  SetSubgraphArguments(std::vector<string>{"A", "B", "C", "D", "E"},
                       std::vector<string>{"K"}, this);

  TF_ASSERT_OK(PlaceRemoteGraphArguments());
  ASSERT_TRUE(IsFuseReady());
  TF_ASSERT_OK(BuildAndAddTensorShape());

  EXPECT_EQ(11, graph_def_.node_size());

  TF_ASSERT_OK(FuseByPlacedArguments());

  EXPECT_EQ(7, result_graph_def_.node_size())
      << "=== Before: \n"
      << SummarizeGraphDef(graph_def_) << "\n\n\n=== After: \n"
      << SummarizeGraphDef(result_graph_def_);
}

}  // namespace
}  // namespace tensorflow
