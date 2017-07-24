/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
vcyou may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_GRAPH_TRANSFERER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_GRAPH_TRANSFERER_H_

#include <array>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/graph_transfer_info.pb.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/kernels/i_remote_fused_graph_ops_definitions.h"
#include "tensorflow/core/kernels/remote_fused_graph_execute_utils.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/util/padding.h"

namespace tensorflow {

// GraphTransferer transfers graph definitions into SoC memory.
// This functionality is effective if SoC is capable to run
// the graph on that chip.
// TODO(satok): support transferring subgraphs to be able to split graphs
// to avoid unsupported ops in SoC.
class GraphTransferer {
 public:
  // TODO(satok): Remove. Use proto definition instead.
  static constexpr int MAX_SUPPORTED_RANK = 4;
  // TODO(satok): Remove. Use proto definition instead.
  static constexpr int SHAPE_ARRAY_SIZE = MAX_SUPPORTED_RANK;
  using TensorShapeMap = RemoteFusedGraphExecuteUtils::TensorShapeMap;

  GraphTransferer() = default;

  // Load graph structure into GraphTransferer
  // TODO(satok): Pass a pair of TensorShape and DataType instead of
  // Tensor as input_node_info_list.
  Status LoadGraphFromProto(
      const IRemoteFusedGraphOpsDefinitions& ops_definitions,
      const GraphDef& graph_def,
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      const std::vector<string>& output_node_names,
      const bool shape_inference_for_unknown_shape);

  // Load graph structure into GraphTransferer from protobuf file
  // TODO(satok): Pass a pair of TensorShape and DataType instead of
  // Tensor as input_node_info_list.
  Status LoadGraphFromProtoFile(
      const IRemoteFusedGraphOpsDefinitions& ops_definitions,
      const string& graph_def_path,
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      const std::vector<string>& output_node_names, const bool is_text_proto,
      const bool shape_inference_for_unknown_shape,
      const bool dry_run_for_unknown_shape);

  // Sort params so that all input nodes appear before consumer nodes.
  // CAVEAT: This may be slow if the number of nodes are too large
  void SortParams(const std::vector<string>& output_node_names);

  void EnableStrictCheckMode(bool enable);

  // Import parameters for transfer
  void SetSerializedGraphTransferInfo(const string& serialized_proto);

  // Return parameters for graph transfer
  const GraphTransferInfo& GetGraphTransferInfo() const;

  // Return mutable GraphTransferInfo for graph transfer
  GraphTransferInfo& GetMutableGraphTransferInfo();

  // Dump verification string of parameters to verify with offline tools
  void DumpVerificationStringOfNodeTransferParams() const;

  static std::array<int64, SHAPE_ARRAY_SIZE> ToTensorShapeArray(
      const TensorShape& shape);

 private:
  class TransferParamsComparator {
   public:
    TransferParamsComparator(
        const std::unordered_map<int, std::unordered_set<int>>& dep_map);
    bool operator()(const GraphTransferInfo::NodeInfo& obj0,
                    const GraphTransferInfo::NodeInfo& obj1);
    const std::unordered_map<int, std::unordered_set<int>>& dependency_map_;
  };

  void CacheNode(const Node& node);

  bool AreAllInputsCached(const Node& node) const;

  // Transform a remote fused graph to add an aggregated input node which takes
  // all inputs of the remote graph.
  Status TransformGraphToAddAggregatedInputNode(
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      Graph* graph, ShapeRefiner* shape_refiner);

  Status RegisterNode(
      const IRemoteFusedGraphOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner, const Node& node,
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      const std::vector<string>& output_node_names);

  void RegisterConstantNode(const ShapeRefiner& shape_refiner,
                            const Node& node);

  int RegisterConstantShape(const std::vector<int>& shape);

  int RegisterConstTensor(const Tensor& tensor, const string& suffix);

  int RegisterConstScalar(const DataType dt, const int val, const int dst_id,
                          const int dst_input_count);

  bool HasPaddingAndStrides(const Node& node);

  bool NeedsToAddRank(const Node& node);

  bool IsPadNode(const Node& node);

  // Return true if the node is a reshape op which just flattens input
  // TODO(satok): Remove this method once generic reshape op is implemented in
  // SOC
  bool IsNodeFlattenReshape(const Node& node,
                            const ShapeRefiner& shape_refiner);

  void RegisterNodeWithPaddingAndStrides(
      const IRemoteFusedGraphOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner, const Node& node);

  void RegisterNodeWithRank(
      const IRemoteFusedGraphOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner, const Node& node);

  void RegisterPadNode(const IRemoteFusedGraphOpsDefinitions& ops_definitions,
                       const ShapeRefiner& shape_refiner, const Node& node);

  void RegisterInputNode(const IRemoteFusedGraphOpsDefinitions& ops_definitions,
                         const ShapeRefiner& shape_refiner, const Node& node);

  void RegisterFlattenNode(
      const IRemoteFusedGraphOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner, const Node& node);

  void RegisterGenericNode(
      const IRemoteFusedGraphOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner, const Node& node);

  Status RegisterNodeIfAllInputsAreCached(
      const IRemoteFusedGraphOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner, const Node& node,
      const bool only_register_const_node,
      const std::vector<std::pair<string, Tensor>>& input_node_info_list,
      const std::vector<string>& output_node_names);

  void AppendNodeParams(const string& name, const int id, const string& type,
                        const int type_id, const int padding,
                        const int inputs_size,
                        const std::vector<int>& extra_inputs,
                        const int outputs_size);

  void AddNodeInputByInputIndex(
      const Node& node, const int idx,
      GraphTransferInfo::NodeInputInfo* node_input_info);

  void AppendNodeInputParams(const int id, const Node& node,
                             const std::vector<int>& extra_inputs);

  void AppendNodeOutputParams(const ShapeRefiner& shape_refiner,
                              const int id, const Node& node);

  static std::array<int64, SHAPE_ARRAY_SIZE> BuildShapeArray(
      const shape_inference::ShapeHandle& shape_handle,
      shape_inference::InferenceContext* context);

  void AppendNodeParamsWithIoParams(
      const ShapeRefiner& shape_refiner, const Node& node, const string& name,
      const int id, const string& type, const int type_id, const int padding,
      const int inputs_size, const std::vector<int>& extra_inputs,
      const int outputs_size, const bool append_input_params,
      const bool append_output_params);

  static string ToPaddingDebugString(int padding);

  // Create dependency map
  static void FillDependencyRec(
      int node_id, std::unordered_map<int, std::unordered_set<int>>& dep_map,
      std::unordered_set<int>& completed);

  // Build tensor from proto
  static Status MakeTensorFromProto(const TensorProto& tensor_proto,
                                    Tensor* tensor);

  void ClearCache();

  // Dump pretty print of parameters
  void DumpNodeTransferParams() const;

  GraphTransferInfo graph_transfer_info_{};

  std::vector<const Node*> node_name_cache_list_{};
  std::unordered_map<string, int> node_name_to_id_cache_map_{};

  // strict check mode is true by default.  Disable this if the ops' shape
  // inferences are not implemented correctly.
  bool strict_check_mode_{true};

  TF_DISALLOW_COPY_AND_ASSIGN(GraphTransferer);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_GRAPH_TRANSFERER_H
