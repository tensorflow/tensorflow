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
#include "tensorflow/core/kernels/hexagon/i_graph_transfer_ops_definitions.h"
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
  using OutputTensorMap = std::unordered_map<string, Tensor*>;

  struct InputNodeInfo {
    string name;
    Tensor tensor;
  };

  struct OutputTensorInfo {
    std::vector<Tensor> output_tensors;
    OutputTensorMap output_tensor_map;
  };

  GraphTransferer() = default;

  // Load graph structure into GraphTransferer
  Status LoadGraphFromProto(
      const IGraphTransferOpsDefinitions& ops_definitions,
      const GraphDef& graph_def,
      const std::vector<InputNodeInfo>& input_node_info_list,
      const std::vector<string>& output_node_names,
      const bool shape_inference_for_unkown_shape,
      const OutputTensorMap& output_tensor_map);

  // Load graph structure into GraphTransferer from protobuf file
  Status LoadGraphFromProtoFile(
      const IGraphTransferOpsDefinitions& ops_definitions,
      const string& graph_def_path,
      const std::vector<InputNodeInfo>& input_node_info_list,
      const std::vector<string>& output_node_names, const bool is_text_proto,
      const bool shape_inference_for_unknown_shape,
      const bool dry_run_for_unknown_shape,
      OutputTensorInfo* output_tensor_info);

  // Dry run inference and cache the result to get memory mapping
  static Status DryRunInference(
      const GraphDef& graph_def,
      const std::vector<InputNodeInfo>& input_node_info_list,
      const std::vector<string>& output_node_names,
      const bool initialize_by_zero,
      std::vector<tensorflow::Tensor>* output_tensors);

  // Dry run inference and fill output tensors to output tensor info
  // CAVEAT: Do not add or modify output_tensors in output_tensor_info
  // otherwise, address map may be broken by re-allocation inside
  // std::vector
  static Status DryRunInferenceForAllNode(
      const GraphDef& graph_def,
      const std::vector<InputNodeInfo>& input_node_info_list,
      const bool initialize_by_zero, OutputTensorInfo* output_tensor_info);

  // Sort params so that all input nodes appear before consumer nodes.
  // CAVEAT: This may be slow if the number of nodes are too large
  void SortParams(const std::vector<string>& output_node_names);

  void EnableStrictCheckMode(bool enable);

  // Import parameters for transfer
  void SetSerializedGraphTransferInfo(const string& serialized_proto);

  // Return parameters for graph transfer
  const GraphTransferInfo& GetGraphTransferInfo() const;

 private:
  class TransferParamsComparator {
   public:
    TransferParamsComparator(
        const std::unordered_map<int, std::unordered_set<int>>& dep_map);
    bool operator()(const GraphTransferInfo::NodeInfo& obj0,
                    const GraphTransferInfo::NodeInfo& obj1);
    const std::unordered_map<int, std::unordered_set<int>>& dependency_map_;
  };

  int CacheNode(const Node& node);

  static bool IsInputNode(
      const std::vector<InputNodeInfo>& input_node_info_list,
      const string& node_name);

  bool AreAllInputsCached(const Node& node) const;

  Status RegisterNode(const IGraphTransferOpsDefinitions& ops_definitions,
                      const ShapeRefiner& shape_refiner,
                      const OutputTensorMap& output_tensor_map,
                      const Node& node,
                      const std::vector<InputNodeInfo>& input_node_info_list,
                      const std::vector<string>& output_node_names);

  void RegisterConstantNode(const ShapeRefiner& shape_refiner, const Node& node,
                            const OutputTensorMap& output_tensor_map);

  int RegisterConstantShape(const std::vector<int>& shape);

  bool HasPaddingAndStrides(const Node& node);

  // Return true if the node is a reshape op which just flattens input
  // TODO(satok): Remove this method once generic reshape op is implemented in
  // SOC
  bool IsNodeFlattenReshape(const Node& node,
                            const OutputTensorMap& output_tensor_map,
                            const ShapeRefiner& shape_refiner);

  void RegisterNodeWithPaddingAndStrides(
      const IGraphTransferOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner,
      const OutputTensorMap& output_tensor_map, const Node& node);

  void RegisterInputNode(const IGraphTransferOpsDefinitions& ops_definitions,
                         const ShapeRefiner& shape_refiner,
                         const OutputTensorMap& output_tensor_map,
                         const Node& node);

  void RegisterOutputNode(const IGraphTransferOpsDefinitions& ops_definitions,
                          const ShapeRefiner& shape_refiner,
                          const OutputTensorMap& output_tensor_map,
                          const Node& node);

  void RegisterFlattenNode(const IGraphTransferOpsDefinitions& ops_definitions,
                           const ShapeRefiner& shape_refiner,
                           const OutputTensorMap& output_tensor_map,
                           const Node& node);

  void RegisterGenericNode(const IGraphTransferOpsDefinitions& ops_definitions,
                           const ShapeRefiner& shape_refiner,
                           const OutputTensorMap& output_tensor_map,
                           const Node& node);

  Status RegisterNodeIfAllInputsAreCached(
      const IGraphTransferOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner, const Node& node,
      const bool only_register_const_node,
      const std::vector<InputNodeInfo>& input_node_info_list,
      const std::vector<string>& output_node_names,
      const OutputTensorMap& output_tensor_map);

  void AppendNodeParams(const string& name, const int id, const string& type,
                        const int type_id, const int padding,
                        const int inputs_size,
                        const std::vector<int>& extra_inputs,
                        const int outputs_size);

  void AppendNodeInputParams(const int id, const Node& node,
                             const std::vector<int>& extra_inputs);

  void AppendNodeOutputParams(const ShapeRefiner& shape_refiner,
                              const OutputTensorMap& output_tensor_map,
                              const int id, const Node& node);

  static std::array<int64, SHAPE_ARRAY_SIZE> BuildShapeArray(
      const shape_inference::ShapeHandle& shape_handle,
      shape_inference::InferenceContext* context);

  void AppendNodeParamsWithIoParams(
      const ShapeRefiner& shape_refiner,
      const OutputTensorMap& output_tensor_map, const Node& node,
      const string& name, const int id, const string& type, const int type_id,
      const int padding, const int inputs_size,
      const std::vector<int>& extra_inputs, const int outputs_size,
      const bool append_input_params, const bool append_output_params);

  static std::array<int64, SHAPE_ARRAY_SIZE> ToTensorShapeArray(
      const TensorShape& shape);

  static string ToPaddingDebugString(int padding);

  static void CheckShape(const OutputTensorMap& output_tensor_map,
                         const string& node_name,
                         const std::array<int64, SHAPE_ARRAY_SIZE>& actual);

  // Create dependency map
  static void FillDependencyRec(
      int node_id, std::unordered_map<int, std::unordered_set<int>>& dep_map,
      std::unordered_set<int>& completed);

  void ClearCache();

  // Dump pretty print of parameters
  void DumpNodeTransferParams() const;

  // Dump verification string of parameters to verify with offline tools
  void DumpVerificationStringOfNodeTransferParams() const;

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
