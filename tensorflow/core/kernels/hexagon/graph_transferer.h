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
#include <vector>

#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/graph.pb.h"
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
  static constexpr int MAX_SUPPORTED_RANK = 5;
  static constexpr int SHAPE_ARRAY_SIZE = MAX_SUPPORTED_RANK - 1;

  // Node parameters for transfer
  struct NodeTransferParams {
    string name;
    int node_id;
    string type;  // for debug info
    int soc_op_id;
    string padding;
    string inputs_name;  // for debug info TODO(satok): remove
    int inputs_size;
    string outputs_name;  // for debug info TODO(satok): remove
    int outputs_size;
  };

  // Const node parameters for transfer
  struct ConstNodeTransferParams {
    string name;
    int node_id;
    std::array<int64, MAX_SUPPORTED_RANK> shape;
    string data_name;  // for debug info
    int data_size;
  };

  // Input parameters of a node for transfer
  struct NodeInputParams {
    int node_id;
    std::vector<std::tuple<int, int>> input_node_id_and_output_port_list;
  };

  // Output parameters of a node for transfer
  struct NodeOutputParams {
    int node_id;
    std::vector<int> max_sizes;
  };

  GraphTransferer() = default;

  // Load graph structure into GraphTransferer
  Status LoadGraphFromProto(const IGraphTransferOpsDefinitions& ops_definitions,
                            const GraphDef& graph_def,
                            const std::vector<string>& input_node_names,
                            const std::vector<string>& output_node_names);

  // Load graph structure into GraphTransferer from protobuf file
  Status LoadGraphFromProtoFile(
      const IGraphTransferOpsDefinitions& ops_definitions,
      const string& graph_def_path, const std::vector<string>& input_node_names,
      const std::vector<string>& output_node_names, const bool is_text_proto);

  // Load graph structure into GraphTransferer from protobuf file
  Status LoadGraphFromProtoFile(
      const IGraphTransferOpsDefinitions& ops_definitions,
      const string& graph_def_path, const std::vector<string>& input_node_names,
      const std::vector<string>& output_node_names);

  // Return const node parameters for transfer
  const std::vector<ConstNodeTransferParams>& GetConstNodeParams() const;

  // Return op node parameters for transfer
  const std::vector<NodeTransferParams>& GetOpNodeParams() const;

  // Return input params of nodes
  const std::vector<NodeInputParams>& GetNodeInputParams() const;

  // Return output params of nodes
  const std::vector<NodeOutputParams>& GetNodeOutputParams() const;

 private:
  int CacheNode(const Node& node);
  bool AreAllInputsCached(const Node& node) const;
  void RegisterNode(const IGraphTransferOpsDefinitions& ops_definitions,
                    const ShapeRefiner& shape_refiner, const Node& node,
                    const std::vector<string>& input_node_names,
                    const std::vector<string>& output_node_names);
  void RegisterConstantNode(const ShapeRefiner& shape_refiner,
                            const Node& node);
  int RegisterConstantShape(const std::vector<int>& shape);
  bool HasPaddingAndStrides(const Node& node);
  void RegisterNodeWithPaddingAndStrides(
      const IGraphTransferOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner, const Node& node);
  void RegisterInputNode(const IGraphTransferOpsDefinitions& ops_definitions,
                         const ShapeRefiner& shape_refiner, const Node& node);
  void RegisterOutputNode(const IGraphTransferOpsDefinitions& ops_definitions,
                          const ShapeRefiner& shape_refiner, const Node& node);
  bool RegisterNodeIfAllInputsAreCached(
      const IGraphTransferOpsDefinitions& ops_definitions,
      const ShapeRefiner& shape_refiner, const Node& node,
      const bool only_register_const_node,
      const std::vector<string>& input_node_names,
      const std::vector<string>& output_node_names);
  void AppendNodeParams(const string& name, const int id, const string& type,
                        const int type_id, const string& padding_str,
                        const int inputs_size,
                        const std::vector<int>& extra_inputs,
                        const int outputs_size);
  void AppendNodeInputParams(const int id, const Node& node,
                             const std::vector<int>& extra_inputs);
  void AppendNodeOutputParams(const ShapeRefiner& shape_refiner, const int id,
                              const Node& node);
  static std::array<int64, SHAPE_ARRAY_SIZE> BuildShapeArray(
      const shape_inference::ShapeHandle& shape_handle,
      shape_inference::InferenceContext* context);
  void AppendNodeParamsWithIoParams(
      const ShapeRefiner& shape_refiner, const Node& node, const string& name,
      const int id, const string& type, const int type_id,
      const string& padding_str, const int inputs_size,
      const std::vector<int>& extra_inputs, const int outputs_size,
      const bool append_input_params, const bool append_output_params);
  // Dump pretty print of parameters
  void DumpNodeTransferParams() const;
  // Dump verification string of parameters to verify with offline tools
  void DumpVerificationStringOfNodeTransferParams() const;

  std::vector<NodeTransferParams> node_transfer_params_list_;
  std::vector<ConstNodeTransferParams> const_node_transfer_params_list_;
  std::vector<NodeInputParams> node_input_params_list_;
  std::vector<NodeOutputParams> node_output_params_list_;

  std::vector<const Node*> node_name_cache_list_;
  std::unordered_map<string, int> node_name_to_id_cache_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphTransferer);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_GRAPH_TRANSFERER_H
