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
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"

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
    int id;
    string type;
    string padding;
    string inputs_name;  // for debug info
    int inputs_size;
    string outputs_name;  // for debug info
    int outputs_size;
  };

  // Const node parameters for transfer
  struct ConstNodeTransferParams {
    string name;
    int id;
    std::array<int64, MAX_SUPPORTED_RANK> shape;
    string data_name;  // for debug info
    int data_size;
  };

  GraphTransferer() = default;

  // Load graph structure into GraphTransferer
  void LoadGraphFromProto(const GraphDef& graph_def);

  // Return const node parameters for transfer
  const std::vector<ConstNodeTransferParams>& GetConstNodeParams() const;

 private:
  int CacheNode(const Node& node);
  bool AreAllInputsCached(const Node& node) const;
  void RegisterConstantNode(const ShapeRefiner& shape_refiner,
                            const Node& node);
  void RegisterNode(const ShapeRefiner& shape_refiner, const Node& node);
  bool RegisterNodeIfAllInputsAreCached(const ShapeRefiner& shape_refiner,
                                        const Node& node,
                                        const bool only_register_const_node);
  void AppendNodeParams(const string& name, const int id, const string& type,
                        const string& padding, const int inputs_size,
                        const int outputs_size);
  static std::array<int64, SHAPE_ARRAY_SIZE> BuildShapeArray(
      const shape_inference::ShapeHandle& shape_handle,
      shape_inference::InferenceContext* context);
  void DumpNodeTransferParams() const;

  std::vector<NodeTransferParams> node_transfer_params_list_;
  std::vector<ConstNodeTransferParams> const_node_transfer_params_list_;

  std::vector<const Node*> node_name_cache_list_;
  std::unordered_map<string, int> node_name_to_id_cache_map_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphTransferer);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_HEXAGON_GRAPH_TRANSFERER_H
