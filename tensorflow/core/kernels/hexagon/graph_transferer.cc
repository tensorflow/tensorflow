/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/hexagon/graph_transferer.h"

#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

static constexpr bool DBG = false;
static constexpr const char* const INPUTS_NODE_PREFIX = "inputs_for_";
static constexpr const char* const OUTPUTS_NODE_PREFIX = "outputs_for_";
static constexpr const char* const DATA_NODE_PREFIX = "data_for_op_";

void GraphTransferer::LoadGraphFromProto(const GraphDef& graph_def) {
  ImportGraphDefOptions opts;
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.op_registry());
  ImportGraphDef(opts, graph_def, &graph, &shape_refiner);

  std::unordered_multimap<string, const Node*> op_name_to_node_multimap(
      graph.num_nodes());

  for (const Node* const node : graph.nodes()) {
    if (DBG) {
      LOG(INFO) << "<Node> " << node->name();
    }
    for (const Node* const input_node : node->in_nodes()) {
      const string& name = input_node->name();
      op_name_to_node_multimap.emplace(name, node);
      if (DBG) {
        LOG(INFO) << "Add dependency: " << name << " -> " << node->name();
      }
    }
  }

  for (const Node* const node : graph.nodes()) {
    RegisterNodeIfAllInputsAreCached(shape_refiner, *node, false);
  }
  if (DBG) {
    DumpNodeTransferParams();
  }
}

const std::vector<GraphTransferer::ConstNodeTransferParams>&
GraphTransferer::GetConstNodeParams() const {
  return const_node_transfer_params_list_;
}

int GraphTransferer::CacheNode(const Node& node) {
  if (node_name_to_id_cache_map_.count(node.name()) > 0) {
    if (DBG) {
      LOG(INFO) << "Emplace node to cache failed";
    }
    // TODO(satok): check here?
    return -1;
  }
  node_name_cache_list_.emplace_back(&node);
  bool emplace_succeeded = false;
  std::tie(std::ignore, emplace_succeeded) = node_name_to_id_cache_map_.emplace(
      node.name(), node_name_cache_list_.size() - 1);
  CHECK(emplace_succeeded);
  return node_name_cache_list_.size() - 1;
}

bool GraphTransferer::AreAllInputsCached(const Node& node) const {
  for (const Node* const input_node : node.in_nodes()) {
    if (node_name_to_id_cache_map_.count(input_node->name()) <= 0) {
      if (DBG) {
        LOG(INFO) << "input_node " << input_node->name() << " of "
                  << node.name() << " is not cached yet.";
      }
      return false;
    }
  }
  return true;
}

void GraphTransferer::RegisterNode(const ShapeRefiner& shape_refiner,
                                   const Node& node) {
  CacheNode(node);
  if (node.IsSource()) {
    // TODO(satok): register params for source node
    if (DBG) {
      LOG(INFO) << "Not implemented for source";
    }
  } else if (node.IsSink()) {
    // TODO(satok): register params for sink node
    if (DBG) {
      LOG(INFO) << "Not implemented for sink";
    }
  } else if (node.IsConstant()) {
    RegisterConstantNode(shape_refiner, node);
  } else {
    // TODO(satok): register params for nodes which are supported by SOC
    if (DBG) {
      LOG(INFO) << "Not implemented for " << node.type_string();
    }
  }
}

void GraphTransferer::RegisterConstantNode(const ShapeRefiner& shape_refiner,
                                           const Node& node) {
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string data_name = string(DATA_NODE_PREFIX) + std::to_string(id);
  const int output_node_size = node.num_outputs();
  CHECK(output_node_size > 0);  // output_node_size == 1?
  // TODO(satok): support multiple outputs?
  const int output_index = 0;
  const DataType dt = node.output_type(output_index);
  const size_t max_bytes_per_data =
      checkpoint::TensorSliceWriter::MaxBytesPerElement(dt);
  shape_inference::InferenceContext* context = shape_refiner.GetContext(&node);
  shape_inference::ShapeHandle shape_handle = context->output(output_index);
  const shape_inference::DimensionHandle num_elements_dim =
      context->NumElements(shape_handle);
  CHECK(context->ValueKnown(num_elements_dim));
  const int64 num_output_elements = context->Value(num_elements_dim);
  const int data_size = max_bytes_per_data * num_output_elements;
  const int rank = context->Rank(shape_handle);
  CHECK(rank == 0);
  const std::array<int64, SHAPE_ARRAY_SIZE> shape =
      BuildShapeArray(shape_handle, context);
  const_node_transfer_params_list_.emplace_back(
      ConstNodeTransferParams{node.name(),
                              id,
                              {{shape[0], shape[1], shape[2], shape[3]}},
                              data_name,
                              data_size});
}

bool GraphTransferer::RegisterNodeIfAllInputsAreCached(
    const ShapeRefiner& shape_refiner, const Node& node,
    const bool only_register_const_node) {
  if (only_register_const_node && !node.IsConstant()) {
    return false;
  }
  if (!AreAllInputsCached(node)) {
    return false;
  }
  RegisterNode(shape_refiner, node);
  return true;
}

void GraphTransferer::AppendNodeParams(const string& name, const int id,
                                       const string& type,
                                       const string& padding,
                                       const int inputs_size,
                                       const int outputs_size) {
  node_transfer_params_list_.emplace_back(NodeTransferParams{
      name, id, type, padding,
      string(INPUTS_NODE_PREFIX) + std::to_string(inputs_size), inputs_size,
      string(OUTPUTS_NODE_PREFIX) + std::to_string(outputs_size),
      outputs_size});
}

/* static */ std::array<int64, GraphTransferer::SHAPE_ARRAY_SIZE>
GraphTransferer::BuildShapeArray(
    const shape_inference::ShapeHandle& shape_handle,
    shape_inference::InferenceContext* context) {
  switch (context->Rank(shape_handle)) {
    case 0:
      return std::array<int64, SHAPE_ARRAY_SIZE>{{1, 1, 1, 1}};
    case 1:
      return std::array<int64, SHAPE_ARRAY_SIZE>{
          {1, 1, 1, context->Value(context->Dim(shape_handle, 0))}};
    case 2:
      return std::array<int64, SHAPE_ARRAY_SIZE>{
          {1, 1, context->Value(context->Dim(shape_handle, 0)),
           context->Value(context->Dim(shape_handle, 1))}};
    case 3:
      return std::array<int64, SHAPE_ARRAY_SIZE>{
          {1, context->Value(context->Dim(shape_handle, 0)),
           context->Value(context->Dim(shape_handle, 1)),
           context->Value(context->Dim(shape_handle, 2))}};
    case 4:
      return std::array<int64, SHAPE_ARRAY_SIZE>{
          {context->Value(context->Dim(shape_handle, 0)),
           context->Value(context->Dim(shape_handle, 1)),
           context->Value(context->Dim(shape_handle, 2)),
           context->Value(context->Dim(shape_handle, 3))}};
    default:
      // TODO(satok): Support more ranks?
      CHECK(false);
      return std::array<int64, SHAPE_ARRAY_SIZE>();
  }
}

void GraphTransferer::DumpNodeTransferParams() const {
  // TODO(satok): Dump all params
  for (const ConstNodeTransferParams& params :
       const_node_transfer_params_list_) {
    LOG(INFO) << "[ " << params.id << " \"" << params.name << "\" (Const)";
    LOG(INFO) << "  shape: " << params.shape[0] << params.shape[1]
              << params.shape[2] << params.shape[3];
    LOG(INFO) << "  data_name: " << params.data_name;
    LOG(INFO) << "  data_size: " << params.data_size << " bytes"
              << " ]";
  }
}

}  // namespace tensorflow
