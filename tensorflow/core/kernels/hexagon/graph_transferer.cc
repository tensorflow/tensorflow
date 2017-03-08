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

#include <algorithm>
#include <cinttypes>
#include <unordered_map>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

constexpr bool DBG_DUMP_VERIFICATION_STRING = false;
constexpr bool DBG_DUMP_PARAMS = false;

const string INPUTS_NODE_PREFIX = "inputs_for_";
const string OUTPUTS_NODE_PREFIX = "outputs_for_";
const string DATA_NODE_PREFIX = "data_for_op_";
const string CONST_SHAPE_PREFIX = "const_shape_";
const string PADDING_PREFIX = "NN_PAD_";
const string PADDING_ATTR_NAME = "padding";
const string STRIDES_ATTR_NAME = "strides";
const string KSIZE_ATTR_NAME = "ksize";
const string PADDING_VALID_STR = "VALID";
const string PADDING_SAME_STR = "SAME";
const string PADDING_NA = "NA";
const string NULL_OUTPUT_NAME = "NULL";

Status GraphTransferer::LoadGraphFromProto(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const GraphDef& graph_def, const std::vector<string>& input_node_names,
    const std::vector<string>& output_node_names) {
  ImportGraphDefOptions opts;
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.op_registry());
  VLOG(1) << "Start import graph";
  Status status = ImportGraphDef(opts, graph_def, &graph, &shape_refiner);
  if (!status.ok()) {
    VLOG(1) << "Failed to import graph " << status.ToString();
    return status;
  }

  std::unordered_multimap<string, const Node*> op_name_to_node_multimap(
      graph.num_nodes());
  for (const Node* const node : graph.nodes()) {
    CacheNode(*node);
  }

  for (const Node* const node : graph.nodes()) {
    VLOG(1) << "<Node> " << node->name();
    for (const Node* const input_node : node->in_nodes()) {
      const string& name = input_node->name();
      op_name_to_node_multimap.emplace(name, node);
      VLOG(1) << "Add dependency: " << name << " -> " << node->name();
    }
  }

  for (const Node* const node : graph.nodes()) {
    RegisterNodeIfAllInputsAreCached(ops_definitions, shape_refiner, *node,
                                     false, input_node_names,
                                     output_node_names);
  }
  if (DBG_DUMP_PARAMS) {
    DumpNodeTransferParams();
  }
  if (DBG_DUMP_VERIFICATION_STRING) {
    DumpVerificationStringOfNodeTransferParams();
  }
  return Status();
}

Status GraphTransferer::LoadGraphFromProtoFile(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const string& graph_def_path, const std::vector<string>& input_node_names,
    const std::vector<string>& output_node_names, const bool is_text_proto) {
  GraphDef graph_def;
  string output;
  Status status;
  if (is_text_proto) {
    status = ReadFileToString(Env::Default(), graph_def_path, &output);
    if (!protobuf::TextFormat::ParseFromString(output, &graph_def)) {
      return errors::InvalidArgument("Cannot parse proto string.");
    }
  } else {
    status = ReadBinaryProto(Env::Default(), graph_def_path, &graph_def);
  }
  if (!status.ok()) {
    return status;
  }
  return LoadGraphFromProto(ops_definitions, graph_def, input_node_names,
                            output_node_names);
}

Status GraphTransferer::LoadGraphFromProtoFile(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const string& graph_def_path, const std::vector<string>& input_node_names,
    const std::vector<string>& output_node_names) {
  GraphDef graph_def;
  string output;
  Status status = ReadFileToString(Env::Default(), graph_def_path, &output);
  if (!status.ok()) {
    return status;
  }
  if (!protobuf::TextFormat::ParseFromString(output, &graph_def)) {
    return errors::InvalidArgument("Cannot parse proto string.");
  }
  LoadGraphFromProto(ops_definitions, graph_def, input_node_names,
                     output_node_names);
  return Status();
}

const std::vector<GraphTransferer::ConstNodeTransferParams>&
GraphTransferer::GetConstNodeParams() const {
  return const_node_transfer_params_list_;
}

const std::vector<GraphTransferer::NodeTransferParams>&
GraphTransferer::GetOpNodeParams() const {
  return node_transfer_params_list_;
}

const std::vector<GraphTransferer::NodeInputParams>&
GraphTransferer::GetNodeInputParams() const {
  return node_input_params_list_;
}

const std::vector<GraphTransferer::NodeOutputParams>&
GraphTransferer::GetNodeOutputParams() const {
  return node_output_params_list_;
}

int GraphTransferer::CacheNode(const Node& node) {
  if (node_name_to_id_cache_map_.count(node.name()) > 0) {
    VLOG(1) << "Emplace node to cache failed";
    // TODO(satok): check here?
    return -1;
  }
  VLOG(1) << "Cache node: " << node.name() << ", " << node.op_def().name();
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
      VLOG(1) << "input_node " << input_node->name() << " of " << node.name()
              << " is not cached yet.";
      return false;
    }
  }
  return true;
}

void GraphTransferer::RegisterNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const Node& node,
    const std::vector<string>& input_node_names,
    const std::vector<string>& output_node_names) {
  VLOG(1) << "Register node: " << node.name();
  if (std::find(input_node_names.begin(), input_node_names.end(),
                node.name()) != input_node_names.end()) {
    RegisterInputNode(ops_definitions, shape_refiner, node);
  } else if (std::find(output_node_names.begin(), output_node_names.end(),
                       node.name()) != output_node_names.end()) {
    RegisterOutputNode(ops_definitions, shape_refiner, node);
  } else if (node.IsConstant()) {
    RegisterConstantNode(shape_refiner, node);
  } else if (HasPaddingAndStrides(node)) {
    RegisterNodeWithPaddingAndStrides(ops_definitions, shape_refiner, node);
  } else {
    // TODO(satok): register params for nodes which are supported by SOC
    VLOG(1) << "Not implemented for " << node.type_string();
  }
}

void GraphTransferer::RegisterConstantNode(const ShapeRefiner& shape_refiner,
                                           const Node& node) {
  VLOG(1) << "Register constant node: " << node.name();
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string data_name = DATA_NODE_PREFIX + std::to_string(id);
  const int output_node_size = node.num_outputs();
  CHECK(output_node_size == 1);
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
  const std::array<int64, SHAPE_ARRAY_SIZE> shape =
      BuildShapeArray(shape_handle, context);
  const_node_transfer_params_list_.emplace_back(
      ConstNodeTransferParams{node.name(),
                              id,
                              {{shape[0], shape[1], shape[2], shape[3]}},
                              data_name,
                              data_size});
}

int GraphTransferer::RegisterConstantShape(const std::vector<int>& shape) {
  VLOG(1) << "Cache constant shape.";
  // TODO(satok): Handle non-4dim strides
  CHECK(shape.size() == 4);
  const string shape_name = CONST_SHAPE_PREFIX + std::to_string(shape.at(0)) +
                            'x' + std::to_string(shape.at(1)) + 'x' +
                            std::to_string(shape.at(2)) + 'x' +
                            std::to_string(shape.at(3));
  if (node_name_to_id_cache_map_.count(shape_name) <= 0) {
    node_name_cache_list_.emplace_back(nullptr);
    const int id = node_name_cache_list_.size() - 1;
    node_name_to_id_cache_map_.emplace(shape_name, id);
    const_node_transfer_params_list_.emplace_back(ConstNodeTransferParams{
        shape_name, id, {{shape[0], shape[1], shape[2], shape[3]}}, "", 0});
  }
  return node_name_to_id_cache_map_[shape_name];
}

bool GraphTransferer::HasPaddingAndStrides(const Node& node) {
  return node.def().attr().count(PADDING_ATTR_NAME) > 0 &&
         node.def().attr().count(STRIDES_ATTR_NAME) > 0;
}

void GraphTransferer::RegisterNodeWithPaddingAndStrides(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const Node& node) {
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  shape_inference::InferenceContext* context = shape_refiner.GetContext(&node);
  CHECK(node.def().attr().count(PADDING_ATTR_NAME) > 0);
  // TODO(satok): Use context->GetAttr(...) instead?
  Padding padding;
  context->GetAttr(PADDING_ATTR_NAME, &padding);
  CHECK(node.def().attr().count(STRIDES_ATTR_NAME) > 0);
  std::vector<int32> strides;
  context->GetAttr(STRIDES_ATTR_NAME, &strides);
  const int stride_id = RegisterConstantShape(strides);
  std::vector<int> extra_inputs{stride_id};
  if (node.def().attr().count(KSIZE_ATTR_NAME) > 0) {
    std::vector<int32> kernel_sizes;
    context->GetAttr(KSIZE_ATTR_NAME, &kernel_sizes);
    const int ksize_id = RegisterConstantShape(kernel_sizes);
    extra_inputs.push_back(ksize_id);
  }
  const std::string padding_str =
      padding == VALID ? PADDING_VALID_STR : PADDING_SAME_STR;
  const int op_type_id = ops_definitions.GetOpIdFor(node.type_string());
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount())
      << node.type_string();
  AppendNodeParamsWithIoParams(
      shape_refiner, node, node.name(), id, node.type_string(), op_type_id,
      padding_str, node.num_inputs(), extra_inputs, node.num_outputs(),
      true /* append_input */, true /* append_output */);
}

void GraphTransferer::RegisterInputNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const Node& node) {
  VLOG(1) << "Register input node: " << node.name();
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string op_type = IGraphTransferOpsDefinitions::INPUT_OP_NAME;
  const int op_type_id = ops_definitions.GetOpIdFor(op_type);
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());
  AppendNodeParamsWithIoParams(shape_refiner, node, node.name(), id,
                               IGraphTransferOpsDefinitions::INPUT_OP_NAME,
                               op_type_id, PADDING_NA, node.num_inputs(), {},
                               node.num_outputs(), true /* append_input */,
                               true /* append_output */);
}

void GraphTransferer::RegisterOutputNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const Node& node) {
  VLOG(1) << "Register output node: " << node.name();
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string op_type = IGraphTransferOpsDefinitions::OUTPUT_OP_NAME;
  const int op_type_id = ops_definitions.GetOpIdFor(op_type);
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());
  // TODO(satok): Set output for output node?
  AppendNodeParamsWithIoParams(shape_refiner, node, node.name(), id, op_type,
                               op_type_id, PADDING_NA, node.num_inputs(), {},
                               0 /* outputs_size */, true /* append_input */,
                               false /* append_output */);
}

// TODO(satok): Remove this function.
// TODO(satok): Remove only_register_const_node.
bool GraphTransferer::RegisterNodeIfAllInputsAreCached(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const Node& node,
    const bool only_register_const_node,
    const std::vector<string>& input_node_names,
    const std::vector<string>& output_node_names) {
  if (only_register_const_node && !node.IsConstant()) {
    return false;
  }
  CHECK(AreAllInputsCached(node));
  RegisterNode(ops_definitions, shape_refiner, node, input_node_names,
               output_node_names);
  return true;
}

// CAVEAT: Append inputs and outputs params accordingly
void GraphTransferer::AppendNodeParams(const string& name, const int id,
                                       const string& type, const int type_id,
                                       const string& padding_str,
                                       const int inputs_size,
                                       const std::vector<int>& extra_inputs,
                                       const int outputs_size) {
  VLOG(1) << "Append node params: " << name;
  // TODO(satok): store padding as Padding?
  const string output_name = OUTPUTS_NODE_PREFIX + std::to_string(id);
  node_transfer_params_list_.emplace_back(
      NodeTransferParams{name, id, type, type_id, PADDING_PREFIX + padding_str,
                         INPUTS_NODE_PREFIX + std::to_string(id),
                         inputs_size + static_cast<int>(extra_inputs.size()),
                         outputs_size <= 0 ? NULL_OUTPUT_NAME : output_name,
                         static_cast<int>(outputs_size)});
}

void GraphTransferer::AppendNodeInputParams(
    const int id, const Node& node, const std::vector<int>& extra_inputs) {
  VLOG(1) << "Append input params: " << node.name() << ", " << node.num_inputs()
          << ", " << extra_inputs.size();
  NodeInputParams input_params;
  input_params.node_id = id;
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Node* input_node = nullptr;
    TF_CHECK_OK(node.input_node(i, &input_node));
    const std::string& op_name = input_node->name();
    CHECK(node_name_to_id_cache_map_.count(op_name) > 0) << op_name;
    const int src_id = node_name_to_id_cache_map_[op_name];
    input_params.input_node_id_and_output_port_list.emplace_back(
        std::make_tuple(src_id, i));
  }
  for (const int extra_input : extra_inputs) {
    input_params.input_node_id_and_output_port_list.emplace_back(
        std::make_tuple(extra_input, 0));
  }
  node_input_params_list_.emplace_back(input_params);
}

void GraphTransferer::AppendNodeOutputParams(const ShapeRefiner& shape_refiner,
                                             const int id, const Node& node) {
  VLOG(1) << "Append output params: " << node.name() << ", "
          << node.num_outputs();
  NodeOutputParams node_output_params;
  node_output_params.node_id = id;
  for (int i = 0; i < node.num_outputs(); ++i) {
    const Node* output_node = nullptr;
    for (const Edge* const output_edge : node.out_edges()) {
      if (output_edge->src_output() == i) {
        output_node = output_edge->src();
      }
    }
    CHECK(output_node != nullptr);
    const int output_index = i;
    const DataType dt = node.output_type(output_index);
    const size_t max_bytes_per_data =
        checkpoint::TensorSliceWriter::MaxBytesPerElement(dt);
    shape_inference::InferenceContext* context =
        shape_refiner.GetContext(output_node);
    shape_inference::ShapeHandle shape_handle = context->output(output_index);
    const shape_inference::DimensionHandle num_elements_dim =
        context->NumElements(shape_handle);
    CHECK(context->ValueKnown(num_elements_dim));
    const int64 num_output_elements = context->Value(num_elements_dim);
    const int data_size = max_bytes_per_data * num_output_elements;
    node_output_params.max_sizes.push_back(data_size);
  }
  node_output_params_list_.emplace_back(node_output_params);
}

void GraphTransferer::AppendNodeParamsWithIoParams(
    const ShapeRefiner& shape_refiner, const Node& node, const string& name,
    const int id, const string& type, const int type_id,
    const string& padding_str, const int inputs_size,
    const std::vector<int>& extra_inputs, const int outputs_size,
    const bool append_input_params, const bool append_output_params) {
  VLOG(1) << "Append node with io params: " << node.name();
  if (append_input_params) {
    AppendNodeInputParams(id, node, extra_inputs);
  }
  if (append_output_params) {
    AppendNodeOutputParams(shape_refiner, id, node);
  }
  AppendNodeParams(name, id, type, type_id, padding_str, inputs_size,
                   extra_inputs, outputs_size);
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
  LOG(INFO) << "*** Const Nodes ***";
  for (const ConstNodeTransferParams& params :
       const_node_transfer_params_list_) {
    LOG(INFO) << "[ " << params.node_id << " \"" << params.name << "\" (Const)";
    LOG(INFO) << "  shape: " << params.shape[0] << params.shape[1]
              << params.shape[2] << params.shape[3];
    LOG(INFO) << "  data_name: " << params.data_name;
    LOG(INFO) << "  data_size: " << params.data_size << " bytes"
              << " ]";
  }
  LOG(INFO) << "******\n";
  LOG(INFO) << "*** Op Nodes ***";
  for (const NodeTransferParams& params : node_transfer_params_list_) {
    LOG(INFO) << "[ " << params.node_id << " \"" << params.name;
    LOG(INFO) << "  type: " << params.type;
    LOG(INFO) << "  padding: " << params.padding;
    LOG(INFO) << "  inputs: " << params.inputs_name
              << ", size = " << params.inputs_size;
    LOG(INFO) << "  outputs: " << params.outputs_name
              << ", size = " << params.outputs_size << " ]";
  }
  LOG(INFO) << "******\n";
  LOG(INFO) << "*** Node input params ***";
  for (const NodeInputParams& params : node_input_params_list_) {
    LOG(INFO) << "[ " << params.node_id << " ]";
    for (const std::tuple<int, int>& pair :
         params.input_node_id_and_output_port_list) {
      LOG(INFO) << "    src node id = " << std::get<0>(pair)
                << ", output port = " << std::get<1>(pair);
    }
  }
  LOG(INFO) << "******\n";
  LOG(INFO) << "*** Node output params ***";
  for (const NodeOutputParams& params : node_output_params_list_) {
    LOG(INFO) << "[ " << params.node_id << " ]";
    for (const int max_size : params.max_sizes) {
      LOG(INFO) << "    max_size = " << max_size;
    }
  }
  LOG(INFO) << "******\n";
}

void GraphTransferer::DumpVerificationStringOfNodeTransferParams() const {
  for (const ConstNodeTransferParams& params :
       const_node_transfer_params_list_) {
    std::stringstream sstream;
    sstream << "---(CONST) [" << std::hex << params.node_id << ","
            << params.shape[0] << "," << params.shape[1] << ","
            << params.shape[2] << "," << params.shape[3] << ","
            << params.data_name << "," << params.data_size << "," << params.name
            << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Const node count = " << const_node_transfer_params_list_.size();
  for (const NodeTransferParams& params : node_transfer_params_list_) {
    std::stringstream sstream;
    sstream << "---(OP) [" << params.name.c_str() << "," << std::hex
            << params.node_id << "," << params.soc_op_id << ","
            << params.padding << "," << params.inputs_name << ","
            << params.inputs_size << "," << params.outputs_name << ","
            << params.outputs_size << "," << params.type << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Op node count = " << node_transfer_params_list_.size();
  for (const NodeInputParams& params : node_input_params_list_) {
    std::stringstream sstream;
    sstream << "---(INPUT) [" << std::hex << params.node_id;
    for (const std::tuple<int, int>& pair :
         params.input_node_id_and_output_port_list) {
      sstream << "," << std::get<0>(pair) << "," << std::get<1>(pair);
    }
    sstream << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Input params count = " << node_input_params_list_.size();
  for (const NodeOutputParams& params : node_output_params_list_) {
    std::stringstream sstream;
    sstream << "---(OUTPUT) [" << std::hex << params.node_id;
    for (const int max_size : params.max_sizes) {
      sstream << "," << max_size;
    }
    sstream << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Output params count = " << node_input_params_list_.size();
}

}  // namespace tensorflow
