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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

constexpr bool DBG_DUMP_VERIFICATION_STRING = false;
constexpr bool DBG_DUMP_PARAMS = false;

const string RESHAPE_NODE_TYPE_STRING = "Reshape";
const string SOURCE_NODE_NAME = "_SOURCE";
const string SINK_NODE_NAME = "_SINK";
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

// This is a temporary workaround to support android build
// where std::string is not supported even with c++11 option.
template <typename T>
static string ToString(T val) {
  std::stringstream stream;
  stream << val;
  return stream.str();
}

/**
 * graph loading functions
 * - LoadGraphFromProto
 * - LoadGraphFromProtoFile
 * These functions read a graph definition and store parameters
 * of node to transfer the graph to SOC.
 */
Status GraphTransferer::LoadGraphFromProto(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const GraphDef& graph_def,
    const std::vector<InputNodeInfo>& input_node_info_list,
    const std::vector<string>& output_node_names,
    const OutputTensorMap& output_tensor_map) {
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
    status = RegisterNodeIfAllInputsAreCached(
        ops_definitions, shape_refiner, *node, false, input_node_info_list,
        output_node_names, output_tensor_map);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to transfer graph " << status;
      return status;
    }
  }
  SortParams(output_node_names);
  ClearCache();
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
    const string& graph_def_path,
    const std::vector<InputNodeInfo>& input_node_info_list,
    const std::vector<string>& output_node_names, const bool is_text_proto,
    const bool dry_run_for_unknown_shape,
    OutputTensorInfo* output_tensor_info) {
  GraphDef graph_def;
  string output;
  Status status;
  VLOG(1) << "Parse file " << graph_def_path;
  if (is_text_proto) {
    status = ReadFileToString(Env::Default(), graph_def_path, &output);
    if (!protobuf::TextFormat::ParseFromString(output, &graph_def)) {
      return errors::InvalidArgument("Cannot parse proto string.");
    }
  } else {
    status = ReadBinaryProto(Env::Default(), graph_def_path, &graph_def);
  }
  if (!status.ok()) {
    VLOG(1) << "Failed to load graph " << status;
    return status;
  }
  if (dry_run_for_unknown_shape) {
    VLOG(1) << "Dry run graph to obtain shape of nodes";
    status = DryRunInferenceForAllNode(graph_def, input_node_info_list, true,
                                       output_tensor_info);
    if (!status.ok()) {
      return status;
    }
  }
  VLOG(1) << "Load graph with output tensors";
  return LoadGraphFromProto(ops_definitions, graph_def, input_node_info_list,
                            output_node_names,
                            output_tensor_info->output_tensor_map);
}

/**
 * Dryrun functions
 * - DryRunInference
 * To determine shapes of output tensors of all nodes, dryrun the graph.
 * This function supplies memory allocation information when loading
 * the graph.
 * TODO(satok): Delete this function when all shapes of ops are implemented.
 * This function doesn't work if some ops changes its shape even if input shape
 * is fixed.
 */
/* static */ Status GraphTransferer::DryRunInference(
    const GraphDef& graph_def,
    const std::vector<InputNodeInfo>& input_node_info_list,
    const std::vector<string>& output_node_names, const bool initialize_by_zero,
    std::vector<tensorflow::Tensor>* output_tensors) {
  // Create input tensor vector.  If "initialize_by_zero" is true,
  // input tensor fields are initialized by 0.
  std::vector<std::pair<string, tensorflow::Tensor> > input_tensors;
  for (const InputNodeInfo& input : input_node_info_list) {
    CHECK(input.tensor.IsInitialized());
    if (!initialize_by_zero) {
      input_tensors.push_back({input.name, input.tensor});
      continue;
    }
    // If input tensor is not initialized, initialize by 0-filling
    const DataType data_type = input.tensor.dtype();
    const TensorShape& shape = input.tensor.shape();
    Tensor input_tensor(data_type, shape);
    switch (data_type) {
      case DT_INT32: {
        auto int_tensor = input_tensor.flat<int32>();
        int_tensor = int_tensor.constant(0);
        break;
      }
      case DT_FLOAT: {
        auto float_tensor = input_tensor.flat<float>();
        float_tensor = float_tensor.constant(0.0f);
        break;
      }
      case DT_QUINT8: {
        auto int_tensor = input_tensor.flat<quint8>();
        int_tensor = int_tensor.constant(0);
        break;
      }
      default:
        LOG(FATAL) << "Unsupported input type: " << data_type;
    }
    input_tensors.push_back({input.name, input_tensor});
  }

  // Setup session
  CHECK(output_tensors != nullptr);
  SessionOptions session_options;
  session_options.env = Env::Default();
  std::unique_ptr<Session> session =
      std::unique_ptr<Session>(NewSession(session_options));
  Status status = session->Create(graph_def);
  if (!status.ok()) {
    return status;
  }

  // Setup session arguments
  RunOptions run_options;
  run_options.set_trace_level(RunOptions::FULL_TRACE);
  RunMetadata run_metadata;

  // Run inference with all node as output
  status = session->Run(run_options, input_tensors, output_node_names, {},
                        output_tensors, &run_metadata);
  if (!status.ok()) {
    LOG(ERROR) << "Error during inference: " << status;
    return status;
  }
  return Status();
}

/* static */ Status GraphTransferer::DryRunInferenceForAllNode(
    const GraphDef& graph_def,
    const std::vector<GraphTransferer::InputNodeInfo>& input_node_info_list,
    const bool initialize_by_zero, OutputTensorInfo* const output_tensor_info) {
  CHECK(output_tensor_info != nullptr);
  auto& output_tensors = output_tensor_info->output_tensors;
  output_tensors.reserve(graph_def.node_size());
  auto& output_tensor_map = output_tensor_info->output_tensor_map;
  std::vector<string> output_node_names;
  for (const NodeDef& node : graph_def.node()) {
    if (!IsInputNode(input_node_info_list, node.name())) {
      output_node_names.emplace_back(node.name());
    }
  }
  const Status status =
      DryRunInference(graph_def, input_node_info_list, output_node_names,
                      initialize_by_zero, &output_tensors);
  if (!status.ok()) {
    VLOG(1) << "Failed to dryrun " << status;
    return status;
  }
  CHECK(output_node_names.size() == output_tensors.size())
      << output_node_names.size() << ", " << output_tensors.size();

  // Append output tensor of input node in advance to create a map
  // to avoid memory reallocation inside vector
  for (const InputNodeInfo& input_node_info : input_node_info_list) {
    output_tensors.push_back(input_node_info.tensor);
  }

  for (int i = 0; i < output_node_names.size(); ++i) {
    const string& name = output_node_names.at(i);
    CHECK(output_tensor_map.count(name) == 0);
    output_tensor_map[name] = &output_tensors.at(i);
  }
  for (int i = 0; i < input_node_info_list.size(); ++i) {
    const string& name = input_node_info_list.at(i).name;
    CHECK(output_tensor_map.count(name) == 0);
    output_tensor_map.emplace(name,
                              &output_tensors.at(output_node_names.size() + i));
  }
  CHECK(graph_def.node_size() == output_tensors.size());
  return status;
}

void GraphTransferer::SortParams(const std::vector<string>& output_node_names) {
  // TODO(satok): optimize complexity
  std::unordered_map<int, NodeInputParams*> input_map;
  for (NodeInputParams& input : node_input_params_list_) {
    input_map.emplace(input.node_id, &input);
  }

  // Setup dependency map placeholder
  std::vector<int> output_node_ids;
  std::unordered_map<int, std::unordered_set<int>> dependency_map;
  for (const NodeTransferParams& params : node_transfer_params_list_) {
    const int node_id = params.node_id;
    for (const string& output_node_name : output_node_names) {
      if (params.name == output_node_name) {
        output_node_ids.emplace_back(node_id);
      }
    }

    dependency_map.emplace(std::piecewise_construct, std::make_tuple(node_id),
                           std::make_tuple());
    if (params.inputs_size == 0) {
      continue;
    }
    CHECK(input_map.count(node_id) == 1);
    for (std::tuple<int, int>& id_and_port :
         input_map.at(node_id)->input_node_id_and_output_port_list) {
      dependency_map.at(node_id).emplace(std::get<0>(id_and_port));
    }
  }

  // Create dependency map traversed from output nodes
  std::unordered_set<int> completed;
  for (int output_node_id : output_node_ids) {
    FillDependencyRec(output_node_id, dependency_map, completed);
  }

  std::sort(node_transfer_params_list_.begin(),
            node_transfer_params_list_.end(),
            TransferParamsComparator(dependency_map));
}

void GraphTransferer::EnableStrictCheckMode(const bool enable) {
  strict_check_mode_ = enable;
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

/* static */ bool GraphTransferer::IsInputNode(
    const std::vector<InputNodeInfo>& input_node_info_list,
    const string& node_name) {
  for (const InputNodeInfo& input_node_info : input_node_info_list) {
    if (node_name == input_node_info.name) {
      return true;
    }
  }
  return false;
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

Status GraphTransferer::RegisterNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const Node& node, const std::vector<InputNodeInfo>& input_node_info_list,
    const std::vector<string>& output_node_names) {
  VLOG(1) << "Register node: " << node.name();
  if (node.name() == SOURCE_NODE_NAME || node.name() == SINK_NODE_NAME) {
    // Just ignore sink and source
    return Status();
  } else if (IsInputNode(input_node_info_list, node.name())) {
    RegisterInputNode(ops_definitions, shape_refiner, output_tensor_map, node);
  } else if (std::find(output_node_names.begin(), output_node_names.end(),
                       node.name()) != output_node_names.end()) {
    RegisterOutputNode(ops_definitions, shape_refiner, output_tensor_map, node);
  } else if (node.IsConstant()) {
    RegisterConstantNode(shape_refiner, node, output_tensor_map);
  } else if (HasPaddingAndStrides(node)) {
    RegisterNodeWithPaddingAndStrides(ops_definitions, shape_refiner,
                                      output_tensor_map, node);
  } else if (IsNodeFlattenReshape(node, output_tensor_map, shape_refiner)) {
    RegisterFlattenNode(ops_definitions, shape_refiner, output_tensor_map,
                        node);
  } else if (ops_definitions.GetOpIdFor(node.type_string()) !=
             IGraphTransferOpsDefinitions::INVALID_OP_ID) {
    RegisterGenericNode(ops_definitions, shape_refiner, output_tensor_map,
                        node);
  } else {
    return errors::InvalidArgument(node.type_string() +
                                   " has not implemented yet.");
  }
  return Status();
}

void GraphTransferer::RegisterConstantNode(
    const ShapeRefiner& shape_refiner, const Node& node,
    const OutputTensorMap& output_tensor_map) {
  VLOG(1) << "Register constant node: " << node.name();
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string data_name = DATA_NODE_PREFIX + ToString(id);
  const int output_node_size = node.num_outputs();
  CHECK(output_node_size == 1);
  // TODO(satok): support multiple outputs?
  const int output_index = 0;
  const DataType dt = node.output_type(output_index);
  const size_t max_bytes_per_data = DataTypeSize(dt);
  CHECK(max_bytes_per_data > 0) << "dt = " << dt << ", " + DataTypeString(dt)
                                << ", " << max_bytes_per_data << ", "
                                << (int)(DataTypeSize(dt)) << ",,,,,,,";
  shape_inference::InferenceContext* context = shape_refiner.GetContext(&node);
  shape_inference::ShapeHandle shape_handle = context->output(output_index);
  const shape_inference::DimensionHandle num_elements_dim =
      context->NumElements(shape_handle);
  std::array<int64, SHAPE_ARRAY_SIZE> shape;
  int data_size;
  if (context->ValueKnown(num_elements_dim)) {
    const int64 num_output_elements = context->Value(num_elements_dim);
    data_size = max_bytes_per_data * num_output_elements;
    shape = BuildShapeArray(shape_handle, context);
    CheckShape(output_tensor_map, node.name(), shape);
  } else {
    // Use output tensor for unknown shape
    // TODO(stok): Remove this fallback
    CHECK(!output_tensor_map.empty());
    const TensorShape& tensor_shape =
        output_tensor_map.at(node.name())->shape();
    shape = ToTensorShapeArray(tensor_shape);
    data_size = max_bytes_per_data * tensor_shape.num_elements();
  }
  CHECK(context->ValueKnown(num_elements_dim));
  const_node_transfer_params_list_.emplace_back(
      ConstNodeTransferParams{node.name(),
                              id,
                              {{shape[0], shape[1], shape[2], shape[3]}},
                              data_name,
                              data_size});
  // TODO(satok): Remove. Determine constant value without dryrun
  if (!output_tensor_map.empty() && data_size != 0) {
    const Tensor* tensor = output_tensor_map.at(node.name());
    CHECK(tensor != nullptr);
    StringPiece sp = tensor->tensor_data();
    CHECK(data_size == sp.size());
    std::vector<uint8>& data = const_node_transfer_params_list_.back().data;
    data.resize(sp.size());
    std::memcpy(&data[0], &sp.data()[0], data_size);
  }
}

int GraphTransferer::RegisterConstantShape(const std::vector<int>& shape) {
  VLOG(1) << "Cache constant shape.";
  // TODO(satok): Handle non-4dim strides
  CHECK(shape.size() == 4);
  const string shape_name = CONST_SHAPE_PREFIX + ToString(shape.at(0)) + 'x' +
                            ToString(shape.at(1)) + 'x' +
                            ToString(shape.at(2)) + 'x' + ToString(shape.at(3));
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

bool GraphTransferer::IsNodeFlattenReshape(
    const Node& node, const OutputTensorMap& output_tensor_map,
    const ShapeRefiner& shape_refiner) {
  // Check if node is reshape op
  if (node.type_string() != RESHAPE_NODE_TYPE_STRING) {
    return false;
  }

  shape_inference::InferenceContext* context = shape_refiner.GetContext(&node);
  // Check if output count is valid
  if (context->num_outputs() != 1) {
    return false;
  }

  shape_inference::ShapeHandle shape_handle = context->output(0);
  std::array<int64, SHAPE_ARRAY_SIZE> shape;
  const shape_inference::DimensionHandle dim_handle =
      context->NumElements(shape_handle);

  // Obtain shape of output of node
  if (context->ValueKnown(dim_handle)) {
    shape = BuildShapeArray(shape_handle, context);
  } else {
    // Use output tensor for unknown shape
    // TODO(stok): Remove this fallback
    CHECK(!output_tensor_map.empty());
    const TensorShape& tensor_shape =
        output_tensor_map.at(node.name())->shape();
    shape = ToTensorShapeArray(tensor_shape);
  }

  // check if reshape op just does flatten
  if (shape[0] == 1 && shape[1] == 1 && shape[2] == 1) {
    return true;
  } else {
    return false;
  }
}

void GraphTransferer::RegisterNodeWithPaddingAndStrides(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const Node& node) {
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
    extra_inputs.insert(extra_inputs.begin(), ksize_id);
  }
  const std::string padding_str =
      padding == VALID ? PADDING_VALID_STR : PADDING_SAME_STR;
  const int op_type_id = ops_definitions.GetOpIdFor(node.type_string());
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount())
      << "Op " << node.type_string() << " not found in map(id = " << op_type_id
      << ")";
  AppendNodeParamsWithIoParams(shape_refiner, output_tensor_map, node,
                               node.name(), id, node.type_string(), op_type_id,
                               padding_str, node.num_inputs(), extra_inputs,
                               node.num_outputs(), true /* append_input */,
                               true /* append_output */);
}

void GraphTransferer::RegisterInputNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const Node& node) {
  VLOG(1) << "Register input node: " << node.name();
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string op_type = IGraphTransferOpsDefinitions::INPUT_OP_NAME;
  const int op_type_id = ops_definitions.GetOpIdFor(op_type);
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());
  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, PADDING_NA, node.num_inputs(), {},
      node.num_outputs(), true /* append_input */, true /* append_output */);
}

void GraphTransferer::RegisterOutputNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const Node& node) {
  VLOG(1) << "Register output node: " << node.name();
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string op_type = IGraphTransferOpsDefinitions::OUTPUT_OP_NAME;
  const int op_type_id = ops_definitions.GetOpIdFor(op_type);
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());
  // TODO(satok): Set output for output node?
  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, PADDING_NA, node.num_inputs(), {},
      0 /* outputs_size */, true /* append_input */, false /* append_output */);
}

void GraphTransferer::RegisterFlattenNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const Node& node) {
  VLOG(1) << "Register flatten node: " << node.name();
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string op_type = IGraphTransferOpsDefinitions::FLATTEN_OP_NAME;
  const int op_type_id = ops_definitions.GetOpIdFor(op_type);
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());

  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, PADDING_NA, node.num_inputs(), {},
      node.num_outputs(), true /* append_input */, true /* append_output */);
}

void GraphTransferer::RegisterGenericNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const Node& node) {
  VLOG(1) << "Register generic node: " << node.name();
  CHECK(node_name_to_id_cache_map_.count(node.name()) == 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const int op_type_id = ops_definitions.GetOpIdFor(node.type_string());
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());

  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, PADDING_NA, node.num_inputs(), {},
      node.num_outputs(), true /* append_input */, true /* append_output */);
}

// TODO(satok): Remove this function.
// TODO(satok): Remove only_register_const_node.
Status GraphTransferer::RegisterNodeIfAllInputsAreCached(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const Node& node,
    const bool only_register_const_node,
    const std::vector<InputNodeInfo>& input_node_info_list,
    const std::vector<string>& output_node_names,
    const OutputTensorMap& output_tensor_map) {
  if (only_register_const_node && !node.IsConstant()) {
    return Status();
  }
  CHECK(AreAllInputsCached(node));
  return RegisterNode(ops_definitions, shape_refiner, output_tensor_map, node,
                      input_node_info_list, output_node_names);
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
  const string output_name = OUTPUTS_NODE_PREFIX + ToString(id);
  node_transfer_params_list_.emplace_back(
      NodeTransferParams{name, id, type, type_id, PADDING_PREFIX + padding_str,
                         INPUTS_NODE_PREFIX + ToString(id),
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
    const Edge* edge = nullptr;
    TF_CHECK_OK(node.input_edge(i, &edge));
    const Node* input_node = edge->src();
    const int port = edge->src_output();

    const std::string& op_name = input_node->name();
    CHECK(node_name_to_id_cache_map_.count(op_name) > 0) << op_name;
    const int src_id = node_name_to_id_cache_map_[op_name];
    input_params.input_node_id_and_output_port_list.emplace_back(
        std::make_tuple(src_id, port));
  }
  for (const int extra_input : extra_inputs) {
    input_params.input_node_id_and_output_port_list.emplace_back(
        std::make_tuple(extra_input, 0));
  }
  node_input_params_list_.emplace_back(input_params);
}

void GraphTransferer::AppendNodeOutputParams(
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
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
    CHECK(output_node != nullptr) << node.name() << ", " << node.type_string();
    const int output_index = i;
    const DataType dt = node.output_type(output_index);
    const size_t max_bytes_per_data = DataTypeSize(dt);
    shape_inference::InferenceContext* context =
        shape_refiner.GetContext(output_node);
    shape_inference::ShapeHandle shape_handle = context->output(output_index);
    const shape_inference::DimensionHandle num_elements_dim =
        context->NumElements(shape_handle);
    int data_size = -1;
    if (context->ValueKnown(num_elements_dim)) {
      const int64 num_output_elements = context->Value(num_elements_dim);
      data_size = max_bytes_per_data * num_output_elements;
      if (!output_tensor_map.empty() && strict_check_mode_) {
        CHECK(output_tensor_map.count(node.name()) == 1) << node.name();
        const TensorShape& tensor_shape =
            output_tensor_map.at(node.name())->shape();
        CHECK(num_output_elements == tensor_shape.num_elements())
            << "num elements of node " << node.name() << " doesn't match "
            << num_output_elements << " vs " << tensor_shape.num_elements()
            << ", " << node.type_string();
      }
    } else {
      // Use dryrun result to get the output data size
      // TODO(satok): Remove and stop using dryrun result
      CHECK(!output_tensor_map.empty());
      CHECK(output_tensor_map.count(node.name()) == 1);
      const TensorShape& tensor_shape =
          output_tensor_map.at(node.name())->shape();
      data_size = max_bytes_per_data * tensor_shape.num_elements();
    }
    CHECK(data_size >= 0);
    node_output_params.max_sizes.push_back(data_size);
  }
  node_output_params_list_.emplace_back(node_output_params);
}

void GraphTransferer::AppendNodeParamsWithIoParams(
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const Node& node, const string& name, const int id, const string& type,
    const int type_id, const string& padding_str, const int inputs_size,
    const std::vector<int>& extra_inputs, const int outputs_size,
    const bool append_input_params, const bool append_output_params) {
  VLOG(1) << "Append node with io params: " << node.name();
  if (append_input_params) {
    AppendNodeInputParams(id, node, extra_inputs);
  }
  if (append_output_params) {
    AppendNodeOutputParams(shape_refiner, output_tensor_map, id, node);
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

/* static */ std::array<int64, GraphTransferer::SHAPE_ARRAY_SIZE>
GraphTransferer::ToTensorShapeArray(const TensorShape& shape) {
  switch (shape.dims()) {
    case 0:
      return std::array<int64, SHAPE_ARRAY_SIZE>{{1, 1, 1, 1}};
    case 1:
      return std::array<int64, SHAPE_ARRAY_SIZE>{{1, 1, 1, shape.dim_size(0)}};
    case 2:
      return std::array<int64, SHAPE_ARRAY_SIZE>{
          {1, 1, shape.dim_size(0), shape.dim_size(1)}};
    case 3:
      return std::array<int64, SHAPE_ARRAY_SIZE>{
          {1, shape.dim_size(0), shape.dim_size(1), shape.dim_size(2)}};
    case 4:
      return std::array<int64, SHAPE_ARRAY_SIZE>{
          {shape.dim_size(0), shape.dim_size(1), shape.dim_size(2),
           shape.dim_size(3)}};
    default:
      // TODO(satok): Support more ranks?
      CHECK(false);
      return std::array<int64, SHAPE_ARRAY_SIZE>();
  }
}

/* static */ void GraphTransferer::CheckShape(
    const OutputTensorMap& output_tensor_map, const string& node_name,
    const std::array<int64, SHAPE_ARRAY_SIZE>& expected) {
  if (output_tensor_map.empty()) {
    // As output_tensor_map is empty, skip checking tensor shape.
    return;
  }
  VLOG(1) << "Check shape for " << node_name;
  CHECK(output_tensor_map.count(node_name) == 1);
  const std::array<int64, SHAPE_ARRAY_SIZE> actual =
      ToTensorShapeArray(output_tensor_map.at(node_name)->shape());
  for (int i = 0; i < SHAPE_ARRAY_SIZE; ++i) {
    CHECK(expected[i] == actual[i]);
  }
}

GraphTransferer::TransferParamsComparator::TransferParamsComparator(
    const std::unordered_map<int, std::unordered_set<int>>& dep_map)
    : dependency_map_(dep_map) {}

bool GraphTransferer::TransferParamsComparator::operator()(
    const GraphTransferer::NodeTransferParams& obj0,
    const GraphTransferer::NodeTransferParams& obj1) {
  const int node_id0 = obj0.node_id;
  const int node_id1 = obj1.node_id;
  bool obj0_uses_obj1 = false;
  if (dependency_map_.count(node_id0)) {
    obj0_uses_obj1 = dependency_map_.at(node_id0).count(node_id1) > 0;
  }
  bool obj1_uses_obj0 = false;
  if (dependency_map_.count(node_id1)) {
    obj1_uses_obj0 = dependency_map_.at(node_id1).count(node_id0) > 0;
  }
  CHECK(!obj0_uses_obj1 || !obj1_uses_obj0);
  if (obj0_uses_obj1) {
    return false;
  } else if (obj1_uses_obj0) {
    return true;
  }
  return node_id0 > node_id1;
}

/* static */ void GraphTransferer::FillDependencyRec(
    const int node_id,
    std::unordered_map<int, std::unordered_set<int>>& dep_map,
    std::unordered_set<int>& completed) {
  if (dep_map.count(node_id) == 0 || dep_map.at(node_id).empty() ||
      completed.count(node_id) == 1) {
    return;
  }
  CHECK(dep_map.count(node_id) == 1);

  // Complete children's dependency map
  for (int child_node_id : dep_map.at(node_id)) {
    CHECK(child_node_id != node_id);
    if (completed.count(child_node_id) != 0) {
      continue;
    }
    FillDependencyRec(child_node_id, dep_map, completed);
  }

  // Find additional depending ids
  std::vector<int> depending_ids;
  for (int child_node_id : dep_map.at(node_id)) {
    if (dep_map.count(child_node_id) == 0) {
      continue;
    }
    for (int depending_id : dep_map.at(child_node_id)) {
      depending_ids.emplace_back(depending_id);
    }
  }

  // Insert additional depending ids
  for (int depending_id : depending_ids) {
    if (dep_map.at(node_id).count(depending_id) == 0) {
      dep_map.at(node_id).emplace(depending_id);
    }
  }

  // DP: Record completed node id
  completed.emplace(node_id);
}

void GraphTransferer::ClearCache() {
  node_name_cache_list_.clear();
  node_name_to_id_cache_map_.clear();
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
    sstream << "---(CONST) [" << std::hex << params.node_id << std::dec << ","
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
            << params.node_id << std::dec << "," << params.soc_op_id << ","
            << params.padding << "," << params.inputs_name << ","
            << params.inputs_size << "," << params.outputs_name << ","
            << params.outputs_size << "," << params.type << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Op node count = " << node_transfer_params_list_.size();
  for (const NodeInputParams& params : node_input_params_list_) {
    std::stringstream sstream;
    sstream << "---(INPUT) [" << std::hex << params.node_id << std::dec;
    for (const std::tuple<int, int>& pair :
         params.input_node_id_and_output_port_list) {
      sstream << "," << std::hex << std::get<0>(pair) << std::dec << ","
              << std::get<1>(pair);
    }
    sstream << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Input params count = " << node_input_params_list_.size();
  for (const NodeOutputParams& params : node_output_params_list_) {
    std::stringstream sstream;
    sstream << "---(OUTPUT) [" << std::hex << params.node_id << std::dec;
    for (const int max_size : params.max_sizes) {
      sstream << "," << max_size;
    }
    sstream << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Output params count = " << node_input_params_list_.size();
}

}  // namespace tensorflow
