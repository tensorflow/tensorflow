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
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"
#include "tensorflow/core/util/tensor_slice_writer.h"

namespace tensorflow {

constexpr bool DBG_DUMP_VERIFICATION_STRING = false;
constexpr bool DBG_DUMP_PARAMS = false;

const char RESHAPE_NODE_TYPE_STRING[] = "Reshape";
const char SOURCE_NODE_NAME[] = "_SOURCE";
const char SINK_NODE_NAME[] = "_SINK";
const char INPUTS_NODE_PREFIX[] = "inputs_for_";
const char OUTPUTS_NODE_PREFIX[] = "outputs_for_";
const char DATA_NODE_PREFIX[] = "data_for_op_";
const char CONST_SHAPE_PREFIX[] = "const_shape_";
const char PADDING_ATTR_NAME[] = "padding";
const char STRIDES_ATTR_NAME[] = "strides";
const char KSIZE_ATTR_NAME[] = "ksize";
const char NULL_OUTPUT_NAME[] = "NULL";
const int PADDING_NA_ID = 0;  // VALID = 1, SAME = 2

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
 * - LoadGraphFromProptoFile
 * These functions read a graph definition and store parameters
 * of node to transfer the graph to SOC.
 */
Status GraphTransferer::LoadGraphFromProto(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const GraphDef& graph_def,
    const std::vector<std::pair<string, Tensor>>& input_node_info_list,
    const std::vector<string>& output_node_names,
    const bool shape_inference_for_unknown_shape,
    const TensorShapeMap& output_tensor_map) {
  ImportGraphDefOptions opts;
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions().producer(), graph.op_registry());
  Status status = ImportGraphDef(opts, graph_def, &graph, &shape_refiner);
  if (!status.ok()) {
    return status;
  }

  if (shape_inference_for_unknown_shape) {
    status = RemoteFusedGraphExecuteUtils::PropagateShapeInference(
        graph_def, input_node_info_list, &graph, &shape_refiner);
    if (!status.ok()) {
      return status;
    }
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

  for (const std::pair<string, Tensor>& input_node_info :
       input_node_info_list) {
    GraphTransferInfo::GraphInputNodeInfo& graph_input_node_info =
        *graph_transfer_info_.add_graph_input_node_info();
    graph_input_node_info.set_name(input_node_info.first);
    graph_input_node_info.set_dtype(input_node_info.second.dtype());
    for (const int64 dim : ToTensorShapeArray(input_node_info.second.shape())) {
      graph_input_node_info.add_shape(dim);
    }
  }

  for (const string& output_node_name : output_node_names) {
    GraphTransferInfo::GraphOutputNodeInfo& graph_output_node_info =
        *graph_transfer_info_.add_graph_output_node_info();
    graph_output_node_info.set_name(output_node_name);
    if (!output_tensor_map.empty()) {
      const DataType* dt;
      const TensorShape* shape;
      CHECK(FindShapeType(output_tensor_map, output_node_name, &dt, &shape));
      graph_output_node_info.set_dtype(*dt);
      for (const int64 dim : ToTensorShapeArray(*shape)) {
        graph_output_node_info.add_shape(dim);
      }
    }
  }

  graph_transfer_info_.set_destination(
      ops_definitions.GetTransferDestination());

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
    const std::vector<std::pair<string, Tensor>>& input_node_info_list,
    const std::vector<string>& output_node_names, const bool is_text_proto,
    const bool shape_inference_for_unknown_shape,
    const bool dry_run_for_unknown_shape,
    RemoteFusedGraphExecuteUtils::TensorShapeMap* tensor_shape_map) {
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
    status = RemoteFusedGraphExecuteUtils::DryRunInferenceForAllNode(
        graph_def, input_node_info_list, true, tensor_shape_map);
    if (!status.ok()) {
      return status;
    }
  }
  VLOG(1) << "Load graph with output tensors";
  return LoadGraphFromProto(
      ops_definitions, graph_def, input_node_info_list, output_node_names,
      shape_inference_for_unknown_shape, *tensor_shape_map);
}

void GraphTransferer::SortParams(const std::vector<string>& output_node_names) {
  // TODO(satok): optimize complexity
  std::unordered_map<int, GraphTransferInfo::NodeInputInfo*> input_map;
  for (GraphTransferInfo::NodeInputInfo& input :
       *graph_transfer_info_.mutable_node_input_info()) {
    input_map.emplace(input.node_id(), &input);
  }

  // Setup dependency map placeholder
  std::vector<int> output_node_ids;
  std::unordered_map<int, std::unordered_set<int>> dependency_map;
  for (const GraphTransferInfo::NodeInfo& params :
       graph_transfer_info_.node_info()) {
    const int node_id = params.node_id();
    for (const string& output_node_name : output_node_names) {
      if (params.name() == output_node_name) {
        output_node_ids.emplace_back(node_id);
      }
    }

    dependency_map.emplace(std::piecewise_construct, std::make_tuple(node_id),
                           std::make_tuple());
    if (params.input_count() == 0) {
      continue;
    }
    CHECK_EQ(input_map.count(node_id), 1);
    for (const GraphTransferInfo::NodeInput& node_input :
         input_map.at(node_id)->node_input()) {
      dependency_map.at(node_id).emplace(node_input.node_id());
    }
  }

  // Create dependency map traversed from output nodes
  std::unordered_set<int> completed;
  for (int output_node_id : output_node_ids) {
    FillDependencyRec(output_node_id, dependency_map, completed);
  }

  std::sort(graph_transfer_info_.mutable_node_info()->begin(),
            graph_transfer_info_.mutable_node_info()->end(),
            TransferParamsComparator(dependency_map));
}

void GraphTransferer::EnableStrictCheckMode(const bool enable) {
  strict_check_mode_ = enable;
}

void GraphTransferer::SetSerializedGraphTransferInfo(
    const string& serialized_proto) {
  graph_transfer_info_.ParseFromString(serialized_proto);
}

const GraphTransferInfo& GraphTransferer::GetGraphTransferInfo() const {
  return graph_transfer_info_;
}

GraphTransferInfo& GraphTransferer::GetMutableGraphTransferInfo() {
  return graph_transfer_info_;
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

Status GraphTransferer::RegisterNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const TensorShapeMap& output_tensor_map,
    const Node& node,
    const std::vector<std::pair<string, Tensor>>& input_node_info_list,
    const std::vector<string>& output_node_names) {
  VLOG(1) << "Register node: " << node.name();
  if (node.name() == SOURCE_NODE_NAME || node.name() == SINK_NODE_NAME) {
    // Just ignore sink and source
    return Status();
  } else if (RemoteFusedGraphExecuteUtils::IsInputNode(input_node_info_list,
                                                       node.name())) {
    RegisterInputNode(ops_definitions, shape_refiner, output_tensor_map, node);
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
                                   " has not been implemented yet.");
  }

  return Status();
}

void GraphTransferer::RegisterConstantNode(
    const ShapeRefiner& shape_refiner, const Node& node,
    const TensorShapeMap& output_tensor_map) {
  VLOG(1) << "Register constant node: " << node.name();
  CHECK_EQ(node_name_to_id_cache_map_.count(node.name()), 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const int output_node_size = node.num_outputs();
  CHECK_EQ(output_node_size, 1);
  // TODO(satok): support multiple outputs?
  const int output_index = 0;
  const DataType dt = node.output_type(output_index);
  const size_t max_bytes_per_data = DataTypeSize(dt);
  CHECK_GT(max_bytes_per_data, 0)
      << "dt = " << dt << ", " + DataTypeString(dt) << ", "
      << max_bytes_per_data << ", " << static_cast<int>(DataTypeSize(dt))
      << ",,,,,,,";
  shape_inference::InferenceContext* context = shape_refiner.GetContext(&node);
  shape_inference::ShapeHandle shape_handle = context->output(output_index);
  const shape_inference::DimensionHandle num_elements_dim =
      context->NumElements(shape_handle);
  std::array<int64, SHAPE_ARRAY_SIZE> shape_array;
  int data_size;
  if (context->ValueKnown(num_elements_dim)) {
    const int64 num_output_elements = context->Value(num_elements_dim);
    data_size = max_bytes_per_data * num_output_elements;
    shape_array = BuildShapeArray(shape_handle, context);
    CheckShape(output_tensor_map, node.name(), shape_array);
  } else {
    // Use output tensor for unknown shape
    const TensorShape* shape;
    CHECK(FindShapeType(output_tensor_map, node.name(), nullptr, &shape));
    shape_array = ToTensorShapeArray(*shape);
    data_size = max_bytes_per_data * shape->num_elements();
  }
  CHECK(context->ValueKnown(num_elements_dim));
  GraphTransferInfo::ConstNodeInfo& const_node_info =
      *graph_transfer_info_.add_const_node_info();
  const_node_info.set_name(node.name());
  const_node_info.set_node_id(id);
  // TODO(satok): Make this generic. Never assume rank is 4.
  CHECK_EQ(4, SHAPE_ARRAY_SIZE);
  const_node_info.add_shape(shape_array[0]);
  const_node_info.add_shape(shape_array[1]);
  const_node_info.add_shape(shape_array[2]);
  const_node_info.add_shape(shape_array[3]);
  const TensorProto* proto = nullptr;
  // TODO(b/32704451): Don't just ignore this status!
  GetNodeAttr(node.def(), "value", &proto).IgnoreError();
  Tensor const_tensor;
  // TODO(b/32704451): Don't just ignore this status!
  MakeTensorFromProto(*proto, &const_tensor).IgnoreError();

  const_node_info.set_dtype(const_tensor.dtype());
  // TODO(satok): Remove. Determine constant value without dryrun
  if (data_size > 0) {
    const_node_info.set_data(const_tensor.tensor_data().data(), data_size);
  }
}

int GraphTransferer::RegisterConstantShape(const std::vector<int>& shape) {
  VLOG(1) << "Cache constant shape.";
  // TODO(satok): Handle non-4dim strides
  CHECK_EQ(shape.size(), 4);
  const string shape_name = CONST_SHAPE_PREFIX + ToString(shape.at(0)) + 'x' +
                            ToString(shape.at(1)) + 'x' +
                            ToString(shape.at(2)) + 'x' + ToString(shape.at(3));
  if (node_name_to_id_cache_map_.count(shape_name) <= 0) {
    node_name_cache_list_.emplace_back(nullptr);
    const int id = node_name_cache_list_.size() - 1;
    node_name_to_id_cache_map_.emplace(shape_name, id);
    GraphTransferInfo::ConstNodeInfo& const_node_info =
        *graph_transfer_info_.add_const_node_info();
    const_node_info.set_name(shape_name);
    const_node_info.set_node_id(id);
    // TODO(satok): Make this generic. Never assume rank is 5.
    const_node_info.add_shape(static_cast<int64>(shape[0]));
    const_node_info.add_shape(static_cast<int64>(shape[1]));
    const_node_info.add_shape(static_cast<int64>(shape[2]));
    const_node_info.add_shape(static_cast<int64>(shape[3]));
  }
  return node_name_to_id_cache_map_[shape_name];
}

bool GraphTransferer::HasPaddingAndStrides(const Node& node) {
  return node.def().attr().count(PADDING_ATTR_NAME) > 0 &&
         node.def().attr().count(STRIDES_ATTR_NAME) > 0;
}

bool GraphTransferer::IsNodeFlattenReshape(
    const Node& node, const TensorShapeMap& output_tensor_map,
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
  std::array<int64, SHAPE_ARRAY_SIZE> shape_array;
  const shape_inference::DimensionHandle dim_handle =
      context->NumElements(shape_handle);

  // Obtain shape of output of node
  if (context->ValueKnown(dim_handle)) {
    shape_array = BuildShapeArray(shape_handle, context);
  } else {
    // Use output tensor for unknown shape
    const TensorShape* shape;
    CHECK(FindShapeType(output_tensor_map, node.name(), nullptr, &shape));
    shape_array = ToTensorShapeArray(*shape);
  }

  // check if reshape op just does flatten
  if (shape_array[0] == 1 && shape_array[1] == 1 && shape_array[2] == 1) {
    return true;
  } else {
    return false;
  }
}

void GraphTransferer::RegisterNodeWithPaddingAndStrides(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const TensorShapeMap& output_tensor_map,
    const Node& node) {
  CHECK_EQ(node_name_to_id_cache_map_.count(node.name()), 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  shape_inference::InferenceContext* context = shape_refiner.GetContext(&node);
  CHECK_GT(node.def().attr().count(PADDING_ATTR_NAME), 0);
  // TODO(satok): Use context->GetAttr(...) instead?
  Padding padding;
  TF_CHECK_OK(context->GetAttr(PADDING_ATTR_NAME, &padding));
  CHECK_GT(node.def().attr().count(STRIDES_ATTR_NAME), 0);
  std::vector<int32> strides;
  TF_CHECK_OK(context->GetAttr(STRIDES_ATTR_NAME, &strides));
  const int stride_id = RegisterConstantShape(strides);
  std::vector<int> extra_inputs{stride_id};
  if (node.def().attr().count(KSIZE_ATTR_NAME) > 0) {
    std::vector<int32> kernel_sizes;
    TF_CHECK_OK(context->GetAttr(KSIZE_ATTR_NAME, &kernel_sizes));
    const int ksize_id = RegisterConstantShape(kernel_sizes);
    extra_inputs.insert(extra_inputs.begin(), ksize_id);
  }
  const int op_type_id = ops_definitions.GetOpIdFor(node.type_string());
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount())
      << "Op " << node.type_string() << " not found in map(id = " << op_type_id
      << ")";
  // Safety check of padding id
  CHECK(padding == Padding::VALID ? 1 : 2);
  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, static_cast<int>(padding),
      node.num_inputs(), extra_inputs, node.num_outputs(),
      true /* append_input */, true /* append_output */);
}

void GraphTransferer::RegisterInputNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const TensorShapeMap& output_tensor_map,
    const Node& node) {
  VLOG(1) << "Register input node: " << node.name();
  CHECK_EQ(node_name_to_id_cache_map_.count(node.name()), 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string op_type = node.type_string();
  const int op_type_id = ops_definitions.GetOpIdFor(op_type);
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount())
      << "Op" << node.name() << ", " << op_type << " is not supported,"
      << op_type_id;
  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, PADDING_NA_ID, node.num_inputs(), {},
      node.num_outputs(), true /* append_input */, true /* append_output */);
}

void GraphTransferer::RegisterFlattenNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const TensorShapeMap& output_tensor_map,
    const Node& node) {
  VLOG(1) << "Register flatten node: " << node.name();
  CHECK_EQ(node_name_to_id_cache_map_.count(node.name()), 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string op_type = IGraphTransferOpsDefinitions::FLATTEN_OP_NAME;
  const int op_type_id = ops_definitions.GetOpIdFor(op_type);
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());

  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, PADDING_NA_ID, node.num_inputs(), {},
      node.num_outputs(), true /* append_input */, true /* append_output */);
}

void GraphTransferer::RegisterGenericNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const TensorShapeMap& output_tensor_map,
    const Node& node) {
  VLOG(1) << "Register generic node: " << node.name();
  CHECK_EQ(node_name_to_id_cache_map_.count(node.name()), 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const int op_type_id = ops_definitions.GetOpIdFor(node.type_string());
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());

  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, PADDING_NA_ID, node.num_inputs(), {},
      node.num_outputs(), true /* append_input */, true /* append_output */);
}

// TODO(satok): Remove this function.
// TODO(satok): Remove only_register_const_node.
Status GraphTransferer::RegisterNodeIfAllInputsAreCached(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const Node& node,
    const bool only_register_const_node,
    const std::vector<std::pair<string, Tensor>>& input_node_info_list,
    const std::vector<string>& output_node_names,
    const TensorShapeMap& output_tensor_map) {
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
                                       const int padding, const int inputs_size,
                                       const std::vector<int>& extra_inputs,
                                       const int outputs_size) {
  VLOG(1) << "Append node params: " << name;
  GraphTransferInfo::NodeInfo& node_info =
      *graph_transfer_info_.add_node_info();
  node_info.set_name(name);
  node_info.set_node_id(id);
  node_info.set_type_name(type);
  node_info.set_soc_op_id(type_id);
  node_info.set_padding_id(padding);
  node_info.set_input_count(inputs_size +
                            static_cast<int>(extra_inputs.size()));
  node_info.set_output_count(static_cast<int>(outputs_size));
}

void GraphTransferer::AppendNodeInputParams(
    const int id, const Node& node, const std::vector<int>& extra_inputs) {
  VLOG(1) << "Append input params: " << node.name() << ", " << node.num_inputs()
          << ", " << extra_inputs.size();
  GraphTransferInfo::NodeInputInfo& node_input_info =
      *graph_transfer_info_.add_node_input_info();
  node_input_info.set_node_id(id);
  for (int i = 0; i < node.num_inputs(); ++i) {
    const Edge* edge = nullptr;
    TF_CHECK_OK(node.input_edge(i, &edge));
    const Node* input_node = edge->src();
    const int port = edge->src_output();

    const std::string& op_name = input_node->name();
    CHECK_GT(node_name_to_id_cache_map_.count(op_name), 0) << op_name;
    const int src_id = node_name_to_id_cache_map_[op_name];
    GraphTransferInfo::NodeInput& node_input =
        *node_input_info.add_node_input();
    node_input.set_node_id(src_id);
    node_input.set_output_port(port);
  }
  for (const int extra_input : extra_inputs) {
    GraphTransferInfo::NodeInput& node_input =
        *node_input_info.add_node_input();
    node_input.set_node_id(extra_input);
    node_input.set_output_port(0);
  }
}

void GraphTransferer::AppendNodeOutputParams(
    const ShapeRefiner& shape_refiner, const TensorShapeMap& output_tensor_map,
    const int id, const Node& node) {
  VLOG(1) << "Append output params: " << node.name() << ", "
          << node.num_outputs();
  GraphTransferInfo::NodeOutputInfo& node_output_info =
      *graph_transfer_info_.add_node_output_info();
  node_output_info.set_node_id(id);
  for (int i = 0; i < node.num_outputs(); ++i) {
    int data_size = -1;
    const int output_index = i;
    const DataType dt = node.output_type(output_index);
    const size_t max_bytes_per_data = DataTypeSize(dt);

    shape_inference::InferenceContext* context =
        shape_refiner.GetContext(&node);
    shape_inference::ShapeHandle shape_handle = context->output(output_index);
    const shape_inference::DimensionHandle num_elements_dim =
        context->NumElements(shape_handle);
    if (context->ValueKnown(num_elements_dim)) {
      const int64 num_output_elements = context->Value(num_elements_dim);
      data_size = max_bytes_per_data * num_output_elements;
      if (!output_tensor_map.empty() && strict_check_mode_) {
        const TensorShape* shape;
        CHECK(FindShapeType(output_tensor_map, node.name(), nullptr, &shape));
        CHECK_EQ(num_output_elements, shape->num_elements())
            << "num elements of node " << node.name() << " doesn't match "
            << num_output_elements << " vs " << shape->num_elements() << ", "
            << node.type_string();
      }
    } else {
      // Use TensorShapeMap for unknown shapes
      const TensorShape* shape;
      CHECK(FindShapeType(output_tensor_map, node.name(), nullptr, &shape));
      data_size = max_bytes_per_data * shape->num_elements();
    }
    CHECK_GE(data_size, 0);
    node_output_info.add_max_byte_size(data_size);
  }
}

void GraphTransferer::AppendNodeParamsWithIoParams(
    const ShapeRefiner& shape_refiner, const TensorShapeMap& output_tensor_map,
    const Node& node, const string& name, const int id, const string& type,
    const int type_id, const int padding, const int inputs_size,
    const std::vector<int>& extra_inputs, const int outputs_size,
    const bool append_input_params, const bool append_output_params) {
  VLOG(1) << "Append node with io params: " << node.name();
  if (append_input_params) {
    AppendNodeInputParams(id, node, extra_inputs);
  }
  if (append_output_params) {
    AppendNodeOutputParams(shape_refiner, output_tensor_map, id, node);
  }
  AppendNodeParams(name, id, type, type_id, padding, inputs_size, extra_inputs,
                   outputs_size);
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

/* static */ string GraphTransferer::ToPaddingDebugString(const int padding) {
  switch (padding) {
    case 0:
      return "NN_PAD_NA";
    case Padding::VALID:
      return "NN_PAD_VALID";
    case Padding::SAME:
      return "NN_PAD_SAME";
    default:
      CHECK(false);
      return "";
  }
}

/* static */ void GraphTransferer::CheckShape(
    const TensorShapeMap& output_tensor_map, const string& node_name,
    const std::array<int64, SHAPE_ARRAY_SIZE>& expected) {
  if (output_tensor_map.empty()) {
    // As output_tensor_map is empty, skip checking tensor shape.
    return;
  }
  const TensorShape* shape;
  CHECK(FindShapeType(output_tensor_map, node_name, nullptr, &shape));
  VLOG(1) << "Check shape for " << node_name;
  const std::array<int64, SHAPE_ARRAY_SIZE> actual = ToTensorShapeArray(*shape);
  for (int i = 0; i < SHAPE_ARRAY_SIZE; ++i) {
    CHECK_EQ(expected[i], actual[i]) << node_name;
  }
}

GraphTransferer::TransferParamsComparator::TransferParamsComparator(
    const std::unordered_map<int, std::unordered_set<int>>& dep_map)
    : dependency_map_(dep_map) {}

bool GraphTransferer::TransferParamsComparator::operator()(
    const GraphTransferInfo::NodeInfo& obj0,
    const GraphTransferInfo::NodeInfo& obj1) {
  const int node_id0 = obj0.node_id();
  const int node_id1 = obj1.node_id();
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
  CHECK_EQ(dep_map.count(node_id), 1);

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

/* static */ Status GraphTransferer::MakeTensorFromProto(
    const TensorProto& tensor_proto, Tensor* tensor) {
  if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
    Tensor parsed(tensor_proto.dtype());
    if (parsed.FromProto(cpu_allocator(), tensor_proto)) {
      *tensor = parsed;
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Cannot parse tensor from proto: ",
                                 tensor_proto.DebugString());
}

/* static */ bool GraphTransferer::FindShapeType(
    const TensorShapeMap& tensor_shape_map, const string& name, const int port,
    const DataType** dt, const TensorShape** shape) {
  const std::pair<DataType, TensorShape>* tensor_shape_type =
      RemoteFusedGraphExecuteUtils::GetTensorShapeType(tensor_shape_map, name,
                                                       port);
  if (tensor_shape_type == nullptr) {
    return false;
  }
  if (dt != nullptr) {
    *dt = &tensor_shape_type->first;
  }
  if (shape != nullptr) {
    *shape = &tensor_shape_type->second;
  }
  return true;
}

/* static */ bool GraphTransferer::FindShapeType(
    const TensorShapeMap& tensor_shape_map, const string& name,
    const DataType** dt, const TensorShape** shape) {
  const TensorId tid = ParseTensorName(name);
  return FindShapeType(tensor_shape_map, tid.first.ToString(), tid.second, dt,
                       shape);
}

void GraphTransferer::ClearCache() {
  node_name_cache_list_.clear();
  node_name_to_id_cache_map_.clear();
}

void GraphTransferer::DumpNodeTransferParams() const {
  LOG(INFO) << "*** Const Nodes ***";
  for (const GraphTransferInfo::ConstNodeInfo& params :
       graph_transfer_info_.const_node_info()) {
    // TODO(satok): Stop assuming shape size is 4.
    CHECK_EQ(params.shape_size(), 4);
    LOG(INFO) << "[ " << params.node_id() << " \"" << params.name()
              << "\" (Const)";
    LOG(INFO) << "  shape: " << params.shape(0) << params.shape(1)
              << params.shape(2) << params.shape(3);
    LOG(INFO) << "  data_name: "
              << (params.data().length() <= 0
                      ? ""
                      : DATA_NODE_PREFIX + ToString(params.node_id()));
    LOG(INFO) << "  data_size: " << params.data().length() << " bytes"
              << " ]";
  }
  LOG(INFO) << "******\n";
  LOG(INFO) << "*** Op Nodes ***";
  for (const GraphTransferInfo::NodeInfo& params :
       graph_transfer_info_.node_info()) {
    LOG(INFO) << "[ " << params.node_id() << " \"" << params.name();
    LOG(INFO) << "  type: " << params.type_name();
    LOG(INFO) << "  padding: " << ToPaddingDebugString(params.padding_id());
    LOG(INFO) << "  inputs: " << INPUTS_NODE_PREFIX + ToString(params.node_id())
              << ", size = " << params.input_count();
    LOG(INFO) << "  outputs: "
              << (params.output_count() <= 0
                      ? NULL_OUTPUT_NAME
                      : (OUTPUTS_NODE_PREFIX + ToString(params.node_id())))
              << ", size = " << params.output_count() << " ]";
  }
  LOG(INFO) << "******\n";
  LOG(INFO) << "*** Node input params ***";
  for (const GraphTransferInfo::NodeInputInfo& params :
       graph_transfer_info_.node_input_info()) {
    LOG(INFO) << "[ " << params.node_id() << " ]";
    for (const GraphTransferInfo::NodeInput& node_input : params.node_input()) {
      LOG(INFO) << "    src node id = " << node_input.node_id()
                << ", output port = " << node_input.output_port();
    }
  }
  LOG(INFO) << "******\n";
  LOG(INFO) << "*** Node output params ***";
  for (const GraphTransferInfo::NodeOutputInfo& params :
       graph_transfer_info_.node_output_info()) {
    LOG(INFO) << "[ " << params.node_id() << " ]";
    for (const int max_size : params.max_byte_size()) {
      LOG(INFO) << "    max_size = " << max_size;
    }
  }
  LOG(INFO) << "******\n";
}

void GraphTransferer::DumpVerificationStringOfNodeTransferParams() const {
  for (const GraphTransferInfo::ConstNodeInfo& params :
       graph_transfer_info_.const_node_info()) {
    std::stringstream sstream;
    // TODO(satok): Stop assuming shape size is 4.
    CHECK_EQ(params.shape_size(), 4);
    sstream << "---(CONST) [" << std::hex << params.node_id() << std::dec << ","
            << params.shape(0) << "," << params.shape(1) << ","
            << params.shape(2) << "," << params.shape(3) << ","
            << (params.data().length() <= 0
                    ? ""
                    : DATA_NODE_PREFIX + ToString(params.node_id()))
            << "," << params.data().length() << "," << params.name() << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Const node count = "
            << graph_transfer_info_.const_node_info_size();
  for (const GraphTransferInfo::NodeInfo& params :
       graph_transfer_info_.node_info()) {
    std::stringstream sstream;
    sstream << "---(OP) [" << params.name().c_str() << "," << std::hex
            << params.node_id() << std::dec << "," << params.soc_op_id() << ","
            << ToPaddingDebugString(params.padding_id()) << ","
            << INPUTS_NODE_PREFIX + ToString(params.node_id()) << ","
            << params.input_count() << ","
            << (params.output_count() <= 0
                    ? NULL_OUTPUT_NAME
                    : (OUTPUTS_NODE_PREFIX + ToString(params.node_id())))
            << "," << params.output_count() << "," << params.type_name() << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Op node count = " << graph_transfer_info_.node_info_size();
  for (const GraphTransferInfo::NodeInputInfo& params :
       graph_transfer_info_.node_input_info()) {
    std::stringstream sstream;
    sstream << "---(INPUT) [" << std::hex << params.node_id() << std::dec;
    for (const GraphTransferInfo::NodeInput& node_input : params.node_input()) {
      sstream << "," << std::hex << node_input.node_id() << std::dec << ","
              << node_input.output_port();
    }
    sstream << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Input params count = "
            << graph_transfer_info_.node_input_info_size();
  for (const GraphTransferInfo::NodeOutputInfo& params :
       graph_transfer_info_.node_output_info()) {
    std::stringstream sstream;
    sstream << "---(OUTPUT) [" << std::hex << params.node_id() << std::dec;
    for (const int max_size : params.max_byte_size()) {
      sstream << "," << max_size;
    }
    sstream << "]";
    LOG(INFO) << sstream.str();
  }
  LOG(INFO) << "Output params count = "
            << graph_transfer_info_.node_output_info_size();
}

}  // namespace tensorflow
