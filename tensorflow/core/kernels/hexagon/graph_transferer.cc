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
 * - LoadGraphFromProtoFile
 * These functions read a graph definition and store parameters
 * of node to transfer the graph to SOC.
 */
Status GraphTransferer::LoadGraphFromProto(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const GraphDef& graph_def,
    const std::vector<InputNodeInfo>& input_node_info_list,
    const std::vector<string>& output_node_names,
    const bool shape_inference_for_unknown_shape,
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

  if (shape_inference_for_unknown_shape && !input_node_info_list.empty()) {
    auto visit = [&shape_refiner, &input_node_info_list, &status](Node* node) {
      CHECK_NE(node, nullptr);
      // If we visit an input node, we use the shape provided and set the
      // shape accordingly.
      bool is_input_node = false;
      for (const InputNodeInfo& input_node_info : input_node_info_list) {
        if (node->name() == input_node_info.name) {
          shape_inference::InferenceContext* context =
              shape_refiner.GetContext(node);
          TensorShapeProto proto;
          input_node_info.tensor.shape().AsProto(&proto);
          shape_inference::ShapeHandle handle;
          context->MakeShapeFromShapeProto(proto, &handle);
          shape_refiner.SetShape(node, 0, handle);
          is_input_node = true;
        }
      }
      // If not an input node call AddNode() that recomputes the shape.
      if (!is_input_node) {
        status = shape_refiner.AddNode(node);
        if (!status.ok()) {
          VLOG(1) << "Shape inference failed for node: " << node->name();
        }
      }
    };

    // Runs a reverse DFS over the entire graph setting the shape for the input
    // nodes provided and then recomputing the shape of all the nodes downstream
    // from them. The "visit" function is executed for each node after all its
    // parents have been visited.
    ReverseDFS(graph, {}, visit);

    if (!status.ok()) {
      VLOG(1) << "Failed to run shape inference: " << status.ToString();
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

  for (const InputNodeInfo& input_node_info : input_node_info_list) {
    GraphTransferInfo::GraphInputNodeInfo& graph_input_node_info =
        *graph_transfer_info_.add_graph_input_node_info();
    graph_input_node_info.set_name(input_node_info.name);
    for (const int64 dim : ToTensorShapeArray(input_node_info.tensor.shape())) {
      graph_input_node_info.add_shape(dim);
    }
  }

  for (const string& output_node_name : output_node_names) {
    GraphTransferInfo::GraphOutputNodeInfo& graph_output_node_info =
        *graph_transfer_info_.add_graph_output_node_info();
    graph_output_node_info.set_name(output_node_name);
    // TODO(satok): Use shape inference to obtain output shapes
    if (!output_tensor_map.empty()) {
      CHECK_EQ(output_tensor_map.count(output_node_name), 1)
          << output_tensor_map.count(output_node_name);
      Tensor* output_tensor = output_tensor_map.at(output_node_name);
      CHECK(output_tensor != nullptr);
      for (const int64 dim : ToTensorShapeArray(output_tensor->shape())) {
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
    const std::vector<InputNodeInfo>& input_node_info_list,
    const std::vector<string>& output_node_names, const bool is_text_proto,
    const bool shape_inference_for_unknown_shape,
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
  return LoadGraphFromProto(
      ops_definitions, graph_def, input_node_info_list, output_node_names,
      shape_inference_for_unknown_shape, output_tensor_info->output_tensor_map);
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
  CHECK_EQ(output_node_names.size(), output_tensors.size())
      << output_node_names.size() << ", " << output_tensors.size();

  // Append output tensor of input node in advance to create a map
  // to avoid memory reallocation inside vector
  for (const InputNodeInfo& input_node_info : input_node_info_list) {
    output_tensors.push_back(input_node_info.tensor);
  }

  for (int i = 0; i < output_node_names.size(); ++i) {
    const string& name = output_node_names.at(i);
    CHECK_EQ(output_tensor_map.count(name), 0);
    output_tensor_map[name] = &output_tensors.at(i);
  }
  for (int i = 0; i < input_node_info_list.size(); ++i) {
    const string& name = input_node_info_list.at(i).name;
    CHECK_EQ(output_tensor_map.count(name), 0);
    output_tensor_map.emplace(name,
                              &output_tensors.at(output_node_names.size() + i));
  }
  CHECK(graph_def.node_size() == output_tensors.size());
  return status;
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
  GraphTransferInfo::ConstNodeInfo& const_node_info =
      *graph_transfer_info_.add_const_node_info();
  const_node_info.set_name(node.name());
  const_node_info.set_node_id(id);
  // TODO(satok): Make this generic. Never assume rank is 4.
  const_node_info.add_shape(shape[0]);
  const_node_info.add_shape(shape[1]);
  const_node_info.add_shape(shape[2]);
  const_node_info.add_shape(shape[3]);
  // TODO(satok): Remove. Determine constant value without dryrun
  if (data_size > 0) {
    if (output_tensor_map.empty()) {
      // setting dummy data if we don't generate node output
      std::vector<uint8> dummy_data(data_size);
      const_node_info.set_data(dummy_data.data(), data_size);
    } else {
      const Tensor* tensor = output_tensor_map.at(node.name());
      CHECK(tensor != nullptr);
      StringPiece sp = tensor->tensor_data();
      CHECK_EQ(data_size, sp.size());
      const_node_info.set_data(sp.data(), data_size);
    }
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
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const Node& node) {
  VLOG(1) << "Register input node: " << node.name();
  CHECK_EQ(node_name_to_id_cache_map_.count(node.name()), 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string op_type = IGraphTransferOpsDefinitions::INPUT_OP_NAME;
  const int op_type_id = ops_definitions.GetOpIdFor(op_type);
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());
  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, PADDING_NA_ID, node.num_inputs(), {},
      node.num_outputs(), true /* append_input */, true /* append_output */);
}

void GraphTransferer::RegisterOutputNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const Node& node) {
  VLOG(1) << "Register output node: " << node.name();
  CHECK_EQ(node_name_to_id_cache_map_.count(node.name()), 1);
  const int id = node_name_to_id_cache_map_[node.name()];
  const string op_type = IGraphTransferOpsDefinitions::OUTPUT_OP_NAME;
  const int op_type_id = ops_definitions.GetOpIdFor(op_type);
  CHECK(op_type_id >= 0 && op_type_id < ops_definitions.GetTotalOpsCount());
  // TODO(satok): Set output for output node?
  AppendNodeParamsWithIoParams(
      shape_refiner, output_tensor_map, node, node.name(), id,
      node.type_string(), op_type_id, PADDING_NA_ID, node.num_inputs(), {},
      0 /* outputs_size */, true /* append_input */, false /* append_output */);
}

void GraphTransferer::RegisterFlattenNode(
    const IGraphTransferOpsDefinitions& ops_definitions,
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
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
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
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
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
    const int id, const Node& node) {
  VLOG(1) << "Append output params: " << node.name() << ", "
          << node.num_outputs();
  GraphTransferInfo::NodeOutputInfo& node_output_info =
      *graph_transfer_info_.add_node_output_info();
  node_output_info.set_node_id(id);
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
        CHECK_EQ(output_tensor_map.count(node.name()), 1) << node.name();
        const TensorShape& tensor_shape =
            output_tensor_map.at(node.name())->shape();
        CHECK_EQ(num_output_elements, tensor_shape.num_elements())
            << "num elements of node " << node.name() << " doesn't match "
            << num_output_elements << " vs " << tensor_shape.num_elements()
            << ", " << node.type_string();
      }
    } else {
      // Use dryrun result to get the output data size
      // TODO(satok): Remove and stop using dryrun result
      CHECK(!output_tensor_map.empty());
      CHECK_EQ(output_tensor_map.count(node.name()), 1);
      const TensorShape& tensor_shape =
          output_tensor_map.at(node.name())->shape();
      data_size = max_bytes_per_data * tensor_shape.num_elements();
    }
    CHECK_GE(data_size, 0);
    node_output_info.add_max_byte_size(data_size);
  }
}

void GraphTransferer::AppendNodeParamsWithIoParams(
    const ShapeRefiner& shape_refiner, const OutputTensorMap& output_tensor_map,
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
    const OutputTensorMap& output_tensor_map, const string& node_name,
    const std::array<int64, SHAPE_ARRAY_SIZE>& expected) {
  if (output_tensor_map.empty()) {
    // As output_tensor_map is empty, skip checking tensor shape.
    return;
  }
  VLOG(1) << "Check shape for " << node_name;
  CHECK_EQ(output_tensor_map.count(node_name), 1);
  const std::array<int64, SHAPE_ARRAY_SIZE> actual =
      ToTensorShapeArray(output_tensor_map.at(node_name)->shape());
  for (int i = 0; i < SHAPE_ARRAY_SIZE; ++i) {
    CHECK_EQ(expected[i], actual[i]);
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
