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

#include <algorithm>
#include <queue>
#include <utility>

#include "tensorflow/core/common_runtime/shape_refiner.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {
namespace {
const Node* FindNodeByName(const string& name, const Graph& graph) {
  for (const Node* node : graph.nodes()) {
    CHECK_NOTNULL(node);
    if (node->name() == name) {
      return node;
    }
  }
  return nullptr;
}

std::unordered_set<string> BuildNodeSetFromNodeNamesAndPorts(
    const std::vector<string>& node_names_and_ports) {
  std::unordered_set<string> retval;
  for (const string& node_name_and_port : node_names_and_ports) {
    const TensorId tid = ParseTensorName(node_name_and_port);
    retval.emplace(tid.first.ToString());
  }
  return retval;
}

Node* FindMutableNodeByName(const string& name, Graph* graph) {
  for (Node* node : graph->nodes()) {
    if (node != nullptr && node->name() == name) {
      return node;
    }
  }
  return nullptr;
}

const NodeDef* FindNodeDefByName(const string& input,
                                 const GraphDef& graph_def) {
  const TensorId tid = ParseTensorName(input);
  const string name = tid.first.ToString();
  for (const NodeDef& node_def : graph_def.node()) {
    if (node_def.name() == name) {
      return &node_def;
    }
  }
  return nullptr;
}

bool IsSameNodeName(const NodeDef& node_def, const string& node_name_and_port,
                    TensorId* tid) {
  CHECK_NOTNULL(tid);
  *tid = ParseTensorName(node_name_and_port);
  if (node_def.name() == tid->first.ToString()) {
    return true;
  }
  return false;
}

bool ContainsSameTensorId(const string& tensor_name,
                          const std::vector<string>& tensor_names) {
  const TensorId tid0 = ParseTensorName(tensor_name);
  for (const string& name : tensor_names) {
    const TensorId tid1 = ParseTensorName(name);
    if (tid0.first == tid1.first && tid0.second == tid1.second) {
      return true;
    }
  }
  return false;
}

void AppendDeliminator(string* str) {
  CHECK_NOTNULL(str);
  if (!str->empty()) {
    *str += ":";
  }
}

void ConvertMapToVector(const std::unordered_map<int, string>& in,
                        std::vector<string>* out) {
  CHECK_NOTNULL(out);
  out->resize(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    CHECK(in.count(i) > 0);
    out->at(i) = in.at(i);
  }
}

string DumpGraphDef(const GraphDef& graph_def) {
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

string DumpCluster(const RemoteFusedGraphExecuteUtils::ClusterInfo& cluster) {
  string out;
  out += "Nodes:\n";
  for (const string& str : std::get<0>(cluster)) {
    out += str + ", ";
  }
  out += "\nInput border:\n";
  for (const string& str : std::get<1>(cluster)) {
    out += str + ", ";
  }
  out += "\nOutput border:\n";
  for (const string& str : std::get<2>(cluster)) {
    out += str + ", ";
  }
  return out;
}

}  // namespace

/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_DATA_TYPES;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_SHAPES;
/* static */ constexpr const char* const RemoteFusedGraphExecuteUtils::
    ATTR_SERIALIZED_REMOTE_FUSED_GRAPH_EXECUTE_INFO;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::ATTR_NODE_TYPE;
/* static */ constexpr const char* const RemoteFusedGraphExecuteUtils::
    TRANSFORM_ARG_REMOTE_FUSED_GRAPH_EXECUTOR_NAME;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_REMOTE_FUSED_GRAPH_NODE_NAME;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_FUSED_NODES;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_BORDER_INPUTS;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_BORDER_OUTPUTS;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_FUSED_OP_TYPES;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_FUSE_BY_EXECUTOR;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_INPUT_TYPES;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::TRANSFORM_ARG_INPUT_SHAPES;

RemoteFusedGraphExecuteUtils::ExecutorBuildRegistrar::ExecutorBuildRegistrar(
    const string& name, ExecutorBuildFunc executor_build_func) {
  ExecutorBuildRegistry& executor_build_registry = *GetExecutorBuildRegistry();
  executor_build_registry[name] = std::move(executor_build_func);
}

/* static */ const RemoteFusedGraphExecuteUtils::ExecutorBuildFunc*
RemoteFusedGraphExecuteUtils::GetExecutorBuildFunc(const string& name) {
  ExecutorBuildRegistry& executor_build_registry = *GetExecutorBuildRegistry();
  if (executor_build_registry.count(name) <= 0) {
    return nullptr;
  }
  return &executor_build_registry.at(name);
}

/* static */ RemoteFusedGraphExecuteUtils::ExecutorBuildRegistry*
RemoteFusedGraphExecuteUtils::GetExecutorBuildRegistry() {
  static ExecutorBuildRegistry executor_builder_registry;
  return &executor_builder_registry;
}

/**
 * - DryRunInference
 * To determine shapes of output tensors of all nodes, dryrun the graph.
 * This function supplies memory allocation information when loading
 * the graph. This function is used to verify shape inference and actual
 * output shape.
 */
/* static */ Status RemoteFusedGraphExecuteUtils::DryRunInference(
    const GraphDef& graph_def,
    const std::vector<std::pair<string, Tensor>>& input_node_info_list,
    const std::vector<string>& output_node_names, const bool initialize_by_zero,
    std::vector<tensorflow::Tensor>* output_tensors) {
  // Create input tensor vector.  If "initialize_by_zero" is true,
  // input tensor fields are initialized by 0.
  std::vector<std::pair<string, tensorflow::Tensor>> input_tensors;
  for (const std::pair<string, Tensor>& input : input_node_info_list) {
    CHECK(input.second.IsInitialized());
    if (!initialize_by_zero) {
      input_tensors.push_back({input.first, input.second});
      continue;
    }
    // If input tensor is not initialized, initialize by 0-filling
    const DataType data_type = input.second.dtype();
    const TensorShape& shape = input.second.shape();
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
    input_tensors.push_back({input.first, input_tensor});
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

/* static */ Status RemoteFusedGraphExecuteUtils::DryRunInferenceForAllNode(
    const GraphDef& graph_def,
    const std::vector<std::pair<string, Tensor>>& input_node_info_list,
    const bool initialize_by_zero,
    RemoteFusedGraphExecuteUtils::TensorShapeMap* tensor_shape_map) {
  CHECK(tensor_shape_map != nullptr);
  std::vector<Tensor> output_tensors;
  output_tensors.reserve(graph_def.node_size());
  std::vector<string> output_node_names;

  Graph graph(OpRegistry::Global());
  Status status = ImportGraphDef({}, graph_def, &graph, nullptr);
  if (!status.ok()) {
    return status;
  }

  for (const Node* node : graph.nodes()) {
    if (IsInputNode(input_node_info_list, node->name())) {
      continue;
    }
    for (int i = 0; i < node->num_outputs(); ++i) {
      output_node_names.emplace_back(strings::StrCat(node->name(), ":", i));
    }
  }

  status = DryRunInference(graph_def, input_node_info_list, output_node_names,
                           initialize_by_zero, &output_tensors);
  if (!status.ok()) {
    VLOG(1) << "Failed to dryrun " << status;
    return status;
  }

  CHECK_EQ(output_node_names.size(), output_tensors.size())
      << output_node_names.size() << ", " << output_tensors.size();

  // Append output tensor of input node in advance to create a map
  // to avoid memory reallocation inside vector
  for (const std::pair<string, Tensor>& input_node_info :
       input_node_info_list) {
    output_tensors.push_back(input_node_info.second);
  }

  for (int i = 0; static_cast<size_t>(i) < output_node_names.size(); ++i) {
    const string& name = output_node_names.at(i);
    const Tensor& tensor = output_tensors.at(i);
    EmplaceTensorShapeType(name, tensor, tensor_shape_map);
  }
  for (int i = 0; static_cast<size_t>(i) < input_node_info_list.size(); ++i) {
    const string& name = input_node_info_list.at(i).first;
    const Tensor& tensor = output_tensors.at(output_node_names.size() + i);
    EmplaceTensorShapeType(name, tensor, tensor_shape_map);
  }
  CHECK_EQ(output_node_names.size() + input_node_info_list.size(),
           output_tensors.size());
  return status;
}

/* static */ bool RemoteFusedGraphExecuteUtils::IsInputNode(
    const std::vector<std::pair<string, Tensor>>& input_tensor_vector,
    const string& node_name) {
  for (const std::pair<string, Tensor>& pair : input_tensor_vector) {
    const TensorId tid = ParseTensorName(pair.first);
    if (node_name == tid.first.ToString()) {
      return true;
    }
  }
  return false;
}

/* static */ void RemoteFusedGraphExecuteUtils::ConvertToTensorShapeMap(
    const std::vector<std::pair<string, Tensor>>& input_node_info_list,
    const std::vector<string>& output_node_names,
    const std::vector<tensorflow::Tensor>& output_tensors,
    TensorShapeMap* tensor_shape_map) {
  CHECK_NE(tensor_shape_map, nullptr);
  tensor_shape_map->clear();
  tensor_shape_map->reserve(input_node_info_list.size() +
                            output_node_names.size());
  const int output_node_count = output_node_names.size();
  CHECK_EQ(output_node_count, output_tensors.size());
  for (int i = 0; i < output_node_count; ++i) {
    const string& node_name = output_node_names.at(i);
    const Tensor& tensor = output_tensors.at(i);
    EmplaceTensorShapeType(node_name, tensor, tensor_shape_map);
  }
}

/* static */ Status RemoteFusedGraphExecuteUtils::MakeTensorFromProto(
    const TensorProto& tensor_proto, Tensor* tensor) {
  if (tensor_proto.dtype() > 0 && tensor_proto.dtype() <= DataType_MAX) {
    Tensor parsed(tensor_proto.dtype());
    if (parsed.FromProto(cpu_allocator(), tensor_proto)) {
      *tensor = parsed;
      return Status::OK();
    }
  }
  return errors::InvalidArgument("Cannot parse tensor from proto");
}

/* static */ bool RemoteFusedGraphExecuteUtils::AddOutputTensorShapeType(
    const std::vector<DataType>& data_types,
    const std::vector<TensorShape>& shapes, NodeDef* node_def) {
  AddNodeAttr(ATTR_OUTPUT_DATA_TYPES, data_types, node_def);
  AddNodeAttr(ATTR_OUTPUT_SHAPES, shapes, node_def);
  return true;
}

/* static */ Status
RemoteFusedGraphExecuteUtils::AddOutputTensorShapeTypeByTensorShapeMap(
    const TensorShapeMap& tensor_shape_map, NodeDef* node_def) {
  CHECK_NE(node_def, nullptr);
  std::priority_queue<std::tuple<int, const TensorShapeType*>> queue;
  auto its = tensor_shape_map.equal_range(node_def->name());
  for (auto it = its.first; it != its.second; ++it) {
    queue.emplace(std::make_tuple(it->second.first, &it->second.second));
  }
  int last_port = queue.size();
  std::vector<DataType> data_types;
  std::vector<TensorShape> shapes;
  while (!queue.empty()) {
    const int port = std::get<0>(queue.top());
    const TensorShapeType* tst = std::get<1>(queue.top());
    CHECK_NE(tst, nullptr);
    data_types.emplace(data_types.begin(), tst->first);
    shapes.emplace(shapes.begin(), tst->second);
    CHECK_EQ(last_port - 1, port);
    last_port = port;
    queue.pop();
  }
  AddOutputTensorShapeType(data_types, shapes, node_def);
  return Status::OK();
}

/* static */ Status RemoteFusedGraphExecuteUtils::GetOutputTensorShapeType(
    AttrSlice attrs, std::vector<DataType>* data_types,
    std::vector<TensorShape>* shapes) {
  Status status;
  if (data_types != nullptr) {
    status = GetNodeAttr(attrs, ATTR_OUTPUT_DATA_TYPES, data_types);
  }
  if (!status.ok()) {
    return status;
  }
  if (shapes != nullptr) {
    status = GetNodeAttr(attrs, ATTR_OUTPUT_SHAPES, shapes);
    if (status.ok() && data_types != nullptr) {
      CHECK_EQ(data_types->size(), shapes->size());
    }
  }

  return status;
}

/* static */ bool RemoteFusedGraphExecuteUtils::GetOutputTensorShapeType(
    const GraphDef& graph_def, const string& name_and_port, DataType* data_type,
    TensorShape* shape) {
  std::vector<DataType> data_types;
  std::vector<TensorShape> shapes;
  const TensorId tid = ParseTensorName(name_and_port);
  const string node_name = tid.first.ToString();
  const int port = tid.second;
  const NodeDef* node_def = FindNodeDefByName(node_name, graph_def);
  CHECK_NOTNULL(node_def);
  GetOutputTensorShapeType(*node_def, &data_types, &shapes).IgnoreError();
  if (data_types.empty()) {
    return false;
  }
  CHECK(data_types.size() > port);
  *data_type = data_types.at(port);
  *shape = shapes.at(port);
  return true;
}

/* static */ Status RemoteFusedGraphExecuteUtils::PropagateShapeInference(
    const GraphDef& graph_def,
    const std::vector<std::pair<string, Tensor>>& input_node_info_list,
    Graph* graph, ShapeRefiner* shape_refiner) {
  Status status;
  auto visit = [&shape_refiner, &input_node_info_list, &status](Node* node) {
    if (!status.ok()) {
      return;
    }
    CHECK_NE(node, nullptr);
    // If we visit an input node, we use the shape provided and set the
    // shape accordingly.
    bool is_input_node = false;
    for (const std::pair<string, Tensor>& input_node_info :
         input_node_info_list) {
      if (node->name() == input_node_info.first) {
        shape_inference::InferenceContext* context =
            shape_refiner->GetContext(node);
        shape_inference::ShapeHandle handle;
        status = context->MakeShapeFromTensorShape(
            input_node_info.second.shape(), &handle);
        if (!status.ok()) {
          break;
        }
        status = shape_refiner->SetShape(node, 0, handle);
        if (!status.ok()) {
          break;
        }
        is_input_node = true;
      }
      if (!status.ok()) {
        break;
      }
    }
    // If not an input node call AddNode() that recomputes the shape.
    if (!is_input_node && status.ok()) {
      status = shape_refiner->AddNode(node);
    }
    if (!status.ok()) {
      VLOG(1) << "Shape inference failed for node: " << node->name();
    }
  };

  ReverseDFS(*graph, {}, visit);

  return status;
}

/* static */ Status RemoteFusedGraphExecuteUtils::BuildTensorShapeMapFromGraph(
    const Graph& graph, const ShapeRefiner& shape_refiner,
    TensorShapeMap* tensor_shape_map) {
  for (int i = 0; i < graph.num_node_ids(); ++i) {
    const Node* node = graph.FindNodeId(i);
    CHECK_NE(node, nullptr);
    for (int j = 0; j < node->num_outputs(); ++j) {
      const int output_index = j;
      const DataType dt = node->output_type(output_index);
      shape_inference::InferenceContext* context =
          shape_refiner.GetContext(node);
      CHECK_NE(context, nullptr);
      shape_inference::ShapeHandle shape_handle = context->output(output_index);
      if (context->RankKnown(shape_handle)) {
        TensorShape ts;
        for (int k = 0; k < context->Rank(shape_handle); ++k) {
          shape_inference::DimensionHandle dh = context->Dim(shape_handle, k);
          CHECK(context->ValueKnown(dh));
          ts.AddDim(context->Value(dh));
        }
        const string& node_name = node->name();
        CHECK(tensor_shape_map->count(node_name) == 0);
        tensor_shape_map->emplace(node_name,
                                  std::make_pair(j, std::make_pair(dt, ts)));
      } else {
        return errors::InvalidArgument("Graph contains unknow shapes");
      }
    }
  }
  return Status::OK();
}

/* static */ const RemoteFusedGraphExecuteUtils::TensorShapeType*
RemoteFusedGraphExecuteUtils::GetTensorShapeType(
    const TensorShapeMap& tensor_shape_map, const string& node_name) {
  if (node_name.find(':') != string::npos) {
    const TensorId tid = ParseTensorName(node_name);
    return GetTensorShapeType(tensor_shape_map, tid.first.ToString(),
                              tid.second);
  } else {
    return GetTensorShapeType(tensor_shape_map, node_name, 0);
  }
}

/* static */ const RemoteFusedGraphExecuteUtils::TensorShapeType*
RemoteFusedGraphExecuteUtils::GetTensorShapeType(
    const TensorShapeMap& tensor_shape_map, const string& node_name,
    const int port) {
  CHECK_EQ(node_name.find(':'), string::npos);
  if (tensor_shape_map.count(node_name) <= 0) {
    return nullptr;
  }
  auto its = tensor_shape_map.equal_range(node_name);
  for (auto it = its.first; it != its.second; ++it) {
    if (it->second.first == port) {
      return &it->second.second;
    }
  }
  return nullptr;
}

/* static */ void
RemoteFusedGraphExecuteUtils::BuildRemoteGraphInputsAndOutputsFromProto(
    const RemoteFusedGraphExecuteInfo& proto,
    std::vector<std::pair<string, Tensor>>* inputs,
    std::vector<string>* outputs) {
  CHECK_EQ(proto.graph_input_node_name_size(),
           proto.default_graph_input_tensor_shape_size());
  for (int i = 0; i < proto.graph_input_node_name_size(); ++i) {
    inputs->emplace_back(
        proto.graph_input_node_name(i),
        Tensor(proto.default_graph_input_tensor_shape(i).dtype(),
               TensorShape(proto.default_graph_input_tensor_shape(i).shape())));
  }
  for (const string& output_node_name : proto.graph_output_node_name()) {
    outputs->emplace_back(output_node_name);
  }
}

/* static */ void RemoteFusedGraphExecuteUtils::EmplaceTensorShapeType(
    const string& name, const Tensor& tensor,
    TensorShapeMap* tensor_shape_map) {
  const TensorId tid = ParseTensorName(name);
  CHECK_EQ(tensor_shape_map->count(name), 0);
  tensor_shape_map->emplace(
      tid.first.ToString(),
      std::make_pair(tid.second,
                     std::make_pair(tensor.dtype(), tensor.shape())));
}

/* static */ Status RemoteFusedGraphExecuteUtils::BuildAndAddTensorShapes(
    const std::vector<std::pair<string, Tensor>>& input_tensors,
    const bool dry_run_inference, GraphDef* graph_def) {
  TensorShapeMap tensor_shape_map;
  if (dry_run_inference) {
    TF_RETURN_IF_ERROR(DryRunInferenceForAllNode(*graph_def, input_tensors,
                                                 /*initialize_by_zero=*/true,
                                                 &tensor_shape_map));
  } else {
    ImportGraphDefOptions opts;
    Graph graph(OpRegistry::Global());
    ShapeRefiner shape_refiner(graph.versions(), graph.op_registry());
    TF_RETURN_IF_ERROR(
        ImportGraphDef(opts, *graph_def, &graph, &shape_refiner));
    TF_RETURN_IF_ERROR(PropagateShapeInference(*graph_def, input_tensors,
                                               &graph, &shape_refiner));
    TF_RETURN_IF_ERROR(
        BuildTensorShapeMapFromGraph(graph, shape_refiner, &tensor_shape_map));
  }

  for (NodeDef& node_def : *graph_def->mutable_node()) {
    TF_RETURN_IF_ERROR(
        AddOutputTensorShapeTypeByTensorShapeMap(tensor_shape_map, &node_def));
  }

  return Status::OK();
}

/* static */ Status
RemoteFusedGraphExecuteUtils::BuildRemoteFusedGraphExecuteInfo(
    const string& executor_name, const GraphDef& subgraph_def,
    const std::vector<string>& inputs, const std::vector<string>& outputs,
    const bool require_shape_type, RemoteFusedGraphExecuteInfo* execute_info,
    DataTypeVector* input_types, DataTypeVector* output_types) {
  CHECK_NOTNULL(execute_info);
  CHECK_NOTNULL(input_types);
  CHECK_NOTNULL(output_types);

  execute_info->Clear();
  execute_info->set_executor_name(executor_name);

  // copy graph
  *execute_info->mutable_remote_graph() = subgraph_def;

  for (const string& input : inputs) {
    DataType dt;
    TensorShape shape;
    const bool has_shapetype =
        GetOutputTensorShapeType(subgraph_def, input, &dt, &shape);

    execute_info->add_graph_input_node_name(input);
    if (has_shapetype) {
      RemoteFusedGraphExecuteInfo::TensorShapeTypeProto& tensor_shape_type =
          *execute_info->add_default_graph_input_tensor_shape();
      tensor_shape_type.set_dtype(dt);
      TensorShapeProto& tensor_shape_proto = *tensor_shape_type.mutable_shape();
      for (const int64 dim : shape.dim_sizes()) {
        tensor_shape_proto.add_dim()->set_size(dim);
      }
      input_types->push_back(dt);
    } else {
      CHECK(!require_shape_type)
          << "No shape type found for " << input << DumpGraphDef(subgraph_def);
      // Assuming input type is float if no data provided.
      input_types->push_back(DT_FLOAT);
    }
  }

  for (const string& output : outputs) {
    DataType dt;
    TensorShape shape;
    const bool has_shapetype =
        GetOutputTensorShapeType(subgraph_def, output, &dt, &shape);

    execute_info->add_graph_output_node_name(output);
    if (has_shapetype) {
      RemoteFusedGraphExecuteInfo::TensorShapeTypeProto&
          tensor_shape_type_proto =
              *execute_info->add_default_graph_output_tensor_shape();
      tensor_shape_type_proto.set_dtype(dt);
      TensorShapeProto& tensor_shape_proto =
          *tensor_shape_type_proto.mutable_shape();
      for (const int64 dim : shape.dim_sizes()) {
        tensor_shape_proto.add_dim()->set_size(dim);
      }
      output_types->push_back(dt);
    } else {
      CHECK(!require_shape_type)
          << "No shape type found for " << output << DumpGraphDef(subgraph_def);
      // Assuming output type is float if no data provided.
      output_types->push_back(DT_FLOAT);
    }
  }

  return Status::OK();
}

/* static */ Status
RemoteFusedGraphExecuteUtils::BuildRemoteFusedGraphExecuteOpNode(
    const string& node_name, const string& executor_name,
    const GraphDef& subgraph_def, const std::vector<string>& inputs,
    const std::vector<string>& outputs, const bool require_shape_type,
    Graph* graph, Node** created_node) {
  CHECK_NOTNULL(graph);
  CHECK_NOTNULL(created_node);

  RemoteFusedGraphExecuteInfo execute_info;
  DataTypeVector input_types;
  DataTypeVector output_types;

  TF_CHECK_OK(RemoteFusedGraphExecuteUtils::BuildRemoteFusedGraphExecuteInfo(
      executor_name, subgraph_def, inputs, outputs, require_shape_type,
      &execute_info, &input_types, &output_types));

  std::vector<NodeBuilder::NodeOut> node_out_list;
  for (const string& input : inputs) {
    const TensorId tid = ParseTensorName(input);
    Node* node = FindMutableNodeByName(tid.first.ToString(), graph);
    CHECK_NOTNULL(node);
    node_out_list.emplace_back(node, tid.second);
  }

  const string execute_info_str = execute_info.SerializeAsString();

  auto builder =
      NodeBuilder(node_name, "RemoteFusedGraphExecute")
          .Input(node_out_list)
          .Attr("Tinputs", input_types)
          .Attr("Toutputs", output_types)
          .Attr("serialized_remote_fused_graph_execute_info", execute_info_str);

  TF_RETURN_IF_ERROR(builder.Finalize(graph, created_node));
  return Status::OK();
}

/* static */ Status RemoteFusedGraphExecuteUtils::BuildIdentityOpNode(
    const string& node_name, const string& input_node_name,
    const int input_node_port, const DataType dt, Graph* graph,
    Node** created_node) {
  Node* node = FindMutableNodeByName(input_node_name, graph);
  CHECK_NOTNULL(node);
  NodeBuilder::NodeOut node_out(node, input_node_port);

  auto builder =
      NodeBuilder(node_name, "Identity").Input(node_out).Attr("T", dt);

  TF_RETURN_IF_ERROR(builder.Finalize(graph, created_node));
  return Status::OK();
}

/* static */ Status RemoteFusedGraphExecuteUtils::ClusterizeNodes(
    const std::unordered_set<string>& node_names, const GraphDef& graph_def,
    std::vector<ClusterInfo>* cluster_infos) {
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions(), graph.op_registry());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, &shape_refiner));
  std::unordered_set<string> remaining_nodes = node_names;

  while (!remaining_nodes.empty()) {
    ClusterInfo ci;

    // Determine one cluster nodes
    std::unordered_set<const Node*> visited;
    std::deque<const Node*> queue;
    queue.emplace_back(FindNodeByName(*remaining_nodes.begin(), graph));
    while (!queue.empty()) {
      const Node* node = queue.front();
      CHECK_NOTNULL(node);
      queue.pop_front();
      const string& node_name = node->name();
      if (node_names.count(node_name) > 0) {
        std::get<0>(ci).emplace(node_name);
        remaining_nodes.erase(node_name);
      } else {
        // Edge of subgraph.  Do nothing.
        continue;
      }
      for (const Node* in : node->in_nodes()) {
        if (visited.insert(in).second) {
          queue.push_back(in);
        }
      }
      for (const Node* out : node->out_nodes()) {
        if (visited.insert(out).second) {
          queue.push_back(out);
        }
      }
    }

    // Determine one cluster border
    std::vector<string>& border_inputs = std::get<1>(ci);
    std::vector<string>& border_outputs = std::get<2>(ci);
    for (const string& node_name : node_names) {
      Node* node = FindMutableNodeByName(node_name, &graph);
      CHECK_NOTNULL(node);
      int input_count = 0;
      for (const Edge* in_edge : node->in_edges()) {
        const Node* src_node = in_edge->src();
        const bool src_is_outside =
            node_names.count(src_node->name()) <= 0 && !src_node->IsSource();
        if (src_is_outside) {
          const string src_name =
              strings::StrCat(src_node->name(), ":", in_edge->src_output());
          CHECK_EQ(1, src_node->num_outputs())
              << "output count of input border node must be one."
              << src_node->name();
          if (std::find(border_inputs.begin(), border_inputs.end(), src_name) ==
              border_inputs.end()) {
            border_inputs.emplace_back(src_name);
          }
        } else {
          ++input_count;
        }
      }
      CHECK(input_count == 0 || input_count == node->in_edges().size())
          << "Invalid input_count(" << input_count << ", "
          << node->in_edges().size() << ") " << node_name;

      for (const Edge* out_edge : node->out_edges()) {
        const Node* dst_node = out_edge->dst();
        CHECK_NOTNULL(dst_node);
        const bool dst_is_outside = node_names.count(dst_node->name()) <= 0;
        const string dst_name =
            strings::StrCat(node->name(), ":", out_edge->src_output());
        if (dst_is_outside) {
          if (dst_node->IsSink()) {
            CHECK_EQ(1, node->num_outputs())
                << "If you want to specify output node as subgraph output node "
                << "the output count of the node must be 1 "
                << "because that node is replaced by identity node.";
            const string identity_dst_name =
                strings::StrCat(node->name(), ":", 0);
            if (std::find(border_outputs.begin(), border_outputs.end(),
                          identity_dst_name) == border_outputs.end()) {
              border_outputs.emplace_back(identity_dst_name);
            }
          } else {
            if (std::find(border_outputs.begin(), border_outputs.end(),
                          dst_name) == border_outputs.end()) {
              border_outputs.emplace_back(dst_name);
            }
          }
        }
      }
    }
    cluster_infos->emplace_back(ci);
    VLOG(1) << DumpCluster(ci);
  }
  return Status::OK();
}

/* static */ Status RemoteFusedGraphExecuteUtils::BuildClusterSubgraphDef(
    const ClusterInfo& cluster, const GraphDef& graph_def,
    GraphDef* subgraph_def) {
  const std::unordered_set<string>& node_names = std::get<0>(cluster);
  const std::unordered_set<string>& border_input_names =
      BuildNodeSetFromNodeNamesAndPorts(std::get<1>(cluster));

  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions(), graph.op_registry());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, &shape_refiner));

  for (Node* node : graph.nodes()) {
    if (node != nullptr && node_names.count(node->name()) <= 0 &&
        border_input_names.count(node->name()) <= 0 && !node->IsSource() &&
        !node->IsSink()) {
      graph.RemoveNode(node);
    }
  }
  graph.ToGraphDef(subgraph_def);

  for (const string& subgraph_input : std::get<1>(cluster)) {
    const TensorId tid = ParseTensorName(subgraph_input);
    const string subgraph_input_name = tid.first.ToString();
    const int subgraph_input_port = tid.second;
    const NodeDef* node_def = FindNodeDefByName(subgraph_input_name, graph_def);
    CHECK_NOTNULL(node_def);
    std::vector<DataType> dt_vec;
    std::vector<TensorShape> shape_vec;
    GetOutputTensorShapeType(*node_def, &dt_vec, &shape_vec).IgnoreError();
    const DataType& dt =
        dt_vec.empty() ? DT_FLOAT : dt_vec.at(subgraph_input_port);
    const TensorShape& shape =
        shape_vec.empty() ? TensorShape({}) : shape_vec.at(subgraph_input_port);

    TF_RETURN_IF_ERROR(ReplaceInputNodeByPlaceHolder(subgraph_input_name, dt,
                                                     shape, subgraph_def));
  }

  // sort subgraph_def to align order in graph_def
  std::unordered_map<string, int> name_to_id_map;
  for (int i = 0; i < graph_def.node_size(); ++i) {
    name_to_id_map.emplace(graph_def.node(i).name(), i);
  }
  std::sort(subgraph_def->mutable_node()->begin(),
            subgraph_def->mutable_node()->end(),
            [&name_to_id_map](const NodeDef& node0, const NodeDef& node1) {
              CHECK(name_to_id_map.count(node0.name()) > 0);
              CHECK(name_to_id_map.count(node1.name()) > 0);
              const int id0 = name_to_id_map.at(node0.name());
              const int id1 = name_to_id_map.at(node1.name());
              return id0 < id1;
            });

  VLOG(1) << DumpGraphDef(*subgraph_def);
  return Status::OK();
}

/* static */ Status RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
    const std::vector<string>& border_inputs,
    const std::vector<string>& border_outputs, const GraphDef& graph_def,
    ClusterInfo* cluster) {
  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions(), graph.op_registry());
  TF_RETURN_IF_ERROR(ImportGraphDef({}, graph_def, &graph, &shape_refiner));

  std::unordered_set<const Node*> visited;
  std::deque<const Node*> queue;
  for (const string& output : border_outputs) {
    const TensorId tid = ParseTensorName(output);
    const string& output_node_name = tid.first.ToString();
    for (const Node* node : graph.nodes()) {
      if (output_node_name == node->name()) {
        queue.push_back(node);
        visited.insert(node);
      }
    }
  }

  std::unordered_set<const Node*> border_input_nodes;
  // propagate visit to parent nodes until input nodes
  while (!queue.empty()) {
    const Node* node = queue.front();
    queue.pop_front();
    for (const Edge* edge : node->in_edges()) {
      const Node* src_node = edge->src();
      CHECK_NOTNULL(src_node);
      const int src_port = edge->src_output();
      bool input_found = false;
      for (const string& input : border_inputs) {
        const TensorId tid = ParseTensorName(input);
        if (tid.first.ToString() == src_node->name() &&
            tid.second == src_port) {
          input_found = true;
          border_input_nodes.insert(src_node);
        }
      }
      if (visited.insert(src_node).second) {
        if (!input_found) {
          queue.push_back(src_node);
        }
      }
    }
  }

  for (const Node* node : visited) {
    if (node != nullptr && !node->IsSource() && !node->IsSink() &&
        border_input_nodes.count(node) <= 0) {
      std::get<0>(*cluster).insert(node->name());
    }
  }
  std::get<1>(*cluster) = border_inputs;
  std::get<2>(*cluster) = border_outputs;
  return Status::OK();
}

/* static */ Status RemoteFusedGraphExecuteUtils::FuseCluster(
    const GraphDef& input_graph_def, const std::vector<string>& inputs,
    const std::vector<string>& outputs,
    const string& remote_fused_graph_node_name, const ClusterInfo& cluster,
    const string& remote_graph_executor_name, const bool require_shape_type,
    GraphDef* output_graph_def) {
  LOG(INFO) << "Transforming quantized stripped model to a remote fused "
               "graph execute op by fusing a specified subgraph...";

  CHECK(!remote_graph_executor_name.empty());

  const std::vector<string>& border_inputs = std::get<1>(cluster);
  const std::vector<string>& border_outputs = std::get<2>(cluster);

  GraphDef subgraph_def;
  TF_RETURN_IF_ERROR(
      BuildClusterSubgraphDef(cluster, input_graph_def, &subgraph_def));

  Graph graph(OpRegistry::Global());
  ShapeRefiner shape_refiner(graph.versions(), graph.op_registry());
  TF_RETURN_IF_ERROR(
      ImportGraphDef({}, input_graph_def, &graph, &shape_refiner));

  Node* fused_node;
  TF_RETURN_IF_ERROR(BuildRemoteFusedGraphExecuteOpNode(
      remote_fused_graph_node_name, remote_graph_executor_name, subgraph_def,
      border_inputs, border_outputs, require_shape_type, &graph, &fused_node));

  for (const Node* node : graph.nodes()) {
    for (int i = 0; i < node->num_inputs(); ++i) {
      const Edge* edge = nullptr;
      TF_RETURN_IF_ERROR(node->input_edge(i, &edge));
      for (int j = 0; j < border_outputs.size(); ++j) {
        const string& output = border_outputs.at(j);
        const TensorId tid = ParseTensorName(output);
        const string output_name = tid.first.ToString();
        Node* src_node = edge->src();
        if (src_node != nullptr && src_node->name() == output_name &&
            edge->src_output() == tid.second) {
          // Source node is replaced by new fused node.
          Node* dst_node = edge->dst();
          const int dst_input = edge->dst_input();
          LOG(INFO) << "Removing existing edge to " << edge->dst()->name()
                    << " from " << edge->src()->name();
          graph.RemoveEdge(edge);
          graph.AddEdge(fused_node, j, dst_node, dst_input);
        }
      }
    }
  }

  // Replace output nodes by identity nodes which forward outputs from
  // RemoteFusedGraphExecuteOpNode
  for (const string& output : outputs) {
    const TensorId output_tid = ParseTensorName(output);
    const string output_name = output_tid.first.ToString();
    for (size_t i = 0; i < border_outputs.size(); ++i) {
      const TensorId subgraph_output_tid =
          ParseTensorName(border_outputs.at(i));
      const string& subgraph_output_name = subgraph_output_tid.first.ToString();
      if (output_name == subgraph_output_name) {
        LOG(INFO) << "As graph output and subgraph output are same, "
                  << "the graph output node is replaced by identity node";
        Node* original_output_node = FindMutableNodeByName(output_name, &graph);
        CHECK_NOTNULL(original_output_node);
        CHECK_EQ(1, original_output_node->num_outputs())
            << "Num outputs should be 1 for " << output << ".";
        graph.RemoveNode(original_output_node);
        Node* new_node;
        TF_RETURN_IF_ERROR(BuildIdentityOpNode(output_name,
                                               remote_fused_graph_node_name, i,
                                               DT_FLOAT, &graph, &new_node));
        CHECK_NOTNULL(new_node);
      }
    }
  }

  GraphDef result_graph_def;

  graph.ToGraphDef(&result_graph_def);

  ClusterInfo graph_cluster;
  TF_RETURN_IF_ERROR(
      BuildClusterByBorder(inputs, outputs, result_graph_def, &graph_cluster));

  // Remove unvisited nodes
  TF_RETURN_IF_ERROR(BuildClusterSubgraphDef(graph_cluster, result_graph_def,
                                             output_graph_def));

  return Status::OK();
}

/* static */ Status RemoteFusedGraphExecuteUtils::FuseRemoteGraphByNodeNames(
    const GraphDef& input_graph_def, const std::vector<string>& inputs,
    const std::vector<string>& outputs,
    const string& remote_fused_graph_node_name_prefix,
    const std::unordered_set<string>& subgraph_nodes,
    const string& remote_fused_graph_executor_name,
    const bool require_shape_type, GraphDef* output_graph_def) {
  std::vector<ClusterInfo> ci_vec;
  TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::ClusterizeNodes(
      subgraph_nodes, input_graph_def, &ci_vec));

  for (size_t i = 0; i < ci_vec.size(); ++i) {
    const string remote_fused_graph_node_name =
        strings::StrCat(remote_fused_graph_node_name_prefix, "/", i);
    TF_RETURN_IF_ERROR(FuseCluster(input_graph_def, inputs, outputs,
                                   remote_fused_graph_node_name, ci_vec.at(i),
                                   remote_fused_graph_executor_name,
                                   require_shape_type, output_graph_def));
  }
  return Status::OK();
}

/* static */ Status RemoteFusedGraphExecuteUtils::FuseRemoteGraphByBorder(
    const GraphDef& input_graph_def, const std::vector<string>& inputs,
    const std::vector<string>& outputs,
    const string& remote_fused_graph_node_name,
    const std::vector<string>& border_inputs,
    const std::vector<string>& border_outputs,
    const string& remote_graph_executor_name, const bool require_shape_type,
    GraphDef* output_graph_def) {
  ClusterInfo cluster;
  TF_RETURN_IF_ERROR(RemoteFusedGraphExecuteUtils::BuildClusterByBorder(
      border_inputs, border_outputs, input_graph_def, &cluster));

  return FuseCluster(
      input_graph_def, inputs, outputs, remote_fused_graph_node_name, cluster,
      remote_graph_executor_name, require_shape_type, output_graph_def);
}

/* static */ Status RemoteFusedGraphExecuteUtils::FuseRemoteGraphByOpTypes(
    const GraphDef& input_graph_def, const std::vector<string>& inputs,
    const std::vector<string>& outputs,
    const string& remote_fused_graph_node_name_prefix,
    const std::unordered_set<string>& fused_op_types,
    const string& remote_fused_graph_executor_name,
    const bool require_shape_type, GraphDef* output_graph_def) {
  const std::unordered_set<string> fused_nodes_filtered_by_op_types =
      BuildNodeMapFromOpTypes(input_graph_def, fused_op_types);

  return FuseRemoteGraphByNodeNames(
      input_graph_def, inputs, outputs, remote_fused_graph_node_name_prefix,
      fused_nodes_filtered_by_op_types, remote_fused_graph_executor_name,
      require_shape_type, output_graph_def);
}

/* static */ Status RemoteFusedGraphExecuteUtils::FuseRemoteGraphByExecutor(
    const GraphDef& input_graph_def, const std::vector<string>& inputs,
    const std::vector<string>& outputs, const string& executor_name,
    GraphDef* output_graph_def) {
  const ExecutorBuildFunc* build_func = GetExecutorBuildFunc(executor_name);
  if (build_func == nullptr) {
    return errors::InvalidArgument("Unknown executor name: " + executor_name);
  }
  std::unique_ptr<IRemoteFusedGraphExecutor> executor;
  TF_RETURN_IF_ERROR((*build_func)(&executor));
  CHECK_NOTNULL(executor.get());
  if (!executor->IsEnabled()) {
    // As this executor is not enabled, just return original graph as is.
    *output_graph_def = input_graph_def;
    return Status::OK();
  }
  return executor->FuseRemoteGraph(input_graph_def, inputs, outputs,
                                   output_graph_def);
}

/* static */ Status RemoteFusedGraphExecuteUtils::PlaceRemoteGraphArguments(
    const std::vector<string>& inputs, const std::vector<string>& outputs,
    const std::unordered_set<string>& fused_node_names,
    const std::vector<string>& border_inputs,
    const std::vector<string>& border_outputs,
    const std::unordered_set<string>& fused_op_types,
    const string& remote_fused_graph_node_name,
    const string& remote_graph_executor_name, GraphDef* graph_def) {
  CHECK_NOTNULL(graph_def);

  const std::unordered_set<string> fused_nodes_filtered_by_op_types =
      BuildNodeMapFromOpTypes(*graph_def, fused_op_types);

  for (NodeDef& node_def : *graph_def->mutable_node()) {
    string attr_str;
    TensorId tid;
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (IsSameNodeName(node_def, inputs.at(i), &tid)) {
        AppendDeliminator(&attr_str);
        attr_str += BuildNodeTypeAttr(RemoteFusedGraphExecuteInfo::GRAPH_INPUT,
                                      tid.second, i, remote_graph_executor_name,
                                      remote_fused_graph_node_name);
      }
    }
    for (size_t i = 0; i < outputs.size(); ++i) {
      if (IsSameNodeName(node_def, outputs.at(i), &tid)) {
        AppendDeliminator(&attr_str);
        attr_str += BuildNodeTypeAttr(RemoteFusedGraphExecuteInfo::GRAPH_OUTPUT,
                                      tid.second, i);
      }
    }
    for (const string& fused_node_name : fused_node_names) {
      if (fused_node_name == node_def.name()) {
        AppendDeliminator(&attr_str);
        attr_str += BuildNodeTypeAttr(RemoteFusedGraphExecuteInfo::FUSED_NODE);
      }
    }
    for (const string& fused_node_name : fused_nodes_filtered_by_op_types) {
      if (fused_node_name == node_def.name()) {
        AppendDeliminator(&attr_str);
        attr_str += BuildNodeTypeAttr(RemoteFusedGraphExecuteInfo::FUSED_NODE);
      }
    }
    for (size_t i = 0; i < border_inputs.size(); ++i) {
      if (IsSameNodeName(node_def, border_inputs.at(i), &tid)) {
        AppendDeliminator(&attr_str);
        attr_str += BuildNodeTypeAttr(RemoteFusedGraphExecuteInfo::BORDER_INPUT,
                                      tid.second, i);
      }
    }
    for (size_t i = 0; i < border_outputs.size(); ++i) {
      if (IsSameNodeName(node_def, border_outputs.at(i), &tid)) {
        AppendDeliminator(&attr_str);
        attr_str += BuildNodeTypeAttr(
            RemoteFusedGraphExecuteInfo::BORDER_OUTPUT, tid.second, i);
      }
    }
    if (attr_str.empty()) {
      attr_str += BuildNodeTypeAttr(RemoteFusedGraphExecuteInfo::UNUSED);
    }
    AddNodeAttr(ATTR_NODE_TYPE, attr_str, &node_def);
  }
  return Status::OK();
}

/* static */ Status
RemoteFusedGraphExecuteUtils::FuseRemoteGraphByPlacedArguments(
    const GraphDef& input_graph_def,
    const std::vector<std::pair<string, Tensor>>& input_tensors,
    GraphDef* output_graph_def) {
  std::unordered_map<int, string> input_map;
  std::unordered_map<int, string> output_map;
  std::unordered_set<string> fused_node_names;
  std::unordered_map<int, string> border_input_map;
  std::unordered_map<int, string> border_output_map;
  string remote_graph_executor_name;
  string remote_fused_graph_node_name;

  for (const NodeDef& node_def : input_graph_def.node()) {
    string attr_str;
    TF_RETURN_IF_ERROR(GetNodeAttr(node_def, ATTR_NODE_TYPE, &attr_str));
    std::vector<std::vector<string>> attr_strs;
    for (const string& str : str_util::Split(attr_str, ":")) {
      attr_strs.emplace_back(str_util::Split(str, ","));
    }
    if (attr_strs.empty()) {
      return errors::InvalidArgument("Remote graph node type not found.");
    }
    for (const std::vector<string>& attr : attr_strs) {
      if (attr.empty()) {
        return errors::InvalidArgument("Empty remote graph node type attr.");
      }
      int node_type_int;
      CHECK(strings::safe_strto32(attr.at(0), &node_type_int)) << attr.at(0);
      const RemoteFusedGraphExecuteInfo::NodeType node_type =
          static_cast<RemoteFusedGraphExecuteInfo::NodeType>(node_type_int);
      const string& name = node_def.name();
      int port;
      int index;

      switch (node_type) {
        case RemoteFusedGraphExecuteInfo::GRAPH_INPUT:
          VLOG(2) << "Graph input: " << name;
          CHECK_EQ(5, attr.size());
          CHECK(strings::safe_strto32(attr.at(1), &port));
          CHECK(strings::safe_strto32(attr.at(2), &index));
          CHECK(!attr.at(3).empty());
          remote_graph_executor_name = attr.at(3);
          CHECK(!attr.at(4).empty());
          remote_fused_graph_node_name = attr.at(4);
          input_map.emplace(index, strings::StrCat(name, ":", port));
          if (GetExecutorBuildFunc(remote_graph_executor_name) == nullptr) {
            LOG(INFO) << "Executor for " << remote_graph_executor_name
                      << " not registered.  Do not fuse.";
            *output_graph_def = input_graph_def;
            return Status::OK();
          }
          break;
        case RemoteFusedGraphExecuteInfo::GRAPH_OUTPUT:
          VLOG(2) << "Graph output: " << name;
          CHECK_EQ(3, attr.size());
          CHECK(strings::safe_strto32(attr.at(1), &port));
          CHECK(strings::safe_strto32(attr.at(2), &index));
          output_map.emplace(index, strings::StrCat(name, ":", port));
          break;
        case RemoteFusedGraphExecuteInfo::FUSED_NODE:
          VLOG(2) << "Fused node: " << name;
          CHECK_EQ(1, attr.size());
          fused_node_names.emplace(name);
          break;
        case RemoteFusedGraphExecuteInfo::BORDER_INPUT:
          VLOG(2) << "Border input: " << name;
          CHECK_EQ(3, attr.size());
          CHECK(strings::safe_strto32(attr.at(1), &port));
          CHECK(strings::safe_strto32(attr.at(2), &index));
          border_input_map.emplace(index, strings::StrCat(name, ":", port));
          break;
        case RemoteFusedGraphExecuteInfo::BORDER_OUTPUT:
          VLOG(2) << "Border output: " << name;
          CHECK_EQ(3, attr.size());
          CHECK(strings::safe_strto32(attr.at(1), &port));
          CHECK(strings::safe_strto32(attr.at(2), &index));
          border_output_map.emplace(index, strings::StrCat(name, ":", port));
          break;
        case RemoteFusedGraphExecuteInfo::UNUSED:
          // do nothing
          break;
        default:
          // unsupported value
          LOG(FATAL);
      }
    }
  }
  bool require_shape_type = false;
  std::vector<string> inputs;
  std::vector<string> outputs;
  std::vector<string> border_inputs;
  std::vector<string> border_outputs;
  ConvertMapToVector(input_map, &inputs);
  ConvertMapToVector(output_map, &outputs);
  ConvertMapToVector(border_input_map, &border_inputs);
  ConvertMapToVector(border_output_map, &border_outputs);

  if (!input_tensors.empty()) {
    bool input_match = false;
    if (inputs.size() == input_tensors.size()) {
      for (const std::pair<string, Tensor>& input_tensor : input_tensors) {
        if (!ContainsSameTensorId(input_tensor.first, inputs)) {
          break;
        }
        DataType data_type;
        TensorShape shape;
        if (GetOutputTensorShapeType(input_graph_def, input_tensor.first,
                                     &data_type, &shape)) {
          if (data_type == input_tensor.second.dtype() &&
              shape == input_tensor.second.shape()) {
            VLOG(2) << "Input matched!";
            // Shape type matched.
            input_match = true;
            require_shape_type = true;
          }
        } else {
          // Shape type not required.
          input_match = true;
        }
      }
    }
    if (!input_match) {
      // Input mismatch.  Just copy original graph
      *output_graph_def = input_graph_def;
      return Status::OK();
    }
  }

  if (!fused_node_names.empty()) {
    TF_RETURN_IF_ERROR(FuseRemoteGraphByNodeNames(
        input_graph_def, inputs, outputs, remote_fused_graph_node_name,
        fused_node_names, remote_graph_executor_name, require_shape_type,
        output_graph_def));
  } else if (!border_inputs.empty() || !border_outputs.empty()) {
    TF_RETURN_IF_ERROR(FuseRemoteGraphByBorder(
        input_graph_def, inputs, outputs, remote_fused_graph_node_name,
        border_inputs, border_outputs, remote_graph_executor_name,
        require_shape_type, output_graph_def));
  } else {
    *output_graph_def = input_graph_def;
  }

  return Status::OK();
}

/* static */ bool RemoteFusedGraphExecuteUtils::IsFuseReady(
    const GraphDef& graph_def,
    const std::vector<std::pair<string, Tensor>>& input_tensors) {
  for (const std::pair<string, Tensor>& input_tensor : input_tensors) {
    const NodeDef* node_def = FindNodeDefByName(input_tensor.first, graph_def);
    if (node_def == nullptr) {
      return false;
    }
    string attr;
    const Status status = GetNodeAttr(*node_def, ATTR_NODE_TYPE, &attr);
    if (!status.ok() || attr.empty()) {
      return false;
    }
  }
  return true;
}

/* static */ Status RemoteFusedGraphExecuteUtils::CopyByteArrayToTensor(
    const void* src_ptr, const int src_size, Tensor* tensor) {
  CHECK(tensor->TotalBytes() >= src_size)
      << tensor->TotalBytes() << ", " << src_size;
  void* dst_ptr;
  switch (tensor->dtype()) {
    case DT_FLOAT:
      dst_ptr = tensor->flat<float>().data();
      break;
    case DT_DOUBLE:
      dst_ptr = tensor->flat<double>().data();
      break;
    case DT_INT32:
      dst_ptr = tensor->flat<int32>().data();
      break;
    case DT_UINT8:
      dst_ptr = tensor->flat<uint8>().data();
      break;
    case DT_INT16:
      dst_ptr = tensor->flat<int16>().data();
      break;
    case DT_INT8:
      dst_ptr = tensor->flat<int8>().data();
      break;
    case DT_STRING:
      dst_ptr = tensor->flat<string>().data();
      break;
    case DT_INT64:
      dst_ptr = tensor->flat<int64>().data();
      break;
    case DT_BOOL:
      dst_ptr = tensor->flat<bool>().data();
      break;
    case DT_QINT8:
      dst_ptr = tensor->flat<qint8>().data();
      break;
    case DT_QUINT8:
      dst_ptr = tensor->flat<quint8>().data();
      break;
    case DT_QINT32:
      dst_ptr = tensor->flat<qint32>().data();
      break;
    case DT_BFLOAT16:
      dst_ptr = tensor->flat<bfloat16>().data();
      break;
    case DT_QINT16:
      dst_ptr = tensor->flat<qint16>().data();
      break;
    case DT_QUINT16:
      dst_ptr = tensor->flat<quint16>().data();
      break;
    case DT_UINT16:
      dst_ptr = tensor->flat<uint16>().data();
      break;
    default:
      LOG(FATAL) << "type " << tensor->dtype() << " is not supported.";
      break;
  }
  CHECK_NOTNULL(dst_ptr);
  std::memcpy(dst_ptr, src_ptr, src_size);
  return Status::OK();
}

/* static */ std::unordered_set<string>
RemoteFusedGraphExecuteUtils::BuildNodeMapFromOpTypes(
    const GraphDef& graph_def, const std::unordered_set<string>& op_types) {
  std::unordered_set<string> retval;
  for (const NodeDef& node_def : graph_def.node()) {
    if (op_types.count(node_def.op()) > 0) {
      retval.emplace(node_def.name());
    }
  }
  return retval;
}

/* static */ std::unordered_set<string>
RemoteFusedGraphExecuteUtils::BuildNodeMapFromOpsDefinitions(
    const GraphDef& graph_def,
    const IRemoteFusedGraphOpsDefinitions& ops_definitions) {
  std::unordered_set<string> retval;
  for (const NodeDef& node_def : graph_def.node()) {
    std::vector<DataType> dt_vec;
    std::vector<TensorShape> shape_vec;
    const Status status =
        GetOutputTensorShapeType(node_def, &dt_vec, &shape_vec);
    if (!status.ok()) {
      shape_vec.clear();
    }
    if (ops_definitions.GetOpIdFor(
            node_def.op(), DataTypeVector(dt_vec.begin(), dt_vec.end())) !=
        IRemoteFusedGraphOpsDefinitions::INVALID_OP_ID) {
      retval.emplace(node_def.name());
    }
  }
  return retval;
}

/* static */ Status RemoteFusedGraphExecuteUtils::ReplaceInputNodeByPlaceHolder(
    const string& input, const DataType type, const TensorShape& shape,
    GraphDef* graph_def) {
  const TensorId tid = ParseTensorName(input);
  CHECK_EQ(0, tid.second);
  const string node_name = tid.first.ToString();
  for (NodeDef& node : *graph_def->mutable_node()) {
    if (node.name() != node_name) {
      continue;
    }
    if (node.op() == "Placeholder") {
      return Status::OK();
    } else {
      NodeDef placeholder_node;
      placeholder_node.set_op("Placeholder");
      placeholder_node.set_name(node_name);
      AddNodeAttr("dtype", type, &placeholder_node);
      AddNodeAttr("shape", shape, &placeholder_node);
      // TODO(satok): Remove once we merge attributes
      AddOutputTensorShapeType({type}, {shape}, &placeholder_node);
      node.Clear();
      node = placeholder_node;
      return Status::OK();
    }
  }
  return errors::InvalidArgument(
      strings::StrCat(node_name, " not found for replacement."));
}

/* static */ string RemoteFusedGraphExecuteUtils::BuildNodeTypeAttr(
    const RemoteFusedGraphExecuteInfo::NodeType node_type, const int port,
    const int index, const string& executor_name, const string& node_name) {
  return strings::StrCat(static_cast<int>(node_type), ",", port, ",", index,
                         ",", executor_name, ",", node_name);
}

/* static */ string RemoteFusedGraphExecuteUtils::BuildNodeTypeAttr(
    const RemoteFusedGraphExecuteInfo::NodeType node_type, const int port,
    const int index) {
  return strings::StrCat(static_cast<int>(node_type), ",", port, ",", index);
}

/* static */ string RemoteFusedGraphExecuteUtils::BuildNodeTypeAttr(
    const RemoteFusedGraphExecuteInfo::NodeType node_type) {
  return strings::StrCat(static_cast<int>(node_type));
}

}  // namespace tensorflow
