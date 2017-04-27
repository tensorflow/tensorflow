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
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_DATA_TYPES;
/* static */ constexpr const char* const
    RemoteFusedGraphExecuteUtils::ATTR_OUTPUT_SHAPES;

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
    const NodeDef& node_def, std::vector<DataType>* data_types,
    std::vector<TensorShape>* shapes) {
  Status status;
  if (data_types != nullptr) {
    status = GetNodeAttr(node_def, ATTR_OUTPUT_DATA_TYPES, data_types);
  }
  if (!status.ok()) {
    return status;
  }
  if (shapes != nullptr) {
    status = GetNodeAttr(node_def, ATTR_OUTPUT_SHAPES, shapes);
    if (status.ok() && data_types != nullptr) {
      CHECK_EQ(data_types->size(), shapes->size());
    }
  }

  return status;
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

}  // namespace tensorflow
