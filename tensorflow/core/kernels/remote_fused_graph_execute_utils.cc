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

#include <utility>

#include "tensorflow/core/public/session.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

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
  for (const std::pair<string, Tensor>& input_node_info :
       input_node_info_list) {
    output_tensors.push_back(input_node_info.second);
  }

  for (int i = 0; i < output_node_names.size(); ++i) {
    const string& name = output_node_names.at(i);
    CHECK_EQ(tensor_shape_map->count(name), 0);
    const Tensor& tensor = output_tensors.at(i);
    tensor_shape_map->emplace(name,
                              std::make_pair(tensor.dtype(), tensor.shape()));
  }
  for (int i = 0; i < input_node_info_list.size(); ++i) {
    const string& name = input_node_info_list.at(i).first;
    CHECK_EQ(tensor_shape_map->count(name), 0);
    const Tensor& tensor = output_tensors.at(output_node_names.size() + i);
    tensor_shape_map->emplace(name,
                              std::make_pair(tensor.dtype(), tensor.shape()));
  }
  CHECK(graph_def.node_size() == output_tensors.size());
  return status;
}

/* static */ bool RemoteFusedGraphExecuteUtils::IsInputNode(
    const std::vector<std::pair<string, Tensor>>& input_tensor_vector,
    const string& node_name) {
  for (const std::pair<string, Tensor>& pair : input_tensor_vector) {
    if (node_name == pair.first) {
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
    tensor_shape_map->emplace(node_name,
                              std::make_pair(tensor.dtype(), tensor.shape()));
  }
}

}  // namespace tensorflow
