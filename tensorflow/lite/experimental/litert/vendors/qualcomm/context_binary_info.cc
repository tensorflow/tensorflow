// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/context_binary_info.h"

#include <cstddef>
#include <utility>
#include <vector>

#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "third_party/qairt/latest/include/QNN/System/QnnSystemContext.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_tensor.h"

namespace litert {
namespace qnn {

namespace {

Expected<void> InsertQnnTensors(int num_qnn_tensors, Qnn_Tensor_t* qnn_tensors,
                                std::vector<QnnTensor>* tensors) {
  tensors->clear();
  tensors->reserve(num_qnn_tensors);
  for (auto i = 0; i < num_qnn_tensors; ++i) {
    auto tensor = QnnTensor::Create(qnn_tensors[i]);
    if (!tensor) {
      return Unexpected(tensor.Error());
    }
    tensors->push_back(std::move(*tensor));
  }
  return {};
}

Expected<void> InsertQnnGraphInfos(
    int num_qnn_graph_infos, QnnSystemContext_GraphInfo_t* qnn_graph_infos,
    std::vector<GraphInfo>* graphs) {
  graphs->clear();
  graphs->reserve(num_qnn_graph_infos);
  for (auto i = 0; i < num_qnn_graph_infos; ++i) {
    auto graph = GraphInfo::Create(qnn_graph_infos[i]);
    if (!graph) {
      return Unexpected(graph.Error());
    }
    graphs->push_back(std::move(*graph));
  }

  return {};
}

}  // namespace

Expected<GraphInfo> GraphInfo::Create(
    const QnnSystemContext_GraphInfo_t& graph_info) {
  GraphInfo info;
  auto status = info.Init(graph_info);
  if (status) {
    return info;
  } else {
    return Unexpected(status.Error());
  }
}

Expected<void> GraphInfo::Init(const QnnSystemContext_GraphInfo_t& graph_info) {
  if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_1) {
    const auto& graph_info_ = graph_info.graphInfoV1;
    name_ = graph_info_.graphName;
    LITERT_LOG(LITERT_INFO, "Found qnn graph: %s", name_.c_str());

    if (auto status = InsertQnnTensors(graph_info_.numGraphInputs,
                                       graph_info_.graphInputs, &inputs_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnTensors(graph_info_.numGraphOutputs,
                                       graph_info_.graphOutputs, &outputs_);
        !status) {
      return Unexpected(status.Error());
    }

  } else if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_2) {
    const auto& graph_info_ = graph_info.graphInfoV2;
    name_ = graph_info_.graphName;
    LITERT_LOG(LITERT_INFO, "Found qnn graph: %s", name_.c_str());

    if (auto status = InsertQnnTensors(graph_info_.numGraphInputs,
                                       graph_info_.graphInputs, &inputs_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnTensors(graph_info_.numGraphOutputs,
                                       graph_info_.graphOutputs, &outputs_);
        !status) {
      return Unexpected(status.Error());
    }
  } else if (graph_info.version == QNN_SYSTEM_CONTEXT_GRAPH_INFO_VERSION_3) {
    const auto& graph_info_ = graph_info.graphInfoV3;
    name_ = graph_info_.graphName;
    LITERT_LOG(LITERT_INFO, "Found qnn graph: %s", name_.c_str());

    if (auto status = InsertQnnTensors(graph_info_.numGraphInputs,
                                       graph_info_.graphInputs, &inputs_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnTensors(graph_info_.numGraphOutputs,
                                       graph_info_.graphOutputs, &outputs_);
        !status) {
      return Unexpected(status.Error());
    }

  } else {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported graph info version.");
  }
  return {};
}

Expected<void> ContextBinaryInfo::Init(
    const QnnSystemContext_BinaryInfo_t& binary_info) {
  if (binary_info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_1) {
    const auto& context_binary_info = binary_info.contextBinaryInfoV1;
    if (auto status = InsertQnnTensors(context_binary_info.numContextTensors,
                                       context_binary_info.contextTensors,
                                       &context_tensors_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnGraphInfos(context_binary_info.numGraphs,
                                          context_binary_info.graphs, &graphs_);
        !status) {
      return Unexpected(status.Error());
    }

  } else if (binary_info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_2) {
    const auto& context_binary_info = binary_info.contextBinaryInfoV2;
    if (auto status = InsertQnnTensors(context_binary_info.numContextTensors,
                                       context_binary_info.contextTensors,
                                       &context_tensors_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnGraphInfos(context_binary_info.numGraphs,
                                          context_binary_info.graphs, &graphs_);
        !status) {
      return Unexpected(status.Error());
    }
  } else if (binary_info.version == QNN_SYSTEM_CONTEXT_BINARY_INFO_VERSION_3) {
    const auto& context_binary_info = binary_info.contextBinaryInfoV3;
    if (auto status = InsertQnnTensors(context_binary_info.numContextTensors,
                                       context_binary_info.contextTensors,
                                       &context_tensors_);
        !status) {
      return Unexpected(status.Error());
    }
    if (auto status = InsertQnnGraphInfos(context_binary_info.numGraphs,
                                          context_binary_info.graphs, &graphs_);
        !status) {
      return Unexpected(status.Error());
    }
  } else {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Unsupported context binary version.");
  }
  return {};
}

Expected<ContextBinaryInfo> ContextBinaryInfo::Create(
    QnnManager& qnn, const void* exec_bytecode_ptr, size_t exec_bytecode_size) {
  auto system_context_handle = qnn.CreateSystemContextHandle();
  if (!system_context_handle) {
    return Unexpected(system_context_handle.Error());
  }

  const QnnSystemContext_BinaryInfo_t* binary_info = nullptr;
  Qnn_ContextBinarySize_t binary_info_size = 0;
  if (auto status = qnn.SystemApi()->systemContextGetBinaryInfo(
          system_context_handle->get(), const_cast<void*>(exec_bytecode_ptr),
          exec_bytecode_size, &binary_info, &binary_info_size);
      status != QNN_SUCCESS) {
    LITERT_LOG(LITERT_ERROR, "Failed to get context binary info: %d", status);
    return Unexpected(kLiteRtStatusErrorRuntimeFailure,
                      "Failed to get context binary info");
  }

  if (!binary_info) {
    LITERT_LOG(LITERT_ERROR, "Null binary info", "");
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Null binary info");
  }

  ContextBinaryInfo info;
  auto status = info.Init(*binary_info);

  if (status) {
    return info;
  } else {
    return Unexpected(status.Error());
  }
}

}  // namespace qnn
}  // namespace litert
