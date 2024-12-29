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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"

#include <alloca.h>
#include <stdio.h>

#include <algorithm>
#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/HTP/QnnHtpGraph.h"
#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnGraph.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

// Get empty configurations for graph building.
inline absl::Span<const QnnGraph_Config_t*> GetFp32GraphConfigs() {
  static QnnHtpGraph_CustomConfig_t htp_graph_config =
      QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  htp_graph_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
  htp_graph_config.precision = QNN_PRECISION_FLOAT16;

  static QnnGraph_Config_t graph_config = QNN_GRAPH_CONFIG_INIT;
  graph_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_config.customConfig = &htp_graph_config;

  static const QnnGraph_Config_t* configs[2] = {&graph_config, nullptr};
  return absl::MakeSpan(configs);
}

inline absl::Span<const QnnGraph_Config_t*> GetDefaultGraphConfigs() {
  static const QnnGraph_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

absl::Span<const QnnGraph_Config_t*> GraphMapper::PickGraphConfigHeuristic() {
  for (const auto& input : subgraph_.Inputs()) {
    if (input.ElementType() == ElementType::Float32) {
      return GetFp32GraphConfigs();
    }
  }
  for (const auto& output : subgraph_.Outputs()) {
    if (output.ElementType() == ElementType::Float32) {
      return GetFp32GraphConfigs();
    }
  }
  return GetDefaultGraphConfigs();
}

LiteRtStatus GraphMapper::AssignTensorName(Qnn_Tensor_t& qnn_tensor) {
  char* name = nullptr;
  const int written = asprintf(&name, "Tensor_%d", cur_tensor_num_++);
  LITERT_ENSURE(written != -1 && name != nullptr, kLiteRtStatusErrorNotFound,
                "Failed to make tensor name");
  qnn_tensor.v2.name = name;
  return kLiteRtStatusOk;
}

absl::flat_hash_map<LiteRtTensor, uint32_t>& GraphMapper::CurrentScope() {
  return current_scope_;
}

LiteRtStatus GraphMapper::LookupInScope(LiteRtTensor litert_tensor,
                                        Qnn_Tensor_t& qnn_tensor) {
  // If we go in topological order, this should never happen. TODO: add
  // "internal error" status code.
  const auto qnn_id = CurrentScope().find(litert_tensor);
  // when qnn_id is not found, the tensor is a constant tensor thats not been
  // added qnn graph.
  if (qnn_id == CurrentScope().end()) {
    LITERT_LOG(LITERT_INFO, "Adding constant tensor %s to qnn graph",
               qnn_tensor.v2.name);
    LITERT_RETURN_STATUS_IF_NOT_OK(
        LegalizeAndRegister(litert_tensor, qnn_tensor));
    LITERT_RETURN_STATUS_IF_NOT_OK(PushToScope(litert_tensor, qnn_tensor));
    // }
    return kLiteRtStatusOk;
  }
  LITERT_LOG(LITERT_INFO, "Found tensor %d in current_scope.", qnn_id->second);
  ResetTensor(qnn_tensor);
  qnn_tensor.v2.id = qnn_id->second;

  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::PushToScope(LiteRtTensor litert_tensor,
                                      Qnn_Tensor_t& qnn_tensor) {
  CurrentScope()[litert_tensor] = MoveToId(qnn_tensor);
  return kLiteRtStatusOk;
}

QnnManager& GraphMapper::Qnn() { return qnn_; }

Qnn_GraphHandle_t& GraphMapper::QnnGraph() { return qnn_graph_; }

LiteRtStatus GraphMapper::LegalizeAndRegister(LiteRtTensor litert_tensor,
                                              Qnn_Tensor_t& qnn_tensor) {
  litert::Tensor tensor(litert_tensor);
  LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeTensor(tensor, qnn_tensor));
  LITERT_RETURN_STATUS_IF_NOT_OK(AssignTensorName(qnn_tensor));

  // Set tensor as graph output if it is used by other Ops.
  if (graph_outpus_.contains(litert_tensor)) {
    LITERT_LOG(LITERT_INFO, "Setting tensor %d as Graph output",
               qnn_tensor.v2.id);
    qnn_tensor.v2.type = QNN_TENSOR_TYPE_APP_READ;
  }

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->tensorCreateGraphTensor(QnnGraph(), &qnn_tensor));

  LITERT_LOG(LITERT_INFO, "Legalized and registered tensor %d",
             qnn_tensor.v2.id);

  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::IsLiteRtSubgraphSupported() {
  // For now, we assume all LiteRt subgraphs are supported.
  // TODO: b/381133565: Implement or remove this function.
  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::InitQnnGraph(absl::string_view qnn_graph_name) {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->graphCreate(context_handle_, qnn_graph_name.data(),
                              PickGraphConfigHeuristic().data(), &QnnGraph()));
  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::Finalize() {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->graphFinalize(QnnGraph(), nullptr, nullptr));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
