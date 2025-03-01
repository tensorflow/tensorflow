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

#include <array>
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
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

inline absl::Span<const QnnGraph_Config_t*> GetDefaultGraphConfigs() {
  static std::array<QnnHtpGraph_CustomConfig_t, 2> graph_custom_configs;
  // QNN suggest always enable relax precision.
  graph_custom_configs[0] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[0].option = QNN_HTP_GRAPH_CONFIG_OPTION_PRECISION;
  graph_custom_configs[0].precision = QNN_PRECISION_FLOAT16;
  // Default use O3 for now.
  graph_custom_configs[1] = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_configs[1].option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  graph_custom_configs[1].optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  // Change to 2 if you want to use O2 (default).
  graph_custom_configs[1].optimizationOption.floatValue = 3;

  static std::array<QnnGraph_Config_t, 2> graph_configs;
  graph_configs[0] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[0].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[0].customConfig = &graph_custom_configs[0];

  graph_configs[1] = QNN_GRAPH_CONFIG_INIT;
  graph_configs[1].option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_configs[1].customConfig = &graph_custom_configs[1];

  static std::array<const QnnGraph_Config_t*, 3> result = {
      &graph_configs[0], &graph_configs[1], nullptr};

  return absl::MakeSpan(result.data(), result.size());
}

inline absl::Span<const QnnGraph_Config_t*> GetLegacyGraphConfigs() {
  static QnnHtpGraph_CustomConfig_t graph_custom_config;
  // Default use O3 for now.
  graph_custom_config = QNN_HTP_GRAPH_CUSTOM_CONFIG_INIT;
  graph_custom_config.option = QNN_HTP_GRAPH_CONFIG_OPTION_OPTIMIZATION;
  graph_custom_config.optimizationOption.type =
      QNN_HTP_GRAPH_OPTIMIZATION_TYPE_FINALIZE_OPTIMIZATION_FLAG;
  // Change to 2 if you want to use O2 (default).
  graph_custom_config.optimizationOption.floatValue = 3;

  static QnnGraph_Config_t graph_config;
  graph_config = QNN_GRAPH_CONFIG_INIT;
  graph_config.option = QNN_GRAPH_CONFIG_OPTION_CUSTOM;
  graph_config.customConfig = &graph_custom_config;

  static std::array<const QnnGraph_Config_t*, 2> result = {&graph_config,
                                                           nullptr};

  return absl::MakeSpan(result.data(), result.size());
}

absl::Span<const QnnGraph_Config_t*> GraphMapper::PickGraphConfigHeuristic() {
  if (qnn_.IsLegacySocModel()) {
    return GetLegacyGraphConfigs();
  } else {
    return GetDefaultGraphConfigs();
  }
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
    LITERT_RETURN_IF_ERROR(LegalizeAndRegister(litert_tensor, qnn_tensor));
    LITERT_RETURN_IF_ERROR(PushToScope(litert_tensor, qnn_tensor));
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
  LITERT_RETURN_IF_ERROR(LegalizeTensor(tensor, qnn_tensor));
  LITERT_RETURN_IF_ERROR(AssignTensorName(qnn_tensor));

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

  for (int i = 0; i < qnn_tensor.v2.rank; ++i) {
    LITERT_LOG(LITERT_INFO, "qnn_tensor dim[%d] = %d", i,
               qnn_tensor.v2.dimensions[i]);
  }

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
