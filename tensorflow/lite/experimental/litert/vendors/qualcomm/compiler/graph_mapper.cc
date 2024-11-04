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

#include <cstdint>
#include <string>
#include <unordered_map>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnGraph.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_support.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

// Get empty configurations for graph building.
inline absl::Span<const QnnGraph_Config_t*> GetDefaultGraphConfigs() {
  static const QnnGraph_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

LiteRtStatus GraphMapper::AssignTensorName(Qnn_Tensor_t& qnn_tensor) {
  char* name = nullptr;
  const int written = asprintf(&name, "Tensor_%d", cur_tensor_num_++);
  LITERT_ENSURE(written != -1 && name != nullptr, kLiteRtStatusErrorNotFound,
                "Failed to make tensor name");
  qnn_tensor.v2.name = name;
  return kLiteRtStatusOk;
}

LiteRtSubgraph GraphMapper::Subgraph() { return subgraph_; }

absl::Span<LiteRtTensor> GraphMapper::LiteRtSubgraphInputs() {
  return litert_subgraph_inputs_;
}

absl::Span<LiteRtTensor> GraphMapper::LiteRtSubgraphOutputs() {
  return litert_subgraph_outputs_;
}

absl::Span<LiteRtOp> GraphMapper::LiteRtSubgraphOps() {
  return litert_subgraph_ops_;
}

absl::flat_hash_map<LiteRtTensor, uint32_t>& GraphMapper::CurrentScope() {
  return current_scope_;
}

LiteRtStatus GraphMapper::LookupInScope(LiteRtTensor litert_tensor,
                                        Qnn_Tensor_t& qnn_tensor) {
  // If we go in topological order, this should never happen. TODO: add
  // "internal error" status code.
  const auto qnn_id = CurrentScope().find(litert_tensor);
  LITERT_ENSURE(qnn_id != CurrentScope().end(), kLiteRtStatusErrorNotFound,
                "Couldn't find tensor in current_scope.");

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
  LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeTensor(litert_tensor, qnn_tensor));
  LITERT_RETURN_STATUS_IF_NOT_OK(AssignTensorName(qnn_tensor));
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->tensorCreateGraphTensor(QnnGraph(), &qnn_tensor));

  LITERT_LOG(LITERT_INFO, "Legalized and registered tensor %d",
             qnn_tensor.v2.id);

  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::ParseLiteRtSubgraph() {
  LITERT_ASSIGN_OR_RETURN_STATUS(auto inputs,
                                 graph_tools::GetSubgraphInputs(Subgraph()));
  litert_subgraph_inputs_ =
      absl::MakeSpan(const_cast<LiteRtTensor*>(inputs.data()), inputs.size());

  LITERT_ASSIGN_OR_RETURN_STATUS(auto outputs,
                                 graph_tools::GetSubgraphOutputs(Subgraph()));
  litert_subgraph_outputs_ =
      absl::MakeSpan(const_cast<LiteRtTensor*>(outputs.data()), outputs.size());

  LITERT_ASSIGN_OR_RETURN_STATUS(auto ops,
                                 graph_tools::GetSubgraphOps(Subgraph()));
  litert_subgraph_ops_ =
      absl::MakeSpan(const_cast<LiteRtOp*>(ops.data()), ops.size());

  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::IsLiteRtSubgraphSupported() {
  LITERT_ENSURE_SUPPORTED(
      LiteRtSubgraphInputs().size() < 4,
      "Only subgraphs with less than 4 inputs currently supported.");

  LITERT_ENSURE_SUPPORTED(LiteRtSubgraphOutputs().size() == 1,
                          "Only subgraphs with 1 output currently supported.");

  LITERT_ENSURE_SUPPORTED(LiteRtSubgraphOps().size() == 1,
                          "Only subgraphs with 1 op currently supported.");

  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::InitQnnGraph(absl::string_view qnn_graph_name) {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->graphCreate(context_handle_, qnn_graph_name.data(),
                              GetDefaultGraphConfigs().data(), &QnnGraph()));
  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::Finalize() {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      qnn_.Api()->graphFinalize(QnnGraph(), nullptr, nullptr));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
