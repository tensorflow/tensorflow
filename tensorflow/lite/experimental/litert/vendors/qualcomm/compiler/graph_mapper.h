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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_GRAPH_MAPPER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_GRAPH_MAPPER_H_

#include <cstdint>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnGraph.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

// Algorithm class for managing "scope" when mapping litert Subgraphs
// to QNN Graphs.
class GraphMapper {
 public:
  GraphMapper(LiteRtSubgraph subgraph, QnnManager& qnn,
              Qnn_ContextHandle_t context_handle)
      : subgraph_(Subgraph(subgraph)),
        qnn_(qnn),
        context_handle_(context_handle) {}

  // Legalize given LiteRtTensors attributes into QNN Tensor registered with
  // QNN context. Result QNN Tensor is empty except for the canonical id
  // assigned by QNN Api.
  LiteRtStatus LegalizeAndRegister(LiteRtTensor litert_tensor,
                                   Qnn_Tensor_t& qnn_tensor);

  // Find ID associated with evaluated litert Tensor and add it to given
  // QNN Tensor.
  LiteRtStatus LookupInScope(LiteRtTensor litert_tensor,
                             Qnn_Tensor_t& qnn_tensor);

  // Adds new mapping to scope. All fields other than ID in given QNN Tensor are
  // cleared and its ID is added to "current_scope". Expects QNN Tensor has
  // already been registered with context.
  LiteRtStatus PushToScope(LiteRtTensor litert_tensor,
                           Qnn_Tensor_t& qnn_tensor);

  // NOTE: QNN Tensors must be created with a unique name. This will ensure
  // uniqueness but will want to have more meaningful names in the future.
  LiteRtStatus AssignTensorName(Qnn_Tensor_t& qnn_tensor);

  // QNN Sdk Accessors
  QnnManager& Qnn();
  Qnn_GraphHandle_t& QnnGraph();

  // CC Convenience Accessors
  const Subgraph& Graph() const { return subgraph_; }

  // Accessor for current scope.
  // Since each QNN Tensor needs to have a unique name globally within each QNN
  // context, we maintain "Current scope", which is a map of evaluated
  // LiteRtTensors to their resolved QNN Tensor ID.
  absl::flat_hash_map<LiteRtTensor, uint32_t>& CurrentScope();

  // Can implementation handle given LiteRtSubgraph topology (see comment at
  // bottom of file).
  LiteRtStatus IsLiteRtSubgraphSupported();

  // Initialize QNN Graph with given name. Call this after parsing
  // LiteRtSubgraph.
  LiteRtStatus InitQnnGraph(absl::string_view qnn_graph_name);

  // Finalize QNN Graph. Call this after all ops have been mapped.
  LiteRtStatus Finalize();

  // Pick graph config based on subgraph.
  absl::Span<const QnnGraph_Config_t*> PickGraphConfigHeuristic();

  inline void RegisterOutput(LiteRtTensor litert_tensor) {
    graph_outpus_.insert(litert_tensor);
  }

 private:
  const Subgraph subgraph_;

  // Set of all outputs of the graph.
  absl::flat_hash_set<LiteRtTensor> graph_outpus_;

  // Maps evaluated tensors to their resolved QNN Tensor ID.
  absl::flat_hash_map<LiteRtTensor, uint32_t> current_scope_;

  //
  // QNN Sdk State
  //
  QnnManager& qnn_;
  Qnn_ContextHandle_t context_handle_;
  Qnn_GraphHandle_t qnn_graph_ = nullptr;

  //
  // Tensor Naming
  //

  uint32_t cur_tensor_num_ = 0;
};

}  // namespace litert::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_GRAPH_MAPPER_H_
