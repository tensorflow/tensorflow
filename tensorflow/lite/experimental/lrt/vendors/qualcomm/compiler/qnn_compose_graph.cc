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

#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/compiler/qnn_compose_graph.h"

#include <alloca.h>
#include <stdio.h>

#include <cstdint>
#include <unordered_map>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_support.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn_manager.h"

#define LITERT_RETURN_STATUS_IF_QNN_NOT_OK(expr) \
  if (QNN_SUCCESS != (expr)) {                   \
    return kLiteRtStatusErrorNotFound;           \
  }

namespace litert::qnn {

namespace {

// Get empty configurations for graph building.
inline absl::Span<const QnnGraph_Config_t*> GetDefaultGraphConfigs() {
  static const QnnGraph_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

// Algorithm class for managing "scope" when mapping LiteRt Subgraphs
// to QNN Graphs.
class GraphMapper {
 public:
  static LiteRtStatus MapGraph(QnnManager& qnn, LiteRtSubgraph subgraph,
                               absl::string_view qnn_graph_name);

 private:
  GraphMapper(LiteRtSubgraph subgraph, QnnManager* qnn)
      : subgraph_(subgraph), qnn_(qnn) {}

  // Can implementation handle given LiteRtSubgraph topology (see comment at
  // bottom of file).
  LiteRtStatus IsLiteRtSubgraphSupported();

  // Legalize given LiteRtTensors attributes into QNN Tensor registered with
  // QNN context. Result QNN Tensor is empty except for the canonical id
  // assigned by QNN Api.
  LiteRtStatus LegalizeAndRegister(LiteRtTensor litert_tensor,
                                   Qnn_Tensor_t& qnn_tensor);

  //
  // CC Convienence Accessors
  //

  // Parse LiteRtSubgraph entities into usable types. Call this before
  // doing anything else.
  LiteRtStatus ParseLiteRtSubgraph();

  absl::Span<LiteRtTensor> LiteRtSubgraphInputs();
  absl::Span<LiteRtTensor> litert_subgraph_inputs_;

  absl::Span<LiteRtTensor> LiteRtSubgraphOutputs();
  absl::Span<LiteRtTensor> litert_subgraph_outputs_;

  absl::Span<LiteRtOp> LiteRtSubgraphOps();
  absl::Span<LiteRtOp> litert_subgraph_ops_;

  LiteRtSubgraph Subgraph();
  LiteRtSubgraph subgraph_;

  //
  // Scope Management
  //

  // Maps evaluated tensors to their resolved QNN Tensor ID.
  std::unordered_map<LiteRtTensor, uint32_t>& CurrentScope();

  // Find ID associated with evaluated LiteRt Tensor and add it to given
  // QNN Tensor.
  LiteRtStatus LookupInScope(LiteRtTensor litert_tensor,
                             Qnn_Tensor_t& qnn_tensor);

  // Adds new mapping to scope. All fields other than ID in given QNN Tensor are
  // cleared and its ID is added to "current_scope". Expects QNN Tensor has
  // already been registered with context.
  LiteRtStatus PushToScope(LiteRtTensor litert_tensor,
                           Qnn_Tensor_t& qnn_tensor);

  std::unordered_map<LiteRtTensor, uint32_t> current_scope_;

  //
  // QNN Sdk State
  //

  QnnManager& Qnn();
  QnnManager* qnn_;

  LiteRtStatus Finalize();
  LiteRtStatus InitQnnGraph(absl::string_view qnn_graph_name);
  Qnn_GraphHandle_t& QnnGraph();
  Qnn_GraphHandle_t qnn_graph_ = nullptr;

  //
  // Tensor Naming
  //

  // NOTE: QNN Tensors must be created with a unique name. This will ensure
  // uniqueness but will want to have more meaningful names in the future.
  LiteRtStatus AssignName(Qnn_Tensor_t& qnn_tensor);
  uint32_t cur_tensor_num_ = 0;
};

LiteRtStatus GraphMapper::AssignName(Qnn_Tensor_t& qnn_tensor) {
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

std::unordered_map<LiteRtTensor, uint32_t>& GraphMapper::CurrentScope() {
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

QnnManager& GraphMapper::Qnn() { return *qnn_; }

Qnn_GraphHandle_t& GraphMapper::QnnGraph() { return qnn_graph_; }

LiteRtStatus GraphMapper::LegalizeAndRegister(LiteRtTensor litert_tensor,
                                              Qnn_Tensor_t& qnn_tensor) {
  LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeTensor(litert_tensor, qnn_tensor));
  LITERT_RETURN_STATUS_IF_NOT_OK(AssignName(qnn_tensor));
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      Qnn().Api()->tensorCreateGraphTensor(QnnGraph(), &qnn_tensor));

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
  LITERT_ENSURE_SUPPORTED(LiteRtSubgraphInputs().size() == 2,
                          "Only subgraphs with 2 inputs currently supported.");

  LITERT_ENSURE_SUPPORTED(LiteRtSubgraphOutputs().size() == 1,
                          "Only subgraphs with 1 output currently supported.");

  LITERT_ENSURE_SUPPORTED(LiteRtSubgraphOps().size() == 1,
                          "Only subgraphs with 1 op currently supported.");

  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::InitQnnGraph(absl::string_view qnn_graph_name) {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      Qnn().Api()->graphCreate(Qnn().ContextHandle(), qnn_graph_name.data(),
                               GetDefaultGraphConfigs().data(), &QnnGraph()));
  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::Finalize() {
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      Qnn().Api()->graphFinalize(QnnGraph(), nullptr, nullptr));
  return kLiteRtStatusOk;
}

LiteRtStatus GraphMapper::MapGraph(QnnManager& qnn, LiteRtSubgraph subgraph,
                                   absl::string_view qnn_graph_name) {
  GraphMapper graph_mapper(subgraph, &qnn);
  LITERT_RETURN_STATUS_IF_NOT_OK(graph_mapper.ParseLiteRtSubgraph());
  LITERT_RETURN_STATUS_IF_NOT_OK(graph_mapper.IsLiteRtSubgraphSupported());
  LITERT_RETURN_STATUS_IF_NOT_OK(graph_mapper.InitQnnGraph(qnn_graph_name));

  //
  // Legalize subgraph inputs and update tensors in scope
  //

  for (auto subgraph_input : graph_mapper.LiteRtSubgraphInputs()) {
    Qnn_Tensor_t qnn_subgraph_input = BuildInputTensor();

    LITERT_RETURN_STATUS_IF_NOT_OK(
        graph_mapper.LegalizeAndRegister(subgraph_input, qnn_subgraph_input));

    LITERT_RETURN_STATUS_IF_NOT_OK(
        graph_mapper.PushToScope(subgraph_input, qnn_subgraph_input));
  }

  //
  // Toplogically traverse graph, legalizing and updating tensors in scope
  //

  // TODO: Drive traversal here.

  {
    LiteRtOp op = graph_mapper.LiteRtSubgraphOps()[0];

    Qnn_OpConfig_t qnn_op = BuildDefaultOp();
    // TODO: Add optional support for "validateOpConfig".
    LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeOp(op, qnn_op));

    // Look up op input tensors in scope

    LITERT_ASSIGN_OR_RETURN_STATUS(auto op_ins, ::graph_tools::GetOpIns(op));
    LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, op_ins.size(),
                       QNN_TENSOR_INIT);

    Qnn_Tensor_t* cur_qnn_op_in = qnn_op_ins;
    for (auto op_in : op_ins) {
      LITERT_RETURN_STATUS_IF_NOT_OK(
          graph_mapper.LookupInScope(op_in, *cur_qnn_op_in));
      ++cur_qnn_op_in;
    }

    // Legalize op outputs and update scope

    LITERT_ASSIGN_OR_RETURN_STATUS(auto op_outs, ::graph_tools::GetOpOuts(op));
    LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, op_outs.size(),
                       QNN_TENSOR_INIT);

    Qnn_Tensor_t* cur_qnn_op_out = qnn_op_outs;
    for (auto op_out : op_outs) {
      LITERT_RETURN_STATUS_IF_NOT_OK(
          graph_mapper.LegalizeAndRegister(op_out, *cur_qnn_op_out));
      LITERT_RETURN_STATUS_IF_NOT_OK(
          graph_mapper.PushToScope(op_out, *cur_qnn_op_out));
      ++cur_qnn_op_out;
    }

    qnn_op.v1.numOfInputs = op_ins.size();
    qnn_op.v1.inputTensors = qnn_op_ins;

    qnn_op.v1.numOfOutputs = op_outs.size();
    qnn_op.v1.outputTensors = qnn_op_outs;

    LITERT_RETURN_STATUS_IF_QNN_NOT_OK(graph_mapper.Qnn().Api()->graphAddNode(
        graph_mapper.QnnGraph(), qnn_op));
  }

  // NOTE: Subgraph outputs are fully configured during "LegalizeAndRegister".

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(graph_mapper.Finalize());

  return kLiteRtStatusOk;
}

}  // namespace

//===----------------------------------------------------------------------===//
//
//                                              [WIP] LRT SUBGRAPH -> QNN GRAPH
//
// Core driver for IR translation. Traverses LiteRt Subgraph, iteratively
// "legalizing" (mapping) LiteRt entities to their QNN counterpart.
//
// APPROACH:
//
// Currently demoing by just handling a simple case where there is one
// partitions and the partitions is as follows:
//
// func(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>)
//   %0 = tfl.mul(%arg0, %arg1)
//   return %0
//
// To support the general case we will need a driver loop that either
// traverses input recursively through edges or just iterates topologically.
// Currently we just have only implemented n=1.
//
// The algorithm is pretty straightforward:
// * Store mapping between already evaluated LiteRtTensors and their
//   newly constructed Qnn Tensor counterpart.
// * Look up QNN Tensors when setting QNN Op inputs.
// * Add new QNN Tensor when setting QNN Op outputs.
//
// NOTES ON QNN API:
//
// After QNN Tensors are registered in the context, they need only
// be stored as their ID. QNN Tensor and "id" : uint32_t are used
// interchangeably.
//
//===----------------------------------------------------------------------===//

LiteRtStatus ComposeGraph(QnnManager& qnn, LiteRtSubgraph subgraph,
                          absl::string_view qnn_graph_name) {
  LITERT_RETURN_STATUS_IF_NOT_OK(
      GraphMapper::MapGraph(qnn, subgraph, qnn_graph_name));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
