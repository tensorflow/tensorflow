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
#include "third_party/qairt/include/QNN/QnnCommon.h"
#include "third_party/qairt/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_common.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/lite/experimental/lrt/c/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/lrt/vendors/qualcomm/qnn_manager.h"

#define LRT_RETURN_STATUS_IF_QNN_NOT_OK(expr) \
  if (QNN_SUCCESS != (expr)) {                \
    return kLrtStatusErrorNotFound;           \
  }

namespace lrt::qnn {

namespace {

// Get empty configurations for graph building.
inline absl::Span<const QnnGraph_Config_t*> GetDefaultGraphConfigs() {
  static const QnnGraph_Config_t* configs[] = {nullptr};
  return absl::MakeSpan(configs);
}

// Algorithm class for managing "scope" when mapping Lrt Subgraphs
// to QNN Graphs.
class GraphMapper {
 public:
  static LrtStatus MapGraph(QnnManager& qnn, LrtSubgraph subgraph,
                            absl::string_view qnn_graph_name);

 private:
  GraphMapper(LrtSubgraph subgraph, QnnManager* qnn)
      : subgraph_(subgraph), qnn_(qnn) {}

  // Can implementation handle given LrtSubgraph topology (see comment at bottom
  // of file).
  LrtStatus IsLrtSubgraphSupported();

  // Legalize given LrtTensors attributes into QNN Tensor registered with
  // QNN context. Result QNN Tensor is empty except for the canonical id
  // assigned by QNN Api.
  LrtStatus LegalizeAndRegister(LrtTensor lrt_tensor, Qnn_Tensor_t& qnn_tensor);

  //
  // CC Convienence Accessors
  //

  // Parse LrtSubgraph entities into usable types. Call this before
  // doing anything else.
  LrtStatus ParseLrtSubgraph();

  absl::Span<LrtTensor> LrtSubgraphInputs();
  absl::Span<LrtTensor> lrt_subgraph_inputs_;

  absl::Span<LrtTensor> LrtSubgraphOutputs();
  absl::Span<LrtTensor> lrt_subgraph_outputs_;

  absl::Span<LrtOp> LrtSubgraphOps();
  absl::Span<LrtOp> lrt_subgraph_ops_;

  LrtSubgraph Subgraph();
  LrtSubgraph subgraph_;

  //
  // Scope Management
  //

  // Maps evaluated tensors to their resolved QNN Tensor ID.
  std::unordered_map<LrtTensor, uint32_t>& CurrentScope();

  // Find ID associated with evaluated Lrt Tensor and add it to given
  // QNN Tensor.
  LrtStatus LookupInScope(LrtTensor lrt_tensor, Qnn_Tensor_t& qnn_tensor);

  // Adds new mapping to scope. All fields other than ID in given QNN Tensor are
  // cleared and its ID is added to "current_scope". Expects QNN Tensor has
  // already been registered with context.
  LrtStatus PushToScope(LrtTensor lrt_tensor, Qnn_Tensor_t& qnn_tensor);

  std::unordered_map<LrtTensor, uint32_t> current_scope_;

  //
  // QNN Sdk State
  //

  QnnManager& Qnn();
  QnnManager* qnn_;

  LrtStatus Finalize();
  LrtStatus InitQnnGraph(absl::string_view qnn_graph_name);
  Qnn_GraphHandle_t& QnnGraph();
  Qnn_GraphHandle_t qnn_graph_ = nullptr;

  //
  // Tensor Naming
  //

  // NOTE: QNN Tensors must be created with a unique name. This will ensure
  // uniqueness but will want to have more meaningful names in the future.
  LrtStatus AssignName(Qnn_Tensor_t& qnn_tensor);
  uint32_t cur_tensor_num_ = 0;
};

LrtStatus GraphMapper::AssignName(Qnn_Tensor_t& qnn_tensor) {
  char* name = nullptr;
  const int written = asprintf(&name, "Tensor_%d", cur_tensor_num_++);
  LRT_ENSURE(written != -1 && name != nullptr, kLrtStatusErrorNotFound,
             "Failed to make tensor name");
  qnn_tensor.v2.name = name;
  return kLrtStatusOk;
}

LrtSubgraph GraphMapper::Subgraph() { return subgraph_; }

absl::Span<LrtTensor> GraphMapper::LrtSubgraphInputs() {
  return lrt_subgraph_inputs_;
}

absl::Span<LrtTensor> GraphMapper::LrtSubgraphOutputs() {
  return lrt_subgraph_outputs_;
}

absl::Span<LrtOp> GraphMapper::LrtSubgraphOps() { return lrt_subgraph_ops_; }

std::unordered_map<LrtTensor, uint32_t>& GraphMapper::CurrentScope() {
  return current_scope_;
}

LrtStatus GraphMapper::LookupInScope(LrtTensor lrt_tensor,
                                     Qnn_Tensor_t& qnn_tensor) {
  // If we go in topological order, this should never happen. TODO: add
  // "internal error" status code.
  const auto qnn_id = CurrentScope().find(lrt_tensor);
  LRT_ENSURE(qnn_id != CurrentScope().end(), kLrtStatusErrorNotFound,
             "Couldn't find tensor in current_scope.");

  ResetTensor(qnn_tensor);
  qnn_tensor.v2.id = qnn_id->second;

  return kLrtStatusOk;
}

LrtStatus GraphMapper::PushToScope(LrtTensor lrt_tensor,
                                   Qnn_Tensor_t& qnn_tensor) {
  CurrentScope()[lrt_tensor] = MoveToId(qnn_tensor);
  return kLrtStatusOk;
}

QnnManager& GraphMapper::Qnn() { return *qnn_; }

Qnn_GraphHandle_t& GraphMapper::QnnGraph() { return qnn_graph_; }

LrtStatus GraphMapper::LegalizeAndRegister(LrtTensor lrt_tensor,
                                           Qnn_Tensor_t& qnn_tensor) {
  LRT_RETURN_STATUS_IF_NOT_OK(LegalizeTensor(lrt_tensor, qnn_tensor));
  LRT_RETURN_STATUS_IF_NOT_OK(AssignName(qnn_tensor));
  LRT_RETURN_STATUS_IF_QNN_NOT_OK(
      Qnn().Api()->tensorCreateGraphTensor(QnnGraph(), &qnn_tensor));

  return kLrtStatusOk;
}

LrtStatus GraphMapper::ParseLrtSubgraph() {
  LRT_ASSIGN_OR_RETURN_STATUS(auto inputs,
                              graph_tools::GetSubgraphInputs(Subgraph()));
  lrt_subgraph_inputs_ =
      absl::MakeSpan(const_cast<LrtTensor*>(inputs.data()), inputs.size());

  LRT_ASSIGN_OR_RETURN_STATUS(auto outputs,
                              graph_tools::GetSubgraphOutputs(Subgraph()));
  lrt_subgraph_outputs_ =
      absl::MakeSpan(const_cast<LrtTensor*>(outputs.data()), outputs.size());

  LRT_ASSIGN_OR_RETURN_STATUS(auto ops,
                              graph_tools::GetSubgraphOps(Subgraph()));
  lrt_subgraph_ops_ =
      absl::MakeSpan(const_cast<LrtOp*>(ops.data()), ops.size());

  return kLrtStatusOk;
}

LrtStatus GraphMapper::IsLrtSubgraphSupported() {
  LRT_ENSURE_SUPPORTED(LrtSubgraphInputs().size() == 2,
                       "Only subgraphs with 2 inputs currently supported.");

  LRT_ENSURE_SUPPORTED(LrtSubgraphOutputs().size() == 1,
                       "Only subgraphs with 1 output currently supported.");

  LRT_ENSURE_SUPPORTED(LrtSubgraphOps().size() == 1,
                       "Only subgraphs with 1 op currently supported.");

  return kLrtStatusOk;
}

LrtStatus GraphMapper::InitQnnGraph(absl::string_view qnn_graph_name) {
  LRT_RETURN_STATUS_IF_QNN_NOT_OK(
      Qnn().Api()->graphCreate(Qnn().ContextHandle(), qnn_graph_name.data(),
                               GetDefaultGraphConfigs().data(), &QnnGraph()));
  return kLrtStatusOk;
}

LrtStatus GraphMapper::Finalize() {
  LRT_RETURN_STATUS_IF_QNN_NOT_OK(
      Qnn().Api()->graphFinalize(QnnGraph(), nullptr, nullptr));
  return kLrtStatusOk;
}

LrtStatus GraphMapper::MapGraph(QnnManager& qnn, LrtSubgraph subgraph,
                                absl::string_view qnn_graph_name) {
  GraphMapper graph_mapper(subgraph, &qnn);
  LRT_RETURN_STATUS_IF_NOT_OK(graph_mapper.ParseLrtSubgraph());
  LRT_RETURN_STATUS_IF_NOT_OK(graph_mapper.IsLrtSubgraphSupported());
  LRT_RETURN_STATUS_IF_NOT_OK(graph_mapper.InitQnnGraph(qnn_graph_name));

  //
  // Legalize subgraph inputs and update tensors in scope
  //

  for (auto subgraph_input : graph_mapper.LrtSubgraphInputs()) {
    Qnn_Tensor_t qnn_subgraph_input = BuildInputTensor();

    LRT_RETURN_STATUS_IF_NOT_OK(
        graph_mapper.LegalizeAndRegister(subgraph_input, qnn_subgraph_input));

    LRT_RETURN_STATUS_IF_NOT_OK(
        graph_mapper.PushToScope(subgraph_input, qnn_subgraph_input));
  }

  //
  // Toplogically traverse graph, legalizing and updating tensors in scope
  //

  // TODO: Drive traversal here.

  {
    LrtOp op = graph_mapper.LrtSubgraphOps()[0];

    Qnn_OpConfig_t qnn_op = BuildDefaultOp();
    // TODO: Add optional support for "validateOpConfig".
    LRT_RETURN_STATUS_IF_NOT_OK(LegalizeOp(op, qnn_op));

    // Look up op input tensors in scope

    LRT_ASSIGN_OR_RETURN_STATUS(auto op_ins, ::graph_tools::GetOpIns(op));
    LRT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, op_ins.size(), QNN_TENSOR_INIT);

    Qnn_Tensor_t* cur_qnn_op_in = qnn_op_ins;
    for (auto op_in : op_ins) {
      LRT_RETURN_STATUS_IF_NOT_OK(
          graph_mapper.LookupInScope(op_in, *cur_qnn_op_in));
      ++cur_qnn_op_in;
    }

    // Legalize op outputs and update scope

    LRT_ASSIGN_OR_RETURN_STATUS(auto op_outs, ::graph_tools::GetOpOuts(op));
    LRT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, op_outs.size(), QNN_TENSOR_INIT);

    Qnn_Tensor_t* cur_qnn_op_out = qnn_op_outs;
    for (auto op_out : op_outs) {
      LRT_RETURN_STATUS_IF_NOT_OK(
          graph_mapper.LegalizeAndRegister(op_out, *cur_qnn_op_out));
      LRT_RETURN_STATUS_IF_NOT_OK(
          graph_mapper.PushToScope(op_out, *cur_qnn_op_out));
      ++cur_qnn_op_out;
    }

    qnn_op.v1.numOfInputs = op_ins.size();
    qnn_op.v1.inputTensors = qnn_op_ins;

    qnn_op.v1.numOfOutputs = op_outs.size();
    qnn_op.v1.outputTensors = qnn_op_outs;

    LRT_RETURN_STATUS_IF_QNN_NOT_OK(graph_mapper.Qnn().Api()->graphAddNode(
        graph_mapper.QnnGraph(), qnn_op));
  }

  // NOTE: Subgraph outputs are fully configured during "LegalizeAndRegister".

  LRT_RETURN_STATUS_IF_QNN_NOT_OK(graph_mapper.Finalize());

  return kLrtStatusOk;
}

}  // namespace

//===----------------------------------------------------------------------===//
//
//                                              [WIP] LRT SUBGRAPH -> QNN GRAPH
//
// Core driver for IR translation. Traverses Lrt Subgraph, iteratively
// "legalizing" (mapping) Lrt entities to their QNN counterpart.
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
// * Store mapping between already evaluated LrtTensors and their
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

LrtStatus ComposeGraph(QnnManager& qnn, LrtSubgraph subgraph,
                       absl::string_view qnn_graph_name) {
  LRT_RETURN_STATUS_IF_NOT_OK(
      GraphMapper::MapGraph(qnn, subgraph, qnn_graph_name));
  return kLrtStatusOk;
}

}  // namespace lrt::qnn
