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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/qnn_compose_graph.h"

#include <alloca.h>
#include <stdio.h>

#include <memory>
#include <vector>

#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnCommon.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/add_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/batch_matmul_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/cast_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/concatenation_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/cos_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/div_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/fully_connected_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/mul_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/reshape_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/rsqrt_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/select_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/sin_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/slice_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/softmax_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/sub_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/sum_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/tanh_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/transpose_op_legalization.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

namespace {

LiteRtStatus RegisterAllLegalizations(
    std::vector<std::unique_ptr<Legalization>>& legalizations) {
  legalizations.push_back(MulOpLegalization::Create());
  legalizations.push_back(BatchMatmulOpLegalization::Create());
  legalizations.push_back(SliceOpLegalization::Create());
  legalizations.push_back(AddOpLegalization::Create());
  legalizations.push_back(DivOpLegalization::Create());
  legalizations.push_back(RsqrtOpLegalization::Create());
  legalizations.push_back(TanhOpLegalization::Create());
  legalizations.push_back(SubOpLegalization::Create());
  legalizations.push_back(ReshapeOpLegalization::Create());
  legalizations.push_back(SumOpLegalization::Create());
  legalizations.push_back(ConcatenationOpLegalization::Create());
  legalizations.push_back(SoftmaxOpLegalization::Create());
  legalizations.push_back(CastOpLegalization::Create());
  legalizations.push_back(TransposeOpLegalization::Create());
  legalizations.push_back(SinOpLegalization::Create());
  legalizations.push_back(CosOpLegalization::Create());
  legalizations.push_back(SelectOpLegalization::Create());
  legalizations.push_back(FullyConnectedOpLegalization::Create());
  LITERT_LOG(LITERT_INFO, "Scheduling %lu legalizations", legalizations.size());
  return kLiteRtStatusOk;
}

LiteRtStatus MapGraph(QnnManager& qnn, Qnn_ContextHandle_t context_handle,
                      LiteRtSubgraph subgraph,
                      absl::string_view qnn_graph_name) {
  // Register legalizations.
  std::vector<std::unique_ptr<Legalization>> legalizations;
  LITERT_RETURN_STATUS_IF_NOT_OK(RegisterAllLegalizations(legalizations));

  GraphMapper graph_mapper(subgraph, qnn, context_handle);
  LITERT_RETURN_STATUS_IF_NOT_OK(graph_mapper.IsLiteRtSubgraphSupported());
  LITERT_RETURN_STATUS_IF_NOT_OK(graph_mapper.InitQnnGraph(qnn_graph_name));

  //
  // Legalize subgraph inputs and update tensors in scope
  //

  for (const auto& subgraph_input : graph_mapper.Graph().Inputs()) {
    Qnn_Tensor_t qnn_subgraph_input = BuildInputTensor();

    LITERT_RETURN_STATUS_IF_NOT_OK(graph_mapper.LegalizeAndRegister(
        subgraph_input.Get(), qnn_subgraph_input));

    LITERT_RETURN_STATUS_IF_NOT_OK(
        graph_mapper.PushToScope(subgraph_input.Get(), qnn_subgraph_input));
  }

  //
  // Topologically traverse graph, legalizing and updating tensors in scope
  //

  for (const auto& op : graph_mapper.Graph().Ops()) {
    Qnn_OpConfig_t qnn_op = BuildDefaultOp();
    for (auto it = legalizations.begin(); it != legalizations.end(); ++it) {
      LITERT_RETURN_STATUS_IF_NOT_OK_OR_NOT_MATCHED(
          (*it)->LegalizeOp(op, qnn_op, graph_mapper));
    }
  }

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(graph_mapper.Finalize());

  return kLiteRtStatusOk;
}

}  // namespace

//===----------------------------------------------------------------------===//
//
//                                           [WIP] LiteRT SUBGRAPH -> QNN GRAPH
//
// Core driver for IR translation. Traverses LiteRt Subgraph, iteratively
// "legalizing" (mapping) LiteRt entities to their QNN counterpart.
//
// APPROACH:
//
// To support the general case we will need a driver loop that either
// traverses input recursively through edges or just iterates topologically.
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

LiteRtStatus ComposeGraph(QnnManager& qnn, Qnn_ContextHandle_t context_handle,
                          LiteRtSubgraph subgraph,
                          absl::string_view qnn_graph_name) {
  LITERT_RETURN_STATUS_IF_NOT_OK(
      MapGraph(qnn, context_handle, subgraph, qnn_graph_name));
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
