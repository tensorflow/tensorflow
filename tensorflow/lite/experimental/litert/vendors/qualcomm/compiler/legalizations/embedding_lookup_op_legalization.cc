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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/embedding_lookup_op_legalization.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

namespace litert::qnn {

static constexpr absl::string_view kQnnEmbeddingLookupOpTypeName = "Gather";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kEmbeddingLookupOpFmt =
    "embedding_lookup_%d";

static constexpr int kReduceEmbeddingLookupOpOutputSize = 1;
static constexpr int kReduceEmbeddingLookupOpParamSize = 1;

static constexpr int kEmbeddingLookupOpTableInputIndex = 1;
static constexpr int kEmbeddingLookupOpLookipInputIndex = 0;
static constexpr int kQnnGatherOpTableInputIndex = 0;
static constexpr int kQnnGatherOpLookupInputIndex = 1;

LiteRtStatus EmbeddingLookupOpLegalization::LegalizeOp(
    const Op& src, Qnn_OpConfig_t& dest, GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflEmbeddingLookup) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  DumpLegalization(*src.Get());
  std::string op_name = absl::StrFormat(kEmbeddingLookupOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      SetOpInfo(op_name.c_str(), kDefaultQnnOpPackageName.data(),
                kQnnEmbeddingLookupOpTypeName.data(), dest));

  // Look up op input tensors in scope.
  const auto op_ins = src.Inputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, op_ins.size(), QNN_TENSOR_INIT);

  LITERT_RETURN_STATUS_IF_NOT_OK(graph_mapper.LookupInScope(
      op_ins[kEmbeddingLookupOpLookipInputIndex].Get(),
      qnn_op_ins[kQnnGatherOpLookupInputIndex]));
  LITERT_RETURN_STATUS_IF_NOT_OK(graph_mapper.LookupInScope(
      op_ins[kEmbeddingLookupOpTableInputIndex].Get(),
      qnn_op_ins[kQnnGatherOpTableInputIndex]));

  // QNN embedding_lookup op expects 1 output tensor.
  const auto op_outs = src.Outputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs,
                     kReduceEmbeddingLookupOpOutputSize, QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.LegalizeAndRegister(op_outs.front().Get(), qnn_op_outs[0]));
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.PushToScope(op_outs.front().Get(), qnn_op_outs[0]));

  // Construct the scalar "axis" param.
  Qnn_Param_t axis_param = BuildDefaultParam();
  axis_param.paramType = QNN_PARAMTYPE_SCALAR;
  axis_param.name = "axis";
  Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
  axis_scalar.dataType = QNN_DATATYPE_INT_32;
  // Embedding lookup op expects axis to always be 0.
  axis_scalar.int32Value = 0;
  axis_param.scalarParam = axis_scalar;

  Qnn_Param_t embedding_lookup_params[] = {axis_param};
  dest.v1.inputTensors = qnn_op_ins;
  dest.v1.numOfInputs = op_ins.size();
  dest.v1.outputTensors = qnn_op_outs;
  dest.v1.numOfOutputs = kReduceEmbeddingLookupOpOutputSize;
  dest.v1.numOfParams = kReduceEmbeddingLookupOpParamSize;
  dest.v1.params = embedding_lookup_params;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->graphAddNode(graph_mapper.QnnGraph(), dest));

  LITERT_LOG(LITERT_INFO, "Legalized embedding_lookup op", "");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
