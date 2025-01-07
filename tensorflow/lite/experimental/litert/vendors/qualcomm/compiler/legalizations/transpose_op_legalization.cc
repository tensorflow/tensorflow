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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/transpose_op_legalization.h"

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
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

namespace litert::qnn {

static constexpr absl::string_view kQnnTransposeOpTypeName = "Transpose";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kTransposeOpFmt = "transpose_%d";

static constexpr int kTransposeOpInputSize = 1;
static constexpr int kTransposeOpOutputSize = 1;
static constexpr int kTransposeOpParamSize = 1;
static constexpr int kTransposeOpParamRank = 1;

LiteRtStatus TransposeOpLegalization::LegalizeOp(const Op& src,
                                                 Qnn_OpConfig_t& dest,
                                                 GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflTranspose) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  DumpLegalization(*src.Get());
  std::string op_name = absl::StrFormat(kTransposeOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      SetOpInfo(op_name.c_str(), kDefaultQnnOpPackageName.data(),
                kQnnTransposeOpTypeName.data(), dest));

  // QNN transpose op expects 1 input tensor.
  const auto op_ins = src.Inputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, kTransposeOpInputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.LookupInScope(op_ins.front().Get(), qnn_op_ins[0]));

  // QNN transpose op expects 1 output tensor.
  const auto op_outs = src.Outputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, kTransposeOpOutputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.LegalizeAndRegister(op_outs.front().Get(), qnn_op_outs[0]));
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.PushToScope(op_outs.front().Get(), qnn_op_outs[0]));

  // Prepare QNN transpose parameters.
  auto perm = Tensor(op_ins.at(1).Get());

  // Check if src_axes are weights tensors.
  if (!perm.HasWeights()) {
    return kLiteRtStatusErrorInvalidLegalization;
  }
  auto perm_data = perm.Weights().Bytes();
  int32_t dest_axes_size = perm_data.size();
  Qnn_ClientBuffer_t perm_tensor_client_buf = BuildDefaultClientBuffer();
  perm_tensor_client_buf.data = (void*)perm_data.data();
  perm_tensor_client_buf.dataSize = dest_axes_size;

  // Construct the const tensor "perm".
  Qnn_Tensor_t perm_tensor = BuildDefaultTensor();
  graph_mapper.AssignTensorName(perm_tensor);
  perm_tensor.v2.dataType = QNN_DATATYPE_INT_32;
  perm_tensor.v2.type = QNN_TENSOR_TYPE_STATIC;
  perm_tensor.v2.rank = kTransposeOpParamRank;
  perm_tensor.v2.dimensions = new uint32_t[kTransposeOpParamRank];
  perm_tensor.v2.dimensions[0] = dest_axes_size;
  perm_tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  perm_tensor.v2.clientBuf = perm_tensor_client_buf;
  perm_tensor.v2.isDynamicDimensions = nullptr;
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->tensorCreateGraphTensor(graph_mapper.QnnGraph(),
                                                        &perm_tensor));

  Qnn_Param_t perm_param = BuildDefaultParam();
  perm_param.paramType = QNN_PARAMTYPE_TENSOR;
  perm_param.name = "perm";
  perm_param.tensorParam = perm_tensor;

  Qnn_Param_t transpose_params[] = {perm_param};
  dest.v1.inputTensors = qnn_op_ins;
  dest.v1.numOfInputs = kTransposeOpInputSize;
  dest.v1.outputTensors = qnn_op_outs;
  dest.v1.numOfOutputs = kTransposeOpOutputSize;
  dest.v1.numOfParams = kTransposeOpParamSize;
  dest.v1.params = transpose_params;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->graphAddNode(graph_mapper.QnnGraph(), dest));

  LITERT_LOG(LITERT_INFO, "Legalized transpose op", "");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
