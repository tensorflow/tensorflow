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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/quantize_op_legalization.h"

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

static constexpr absl::string_view kQnnConvertOpTypeName = "Convert";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kConvertOpFmt = "convert_%d";

static constexpr absl::string_view kQnnQuantizeOpTypeName = "Quantize";
static constexpr absl::string_view kQuantizeOpFmt = "quantize_%d";

static constexpr absl::string_view kQnnDequantizeOpTypeName = "Dequantize";
static constexpr absl::string_view kDequantizeOpFmt = "dequantize_%d";

static constexpr int kQuantizeOpInputSize = 1;
static constexpr int kQuantizeOpOutputSize = 1;

// Option 1: dequantize + quantize
LiteRtStatus QuantizeOpLegalization::DequantQauntLeaglization(
    const litert::Op& src, Qnn_OpConfig_t& dest, GraphMapper& graph_mapper) {
  // Add dequantize op to graph.
  Qnn_OpConfig_t dequantize_op = BuildDefaultOp();
  std::string dequantize_op_name =
      absl::StrFormat(kDequantizeOpFmt, op_counter_);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      SetOpInfo(dequantize_op_name.c_str(), kDefaultQnnOpPackageName.data(),
                kQnnDequantizeOpTypeName.data(), dequantize_op));
  // Look up op input tensors in scope.
  const auto op_ins = src.Inputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, kQuantizeOpInputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.LookupInScope(op_ins.front().Get(), qnn_op_ins[0]));

  // Create output tensor for dequantize op.
  Qnn_Tensor_t dequantize_op_out = BuildDefaultTensor();
  SetResultTensorAttrs(dequantize_op_out);
  std::string dequantize_op_out_name =
      absl::StrFormat("%s_out", dequantize_op_name);
  dequantize_op_out.v2.name = dequantize_op_out_name.c_str();
  dequantize_op_out.v2.dataType = QNN_DATATYPE_FLOAT_32;
  // legalize shape info
  dequantize_op_out.v2.rank = qnn_op_ins[0].v2.rank;
  dequantize_op_out.v2.dimensions = new uint32_t[dequantize_op_out.v2.rank];
  for (int i = 0; i < dequantize_op_out.v2.rank; ++i) {
    const auto src_dim = qnn_op_ins[0].v2.dimensions[i];
    LITERT_ENSURE(src_dim >= 1, kLiteRtStatusErrorInvalidArgument,
                  "Cannot pass dim < 1 to QNN Tensor.");
    dequantize_op_out.v2.dimensions[i] = src_dim;
  }
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->tensorCreateGraphTensor(graph_mapper.QnnGraph(),
                                                        &dequantize_op_out));
  LITERT_LOG(LITERT_INFO, "Add dequantize op output tensor to Qnn Graph",
             dequantize_op_out.v2.id);

  dequantize_op.v1.numOfInputs = kQuantizeOpInputSize;
  dequantize_op.v1.inputTensors = qnn_op_ins;
  dequantize_op.v1.numOfOutputs = kQuantizeOpOutputSize;
  dequantize_op.v1.outputTensors = &dequantize_op_out;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(graph_mapper.Qnn().Api()->graphAddNode(
      graph_mapper.QnnGraph(), dequantize_op));

  // legalize quantize op.
  std::string op_name = absl::StrFormat(kQuantizeOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      SetOpInfo(op_name.c_str(), kDefaultQnnOpPackageName.data(),
                kQnnQuantizeOpTypeName.data(), dest));

  // Legalize op outputs and update scope.
  const auto op_outs = src.Outputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, kQuantizeOpOutputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.LegalizeAndRegister(op_outs.front().Get(), qnn_op_outs[0]));
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.PushToScope(op_outs.front().Get(), qnn_op_outs[0]));

  dest.v1.numOfInputs = kQuantizeOpInputSize;
  dest.v1.inputTensors = &dequantize_op_out;

  dest.v1.numOfOutputs = kQuantizeOpOutputSize;
  dest.v1.outputTensors = qnn_op_outs;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->graphAddNode(graph_mapper.QnnGraph(), dest));

  ResetTensor(dequantize_op_out);
  return kLiteRtStatusOk;
}

// Option 2: convert
LiteRtStatus QuantizeOpLegalization::ConvertLeaglization(
    const litert::Op& src, Qnn_OpConfig_t& dest, GraphMapper& graph_mapper) {
  std::string op_name = absl::StrFormat(kConvertOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(SetOpInfo(op_name.c_str(),
                                           kDefaultQnnOpPackageName.data(),
                                           kQnnConvertOpTypeName.data(), dest));
  LITERT_RETURN_STATUS_IF_NOT_OK(LegalizeSimpleOp(src, dest, graph_mapper));
  return kLiteRtStatusOk;
}

LiteRtStatus QuantizeOpLegalization::LegalizeOp(const litert::Op& src,
                                                Qnn_OpConfig_t& dest,
                                                GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflQuantize) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  DumpLegalization(*src.Get());
  if (src.Inputs().front().RankedTensorType().ElementType() ==
      src.Outputs().front().RankedTensorType().ElementType()) {
    LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
        ConvertLeaglization(src, dest, graph_mapper));
  } else {
    return kLiteRtStatusErrorInvalidLegalization;
  }
  LITERT_LOG(LITERT_INFO, "Legalized quantize op");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
