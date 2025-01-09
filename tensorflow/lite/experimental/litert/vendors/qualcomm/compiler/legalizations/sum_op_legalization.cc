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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/sum_op_legalization.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

namespace litert::qnn {

static constexpr absl::string_view kQnnSumOpTypeName = "ReduceSum";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kSumOpFmt = "sum_%d";

static constexpr int kReduceSumOpInputSize = 1;
static constexpr int kReduceSumOpOutputSize = 1;
static constexpr int kReduceSumOpParamSize = 1;
static constexpr int kReduceSumOpParamRank = 1;

LiteRtStatus SumOpLegalization::LegalizeOp(const Op& src, Qnn_OpConfig_t& dest,
                                           GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflSum) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  DumpLegalization(*src.Get());
  std::string op_name = absl::StrFormat(kSumOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(SetOpInfo(op_name.c_str(),
                                           kDefaultQnnOpPackageName.data(),
                                           kQnnSumOpTypeName.data(), dest));

  // QNN reduce sum op expects 1 input tensor.
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, kReduceSumOpInputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.LookupInScope(src.Inputs().front().Get(), qnn_op_ins[0]));

  // QNN sum op expects 1 output tensor.
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, kReduceSumOpOutputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(graph_mapper.LegalizeAndRegister(
      src.Outputs().front().Get(), qnn_op_outs[0]));
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.PushToScope(src.Outputs().front().Get(), qnn_op_outs[0]));

  // Prepare QNN reduce sum parameters.
  const auto inputs = src.Inputs();
  const auto& src_axes = inputs.at(1);

  // Check if src_axes are weights tensors.
  if (!src_axes.HasWeights()) {
    LITERT_LOG(LITERT_ERROR, "Sum op axes are not weights tensors");
    return kLiteRtStatusErrorInvalidLegalization;
  }

  auto src_axes_tensor_type = src_axes.RankedTensorType();
  if (!src_axes_tensor_type) {
    LITERT_LOG(LITERT_ERROR, "%s",
               src_axes_tensor_type.Error().Message().data());
    return src_axes_tensor_type.Error().Status();
  }

  int32_t dest_axes_size = src_axes_tensor_type->Layout().Dimensions()[0];
  auto src_axes_data = src_axes.Weights().Bytes();
  Qnn_ClientBuffer_t axes_tensor_client_buf = BuildDefaultClientBuffer();
  axes_tensor_client_buf.data = (void*)src_axes_data.data();
  axes_tensor_client_buf.dataSize = src_axes_data.size();

  // Extract keepdims option from sum op.
  bool keep_dims;
  LITERT_RETURN_STATUS_IF_NOT_OK(
      LiteRtGetSumKeepDimsOption(src.Get(), &keep_dims));

  // Construct the scalar "keep_dims" param.
  if (keep_dims) {
    Qnn_Param_t range_param = BuildDefaultParam();
    range_param.paramType = QNN_PARAMTYPE_SCALAR;
    range_param.name = "keep_dims";
    Qnn_Scalar_t keep_dims_scalar = QNN_SCALAR_INIT;
    keep_dims_scalar.dataType = QNN_DATATYPE_BOOL_8;
    keep_dims_scalar.bool8Value = true;
    range_param.scalarParam = keep_dims_scalar;
  }

  // Construct the const tensor "axes".
  Qnn_Tensor_t range_tensor = BuildDefaultTensor();
  graph_mapper.AssignTensorName(range_tensor);
  range_tensor.v2.dataType = QNN_DATATYPE_INT_32;
  range_tensor.v2.type = QNN_TENSOR_TYPE_STATIC;
  range_tensor.v2.rank = kReduceSumOpParamRank;
  range_tensor.v2.dimensions = new uint32_t[kReduceSumOpParamRank];
  range_tensor.v2.dimensions[0] = dest_axes_size;
  range_tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  range_tensor.v2.clientBuf = axes_tensor_client_buf;
  range_tensor.v2.isDynamicDimensions = nullptr;
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->tensorCreateGraphTensor(graph_mapper.QnnGraph(),
                                                        &range_tensor));

  Qnn_Param_t range_param = BuildDefaultParam();
  range_param.paramType = QNN_PARAMTYPE_TENSOR;
  range_param.name = "axes";
  range_param.tensorParam = range_tensor;

  Qnn_Param_t reduce_sum_params[] = {range_param};
  dest.v1.inputTensors = qnn_op_ins;
  dest.v1.numOfInputs = kReduceSumOpInputSize;
  dest.v1.outputTensors = qnn_op_outs;
  dest.v1.numOfOutputs = kReduceSumOpOutputSize;
  dest.v1.numOfParams = kReduceSumOpParamSize;
  dest.v1.params = reduce_sum_params;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->graphAddNode(graph_mapper.QnnGraph(), dest));

  LITERT_LOG(LITERT_INFO, "Legalized sum op", "");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
