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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/concatenation_op_legalization.h"

#include <cstdint>
#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

namespace litert::qnn {

static constexpr absl::string_view kQnnConcatenationOpTypeName = "Concat";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kConcatenationOpFmt = "concatenation_%d";

static constexpr int kReduceConcatenationOpOutputSize = 1;
static constexpr int kReduceConcatenationOpParamSize = 1;

LiteRtStatus ConcatenationOpLegalization::LegalizeOp(
    const Op& src, Qnn_OpConfig_t& dest, GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflConcatenation) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  DumpLegalization(*src.Get());
  std::string op_name = absl::StrFormat(kConcatenationOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      SetOpInfo(op_name.c_str(), kDefaultQnnOpPackageName.data(),
                kQnnConcatenationOpTypeName.data(), dest));

  // Look up op input tensors in scope.
  const auto op_ins = src.Inputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, op_ins.size(), QNN_TENSOR_INIT);

  Qnn_Tensor_t* cur_qnn_op_in = qnn_op_ins;
  for (const auto& op_in : op_ins) {
    LITERT_RETURN_STATUS_IF_NOT_OK(
        graph_mapper.LookupInScope(op_in.Get(), *cur_qnn_op_in));
    ++cur_qnn_op_in;
  }

  // QNN concatenation op expects 1 output tensor.
  const auto op_outs = src.Outputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs,
                     kReduceConcatenationOpOutputSize, QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.LegalizeAndRegister(op_outs.front().Get(), qnn_op_outs[0]));
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.PushToScope(op_outs.front().Get(), qnn_op_outs[0]));

  // Extract axis option from concatenation op.
  int32_t axis;
  LITERT_RETURN_STATUS_IF_NOT_OK(
      LiteRtGetConcatenationAxisOption(src.Get(), &axis));

  // Construct the scalar "axis" param.
  Qnn_Param_t axis_param = BuildDefaultParam();
  axis_param.paramType = QNN_PARAMTYPE_SCALAR;
  axis_param.name = "axis";
  Qnn_Scalar_t axis_scalar = QNN_SCALAR_INIT;
  axis_scalar.dataType = QNN_DATATYPE_UINT_32;
  axis_scalar.int32Value = axis;
  axis_param.scalarParam = axis_scalar;

  Qnn_Param_t concatenation_params[] = {axis_param};
  dest.v1.inputTensors = qnn_op_ins;
  dest.v1.numOfInputs = op_ins.size();
  dest.v1.outputTensors = qnn_op_outs;
  dest.v1.numOfOutputs = kReduceConcatenationOpOutputSize;
  dest.v1.numOfParams = kReduceConcatenationOpParamSize;
  dest.v1.params = concatenation_params;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->graphAddNode(graph_mapper.QnnGraph(), dest));

  LITERT_LOG(LITERT_INFO, "Legalized concatenation op", "");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
