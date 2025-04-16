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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/softmax_op_legalization.h"

#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/c/litert_options.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_op.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

namespace litert::qnn {

static constexpr absl::string_view kQnnSoftmaxOpTypeName = "Softmax";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kSoftmaxOpFmt = "softmax_%d";

static constexpr int kSoftmaxOpInputSize = 1;
static constexpr int kSoftmaxOpOutputSize = 1;
static constexpr int kSoftmaxOpParamSize = 1;

LiteRtStatus SoftmaxOpLegalization::LegalizeOp(const Op& src,
                                               Qnn_OpConfig_t& dest,
                                               GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflSoftmax) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  DumpLegalization(*src.Get());
  std::string op_name = absl::StrFormat(kSoftmaxOpFmt, op_counter_++);
  LITERT_RETURN_IF_ERROR(SetOpInfo(op_name.c_str(),
                                   kDefaultQnnOpPackageName.data(),
                                   kQnnSoftmaxOpTypeName.data(), dest));

  // QNN reduce softmax op expects 1 input tensor.
  const auto op_ins = src.Inputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, kSoftmaxOpInputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_IF_ERROR(
      graph_mapper.LookupInScope(op_ins.front().Get(), qnn_op_ins[0]));

  // QNN softmax op expects 1 output tensor.
  const auto op_outs = src.Outputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, kSoftmaxOpOutputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_IF_ERROR(
      graph_mapper.LegalizeAndRegister(op_outs.front().Get(), qnn_op_outs[0]));
  LITERT_RETURN_IF_ERROR(
      graph_mapper.PushToScope(op_outs.front().Get(), qnn_op_outs[0]));

  // Prepare QNN reduce softmax parameters.

  // Extract beta option from softmax op.
  float beta;
  LITERT_RETURN_IF_ERROR(LiteRtGetSoftmaxBetaOption(src.Get(), &beta));
  Qnn_Param_t beta_param = BuildDefaultParam();
  beta_param.paramType = QNN_PARAMTYPE_SCALAR;
  beta_param.name = "beta";
  Qnn_Scalar_t keep_dims_scalar = QNN_SCALAR_INIT;
  keep_dims_scalar.dataType = QNN_DATATYPE_FLOAT_32;
  keep_dims_scalar.floatValue = beta;
  beta_param.scalarParam = keep_dims_scalar;

  Qnn_Param_t reduce_softmax_params[] = {beta_param};
  dest.v1.inputTensors = qnn_op_ins;
  dest.v1.numOfInputs = kSoftmaxOpInputSize;
  dest.v1.outputTensors = qnn_op_outs;
  dest.v1.numOfOutputs = kSoftmaxOpOutputSize;
  dest.v1.numOfParams = kSoftmaxOpParamSize;
  dest.v1.params = reduce_softmax_params;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->graphAddNode(graph_mapper.QnnGraph(), dest));

  LITERT_LOG(LITERT_INFO, "Legalized softmax op", "");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
