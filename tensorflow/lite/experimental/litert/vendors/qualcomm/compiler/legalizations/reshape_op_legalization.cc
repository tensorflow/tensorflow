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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/reshape_op_legalization.h"

#include <string>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/c/litert_support.h"
#include "tensorflow/lite/experimental/litert/cc/litert_op.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

namespace litert::qnn {

static constexpr absl::string_view kQnnReshapeOpTypeName = "Reshape";
static constexpr absl::string_view kDefaultQnnOpPackageName = "qti.aisw";
static constexpr absl::string_view kReshapeOpFmt = "reshape_%d";

static constexpr int kReshapeOpInputSize = 1;
static constexpr int kReshapeOpOutputSize = 1;

LiteRtStatus ReshapeOpLegalization::LegalizeOp(LiteRtOpManager& src,
                                               Qnn_OpConfig_t& dest,
                                               GraphMapper& graph_mapper) {
  if (src.Code() != kLiteRtOpCodeTflReshape) {
    return kLiteRtStatusLegalizeNoMatch;
  }
  std::string op_name = absl::StrFormat(kReshapeOpFmt, op_counter_++);
  LITERT_RETURN_STATUS_IF_NOT_OK(SetOpInfo(op_name.c_str(),
                                           kDefaultQnnOpPackageName.data(),
                                           kQnnReshapeOpTypeName.data(), dest));
  DumpLegalization(*src.Op());
  // Look up op input tensors in scope.
  LITERT_ASSIGN_OR_RETURN_STATUS(auto op_ins,
                                 ::graph_tools::GetOpIns(src.Op()));
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, kReshapeOpInputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.LookupInScope(op_ins[0], qnn_op_ins[0]));

  // Legalize op outputs and update scope.

  LITERT_ASSIGN_OR_RETURN_STATUS(auto op_outs,
                                 ::graph_tools::GetOpOuts(src.Op()));
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, kReshapeOpOutputSize,
                     QNN_TENSOR_INIT);
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.LegalizeAndRegister(op_outs[0], qnn_op_outs[0]));
  LITERT_RETURN_STATUS_IF_NOT_OK(
      graph_mapper.PushToScope(op_outs[0], qnn_op_outs[0]));

  dest.v1.numOfInputs = kReshapeOpInputSize;
  dest.v1.inputTensors = qnn_op_ins;

  dest.v1.numOfOutputs = kReshapeOpOutputSize;
  dest.v1.outputTensors = qnn_op_outs;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->graphAddNode(graph_mapper.QnnGraph(), dest));

  LITERT_LOG(LITERT_INFO, "Legalized reshape op", "");
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
