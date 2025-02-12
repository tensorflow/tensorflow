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

#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/legalizations/util.h"

#include <cstdint>
#include <sstream>

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/tools/dump.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/qnn_manager.h"

namespace litert::qnn {

using ::litert::internal::Dump;
using ::litert::internal::DumpOptions;

// Dump source Op details.
void DumpLegalization(const LiteRtOpT& op) {
  std::ostringstream dump;
  // TODO Make dump tools part of stable api.
  Dump(op, dump);
  DumpOptions(op, dump);
  std::string s = dump.str();
  LITERT_LOG(LITERT_INFO, "%s", s.data());
}

LiteRtStatus LegalizeSimpleOp(const Op& src, Qnn_OpConfig_t& dest,
                              GraphMapper& graph_mapper) {
  DumpLegalization(*src.Get());
  // Look up op input tensors in scope.
  const auto op_ins = src.Inputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_ins, op_ins.size(), QNN_TENSOR_INIT);

  Qnn_Tensor_t* cur_qnn_op_in = qnn_op_ins;
  for (const auto& op_in : op_ins) {
    LITERT_RETURN_IF_ERROR(
        graph_mapper.LookupInScope(op_in.Get(), *cur_qnn_op_in));
    ++cur_qnn_op_in;
  }

  // Legalize op outputs and update scope.

  const auto op_outs = src.Outputs();
  LITERT_STACK_ARRAY(Qnn_Tensor_t, qnn_op_outs, op_outs.size(),
                     QNN_TENSOR_INIT);

  Qnn_Tensor_t* cur_qnn_op_out = qnn_op_outs;
  for (const auto& op_out : op_outs) {
    LITERT_RETURN_IF_ERROR(
        graph_mapper.LegalizeAndRegister(op_out.Get(), *cur_qnn_op_out));
    LITERT_RETURN_IF_ERROR(
        graph_mapper.PushToScope(op_out.Get(), *cur_qnn_op_out));
    ++cur_qnn_op_out;
  }
  dest.v1.numOfInputs = op_ins.size();
  dest.v1.inputTensors = qnn_op_ins;

  dest.v1.numOfOutputs = op_outs.size();
  dest.v1.outputTensors = qnn_op_outs;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->graphAddNode(graph_mapper.QnnGraph(), dest));

  return kLiteRtStatusOk;
}

LiteRtStatus BuildAndRegisterQnnNativeTensor(Qnn_DataType_t param_data_type,
                                             uint32_t rank, uint32_t* dims,
                                             GraphMapper& graph_mapper,
                                             Qnn_Tensor_t& tensor) {
  graph_mapper.AssignTensorName(tensor);
  tensor.v2.dataType = param_data_type;
  tensor.v2.type = QNN_TENSOR_TYPE_NATIVE;
  tensor.v2.rank = rank;
  tensor.v2.dimensions = dims;
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->tensorCreateGraphTensor(graph_mapper.QnnGraph(),
                                                        &tensor));
  return kLiteRtStatusOk;
}

LiteRtStatus BuildAndRegisterQnnOp(uint32_t input_size, Qnn_Tensor_t* op_ins,
                                   uint32_t output_size, Qnn_Tensor_t* op_outs,
                                   Qnn_OpConfig_t& op, uint32_t param_size,
                                   Qnn_Param_t* params,
                                   GraphMapper& graph_mapper) {
  op.v1.numOfInputs = input_size;
  op.v1.inputTensors = op_ins;
  op.v1.numOfOutputs = output_size;
  op.v1.outputTensors = op_outs;
  op.v1.numOfParams = param_size;
  op.v1.params = params;

  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->graphAddNode(graph_mapper.QnnGraph(), op));

  return kLiteRtStatusOk;
}

}  // namespace litert::qnn
