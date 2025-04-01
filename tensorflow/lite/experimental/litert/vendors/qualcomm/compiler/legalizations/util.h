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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_UTIL_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_UTIL_H_

#include <cstdint>

#include "third_party/qairt/latest/include/QNN/QnnTypes.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/common.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/IR/qnn_tensor.h"
#include "tensorflow/lite/experimental/litert/vendors/qualcomm/compiler/graph_mapper.h"

namespace litert::qnn {

// Use this function to legalize a LiteRtOp to a Qnn Op when:
// 1. Source input/output tensor and destination input/ouptut tensor are 1 : 1
// mapped
// 2. Assigning params to destination OP does not depending on input tensor of
// source OP.
LiteRtStatus LegalizeSimpleOp(const Op& src, Qnn_OpConfig_t& dest,
                              GraphMapper& graph_mapper);

// Dump source Op details.
void DumpLegalization(const LiteRtOpT& op);

// Build and register a QNN native tensor in the QNN graph.
LiteRtStatus BuildAndRegisterQnnNativeTensor(Qnn_DataType_t param_data_type,
                                             uint32_t rank, uint32_t* dims,
                                             GraphMapper& graph_mapper,
                                             Qnn_Tensor_t& tensor);

// Build and register a QNN op in the QNN graph.
LiteRtStatus BuildAndRegisterQnnOp(uint32_t input_size, Qnn_Tensor_t* op_ins,
                                   uint32_t output_size, Qnn_Tensor_t* op_outs,
                                   Qnn_OpConfig_t& op, uint32_t param_size,
                                   Qnn_Param_t* params,
                                   GraphMapper& graph_mapper);

// Build and register a QNN tensor param in the QNN graph.
template <typename T>
LiteRtStatus BuildQnnTesnorParam(T* param_data, uint32_t* param_dims,
                                 Qnn_DataType_t param_data_type,
                                 uint32_t param_rank, const char* param_name,
                                 GraphMapper& graph_mapper,
                                 Qnn_Param_t& param) {
  // Build ClientBuffer for the param tensor.
  Qnn_ClientBuffer_t tensor_client_buf = BuildDefaultClientBuffer();
  tensor_client_buf.data = param_data;
  tensor_client_buf.dataSize = sizeof(param_data);

  // Build QNN param tensor.
  Qnn_Tensor_t param_tensor = BuildDefaultTensor();
  graph_mapper.AssignTensorName(param_tensor);
  param_tensor.v2.dataType = param_data_type;
  param_tensor.v2.type = QNN_TENSOR_TYPE_STATIC;
  param_tensor.v2.rank = param_rank;
  param_tensor.v2.dimensions = param_dims;
  param_tensor.v2.memType = QNN_TENSORMEMTYPE_RAW;
  param_tensor.v2.clientBuf = tensor_client_buf;

  // Register param tensor in QNN graph.
  LITERT_RETURN_STATUS_IF_QNN_NOT_OK(
      graph_mapper.Qnn().Api()->tensorCreateGraphTensor(graph_mapper.QnnGraph(),
                                                        &param_tensor));
  param.paramType = QNN_PARAMTYPE_TENSOR;
  param.name = param_name;
  param.tensorParam = param_tensor;
  return kLiteRtStatusOk;
}

template <typename T>
LiteRtStatus BuildQnnScalarParam(T& param_data, Qnn_DataType_t param_data_type,
                                 const char* param_name,
                                 GraphMapper& graph_mapper,
                                 Qnn_Param_t& param) {
  // Build QNN scalar.
  Qnn_Scalar_t scalar = QNN_SCALAR_INIT;
  scalar.dataType = param_data_type;

  // Build QNN scalar param.
  switch (param_data_type) {
    case QNN_DATATYPE_BOOL_8:
      scalar.bool8Value = param_data;
      break;
    case QNN_DATATYPE_UINT_32:
      scalar.uint32Value = param_data;
      break;
    case QNN_DATATYPE_INT_32:
      scalar.int32Value = param_data;
      break;
    default:
      return kLiteRtStatusErrorUnsupported;
  }
  param.paramType = QNN_PARAMTYPE_SCALAR;
  param.name = param_name;
  param.scalarParam = scalar;
  return kLiteRtStatusOk;
}

}  // namespace litert::qnn

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_VENDORS_QUALCOMM_COMPILER_LEGALIZATIONS_UTIL_H_
