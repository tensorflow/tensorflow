/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
vcyou may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/kernels/hexagon/hexagon_ops_definitions.h"

#include <unordered_map>

#include "tensorflow/core/framework/types.h"

namespace tensorflow {

// HVX internal supported ops names
enum class SupportedOpType {
  INPUT,
  OUTPUT,
  NOP,
  OP_CONST, /* OP_ is required to avoid compilation error on windows */
  CHECK,
  CLOSE_FLOAT32,
  CLOSE_QINT8,
  CLOSE_Q_QINT8,
  CLOSE_INT32,
  CLOSE_QINT32,
  PPRINT_8,
  PPRINT_32,
  PPRINT_FLOAT,
  PREFREE,
  FLATTEN,
  // With Reference
  QUANTIZEDCONV2D_8X8TO32,
  QUANTIZEDCONV2D_8X8TO32_REF,
  QUANTIZEDMATMUL_8X8TO32,
  QUANTIZEDMATMUL_8X8TO32_REF,
  QUANTIZEDOWNANDSHRINKRANGE_32TO8,
  QUANTIZEDOWNANDSHRINKRANGE_32TO8_REF,
  QUANTIZEDRELU_8,
  QUANTIZEDRELU_8_REF,
  QUANTIZEDRELUX_8,
  QUANTIZEDRELUX_8_REF,
  QUANTIZEDSIGMOID_8,
  QUANTIZEDSIGMOID_8_REF,
  QUANTIZEDTANH_8,
  QUANTIZEDTANH_8_REF,
  QUANTIZEDMAXPOOL_8,
  QUANTIZEDMAXPOOL_8_REF,
  QUANTIZEDAVGPOOL_8,
  QUANTIZEDAVGPOOL_8_REF,
  QUANTIZEDCONCAT_8,
  QUANTIZEDCONCAT_8_REF,
  QUANTIZEDBIASADD_8P8TO32,
  QUANTIZEDBIASADD_8P8TO32_REF,
  QUANTIZEDSOFTMAX_8,
  QUANTIZEDSOFTMAX_8_REF,
  QUANTIZEDLRN_8,
  QUANTIZEDLRN_8_REF,
  MIN_F,
  MIN_F_REF,
  MAX_F,
  MAX_F_REF,
  QUANTIZE,
  QUANTIZE_REF,
  DEQUANTIZE,
  DEQUANTIZE_REF,
  SUPERNODE_8X8P8TO8,
  SUPERNODE_8X8P8TO8_REF,

  QUANTIZEDFLATTEN,
  SOFTMAX_F,
  CONV2D_F,
  MATMUL_F,
  RELU_F,
  RELUX_F,
  AVGPOOL_F,
  MAXPOOL_F,
  CONCAT_F,
  BIASADD_F,
  LRN_F,

  VARIABLE,
  ASSIGN,
  RESHAPE,
  QUANTIZED_RESHAPE,
  TANH_F,
  SIGMOID_F,
  SLICE_8,
  SLICE_F,
  QUANTIZED_SLICE_8,
  ADD_F,
  MUL_F,
  MINIMUM_F,
  MAXIMAM_F,

  REQUANTIZE_32_TO_8,
  REQUANTIZE_32_TO_8_REF,
  REQUANTIZATION_RANGE_32,
  REQUANTIZATION_RANGE_32_REF,

  NEG_F,
  SUB_F,
  ADD_N_F,
  RANGE_INT32,
  RANK_INT32,
  TRANSPOSE_INT32,
  TRANSPOSE_F,
  INSTANCE_NORM_F,
  QUANTIZED_INSTANCENORM_8,
  QUANTIZED_INSTANCENORM_8_REF,
  SUB_INT32,
  ADD_INT32,
  SPLIT_F,
  DEQUANTIZE_QINT32_F,
  PRELU_F,
  QUANTIZED_PRELU_8,
  SUM_F,
  PROD_F,
  MUL_INT32,
  LOGICAL_AND_INT32,
  LOGICALOR_INT32,
  LOGICAL_XOR_INT32,
  SPAPE_INT32,
  PACK_INT32,
  MIRROR_PAD_F,
  RESIZE_NEAREST_NEIGHBOR_F,
  STRIDED_SLICE_INT32,
  STRIDED_SLICE_F,
  EXPAND_DIMS_INT32,
  EXPAND_DIMS_F,

  LOG_SOFTMAX_F,
  SPLIT_INT32,
  QUANTIZED_SPLIT_8,

  DECONV_F,
  QUANTIZED_DECONV_8X8TO32,
  QUANTIZED_DECONV_8X8TO32_REF,

  SUPPORTED_OP_TYPE_COUNT  // TERMINATOR. DO NOT REMOVE
};

const std::unordered_map<string, SupportedOpType> OP_NAME_TO_SOC_OP_TYPE_MAP{
    // Custom Op name
    {"INPUT", SupportedOpType::INPUT},
    {"OUTPUT", SupportedOpType::OUTPUT},
    {"NoOp", SupportedOpType::NOP},
    {IGraphTransferOpsDefinitions::FLATTEN_OP_NAME, SupportedOpType::FLATTEN},
    // Tensorflow op name
    {"Const", SupportedOpType::OP_CONST},
    {"QuantizedConv2D", SupportedOpType::QUANTIZEDCONV2D_8X8TO32},
    {"QuantizedMatMul", SupportedOpType::QUANTIZEDMATMUL_8X8TO32},
    {"QuantizeDownAndShrinkRange",
     SupportedOpType::QUANTIZEDOWNANDSHRINKRANGE_32TO8},
    {"QuantizedRelu", SupportedOpType::QUANTIZEDRELU_8},
    {"QuantizedReluX", SupportedOpType::QUANTIZEDRELUX_8},
    {"QuantizedMaxPool", SupportedOpType::QUANTIZEDMAXPOOL_8},
    {"QuantizedAvgPool", SupportedOpType::QUANTIZEDAVGPOOL_8},
    {"QuantizedConcat", SupportedOpType::QUANTIZEDCONCAT_8},
    {"QuantizedBiasAdd", SupportedOpType::QUANTIZEDBIASADD_8P8TO32},
    {"Min", SupportedOpType::MIN_F},
    {"Max", SupportedOpType::MAX_F},
    {"QuantizeV2", SupportedOpType::QUANTIZE},
    {"Dequantize", SupportedOpType::DEQUANTIZE},
    {"Softmax", SupportedOpType::SOFTMAX_F},
    {"Placeholder", SupportedOpType::NOP},
    {"RequantizationRange", SupportedOpType::REQUANTIZATION_RANGE_32},
    {"Requantize", SupportedOpType::REQUANTIZE_32_TO_8},
    {"QuantizedReshape", SupportedOpType::QUANTIZED_RESHAPE},
    {"Add", SupportedOpType::ADD_F},
    {"Sub", SupportedOpType::SUB_F},
    {"Reshape", SupportedOpType::RESHAPE},
    {"Identity", SupportedOpType::NOP},
};

/* static */ const IGraphTransferOpsDefinitions&
HexagonOpsDefinitions::getInstance() {
  const static HexagonOpsDefinitions instance{};
  return instance;
}

int HexagonOpsDefinitions::GetTotalOpsCount() const {
  return static_cast<int>(SupportedOpType::SUPPORTED_OP_TYPE_COUNT);
}

int HexagonOpsDefinitions::GetOpIdFor(const string& op_type) const {
  if (OP_NAME_TO_SOC_OP_TYPE_MAP.count(op_type) > 0) {
    return static_cast<int>(OP_NAME_TO_SOC_OP_TYPE_MAP.at(op_type));
  }
  return IGraphTransferOpsDefinitions::INVALID_OP_ID;
}

GraphTransferInfo::Destination HexagonOpsDefinitions::GetTransferDestination()
    const {
  return GraphTransferInfo::HEXAGON;
}
};
