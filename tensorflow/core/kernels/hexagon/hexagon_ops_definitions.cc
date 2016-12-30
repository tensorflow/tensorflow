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
  QUANTIZEDMAXPOOL_8,
  QUANTIZEDMAXPOOL_8_REF,
  QUANTIZEDAVGPOOL_8,
  QUANTIZEDAVGPOOL_8_REF,
  QUANTIZEDCONCAT_8,
  QUANTIZEDCONCAT_8_REF,
  QUANTIZEDBIASADD_8P8TO32,
  QUANTIZEDBIASADD_8P8TO32_REF,
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
  SUPPORTED_OP_TYPE_COUNT,
};

const std::unordered_map<string, SupportedOpType> OP_NAME_TO_SOC_OP_TYPE_MAP{
    // Custom Op name
    {IGraphTransferOpsDefinitions::INPUT_OP_NAME, SupportedOpType::INPUT},
    {IGraphTransferOpsDefinitions::OUTPUT_OP_NAME, SupportedOpType::OUTPUT},
    {"NoOp", SupportedOpType::NOP},
    {IGraphTransferOpsDefinitions::FLATTEN_OP_NAME, SupportedOpType::FLATTEN},
    // Tensorflow op name
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
};

/* static */ const IGraphTransferOpsDefinitions&
HexagonOpsDefinitions::getInstance() {
  const static HexagonOpsDefinitions instance{};
  return instance;
}

int HexagonOpsDefinitions::GetTotalOpsCount() const {
  return static_cast<int>(SupportedOpType::SUPPORTED_OP_TYPE_COUNT);
}

int HexagonOpsDefinitions::GetInputNodeOpId() const {
  return static_cast<int>(SupportedOpType::INPUT);
}

int HexagonOpsDefinitions::GetOutputNodeOpId() const {
  return static_cast<int>(SupportedOpType::OUTPUT);
}

int HexagonOpsDefinitions::GetOpIdFor(const string& op_type) const {
  if (OP_NAME_TO_SOC_OP_TYPE_MAP.count(op_type) > 0) {
    return static_cast<int>(OP_NAME_TO_SOC_OP_TYPE_MAP.at(op_type));
  }
  return IGraphTransferOpsDefinitions::INVALID_OP_ID;
}
};
