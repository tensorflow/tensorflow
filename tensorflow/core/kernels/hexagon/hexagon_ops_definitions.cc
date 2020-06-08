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

#include "tensorflow/core/framework/types.h"

// CAVEAT: Comment-out the following macro if you want to use experimental
// hexagon ops.
//#define ENABLE_EXPERIMENTAL_HEXNN_OPS

namespace tensorflow {

// HVX internal supported ops names
// TODO(satok): Remove this map once hexnn lib supports an API to retrieve op id
// from op name and data type
enum class HexagonOpsDefinitions::SupportedOpType {
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

#ifdef ENABLE_EXPERIMENTAL_HEXNN_OPS
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
  MAXIMUM_F,

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

  QUANTIZED_MUL_8x8to32,
  QUANTIZED_MUL_8x8to32_REF,
  QUANTIZED_ADD_8p8to32,
  QUANTIZED_ADD_8p8to32_REF,
  QUANTIZED_SIGMOID_8,
  QUANTIZED_SIGMOID_8_REF,
  QUANTIZED_TANH_8,
  QUANTIZED_TANH_8_REF,
  QUANTIZED_SOFTMAX_8,
  QUANTIZED_SOFTMAX_8_REF,
  QUANTIZED_LRN_8,
  QUANTIZED_LRN_8_REF,
  QUANTIZED_PAD2D_FRAME_8P,
  QUANTIZED_PAD2D_FRAME_8P_REF,
  QUANTIZED_SUB_8P8TO32,
  QUANTIZED_SUB_8P8TO32_REF,
  QUANTIZED_MAXIMUM_8,
  QUANTIZED_MAXIMUM_8_REF,
  QUANTIZED_MINIMUM_8,
  QUANTIZED_MINIMUM_8_REF,

  PAD_F,
  SPACE_TO_BATCH_ND_F,
  BATCH_TO_SPACE_ND_F,
  RESIZE_BILINEAR_F,
  CONCAT_V2_F,

#else
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
  MAXIMUM_F,

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
#endif

  SUPPORTED_OP_TYPE_COUNT  // TERMINATOR. DO NOT REMOVE
};

/* static */ void HexagonOpsDefinitions::EmplaceOpType(
    const string& op_type, const DataTypeVector& dt_vec,
    const SupportedOpType supported_op_type,
    std::unordered_map<string, std::vector<DataTypeToOp>>* map) {
  if (map->count(op_type) <= 0) {
    map->emplace(op_type, std::vector<DataTypeToOp>());
  }
  map->at(op_type).emplace_back(
      std::forward_as_tuple(dt_vec, supported_op_type));
}

/* static */ std::unordered_map<
    string, std::vector<HexagonOpsDefinitions::DataTypeToOp>>
HexagonOpsDefinitions::BuildOpNameToSocOpTypeMap() {
  std::unordered_map<string, std::vector<DataTypeToOp>> op_map;
  // Custom Op name
  EmplaceOpType("INPUT", {}, SupportedOpType::INPUT, &op_map);
  EmplaceOpType("OUTPUT", {}, SupportedOpType::OUTPUT, &op_map);
  EmplaceOpType("NoOp", {}, SupportedOpType::NOP, &op_map);
  // Special op type for hexagon
  EmplaceOpType("FLATTEN", {}, SupportedOpType::FLATTEN, &op_map);
  // Tensorflow op name
  // CAVEAT: Keep order of SupportedOpType
  EmplaceOpType("Identity", {}, SupportedOpType::NOP, &op_map);
  EmplaceOpType("Placeholder", {}, SupportedOpType::NOP, &op_map);
  EmplaceOpType("Const", {}, SupportedOpType::OP_CONST, &op_map);
  EmplaceOpType("QuantizedConv2D", {}, SupportedOpType::QUANTIZEDCONV2D_8X8TO32,
                &op_map);
  EmplaceOpType("QuantizedMatMul", {}, SupportedOpType::QUANTIZEDMATMUL_8X8TO32,
                &op_map);
  EmplaceOpType("QuantizeDownAndShrinkRange", {},
                SupportedOpType::QUANTIZEDOWNANDSHRINKRANGE_32TO8, &op_map);
  EmplaceOpType("QuantizedRelu", {}, SupportedOpType::QUANTIZEDRELU_8, &op_map);
  EmplaceOpType("QuantizedReluX", {}, SupportedOpType::QUANTIZEDRELUX_8,
                &op_map);
  EmplaceOpType("QuantizedMaxPool", {}, SupportedOpType::QUANTIZEDMAXPOOL_8,
                &op_map);
  EmplaceOpType("QuantizedAvgPool", {}, SupportedOpType::QUANTIZEDAVGPOOL_8,
                &op_map);
  EmplaceOpType("QuantizedConcat", {}, SupportedOpType::QUANTIZEDCONCAT_8,
                &op_map);
  EmplaceOpType("QuantizedBiasAdd", {},
                SupportedOpType::QUANTIZEDBIASADD_8P8TO32, &op_map);
  EmplaceOpType("Min", {}, SupportedOpType::MIN_F, &op_map);
  EmplaceOpType("Max", {}, SupportedOpType::MAX_F, &op_map);
  EmplaceOpType("QuantizeV2", {}, SupportedOpType::QUANTIZE, &op_map);
  EmplaceOpType("Dequantize", {}, SupportedOpType::DEQUANTIZE, &op_map);
  EmplaceOpType("Softmax", {}, SupportedOpType::SOFTMAX_F, &op_map);
  EmplaceOpType("Reshape", {}, SupportedOpType::RESHAPE, &op_map);
  EmplaceOpType("QuantizedReshape", {}, SupportedOpType::QUANTIZED_RESHAPE,
                &op_map);
  EmplaceOpType("Sigmoid", {}, SupportedOpType::SIGMOID_F, &op_map);
  EmplaceOpType("Slice", {}, SupportedOpType::SLICE_F, &op_map);
  EmplaceOpType("Add", {}, SupportedOpType::ADD_F, &op_map);
  EmplaceOpType("Mul", {}, SupportedOpType::MUL_F, &op_map);
  EmplaceOpType("Requantize", {}, SupportedOpType::REQUANTIZE_32_TO_8, &op_map);
  EmplaceOpType("RequantizationRange", {},
                SupportedOpType::REQUANTIZATION_RANGE_32, &op_map);
  EmplaceOpType("Sub", {}, SupportedOpType::SUB_F, &op_map);
  EmplaceOpType("Pack", {}, SupportedOpType::PACK_INT32, &op_map);
  EmplaceOpType("StridedSlice", {}, SupportedOpType::STRIDED_SLICE_F, &op_map);
  EmplaceOpType("ExpandDims", {}, SupportedOpType::EXPAND_DIMS_F, &op_map);
#ifdef ENABLE_EXPERIMENTAL_HEXNN_OPS
  EmplaceOpType("QuantizedMul", {}, SupportedOpType::QUANTIZED_MUL_8x8to32,
                &op_map);
  EmplaceOpType("QuantizedAdd", {}, SupportedOpType::QUANTIZED_ADD_8p8to32,
                &op_map);
  EmplaceOpType("Pad", {}, SupportedOpType::PAD_F, &op_map);
  EmplaceOpType("SpaceToBatchND", {}, SupportedOpType::SPACE_TO_BATCH_ND_F,
                &op_map),
      EmplaceOpType("BatchToSpaceND", {}, SupportedOpType::BATCH_TO_SPACE_ND_F,
                    &op_map);
  EmplaceOpType("ResizeBilinear", {}, SupportedOpType::RESIZE_BILINEAR_F,
                &op_map);
  EmplaceOpType("ConcatV2", {}, SupportedOpType::CONCAT_V2_F, &op_map);
  EmplaceOpType("Conv2DBackpropInput", {}, SupportedOpType::DECONV_F, &op_map);

  EmplaceOpType("Tanh", {}, SupportedOpType::TANH_F, &op_map);
  EmplaceOpType("Split", {}, SupportedOpType::SPLIT_F, &op_map);
  EmplaceOpType("Transpose", {}, SupportedOpType::TRANSPOSE_F, &op_map);
  EmplaceOpType("Concat", {}, SupportedOpType::CONCAT_F, &op_map);
#endif
  return op_map;
};

HexagonOpsDefinitions::HexagonOpsDefinitions()
    : op_name_to_soc_op_type_map_(BuildOpNameToSocOpTypeMap()) {}

/* static */ const IRemoteFusedGraphOpsDefinitions&
HexagonOpsDefinitions::getInstance() {
  const static HexagonOpsDefinitions instance{};
  return instance;
}

int HexagonOpsDefinitions::GetTotalOpsCount() const {
  return static_cast<int>(SupportedOpType::SUPPORTED_OP_TYPE_COUNT);
}

int HexagonOpsDefinitions::GetOpIdFor(const string& op_type,
                                      const DataTypeVector& dt_vec) const {
  if (op_name_to_soc_op_type_map_.count(op_type) > 0) {
    const std::vector<DataTypeToOp>& dt_to_op_vec =
        op_name_to_soc_op_type_map_.at(op_type);
    CHECK(!dt_to_op_vec.empty());
    // If argument DataType is empty, return the first entry.
    if (dt_vec.empty()) {
      return static_cast<int>(std::get<1>(dt_to_op_vec.front()));
    }
    // If there is only one op_id registered for empty op_vec, we assume
    // that the op supports any data types.
    if (dt_to_op_vec.size() == 1 && std::get<0>(dt_to_op_vec.front()).empty()) {
      return static_cast<int>(std::get<1>(dt_to_op_vec.front()));
    }
    for (const DataTypeToOp& data_type_to_op : dt_to_op_vec) {
      if (std::get<0>(data_type_to_op) == dt_vec) {
        return static_cast<int>(std::get<1>(data_type_to_op));
      }
    }
  }
  return IRemoteFusedGraphOpsDefinitions::INVALID_OP_ID;
}
}  // namespace tensorflow
