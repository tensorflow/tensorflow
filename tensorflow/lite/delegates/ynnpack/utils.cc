/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/lite/delegates/ynnpack/utils.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/types/half.h"

namespace tflite {
namespace ynnpack {

namespace {
uint16_t FloatToBfloat16(float f) {
  uint32_t val;
  std::memcpy(&val, &f, sizeof(float));
  // Round to nearest even.
  uint32_t rounding_bias = 0x7fff + ((val >> 16) & 1);
  val += rounding_bias;
  return static_cast<uint16_t>(val >> 16);
}

template <typename T>
T QuantizeValue(float value, float scale, int32_t zero_point) {
  int32_t quantized = zero_point + std::round(value / scale);
  return static_cast<T>(std::max<int32_t>(
      std::numeric_limits<T>::min(),
      std::min<int32_t>(std::numeric_limits<T>::max(), quantized)));
}
}  // namespace

TfLiteStatus GetTfLiteTensorValueAsDouble(TfLiteContext* context,
                                          const TfLiteTensor& tensor, int index,
                                          double* value) {
  TF_LITE_ENSURE(context, tensor.data.raw != nullptr);
  int num_elements = tflite::NumElements(&tensor);
  TF_LITE_ENSURE_MSG(context, index >= 0 && index < num_elements,
                     "Index %d out of bounds for tensor with %d elements",
                     index, num_elements);

  switch (tensor.type) {
    case kTfLiteFloat32:
      *value = reinterpret_cast<const float*>(tensor.data.raw)[index];
      break;
    case kTfLiteInt32:
      *value = reinterpret_cast<const int32_t*>(tensor.data.raw)[index];
      break;
    case kTfLiteFloat16:
      *value = static_cast<float>(
          reinterpret_cast<const tflite::half*>(tensor.data.raw)[index]);
      break;
    case kTfLiteInt8:
      *value = reinterpret_cast<const int8_t*>(tensor.data.raw)[index];
      break;
    case kTfLiteUInt8:
      *value = reinterpret_cast<const uint8_t*>(tensor.data.raw)[index];
      break;
    default:
      TF_LITE_ENSURE_MSG(context, false, "Unsupported type %d", tensor.type);
  }
  return kTfLiteOk;
}

ynn_type GetYnnType(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat32:
      return ynn_type_fp32;
    case kTfLiteInt32:
      return ynn_type_int32;
    case kTfLiteFloat16:
      return ynn_type_fp16;
    case kTfLiteFloat64:
      return ynn_type_fp64;
    case kTfLiteBFloat16:
      return ynn_type_bf16;
    case kTfLiteInt8:
      return ynn_type_int8;
    case kTfLiteUInt8:
      return ynn_type_uint8;
    case kTfLiteInt4:
      return ynn_type_int4;
    case kTfLiteUInt4:
      return ynn_type_uint4;
    case kTfLiteInt2:
      return ynn_type_int2;
    default:
      return ynn_type_invalid;
  }
}

ynn_unary_operator GetYnnUnaryOperator(int builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinAbs:
    case kTfLiteBuiltinStablehloAbs:
      return ynn_unary_abs;
    case kTfLiteBuiltinCeil:
      return ynn_unary_ceil;
    case kTfLiteBuiltinCos:
    case kTfLiteBuiltinStablehloCosine:
      return ynn_unary_cosine;
    case kTfLiteBuiltinExp:
    case kTfLiteBuiltinStablehloExponential:
      return ynn_unary_exp;
    case kTfLiteBuiltinFloor:
    case kTfLiteBuiltinStablehloFloor:
      return ynn_unary_floor;
    case kTfLiteBuiltinLog:
    case kTfLiteBuiltinStablehloLog:
      return ynn_unary_log;
    case kTfLiteBuiltinNeg:
    case kTfLiteBuiltinStablehloNegate:
      return ynn_unary_negate;
    case kTfLiteBuiltinRsqrt:
    case kTfLiteBuiltinStablehloRsqrt:
      return ynn_unary_reciprocal_square_root;
    case kTfLiteBuiltinRound:
      return ynn_unary_round;
    case kTfLiteBuiltinLogistic:
    case kTfLiteBuiltinStablehloLogistic:
      return ynn_unary_sigmoid;
    case kTfLiteBuiltinSin:
      return ynn_unary_sine;
    case kTfLiteBuiltinSquare:
      return ynn_unary_square;
    case kTfLiteBuiltinSqrt:
      return ynn_unary_square_root;
    case kTfLiteBuiltinTanh:
    case kTfLiteBuiltinStablehloTanh:
      return ynn_unary_tanh;
    case kTfLiteBuiltinSign:
      return ynn_unary_sign;
    case kTfLiteBuiltinCast:
    case kTfLiteBuiltinStablehloConvert:
      return ynn_unary_convert;
    default:
      return ynn_unary_invalid;
  }
}

ynn_binary_operator GetYnnBinaryOperator(int builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinAdd:
    case kTfLiteBuiltinStablehloAdd:
      return ynn_binary_add;
    case kTfLiteBuiltinDiv:
    case kTfLiteBuiltinStablehloDivide:
      return ynn_binary_divide;
    case kTfLiteBuiltinMaximum:
    case kTfLiteBuiltinStablehloMaximum:
      return ynn_binary_max;
    case kTfLiteBuiltinMinimum:
    case kTfLiteBuiltinStablehloMinimum:
      return ynn_binary_min;
    case kTfLiteBuiltinMul:
    case kTfLiteBuiltinStablehloMultiply:
      return ynn_binary_multiply;
    case kTfLiteBuiltinPow:
    case kTfLiteBuiltinStablehloPower:
      return ynn_binary_pow;
    case kTfLiteBuiltinSub:
    case kTfLiteBuiltinStablehloSubtract:
      return ynn_binary_subtract;
    case kTfLiteBuiltinSquaredDifference:
      return ynn_binary_squared_difference;
    case kTfLiteBuiltinPrelu:
      return ynn_binary_leaky_relu;
    default:
      return ynn_binary_invalid;
  }
}

ynn_reduce_operator GetYnnReduceOperator(int builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinSum:
    case kTfLiteBuiltinMean:
      return ynn_reduce_sum;
    case kTfLiteBuiltinReduceMin:
      return ynn_reduce_min;
    case kTfLiteBuiltinReduceMax:
      return ynn_reduce_max;
    default:
      return ynn_reduce_invalid;
  }
}

bool IsUnaryOp(int builtin_code) {
  return GetYnnUnaryOperator(builtin_code) != ynn_unary_invalid ||
         builtin_code == kTfLiteBuiltinGelu ||
         builtin_code == kTfLiteBuiltinElu ||
         builtin_code == kTfLiteBuiltinLeakyRelu ||
         builtin_code == kTfLiteBuiltinHardSwish ||
         builtin_code == kTfLiteBuiltinRelu ||
         builtin_code == kTfLiteBuiltinRelu6 ||
         builtin_code == kTfLiteBuiltinReluN1To1 ||
         builtin_code == kTfLiteBuiltinRelu0To1;
}

bool IsBinaryOp(int builtin_code) {
  return GetYnnBinaryOperator(builtin_code) != ynn_binary_invalid;
}

bool IsStablehloOp(int builtin_code) {
  switch (builtin_code) {
    case kTfLiteBuiltinStablehloLogistic:
    case kTfLiteBuiltinStablehloAdd:
    case kTfLiteBuiltinStablehloDivide:
    case kTfLiteBuiltinStablehloMultiply:
    case kTfLiteBuiltinStablehloMaximum:
    case kTfLiteBuiltinStablehloAbs:
    case kTfLiteBuiltinStablehloCosine:
    case kTfLiteBuiltinStablehloExponential:
    case kTfLiteBuiltinStablehloFloor:
    case kTfLiteBuiltinStablehloLog:
    case kTfLiteBuiltinStablehloMinimum:
    case kTfLiteBuiltinStablehloNegate:
    case kTfLiteBuiltinStablehloPower:
    case kTfLiteBuiltinStablehloRsqrt:
    case kTfLiteBuiltinStablehloSubtract:
    case kTfLiteBuiltinStablehloTanh:
    case kTfLiteBuiltinStablehloConvert:
    case kTfLiteBuiltinStablehloClamp:
      return true;
    default:
      return false;
  }
}

bool IsQuantized(const TfLiteTensor& tensor) {
  if (tensor.type == kTfLiteFloat32 || tensor.type == kTfLiteFloat16 ||
      tensor.type == kTfLiteBFloat16 || tensor.type == kTfLiteFloat64 ||
      tensor.type == kTfLiteFloat8E4M3FN || tensor.type == kTfLiteFloat8E5M2) {
    return false;
  }
  return tensor.quantization.type != kTfLiteNoQuantization ||
         tensor.params.scale != 0.0f;
}

bool IsSupportedQuantization(const TfLiteTensor& tensor,
                             bool allow_per_channel) {
  if (!IsQuantized(tensor)) return true;
  if (tensor.quantization.type != kTfLiteAffineQuantization) return false;
  const auto* params =
      static_cast<const TfLiteAffineQuantization*>(tensor.quantization.params);
  if (!params) return false;
  if (!params->scale) return false;
  if (params->scale->size != 1 && !allow_per_channel) return false;
  if (tensor.type == kTfLiteUInt8) {
    if (!params->zero_point) return false;
    if (params->zero_point->size != params->scale->size &&
        params->zero_point->size != 1) {
      return false;
    }
  }
  return true;
}

size_t YnnTypeElementCount(ynn_type type) {
  switch (type) {
    case ynn_type_int2:
    case ynn_type_uint2:
      return 4;
    case ynn_type_int4:
    case ynn_type_uint4:
      return 2;
    default:
      return 1;
  }
}

bool IsTensorSupported(const TfLiteTensor& tensor, bool allow_per_channel) {
  ynn_type type = GetYnnType(tensor.type);
  if (type == ynn_type_invalid) return false;
  size_t element_count = YnnTypeElementCount(type);
  if (element_count > 1) {
    if (tensor.dims == nullptr) return false;
    size_t dense_dim =
        tensor.dims->size == 0 ? 1 : tensor.dims->data[tensor.dims->size - 1];
    if (dense_dim % element_count != 0) return false;
  }
  return IsSupportedQuantization(tensor, allow_per_channel);
}

bool QuantizationParamsEqual(const TfLiteTensor& tensor1,
                             const TfLiteTensor& tensor2) {
  if (IsQuantized(tensor1) != IsQuantized(tensor2)) return false;
  if (!IsQuantized(tensor1)) return true;
  return tensor1.params.scale == tensor2.params.scale &&
         tensor1.params.zero_point == tensor2.params.zero_point;
}

bool IsActivationSupported(TfLiteFusedActivation activation,
                           TfLiteType output_type) {
  if (activation == kTfLiteActNone) return true;

  return (activation == kTfLiteActRelu || activation == kTfLiteActRelu6 ||
          activation == kTfLiteActReluN1To1 || activation == kTfLiteActTanh ||
          activation == kTfLiteActSigmoid);
}

TfLiteFusedActivation GetFusedActivation(const TfLiteRegistration* registration,
                                         const TfLiteNode* node) {
  TfLiteFusedActivation activation = kTfLiteActNone;
  switch (registration->builtin_code) {
    case kTfLiteBuiltinAdd: {
      const auto* params =
          reinterpret_cast<const TfLiteAddParams*>(node->builtin_data);
      if (params) activation = params->activation;
      break;
    }
    case kTfLiteBuiltinSub: {
      const auto* params =
          reinterpret_cast<const TfLiteSubParams*>(node->builtin_data);
      if (params) activation = params->activation;
      break;
    }
    case kTfLiteBuiltinMul: {
      const auto* params =
          reinterpret_cast<const TfLiteMulParams*>(node->builtin_data);
      if (params) activation = params->activation;
      break;
    }
    case kTfLiteBuiltinDiv: {
      const auto* params =
          reinterpret_cast<const TfLiteDivParams*>(node->builtin_data);
      if (params) activation = params->activation;
      break;
    }
    case kTfLiteBuiltinMaxPool2d:
    case kTfLiteBuiltinAveragePool2d: {
      const auto* params =
          reinterpret_cast<const TfLitePoolParams*>(node->builtin_data);
      if (params) activation = params->activation;
      break;
    }
    default:
      break;
  }
  return activation;
}

TfLiteStatus DefineScalarConstant(TfLiteContext* context,
                                  ynn_subgraph_t subgraph, ynn_type type,
                                  double value, uint32_t* id_out) {
  TF_LITE_ENSURE(context, type != ynn_type_invalid);
  switch (type) {
    case ynn_type_fp32: {
      float f_val = static_cast<float>(value);
      TF_LITE_ENSURE_EQ(context,
                        ynn_define_tensor(subgraph, type, 0, nullptr, &f_val,
                                          YNN_VALUE_FLAG_COPY_DATA, id_out),
                        ynn_status_success);
      return kTfLiteOk;
    }
    case ynn_type_int32: {
      int32_t i_val = static_cast<int32_t>(value);
      TF_LITE_ENSURE_EQ(context,
                        ynn_define_tensor(subgraph, type, 0, nullptr, &i_val,
                                          YNN_VALUE_FLAG_COPY_DATA, id_out),
                        ynn_status_success);
      return kTfLiteOk;
    }
    case ynn_type_int8: {
      int8_t i8_val = static_cast<int8_t>(value);
      TF_LITE_ENSURE_EQ(context,
                        ynn_define_tensor(subgraph, type, 0, nullptr, &i8_val,
                                          YNN_VALUE_FLAG_COPY_DATA, id_out),
                        ynn_status_success);
      return kTfLiteOk;
    }
    case ynn_type_uint8: {
      uint8_t u8_val = static_cast<uint8_t>(value);
      TF_LITE_ENSURE_EQ(context,
                        ynn_define_tensor(subgraph, type, 0, nullptr, &u8_val,
                                          YNN_VALUE_FLAG_COPY_DATA, id_out),
                        ynn_status_success);
      return kTfLiteOk;
    }
    case ynn_type_fp64: {
      TF_LITE_ENSURE_EQ(context,
                        ynn_define_tensor(subgraph, type, 0, nullptr, &value,
                                          YNN_VALUE_FLAG_COPY_DATA, id_out),
                        ynn_status_success);
      return kTfLiteOk;
    }
    case ynn_type_fp16: {
      tflite::half f16_val(static_cast<float>(value));
      uint16_t bits = f16_val.to_bits();
      TF_LITE_ENSURE_EQ(context,
                        ynn_define_tensor(subgraph, type, 0, nullptr, &bits,
                                          YNN_VALUE_FLAG_COPY_DATA, id_out),
                        ynn_status_success);
      return kTfLiteOk;
    }
    case ynn_type_bf16: {
      uint16_t bits = FloatToBfloat16(static_cast<float>(value));
      TF_LITE_ENSURE_EQ(context,
                        ynn_define_tensor(subgraph, type, 0, nullptr, &bits,
                                          YNN_VALUE_FLAG_COPY_DATA, id_out),
                        ynn_status_success);
      return kTfLiteOk;
    }
    default:
      TF_LITE_ENSURE(context, false);
  }
}

TfLiteStatus DefineQuantizationParams(TfLiteContext* context,
                                      ynn_subgraph_t subgraph,
                                      const TfLiteTensor& tensor,
                                      uint32_t* scale_id, uint32_t* zp_id,
                                      int32_t zp_offset) {
  if (!IsQuantized(tensor)) {
    *scale_id = YNN_INVALID_VALUE_ID;
    *zp_id = YNN_INVALID_VALUE_ID;
    return kTfLiteOk;
  }
  const auto* params =
      static_cast<const TfLiteAffineQuantization*>(tensor.quantization.params);
  if (params && params->scale && params->scale->size > 1) {
    TF_LITE_ENSURE(context, params->zero_point != nullptr);
    TF_LITE_ENSURE(context, params->zero_point->size == params->scale->size ||
                                params->zero_point->size == 1);

    size_t size = params->scale->size;
    size_t dims[] = {size};

    TF_LITE_ENSURE_EQ(
        context,
        ynn_define_tensor(subgraph, ynn_type_fp32, 1, dims, params->scale->data,
                          YNN_VALUE_FLAG_COPY_DATA, scale_id),
        ynn_status_success);

    std::vector<int32_t> adjusted_zp(size);
    for (size_t i = 0; i < size; ++i) {
      int32_t zp = params->zero_point->size == 1 ? params->zero_point->data[0]
                                                 : params->zero_point->data[i];
      adjusted_zp[i] = zp + zp_offset;
    }
    bool all_zero = std::all_of(adjusted_zp.begin(), adjusted_zp.end(),
                                [](int32_t zp) { return zp == 0; });
    if (all_zero) {
      *zp_id = YNN_INVALID_VALUE_ID;
    } else {
      TF_LITE_ENSURE_EQ(context,
                        ynn_define_tensor(subgraph, ynn_type_int32, 1, dims,
                                          adjusted_zp.data(),
                                          YNN_VALUE_FLAG_COPY_DATA, zp_id),
                        ynn_status_success);
    }
  } else {
    float scale = 0.0f;
    int32_t zp = 0;
    if (params && params->scale) {
      scale = params->scale->data[0];
      zp = params->zero_point ? params->zero_point->data[0] : 0;
    } else {
      TFLITE_LOG_PROD(
          TFLITE_LOG_WARNING,
          "DefineQuantizationParams: missing or invalid affine quantization "
          "params for tensor, using legacy params. scale: %f, zp: %d",
          tensor.params.scale, tensor.params.zero_point);
      scale = tensor.params.scale;
      zp = tensor.params.zero_point;
      TF_LITE_ENSURE_MSG(context, scale != 0.0f,
                         "DefineQuantizationParams: scale is 0.0");
    }

    zp += zp_offset;

    TF_LITE_ENSURE_EQ(
        context,
        ynn_define_tensor(subgraph, ynn_type_fp32, 0, nullptr, &scale,
                          YNN_VALUE_FLAG_COPY_DATA, scale_id),
        ynn_status_success);

    if (zp == 0) {
      *zp_id = YNN_INVALID_VALUE_ID;
    } else {
      TF_LITE_ENSURE_EQ(context,
                        ynn_define_tensor(subgraph, ynn_type_int32, 0, nullptr,
                                          &zp, YNN_VALUE_FLAG_COPY_DATA, zp_id),
                        ynn_status_success);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus ApplyClamp(TfLiteContext* context, ynn_subgraph_t subgraph,
                        double min_val, double max_val, uint32_t input_id,
                        uint32_t& output_id, int output_tensor_index,
                        ynn_type internal_type) {
  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];
  const bool is_quantized =
      IsQuantized(output_tensor) &&
      (internal_type == ynn_type_int8 || internal_type == ynn_type_uint8);

  auto quantize_val = [&](double val) -> double {
    if (!is_quantized) return val;
    const float scale = output_tensor.params.scale;
    const int32_t zero_point = output_tensor.params.zero_point;
    if (internal_type == ynn_type_int8) {
      return QuantizeValue<int8_t>(val, scale, zero_point);
    } else {
      return QuantizeValue<uint8_t>(val, scale, zero_point);
    }
  };

  uint32_t current_input_id = input_id;

  uint32_t min_const_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineScalarConstant(
      context, subgraph, internal_type, quantize_val(min_val), &min_const_id));

  uint32_t max_output_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_EQ(
      context,
      ynn_define_binary(subgraph, ynn_binary_max, current_input_id,
                        min_const_id, &max_output_id, 0),
      ynn_status_success);

  current_input_id = max_output_id;

  uint32_t max_const_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(DefineScalarConstant(
      context, subgraph, internal_type, quantize_val(max_val), &max_const_id));

  TF_LITE_ENSURE_EQ(
      context,
      ynn_define_binary(subgraph, ynn_binary_min, current_input_id,
                        max_const_id, &output_id, 0),
      ynn_status_success);

  return kTfLiteOk;
}

TfLiteStatus ApplyActivation(TfLiteContext* context, ynn_subgraph_t subgraph,
                             TfLiteFusedActivation activation,
                             uint32_t input_id, uint32_t& output_id,
                             int output_tensor_index, ynn_type internal_type) {
  if (activation == kTfLiteActNone) {
    return kTfLiteOk;
  }

  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];
  bool is_quantized =
      IsQuantized(output_tensor) &&
      (internal_type == ynn_type_int8 || internal_type == ynn_type_uint8);

  auto quantize_val = [&](double val) -> double {
    if (!is_quantized) return val;
    float scale = output_tensor.params.scale;
    int32_t zero_point = output_tensor.params.zero_point;
    if (internal_type == ynn_type_int8) {
      return QuantizeValue<int8_t>(val, scale, zero_point);
    } else {
      return QuantizeValue<uint8_t>(val, scale, zero_point);
    }
  };

  uint32_t current_input_id = input_id;

  if (activation == kTfLiteActRelu) {
    uint32_t min_const_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(DefineScalarConstant(
        context, subgraph, internal_type, quantize_val(0.0), &min_const_id));
    TF_LITE_ENSURE_EQ(context,
                      ynn_define_binary(subgraph, ynn_binary_max, input_id,
                                        min_const_id, &output_id, 0),
                      ynn_status_success);
  } else if (activation == kTfLiteActRelu6 ||
             activation == kTfLiteActReluN1To1) {
    double min_val = (activation == kTfLiteActReluN1To1) ? -1.0 : 0.0;
    double max_val = (activation == kTfLiteActRelu6) ? 6.0 : 1.0;
    return ApplyClamp(context, subgraph, min_val, max_val, input_id, output_id,
                      output_tensor_index, internal_type);
  } else if (activation == kTfLiteActTanh) {
    TF_LITE_ENSURE_EQ(context,
                      ynn_define_unary(subgraph, ynn_unary_tanh,
                                       current_input_id, &output_id, 0),
                      ynn_status_success);
  } else if (activation == kTfLiteActSigmoid) {
    TF_LITE_ENSURE_EQ(context,
                      ynn_define_unary(subgraph, ynn_unary_sigmoid,
                                       current_input_id, &output_id, 0),
                      ynn_status_success);
  } else {
    TF_LITE_ENSURE(context, false);
  }

  return kTfLiteOk;
}

TfLiteStatus DequantizeIfNeeded(TfLiteContext* context, ynn_subgraph_t subgraph,
                                TensorToValueIdMap& tensor_to_value_id,
                                int tensor_index, uint32_t val_id,
                                uint32_t* float_val_id) {
  const TfLiteTensor& tensor = context->tensors[tensor_index];
  if (IsQuantized(tensor)) {
    uint32_t scale_id = YNN_INVALID_VALUE_ID;
    uint32_t zp_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_STATUS(
        DefineQuantizationParams(context, subgraph, tensor, &scale_id, &zp_id));
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_dequantize(
        subgraph, val_id, zp_id, scale_id, ynn_type_fp32, float_val_id, 0));
  } else {
    *float_val_id = val_id;
  }
  return kTfLiteOk;
}

TfLiteStatus Quantize(TfLiteContext* context, ynn_subgraph_t subgraph,
                      TensorToValueIdMap& tensor_to_value_id, int tensor_index,
                      uint32_t float_val_id, uint32_t quant_val_id) {
  const TfLiteTensor& tensor = context->tensors[tensor_index];
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  TF_LITE_ENSURE_STATUS(
      DefineQuantizationParams(context, subgraph, tensor, &scale_id, &zp_id));
  ynn_type ynn_type = GetYnnType(tensor.type);
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(
      subgraph, float_val_id, ynn_type, zp_id, scale_id, &quant_val_id, 0));
  return kTfLiteOk;
}

uint32_t GetOrCreateValueId(TfLiteContext* context, ynn_subgraph_t subgraph,
                            TensorToValueIdMap& mapping, int tensor_index) {
  if (tensor_index < 0 || tensor_index >= context->tensors_size) {
    return YNN_INVALID_VALUE_ID;
  }

  auto it = mapping.find(tensor_index);
  if (it != mapping.end()) return it->second;

  const TfLiteTensor& tensor = context->tensors[tensor_index];
  uint32_t val_id = YNN_INVALID_VALUE_ID;
  ynn_type ynn_type = GetYnnType(tensor.type);

  if (ynn_type == ynn_type_invalid) {
    return YNN_INVALID_VALUE_ID;
  }
  if (tensor.dims->size > YNN_MAX_TENSOR_RANK) {
    return YNN_INVALID_VALUE_ID;
  }

  const void* data = nullptr;
  uint32_t flags = 0;
  size_t dims_data[YNN_MAX_TENSOR_RANK];
  const size_t* dims = nullptr;

  if (tensor.allocation_type == kTfLiteMmapRo) {
    data = tensor.data.raw;
    std::copy_n(tensor.dims->data, tensor.dims->size, dims_data);
    dims = dims_data;
  }

  ynn_status status = ynn_define_tensor(subgraph, ynn_type, tensor.dims->size,
                                        dims, data, flags, &val_id);
  if (status != ynn_status_success) return YNN_INVALID_VALUE_ID;
  mapping[tensor_index] = val_id;
  return val_id;
}

TfLiteStatus ImplementMutualBroadcasting(TfLiteContext* context,
                                         ynn_subgraph_t subgraph, int rank_a,
                                         int rank_b, int exclude_a,
                                         int exclude_b, uint32_t& current_a_id,
                                         uint32_t& current_b_id) {
  int max_rank = std::max(rank_a, rank_b);

  uint32_t input_a_id = current_a_id;
  uint32_t input_b_id = current_b_id;

  // 1. Expand A if needed.
  uint32_t expanded_a_id = input_a_id;
  if (max_rank > rank_a) {
    int diff = max_rank - rank_a;
    int32_t axes[YNN_MAX_TENSOR_RANK];
    std::iota(axes, axes + diff, 0);
    expanded_a_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, diff, axes, input_a_id, &expanded_a_id, 0));
  }

  // 2. Expand B if needed.
  uint32_t expanded_b_id = input_b_id;
  if (max_rank > rank_b) {
    int diff = max_rank - rank_b;
    int32_t axes[YNN_MAX_TENSOR_RANK];
    std::iota(axes, axes + diff, 0);
    expanded_b_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, diff, axes, input_b_id, &expanded_b_id, 0));
  }

  uint32_t broadcasted_a_id = expanded_a_id;
  if (max_rank > exclude_a) {
    broadcasted_a_id = YNN_INVALID_VALUE_ID;
    int num_axes = max_rank - exclude_a;
    int32_t axes[YNN_MAX_TENSOR_RANK];
    std::iota(axes, axes + num_axes, 0);
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_broadcast_like(subgraph, num_axes, axes, expanded_a_id,
                                  expanded_b_id, &broadcasted_a_id, 0));
  }

  uint32_t broadcasted_b_id = expanded_b_id;
  if (max_rank > exclude_b) {
    broadcasted_b_id = YNN_INVALID_VALUE_ID;
    int num_axes = max_rank - exclude_b;
    int32_t axes[YNN_MAX_TENSOR_RANK];
    std::iota(axes, axes + num_axes, 0);
    TF_LITE_ENSURE_YNN_STATUS(
        ynn_define_broadcast_like(subgraph, num_axes, axes, expanded_b_id,
                                  expanded_a_id, &broadcasted_b_id, 0));
  }

  current_a_id = broadcasted_a_id;
  current_b_id = broadcasted_b_id;
  return kTfLiteOk;
}

TfLiteStatus DefineYnnStencil(TfLiteContext* context, ynn_subgraph_t subgraph,
                              const TfLiteTensor& input_tensor,
                              uint32_t input_id, size_t filter_height,
                              size_t filter_width, size_t stride_height,
                              size_t stride_width, size_t dilation_height,
                              size_t dilation_width, TfLitePadding padding,
                              float padding_value, uint32_t* stencil_id) {
  uint32_t padding_id = YNN_INVALID_VALUE_ID;
  if (padding == kTfLitePaddingSame) {
    ynn_type padding_type = GetYnnType(input_tensor.type);
    TF_LITE_ENSURE(context, padding_type != ynn_type_invalid);
    TF_LITE_ENSURE_STATUS(DefineScalarConstant(context, subgraph, padding_type,
                                               padding_value, &padding_id));
  }

  const int32_t stencil_axes[] = {1, 2};
  const int32_t new_axes[] = {-3, -2};
  const size_t stencil_dims[] = {filter_height, filter_width};
  const size_t stencil_strides[] = {stride_height, stride_width};
  const size_t stencil_dilations[] = {dilation_height, dilation_width};

  TF_LITE_ENSURE_YNN_STATUS(ynn_define_stencil_copy(
      subgraph, /*num_stencils=*/2, stencil_axes, new_axes, stencil_dims,
      stencil_strides, stencil_dilations, input_id, padding_id, stencil_id,
      /*flags=*/0));

  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
