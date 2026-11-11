/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_DEQUANTIZE_H_
#define TENSORFLOW_LITE_KERNELS_DEQUANTIZE_H_

#include <stddef.h>
#include <stdint.h>

#include <algorithm>
#include <limits>
#include <memory>
#include <vector>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/common.h"
#if defined(TFLITE_ENABLE_EXTRA_REFERENCE_KERNELS)
#include "tensorflow/lite/kernels/internal/float8.h"
#endif
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace dequantize {

// This file has two implementation of Dequantize.
enum KernelType {
  kReference,
  kGenericOptimized,
};

inline bool IsQuantizedPerChannel(const TfLiteTensor* input) {
  if (input->quantization.type == kTfLiteAffineQuantization &&
      input->quantization.params) {
    auto* quant_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    return (quant_params->scale && quant_params->scale->size > 1);
  }
  return false;
}

// Returns the N-D blockwise quantization parameters of `input`, or nullptr if
// it is not quantized that way.
inline const TfLiteBlockwiseQuantizationV2* GetBlockwiseQuantizationV2(
    const TfLiteTensor* input) {
  if (input->quantization.type != kTfLiteBlockwiseQuantizationV2) {
    return nullptr;
  }
  return static_cast<const TfLiteBlockwiseQuantizationV2*>(
      input->quantization.params);
}

// Checks that `params` describe a valid N-D blockwise quantization of `input`:
// one `block_shape` entry per dimension, and scale / zero point tensors of the
// same rank whose extents are either 1 (broadcast) or the number of blocks
// along that dimension.
inline TfLiteStatus ValidateBlockwiseQuantizationV2(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteBlockwiseQuantizationV2* params) {
  TF_LITE_ENSURE(context, params != nullptr);
  TF_LITE_ENSURE(context, params->block_shape != nullptr);
  const int rank = input->dims->size;
  TF_LITE_ENSURE_EQ(context, params->block_shape->size, rank);
  for (int d = 0; d < rank; ++d) {
    TF_LITE_ENSURE(context, params->block_shape->data[d] > 0);
  }

  auto validate_params_tensor = [&](int index, bool is_scale) -> TfLiteStatus {
    TF_LITE_ENSURE(context, index >= 0 && index < context->tensors_size);
    const TfLiteTensor* params_tensor = &context->tensors[index];
    if (is_scale) {
      TF_LITE_ENSURE(context, params_tensor->type == kTfLiteFloat32 ||
                                  params_tensor->type == kTfLiteFloat16 ||
                                  params_tensor->type == kTfLiteBFloat16);
    } else {
      TF_LITE_ENSURE(context, params_tensor->type == kTfLiteFloat32 ||
                                  params_tensor->type == kTfLiteInt32 ||
                                  params_tensor->type == kTfLiteInt16 ||
                                  params_tensor->type == kTfLiteInt8);
    }
    TF_LITE_ENSURE_EQ(context, params_tensor->dims->size, rank);
    for (int d = 0; d < rank; ++d) {
      const int dim = input->dims->data[d];
      TF_LITE_ENSURE(context, dim >= 0);
      const int block = params->block_shape->data[d];
      const int num_blocks = (dim == 0) ? 0 : (1 + (dim - 1) / block);
      const int extent = params_tensor->dims->data[d];
      TF_LITE_ENSURE(context, extent == 1 || extent == num_blocks);
    }
    return kTfLiteOk;
  };
  TF_LITE_ENSURE_OK(context,
                    validate_params_tensor(params->scale, /*is_scale=*/true));
  if (params->zero_point != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context, validate_params_tensor(params->zero_point,
                                                      /*is_scale=*/false));
  }
  return kTfLiteOk;
}

// Reads element `index` of a blockwise quantization parameter tensor as a
// float.
inline float GetBlockwiseParamAsFloat(const TfLiteTensor* tensor,
                                      int64_t index) {
  switch (tensor->type) {
    case kTfLiteFloat32:
      return GetTensorData<float>(tensor)[index];
    case kTfLiteFloat16:
      return static_cast<float>(Eigen::numext::bit_cast<Eigen::half>(
          GetTensorData<TfLiteFloat16>(tensor)[index].data));
    case kTfLiteBFloat16:
      return static_cast<float>(Eigen::numext::bit_cast<Eigen::bfloat16>(
          GetTensorData<TfLiteBFloat16>(tensor)[index].data));
    case kTfLiteInt32:
      return GetTensorData<int32_t>(tensor)[index];
    case kTfLiteInt16:
      return GetTensorData<int16_t>(tensor)[index];
    case kTfLiteInt8:
      return GetTensorData<int8_t>(tensor)[index];
    default:
      return 0.0f;
  }
}

// Reads the elements of a blockwise quantization parameter tensor as floats.
inline std::vector<float> ReadBlockwiseParamsAsFloat(
    const TfLiteTensor* tensor) {
  const int64_t size = GetTensorShape(tensor).FlatSize();
  std::vector<float> values(size);
  for (int64_t i = 0; i < size; ++i) {
    values[i] = GetBlockwiseParamAsFloat(tensor, i);
  }
  return values;
}

// Returns element `index` (in row-major order) of the raw data `data` of a
// blockwise quantized tensor of type `kType`, which must be INT2, INT4, UINT4,
// INT8, UINT8 or INT16. Sub-byte values are packed low bits first. Unlike
// `GetBlockwiseQuantizedValues`, this does not unpack the whole tensor, so it
// suits kernels that only read a few slices of a large tensor.
template <TfLiteType kType>
inline int32_t GetBlockwiseQuantizedValue(const uint8_t* data, int64_t index) {
  static_assert(kType == kTfLiteInt2 || kType == kTfLiteInt4 ||
                kType == kTfLiteUInt4 || kType == kTfLiteInt8 ||
                kType == kTfLiteUInt8 || kType == kTfLiteInt16);
  if constexpr (kType == kTfLiteInt2) {
    const int shift = static_cast<int>(index & 3) * 2;
    // Move the 2 bits to the top of an int8 and shift back to sign extend.
    return static_cast<int8_t>(
               static_cast<uint8_t>(data[index >> 2] << (6 - shift))) >>
           6;
  } else if constexpr (kType == kTfLiteInt4) {
    const int shift = static_cast<int>(index & 1) * 4;
    return static_cast<int8_t>(
               static_cast<uint8_t>(data[index >> 1] << (4 - shift))) >>
           4;
  } else if constexpr (kType == kTfLiteUInt4) {
    return (data[index >> 1] >> (static_cast<int>(index & 1) * 4)) & 0x0F;
  } else if constexpr (kType == kTfLiteInt8) {
    return reinterpret_cast<const int8_t*>(data)[index];
  } else if constexpr (kType == kTfLiteUInt8) {
    return data[index];
  } else {
    return reinterpret_cast<const int16_t*>(data)[index];
  }
}

// Checks that the scale and zero point tensors of `params` are constant or
// persistent, and that they and `input` hold as many bytes as their shapes
// require. Call it before reading the data of any of these tensors.
inline TfLiteStatus ValidateBlockwiseQuantizationV2Buffers(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteBlockwiseQuantizationV2* params) {
  auto validate_buffer = [&](const TfLiteTensor* tensor,
                             bool require_constant_or_persistent) {
    if (require_constant_or_persistent) {
      TF_LITE_ENSURE(context, IsConstantOrPersistentTensor(tensor));
    }
    size_t required_bytes = 0;
    TF_LITE_ENSURE_OK(
        context, BytesRequired(tensor->type, tensor->dims->data,
                               tensor->dims->size, &required_bytes, context));
    TF_LITE_ENSURE(context, tensor->bytes >= required_bytes);
    if (required_bytes > 0) {
      TF_LITE_ENSURE(context, tensor->data.raw != nullptr);
    }
    return kTfLiteOk;
  };
  TF_LITE_ENSURE_OK(context,
                    validate_buffer(&context->tensors[params->scale],
                                    /*require_constant_or_persistent=*/true));
  if (params->zero_point != kTfLiteOptionalTensor) {
    TF_LITE_ENSURE_OK(context,
                      validate_buffer(&context->tensors[params->zero_point],
                                      /*require_constant_or_persistent=*/true));
  }
  return validate_buffer(input, /*require_constant_or_persistent=*/false);
}

// The integer values of a blockwise quantized tensor. Exactly one of the data
// pointers is set; sub-byte values are unpacked into `unpacked`, which
// `int8_data` then points into.
struct BlockwiseQuantizedValues {
  std::vector<int8_t> unpacked;
  const int8_t* int8_data = nullptr;
  const uint8_t* uint8_data = nullptr;
  const int16_t* int16_data = nullptr;
};

// Fills `values` with the integer values of the blockwise quantized `input`,
// which must be of type INT2, INT4, UINT4, INT8, UINT8 or INT16.
inline TfLiteStatus GetBlockwiseQuantizedValues(
    TfLiteContext* context, const TfLiteTensor* input,
    BlockwiseQuantizedValues* values) {
  *values = BlockwiseQuantizedValues();
  const int64_t flat_size = GetTensorShape(input).FlatSize();
  switch (input->type) {
    case kTfLiteInt2:
    case kTfLiteInt4:
    case kTfLiteUInt4:
      TF_LITE_ENSURE(context, flat_size <= std::numeric_limits<int>::max());
      // An empty tensor has no data to unpack (and possibly a null buffer).
      if (flat_size == 0) return kTfLiteOk;
      values->unpacked.resize(flat_size);
      tflite::tensor_utils::UnpackPackedIntToInt8(
          GetTensorData<int8_t>(input), static_cast<int>(flat_size),
          /*bit_width=*/input->type == kTfLiteInt2 ? 2 : 4,
          values->unpacked.data(),
          /*unpack_unsigned=*/input->type == kTfLiteUInt4);
      values->int8_data = values->unpacked.data();
      return kTfLiteOk;
    case kTfLiteInt8:
      values->int8_data = GetTensorData<int8_t>(input);
      return kTfLiteOk;
    case kTfLiteUInt8:
      values->uint8_data = GetTensorData<uint8_t>(input);
      return kTfLiteOk;
    case kTfLiteInt16:
      values->int16_data = GetTensorData<int16_t>(input);
      return kTfLiteOk;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type %s not supported for blockwise quantization.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

// Returns the strides of a blockwise scale or zero point tensor of rank `rank`
// in units of blocks, with 0 for broadcast (extent 1) dimensions. A null
// `params_tensor` (no zero point) yields all-zero strides.
inline std::vector<int64_t> GetBlockwiseParamsStrides(
    const TfLiteTensor* params_tensor, int rank) {
  std::vector<int64_t> strides(rank, 0);
  if (params_tensor == nullptr) return strides;
  int64_t stride = 1;
  for (int d = rank - 1; d >= 0; --d) {
    const int extent = params_tensor->dims->data[d];
    strides[d] = (extent == 1) ? 0 : stride;
    stride *= extent;
  }
  return strides;
}

// Dequantizes `input`, which is N-D blockwise quantized, into `output`:
//
//   output[i] = scale[b] * (input[i] - zero_point[b]),
//   b = floor(i / block_shape)
//
// where a scale or zero point tensor of extent 1 along a dimension broadcasts
// along it, and a missing zero point tensor means a zero point of 0.
inline TfLiteStatus BlockwiseDequantizeImpl(TfLiteContext* context,
                                            const TfLiteTensor* input,
                                            TfLiteTensor* output) {
  const TfLiteBlockwiseQuantizationV2* params =
      GetBlockwiseQuantizationV2(input);
  TF_LITE_ENSURE_OK(context,
                    ValidateBlockwiseQuantizationV2(context, input, params));
  TF_LITE_ENSURE_OK(
      context, ValidateBlockwiseQuantizationV2Buffers(context, input, params));

  const RuntimeShape shape = GetTensorShape(input);
  const int rank = shape.DimensionsCount();
  const int64_t flat_size = shape.FlatSize();
  if (flat_size > 0) {
    TF_LITE_ENSURE(context, GetTensorData<float>(output) != nullptr);
  }

  BlockwiseQuantizedValues quantized_values;
  TF_LITE_ENSURE_OK(
      context, GetBlockwiseQuantizedValues(context, input, &quantized_values));

  const TfLiteTensor* scale_tensor = &context->tensors[params->scale];
  const std::vector<float> scales = ReadBlockwiseParamsAsFloat(scale_tensor);
  std::vector<float> zero_points = {0.0f};
  const TfLiteTensor* zero_point_tensor = nullptr;
  if (params->zero_point != kTfLiteOptionalTensor) {
    zero_point_tensor = &context->tensors[params->zero_point];
    zero_points = ReadBlockwiseParamsAsFloat(zero_point_tensor);
  }

  const std::vector<int64_t> scale_strides =
      GetBlockwiseParamsStrides(scale_tensor, rank);
  const std::vector<int64_t> zero_point_strides =
      GetBlockwiseParamsStrides(zero_point_tensor, rank);

  auto read_q = [&](int64_t index) -> float {
    if (quantized_values.uint8_data != nullptr) {
      return static_cast<float>(quantized_values.uint8_data[index]);
    }
    if (quantized_values.int16_data != nullptr) {
      return static_cast<float>(quantized_values.int16_data[index]);
    }
    return static_cast<float>(quantized_values.int8_data[index]);
  };

  // Walk the tensor in row-major order, tracking the coordinates of the
  // current element and of its block, and the parameter indices they map to.
  std::vector<int> coords(rank, 0);
  std::vector<int> block_coords(rank, 0);
  int64_t scale_index = 0;
  int64_t zero_point_index = 0;
  float* output_data = GetTensorData<float>(output);
  for (int64_t i = 0; i < flat_size; ++i) {
    output_data[i] =
        scales[scale_index] * (read_q(i) - zero_points[zero_point_index]);
    for (int d = rank - 1; d >= 0; --d) {
      const int block = params->block_shape->data[d];
      if (++coords[d] < shape.Dims(d)) {
        if (coords[d] % block == 0) {
          ++block_coords[d];
          scale_index += scale_strides[d];
          zero_point_index += zero_point_strides[d];
        }
        break;
      }
      scale_index -= block_coords[d] * scale_strides[d];
      zero_point_index -= block_coords[d] * zero_point_strides[d];
      coords[d] = 0;
      block_coords[d] = 0;
    }
  }
  return kTfLiteOk;
}

#if defined(TFLITE_ENABLE_EXTRA_REFERENCE_KERNELS)
template <typename Float8T>
inline void DequantizeFloat8(const TfLiteTensor* input, TfLiteTensor* output) {
  const uint8_t* input_data = GetTensorData<uint8_t>(input);
  float* output_data = GetTensorData<float>(output);
  const int flat_size =
      MatchingFlatSize(GetTensorShape(input), GetTensorShape(output));
  for (int i = 0; i < flat_size; ++i) {
    output_data[i] = static_cast<float>(Float8T::FromRep(input_data[i]));
  }
}
#endif

inline TfLiteStatus PerChannelDequantizeImpl(TfLiteContext* context,
                                             TfLiteNode* node,
                                             const TfLiteTensor* input,
                                             TfLiteTensor* output) {
  const auto* quantization_params =
      reinterpret_cast<const TfLiteAffineQuantization*>(
          input->quantization.params);
  PerChannelDequantizationParams per_channel_op_params;
  per_channel_op_params.quantized_dimension =
      quantization_params->quantized_dimension;
  per_channel_op_params.scale = quantization_params->scale->data;
  std::vector<int> zero_points;
  if (quantization_params->zero_point->size ==
      quantization_params->scale->size) {
    per_channel_op_params.zero_point = quantization_params->zero_point->data;
  } else {
    zero_points.resize(quantization_params->scale->size,
                       quantization_params->zero_point->data[0]);
    per_channel_op_params.zero_point = zero_points.data();
  }
  const int8_t* input_data;
  size_t bytes_unpacked;
  if (input->type == kTfLiteInt2) {
    bytes_unpacked = input->bytes * 4;
  } else if (input->type == kTfLiteInt4 || input->type == kTfLiteUInt4) {
    bytes_unpacked = input->bytes * 2;
  } else {
    bytes_unpacked = input->bytes;
  }
  auto unpacked_input_data = std::make_unique<int8_t[]>(bytes_unpacked);

  if (input->type == kTfLiteInt4 || input->type == kTfLiteUInt4) {
    tflite::tensor_utils::UnpackPackedIntToInt8(
        GetTensorData<int8_t>(input), GetTensorShape(input).FlatSize(),
        /*bit_width=*/4, unpacked_input_data.get(),
        /*unpack_unsigned=*/input->type == kTfLiteUInt4);
    input_data = unpacked_input_data.get();
  } else if (input->type == kTfLiteInt2) {
    tflite::tensor_utils::UnpackPackedIntToInt8(
        GetTensorData<int8_t>(input), GetTensorShape(input).FlatSize(),
        /*bit_width=*/2, unpacked_input_data.get());
    input_data = unpacked_input_data.get();
  } else {
    input_data = GetTensorData<int8_t>(input);
  }

  switch (input->type) {
    case kTfLiteUInt8:
      reference_ops::PerChannelDequantize<uint8_t>(
          per_channel_op_params, GetTensorShape(input),
          GetTensorData<uint8_t>(input), GetTensorShape(output),
          GetTensorData<float>(output));
      break;
    case kTfLiteInt2:
    case kTfLiteInt4:
    case kTfLiteUInt4:
    case kTfLiteInt8:
      reference_ops::PerChannelDequantize<int8_t>(
          per_channel_op_params, GetTensorShape(input), input_data,
          GetTensorShape(output), GetTensorData<float>(output));
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %d not supported for per-channel.",
                         input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus DequantizeImpl(TfLiteContext* context, TfLiteNode* node,
                            const TfLiteTensor* input, TfLiteTensor* output) {
  if (GetBlockwiseQuantizationV2(input) != nullptr) {
    return BlockwiseDequantizeImpl(context, input, output);
  }
  if (IsQuantizedPerChannel(input)) {
    return PerChannelDequantizeImpl(context, node, input, output);
  }
  DequantizationParams op_params;
  op_params.zero_point = input->params.zero_point;
  op_params.scale = input->params.scale;
  const int8_t* input_data;
  size_t bytes_unpacked;
  if (input->type == kTfLiteInt2) {
    bytes_unpacked = input->bytes * 4;
  } else if (input->type == kTfLiteInt4 || input->type == kTfLiteUInt4) {
    bytes_unpacked = input->bytes * 2;
  } else {
    bytes_unpacked = input->bytes;
  }
  auto unpacked_input_data = std::make_unique<int8_t[]>(bytes_unpacked);

  if (input->type == kTfLiteInt4 || input->type == kTfLiteUInt4) {
    // Use GetTensorShape(input).FlatSize() for num_elements.
    tflite::tensor_utils::UnpackPackedIntToInt8(
        GetTensorData<int8_t>(input), GetTensorShape(input).FlatSize(),
        /*bit_width=*/4, unpacked_input_data.get(),
        /*unpack_unsigned=*/input->type == kTfLiteUInt4);
    input_data = unpacked_input_data.get();
  } else if (input->type == kTfLiteInt2) {
    // Use GetTensorShape(input).FlatSize() for num_elements.
    tflite::tensor_utils::UnpackPackedIntToInt8(
        GetTensorData<int8_t>(input), GetTensorShape(input).FlatSize(),
        /*bit_width=*/2, unpacked_input_data.get());
    input_data = unpacked_input_data.get();
  } else {
    input_data = GetTensorData<int8_t>(input);
  }

  switch (input->type) {
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        reference_ops::Dequantize(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        optimized_ops::Dequantize(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      break;
    case kTfLiteInt2:
    case kTfLiteInt4:
    case kTfLiteUInt4:
    case kTfLiteInt8:
      if (kernel_type == kReference) {
        reference_integer_ops::Dequantize<int8_t>(
            op_params, GetTensorShape(input), input_data,
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        optimized_ops::Dequantize(op_params, GetTensorShape(input), input_data,
                                  GetTensorShape(output),
                                  GetTensorData<float>(output));
      }
      break;
    case kTfLiteInt16:
      if (kernel_type == kReference) {
        reference_integer_ops::Dequantize<int16_t>(
            op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        optimized_ops::Dequantize(
            op_params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      break;
    case kTfLiteFloat16: {
      const Eigen::half* half_data = reinterpret_cast<const Eigen::half*>(
          GetTensorData<TfLiteFloat16>(input));
      reference_ops::Dequantize(GetTensorShape(input), half_data,
                                GetTensorShape(output),
                                GetTensorData<float>(output));
      break;
    }
#if defined(TFLITE_ENABLE_EXTRA_REFERENCE_KERNELS)
    case kTfLiteFloat8E4M3FN:
      DequantizeFloat8<float8_internal::Float8E4M3FN>(input, output);
      break;
    case kTfLiteFloat8E5M2:
      DequantizeFloat8<float8_internal::Float8E5M2>(input, output);
      break;
#endif
    default:
      TF_LITE_KERNEL_LOG(context, "Type %d not supported.", input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace dequantize
}  // namespace builtin
}  // namespace ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_DEQUANTIZE_H_
