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

#include <stdint.h>

#include <memory>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"

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
  per_channel_op_params.zero_point = quantization_params->zero_point->data;
  const int8_t* input_data;
  const size_t bytes_unpacked = input->bytes * 2;
  auto unpacked_input_data = std::make_unique<int8_t[]>(bytes_unpacked);

  if (input->type == kTfLiteInt4) {
    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
        GetTensorData<int8_t>(input), GetTensorShape(input).FlatSize(),
        unpacked_input_data.get());
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
    case kTfLiteInt4:
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
  if (IsQuantizedPerChannel(input)) {
    return PerChannelDequantizeImpl(context, node, input, output);
  }
  DequantizationParams op_params;
  op_params.zero_point = input->params.zero_point;
  op_params.scale = input->params.scale;
  const int8_t* input_data;
  const size_t bytes_unpacked = input->bytes * 2;
  auto unpacked_input_data = std::make_unique<int8_t[]>(bytes_unpacked);

  if (input->type == kTfLiteInt4) {
    // Use GetTensorShape(input).FlatSize() for num_elements.
    tflite::tensor_utils::UnpackDenseInt4IntoInt8(
        GetTensorData<int8_t>(input), GetTensorShape(input).FlatSize(),
        unpacked_input_data.get());
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
    case kTfLiteInt4:
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
