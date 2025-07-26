/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

#include <functional>
#include <type_traits>

#include "Eigen/Core"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace range {
namespace {

constexpr int kStartTensor = 0;
constexpr int kLimitTensor = 1;
constexpr int kDeltaTensor = 2;
constexpr int kOutputTensor = 0;

struct OpData {
  // Indicates that 'Eval' is a noop as the output as written during 'Prepare'.
  bool noop;
};

template <typename T>
float dequantize_value(const TfLiteTensor* input) {
  const T quantized_input_value = *GetTensorData<T>(input);
  int32_t zero_point = input->params.zero_point;
  const double scale = input->params.scale;
  return (scale * (quantized_input_value - zero_point));
}

template <typename T>
T quantize_value(const float value, const double scale, int32_t zero_point) {
  static constexpr int32_t min_val = std::numeric_limits<T>::min();
  static constexpr int32_t max_val = std::numeric_limits<T>::max();

  int32_t unclamped =
      static_cast<int32_t>(TfLiteRound(value / static_cast<float>(scale))) +
      zero_point;
  int32_t clamped = std::min(std::max(unclamped, min_val), max_val);

  return static_cast<T>(clamped);
}

template <typename T>
TfLiteStatus GetSize(TfLiteContext* context, T start, T limit, T delta,
                     int* size) {
  // TF_LITE_ENSURE(context, !std::equal_to<T>()(delta, 0)); // std::equal_to
  // does not support float16 and bfloat16
  TF_LITE_ENSURE(context, !(delta == 0));
  TF_LITE_ENSURE(
      context, (start >= limit && delta < 0) || (start <= limit && delta > 0));
  *size =
      (std::is_integral<T>::value
           ? ((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta))
           : std::ceil(std::abs((limit - start) / delta)));
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus GetSizeQuantized(TfLiteContext* context, const TfLiteTensor* start,
                              const TfLiteTensor* limit,
                              const TfLiteTensor* delta, int* size) {
  const float dequantized_start_value = dequantize_value<T>(start);
  const float dequantized_delta_value = dequantize_value<T>(delta);
  const float dequantized_limit_value = dequantize_value<T>(limit);

  TF_LITE_ENSURE(context, !(dequantized_delta_value == 0));
  TF_LITE_ENSURE(context,
                 (dequantized_start_value >= dequantized_limit_value &&
                  dequantized_delta_value < 0) ||
                     (dequantized_start_value <= dequantized_limit_value &&
                      dequantized_delta_value > 0));
  *size = (std::is_integral<T>::value
               ? ((std::abs(dequantized_limit_value - dequantized_start_value) +
                   std::abs(dequantized_delta_value) - 1) /
                  std::abs(dequantized_delta_value))
               : std::ceil(std::abs(
                     (dequantized_limit_value - dequantized_start_value) /
                     dequantized_delta_value)));
  return kTfLiteOk;
}

TfLiteStatus ResizeOutput(TfLiteContext* context, const TfLiteTensor* start,
                          const TfLiteTensor* limit, const TfLiteTensor* delta,
                          TfLiteTensor* output) {
  // The output will always be a 1-d array.
  int size = 0;
  switch (start->type) {
    case kTfLiteInt32: {
      TF_LITE_ENSURE_OK(context,
                        GetSize(context, *GetTensorData<int32_t>(start),
                                *GetTensorData<int32_t>(limit),
                                *GetTensorData<int32_t>(delta), &size));
      break;
    }
    case kTfLiteInt64: {
      TF_LITE_ENSURE_OK(context,
                        GetSize(context, *GetTensorData<int64_t>(start),
                                *GetTensorData<int64_t>(limit),
                                *GetTensorData<int64_t>(delta), &size));
      break;
    }
    case kTfLiteFloat32: {
      TF_LITE_ENSURE_OK(context, GetSize(context, *GetTensorData<float>(start),
                                         *GetTensorData<float>(limit),
                                         *GetTensorData<float>(delta), &size));
      break;
    }
    case kTfLiteInt8: {
      if (start->quantization.type == kTfLiteAffineQuantization) {
        TF_LITE_ENSURE_OK(context, GetSizeQuantized<int8_t>(
                                        context, start, limit, delta, &size));
      } else {
        TF_LITE_ENSURE_OK(context,
                          GetSize(context, *GetTensorData<int8_t>(start),
                                  *GetTensorData<int8_t>(limit),
                                  *GetTensorData<int8_t>(delta), &size));
      }
      break;
    }
    case kTfLiteInt16: {
      if (start->quantization.type == kTfLiteAffineQuantization) {
        TF_LITE_ENSURE_OK(context, GetSizeQuantized<int16_t>(
                                       context, start, limit, delta, &size));
      } else {
        TF_LITE_ENSURE_OK(context,
                          GetSize(context, *GetTensorData<int16_t>(start),
                                  *GetTensorData<int16_t>(limit),
                                  *GetTensorData<int16_t>(delta), &size));
      }
      break;
    }
    case kTfLiteFloat16: {
      TF_LITE_ENSURE_OK(context,
                        GetSize(context, *GetTensorData<Eigen::half>(start),
                                *GetTensorData<Eigen::half>(limit),
                                *GetTensorData<Eigen::half>(delta), &size));
      break;
    }
    case kTfLiteBFloat16: {
      TF_LITE_ENSURE_OK(context,
                        GetSize(context, *GetTensorData<Eigen::bfloat16>(start),
                                *GetTensorData<Eigen::bfloat16>(limit),
                                *GetTensorData<Eigen::bfloat16>(delta), &size));
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Unknown data type: %d", start->type);
      return kTfLiteError;
    }
  }
  TfLiteIntArray* output_shape_array = TfLiteIntArrayCreate(1);
  output_shape_array->data[0] = size;
  return context->ResizeTensor(context, output, output_shape_array);
}

template <typename T>
void CalculateRange(const TfLiteTensor* start, const TfLiteTensor* delta,
                    TfLiteTensor* output) {
  const T start_value = *GetTensorData<T>(start);
  const T delta_value = *GetTensorData<T>(delta);
  T* output_data = GetTensorData<T>(output);
  const int num_elements = NumElements(output);
  T value = start_value;
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = value;
    value += delta_value;
  }
}

template <typename T>
void CalculateRangeQuantized(const TfLiteTensor* start,
                             const TfLiteTensor* delta, TfLiteTensor* output) {
  int32_t zero_point = start->params.zero_point;
  const double scale = start->params.scale;

  const float dequantized_start_value = dequantize_value<T>(start);
  const float dequantized_delta_value = dequantize_value<T>(delta);

  T* output_data = GetTensorData<T>(output);

  const int num_elements = NumElements(output);
  float value = dequantized_start_value;
  for (int i = 0; i < num_elements; ++i) {
    output_data[i] = quantize_value<T>(value, scale, zero_point);
    value += dequantized_delta_value;
  }
}

TfLiteStatus EvalImpl(TfLiteContext* context, const TfLiteTensor* start,
                      const TfLiteTensor* delta, TfLiteTensor* output) {
  switch (output->type) {
    case kTfLiteInt32: {
      CalculateRange<int32_t>(start, delta, output);
      break;
    }
    case kTfLiteFloat32: {
      CalculateRange<float>(start, delta, output);
      break;
    }
    case kTfLiteInt64: {
      CalculateRange<int64_t>(start, delta, output);
      break;
    }
    case kTfLiteInt8: {
      start->quantization.type == kTfLiteAffineQuantization 
          ? CalculateRangeQuantized<int8_t>(start, delta, output)
          : CalculateRange<int8_t>(start, delta, output);
      break;
    }
    case kTfLiteInt16: {
      start->quantization.type == kTfLiteAffineQuantization
          ? CalculateRangeQuantized<int16_t>(start, delta, output)
          : CalculateRange<int16_t>(start, delta, output);
      break;
    }
    case kTfLiteFloat16: {
      CalculateRange<Eigen::half>(start, delta, output);
      break;
    }
    case kTfLiteBFloat16: {
      CalculateRange<Eigen::bfloat16>(start, delta, output);
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Unsupported data type: %d", output->type);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  op_data->noop = false;

  const TfLiteTensor* start;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartTensor, &start));
  const TfLiteTensor* limit;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kLimitTensor, &limit));
  const TfLiteTensor* delta;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDeltaTensor, &delta));
  // Make sure all the inputs are scalars.
  TF_LITE_ENSURE_EQ(context, NumDimensions(start), 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(limit), 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(delta), 0);

  // Currently only supports int32, int64 and float.
  // TODO(b/117912892): Support quantization as well.
  const auto dtype = start->type;
  if (dtype != kTfLiteFloat32 && dtype != kTfLiteInt32 &&
      dtype != kTfLiteInt64 && dtype != kTfLiteInt8 &&
      dtype != kTfLiteFloat16 && dtype != kTfLiteBFloat16 &&
      dtype != kTfLiteInt16) {
    TF_LITE_KERNEL_LOG(context, "Unknown index output data type: %s",
                       TfLiteTypeGetName(dtype));
    return kTfLiteError;
  }

  TF_LITE_ENSURE_TYPES_EQ(context, limit->type, dtype);
  TF_LITE_ENSURE_TYPES_EQ(context, delta->type, dtype);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  output->type = dtype;

  if (IsConstantOrPersistentTensor(start) &&
      IsConstantOrPersistentTensor(limit) &&
      IsConstantOrPersistentTensor(delta)) {
    SetTensorToPersistentRo(output);
    TF_LITE_ENSURE_OK(context,
                      ResizeOutput(context, start, limit, delta, output));

    op_data->noop = true;
    return EvalImpl(context, start, delta, output);
  }

  SetTensorToDynamic(output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* start;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kStartTensor, &start));
  const TfLiteTensor* limit;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kLimitTensor, &limit));
  const TfLiteTensor* delta;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kDeltaTensor, &delta));

  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  if (op_data->noop) {
    return kTfLiteOk;
  }
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutput(context, start, limit, delta, output));
  }
  return EvalImpl(context, start, delta, output);
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

}  // namespace
}  // namespace range

TfLiteRegistration* Register_RANGE() {
  static TfLiteRegistration r = {range::Init, range::Free, range::Prepare,
                                 range::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
