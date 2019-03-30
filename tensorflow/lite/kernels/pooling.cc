/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pooling {

// This file has two implementation of each pooling op.
enum KernelType {
  kReference,
  kGenericOptimized,
};

enum PoolType {
  kAverage,
  kMax,
  kL2,
};

struct OpData {
  TfLitePaddingValues padding;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

template <PoolType pool_type>
TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  // Matching GetWindowedOutputSize in TensorFlow.
  auto padding = params->padding;
  auto compute_out_size = [padding](int image_size, int filter_size,
                                    int stride) -> int {
    return padding == kTfLitePaddingSame
               ? (image_size + stride - 1) / stride
               : padding == kTfLitePaddingValid
                     ? (image_size - filter_size + stride) / stride
                     : 0;
  };

  int out_width =
      compute_out_size(width, params->filter_width, params->stride_width);
  int out_height =
      compute_out_size(height, params->filter_height, params->stride_height);

  data->padding.height = ComputePadding(params->stride_height, 1, height,
                                        params->filter_height, out_height);
  data->padding.width = ComputePadding(params->stride_width, 1, width,
                                       params->filter_width, out_width);

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    if (pool_type == kAverage || pool_type == kMax) {
      TF_LITE_ENSURE_EQ(context, input->params.scale, output->params.scale);
      TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                        output->params.zero_point);
    }
    if (pool_type == kL2) {
      // We currently don't have a quantized implementation of L2Pool
      TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
    }
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

template <KernelType kernel_type>
void AverageEvalFloat(TfLiteContext* context, TfLiteNode* node,
                      TfLitePoolParams* params, OpData* data,
                      const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_AVERAGE_POOL(type)                                       \
  tflite::PoolParams op_params;                                          \
  op_params.stride_height = params->stride_height;                       \
  op_params.stride_width = params->stride_width;                         \
  op_params.filter_height = params->filter_height;                       \
  op_params.filter_width = params->filter_width;                         \
  op_params.padding_values.height = data->padding.height;                \
  op_params.padding_values.width = data->padding.width;                  \
  op_params.float_activation_min = activation_min;                       \
  op_params.float_activation_max = activation_max;                       \
  type::AveragePool(op_params, GetTensorShape(input),                    \
                    GetTensorData<float>(input), GetTensorShape(output), \
                    GetTensorData<float>(output))
  if (kernel_type == kReference) {
    TF_LITE_AVERAGE_POOL(reference_ops);
  } else {
    TF_LITE_AVERAGE_POOL(optimized_ops);
  }
#undef TF_LITE_AVERAGE_POOL
}

template <KernelType kernel_type>
void AverageEvalQuantizedUint8(TfLiteContext* context, TfLiteNode* node,
                               TfLitePoolParams* params, OpData* data,
                               const TfLiteTensor* input,
                               TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  CalculateActivationRangeUint8(params->activation, output, &activation_min,
                                &activation_max);
#define TF_LITE_AVERAGE_POOL(type)                                         \
  tflite::PoolParams op_params;                                            \
  op_params.stride_height = params->stride_height;                         \
  op_params.stride_width = params->stride_width;                           \
  op_params.filter_height = params->filter_height;                         \
  op_params.filter_width = params->filter_width;                           \
  op_params.padding_values.height = data->padding.height;                  \
  op_params.padding_values.width = data->padding.width;                    \
  op_params.quantized_activation_min = activation_min;                     \
  op_params.quantized_activation_max = activation_max;                     \
  type::AveragePool(op_params, GetTensorShape(input),                      \
                    GetTensorData<uint8_t>(input), GetTensorShape(output), \
                    GetTensorData<uint8_t>(output))
  if (kernel_type == kReference) {
    TF_LITE_AVERAGE_POOL(reference_ops);
  } else {
    TF_LITE_AVERAGE_POOL(optimized_ops);
  }
#undef TF_LITE_AVERAGE_POOL
}

template <KernelType kernel_type>
void AverageEvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                              TfLitePoolParams* params, OpData* data,
                              const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  CalculateActivationRangeInt8(params->activation, output, &activation_min,
                               &activation_max);
#define TF_LITE_AVERAGE_POOL(type)                                        \
  tflite::PoolParams op_params;                                           \
  op_params.stride_height = params->stride_height;                        \
  op_params.stride_width = params->stride_width;                          \
  op_params.filter_height = params->filter_height;                        \
  op_params.filter_width = params->filter_width;                          \
  op_params.padding_values.height = data->padding.height;                 \
  op_params.padding_values.width = data->padding.width;                   \
  op_params.quantized_activation_min = activation_min;                    \
  op_params.quantized_activation_max = activation_max;                    \
  type::AveragePool(op_params, GetTensorShape(input),                     \
                    GetTensorData<int8_t>(input), GetTensorShape(output), \
                    GetTensorData<int8_t>(output))
  if (kernel_type == kReference) {
    TF_LITE_AVERAGE_POOL(reference_integer_ops);
  } else {
    TF_LITE_AVERAGE_POOL(optimized_integer_ops);
  }
#undef TF_LITE_AVERAGE_POOL
}

template <KernelType kernel_type>
void MaxEvalFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLitePoolParams* params, OpData* data,
                  const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_MAX_POOL(type)                                                 \
  tflite::PoolParams op_params;                                                \
  op_params.stride_height = params->stride_height;                             \
  op_params.stride_width = params->stride_width;                               \
  op_params.filter_height = params->filter_height;                             \
  op_params.filter_width = params->filter_width;                               \
  op_params.padding_values.height = data->padding.height;                      \
  op_params.padding_values.width = data->padding.width;                        \
  op_params.float_activation_min = activation_min;                             \
  op_params.float_activation_max = activation_max;                             \
  type::MaxPool(op_params, GetTensorShape(input), GetTensorData<float>(input), \
                GetTensorShape(output), GetTensorData<float>(output))
  if (kernel_type == kReference) {
    TF_LITE_MAX_POOL(reference_ops);
  } else {
    TF_LITE_MAX_POOL(optimized_ops);
  }
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void MaxEvalQuantizedUInt8(TfLiteContext* context, TfLiteNode* node,
                           TfLitePoolParams* params, OpData* data,
                           const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  CalculateActivationRangeUint8(params->activation, output, &activation_min,
                                &activation_max);
#define TF_LITE_MAX_POOL(type)                                         \
  tflite::PoolParams op_params;                                        \
  op_params.stride_height = params->stride_height;                     \
  op_params.stride_width = params->stride_width;                       \
  op_params.filter_height = params->filter_height;                     \
  op_params.filter_width = params->filter_width;                       \
  op_params.padding_values.height = data->padding.height;              \
  op_params.padding_values.width = data->padding.width;                \
  op_params.quantized_activation_min = activation_min;                 \
  op_params.quantized_activation_max = activation_max;                 \
  type::MaxPool(op_params, GetTensorShape(input),                      \
                GetTensorData<uint8_t>(input), GetTensorShape(output), \
                GetTensorData<uint8_t>(output))
  if (kernel_type == kReference) {
    TF_LITE_MAX_POOL(reference_ops);
  } else {
    TF_LITE_MAX_POOL(optimized_ops);
  }
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void MaxEvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                          TfLitePoolParams* params, OpData* data,
                          const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min;
  int32_t activation_max;
  CalculateActivationRangeInt8(params->activation, output, &activation_min,
                               &activation_max);
#define TF_LITE_MAX_POOL(type)                                        \
  tflite::PoolParams op_params;                                       \
  op_params.stride_height = params->stride_height;                    \
  op_params.stride_width = params->stride_width;                      \
  op_params.filter_height = params->filter_height;                    \
  op_params.filter_width = params->filter_width;                      \
  op_params.padding_values.height = data->padding.height;             \
  op_params.padding_values.width = data->padding.width;               \
  op_params.quantized_activation_min = activation_min;                \
  op_params.quantized_activation_max = activation_max;                \
  type::MaxPool(op_params, GetTensorShape(input),                     \
                GetTensorData<int8_t>(input), GetTensorShape(output), \
                GetTensorData<int8_t>(output))
  if (kernel_type == kReference) {
    TF_LITE_MAX_POOL(reference_integer_ops);
  } else {
    TF_LITE_MAX_POOL(optimized_integer_ops);
  }
#undef TF_LITE_MAX_POOL
}

template <KernelType kernel_type>
void L2EvalFloat(TfLiteContext* context, TfLiteNode* node,
                 TfLitePoolParams* params, OpData* data,
                 const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);
#define TF_LITE_L2_POOL(type)                                                 \
  tflite::PoolParams op_params;                                               \
  op_params.stride_height = params->stride_height;                            \
  op_params.stride_width = params->stride_width;                              \
  op_params.filter_height = params->filter_height;                            \
  op_params.filter_width = params->filter_width;                              \
  op_params.padding_values.height = data->padding.height;                     \
  op_params.padding_values.width = data->padding.width;                       \
  op_params.float_activation_min = activation_min;                            \
  op_params.float_activation_max = activation_max;                            \
  type::L2Pool(op_params, GetTensorShape(input), GetTensorData<float>(input), \
               GetTensorShape(output), GetTensorData<float>(output))
  if (kernel_type == kReference) {
    TF_LITE_L2_POOL(reference_ops);
  } else {
    TF_LITE_L2_POOL(optimized_ops);
  }
#undef TF_LITE_L2_POOL
}

#undef TF_LITE_KERNEL_TYPE_DISPATCH

template <KernelType kernel_type>
TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      AverageEvalFloat<kernel_type>(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
      AverageEvalQuantizedUint8<kernel_type>(context, node, params, data, input,
                                             output);
      break;
    case kTfLiteInt8:
      AverageEvalQuantizedInt8<kernel_type>(context, node, params, data, input,
                                            output);
      break;
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      MaxEvalFloat<kernel_type>(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
      MaxEvalQuantizedUInt8<kernel_type>(context, node, params, data, input,
                                         output);
      break;
    case kTfLiteInt8:
      MaxEvalQuantizedInt8<kernel_type>(context, node, params, data, input,
                                        output);
      break;
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus L2Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      L2EvalFloat<kernel_type>(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
    // We don't have a quantized implementation, so just fall through to the
    // 'default' case.
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace pooling

TfLiteRegistration* Register_AVERAGE_POOL_REF() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kAverage>,
                                 pooling::AverageEval<pooling::kReference>};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_REF() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kMax>,
                                 pooling::MaxEval<pooling::kReference>};
  return &r;
}

TfLiteRegistration* Register_L2_POOL_REF() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kL2>,
                                 pooling::L2Eval<pooling::kReference>};
  return &r;
}

TfLiteRegistration* Register_AVERAGE_POOL_GENERIC_OPT() {
  static TfLiteRegistration r = {
      pooling::Init, pooling::Free, pooling::GenericPrepare<pooling::kAverage>,
      pooling::AverageEval<pooling::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_GENERIC_OPT() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kMax>,
                                 pooling::MaxEval<pooling::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_L2_POOL_GENERIC_OPT() {
  static TfLiteRegistration r = {pooling::Init, pooling::Free,
                                 pooling::GenericPrepare<pooling::kL2>,
                                 pooling::L2Eval<pooling::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_AVERAGE_POOL_2D() {
  return Register_AVERAGE_POOL_GENERIC_OPT();
}

TfLiteRegistration* Register_MAX_POOL_2D() {
  return Register_MAX_POOL_GENERIC_OPT();
}

TfLiteRegistration* Register_L2_POOL_2D() {
  return Register_L2_POOL_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
