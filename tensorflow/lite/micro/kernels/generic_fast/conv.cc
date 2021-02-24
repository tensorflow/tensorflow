/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

/*
GENERIC FAST
This optimized kernel directory contains optimized kernels.
The kernels are portable to every hardware, no custom instructions are used.
The kernels take advantage of precomputations, smaller tweaks and the prepare
phase to reduce runtime and memory overhead.
==============================================================================*/

#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

// EVAL functions are located in these files
#include "tensorflow/lite/micro/kernels/generic_fast/conv/conv_impl.h"
#include "tensorflow/lite/micro/kernels/generic_fast/conv/conv_op_data.h"
#include "tensorflow/lite/micro/kernels/generic_fast/conv/conv_ops_int8.h"
#include "tensorflow/lite/micro/kernels/generic_fast/conv/conv_ops_ref.h"
#include "tensorflow/lite/micro/kernels/generic_fast/conv/conv_ops_uint8.h"


namespace tflite {
namespace {

/*
 * Prepare function. This function is only run once before the invocations
 * start. Do as many operations here as possible.
 */

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  int input_width = input->dims->data[2];
  int input_height = input->dims->data[1];
  int filter_width = filter->dims->data[2];
  int filter_height = filter->dims->data[1];
  int output_width = output->dims->data[2];
  int output_height = output->dims->data[1];

  TF_LITE_ENSURE_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(context,
                   affine_quantization->scale->size == 1 ||
                       affine_quantization->scale->size ==
                           filter->dims->data[kConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
  const int32_t* bias_data = GetTensorData<int32_t>(bias);

  const int32_t filter_offset = -data->filter_zero_point;
  RuntimeShape filter_shape = GetTensorShape(filter);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);

  int output_depth = filter_shape.Dims(0);

  if (filter->type == kTfLiteInt8 || filter->type == kTfLiteUInt8) {
    void* raw = context->AllocatePersistentBuffer(
        context, sizeof(int32_t) * output_depth);
    data->sum_of_filters_factor = reinterpret_cast<int32_t*>(raw);

    raw = context->AllocatePersistentBuffer(context,
                                            sizeof(int32_t) * output_depth);
    data->per_channel_output_multiplier = reinterpret_cast<int32_t*>(raw);

    raw = context->AllocatePersistentBuffer(context,
                                            sizeof(int32_t) * output_depth);
    data->per_channel_output_shift = reinterpret_cast<int32_t*>(raw);
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(
      context, node, params, input_width, input_height, filter_width,
      filter_height, output_width, output_height, input->type, data));

  const bool use_padding =
      (data->padding.height != 0 || data->padding.width != 0 ||
       data->padding.height_offset != 0 || data->padding.width_offset != 0);

  if (filter->type == kTfLiteInt8 || filter->type == kTfLiteUInt8) {
    // Precompute the sum of filters
    const int32_t input_offset = -data->input_zero_point;
    if (filter->type == kTfLiteUInt8) {
      if (use_padding) {
        PrecomputeSumOfFiltersPadding(bias_data, data->sum_of_filters_factor,
                                      filter_shape);
      } else {
        PrecomputeSumOfFilters<uint8_t>(
            bias_data, filter, data->sum_of_filters_factor, filter_shape,
            input_offset, filter_offset);
      }
    } else {
      if (use_padding) {
        PrecomputeSumOfFiltersPadding(bias_data, data->sum_of_filters_factor,
                                      filter_shape);
      } else {
        PrecomputeSumOfFilters<int8_t>(bias_data, filter,
                                       data->sum_of_filters_factor,
                                       filter_shape, input_offset, 0);
      }
    }
  }

  // Determine which version to use
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;
  bool use_reference =
      ((dilation_width_factor != 1) || (dilation_height_factor != 1));

  // Set the function pointer that is used during inference here
  switch (filter->type) {
    case kTfLiteFloat32: {
      data->eval_function = &EvalConvFloat;
      break;
    }
    case kTfLiteInt8: {
      if (use_reference) {
        data->eval_function = &EvalConvInt8Reference;
      } else {
        data->eval_function = &EvalConvInt8;
      }
      break;
    }
    case kTfLiteUInt8: {
      if (use_reference) {
        data->eval_function = &EvalConvUInt8Reference;
      } else {
        data->eval_function = &EvalConvUInt8;
      }
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_CONV_2D() {
  return {/*init=*/Init,
          /*free=*/Free,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
