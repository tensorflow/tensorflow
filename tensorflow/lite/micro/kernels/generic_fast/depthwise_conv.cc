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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

// EVAL functions are located in these files
#include "tensorflow/lite/micro/kernels/generic_fast/depthwise_conv/depthwise_conv_impl.h"
#include "tensorflow/lite/micro/kernels/generic_fast/depthwise_conv/depthwise_conv_op_data.h"
#include "tensorflow/lite/micro/kernels/generic_fast/depthwise_conv/depthwise_conv_ops.h"

namespace tflite {
namespace {

/*
 * Prepare function. This function is only run once before the invocations
 * start. Do as many operations here as possible.
 */
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* params =
      reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kFilterTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  const TfLiteType data_type = input->type;
  int width = SizeOfDimension(input, 2);
  int height = SizeOfDimension(input, 1);
  int filter_width = SizeOfDimension(filter, 2);
  int filter_height = SizeOfDimension(filter, 1);

  // Per channel quantization is only needed for int8_t inference. For other
  // quantized types, only a single scale and zero point is needed.
  const int num_channels = filter->dims->data[kDepthwiseConvQuantizedDimension];
  // Dynamically allocate per-channel quantization parameters.
  data->per_channel_output_multiplier =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));

  TF_LITE_ENSURE(context, data->per_channel_output_multiplier != nullptr);
  data->per_channel_output_shift =
      static_cast<int32_t*>(context->AllocatePersistentBuffer(
          context, num_channels * sizeof(int32_t)));
  TF_LITE_ENSURE(context, data->per_channel_output_multiplier != nullptr);

  // All per-channel quantized tensors need valid zero point and scale arrays.
  if (input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);

    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, affine_quantization->zero_point);
    TF_LITE_ENSURE(
        context, affine_quantization->scale->size == 1 ||
                     affine_quantization->scale->size ==
                         filter->dims->data[kDepthwiseConvQuantizedDimension]);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                      affine_quantization->zero_point->size);
  }

  data->input_zero_point = input->params.zero_point;
  data->filter_zero_point = filter->params.zero_point;
  data->output_zero_point = output->params.zero_point;

  const int32_t input_offset = -input->params.zero_point;
  const int dilation_width_factor = params->dilation_width_factor;
  const int dilation_height_factor = params->dilation_height_factor;

  auto filter_shape = GetTensorShape(filter);

  const int output_depth = SizeOfDimension(filter, 3);
  // Selection structure mirrors that in Eval.   Could select a final
  // kernel variant here...

  if (filter->type == kTfLiteInt8 || filter->type == kTfLiteUInt8) {
    const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
    const int32_t* bias_data = GetTensorData<int32_t>(bias);

    const int32_t filter_offset = -filter->params.zero_point;
    TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);

    void* raw = context->AllocatePersistentBuffer(
        context, sizeof(int32_t) * num_channels);
    data->sum_of_filters_factor = static_cast<int32_t*>(raw);

    // Precompute the sum of filters
    if (filter->type == kTfLiteUInt8) {
      PrecomputeSumOfFiltersFactor<uint8_t>(
          bias_data, filter, data->sum_of_filters_factor, filter_shape,
          input_offset, filter_offset);
    } else {
      PrecomputeSumOfFiltersFactor<int8_t>(bias_data, filter,
                                           data->sum_of_filters_factor,
                                           filter_shape, input_offset, 0);
    }
  }

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, node, params, width, height,
                                        filter_width, filter_height, data_type,
                                        data));

  // Determine which version to use
  bool need_acc_buf = false;
  // Check if optimized filter width is used
  const bool use_optimized_filter_width = (GetTensorShape(filter).Dims(0) != 1);
  const bool use_reference =
      ((dilation_width_factor != 1) || (dilation_height_factor != 1) ||
       use_optimized_filter_width);
  if (!use_reference && !(input->type == kTfLiteFloat32)) {
    need_acc_buf = true;
  }
  if (need_acc_buf) {
    void* raw = context->AllocatePersistentBuffer(
        context, sizeof(int32_t) * output_depth);
    data->acc_buf = static_cast<int32_t*>(raw);
  }

  const bool use_padding =
      (data->padding.height != 0 || data->padding.width != 0 ||
       data->padding.height_offset != 0 || data->padding.width_offset != 0);

  // Set the function pointer that is used during inference here
  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      data->eval_function = &EvalFloat;
      break;
    case kTfLiteInt8: {
      if (use_reference) {
        data->eval_function = &EvalInt8Reference;
      } else if (use_padding) {
        // Use the version that can handle padding
        data->eval_function = &EvalInt8Padding;
      } else {
        data->eval_function = &EvalInt8;
      }
      break;
    }
    case kTfLiteUInt8: {
      if (use_reference) {
        data->eval_function = &EvalUInt8Reference;
      } else if (use_padding) {
        data->eval_function = &EvalUInt8Padding;
      } else {
        data->eval_function = &EvalUInt8;
      }
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_DEPTHWISE_CONV_2D() {
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
