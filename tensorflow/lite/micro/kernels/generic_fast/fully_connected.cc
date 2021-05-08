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
#include "tensorflow/lite/micro/kernels/generic_fast/fully_connected/fully_connected_core.h"
#include "tensorflow/lite/micro/kernels/generic_fast/fully_connected/fully_connected_impl.h"
#include "tensorflow/lite/micro/kernels/generic_fast/fully_connected/fully_connected_op_data.h"
#include "tensorflow/lite/micro/kernels/generic_fast/fully_connected/fully_connected_ops.h"


namespace tflite {
namespace {

/*
 * Prepare function. This function is only run once before the invocations
 * start. Do as many operations here as possible.
 */
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  int rows = 0;
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  const int32_t weights_offset = -weights->params.zero_point;
  RuntimeShape weights_shape = GetTensorShape(weights);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  rows = weights_shape.Dims(0);
  const int cols = weights_shape.Dims(1);
  const int32_t input_offset = -input->params.zero_point;
  const int32_t* bias_data = GetTensorData<int32_t>(bias);

  if (weights->type == kTfLiteInt8 || weights->type == kTfLiteUInt8) {
    // Calculate data for quantized operation

    TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input->type, input,
                                          weights, bias, output, data));
    // Pre-compute factors for quantized operation

    void* raw =
        context->AllocatePersistentBuffer(context, sizeof(int32_t) * rows);
    data->sum_of_weights_factor = reinterpret_cast<int32_t*>(raw);
  }

  switch (weights->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      data->eval_function = &EvalFloat;
      break;
    case kTfLiteInt8:
      switch (output->type) {
        case kTfLiteInt8:
          PrecomputeSumOfWeightsFactor<int8_t>(
              bias_data, GetTensorData<int8_t>(weights),
              data->sum_of_weights_factor, cols, rows, weights_offset,
              input_offset);
          data->eval_function = &EvalQuantizedInt8;
          break;
        default:
          TF_LITE_KERNEL_LOG(context, "Quantized int8 _t expects output int8");
          return kTfLiteError;
      }
      break;

    case kTfLiteUInt8:
      switch (output->type) {
        case kTfLiteUInt8:
          PrecomputeSumOfWeightsFactor<uint8_t>(
              bias_data, GetTensorData<uint8_t>(weights),
              data->sum_of_weights_factor, cols, rows, weights_offset,
              input_offset);
          data->eval_function = &EvalQuantizedUInt8;
          break;
        case kTfLiteInt16:
          data->eval_function = &EvalQuantizedUint8WithOutputInt16;
          break;
        default:
          TF_LITE_KERNEL_LOG(
              context, "Quantized uint8_t expects output uint8_t or int16");
          return kTfLiteError;
      }
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Weight type %d not currently supported.",
                         weights->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_FULLY_CONNECTED() {
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
