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
/*
* Copyright (c) 2020 Cadence Design Systems Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to deal in the Software without restriction, including
* without limitation the rights to use, copy, modify, merge, publish,
* distribute, sublicense, and/or sell copies of the Software, and to
* permit persons to whom the Software is furnished to do so, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "vision_api.h"
#include "utils.h"
#include <string>

namespace tflite {
namespace ops {
namespace micro {
namespace pooling {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  TfLitePaddingValues padding;
  int32_t activation_min;
  int32_t activation_max;
  float activation_min_f32;
  float activation_max_f32;
  // xtensa library context specific
  uint8_t* pContext; // persistent lib context for this instance saved here
  uint32_t contextSize;
};

TfLiteStatus CalculateOpData(const TfLiteContext* context,
                             const TfLitePoolParams* params,
                             const TfLiteTensor* input,
                             const TfLiteTensor* output, OpData* data) {
  // input: batch, height, width, channel
  int height = SizeOfDimension(input, 1);
  int width = SizeOfDimension(input, 2);

  int out_height, out_width;

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      /*dilation_rate_height=*/1,
      /*dilation_rate_width=*/1, height, width, params->filter_height,
      params->filter_width, params->padding, &out_height, &out_width);

  return kTfLiteOk;
}

void AverageEvalFloat(const TfLiteContext* context, const TfLiteNode* node,
                      const TfLitePoolParams* params, const OpData* data,
                      const TfLiteEvalTensor* input, TfLiteEvalTensor* output) {
  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = data->activation_min_f32;
  op_params.float_activation_max = data->activation_max_f32;
  reference_ops::AveragePool(op_params, tflite::micro::GetTensorShape(input),
                             tflite::micro::GetTensorData<float>(input),
                             tflite::micro::GetTensorShape(output),
                             tflite::micro::GetTensorData<float>(output));
}

void AverageEvalQuantized(TfLiteContext* context, const TfLiteNode* node,
                          const TfLitePoolParams* params, const OpData* data,
                          const TfLiteEvalTensor* input,
                          TfLiteEvalTensor* output) {
  TFLITE_DCHECK(input->type == kTfLiteUInt8 || input->type == kTfLiteInt8);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  if (input->type == kTfLiteUInt8) {
    reference_ops::AveragePool(op_params, tflite::micro::GetTensorShape(input),
                               tflite::micro::GetTensorData<uint8_t>(input),
                               tflite::micro::GetTensorShape(output),
                               tflite::micro::GetTensorData<uint8_t>(output));
  }
  else {
    if (op_params.stride_width != op_params.stride_height) {
      reference_integer_ops::AveragePool(
          op_params, tflite::micro::GetTensorShape(input),
          tflite::micro::GetTensorData<int8_t>(input),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<int8_t>(output));
    }
    else {
      int8_t* input_data = (int8_t*)tflite::micro::GetTensorData<int8_t>(input);
      int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
      const RuntimeShape input_shape = tflite::micro::GetTensorShape(input);
      const RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
      TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
      TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
      TFLITE_DCHECK_EQ(op_params.stride_height, op_params.stride_width);

      uint32_t inputN = input_shape.Dims(0);
      uint32_t inputH = input_shape.Dims(1);
      uint32_t inputW = input_shape.Dims(2);
      uint32_t inputD = input_shape.Dims(3);

      uint32_t outputN = output_shape.Dims(0);
      uint32_t outputH = output_shape.Dims(1);
      uint32_t outputW = output_shape.Dims(2);
      uint32_t outputD = output_shape.Dims(3);

      uint32_t inputSize = inputN * inputH * inputW * inputD;
      uint32_t outputSize = outputN * outputH * outputW * outputD;

      uint32_t strideWidth = op_params.stride_width;
      uint32_t filterWidth = op_params.filter_height;
      uint32_t filterHeight = op_params.filter_width;
      uint32_t act_min = op_params.quantized_activation_min;
      uint32_t act_max = op_params.quantized_activation_max;
      xiAverageEvalQuantized(
          data->pContext, data->contextSize, input_data, inputSize, output_data,
          outputSize, inputN, inputH, inputW, inputD, outputN, outputH, outputW,
          outputD, filterWidth, filterHeight, strideWidth, act_min, act_max);
    }
#if (ENABLE_OUTPUT_DUMP_FILES)
    {
      uint32_t output_size = output->dims->data[0] * output->dims->data[1] *
                            output->dims->data[2] * output->dims->data[3];
	  static int level=0;
#if (FLK_USE_GOOGLE_REF)
	  std::string kernelType = "gref";
#else
#if (REF_FLK_POOL)
      std::string kernelType = "cref";
#else
      std::string kernelType = "opt";
#endif
#endif
      std::string filename = "pooling" + std::to_string(level) + ".output.txt." + kernelType;
      dumpOutputToFile(filename, output->dims->data[0], output->dims->data[1],
    		  output->dims->data[2], output->dims->data[3],
    		  (uint8_t *)tflite::micro::GetTensorData<int8_t>(output));
      level ++;
    }
#endif
  }
}

void MaxEvalFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLitePoolParams* params, const OpData* data,
                  const TfLiteEvalTensor* input, TfLiteEvalTensor* output) {
  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = data->activation_min_f32;
  op_params.float_activation_max = data->activation_max_f32;
  reference_ops::MaxPool(op_params, tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorData<float>(input),
                         tflite::micro::GetTensorShape(output),
                         tflite::micro::GetTensorData<float>(output));
}

void MaxEvalQuantized(TfLiteContext* context, TfLiteNode* node,
                      TfLitePoolParams* params, const OpData* data,
                      const TfLiteEvalTensor* input, TfLiteEvalTensor* output) {
  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = data->activation_min;
  op_params.quantized_activation_max = data->activation_max;

  if (input->type == kTfLiteUInt8) {
    reference_ops::MaxPool(op_params, tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<uint8_t>(input),
                           tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<uint8_t>(output));
  } else {
    reference_integer_ops::MaxPool(
        op_params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
  }
}
}  // namespace

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  // Inputs and outputs share the same type, guaranteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AverageEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      AverageEvalQuantized(context, node, params, data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Input type %s is not currently supported",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData* data = static_cast<const OpData*>(node->user_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  switch (input->type) {
    case kTfLiteFloat32:
      MaxEvalFloat(context, node, params, data, input, output);
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
      MaxEvalQuantized(context, node, params, data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
    void* data = context->AllocatePersistentBuffer(context, sizeof(OpData));
    if (data == nullptr) {
        return nullptr;
    }

    uint32_t contextSize=0;
    uint32_t status = xiPoolGetMemReqd_Context(&contextSize);
    if (!status && contextSize) {
      void* data2 = context->AllocatePersistentBuffer(context, contextSize);
      if (data2 == nullptr) {
        return nullptr;
      }
      OpData* opData = (OpData*)data;
      opData->pContext = (uint8_t *)data2;
      opData->contextSize = contextSize;
    }

    if (InitXtensaContext())
  	  return nullptr;
    return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, data));

  if (input->type == kTfLiteFloat32) {
    CalculateActivationRange(params->activation, &data->activation_min_f32,
                             &data->activation_max_f32);
  } else if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    CalculateActivationRangeQuantized(context, params->activation, output,
                                      &data->activation_min,
                                      &data->activation_max);
  }

  return kTfLiteOk;
}

}  // namespace pooling

TfLiteRegistration Register_AVERAGE_POOL_2D() {
  return {/*init=*/pooling::Init,
          /*free=*/nullptr,
          /*prepare=*/pooling::Prepare,
          /*invoke=*/pooling::AverageEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

TfLiteRegistration Register_MAX_POOL_2D() {
  return {/*init=*/pooling::Init,
          /*free=*/nullptr,
          /*prepare=*/pooling::Prepare,
          /*invoke=*/pooling::MaxEval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
