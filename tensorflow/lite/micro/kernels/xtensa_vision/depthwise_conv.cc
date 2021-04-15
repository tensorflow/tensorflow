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

#include "depthwise_conv.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_float.h"
#include "tensorflow/lite/kernels/internal/reference/depthwiseconv_uint8.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "vision_api.h"
#include "utils.h"
#include <string>

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = context->AllocatePersistentBuffer(context, sizeof(OpDataConv));
  if (InitXtensaContext())
     return nullptr;
  return data;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto& params =
      *(reinterpret_cast<TfLiteDepthwiseConvParams*>(node->builtin_data));
  const OpDataConv& data = *(static_cast<const OpDataConv*>(node->user_data));

  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kDepthwiseConvOutputTensor);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kDepthwiseConvWeightsTensor);
  const TfLiteEvalTensor* bias =
      (NumInputs(node) == 3)
          ? tflite::micro::GetEvalInput(context, node, kDepthwiseConvBiasTensor)
          : nullptr;

  switch (input->type) {  // Already know in/out types are same.
    case kTfLiteFloat32: {
  tflite::reference_ops::DepthwiseConv(
          DepthwiseConvParamsFloat(params, data),
          tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<float>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<float>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
      break;
}
    case kTfLiteInt8: {
      if (data.enableXtensaKernel) {
        uint32_t input_size =
            input->dims->data[0] * input->dims->data[1] *
            input->dims->data[2] * input->dims->data[3];
        uint32_t output_size =
            output->dims->data[0] * output->dims->data[1] *
            output->dims->data[2] * output->dims->data[3];
        uint32_t num_channels =
            filter->dims->data[kDepthwiseConvQuantizedDimension];
        xiDepthwiseConv(data.pContext, data.contextSize,
            (int8_t *)tflite::micro::GetTensorData<int8_t>(input), input_size,
            tflite::micro::GetTensorData<int8_t>(output), output_size,
            data.reordCoeffnBias, data.reordCoeffnBiasSize,
            data.per_channel_output_multiplier,
            data.per_channel_output_shift_int8, num_channels,
            data.padding.width, data.padding.height);
      }
      else {
        reference_integer_ops::DepthwiseConvPerChannel(
               DepthwiseConvParamsQuantized(params, data),
               data.per_channel_output_multiplier, data.per_channel_output_shift,
               tflite::micro::GetTensorShape(input),
           tflite::micro::GetTensorData<int8_t>(input),
           tflite::micro::GetTensorShape(filter),
           tflite::micro::GetTensorData<int8_t>(filter),
           tflite::micro::GetTensorShape(bias),
           tflite::micro::GetTensorData<int32_t>(bias),
           tflite::micro::GetTensorShape(output),
           tflite::micro::GetTensorData<int8_t>(output));
      }
#if (ENABLE_OUTPUT_DUMP_FILES)
      {
        uint32_t output_size = output->dims->data[0] * output->dims->data[1] *
                              output->dims->data[2] * output->dims->data[3];
     static int level=0;
#if (FLK_USE_GOOGLE_REF)
     std::string kernelType = "gref";
#else
#if (REF_FLK_DEPTHWISE_CONV2D)
        std::string kernelType = "cref";
#else
        std::string kernelType = "opt";
#endif
#endif
        std::string filename = "depthwise_conv2d" + std::to_string(level) + ".output.txt." + kernelType;
        dumpOutputToFile(filename, output->dims->data[0], output->dims->data[1],
    	      output->dims->data[2], output->dims->data[3],
              (uint8_t *)tflite::micro::GetTensorData<int8_t>(output));
        level ++;
      }
#endif
      break;
    }
    case kTfLiteUInt8: {
      reference_ops::DepthwiseConv(
          DepthwiseConvParamsQuantized(params, data),
          tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<uint8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<uint8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<uint8_t>(output));
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
          /*free=*/nullptr,
          /*prepare=*/DepthwiseConvPrepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
