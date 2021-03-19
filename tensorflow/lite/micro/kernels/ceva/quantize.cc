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
#include "tensorflow/lite/kernels/internal/reference/quantize.h"

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/ceva/ceva_tflm_lib.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/micro_utils.h"
#ifdef MCPS_MEASUREMENT
#include "tensorflow/lite/micro/kernels/ceva/mcps_macros.h "
#endif

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataQuantizeReference));
}

TfLiteStatus EvalCEVA(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  auto* data = static_cast<OpDataQuantizeReference*>(node->user_data);

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  if (input->type == kTfLiteFloat32 && output->type == kTfLiteInt8) {
    const float* input_data = tflite::micro::GetTensorData<float>(input);
    int8_t* output_data = tflite::micro::GetTensorData<int8_t>(output);
    const int flat_size =
        MatchingFlatSize(tflite::micro::GetTensorShape(input),
                         tflite::micro::GetTensorShape(output));

#ifdef MCPS_MEASUREMENT
    MCPS_START_ONE;
#endif
    CEVA_TFLM_AffineQuantize_Int8(input_data, output_data, flat_size,
                                  data->quantization_params.scale,
                                  data->quantization_params.zero_point);
#ifdef MCPS_MEASUREMENT
    MCPS_STOP_ONE("Test params:CEVA_TFLM_AffineQuantize_Int8 loop = %d",
                  flat_size);
#endif
  } else
    return EvalQuantizeReference(context, node);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
#if defined(CEVA_BX1) || defined(CEVA_SP500)
  return EvalCEVA(context, node);
#else
  return EvalQuantizeReference(context, node);
#endif
}

}  // namespace

// This Op (QUANTIZE) quantizes the input and produces quantized output.
// AffineQuantize takes scale and zero point and quantizes the float value to
// quantized output, in int8_t or uint8_t format.
TfLiteRegistration Register_QUANTIZE() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/PrepareQuantizeReference,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
