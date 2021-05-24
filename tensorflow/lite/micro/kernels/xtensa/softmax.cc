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

#include "tensorflow/lite/micro/kernels/softmax.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa_softmax.h"

namespace tflite {
namespace {

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt16) {
    return XtensaEvalSoftmaxInt8Int16(context, node);
  }

  TFLITE_DCHECK(node->user_data != nullptr);
#if defined(FUSION_F1) || defined(HIFI5)
  XtensaSoftmaxOpData op_data =
      *static_cast<XtensaSoftmaxOpData*>(node->user_data);
  SoftmaxParams params = op_data.params;
#else
  SoftmaxParams params = *static_cast<SoftmaxParams*>(node->user_data);
#endif

  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
    tflite::reference_ops::Softmax(
        params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int8_t>(output));
    return kTfLiteOk;
  }

  if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    tflite::reference_ops::SoftmaxInt16(
        params, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int16_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
    return kTfLiteOk;
  }

  if (input->type == kTfLiteFloat32) {
    tflite::reference_ops::Softmax(params, tflite::micro::GetTensorShape(input),
                                   tflite::micro::GetTensorData<float>(input),
                                   tflite::micro::GetTensorShape(output),
                                   tflite::micro::GetTensorData<float>(output));
    return kTfLiteOk;
  }

  TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                     TfLiteTypeGetName(input->type), input->type);
  return kTfLiteError;
}

}  // namespace

TfLiteRegistration Register_SOFTMAX() {
  return {/*init=*/XtensaInitSoftmax,
          /*free=*/nullptr,
          /*prepare=*/XtensaPrepareSoftmax,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
