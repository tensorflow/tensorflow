/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include <cstring>
#include <memory>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace bitcast {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

TfLiteStatus CalculateShape(TfLiteContext* context, const TfLiteTensor* input,
                            const TfLiteTensor* output,
                            TfLiteIntArray** output_shape) {
  const int dims = NumDimensions(input);

  auto input_type = input->type;
  auto output_type = output->type;
  size_t input_type_size, output_type_size;
  TF_LITE_ENSURE_STATUS(GetSizeOfType(context, input_type, &input_type_size));
  TF_LITE_ENSURE_STATUS(GetSizeOfType(context, output_type, &output_type_size));

  TfLiteIntArray* shape = nullptr;
  if (input_type_size > output_type_size) {
    // If the input datatype T is larger than the output datatype type then the
    // shape changes from [...] to [..., sizeof(T)/sizeof(type)].
    shape = TfLiteIntArrayCreate(dims + 1);
    for (int i = 0; i < dims; ++i) {
      shape->data[i] = input->dims->data[i];
    }
    shape->data[dims] = input_type_size / output_type_size;
  } else if (input_type_size < output_type_size) {
    // If T is smaller than type, the operator requires that the rightmost
    // dimension be equal to sizeof(type)/sizeof(T). The shape then goes from
    // [..., sizeof(type)/sizeof(T)] to [...].
    TF_LITE_ENSURE_EQ(context, input->dims->data[dims - 1],
                      output_type_size / input_type_size);
    shape = TfLiteIntArrayCreate(dims - 1);
    for (int i = 0; i < dims - 1; ++i) {
      shape->data[i] = input->dims->data[i];
    }
  } else {
    // Same element type size.
    shape = TfLiteIntArrayCopy(input->dims);
  }
  *output_shape = shape;
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TfLiteIntArray* output_size = nullptr;
  TF_LITE_ENSURE_OK(context,
                    CalculateShape(context, input, output, &output_size));
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Only copy data if input and output do not share a buffer.
  if (output->data.data != input->data.data) {
    memcpy(output->data.data, input->data.data, input->bytes);
  }
  return kTfLiteOk;
}

}  // namespace bitcast

TfLiteRegistration* Register_BITCAST() {
  static TfLiteRegistration r = {
      nullptr,
      nullptr,
      bitcast::Prepare,
      bitcast::Eval,
      /*profiling_string=*/nullptr,
      /*builtin_code=*/0,
      /*custom_name=*/nullptr,
      /*version=*/0,
      /*registration_external=*/nullptr,
      /*async_kernel=*/nullptr,
      kTfLiteInplaceOpInput0Shared | kTfLiteInplaceOpDataUnmodified};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
