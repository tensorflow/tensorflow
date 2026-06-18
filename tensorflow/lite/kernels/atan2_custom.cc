// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cmath>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/custom_ops_register.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace atan2 {

TfLiteStatus EnsureSameShape(TfLiteContext* context, const TfLiteTensor* a,
                             const TfLiteTensor* b) {
  TF_LITE_ENSURE_EQ(context, tflite::NumDimensions(a),
                    tflite::NumDimensions(b));

  return TfLiteStatus::kTfLiteOk;
}

TfLiteStatus Atan2Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, tflite::NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, tflite::NumOutputs(node), 1);

  const TfLiteTensor* input_y = tflite::GetInput(context, node, 0);
  const TfLiteTensor* input_x = tflite::GetInput(context, node, 1);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);

  // Validate size and type constraints
  TF_LITE_ENSURE_OK(context, EnsureSameShape(context, input_y, input_x));
  TF_LITE_ENSURE_TYPES_EQ(context, input_y->type, input_x->type);
  TF_LITE_ENSURE_TYPES_EQ(context, input_y->type, output->type);
  TF_LITE_ENSURE(context, input_y->type == kTfLiteFloat32 ||
                              input_y->type == kTfLiteFloat64);

  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input_y->dims);

  return context->ResizeTensor(context, output, output_shape);
}

template <typename Float>
TfLiteStatus Atan2(const TfLiteTensor* input_y, const TfLiteTensor* input_x,
                   TfLiteTensor* output) {
  const Float* data_y = tflite::GetTensorData<Float>(input_y);
  const Float* data_x = tflite::GetTensorData<Float>(input_x);
  Float* data_output = tflite::GetTensorData<Float>(output);

  const int64_t num_elements = NumElements(input_y);
  for (int64_t i = 0; i < num_elements; ++i) {
    data_output[i] = std::atan2(data_y[i], data_x[i]);
  }

  return TfLiteStatus::kTfLiteOk;
}

TfLiteStatus Atan2Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_y = tflite::GetInput(context, node, 0);
  const TfLiteTensor* input_x = tflite::GetInput(context, node, 1);
  TfLiteTensor* output = tflite::GetOutput(context, node, 0);

  switch (output->type) {
    case kTfLiteFloat32:
      TF_LITE_ENSURE_OK(context, Atan2<float>(input_y, input_x, output));
      break;
    case kTfLiteFloat64:
      TF_LITE_ENSURE_OK(context, Atan2<double>(input_y, input_x, output));
      break;
    default: {
      TF_LITE_KERNEL_LOG(context, "Unsupported datatype for atan2 output: %s",
                         TfLiteTypeGetName(output->type));
      return TfLiteStatus::kTfLiteError;
    }
  }

  return TfLiteStatus::kTfLiteOk;
}

}  // namespace atan2

TfLiteRegistration* Register_ATAN2() {
  static TfLiteRegistration r = {nullptr, nullptr, atan2::Atan2Prepare,
                                 atan2::Atan2Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
