/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "flatbuffers/flexbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

// TODO(b/161933288): Promote this op to builtin-op when we can add new builtin
// ops.

namespace tflite {
namespace ops {
namespace custom {
namespace cumsum {

typedef struct {
  bool exclusive;
  bool reverse;
} TfLiteCumsumParams;

static const int kInputTensor = 0;
static const int kAxisTensor = 1;
static const int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new TfLiteCumsumParams;
  const uint8_t* buffer_data = reinterpret_cast<const uint8_t*>(buffer);

  const flexbuffers::Map& m = flexbuffers::GetRoot(buffer_data, length).AsMap();
  data->exclusive = m["exclusive"].AsBool();
  data->reverse = m["reverse"].AsBool();
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<TfLiteCumsumParams*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis = GetInput(context, node, kAxisTensor);

  TF_LITE_ENSURE(context,
                 input->type == kTfLiteInt32 || input->type == kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, axis->type, kTfLiteInt32);

  TF_LITE_ENSURE_EQ(context, NumElements(axis), 1);

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TfLiteIntArray* output_shape = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_shape);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* axis_tensor = GetInput(context, node, kAxisTensor);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  auto* params = reinterpret_cast<TfLiteCumsumParams*>(node->user_data);

  int axis = *GetTensorData<int>(axis_tensor);
  if (axis < 0) axis += NumDimensions(input);

  if (axis < 0 || axis >= NumDimensions(input)) {
    TF_LITE_KERNEL_LOG(context, "Invalid axis: ", axis);
    return kTfLiteError;
  }

  switch (input->type) {
    case kTfLiteInt32: {
      optimized_ops::CumSum(GetTensorData<int>(input), GetTensorShape(input),
                            axis, params->exclusive, params->reverse,
                            GetTensorData<int>(output));
      break;
    }
    case kTfLiteFloat32: {
      optimized_ops::CumSum(GetTensorData<float>(input), GetTensorShape(input),
                            axis, params->exclusive, params->reverse,
                            GetTensorData<float>(output));
      break;
    }
    default: {
      TF_LITE_KERNEL_LOG(
          context,
          "Unsupported input type, cumsum only supports int32 & float32.");
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace cumsum

TfLiteRegistration* Register_CUMSUM() {
  static TfLiteRegistration r = {cumsum::Init, cumsum::Free, cumsum::Prepare,
                                 cumsum::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
