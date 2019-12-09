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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pack {
namespace {

constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus PackImpl(TfLiteContext* context, TfLiteNode* node,
                      TfLiteTensor* output, int values_count, int axis) {
  const int dimensions = output->dims->size;
  const TfLiteTensor* input0 = &context->tensors[node->inputs->data[0]];
  const TfLiteIntArray* input_dims = input0->dims;
  const TfLiteIntArray* output_dims = output->dims;

  if (axis < 0) {
    axis += dimensions;
  }

  int outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= output_dims->data[i];
  }
  int copy_size = 1;
  for (int i = axis + 1; i < dimensions; ++i) {
    copy_size *= output_dims->data[i];
  }
  int input_size = 1;
  for (int i = 0; i < input_dims->size; ++i) {
    input_size *= input_dims->data[i];
  }
  TFLITE_DCHECK_EQ(input_size, copy_size * outer_size);

  T* output_data = GetTensorData<T>(output);

  for (int i = 0; i < values_count; ++i) {
    TfLiteTensor* t = &context->tensors[node->inputs->data[i]];
    const T* input_data = GetTensorData<T>(t);
    for (int k = 0; k < outer_size; ++k) {
      const T* input_ptr = input_data + copy_size * k;
      int loc = k * values_count * copy_size + i * copy_size;
      T* output_ptr = output_data + loc;
      for (int j = 0; j < copy_size; ++j) output_ptr[j] = input_ptr[j];
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLitePackParams* data =
      reinterpret_cast<TfLitePackParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (output->type) {
    case kTfLiteFloat32: {
      return PackImpl<float>(context, node, output, data->values_count,
                             data->axis);
    }
    case kTfLiteUInt8: {
      return PackImpl<uint8_t>(context, node, output, data->values_count,
                               data->axis);
    }
    case kTfLiteInt8: {
      return PackImpl<int8_t>(context, node, output, data->values_count,
                              data->axis);
    }
    case kTfLiteInt32: {
      return PackImpl<int32_t>(context, node, output, data->values_count,
                               data->axis);
    }
    case kTfLiteInt64: {
      return PackImpl<int64_t>(context, node, output, data->values_count,
                               data->axis);
    }
    default: {
      context->ReportError(context, "Type '%s' is not supported by pack.",
                           TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}

}  // namespace
}  // namespace pack

TfLiteRegistration* Register_PACK() {
  static TfLiteRegistration r = {nullptr, nullptr, pack::Prepare, pack::Eval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
