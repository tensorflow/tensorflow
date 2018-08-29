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

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pack {
namespace {

constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLitePackParams* data =
      reinterpret_cast<TfLitePackParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), data->values_count);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input0 = GetInput(context, node, 0);
  TF_LITE_ENSURE(context, NumDimensions(input0) < 4);
  TF_LITE_ENSURE(context, NumDimensions(input0) >= data->axis);
  // TODO(renjieliu): Support negative axis.
  TF_LITE_ENSURE(context, data->axis >= 0);
  if (input0->type != kTfLiteInt32 && input0->type != kTfLiteFloat32 &&
      input0->type != kTfLiteUInt8 && input0->type != kTfLiteInt16) {
    context->ReportError(context,
                         "Currently pack only supports "
                         "float32/uint8/int16/int32.");
    return kTfLiteError;
  }
  // Make sure all inputs have the same shape and type.
  for (int i = 1; i < data->values_count; ++i) {
    const TfLiteTensor* input = GetInput(context, node, i);
    TF_LITE_ENSURE(context, HaveSameShapes(input0, input));
    TF_LITE_ENSURE_EQ(context, input0->type, input->type);
  }

  // Resize output. rank R will become rank R + 1
  const int dimension_size = NumDimensions(input0) + 1;
  const TfLiteIntArray* input_shape = input0->dims;
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(dimension_size);
  int i = 0;
  for (int index = 0; index < dimension_size; ++index) {
    if (index == data->axis) {
      output_shape->data[index] = data->values_count;
    } else {
      output_shape->data[index] = input_shape->data[i++];
    }
  }

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_EQ(context, output->type, input0->type);

  // Guarantee input/output quantization params match as we do not support
  // packing quantized tensors.
  for (int i = 0; i < data->values_count; i++) {
    const TfLiteTensor* input = GetInput(context, node, i);
    TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                      output->params.zero_point);
    TF_LITE_ENSURE_EQ(context, input->params.scale, output->params.scale);
  }

  return context->ResizeTensor(context, output, output_shape);
}

template <typename T>
void PackImpl(TfLiteContext* context, TfLiteNode* node, TfLiteTensor* output,
              int values_count, int axis) {
  VectorOfTensors<T> all_inputs(*context, *node->inputs);
  reference_ops::Pack<T>(RemapDim(NumDimensions(output), axis),
                         all_inputs.data(), all_inputs.dims(), values_count,
                         GetTensorData<T>(output), GetTensorDims(output));
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLitePackParams* data =
      reinterpret_cast<TfLitePackParams*>(node->builtin_data);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  switch (output->type) {
    case kTfLiteFloat32: {
      PackImpl<float>(context, node, output, data->values_count, data->axis);
      break;
    }
    case kTfLiteUInt8: {
      PackImpl<uint8_t>(context, node, output, data->values_count, data->axis);
      break;
    }
    case kTfLiteInt32: {
      PackImpl<int32_t>(context, node, output, data->values_count, data->axis);
      break;
    }
    default: {
      context->ReportError(context,
                           "Currently pack only supports "
                           "float32/uint8/int32.");
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

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
