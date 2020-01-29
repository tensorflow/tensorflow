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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unpack {
namespace {

constexpr int kInputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteUnpackParams* data =
      reinterpret_cast<TfLiteUnpackParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), data->num);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, NumDimensions(input) <= 4);
  TF_LITE_ENSURE(context, NumElements(input) > 0);
  int axis = data->axis;
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < NumDimensions(input));
  if (input->type != kTfLiteInt32 && input->type != kTfLiteFloat32 &&
      input->type != kTfLiteUInt8 && input->type != kTfLiteInt8 &&
      input->type != kTfLiteBool) {
    context->ReportError(context, "Type '%s' is not supported by unpack.",
                         TfLiteTypeGetName(input->type));
    return kTfLiteError;
  }

  const TfLiteIntArray* input_shape = input->dims;
  // Num should be equal to the shape[axis].
  // Resize outputs. rank will be R - 1.
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(NumDimensions(input) - 1);
  int o = 0;
  for (int index = 0; index < NumDimensions(input); ++index) {
    if (index != axis) {
      output_shape->data[o++] = input_shape->data[index];
    }
  }

  TF_LITE_ENSURE_EQ(context, data->num, input_shape->data[axis]);
  for (int i = 0; i < data->num; ++i) {
    TfLiteIntArray* copied_output_shape = TfLiteIntArrayCopy(output_shape);
    TfLiteTensor* output = GetOutput(context, node, i);
    TF_LITE_ENSURE_EQ(context, output->type, input->type);
    // Guarantee input/output quantization params match as we do not support
    // rescaling of unpacked quantized tensors.
    TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                      output->params.zero_point);
    TF_LITE_ENSURE_EQ(context, input->params.scale, output->params.scale);
    TF_LITE_ENSURE_OK(
        context, context->ResizeTensor(context, output, copied_output_shape));
  }

  TfLiteIntArrayFree(output_shape);
  return kTfLiteOk;
}

template <typename T>
void UnpackImpl(TfLiteContext* context, TfLiteNode* node,
                const TfLiteTensor* input, int output_count, int axis) {
  tflite::UnpackParams op_params;
  op_params.axis = axis;
  op_params.num_split = output_count;
  VectorOfTensors<T> all_outputs(*context, *node->outputs);
  reference_ops::Unpack<T>(op_params, GetTensorShape(input),
                           GetTensorData<T>(input), **all_outputs.shapes(),
                           all_outputs.data());
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteUnpackParams* data =
      reinterpret_cast<TfLiteUnpackParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  switch (input->type) {
    case kTfLiteFloat32: {
      UnpackImpl<float>(context, node, input, data->num, data->axis);
      break;
    }
    case kTfLiteInt32: {
      UnpackImpl<int32_t>(context, node, input, data->num, data->axis);
      break;
    }
    case kTfLiteUInt8: {
      UnpackImpl<uint8_t>(context, node, input, data->num, data->axis);
      break;
    }
    case kTfLiteInt8: {
      UnpackImpl<int8_t>(context, node, input, data->num, data->axis);
      break;
    }
    case kTfLiteBool: {
      UnpackImpl<bool>(context, node, input, data->num, data->axis);
      break;
    }
    default: {
      context->ReportError(context, "Type '%s' is not supported by unpack.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
    }
  }

  return kTfLiteOk;
}
}  // namespace
}  // namespace unpack

TfLiteRegistration* Register_UNPACK() {
  static TfLiteRegistration r = {nullptr, nullptr, unpack::Prepare,
                                 unpack::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
