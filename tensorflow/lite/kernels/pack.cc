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
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pack {
namespace {

constexpr int kOutputTensor = 0;

struct OpData {
  void* all_inputs;
  TfLiteType type;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
  auto* data = reinterpret_cast<OpData*>(buffer);

  switch (data->type) {
    case kTfLiteFloat32:
      delete static_cast<VectorOfTensors<float>*>(data->all_inputs);
      break;
    case kTfLiteInt32:
      delete static_cast<VectorOfTensors<int32_t>*>(data->all_inputs);
      break;
    case kTfLiteUInt8:
      delete static_cast<VectorOfTensors<int8_t>*>(data->all_inputs);
      break;
    case kTfLiteInt8:
      delete static_cast<VectorOfTensors<int8_t>*>(data->all_inputs);
      break;
    case kTfLiteInt64:
      delete static_cast<VectorOfTensors<int64_t>*>(data->all_inputs);
      break;

    default:
      context->ReportError(context, "Unexpected data type - [%s] received.",
                           TfLiteTypeGetName(data->type));
  }

  delete data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TfLitePackParams* data =
      reinterpret_cast<TfLitePackParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), data->values_count);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input0 = GetInput(context, node, 0);
  const int dimension_size = NumDimensions(input0) + 1;
  if (data->axis < 0) {
    data->axis += dimension_size;
  }
  TF_LITE_ENSURE(context, NumDimensions(input0) >= data->axis);
  TF_LITE_ENSURE(context, data->axis >= 0);

  if (input0->type != kTfLiteInt32 && input0->type != kTfLiteFloat32 &&
      input0->type != kTfLiteUInt8 && input0->type != kTfLiteInt8 &&
      input0->type != kTfLiteInt64) {
    context->ReportError(context, "Type '%s' is not supported by pack.",
                         TfLiteTypeGetName(input0->type));
    return kTfLiteError;
  }
  // Make sure all inputs have the same shape and type.
  for (int i = 1; i < data->values_count; ++i) {
    const TfLiteTensor* input = GetInput(context, node, i);
    TF_LITE_ENSURE(context, HaveSameShapes(input0, input));
    TF_LITE_ENSURE_EQ(context, input0->type, input->type);
  }

  // Resize output. rank R will become rank R + 1
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

  OpData* user_data = reinterpret_cast<OpData*>(node->user_data);
  user_data->type = output->type;

  switch (user_data->type) {
    case kTfLiteFloat32:
      user_data->all_inputs = reinterpret_cast<void*>(
          new VectorOfTensors<float>(*context, *node->inputs));
      break;
    case kTfLiteInt32:
      user_data->all_inputs = reinterpret_cast<void*>(
          new VectorOfTensors<int32_t>(*context, *node->inputs));
      break;
    case kTfLiteUInt8:
      user_data->all_inputs = reinterpret_cast<void*>(
          new VectorOfTensors<uint8_t>(*context, *node->inputs));
      break;
    case kTfLiteInt8:
      user_data->all_inputs = reinterpret_cast<void*>(
          new VectorOfTensors<int8_t>(*context, *node->inputs));
      break;
    case kTfLiteInt64:
      user_data->all_inputs = reinterpret_cast<void*>(
          new VectorOfTensors<int64_t>(*context, *node->inputs));
      break;

    default:
      context->ReportError(context, "Unexpected data type - [%s] received.",
                           TfLiteTypeGetName(user_data->type));
  }

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
TfLiteStatus PackImpl(TfLiteContext* context, TfLiteNode* node,
                      TfLiteTensor* output, int values_count, int axis) {
  TF_LITE_ENSURE(context, axis >= 0);

  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  VectorOfTensors<T>* all_inputs =
      static_cast<VectorOfTensors<T>*>(data->all_inputs);
  all_inputs->update(*context, *node->inputs);
  tflite::PackParams op_params;
  op_params.axis = axis;
  op_params.inputs_count = values_count;

  reference_ops::Pack<T>(op_params, all_inputs->shapes(), all_inputs->data(),
                         GetTensorShape(output), GetTensorData<T>(output));
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
  static TfLiteRegistration r = {pack::Init, pack::Free, pack::Prepare,
                                 pack::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
