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

#include <stdint.h>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pack {
namespace {

constexpr int kOutputTensor = 0;

struct OpData {
  // Indicates that 'Eval' is a noop as the output as written during 'Prepare'.
  bool noop;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                      TfLiteTensor* output, const TfLitePackParams* params);
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  bool noop = true;
  TfLitePackParams* params =
      reinterpret_cast<TfLitePackParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), params->values_count);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input0;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input0));
  noop &= IsConstantOrPersistentTensor(input0);
  const int dimension_size = NumDimensions(input0) + 1;
  if (params->axis < 0) {
    params->axis += dimension_size;
  }
  TF_LITE_ENSURE(context, NumDimensions(input0) >= params->axis);
  TF_LITE_ENSURE(context, params->axis >= 0);

  if (input0->type != kTfLiteInt32 && input0->type != kTfLiteFloat32 &&
      input0->type != kTfLiteUInt8 && input0->type != kTfLiteInt8 &&
      input0->type != kTfLiteInt16 && input0->type != kTfLiteInt64) {
    TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by pack.",
                       TfLiteTypeGetName(input0->type));
    return kTfLiteError;
  }
  // Make sure all inputs have the same shape and type.
  for (int i = 1; i < params->values_count; ++i) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &input));
    TF_LITE_ENSURE(context, HaveSameShapes(input0, input));
    TF_LITE_ENSURE_TYPES_EQ(context, input0->type, input->type);
    noop &= IsConstantOrPersistentTensor(input);
  }
  data->noop = noop;

  // Resize output. rank R will become rank R + 1
  const TfLiteIntArray* input_shape = input0->dims;
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(dimension_size);
  int i = 0;
  for (int index = 0; index < dimension_size; ++index) {
    if (index == params->axis) {
      output_shape->data[index] = params->values_count;
    } else {
      output_shape->data[index] = input_shape->data[i++];
    }
  }

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input0->type);

  // Guarantee input/output quantization params match as we do not support
  // packing quantized tensors.
  for (int i = 0; i < params->values_count; i++) {
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, i, &input));
    TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                      output->params.zero_point);
    TF_LITE_ENSURE_EQ(context, input->params.scale, output->params.scale);
  }

  if (noop) {
    SetTensorToPersistentRo(output);
    context->ResizeTensor(context, output, output_shape);
    return EvalImpl(context, node, output, params);
  }
  return context->ResizeTensor(context, output, output_shape);
}

template <typename T>
TfLiteStatus PackImpl(TfLiteContext* context, TfLiteNode* node,
                      TfLiteTensor* output, int values_count, int axis) {
  TF_LITE_ENSURE(context, axis >= 0);

  VectorOfTensors<T> all_inputs(*context, *node->inputs);
  tflite::PackParams op_params;
  op_params.axis = axis;
  op_params.inputs_count = values_count;

  reference_ops::Pack<T>(op_params, all_inputs.shapes(), all_inputs.data(),
                         GetTensorShape(output), GetTensorData<T>(output));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* data = reinterpret_cast<const OpData*>(node->user_data);
  if (data->noop) {
    return kTfLiteOk;
  }
  const TfLitePackParams* params =
      reinterpret_cast<TfLitePackParams*>(node->builtin_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  return EvalImpl(context, node, output, params);
}

TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node,
                      TfLiteTensor* output, const TfLitePackParams* params) {
  switch (output->type) {
    case kTfLiteFloat32: {
      return PackImpl<float>(context, node, output, params->values_count,
                             params->axis);
    }
    case kTfLiteUInt8: {
      return PackImpl<uint8_t>(context, node, output, params->values_count,
                               params->axis);
    }
    case kTfLiteInt8: {
      return PackImpl<int8_t>(context, node, output, params->values_count,
                              params->axis);
    }
    case kTfLiteInt16: {
      return PackImpl<int16_t>(context, node, output, params->values_count,
                               params->axis);
    }
    case kTfLiteInt32: {
      return PackImpl<int32_t>(context, node, output, params->values_count,
                               params->axis);
    }
    case kTfLiteInt64: {
      return PackImpl<int64_t>(context, node, output, params->values_count,
                               params->axis);
    }
    default: {
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by pack.",
                         TfLiteTypeGetName(output->type));
      return kTfLiteError;
    }
  }
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
