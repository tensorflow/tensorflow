/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include <string.h>
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace gather {
constexpr int kInputTensor = 0;
constexpr int kInputPositions = 1;
constexpr int kOutputTensor = 0;

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* positions = GetInput(context, node, kInputPositions);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (positions->type) {
    case kTfLiteInt64:
    case kTfLiteInt32:
      break;
    default:
      context->ReportError(
          context, "Positions of type '%s' are not supported by gather.",
          TfLiteTypeGetName(positions->type));
      return kTfLiteError;
  }

  // Assign to output the input type.
  output->type = input->type;

  // Check conditions for different types.
  switch (input->type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt64:
    case kTfLiteInt32:
      break;
    case kTfLiteString: {
      // Only 1D input is supported.
      TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
    } break;
    default:
      context->ReportError(context, "Type '%s' is not supported by gather.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  int axis = params->axis;
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < NumDimensions(input));

  const int num_dimensions =
      NumDimensions(input) + NumDimensions(positions) - 1;
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  int output_index = 0;
  for (int i = 0; i < axis; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  for (int i = 0; i < positions->dims->size; ++i) {
    output_shape->data[output_index++] = positions->dims->data[i];
  }
  for (int i = axis + 1; i < input->dims->size; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  return context->ResizeTensor(context, output, output_shape);
}

template <typename InputT, typename PositionsT>
TfLiteStatus Gather(const TfLiteGatherParams& params, const TfLiteTensor* input,
                    const TfLiteTensor* positions, TfLiteTensor* output) {
  tflite::GatherParams op_params;
  op_params.axis = params.axis;
  optimized_ops::Gather(op_params, GetTensorShape(input),
                        GetTensorData<InputT>(input), GetTensorShape(positions),
                        GetTensorData<PositionsT>(positions),
                        GetTensorShape(output), GetTensorData<InputT>(output));
  return kTfLiteOk;
}

template <typename PositionT>
TfLiteStatus GatherStrings(TfLiteContext* context, const TfLiteTensor* input,
                           const TfLiteTensor* positions,
                           TfLiteTensor* output) {
  // TODO(mgubin): Currently support only for 1D output tensors.
  DynamicBuffer buffer;
  const PositionT* indexes = GetTensorData<PositionT>(positions);
  const PositionT num_strings = GetStringCount(input);
  for (int i = 0; i < positions->dims->data[0]; ++i) {
    const PositionT pos = indexes[i];
    TF_LITE_ENSURE(context, pos < num_strings);
    const auto string_ref = GetString(input, pos);
    buffer.AddString(string_ref.str, string_ref.len);
  }
  buffer.WriteToTensor(output);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* positions = GetInput(context, node, kInputPositions);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  if (positions->type == kTfLiteInt32) {
    switch (input->type) {
      case kTfLiteFloat32:
        return Gather<float, int32_t>(*params, input, positions, output);
      case kTfLiteUInt8:
        return Gather<uint8_t, int32_t>(*params, input, positions, output);
      case kTfLiteInt32:
        return Gather<int32_t, int32_t>(*params, input, positions, output);
      case kTfLiteInt64:
        return Gather<int64_t, int32_t>(*params, input, positions, output);
      case kTfLiteString:
        return GatherStrings<int32_t>(context, input, positions, output);
      default:
        context->ReportError(context, "Type '%s' is not supported by gather.",
                             TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }
  if (positions->type == kTfLiteInt64) {
    switch (input->type) {
      case kTfLiteFloat32:
        return Gather<float, int64_t>(*params, input, positions, output);
      case kTfLiteUInt8:
        return Gather<uint8_t, int64_t>(*params, input, positions, output);
      case kTfLiteInt32:
        return Gather<int32_t, int64_t>(*params, input, positions, output);
      case kTfLiteInt64:
        return Gather<int64_t, int64_t>(*params, input, positions, output);
      case kTfLiteString:
        return GatherStrings<int64_t>(context, input, positions, output);
      default:
        context->ReportError(context, "Type '%s' is not supported by gather.",
                             TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }
  context->ReportError(context,
                       "Positions of type '%s' are not supported by gather.",
                       TfLiteTypeGetName(positions->type));
  return kTfLiteError;
}
}  // namespace gather

TfLiteRegistration* Register_GATHER() {
  static TfLiteRegistration r = {nullptr, nullptr, gather::Prepare,
                                 gather::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
