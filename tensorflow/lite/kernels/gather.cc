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
#include <stdint.h>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace gather {
constexpr int kInputTensor = 0;
constexpr int kInputPositions = 1;
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

TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node);

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* positions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputPositions, &positions));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  switch (positions->type) {
    case kTfLiteInt64:
    case kTfLiteInt32:
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Positions of type '%s' are not supported by gather.",
                         TfLiteTypeGetName(positions->type));
      return kTfLiteError;
  }

  // Assign to output the input type.
  output->type = input->type;

  // Check conditions for different types.
  switch (input->type) {
    case kTfLiteFloat32:
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt16:
    case kTfLiteInt64:
    case kTfLiteInt32:
    case kTfLiteBool:
      break;
    case kTfLiteString: {
      // Only 1D input is supported.
      TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }

  int axis = params->axis;
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < NumDimensions(input));

  int batch_dims = params->batch_dims;
  // batch_dims should be in range: [-rank(positions), rank(positions)].
  // Negative batch_dims is added with rank of positions.
  if (batch_dims < 0) {
    batch_dims += NumDimensions(positions);
  }
  TF_LITE_ENSURE(context, batch_dims <= axis);
  TF_LITE_ENSURE(context, 0 <= batch_dims && batch_dims < NumDimensions(input));
  TF_LITE_ENSURE(context, batch_dims <= NumDimensions(positions));
  for (int i = 0; i < batch_dims; ++i) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[i], positions->dims->data[i]);
  }

  const int num_dimensions =
      NumDimensions(input) + NumDimensions(positions) - 1 - batch_dims;
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  int output_index = 0;
  for (int i = 0; i < axis; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  for (int i = batch_dims; i < positions->dims->size; ++i) {
    output_shape->data[output_index++] = positions->dims->data[i];
  }
  for (int i = axis + 1; i < input->dims->size; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  data->noop = IsConstantOrPersistentTensor(input) &&
               IsConstantOrPersistentTensor(positions);
  if (data->noop) {
    SetTensorToPersistentRo(output);
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, output, output_shape));
    return EvalImpl(context, node);
  } else {
    return context->ResizeTensor(context, output, output_shape);
  }
}

template <typename InputT, typename PositionsT>
TfLiteStatus Gather(TfLiteContext* context, const TfLiteGatherParams& params,
                    const TfLiteTensor* input, const TfLiteTensor* positions,
                    TfLiteTensor* output) {
  const PositionsT* indexes = GetTensorData<PositionsT>(positions);
  bool indices_has_only_positive_elements = true;
  const size_t num_indices = positions->bytes / sizeof(PositionsT);
  for (size_t i = 0; i < num_indices; i++) {
    if (indexes[i] < 0) {
      indices_has_only_positive_elements = false;
      break;
    }
  }
  TF_LITE_ENSURE(context, indices_has_only_positive_elements);

  tflite::GatherParams op_params;
  op_params.axis = params.axis;
  op_params.batch_dims = params.batch_dims;
  return optimized_ops::Gather(
      op_params, GetTensorShape(input), GetTensorData<InputT>(input),
      GetTensorShape(positions), GetTensorData<PositionsT>(positions),
      GetTensorShape(output), GetTensorData<InputT>(output));
}

template <typename PositionT>
TfLiteStatus GatherStrings(TfLiteContext* context, const TfLiteTensor* input,
                           const TfLiteTensor* positions,
                           TfLiteTensor* output) {
  DynamicBuffer buffer;

  const PositionT* indexes = GetTensorData<PositionT>(positions);
  bool indices_has_only_positive_elements = true;
  const size_t num_indices = positions->bytes / sizeof(PositionT);
  for (size_t i = 0; i < num_indices; i++) {
    if (indexes[i] < 0) {
      indices_has_only_positive_elements = false;
      break;
    }
  }
  TF_LITE_ENSURE(context, indices_has_only_positive_elements);

  const PositionT num_strings = GetStringCount(input);
  const int num_indexes = NumElements(positions);

  for (int i = 0; i < num_indexes; ++i) {
    const PositionT pos = indexes[i];
    TF_LITE_ENSURE(context, pos < num_strings);
    const auto string_ref = GetString(input, pos);
    buffer.AddString(string_ref.str, string_ref.len);
  }
  buffer.WriteToTensor(output, /*new_shape=*/nullptr);
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const OpData* data = reinterpret_cast<OpData*>(node->user_data);
  if (data->noop) {
    return kTfLiteOk;
  } else {
    return EvalImpl(context, node);
  }
}

TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* positions;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputPositions, &positions));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  TfLiteStatus status = kTfLiteError;
  if (positions->type == kTfLiteInt32) {
    switch (input->type) {
      case kTfLiteFloat32:
        status =
            Gather<float, int32_t>(context, *params, input, positions, output);
        break;
      case kTfLiteUInt8:
        status = Gather<uint8_t, int32_t>(context, *params, input, positions,
                                          output);
        break;
      case kTfLiteInt8:
        status =
            Gather<int8_t, int32_t>(context, *params, input, positions, output);
        break;
      case kTfLiteInt16:
        status = Gather<int16_t, int32_t>(context, *params, input, positions,
                                          output);
        break;
      case kTfLiteInt32:
        status = Gather<int32_t, int32_t>(context, *params, input, positions,
                                          output);
        break;
      case kTfLiteInt64:
        status = Gather<int64_t, int32_t>(context, *params, input, positions,
                                          output);
        break;
      case kTfLiteBool:
        status =
            Gather<bool, int32_t>(context, *params, input, positions, output);
        break;
      case kTfLiteString:
        status = GatherStrings<int32_t>(context, input, positions, output);
        break;
      default:
        TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                           TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }
  if (positions->type == kTfLiteInt64) {
    switch (input->type) {
      case kTfLiteFloat32:
        status =
            Gather<float, int64_t>(context, *params, input, positions, output);
        break;
      case kTfLiteUInt8:
        status = Gather<uint8_t, int64_t>(context, *params, input, positions,
                                          output);
        break;
      case kTfLiteInt8:
        status =
            Gather<int8_t, int64_t>(context, *params, input, positions, output);
        break;
      case kTfLiteInt16:
        status = Gather<int16_t, int64_t>(context, *params, input, positions,
                                          output);
        break;
      case kTfLiteInt32:
        status = Gather<int32_t, int64_t>(context, *params, input, positions,
                                          output);
        break;
      case kTfLiteInt64:
        status = Gather<int64_t, int64_t>(context, *params, input, positions,
                                          output);
        break;
      case kTfLiteBool:
        status =
            Gather<bool, int64_t>(context, *params, input, positions, output);
        break;
      case kTfLiteString:
        status = GatherStrings<int64_t>(context, input, positions, output);
        break;
      default:
        TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                           TfLiteTypeGetName(input->type));
        return kTfLiteError;
    }
  }
  if (status != kTfLiteOk) {
    TF_LITE_KERNEL_LOG(context, "gather index out of bounds");
  }
  return status;
  TF_LITE_KERNEL_LOG(context,
                     "Positions of type '%s' are not supported by gather.",
                     TfLiteTypeGetName(positions->type));
  return kTfLiteError;
}
}  // namespace gather

TfLiteRegistration* Register_GATHER() {
  static TfLiteRegistration r = {gather::Init, gather::Free, gather::Prepare,
                                 gather::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
