/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

constexpr int kInputTensor = 0;
constexpr int kInputPositions = 1;
constexpr int kOutputTensor = 0;

template <typename InputT, typename CoordsT = int32_t>
TfLiteStatus Gather(const TfLiteGatherParams* params,
                    const TfLiteEvalTensor* input,
                    const TfLiteEvalTensor* coords, TfLiteEvalTensor* output) {
  const InputT* input_data = tflite::micro::GetTensorData<InputT>(input);
  const CoordsT* coords_data = tflite::micro::GetTensorData<CoordsT>(coords);
  InputT* output_data = tflite::micro::GetTensorData<InputT>(output);
  const TfLiteIntArray* input_dims = input->dims;
  const int input_dims_size = input_dims->size;
  int axis = params->axis;
  if (axis < 0) {
    axis += input_dims_size;
  }
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, input_dims_size);

  int batch_dims = params->batch_dims;
  // batch_dims should be in range: [-rank(coords), rank(coords)].
  // Negative batch_dims is added with rank of coords.
  const TfLiteIntArray* coords_dims = coords->dims;
  const int coords_dims_size = coords_dims->size;
  if (batch_dims < 0) {
    batch_dims += coords_dims_size;
  }
  TFLITE_DCHECK_GE(batch_dims, 0);
  TFLITE_DCHECK_LT(batch_dims, input_dims_size);
  TFLITE_DCHECK_LE(batch_dims, coords_dims_size);
  TFLITE_DCHECK_GE(axis, batch_dims);
  for (int i = 0; i < batch_dims; ++i) {
    TFLITE_DCHECK_EQ(input_dims->data[i], coords_dims->data[i]);
  }

  const int axis_size = input_dims->data[axis];

  int batch_size = 1;
  for (int i = 0; i < batch_dims; ++i) {
    batch_size *= input_dims->data[i];
  }
  int outer_size = 1;
  for (int i = batch_dims; i < axis; ++i) {
    outer_size *= input_dims->data[i];
  }
  int inner_size = 1;
  for (int i = axis + 1; i < input_dims_size; ++i) {
    inner_size *= input_dims->data[i];
  }
  int coord_size = 1;
  for (int i = batch_dims; i < coords_dims_size; ++i) {
    coord_size *= coords_dims->data[i];
  }

  for (int batch = 0; batch < batch_size; ++batch) {
    for (int outer = 0; outer < outer_size; ++outer) {
      for (int coord = 0; coord < coord_size; ++coord) {
        TFLITE_DCHECK_GE(coords_data[coord], 0);
        TFLITE_DCHECK_LT(coords_data[coord], axis_size);
        std::memcpy(output_data +
                        (((batch * outer_size) + outer) * coord_size + coord) *
                            inner_size,
                    input_data + (((batch * outer_size) + outer) * axis_size +
                                  coords_data[batch * coord_size + coord]) *
                                     inner_size,
                    sizeof(InputT) * inner_size);
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  const TfLiteTensor* coords;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputPositions, &coords));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  switch (coords->type) {
    case kTfLiteInt32:
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Positions of type '%s' are not supported by gather.",
                         TfLiteTypeGetName(coords->type));
      return kTfLiteError;
      break;
  }

  // Assign to output the input type.
  output->type = input->type;

  // Check conditions for different types.
  switch (input->type) {
    case kTfLiteFloat32:
    case kTfLiteInt8:
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
      break;
  }

  int axis = params->axis;
  if (axis < 0) {
    axis += NumDimensions(input);
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < NumDimensions(input));

  int batch_dims = params->batch_dims;
  // batch_dims should be in range: [-rank(coords), rank(coords)].
  // Negative batch_dims is added with rank of coords.
  if (batch_dims < 0) {
    batch_dims += NumDimensions(coords);
  }
  TF_LITE_ENSURE(context, batch_dims <= axis);
  TF_LITE_ENSURE(context, 0 <= batch_dims && batch_dims < NumDimensions(input));
  TF_LITE_ENSURE(context, batch_dims <= NumDimensions(coords));
  for (int i = 0; i < batch_dims; ++i) {
    TF_LITE_ENSURE_EQ(context, input->dims->data[i], coords->dims->data[i]);
  }

  // GATHER updates the output tensor dimensions, but TfLiteTensor in the
  // MicroInterpreter is a temporary allocation. We must therefore relocate the
  // dims from the FlatBuffer to the persistant storage arena.
  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));

  TfLiteIntArray* output_shape = output->dims;
  output_shape->size =
      NumDimensions(input) + NumDimensions(coords) - 1 - batch_dims;
  int output_index = 0;
  for (int i = 0; i < axis; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  for (int i = batch_dims; i < coords->dims->size; ++i) {
    output_shape->data[output_index++] = coords->dims->data[i];
  }
  for (int i = axis + 1; i < input->dims->size; ++i) {
    output_shape->data[output_index++] = input->dims->data[i];
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const auto* params =
      reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* coords =
      tflite::micro::GetEvalInput(context, node, kInputPositions);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  if (coords->type == kTfLiteInt32) {
    switch (input->type) {
      case kTfLiteFloat32:
        return Gather<float, int32_t>(params, input, coords, output);
        break;
      case kTfLiteInt8:
        return Gather<int8_t, int32_t>(params, input, coords, output);
        break;
      default:
        TF_LITE_KERNEL_LOG(context, "Type '%s' is not supported by gather.",
                           TfLiteTypeGetName(input->type));
        return kTfLiteError;
        break;
    }
  }
  return kTfLiteOk;
}
}  // namespace

TfLiteRegistration Register_GATHER() {
  return {/*init=*/nullptr,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
