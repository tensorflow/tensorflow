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

#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
namespace tflite {
namespace ops {
namespace builtin {
namespace tile {

constexpr int kInputTensor = 0;
constexpr int kInputMultipliers = 1;
constexpr int kOutputTensor = 0;

namespace {
template <typename T>
TfLiteIntArray* MultiplyShapeDims(const TfLiteIntArray& shape,
                                  const TfLiteTensor* multipliers,
                                  int num_dimensions) {
  const T* multipliers_v = GetTensorData<T>(multipliers);

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(num_dimensions);
  for (int i = 0; i < num_dimensions; ++i) {
    output_shape->data[i] = shape.data[i] * multipliers_v[i];
  }
  return output_shape;
}

TfLiteStatus ResizeOutput(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* multipliers = GetInput(context, node, kInputMultipliers);

  const int num_dimensions = NumDimensions(input);
  const int num_multipliers = NumElements(multipliers);
  TF_LITE_ENSURE_EQ(context, num_dimensions, num_multipliers);
  switch (multipliers->type) {
    case kTfLiteInt32:
      return context->ResizeTensor(
          context, output,
          MultiplyShapeDims<int32_t>(*input->dims, multipliers,
                                     num_dimensions));
    case kTfLiteInt64:
      return context->ResizeTensor(
          context, output,
          MultiplyShapeDims<int64_t>(*input->dims, multipliers,
                                     num_dimensions));
    default:
      context->ReportError(
          context, "Multipliers of type '%s' are not supported by tile.",
          TfLiteTypeGetName(multipliers->type));
      return kTfLiteError;
  }
}

template <typename T, typename M>
void CopyMultipleTimes(const T* in_data, int32_t in_size, M multiplier,
                       T* out_data) {
  for (M i = 0; i < multiplier; ++i) {
    const T* in_end = in_data + in_size;
    T* new_out_data = std::copy(in_data, in_end, out_data);
    in_data = out_data;
    out_data = new_out_data;
  }
}

template <typename T, typename M>
std::pair<int, int> TileOneDimension(const TfLiteIntArray& in_dimensions,
                                     const T* in_data, const M* multipliers,
                                     T* out_data, int dimension) {
  const int dimension_size = in_dimensions.data[dimension];
  if (dimension == in_dimensions.size - 1) {
    CopyMultipleTimes(in_data, dimension_size, multipliers[dimension],
                      out_data);
    return std::make_pair(
        dimension_size,
        dimension_size * static_cast<int>(multipliers[dimension]));
  }
  int total_stride_size = 0, total_tiled_stride_size = 0;
  const T* copy_from_data = in_data;
  T* copy_to_data = out_data;
  for (int i = 0; i < dimension_size; ++i) {
    int stride_size = 0, tiled_stride_size = 0;
    std::tie(stride_size, tiled_stride_size) =
        TileOneDimension(in_dimensions, copy_from_data, multipliers,
                         copy_to_data, dimension + 1);
    copy_from_data += stride_size;
    copy_to_data += tiled_stride_size;
    total_stride_size += stride_size;
    total_tiled_stride_size += tiled_stride_size;
  }
  CopyMultipleTimes(out_data, total_tiled_stride_size,
                    multipliers[dimension] - 1,
                    out_data + total_tiled_stride_size);
  return std::make_pair(
      total_stride_size,
      static_cast<int>(total_tiled_stride_size * multipliers[dimension]));
}

template <typename T>
void Tile(const TfLiteIntArray& in_dimensions, const TfLiteTensor* in_data,
          const TfLiteTensor* multipliers, TfLiteTensor* out_data) {
  // Doing recursively tiling from top to down dimension.
  switch (multipliers->type) {
    case kTfLiteInt32:
      TileOneDimension(in_dimensions, GetTensorData<T>(in_data),
                       GetTensorData<int32_t>(multipliers),
                       GetTensorData<T>(out_data), 0);
      break;
    case kTfLiteInt64:
      TileOneDimension(in_dimensions, GetTensorData<T>(in_data),
                       GetTensorData<int64_t>(multipliers),
                       GetTensorData<T>(out_data), 0);
      break;
    default:
      break;
  }
}
}  // namespace

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);

  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  const TfLiteTensor* multipliers = GetInput(context, node, kInputMultipliers);
  // Only int32 and int64 multipliers type is supported.
  if (multipliers->type != kTfLiteInt32 && multipliers->type != kTfLiteInt64) {
    context->ReportError(context,
                         "Multipliers of type '%s' are not supported by tile.",
                         TfLiteTypeGetName(multipliers->type));
    return kTfLiteError;
  }

  if (IsConstantTensor(multipliers)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  } else {
    SetTensorToDynamic(output);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  const TfLiteTensor* multipliers = GetInput(context, node, kInputMultipliers);

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutput(context, node));
  }

  switch (output->type) {
    case kTfLiteFloat32:
      Tile<float>(*(input->dims), input, multipliers, output);
      break;
    case kTfLiteUInt8:
      Tile<uint8_t>(*(input->dims), input, multipliers, output);
      break;
    case kTfLiteInt32:
      Tile<int32_t>(*(input->dims), input, multipliers, output);
      break;
    case kTfLiteInt64:
      Tile<int64_t>(*(input->dims), input, multipliers, output);
      break;
    case kTfLiteBool:
      Tile<bool>(*(input->dims), input, multipliers, output);
      break;
    default:
      context->ReportError(context, "Type '%s' is not supported by tile.",
                           TfLiteTypeGetName(output->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace tile
TfLiteRegistration* Register_TILE() {
  static TfLiteRegistration r = {nullptr, nullptr, tile::Prepare, tile::Eval};
  return &r;
}
}  // namespace builtin
}  // namespace ops
}  // namespace tflite
