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
#include <cstdint>
#include <cstring>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
namespace tflite {
namespace ops {
namespace custom {

namespace max_unpooling_2d {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

// TODO(b/175003241): Move this logic to lite/kernels/internal when promoting
// this op to a builtin op.
inline TfLiteStatus MaxUnpooling(TfLiteContext* context,
                                 const RuntimeShape& input_shape,
                                 const float* input_data,
                                 const int32_t* indices_data,
                                 const RuntimeShape& output_shape,
                                 float* output_data) {
  int input_count = 0;
  int output_count = 0;
  TF_LITE_ENSURE_MSG(
      context,
      input_shape.CheckedNumElementsInRange(
          0, input_shape.DimensionsCount(), input_count),
      "%s", "MaxUnpooling2D input size overflowed.");
  TF_LITE_ENSURE_MSG(
      context,
      output_shape.CheckedNumElementsInRange(
          0, output_shape.DimensionsCount(), output_count),
      "%s", "MaxUnpooling2D output size overflowed.");
  if (input_count > 0) {
    TF_LITE_ENSURE(context, input_data != nullptr);
    TF_LITE_ENSURE(context, indices_data != nullptr);
  }
  if (output_count > 0) {
    TF_LITE_ENSURE(context, output_data != nullptr);
    std::memset(output_data, 0, output_count * sizeof(float));
  }

  const int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int32_t input_height = input_shape.Dims(1);
  const int32_t input_width = input_shape.Dims(2);
  const int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);
  int batch_stride = 0;
  TF_LITE_ENSURE_MSG(context,
                     output_shape.CheckedSizeFromDimension(1, batch_stride),
                     "%s", "MaxUnpooling2D batch stride overflowed.");
  for (int i = 0; i < input_count; ++i) {
    const int32_t idx = indices_data[i];
    TF_LITE_ENSURE_MSG(context, idx >= 0 && idx < batch_stride,
                       "Invalid MaxUnpooling2D index.");
  }

  for (int32_t batch = 0; batch < batches; ++batch) {
    int batch_offset = 0;
    TF_LITE_ENSURE_OK(
        context, CheckedShapeProductToInt(
                     context, {batch, batch_stride},
                     "MaxUnpooling2D batch offset overflowed.", batch_offset));
    for (int32_t in_y = 0; in_y < input_height; ++in_y) {
      for (int32_t in_x = 0; in_x < input_width; ++in_x) {
        for (int32_t channel = 0; channel < depth; ++channel) {
          const auto input_offset =
              Offset(input_shape, batch, in_y, in_x, channel);
          const int32_t idx = indices_data[input_offset];
          const int output_offset = batch_offset + idx;
          output_data[output_offset] = input_data[input_offset];
        }
      }
    }
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<const TfLitePoolParams*>(node->custom_initial_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* indices = GetInput(context, node, kIndicesTensor);
  TF_LITE_ENSURE(context, indices != nullptr);
  TF_LITE_ENSURE_EQ(context, NumDimensions(indices), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, indices->type, kTfLiteInt32);
  TF_LITE_ENSURE(context, params->padding != kTfLitePaddingUnknown);

  // Size of input and indices tensor must match.
  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape indices_shape = GetTensorShape(indices);
  TF_LITE_ENSURE_MSG(
      context, input_shape.DimensionsCount() == indices_shape.DimensionsCount(),
      "Input and indices must have the same shape.");
  for (int i = 0; i < input_shape.DimensionsCount(); ++i) {
    TF_LITE_ENSURE_MSG(context, input_shape.Dims(i) == indices_shape.Dims(i),
                       "Input and indices must have the same shape.");
  }

  int batches = input->dims->data[0];
  int height = input->dims->data[1];
  int width = input->dims->data[2];
  int channels_out = input->dims->data[3];

  int out_width, out_height;
  if (params->padding == kTfLitePaddingSame) {
    out_width = width * params->stride_width;
    out_height = height * params->stride_height;
  } else {
    out_width = (width - 1) * params->stride_width + params->filter_width;
    out_height = (height - 1) * params->stride_height + params->filter_height;
  }

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(4);
  output_size->data[0] = batches;
  output_size->data[1] = out_height;
  output_size->data[2] = out_width;
  output_size->data[3] = channels_out;
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* indices = GetInput(context, node, kIndicesTensor);
  TF_LITE_ENSURE(context, indices != nullptr);

  const RuntimeShape input_shape = GetTensorShape(input);
  TF_LITE_ENSURE(context, TfLiteIntArrayEqual(input->dims, indices->dims));

  return MaxUnpooling(context, input_shape, GetTensorData<float>(input),
                      GetTensorData<int32_t>(indices), GetTensorShape(output),
                      GetTensorData<float>(output));
}

}  // namespace max_unpooling_2d

TfLiteRegistration* RegisterMaxUnpooling2D() {
  static TfLiteRegistration reg = {/*init=*/nullptr,
                                   /*free=*/nullptr, max_unpooling_2d::Prepare,
                                   max_unpooling_2d::Eval};
  return &reg;
}

// Alias for selective build.
TfLiteRegistration* Register_MAX_UNPOOLING2D() {
  return RegisterMaxUnpooling2D();
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
