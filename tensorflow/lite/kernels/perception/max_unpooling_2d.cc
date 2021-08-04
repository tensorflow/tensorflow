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
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
namespace tflite {
namespace ops {
namespace custom {

namespace max_unpooling_2d {

constexpr int kDataInputTensor = 0;
constexpr int kIndicesTensor = 1;
constexpr int kOutputTensor = 0;

// TODO(b/175003241): Move this logic to lite/kernels/internal when promoting
// this op to a builtin op.
inline void MaxUnpooling(const RuntimeShape& input_shape,
                         const float* input_data, const int32_t* indices_data,
                         const RuntimeShape& output_shape, float* output_data) {
  std::memset(output_data, 0, output_shape.FlatSize() * sizeof(float));
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int batch_stride =
      output_shape.Dims(1) * output_shape.Dims(2) * output_shape.Dims(3);
  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < input_shape.Dims(1); ++in_y) {
      for (int in_x = 0; in_x < input_shape.Dims(2); ++in_x) {
        for (int channel = 0; channel < depth; ++channel) {
          const auto input_offset =
              Offset(input_shape, batch, in_y, in_x, channel);
          int idx = indices_data[input_offset];
          output_data[batch * batch_stride + idx] = input_data[input_offset];
        }
      }
    }
  }
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

  MaxUnpooling(GetTensorShape(input), GetTensorData<float>(input),
               GetTensorData<int32_t>(indices), GetTensorShape(output),
               GetTensorData<float>(output));
  return kTfLiteOk;
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
