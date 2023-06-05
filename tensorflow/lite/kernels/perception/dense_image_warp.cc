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
#include <algorithm>
#include <cmath>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace custom {
namespace dense_image_warp {

constexpr int kInputTensor = 0;
constexpr int kFlowTensor = 1;
constexpr int kOutputTensor = 0;

inline void DenseImageWarp(const RuntimeShape& input_shape,
                           const float* input_data,
                           const RuntimeShape& flow_shape,
                           const float* flow_data, float* output_data) {
  const int batches = MatchingDim(input_shape, 0, flow_shape, 0);
  const int height = MatchingDim(input_shape, 1, flow_shape, 1);
  const int width = MatchingDim(input_shape, 2, flow_shape, 2);
  const int channels = input_shape.Dims(3);
  TFLITE_CHECK_EQ(flow_shape.Dims(3), 2);

  // Max values to make sure the indexes are not out of bound.
  const int max_floor_y = height - 2;
  const int max_floor_x = width - 2;

  for (int batch = 0; batch < batches; ++batch) {
    for (int in_y = 0; in_y < height; ++in_y) {
      for (int in_x = 0; in_x < width; ++in_x) {
        float querry_point_y =
            in_y - flow_data[Offset(flow_shape, batch, in_y, in_x, 0)];
        float querry_point_x =
            in_x - flow_data[Offset(flow_shape, batch, in_y, in_x, 1)];

        int floor_y =
            std::min(std::max(0, static_cast<int>(std::floor(querry_point_y))),
                     max_floor_y);
        int floor_x =
            std::min(std::max(0, static_cast<int>(std::floor(querry_point_x))),
                     max_floor_x);
        float alpha_y =
            std::min(std::max(0.0f, querry_point_y - floor_y), 1.0f);
        float alpha_x =
            std::min(std::max(0.0f, querry_point_x - floor_x), 1.0f);

        for (int c = 0; c < channels; ++c) {
          float top_left =
              input_data[Offset(input_shape, batch, floor_y, floor_x, c)];
          float top_right =
              input_data[Offset(input_shape, batch, floor_y, floor_x + 1, c)];
          float bottom_left =
              input_data[Offset(input_shape, batch, floor_y + 1, floor_x, c)];
          float bottom_right = input_data[Offset(input_shape, batch,
                                                 floor_y + 1, floor_x + 1, c)];

          float interp_top = alpha_x * (top_right - top_left) + top_left;
          float interp_bottom =
              alpha_x * (bottom_right - bottom_left) + bottom_left;
          float interp = alpha_y * (interp_bottom - interp_top) + interp_top;
          output_data[Offset(input_shape, batch, in_y, in_x, c)] = interp;
        }
      }
    }
  }
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  // Check inputs and output.
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* flow = GetInput(context, node, kFlowTensor);
  TF_LITE_ENSURE(context, flow != nullptr);

  // Check types.
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, flow->type, kTfLiteFloat32);

  // Check shapes. If input has shape of [b, h, w, c], flow must have shape of
  // [b, h, w, 2].
  TF_LITE_ENSURE_EQ(context, NumDimensions(flow), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape flow_shape = GetTensorShape(flow);
  TF_LITE_ENSURE_EQ(context, input_shape.Dims(0), flow_shape.Dims(0));
  TF_LITE_ENSURE_EQ(context, input_shape.Dims(1), flow_shape.Dims(1));
  TF_LITE_ENSURE_EQ(context, input_shape.Dims(2), flow_shape.Dims(2));
  TF_LITE_ENSURE_MSG(context, input_shape.Dims(1) >= 2,
                     "Image height must be at least 2.");
  TF_LITE_ENSURE_MSG(context, input_shape.Dims(2) >= 2,
                     "Image width must be at least 2.");
  TF_LITE_ENSURE_MSG(context, flow_shape.Dims(3) == 2,
                     "The last dimension of flow tensor must be 2.");

  TfLiteIntArray* output_size = TfLiteIntArrayCopy(input->dims);
  return context->ResizeTensor(context, output, output_size);
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* flow = GetInput(context, node, kFlowTensor);
  TF_LITE_ENSURE(context, flow != nullptr);

  DenseImageWarp(GetTensorShape(input), GetTensorData<float>(input),
                 GetTensorShape(flow), GetTensorData<float>(flow),
                 GetTensorData<float>(output));
  return kTfLiteOk;
}

}  // namespace dense_image_warp

TfLiteRegistration* RegisterDenseImageWarp() {
  static TfLiteRegistration reg = {/*init=*/nullptr,
                                   /*free=*/nullptr, dense_image_warp::Prepare,
                                   dense_image_warp::Eval};
  return &reg;
}

// Alias for selective build.
TfLiteRegistration* Register_DENSE_IMAGE_WARP() {
  return RegisterDenseImageWarp();
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
