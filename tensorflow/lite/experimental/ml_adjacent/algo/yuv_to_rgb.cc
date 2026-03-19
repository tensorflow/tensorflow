/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/ml_adjacent/algo/image_utils.h"
#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace ml_adj {
namespace yuv_to_rgb {
namespace {

using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;

constexpr float kYuv2RgbKernel[] = {1.0f,         0.0f,
                                    1.13988303f,  //
                                    1.0f,         -0.394642334f,
                                    -0.58062185f,  //
                                    1.0f,         2.03206185f,   0.0f};
constexpr int kYuv2RgbKernelDim =
    sizeof(kYuv2RgbKernel) / sizeof(kYuv2RgbKernel[0]);

// Converts YUV to RGB. Supports `float` datatype only.
void ComputeYuvToRgb(const InputPack& inputs, const OutputPack& outputs) {
  TFLITE_DCHECK(inputs.size() == 1);
  TFLITE_DCHECK(outputs.size() == 1);

  // Extract input image data.
  const DataRef* img = inputs[0];
  const float* input_data = reinterpret_cast<const float*>(img->Data());
  const dim_t batches = img->Dims()[0];
  const dim_t height = img->Dims()[1];
  const dim_t width = img->Dims()[2];
  const dim_t channels = img->Dims()[3];

  // Resize output buffer.
  MutableDataRef* output = outputs[0];
  output->Resize({batches, height, width, channels});
  float* output_data = reinterpret_cast<float*>(output->Data());

  ConvertColorSpace(batches, height, width, input_data, output_data,
                    &kYuv2RgbKernel[0], kYuv2RgbKernelDim);
}

}  // namespace

const Algo* Impl_YuvToRgb() {
  static const Algo yuv_to_rgb = {&ComputeYuvToRgb, nullptr};
  return &yuv_to_rgb;
}

}  // namespace yuv_to_rgb
}  // namespace ml_adj
