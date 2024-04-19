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
#include "tensorflow/lite/experimental/ml_adjacent/algo/rgb_to_grayscale.h"

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace ml_adj {
namespace rgb_to_grayscale {
namespace {

using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;

inline void ConvertRgbToGrayscale(dim_t batches, dim_t height, dim_t width,
                                  const float* input_data, float* output_data) {
  const dim_t output_num_pixels = batches * width * height;
  // Reference for converting between RGB and grayscale. Same as in
  // tf.image.rgb_to_convert: https://en.wikipedia.org/wiki/Luma_%28video%29.
  constexpr float kRgb2GrayscaleKernel[] = {0.2989f, 0.5870f, 0.1140f};
  const float* src_ptr = input_data;
  float* dst_ptr = output_data;
  for (int i = 0; i < output_num_pixels; ++i) {
    *dst_ptr = kRgb2GrayscaleKernel[0] * src_ptr[0] +
               kRgb2GrayscaleKernel[1] * src_ptr[1] +
               kRgb2GrayscaleKernel[2] * src_ptr[2];
    src_ptr += 3;  // Step is number of input channels, which is 3 for RGB.
    dst_ptr++;
  }
}

// Converts each image in input from RGB to grayscale. Works for float datatype.
void ComputeRgbToGrayscale(const InputPack& inputs, const OutputPack& outputs) {
  TFLITE_DCHECK(inputs.size() == 1);
  TFLITE_DCHECK(outputs.size() == 1);

  // Extract input image data.
  const DataRef* img = inputs[0];
  const float* img_data = reinterpret_cast<const float*>(img->Data());
  const dim_t img_num_batches = img->Dims()[0];
  const dim_t img_height = img->Dims()[1];
  const dim_t img_width = img->Dims()[2];
  const dim_t channels = img->Dims()[3];
  TFLITE_DCHECK(channels == 3);

  // Resize output buffer for single-channel output image.
  MutableDataRef* output = outputs[0];
  output->Resize({img_num_batches, img_height, img_width, 1});
  float* output_data = reinterpret_cast<float*>(output->Data());

  ConvertRgbToGrayscale(img_num_batches, img_height, img_width, img_data,
                        output_data);
}

}  // namespace

const Algo* Impl_RgbToGrayscale() {
  static const Algo rgb_to_grayscale = {&ComputeRgbToGrayscale, nullptr};
  return &rgb_to_grayscale;
}

}  // namespace rgb_to_grayscale
}  // namespace ml_adj
