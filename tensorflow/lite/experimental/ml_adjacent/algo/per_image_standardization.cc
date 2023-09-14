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
#include "tensorflow/lite/experimental/ml_adjacent/algo/per_image_standardization.h"

#include <cmath>

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace ml_adj {
namespace per_image_standardization {
namespace {

using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;

inline void PerImageStandardization(dim_t batches, dim_t height, dim_t width,
                                    dim_t num_channels, const float* input_data,
                                    float* output_data) {
  const dim_t num_pixels_per_image = height * width * num_channels;

  const float inv_num_pixels_per_image = 1.0f / num_pixels_per_image;

  for (ind_t b = 0; b < batches; ++b) {
    const dim_t offset = b * num_pixels_per_image;
    const float* input_ptr = input_data + offset;
    float* output_ptr = output_data + offset;

    // Compute mean of all image pixels.
    float mean = 0.0f;
    for (ind_t i = 0; i < num_pixels_per_image; ++i) {
      mean += input_ptr[i];
    }
    mean *= inv_num_pixels_per_image;

    // Compute adjusted_stddev = max(stddev, 1.0/sqrt(N)) which is capped away
    // from 0 to protect against division by 0 when handling uniform images.
    // Reusing computed (x - mean) and saving it to the output here for speed
    // optimization.
    float variance = 0.0f;
    for (ind_t i = 0; i < num_pixels_per_image; ++i) {
      const float diff = input_ptr[i] - mean;
      variance += diff * diff * inv_num_pixels_per_image;
      output_ptr[i] = diff;
    }

    // Divide output by adjusted stddev.
    const float inv_adjusted_stddev =
        fmin(num_pixels_per_image, 1.0f / sqrt(variance));
    for (ind_t i = 0; i < num_pixels_per_image; ++i) {
      output_ptr[i] *= inv_adjusted_stddev;
    }
  }
}

// Linearly scales each image in input to have mean 0 and variance 1. Works for
// float datatype.
void ComputePerImageStandardization(const InputPack& inputs,
                                    const OutputPack& outputs) {
  TFLITE_DCHECK(inputs.size() == 1);
  TFLITE_DCHECK(outputs.size() == 1);

  // Extract input image data.
  const DataRef* img = inputs[0];
  const float* img_data = reinterpret_cast<const float*>(img->Data());
  const dim_t img_num_batches = img->Dims()[0];
  const dim_t img_height = img->Dims()[1];
  const dim_t img_width = img->Dims()[2];
  const dim_t img_num_channels = img->Dims()[3];

  // Resize output buffer for resized image.
  MutableDataRef* output = outputs[0];
  output->Resize({img_num_batches, img_height, img_width, img_num_channels});
  float* output_data = reinterpret_cast<float*>(output->Data());

  PerImageStandardization(img_num_batches, img_height, img_width,
                          img_num_channels, img_data, output_data);
}

}  // namespace

const Algo* Impl_PerImageStandardization() {
  static const Algo per_image_standardization = {
      &ComputePerImageStandardization, nullptr};
  return &per_image_standardization;
}

}  // namespace per_image_standardization
}  // namespace ml_adj
