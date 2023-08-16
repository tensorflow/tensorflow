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
#include <cstring>

#include "tensorflow/lite/experimental/ml_adjacent/lib.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"

namespace ml_adj {
namespace crop {
namespace {

using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;
using ::ml_adj::data::TypeWidth;

// Crop given input from the center. Works on any datatype.
void ComputeCenterCrop(const InputPack& inputs, const OutputPack& outputs) {
#ifndef NDEBUG
  TFLITE_CHECK(inputs.size() == 2);
  TFLITE_CHECK(outputs.size() == 1);
#endif

  const DataRef* img = inputs[0];

  const DataRef* frac = inputs[1];
  const double frac_data = *reinterpret_cast<const double*>(frac->Data());

  // Compute output height.
  const dim_t in_height = img->Dims()[1];
  const dim_t out_height_offset = (in_height - in_height * frac_data) / 2;
  const dim_t out_height = in_height - (2 * out_height_offset);

  // Compute output width.
  const dim_t in_width = img->Dims()[2];
  const dim_t out_width_offset = (in_width - in_width * frac_data) / 2;
  const dim_t out_width = in_width - (2 * out_width_offset);

  // Resize output buffer.
  MutableDataRef* output = outputs[0];
  output->Resize({img->Dims()[0], out_height, out_width, img->Dims()[3]});

  // Write rectangle window from input to output for each batch.
  const dim_t num_channels = img->Dims()[3];
  const dim_t chunk = TypeWidth(img->Type()) * num_channels;
  const dim_t in_img_size = in_height * in_width;

  const dim_t out_img_size = out_height * out_width;

  for (int b = 0; b < img->Dims()[0]; ++b) {
    for (int i = 0; i < out_height; ++i) {
      const dim_t read_byte_ofs =
          (in_img_size * b + (i + out_height_offset) * in_width +
           out_width_offset) *
          chunk;

      const void* read_start_addr =
          reinterpret_cast<const char*>(img->Data()) + read_byte_ofs;

      const dim_t write_byte_ofs = chunk * (out_img_size * b + i * out_width);

      void* write_addr =
          reinterpret_cast<char*>(output->Data()) + write_byte_ofs;

      // Copy slice of each input row by row.
      std::memcpy(write_addr, read_start_addr, chunk * out_width);
    }
  }
}

}  // namespace

const Algo* Impl_CenterCrop() {
  static const Algo center_crop = {&ComputeCenterCrop, nullptr};
  return &center_crop;
}

}  // namespace crop
}  // namespace ml_adj
