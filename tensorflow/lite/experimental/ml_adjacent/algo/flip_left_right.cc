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
namespace flip_left_right {
namespace {

using ::ml_adj::algo::Algo;
using ::ml_adj::algo::InputPack;
using ::ml_adj::algo::OutputPack;
using ::ml_adj::data::DataRef;
using ::ml_adj::data::MutableDataRef;
using ::ml_adj::data::TypeWidth;

// TODO(vdziuba): Move flip functionality to header.
void FlipLeftRight(dim_t batches, dim_t input_height, dim_t input_width,
                   const char* input_data, char* output_data,
                   dim_t chunk_size) {
  const dim_t row_stride = input_width * chunk_size;
  const dim_t batch_stride = row_stride * input_height;

  // Iterate over batches to flip multi-channel image.
  for (int b = 0; b < batches; ++b) {
    const char* src_data_prt = input_data + b * batch_stride;
    char* dst_data_prt = output_data + b * batch_stride;

    for (int y = 0; y < input_height; ++y) {
      const char* src_ptr_row =
          src_data_prt + y * row_stride + (input_width - 1) * chunk_size;
      char* dst_ptr_row = dst_data_prt + y * row_stride;
      for (int x = 0; x < input_width; ++x) {
        std::memcpy(dst_ptr_row, src_ptr_row, chunk_size);

        src_ptr_row -= chunk_size;
        dst_ptr_row += chunk_size;
      }
    }
  }
}

// Flips the given input horizontally. Supports any datatype.
void ComputeFlipLeftRight(const InputPack& inputs, const OutputPack& outputs) {
  TFLITE_DCHECK(inputs.size() == 1);
  TFLITE_DCHECK(outputs.size() == 1);

  // Extract input image data.
  const DataRef* img = inputs[0];
  const char* img_data = reinterpret_cast<const char*>(img->Data());
  const dim_t num_batches = img->Dims()[0];
  const dim_t height = img->Dims()[1];
  const dim_t width = img->Dims()[2];
  const dim_t num_channels = img->Dims()[3];
  const dim_t chunk_size = TypeWidth(img->Type()) * num_channels;

  // Resize output buffer.
  MutableDataRef* output = outputs[0];
  output->Resize({num_batches, height, width, num_channels});
  char* output_data = reinterpret_cast<char*>(output->Data());

  FlipLeftRight(num_batches, height, width, img_data, output_data, chunk_size);
}

}  // namespace

const Algo* Impl_FlipLeftRight() {
  static const Algo flip_left_right = {&ComputeFlipLeftRight, nullptr};
  return &flip_left_right;
}

}  // namespace flip_left_right
}  // namespace ml_adj
