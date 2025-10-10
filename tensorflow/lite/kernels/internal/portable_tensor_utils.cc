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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_

#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#include "tensorflow/lite/core/c/builtin_op_data.h"

#if defined(_MSC_VER)
#define __restrict__ __restrict
#endif

namespace tflite {

// Not all backends support CpuBackendContext usage, so forward declare to avoid
// pulling in its implementation. Use of CpuBackendContext in method
// implementations is purely optional.
class CpuBackendContext;

namespace tensor_utils {

// Apply Rectified Linear to elements of a vector.
void ApplyReluToVector(const float* __restrict__ vector, int v_size,
                       float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(0.0f, vector[v]);
  }
}

// Apply Rectified Linear 1 (cap to [-1;1]) to elements of a vector
void ApplyRelu1ToVector(const float* __restrict__ vector, int v_size,
                        float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(-1.0f, std::min(vector[v], 1.0f));
  }
}

// Apply Rectified Linear 6 (cap to [0;6]) to elements of a vector
void ApplyRelu6ToVector(const float* __restrict__ vector, int v_size,
                        float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::max(0.0f, std::min(vector[v], 6.0f));
  }
}

// Apply signbit to elements of a vector
void ApplySignbitToVector(const float* __restrict__ vector, int v_size,
                          float* __restrict__ result) {
  for (int v = 0; v < v_size; v++) {
    result[v] = std::signbit(vector[v]);
  }
}

void UnpackDenseInt4IntoInt8(const int8_t* src_buffer, int num_elements,
                             int8_t* dst_buffer) {
  // num_elements means the number of elements regardless of packed or unpacked.
  // For example, 3 elements means both
  //   1) Packed: 3 int4's = 12 bit -> 16 bits (padded) = 2 bytes.
  //      stored in src_buffer[0] and src_buffer[1] (i = 0..1)
  //   2) Unpacked: 3 int8's = 3 bytes.
  //.     stored in dst_buffer[0], dst_buffer[1] and dst_buffer[2] (j = 0..2)
  for (int i = 0; i < num_elements / 2; i++) {
    int8_t byte = src_buffer[i];
    // Shift left first so that sign is properly extended when shifted right
    int8_t lower = static_cast<int8_t>(byte << 4) >> 4;
    int8_t higher = byte >> 4;
    dst_buffer[2 * i] = lower;
    dst_buffer[2 * i + 1] = higher;
  }

  // If the buffer size is odd, extract the final lower nibble.
  if (num_elements % 2 != 0) {
    dst_buffer[num_elements - 1] =
        static_cast<int8_t>(src_buffer[num_elements / 2] << 4) >> 4;
  }
}

void UnpackPackedIntToInt8(const int8_t* src_buffer, int num_elements,
                           int bit_width, int8_t* dst_buffer) {
  assert(bit_width == 2 || bit_width == 4);
  if (bit_width == 4) {
    // num_elements means the number of elements regardless of packed or
    // unpacked. For example, 3 elements means both
    //   1) Packed: 3 int4's = 12 bit -> 16 bits (padded) = 2 bytes.
    //      stored in src_buffer[0] and src_buffer[1] (i = 0..1)
    //   2) Unpacked: 3 int8's = 3 bytes.
    //.     stored in dst_buffer[0], dst_buffer[1] and dst_buffer[2] (j = 0..2)
    for (int i = 0; i < num_elements / 2; i++) {
      int8_t byte = src_buffer[i];
      // Shift left first so that sign is properly extended when shifted right
      int8_t lower = static_cast<int8_t>(byte << 4) >> 4;
      int8_t higher = byte >> 4;
      dst_buffer[2 * i] = lower;
      dst_buffer[2 * i + 1] = higher;
    }

    // If the buffer size is odd, extract the final lower nibble.
    if (num_elements % 2 != 0) {
      dst_buffer[num_elements - 1] =
          static_cast<int8_t>(src_buffer[num_elements / 2] << 4) >> 4;
    }
  } else if (bit_width == 2) {
    for (int i = 0; i < num_elements / 4; i++) {
      int8_t byte = src_buffer[i];
      // Shift left first so that sign is properly extended when shifted right
      int8_t val1 = static_cast<int8_t>(byte << 6) >> 6;
      int8_t val2 = static_cast<int8_t>((byte << 4) & 0xFF) >> 6;
      int8_t val3 = static_cast<int8_t>((byte << 2) & 0xFF) >> 6;
      int8_t val4 = byte >> 6;
      dst_buffer[4 * i] = val1;
      dst_buffer[4 * i + 1] = val2;
      dst_buffer[4 * i + 2] = val3;
      dst_buffer[4 * i + 3] = val4;
    }

    // Handle the remaining elements.
    int remaining_elements = num_elements % 4;
    if (remaining_elements > 0) {
      int8_t byte = src_buffer[num_elements / 4];
      for (int i = 0; i < remaining_elements; i++) {
        dst_buffer[num_elements - remaining_elements + i] =
            static_cast<int8_t>((byte << (6 - 2 * i)) & 0xFF) >> 6;
      }
    }
  }
}

void PackInt8IntoDenseInt4(const int8_t* src_buffer, int num_elements,
                           int8_t* dst_buffer) {
  // num_elements means the number of elements regardless of packed or unpacked.
  // For example, 3 elements means both
  //   1) Packed: 3 int4's = 12 bit -> 16 bits (padded) = 2 bytes.
  //      stored in src_buffer[0] and src_buffer[1] (i = 0..1)
  //   2) Unpacked: 3 int8's = 3 bytes.
  //      stored in dst_buffer[0], dst_buffer[1] and dst_buffer[2] (j = 0..2)
  for (int i = 0; i < num_elements - 1; i += 2) {
    dst_buffer[i / 2] = src_buffer[i] & 0x0F;
    dst_buffer[i / 2] |= src_buffer[i + 1] << 4;
  }
  auto packed_size = (num_elements + 1) / 2;

  // Copy the final nibble if the buffer is odd-lengthed
  if (num_elements % 2 != 0) {
    dst_buffer[packed_size - 1] = src_buffer[num_elements - 1] & 0x0F;
  }
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_TENSOR_UTILS_H_
