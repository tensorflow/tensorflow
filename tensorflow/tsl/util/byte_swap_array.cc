/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/tsl/util/byte_swap_array.h"

#include "tensorflow/tsl/platform/errors.h"

namespace tsl {

Status ByteSwapArray(char* array, size_t bytes_per_elem, int array_len) {
  if (bytes_per_elem == 1) {
    // No-op
    return OkStatus();
  } else if (bytes_per_elem == 2) {
    auto array_16 = reinterpret_cast<uint16_t*>(array);
    for (int i = 0; i < array_len; i++) {
      array_16[i] = BYTE_SWAP_16(array_16[i]);
    }
    return OkStatus();
  } else if (bytes_per_elem == 4) {
    auto array_32 = reinterpret_cast<uint32_t*>(array);
    for (int i = 0; i < array_len; i++) {
      array_32[i] = BYTE_SWAP_32(array_32[i]);
    }
    return OkStatus();
  } else if (bytes_per_elem == 8) {
    auto array_64 = reinterpret_cast<uint64_t*>(array);
    for (int i = 0; i < array_len; i++) {
      array_64[i] = BYTE_SWAP_64(array_64[i]);
    }
    return OkStatus();
  } else {
    return errors::Unimplemented("Byte-swapping of ", bytes_per_elem,
                                 "-byte values not supported.");
  }
}

}  // namespace tsl
