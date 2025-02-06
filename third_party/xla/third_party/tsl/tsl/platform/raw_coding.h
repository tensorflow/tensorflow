/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_RAW_CODING_H_
#define TENSORFLOW_TSL_PLATFORM_RAW_CODING_H_

#include <string.h>

#include "xla/tsl/platform/byte_order.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace core {

// Lower-level versions of Get... that read directly from a character buffer
// without any bounds checking.

inline uint16 DecodeFixed16(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint16 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    return ((static_cast<uint16>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint16>(static_cast<unsigned char>(ptr[1])) << 8));
  }
}

inline uint32 DecodeFixed32(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint32 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    return ((static_cast<uint32>(static_cast<unsigned char>(ptr[0]))) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[1])) << 8) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[2])) << 16) |
            (static_cast<uint32>(static_cast<unsigned char>(ptr[3])) << 24));
  }
}

inline uint64 DecodeFixed64(const char* ptr) {
  if (port::kLittleEndian) {
    // Load the raw bytes
    uint64 result;
    memcpy(&result, ptr, sizeof(result));  // gcc optimizes this to a plain load
    return result;
  } else {
    uint64 lo = DecodeFixed32(ptr);
    uint64 hi = DecodeFixed32(ptr + 4);
    return (hi << 32) | lo;
  }
}

}  // namespace core
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_RAW_CODING_H_
