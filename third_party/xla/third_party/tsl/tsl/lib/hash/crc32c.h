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

#ifndef TENSORFLOW_TSL_LIB_HASH_CRC32C_H_
#define TENSORFLOW_TSL_LIB_HASH_CRC32C_H_

#include <stddef.h>

#include "absl/crc/crc32c.h"
#include "absl/strings/string_view.h"
#include "tsl/platform/cord.h"
#include "tsl/platform/platform.h"
#include "tsl/platform/types.h"

namespace tsl {
namespace crc32c {

// Return the crc32c of concat(A, buf[0,size-1]) where init_crc is the
// crc32c of some string A.  Extend() is often used to maintain the
// crc32c of a stream of data.
inline uint32 Extend(uint32 init_crc, const char* buf, size_t size) {
  return static_cast<uint32>(absl::ExtendCrc32c(
      static_cast<absl::crc32c_t>(init_crc), absl::string_view(buf, size)));
}

#if defined(TF_CORD_SUPPORT)
extern uint32 Extend(uint32 init_crc, const absl::Cord& cord);
#endif

// Return the crc32c of data[0,n-1]
inline uint32 Value(const char* data, size_t n) { return Extend(0, data, n); }

#if defined(TF_CORD_SUPPORT)
inline uint32 Value(const absl::Cord& cord) { return Extend(0, cord); }
#endif

static const uint32 kMaskDelta = 0xa282ead8ul;

// Return a masked representation of crc.
//
// Motivation: it is problematic to compute the CRC of a string that
// contains embedded CRCs.  Therefore we recommend that CRCs stored
// somewhere (e.g., in files) should be masked before being stored.
inline uint32 Mask(uint32 crc) {
  // Rotate right by 15 bits and add a constant.
  return ((crc >> 15) | (crc << 17)) + kMaskDelta;
}

// Return the crc whose masked representation is masked_crc.
inline uint32 Unmask(uint32 masked_crc) {
  uint32 rot = masked_crc - kMaskDelta;
  return ((rot >> 17) | (rot << 15));
}

}  // namespace crc32c
}  // namespace tsl

#endif  // TENSORFLOW_TSL_LIB_HASH_CRC32C_H_
