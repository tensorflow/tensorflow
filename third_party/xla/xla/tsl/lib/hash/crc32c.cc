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

// A portable implementation of crc32c, optimized to handle
// four bytes at a time.

#include "xla/tsl/lib/hash/crc32c.h"

#include <stdint.h>

#include "absl/strings/cord.h"
#include "absl/strings/string_view.h"
#include "xla/tsl/platform/types.h"

namespace tsl {
namespace crc32c {

#if defined(TF_CORD_SUPPORT)
uint32 Extend(uint32 crc, const absl::Cord &cord) {
  for (absl::string_view fragment : cord.Chunks()) {
    crc = Extend(crc, fragment.data(), fragment.size());
  }
  return crc;
}
#endif

}  // namespace crc32c
}  // namespace tsl
