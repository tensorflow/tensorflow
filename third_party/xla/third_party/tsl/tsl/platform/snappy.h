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

#ifndef TENSORFLOW_TSL_PLATFORM_SNAPPY_H_
#define TENSORFLOW_TSL_PLATFORM_SNAPPY_H_

#include "tsl/platform/types.h"

#if !defined(PLATFORM_WINDOWS)
#include <sys/uio.h>
namespace tsl {
using ::iovec;  // NOLINT(misc-unused-using-decls)
}  // namespace tsl
#else
namespace tsl {
struct iovec {
  void* iov_base;
  size_t iov_len;
};
}  // namespace tsl
#endif

namespace tsl {
namespace port {

// Snappy compression/decompression support
bool Snappy_Compress(const char* input, size_t length, string* output);

bool Snappy_CompressFromIOVec(const struct iovec* iov,
                              size_t uncompressed_length, string* output);

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result);
bool Snappy_Uncompress(const char* input, size_t length, char* output);

bool Snappy_UncompressToIOVec(const char* compressed, size_t compressed_length,
                              const struct iovec* iov, size_t iov_cnt);

}  // namespace port
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_SNAPPY_H_
