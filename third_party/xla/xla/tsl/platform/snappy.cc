/* Copyright 2025 The OpenXLA Authors.

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

#include "tsl/platform/snappy.h"

#include <cstddef>
#include <string>
#include <tuple>

#ifdef TF_USE_SNAPPY
#include "xla/tsl/lib/strings/resize_uninitialized.h"
#include "snappy.h"
#endif

namespace tsl {
namespace port {

#ifdef TF_USE_SNAPPY
namespace {
template <typename RT, typename... ATs>
struct FunctionTraits;

template <typename RT, typename... ATs>
struct FunctionTraits<RT(ATs...)> {
  using ReturnType = RT;
  using ArgTypes = std::tuple<ATs...>;
};

using SnappyIovecPtrType = std::tuple_element_t<
    0, FunctionTraits<decltype(snappy::RawCompressFromIOVec)>::ArgTypes>;
}  // namespace

bool Snappy_Compress(const char* input, size_t input_length,
                     std::string* output) {
  STLStringResizeUninitialized(output,
                               snappy::MaxCompressedLength(input_length));
  size_t outlen;
  snappy::RawCompress(input, input_length, &(*output)[0], &outlen);
  output->resize(outlen);
  return true;
}

bool Snappy_CompressFromIOVec(const struct iovec* iov,
                              size_t uncompressed_length, std::string* output) {
  STLStringResizeUninitialized(
      output, snappy::MaxCompressedLength(uncompressed_length));
  size_t outlen;
  auto* snappy_iov = reinterpret_cast<SnappyIovecPtrType>(iov);
  snappy::RawCompressFromIOVec(snappy_iov, uncompressed_length, &(*output)[0],
                               &outlen);
  output->resize(outlen);
  return true;
}

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result) {
  return snappy::GetUncompressedLength(input, length, result);
}

bool Snappy_Uncompress(const char* input, size_t length, char* output) {
  return snappy::RawUncompress(input, length, output);
}

bool Snappy_UncompressToIOVec(const char* compressed, size_t compressed_length,
                              const struct iovec* iov, size_t iov_cnt) {
  auto* snappy_iov = reinterpret_cast<SnappyIovecPtrType>(iov);
  return snappy::RawUncompressToIOVec(compressed, compressed_length, snappy_iov,
                                      iov_cnt);
}
#else
bool Snappy_Compress(const char* input, size_t input_length,
                     std::string* output) {
  return false;
}

bool Snappy_CompressFromIOVec(const struct iovec* iov,
                              size_t uncompressed_length, std::string* output) {
  return false;
}

bool Snappy_GetUncompressedLength(const char* input, size_t length,
                                  size_t* result) {
  return false;
}

bool Snappy_Uncompress(const char* input, size_t length, char* output) {
  return false;
}

bool Snappy_UncompressToIOVec(const char* compressed, size_t compressed_length,
                              const struct iovec* iov, size_t iov_cnt) {
  return false;
}
#endif

}  // namespace port
}  // namespace tsl
