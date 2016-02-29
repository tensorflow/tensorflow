/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/lib/hash/hash.h"

#include "tensorflow/core/lib/core/raw_coding.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

#include <string.h>

namespace tensorflow {

// 0xff is in case char is signed.
static inline uint32 ByteAs32(char c) { return static_cast<uint32>(c) & 0xff; }
static inline uint64 ByteAs64(char c) { return static_cast<uint64>(c) & 0xff; }

uint32 Hash32(const char* data, size_t n, uint32 seed) {
  // 'm' and 'r' are mixing constants generated offline.
  // They're not really 'magic', they just happen to work well.

  const uint32 m = 0x5bd1e995;
  const int r = 24;

  // Initialize the hash to a 'random' value
  uint32 h = seed ^ n;

  // Mix 4 bytes at a time into the hash
  while (n >= 4) {
    uint32 k = core::DecodeFixed32(data);

    k *= m;
    k ^= k >> r;
    k *= m;

    h *= m;
    h ^= k;

    data += 4;
    n -= 4;
  }

  // Handle the last few bytes of the input array

  switch (n) {
    case 3:
      h ^= ByteAs32(data[2]) << 16;
      TF_FALLTHROUGH_INTENDED;
    case 2:
      h ^= ByteAs32(data[1]) << 8;
      TF_FALLTHROUGH_INTENDED;
    case 1:
      h ^= ByteAs32(data[0]);
      h *= m;
  }

  // Do a few final mixes of the hash to ensure the last few
  // bytes are well-incorporated.

  h ^= h >> 13;
  h *= m;
  h ^= h >> 15;

  return h;
}

uint64 Hash64(const char* data, size_t n, uint64 seed) {
  const uint64 m = 0xc6a4a7935bd1e995;
  const int r = 47;

  uint64 h = seed ^ (n * m);

  while (n >= 8) {
    uint64 k = core::DecodeFixed64(data);
    data += 8;
    n -= 8;

    k *= m;
    k ^= k >> r;
    k *= m;

    h ^= k;
    h *= m;
  }

  switch (n) {
    case 7:
      h ^= ByteAs64(data[6]) << 48;
      TF_FALLTHROUGH_INTENDED;
    case 6:
      h ^= ByteAs64(data[5]) << 40;
      TF_FALLTHROUGH_INTENDED;
    case 5:
      h ^= ByteAs64(data[4]) << 32;
      TF_FALLTHROUGH_INTENDED;
    case 4:
      h ^= ByteAs64(data[3]) << 24;
      TF_FALLTHROUGH_INTENDED;
    case 3:
      h ^= ByteAs64(data[2]) << 16;
      TF_FALLTHROUGH_INTENDED;
    case 2:
      h ^= ByteAs64(data[1]) << 8;
      TF_FALLTHROUGH_INTENDED;
    case 1:
      h ^= ByteAs64(data[0]);
      h *= m;
  }

  h ^= h >> r;
  h *= m;
  h ^= h >> r;

  return h;
}

}  // namespace tensorflow
