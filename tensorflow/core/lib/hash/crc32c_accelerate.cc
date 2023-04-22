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

#include <stddef.h>
#include <stdint.h>

// SSE4.2 accelerated CRC32c.

// See if the SSE4.2 crc32c instruction is available.
#undef USE_SSE_CRC32C
#ifdef __SSE4_2__
#if defined(__x86_64__) && defined(__GNUC__) && \
    (__GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8))
#define USE_SSE_CRC32C 1
#elif defined(__x86_64__) && defined(__clang__)
#if __has_builtin(__builtin_cpu_supports)
#define USE_SSE_CRC32C 1
#endif
#endif
#endif /* __SSE4_2__ */

// This version of Apple clang has a bug:
// https://llvm.org/bugs/show_bug.cgi?id=25510
#if defined(__APPLE__) && (__clang_major__ <= 8)
#undef USE_SSE_CRC32C
#endif

#ifdef USE_SSE_CRC32C
#include <nmmintrin.h>
#endif

namespace tensorflow {
namespace crc32c {

#ifndef USE_SSE_CRC32C

bool CanAccelerate() { return false; }
uint32_t AcceleratedExtend(uint32_t crc, const char *buf, size_t size) {
  // Should not be called.
  return 0;
}

#else

// SSE4.2 optimized crc32c computation.
bool CanAccelerate() { return __builtin_cpu_supports("sse4.2"); }

uint32_t AcceleratedExtend(uint32_t crc, const char *buf, size_t size) {
  const uint8_t *p = reinterpret_cast<const uint8_t *>(buf);
  const uint8_t *e = p + size;
  uint32_t l = crc ^ 0xffffffffu;

  // Advance p until aligned to 8-bytes..
  // Point x at first 7-byte aligned byte in string.  This might be
  // just past the end of the string.
  const uintptr_t pval = reinterpret_cast<uintptr_t>(p);
  const uint8_t *x = reinterpret_cast<const uint8_t *>(((pval + 7) >> 3) << 3);
  if (x <= e) {
    // Process bytes until finished or p is 8-byte aligned
    while (p != x) {
      l = _mm_crc32_u8(l, *p);
      p++;
    }
  }

  // Process bytes 16 at a time
  uint64_t l64 = l;
  while ((e - p) >= 16) {
    l64 = _mm_crc32_u64(l64, *reinterpret_cast<const uint64_t *>(p));
    l64 = _mm_crc32_u64(l64, *reinterpret_cast<const uint64_t *>(p + 8));
    p += 16;
  }

  // Process remaining bytes one at a time.
  l = l64;
  while (p < e) {
    l = _mm_crc32_u8(l, *p);
    p++;
  }

  return l ^ 0xffffffffu;
}

#endif

}  // namespace crc32c
}  // namespace tensorflow
