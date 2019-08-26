/* Copyright 2019 Google LLC. All Rights Reserved.

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

#include "tensorflow/lite/experimental/ruy/detect_x86.h"

#include <cstdint>

#if RUY_PLATFORM(X86) && RUY_PLATFORM(X86_ENHANCEMENTS)
#include <immintrin.h>  // IWYU pragma: keep

#endif

namespace ruy {
#if RUY_PLATFORM(X86) && RUY_PLATFORM(X86_ENHANCEMENTS)

namespace {

// See Intel docs, such as http://goo.gl/c6IkGX.
inline void RunCpuid(std::uint32_t eax, std::uint32_t ecx,
                     std::uint32_t abcd[4]) {
  std::uint32_t ebx, edx;
#if defined(__i386__) && defined(__PIC__)
  /* in case of PIC under 32-bit EBX cannot be clobbered */
  asm volatile("movl %%ebx, %%edi \n\t cpuid \n\t xchgl %%ebx, %%edi"
               : "=D"(ebx),
#else
  asm volatile("cpuid"
               : "+b"(ebx),
#endif
                 "+a"(eax), "+c"(ecx), "=d"(edx));
  abcd[0] = eax;
  abcd[1] = ebx;
  abcd[2] = ecx;
  abcd[3] = edx;
}

}  // namespace

bool DetectCpuSse42() {
  constexpr std::uint32_t kAvx512EcxSse42 = 1u << 20;
  constexpr std::uint32_t kAvx512EcxAbm = 1u << 5;

  std::uint32_t abcd[4];

  RunCpuid(1, 0, abcd);
  const bool has_sse4_2_base = (abcd[2] & kAvx512EcxSse42) == kAvx512EcxSse42;
  RunCpuid(0x80000001, 0, abcd);
  const bool has_abm = (abcd[2] & kAvx512EcxAbm) == kAvx512EcxAbm;

  return has_sse4_2_base && has_abm;
}

bool DetectCpuAvx2() {
  constexpr std::uint32_t kAvx2Ebx = 1u << 5;

  std::uint32_t abcd[4];
  RunCpuid(7, 0, abcd);

  return (abcd[1] & kAvx2Ebx) == kAvx2Ebx;
}

bool DetectCpuAvx512() {
  constexpr std::uint32_t kAvx512EbxF = 1u << 16;
  constexpr std::uint32_t kAvx512EbxDq = 1u << 17;
  constexpr std::uint32_t kAvx512EbxCd = 1u << 28;
  constexpr std::uint32_t kAvx512EbxBw = 1u << 30;
  constexpr std::uint32_t kAvx512EbxVl = 1u << 31;

  constexpr std::uint32_t kAvx512EbxMask =
      kAvx512EbxF | kAvx512EbxDq | kAvx512EbxCd | kAvx512EbxBw | kAvx512EbxVl;
  std::uint32_t abcd[4];
  RunCpuid(7, 0, abcd);

  return (abcd[1] & kAvx512EbxMask) == kAvx512EbxMask;
}

#endif
}  // namespace ruy
