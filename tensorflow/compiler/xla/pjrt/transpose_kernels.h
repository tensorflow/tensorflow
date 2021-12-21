/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_KERNELS_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_KERNELS_H_

#include <cstdint>

#include "third_party/eigen3/Eigen/Core"

namespace xla {

// Generic transpose kernel.
//
// All of the kernels that follow in this file are optimized versions of this
// generic kernel, specialized to particular block sizes and data types.
//
// The transpose kernel requires its input to be contiguous in one of the two
// dimensions being transposed, and the output to be contiguous in the other
// dimension.
//
// lda, ldb are strides in bytes.
template <typename T, int bs>
struct TransposeMicroKernel {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    for (int i = 0; i < bs; ++i) {
      for (int j = 0; j < bs; ++j) {
        *reinterpret_cast<T*>(b + i * ldb + j * sizeof(T)) =
            *reinterpret_cast<T const*>(a + j * lda + i * sizeof(T));
      }
    }
  }
};

// TODO(phawkins): it would be nice to remove the use of Eigen here, and instead
// allow for runtime dispatch of, say, AVX or AVX2 kernels where they are
// supported. On the other hand, using Eigen makes for easier cross-platform
// portability.
#ifdef EIGEN_VECTORIZE_AVX

template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    __m128i x = _mm_set_epi32(*reinterpret_cast<const uint32_t*>(a + lda * 0),
                              *reinterpret_cast<const uint32_t*>(a + lda * 1),
                              *reinterpret_cast<const uint32_t*>(a + lda * 2),
                              *reinterpret_cast<const uint32_t*>(a + lda * 3));
    __m128i mask =
        _mm_setr_epi8(12, 8, 4, 0, 13, 9, 5, 1, 14, 10, 6, 2, 15, 11, 7, 3);
    x = _mm_shuffle_epi8(x, mask);
    *reinterpret_cast<uint32_t*>(b + ldb * 0) = _mm_extract_epi32(x, 0);
    *reinterpret_cast<uint32_t*>(b + ldb * 1) = _mm_extract_epi32(x, 1);
    *reinterpret_cast<uint32_t*>(b + ldb * 2) = _mm_extract_epi32(x, 2);
    *reinterpret_cast<uint32_t*>(b + ldb * 3) = _mm_extract_epi32(x, 3);
  }
};

// TODO(phawkins): add an 8x8 byte transpose kernel.

// TODO(phawkins): Eigen doesn't have a SSE/AVX byte Packet16c type. Add one
// and call it here rather than using AVX intrinsics.
template <>
struct TransposeMicroKernel<uint8_t, /*bs=*/16> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    std::array<__m128i, 16> packet;
    for (int i = 0; i < 16; ++i) {
      packet[i] =
          _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + lda * i));
    }

    // If we number the elements in the input thus:
    // kernel.packet[ 0] = {00, 01, 02, 03, 04, 05, 06, 07, 08, 09, 0a, 0b, 0c,
    //                      0d, 0e, 0f}
    // kernel.packet[ 1] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 1a, 1b, 1c,
    //                      1d, 1e, 1f}
    // ...
    // kernel.packet[15] = {f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, fa, fb, fc,
    //                      fd, fe, ff},
    //
    // the desired output is:
    // kernel.packet[ 0] = {00, 10, 20, 30, 40, 50, 60, 70, 80, 90, a0, b0, c0,
    //                      d0, e0, f0}
    // kernel.packet[ 1] = {01, 11, 21, 31, 41, 51, 61, 71, 81, 91, a1, b1, c1,
    //                      d1, e1, f1}
    // ...
    // kernel.packet[15] = {0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f, 9f, af, bf, cf,
    //                      df, ef, ff},
    // 00 10 01 11 02 12 03 13 04 14 05 15 06 16 07 17
    __m128i t0 = _mm_unpacklo_epi8(packet[0], packet[1]);
    // 08 18 09 19 0a 1a 0b 1b 0c 1c 0d 1d 0e 1e 0f 1f
    __m128i t1 = _mm_unpackhi_epi8(packet[0], packet[1]);
    // 20 30 21 31 22 32 ...                     27 37
    __m128i t2 = _mm_unpacklo_epi8(packet[2], packet[3]);
    // 28 38 29 39 2a 3a ...                     2f 3f
    __m128i t3 = _mm_unpackhi_epi8(packet[2], packet[3]);
    // 40 50 41 51 42 52                         47 57
    __m128i t4 = _mm_unpacklo_epi8(packet[4], packet[5]);
    // 48 58 49 59 4a 5a
    __m128i t5 = _mm_unpackhi_epi8(packet[4], packet[5]);
    __m128i t6 = _mm_unpacklo_epi8(packet[6], packet[7]);
    __m128i t7 = _mm_unpackhi_epi8(packet[6], packet[7]);
    __m128i t8 = _mm_unpacklo_epi8(packet[8], packet[9]);
    __m128i t9 = _mm_unpackhi_epi8(packet[8], packet[9]);
    __m128i ta = _mm_unpacklo_epi8(packet[10], packet[11]);
    __m128i tb = _mm_unpackhi_epi8(packet[10], packet[11]);
    __m128i tc = _mm_unpacklo_epi8(packet[12], packet[13]);
    __m128i td = _mm_unpackhi_epi8(packet[12], packet[13]);
    __m128i te = _mm_unpacklo_epi8(packet[14], packet[15]);
    __m128i tf = _mm_unpackhi_epi8(packet[14], packet[15]);

    // 00 10 20 30 01 11 21 31 02 12 22 32 03 13 23 33
    __m128i s0 = _mm_unpacklo_epi16(t0, t2);
    __m128i s1 = _mm_unpackhi_epi16(t0, t2);  // 04 14 24 34
    __m128i s2 = _mm_unpacklo_epi16(t1, t3);  // 08 18 28 38 ...
    __m128i s3 = _mm_unpackhi_epi16(t1, t3);  // 0c 1c 2c 3c ...
    // 40 50 60 70 41 51 61 71 42 52 62 72 43 53 63 73
    __m128i s4 = _mm_unpacklo_epi16(t4, t6);
    __m128i s5 = _mm_unpackhi_epi16(t4, t6);  // 44 54 64 74 ...
    __m128i s6 = _mm_unpacklo_epi16(t5, t7);
    __m128i s7 = _mm_unpackhi_epi16(t5, t7);
    __m128i s8 = _mm_unpacklo_epi16(t8, ta);
    __m128i s9 = _mm_unpackhi_epi16(t8, ta);
    __m128i sa = _mm_unpacklo_epi16(t9, tb);
    __m128i sb = _mm_unpackhi_epi16(t9, tb);
    __m128i sc = _mm_unpacklo_epi16(tc, te);
    __m128i sd = _mm_unpackhi_epi16(tc, te);
    __m128i se = _mm_unpacklo_epi16(td, tf);
    __m128i sf = _mm_unpackhi_epi16(td, tf);

    // 00 10 20 30 40 50 60 70 01 11 21 31 41 51 61 71
    __m128i u0 = _mm_unpacklo_epi32(s0, s4);
    // 02 12 22 32 42 52 62 72 03 13 23 33 43 53 63 73
    __m128i u1 = _mm_unpackhi_epi32(s0, s4);
    __m128i u2 = _mm_unpacklo_epi32(s1, s5);
    __m128i u3 = _mm_unpackhi_epi32(s1, s5);
    __m128i u4 = _mm_unpacklo_epi32(s2, s6);
    __m128i u5 = _mm_unpackhi_epi32(s2, s6);
    __m128i u6 = _mm_unpacklo_epi32(s3, s7);
    __m128i u7 = _mm_unpackhi_epi32(s3, s7);
    __m128i u8 = _mm_unpacklo_epi32(s8, sc);
    __m128i u9 = _mm_unpackhi_epi32(s8, sc);
    __m128i ua = _mm_unpacklo_epi32(s9, sd);
    __m128i ub = _mm_unpackhi_epi32(s9, sd);
    __m128i uc = _mm_unpacklo_epi32(sa, se);
    __m128i ud = _mm_unpackhi_epi32(sa, se);
    __m128i ue = _mm_unpacklo_epi32(sb, sf);
    __m128i uf = _mm_unpackhi_epi32(sb, sf);

    packet[0] = _mm_unpacklo_epi64(u0, u8);
    packet[1] = _mm_unpackhi_epi64(u0, u8);
    packet[2] = _mm_unpacklo_epi64(u1, u9);
    packet[3] = _mm_unpackhi_epi64(u1, u9);
    packet[4] = _mm_unpacklo_epi64(u2, ua);
    packet[5] = _mm_unpackhi_epi64(u2, ua);
    packet[6] = _mm_unpacklo_epi64(u3, ub);
    packet[7] = _mm_unpackhi_epi64(u3, ub);
    packet[8] = _mm_unpacklo_epi64(u4, uc);
    packet[9] = _mm_unpackhi_epi64(u4, uc);
    packet[10] = _mm_unpacklo_epi64(u5, ud);
    packet[11] = _mm_unpackhi_epi64(u5, ud);
    packet[12] = _mm_unpacklo_epi64(u6, ue);
    packet[13] = _mm_unpackhi_epi64(u6, ue);
    packet[14] = _mm_unpacklo_epi64(u7, uf);
    packet[15] = _mm_unpackhi_epi64(u7, uf);
    for (int i = 0; i < 16; ++i) {
      _mm_storeu_si128(reinterpret_cast<__m128i*>(b + ldb * i), packet[i]);
    }
  }
};

// TODO(phawkins): add an 4x4 uint16_t transpose kernel.

template <>
struct TransposeMicroKernel<uint16_t, /*bs=*/8> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    using Eigen::internal::Packet8h;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 8;
    PacketBlock<Packet8h, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet8h>(
          reinterpret_cast<const Eigen::half*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<Eigen::half>(
          reinterpret_cast<Eigen::half*>(b + ldb * i), block.packet[i]);
    }
  }
};

template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    using Eigen::internal::Packet4f;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 4;
    PacketBlock<Packet4f, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet4f>(
          reinterpret_cast<const float*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<float>(reinterpret_cast<float*>(b + ldb * i),
                                      block.packet[i]);
    }
  }
};

template <>
struct TransposeMicroKernel<uint32_t, /*bs=*/8> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    using Eigen::internal::Packet8f;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 8;
    PacketBlock<Packet8f, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet8f>(
          reinterpret_cast<const float*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<float>(reinterpret_cast<float*>(b + ldb * i),
                                      block.packet[i]);
    }
  }
};

template <>
struct TransposeMicroKernel<uint64_t, /*bs=*/2> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    using Eigen::internal::Packet2d;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 2;
    PacketBlock<Packet2d, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet2d>(
          reinterpret_cast<const double*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<double>(reinterpret_cast<double*>(b + ldb * i),
                                       block.packet[i]);
    }
  }
};

template <>
struct TransposeMicroKernel<uint64_t, /*bs=*/4> {
  static void Apply(const char* __restrict a, int64_t lda, char* __restrict b,
                    int64_t ldb) {
    using Eigen::internal::Packet4d;
    using Eigen::internal::PacketBlock;
    constexpr int bs = 4;
    PacketBlock<Packet4d, bs> block;
    for (int i = 0; i < bs; ++i) {
      block.packet[i] = Eigen::internal::ploadu<Packet4d>(
          reinterpret_cast<const double*>(a + lda * i));
    }
    Eigen::internal::ptranspose(block);
    for (int i = 0; i < bs; ++i) {
      Eigen::internal::pstoreu<double>(reinterpret_cast<double*>(b + ldb * i),
                                       block.packet[i]);
    }
  }
};

#endif  // EIGEN_VECTORIZE_AVX

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_TRANSPOSE_KERNELS_H_
