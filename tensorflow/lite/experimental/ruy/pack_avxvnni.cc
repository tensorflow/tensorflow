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

#include <cstdint>
#include <cstring>

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/pack.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"

#if RUY_PLATFORM(AVX_VNNI) && RUY_OPT_ENABLED(RUY_OPT_INTRINSICS)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM(AVX_VNNI) && RUY_OPT_ENABLED(RUY_OPT_ASM))

void Pack8bitAvxVnni(const std::int8_t* src_ptr, std::int8_t input_xor,
                     const std::int8_t* zerobuf, int src_stride,
                     int remaining_src_cols, int src_rows,
                     std::int8_t* packed_ptr, std::int32_t* sums_ptr) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void PackFloatAvxVnni(const float* src_ptr, const float* zerobuf,
                      int src_stride, int remaining_src_cols, int src_rows,
                      float* packed_ptr) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM(AVX_VNNI) && RUY_OPT_ENABLED(RUY_OPT_ASM)

// The first int8_t template parameter is arbitrary: this routine is common to
// all 8-bit source matrix types.
using PackImpl8bitAvxVnni =
    PackImpl<Path::kAvxVnni, FixedKernelLayout<Order::kColMajor, 4, 16>,
             std::int8_t, std::int8_t, std::int32_t>;

namespace {

inline void ZeroHalf8bitAvxVnni(int src_rows, std::int8_t packed_zero_point,
                                std::int8_t* packed_ptr) {
  const int non_trailing_blocks = (src_rows & ~31) >> 2;
  // This routine fills half blocks, and typically fills the second halves. Thus
  // packed_ptr is already offset by 8*4.
  for (int k = 0; k < non_trailing_blocks; ++k) {
    for (int j = 0; j < (8 * 4); ++j) {
      packed_ptr[16 * 4 * k + j] = packed_zero_point;
    }
  }
}

inline void HalfPack8bitAvxVnni(const std::int8_t* src_ptr,
                                std::int8_t input_xor,
                                const std::int8_t* zerobuf, int src_stride,
                                int remaining_src_cols, int src_rows,
                                std::int8_t* packed_ptr, std::int32_t* sums_ptr,
                                std::int8_t* trailing_buf) {
  std::int8_t in_data[8][8][4];

  const std::int8_t* src_ptr0 = src_ptr;
  const std::int8_t* src_ptr1 = src_ptr0 + src_stride;
  const std::int8_t* src_ptr2 = src_ptr1 + src_stride;
  const std::int8_t* src_ptr3 = src_ptr2 + src_stride;
  const std::int8_t* src_ptr4 = src_ptr3 + src_stride;
  const std::int8_t* src_ptr5 = src_ptr4 + src_stride;
  const std::int8_t* src_ptr6 = src_ptr5 + src_stride;
  const std::int8_t* src_ptr7 = src_ptr6 + src_stride;
  std::int64_t src_inc0 = 8 * 4;
  std::int64_t src_inc1 = 8 * 4;
  std::int64_t src_inc2 = 8 * 4;
  std::int64_t src_inc3 = 8 * 4;
  std::int64_t src_inc4 = 8 * 4;
  std::int64_t src_inc5 = 8 * 4;
  std::int64_t src_inc6 = 8 * 4;
  std::int64_t src_inc7 = 8 * 4;
  if (remaining_src_cols < 8) {
    if (remaining_src_cols <= 0) {
      src_ptr0 = zerobuf;
      src_inc0 = 0;
    }
    if (remaining_src_cols <= 1) {
      src_ptr1 = zerobuf;
      src_inc1 = 0;
    }
    if (remaining_src_cols <= 2) {
      src_ptr2 = zerobuf;
      src_inc2 = 0;
    }
    if (remaining_src_cols <= 3) {
      src_ptr3 = zerobuf;
      src_inc3 = 0;
    }
    if (remaining_src_cols <= 4) {
      src_ptr4 = zerobuf;
      src_inc4 = 0;
    }
    if (remaining_src_cols <= 5) {
      src_ptr5 = zerobuf;
      src_inc5 = 0;
    }
    if (remaining_src_cols <= 6) {
      src_ptr6 = zerobuf;
      src_inc6 = 0;
    }
    src_ptr7 = zerobuf;
    src_inc7 = 0;
  }

  const std::int8_t zero_point = zerobuf[0];

  if (sums_ptr) {
    for (int i = 0; i < 8; ++i) {
      sums_ptr[i] = 0;
    }
  }

  // The overall packing effectively pads the source rows to
  // (src_rows + 63) & ~63. The iteration over k may skip when m=1, and then we
  // only pack for (src_rows + 31) & ~31. When there is an incomplete
  // destination block, this is stored into trailing_buf instead of packed_ptr.
  for (int k = 0; k < src_rows; k += 16 * 4) {
    for (int m = 0; m < 2; ++m) {
      // Available source rows.
      // If this is less than 0 (for m=1), we skip, having filled trailing
      // buffer for m=0. Also, if source rows is zero on m=1, then we filled
      // exactly to the end of the column in the packed buffer.
      const int packed_rows = src_rows - k - 8 * m * 4;
      // Effectively,
      // packed_rows = std::max(0, std::min(8, src_rows - k - 8 * m));
      // but treat each case separately.
      if (packed_rows >= (8 * 4)) {
        for (int i = 0; i < 8; ++i) {
          for (int s = 0; s < 4; ++s) {
            in_data[0][i][s] = src_ptr0[i * 4 + s];
            in_data[1][i][s] = src_ptr1[i * 4 + s];
            in_data[2][i][s] = src_ptr2[i * 4 + s];
            in_data[3][i][s] = src_ptr3[i * 4 + s];
            in_data[4][i][s] = src_ptr4[i * 4 + s];
            in_data[5][i][s] = src_ptr5[i * 4 + s];
            in_data[6][i][s] = src_ptr6[i * 4 + s];
            in_data[7][i][s] = src_ptr7[i * 4 + s];
          }
        }
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 8; ++j) {
            for (int s = 0; s < 4; ++s) {
              packed_ptr[(16 * i + j) * 4 + s] =
                  static_cast<std::int8_t>(in_data[j][i][s] ^ input_xor);
            }
            if (sums_ptr) {
              for (int s = 0; s < 4; ++s) {
                sums_ptr[j] += in_data[j][i][s] ^ input_xor;
              }
            }
          }
        }
      } else if (packed_rows > 0) {
        RUY_DCHECK_LT(packed_rows >> 2, 8);
        int i = 0;
        for (; i < (packed_rows >> 2); ++i) {
          for (int s = 0; s < 4; ++s) {
            in_data[0][i][s] = src_ptr0[i * 4 + s];
            in_data[1][i][s] = src_ptr1[i * 4 + s];
            in_data[2][i][s] = src_ptr2[i * 4 + s];
            in_data[3][i][s] = src_ptr3[i * 4 + s];
            in_data[4][i][s] = src_ptr4[i * 4 + s];
            in_data[5][i][s] = src_ptr5[i * 4 + s];
            in_data[6][i][s] = src_ptr6[i * 4 + s];
            in_data[7][i][s] = src_ptr7[i * 4 + s];
          }
        }
        if (i < ((packed_rows + 3) >> 2)) {
          int s = 0;
          for (; s < (packed_rows & 3); ++s) {
            in_data[0][i][s] = src_ptr0[i * 4 + s];
            in_data[1][i][s] = src_ptr1[i * 4 + s];
            in_data[2][i][s] = src_ptr2[i * 4 + s];
            in_data[3][i][s] = src_ptr3[i * 4 + s];
            in_data[4][i][s] = src_ptr4[i * 4 + s];
            in_data[5][i][s] = src_ptr5[i * 4 + s];
            in_data[6][i][s] = src_ptr6[i * 4 + s];
            in_data[7][i][s] = src_ptr7[i * 4 + s];
          }
          RUY_DCHECK_LE(s, 4);
          for (; s < 4; ++s) {
            for (int j = 0; j < 8; ++j) {
              in_data[j][i][s] = zero_point;
            }
          }
          ++i;
        }
        // We do not care what goes into the trailing buffer, but we want
        // in_data[...] ^ input_xor == 0 for irrelevant values in the summation.
        //
        // It might prove better in optimized code to pad uniformly with
        // zero_point, and compensate by initializing the summations with the
        // compensating offset, effectively
        // ((input_xor - zero_point) ^ input_xor) *
        //                         4 * (8 - ((packed_rows + 3) >> 2)).
        for (; i < 8; ++i) {
          for (int s = 0; s < 4; ++s) {
            for (int j = 0; j < 8; ++j) {
              in_data[j][i][s] = input_xor;
            }
          }
        }
        // We loop through [0, 8) rather than [0, (packed_rows + 3) >> 2), since
        // that emulates what we might do in fully-optimized code.
        if (sums_ptr) {
          for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
              for (int s = 0; s < 4; ++s) {
                trailing_buf[(16 * i + j) * 4 + s] =
                    static_cast<std::int8_t>(in_data[j][i][s] ^ input_xor);
                sums_ptr[j] += in_data[j][i][s] ^ input_xor;
              }
            }
          }
        } else {
          for (int i = 0; i < 8; ++i) {
            for (int j = 0; j < 8; ++j) {
              for (int s = 0; s < 4; ++s) {
                trailing_buf[(16 * i + j) * 4 + s] =
                    static_cast<std::int8_t>(in_data[j][i][s] ^ input_xor);
              }
            }
          }
        }
      }

      packed_ptr += 16 * 8 * 4;
      src_ptr0 += src_inc0;
      src_ptr1 += src_inc1;
      src_ptr2 += src_inc2;
      src_ptr3 += src_inc3;
      src_ptr4 += src_inc4;
      src_ptr5 += src_inc5;
      src_ptr6 += src_inc6;
      src_ptr7 += src_inc7;
    }
  }
}

inline void HalfPackFloatAvxVnni(const float* src_ptr, const float* zerobuf,
                                 int src_stride, int remaining_src_cols,
                                 int src_rows, float* packed_ptr,
                                 float* trailing_buf) {
  float in_data[8][8];

  const float* src_ptr0 = src_ptr;
  const float* src_ptr1 = src_ptr0 + src_stride;
  const float* src_ptr2 = src_ptr1 + src_stride;
  const float* src_ptr3 = src_ptr2 + src_stride;
  const float* src_ptr4 = src_ptr3 + src_stride;
  const float* src_ptr5 = src_ptr4 + src_stride;
  const float* src_ptr6 = src_ptr5 + src_stride;
  const float* src_ptr7 = src_ptr6 + src_stride;
  std::int64_t src_inc0 = 8;
  std::int64_t src_inc1 = 8;
  std::int64_t src_inc2 = 8;
  std::int64_t src_inc3 = 8;
  std::int64_t src_inc4 = 8;
  std::int64_t src_inc5 = 8;
  std::int64_t src_inc6 = 8;
  std::int64_t src_inc7 = 8;
  if (remaining_src_cols < 8) {
    if (remaining_src_cols <= 0) {
      src_ptr0 = zerobuf;
      src_inc0 = 0;
    }
    if (remaining_src_cols <= 1) {
      src_ptr1 = zerobuf;
      src_inc1 = 0;
    }
    if (remaining_src_cols <= 2) {
      src_ptr2 = zerobuf;
      src_inc2 = 0;
    }
    if (remaining_src_cols <= 3) {
      src_ptr3 = zerobuf;
      src_inc3 = 0;
    }
    if (remaining_src_cols <= 4) {
      src_ptr4 = zerobuf;
      src_inc4 = 0;
    }
    if (remaining_src_cols <= 5) {
      src_ptr5 = zerobuf;
      src_inc5 = 0;
    }
    if (remaining_src_cols <= 6) {
      src_ptr6 = zerobuf;
      src_inc6 = 0;
    }
    src_ptr7 = zerobuf;
    src_inc7 = 0;
  }

  for (int k = 0; k < src_rows; k += 16) {
    for (int m = 0; m < 2; ++m) {
      const int packed_rows = src_rows - k - 8 * m;
      // Effectively,
      // packed_rows = std::max(0, std::min(8, src_rows - k - 8 * m));
      // but treat each case separately.
      if (packed_rows > 7) {
        for (int i = 0; i < 8; ++i) {
          in_data[0][i] = src_ptr0[i];
          in_data[1][i] = src_ptr1[i];
          in_data[2][i] = src_ptr2[i];
          in_data[3][i] = src_ptr3[i];
          in_data[4][i] = src_ptr4[i];
          in_data[5][i] = src_ptr5[i];
          in_data[6][i] = src_ptr6[i];
          in_data[7][i] = src_ptr7[i];
        }
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 8; ++j) {
            packed_ptr[16 * i + j] = in_data[j][i];
          }
        }
      } else if (packed_rows > 0) {
        for (int i = 0; i < packed_rows; ++i) {
          in_data[0][i] = src_ptr0[i];
          in_data[1][i] = src_ptr1[i];
          in_data[2][i] = src_ptr2[i];
          in_data[3][i] = src_ptr3[i];
          in_data[4][i] = src_ptr4[i];
          in_data[5][i] = src_ptr5[i];
          in_data[6][i] = src_ptr6[i];
          in_data[7][i] = src_ptr7[i];
        }
        for (int i = packed_rows; i < 8; ++i) {
          in_data[0][i] = 0.0f;
          in_data[1][i] = 0.0f;
          in_data[2][i] = 0.0f;
          in_data[3][i] = 0.0f;
          in_data[4][i] = 0.0f;
          in_data[5][i] = 0.0f;
          in_data[6][i] = 0.0f;
          in_data[7][i] = 0.0f;
        }
        // We loop through [0, 7) rather than [0, packed_rows), since that
        // emulates what we might do in fully-optimized code.
        for (int i = 0; i < 7; ++i) {
          for (int j = 0; j < 8; ++j) {
            trailing_buf[16 * i + j] = in_data[j][i];
          }
        }
      }

      packed_ptr += 16 * 8;
      src_ptr0 += src_inc0;
      src_ptr1 += src_inc1;
      src_ptr2 += src_inc2;
      src_ptr3 += src_inc3;
      src_ptr4 += src_inc4;
      src_ptr5 += src_inc5;
      src_ptr6 += src_inc6;
      src_ptr7 += src_inc7;
    }
  }
}

inline void ZeroHalfFloatAvxVnni(int src_rows, float* packed_ptr) {
  const int non_trailing_rows = src_rows & ~7;
  for (int k = 0; k < non_trailing_rows; ++k) {
    for (int j = 0; j < 8; ++j) {
      packed_ptr[j] = 0.0f;
    }
    packed_ptr += 16;
  }
}

}  // namespace.

// TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
// Optimization is not finished. In particular the dimensions of the kernel
// blocks can be changed as desired.
//
// When removing this comment, update profiling label below.
void Pack8bitAvxVnni(const std::int8_t* src_ptr, std::int8_t input_xor,
                     const std::int8_t* zerobuf, int src_stride,
                     int remaining_src_cols, int src_rows,
                     std::int8_t* packed_ptr, std::int32_t* sums_ptr) {
  gemmlowp::ScopedProfilingLabel label("Pack kAvxVnni 8bit (UNFINISHED)");

  // Each packed block is 4*16, and there are normally 8. The trailing block is
  // only slightly shorter.
  std::int8_t trailing_buf[8 * 16 * 4];
  memset(trailing_buf, 0, 8 * 16 * 4 * sizeof(std::int8_t));

  std::int32_t* second_sums_ptr = sums_ptr ? sums_ptr + 8 : nullptr;
  if (remaining_src_cols > 8) {
    HalfPack8bitAvxVnni(src_ptr, input_xor, zerobuf, src_stride,
                        remaining_src_cols, src_rows, packed_ptr, sums_ptr,
                        trailing_buf);
    HalfPack8bitAvxVnni(src_ptr + src_stride * 8, input_xor, zerobuf,
                        src_stride, remaining_src_cols - 8, src_rows,
                        packed_ptr + 8 * 4, second_sums_ptr,
                        trailing_buf + 8 * 4);
  } else {
    HalfPack8bitAvxVnni(src_ptr, input_xor, zerobuf, src_stride,
                        remaining_src_cols, src_rows, packed_ptr, sums_ptr,
                        trailing_buf);
    ZeroHalf8bitAvxVnni(src_rows, zerobuf[0] ^ input_xor, packed_ptr + 8 * 4);
    // The kernel may not need the second half-blocks sums to be set.
    if (second_sums_ptr) {
      for (int i = 0; i < 8; ++i) {
        second_sums_ptr[i] = (zerobuf[0] ^ input_xor) * ((src_rows + 3) & ~3);
      }
    }
  }
  const bool trailing_data = (src_rows & 31) > 0;
  // If the number of source rows is not a multiple of 32, there will be data in
  // the trailing buffer,
  if (trailing_data > 0) {
    const int non_trailing_rows = src_rows & ~31;
    // Destination "rows" are padded to next highest multiple of 4.
    const int dst_rows = (src_rows + 3) & ~3;
    const int trailing_rows = dst_rows - non_trailing_rows;
    memcpy(packed_ptr + 16 * non_trailing_rows, trailing_buf,
           16 * trailing_rows * sizeof(std::int8_t));
  }
}

// TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
// Optimization is not finished. In particular the dimensions of the kernel
// blocks can be changed as desired.
//
// When removing this comment, update profiling label below.
void PackFloatAvxVnni(const float* src_ptr, const float* zerobuf,
                      int src_stride, int remaining_src_cols, int src_rows,
                      float* packed_ptr) {
  gemmlowp::ScopedProfilingLabel label("Pack kAvxVnni float (UNFINISHED)");
  float trailing_buf[7 * 16];
  if (remaining_src_cols > 8) {
    HalfPackFloatAvxVnni(src_ptr, zerobuf, src_stride, remaining_src_cols,
                         src_rows, packed_ptr, trailing_buf);
    HalfPackFloatAvxVnni(src_ptr + src_stride * 8, zerobuf, src_stride,
                         remaining_src_cols - 8, src_rows, packed_ptr + 8,
                         trailing_buf + 8);
  } else {
    memset(trailing_buf, 0, sizeof(trailing_buf));
    HalfPackFloatAvxVnni(src_ptr, zerobuf, src_stride, remaining_src_cols,
                         src_rows, packed_ptr, trailing_buf);
    ZeroHalfFloatAvxVnni(src_rows, packed_ptr + 8);
  }
  const int trailing_rows = src_rows & 7;
  if (trailing_rows > 0) {
    const int non_trailing_rows = src_rows & ~7;
    memcpy(packed_ptr + 16 * non_trailing_rows, trailing_buf,
           16 * trailing_rows * sizeof(float));
  }
}

#endif  // RUY_PLATFORM(AVX_VNNI) && RUY_OPT_ENABLED(RUY_OPT_INTRINSICS)

}  // namespace ruy
