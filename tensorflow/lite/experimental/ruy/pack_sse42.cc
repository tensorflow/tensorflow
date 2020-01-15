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

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/matrix.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/pack.h"
#include "tensorflow/lite/experimental/ruy/path.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"

#if RUY_PLATFORM(SSE42) && RUY_OPT_ENABLED(RUY_OPT_INTRINSICS)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM(SSE42) && RUY_OPT_ENABLED(RUY_OPT_ASM))

void Pack8bitSse42(const std::int8_t* src_ptr, std::int8_t input_xor,
                   const std::int8_t* zerobuf, int src_stride,
                   int remaining_src_cols, int src_rows,
                   std::int8_t* packed_ptr, std::int32_t* sums_ptr) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void PackFloatSse42(const float* src_ptr, const float* zerobuf, int src_stride,
                    int remaining_src_cols, int src_rows, float* packed_ptr) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM(SSE42) && RUY_OPT_ENABLED(RUY_OPT_ASM)

// The first int8_t template parameter is arbitrary: this routine is common to
// all 8-bit source matrix types.
using PackImpl8bitSse42 =
    PackImpl<Path::kSse42, FixedKernelLayout<Order::kColMajor, 4, 8>,
             std::int8_t, std::int8_t, std::int32_t>;

using PackImplFloatSse42 =
    PackImpl<Path::kSse42, FixedKernelLayout<Order::kRowMajor, 1, 8>, float,
             float, float>;

namespace {

inline void Pack8bitSse42Packer(const std::int8_t* src_ptr,
                                std::int8_t input_xor,
                                const std::int8_t* zerobuf, int src_stride,
                                int remaining_src_cols, int src_rows,
                                std::int8_t* packed_ptr, std::int32_t* sums_ptr,
                                std::int8_t* trailing_buf) {
  using Layout = PackImpl8bitSse42::Layout;
  RUY_DCHECK_EQ(Layout::kCols, 8);
  RUY_DCHECK_EQ(Layout::kRows, 4);
  // Each Layout::Rows is 4 contiguous input, contiguous packed elements.
  // We process 8 of these chunks at a time, padding short input chunks.
  constexpr int kNumRowChunks = 8;
  constexpr int kNumChunkedSrcRows = kNumRowChunks * Layout::kRows;

  std::int8_t in_data[Layout::kCols][kNumRowChunks][Layout::kRows];

  const std::int8_t* src_ptr0 = src_ptr;
  const std::int8_t* src_ptr1 = src_ptr0 + src_stride;
  const std::int8_t* src_ptr2 = src_ptr1 + src_stride;
  const std::int8_t* src_ptr3 = src_ptr2 + src_stride;
  const std::int8_t* src_ptr4 = src_ptr3 + src_stride;
  const std::int8_t* src_ptr5 = src_ptr4 + src_stride;
  const std::int8_t* src_ptr6 = src_ptr5 + src_stride;
  const std::int8_t* src_ptr7 = src_ptr6 + src_stride;
  std::int64_t src_inc0 = kNumChunkedSrcRows;
  std::int64_t src_inc1 = kNumChunkedSrcRows;
  std::int64_t src_inc2 = kNumChunkedSrcRows;
  std::int64_t src_inc3 = kNumChunkedSrcRows;
  std::int64_t src_inc4 = kNumChunkedSrcRows;
  std::int64_t src_inc5 = kNumChunkedSrcRows;
  std::int64_t src_inc6 = kNumChunkedSrcRows;
  std::int64_t src_inc7 = kNumChunkedSrcRows;
  // Handle cases where source does not have Layout::kCols (8) columns.
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
    // i: Layout::kCols.
    for (int i = 0; i < 8; ++i) {
      sums_ptr[i] = 0;
    }
  }

  // The overall packing effectively pads the source rows to
  // (src_rows + 63) & ~63. The iteration over k may skip when m=1, and then we
  // only pack for (src_rows + 31) & ~31. When there is an incomplete
  // destination block, this is stored into trailing_buf instead of packed_ptr.
  for (int k = 0; k < src_rows; k += kNumChunkedSrcRows) {
    // Available source rows.
    // If this is less than 0 (for m=1), we skip, having filled trailing
    // buffer for m=0. Also, if source rows is zero on m=1, then we filled
    // exactly to the end of the column in the packed buffer.
    const int available_src_rows = src_rows - k;
    // Effectively,
    // available rows = std::max(0, std::min(8, src_rows - k));
    // treat each case separately.
    if (available_src_rows >= kNumChunkedSrcRows) {
      // i: chunks, s: Layout::Rows.
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
      // i: chunks, j: Layout::kCols, s: Layout::Rows.
      for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
          for (int s = 0; s < 4; ++s) {
            // 8 * 4 * i is offset for each block, that is
            // (Layout::kCols * Layout::kRows * i)
            packed_ptr[(8 * i + j) * 4 + s] = in_data[j][i][s] ^ input_xor;
          }
          if (sums_ptr) {
            for (int s = 0; s < 4; ++s) {
              sums_ptr[j] += in_data[j][i][s] ^ input_xor;
            }
          }
        }
      }
    } else if (available_src_rows > 0) {
      RUY_DCHECK_LT(available_src_rows, kNumChunkedSrcRows);
      int i = 0;
      // Consume chunks of 4 rows that are complete.
      for (; i < (available_src_rows >> 2); ++i) {
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
      // Consume any incomplete chunk.
      if (i < ((available_src_rows + 3) >> 2)) {
        int s = 0;
        for (; s < (available_src_rows & 3); ++s) {
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
          // j: Layout::kCols.
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
      //                         4 * (8 - ((available_src_rows + 3) >> 2)).
      for (; i < 8; ++i) {
        for (int s = 0; s < 4; ++s) {
          for (int j = 0; j < 8; ++j) {
            in_data[j][i][s] = input_xor;
          }
        }
      }
      // We loop through [0, 8) rather than
      // [0, (available_src_rows + 3) >> 2), since that emulates what we might
      // do in fully-optimized code.
      //
      // i: chunks, j: Layout::kCols, s: Layout::Rows.
      if (sums_ptr) {
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 8; ++j) {
            for (int s = 0; s < 4; ++s) {
              trailing_buf[(8 * i + j) * 4 + s] = in_data[j][i][s] ^ input_xor;
              sums_ptr[j] = sums_ptr[j] + (in_data[j][i][s] ^ input_xor);
            }
          }
        }
      } else {
        for (int i = 0; i < 8; ++i) {
          for (int j = 0; j < 8; ++j) {
            for (int s = 0; s < 4; ++s) {
              trailing_buf[(8 * i + j) * 4 + s] = in_data[j][i][s] ^ input_xor;
            }
          }
        }
      }
    }

    packed_ptr += 8 * kNumChunkedSrcRows;
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

inline void PackFloatSse42Packer(const float* src_ptr, const float* zerobuf,
                                 int src_stride, int remaining_src_cols,
                                 int src_rows, float* packed_ptr,
                                 float* trailing_buf) {
  using Layout = PackImplFloatSse42::Layout;
  RUY_DCHECK_EQ(Layout::kCols, 8);
  RUY_DCHECK_EQ(Layout::kRows, 1);

  // This packing amounts to tranposition of 8x8 blocks.
  static constexpr int kPackCols = 8;  // Source cols packed together.
  static constexpr int kPackRows = 8;  // Short input is padded.

  float in_data[kPackCols][kPackRows];

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
  // Handle cases where source does not have kPackDim (8) columns.
  if (remaining_src_cols < kPackCols) {
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

  for (int k = 0; k < src_rows; k += kPackRows) {
    const int available_src_rows = src_rows - k;
    // Effectively,
    // available_src_rows = std::max(0, std::min(kPackDim, src_rows - k));
    // but treat each case separately.
    if (available_src_rows >= kPackRows) {
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
          packed_ptr[8 * i + j] = in_data[j][i];
        }
      }
    } else if (available_src_rows > 0) {
      for (int i = 0; i < available_src_rows; ++i) {
        in_data[0][i] = src_ptr0[i];
        in_data[1][i] = src_ptr1[i];
        in_data[2][i] = src_ptr2[i];
        in_data[3][i] = src_ptr3[i];
        in_data[4][i] = src_ptr4[i];
        in_data[5][i] = src_ptr5[i];
        in_data[6][i] = src_ptr6[i];
        in_data[7][i] = src_ptr7[i];
      }
      for (int i = available_src_rows; i < kPackRows; ++i) {
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
      // i: (kPackRows - 1), j: kPackCols.
      for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 8; ++j) {
          trailing_buf[kPackRows * i + j] = in_data[j][i];
        }
      }
    }

    packed_ptr += kPackRows * kPackCols;
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

}  // namespace.

// TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
// Optimization is not finished. In particular the dimensions of the kernel
// blocks can be changed as desired.
//
// When removing this comment, update profiling label below.
void Pack8bitSse42(const std::int8_t* src_ptr, std::int8_t input_xor,
                   const std::int8_t* zerobuf, int src_stride,
                   int remaining_src_cols, int src_rows,
                   std::int8_t* packed_ptr, std::int32_t* sums_ptr) {
  profiler::ScopeLabel label("Pack kSse42 8bit (UNFINISHED)");

  using Layout = PackImpl8bitSse42::Layout;
  RUY_DCHECK_EQ(Layout::kCols, 8);
  RUY_DCHECK_EQ(Layout::kRows, 4);

  // Each Layout::Rows is 4 contiguous input, contiguous packed elements.
  // We process 8 of these chunks at a time, padding short input chunks.
  static constexpr int kNumRowChunks = 8;  // Short input is padded.

  // Each packed block is 4*8, and there are normally 8. The trailing block is
  // only slightly shorter.
  constexpr int kTrailingBufSize =
      kNumRowChunks * Layout::kCols * Layout::kRows;
  std::int8_t trailing_buf[kTrailingBufSize];
  memset(trailing_buf, 0, kTrailingBufSize * sizeof(std::int8_t));

  Pack8bitSse42Packer(src_ptr, input_xor, zerobuf, src_stride,
                      remaining_src_cols, src_rows, packed_ptr, sums_ptr,
                      trailing_buf);

  constexpr int kChunkedRowMask = kNumRowChunks * Layout::kRows - 1;
  const bool trailing_data = (src_rows & kChunkedRowMask) > 0;
  // If the number of source rows is not a multiple of kChunkedRowMask, there
  // will be data in the trailing buffer,
  if (trailing_data > 0) {
    const int non_trailing_rows = src_rows & ~kChunkedRowMask;
    // Destination "rows" are padded to next highest multiple of Layout::kRows.
    const int dst_rows = (src_rows + 3) & ~3;
    const int trailing_rows = dst_rows - non_trailing_rows;
    memcpy(packed_ptr + Layout::kCols * non_trailing_rows, trailing_buf,
           Layout::kCols * trailing_rows * sizeof(std::int8_t));
  }
}

// TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
// Optimization is not finished. In particular the dimensions of the kernel
// blocks can be changed as desired.
//
// When removing this comment, update profiling label below.
void PackFloatSse42(const float* src_ptr, const float* zerobuf, int src_stride,
                    int remaining_src_cols, int src_rows, float* packed_ptr) {
  profiler::ScopeLabel label("Pack kSse42 float (UNFINISHED)");
  static constexpr int kPackCols = 8;  // Source cols packed together.
  static constexpr int kPackRows = 8;  // Short input is padded.
  float trailing_buf[(kPackRows - 1) * kPackCols];
  if (remaining_src_cols < 8) {
    memset(trailing_buf, 0, sizeof(trailing_buf));
  }
  PackFloatSse42Packer(src_ptr, zerobuf, src_stride, remaining_src_cols,
                       src_rows, packed_ptr, trailing_buf);

  const int trailing_rows = src_rows & (kPackRows - 1);
  if (trailing_rows > 0) {
    const int non_trailing_rows = src_rows & ~(kPackRows - 1);
    memcpy(packed_ptr + kPackCols * non_trailing_rows, trailing_buf,
           kPackCols * trailing_rows * sizeof(float));
  }
}

#endif  // RUY_PLATFORM(SSE42) && RUY_OPT_ENABLED(RUY_OPT_INTRINSICS)

}  // namespace ruy
