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

#include <algorithm>
#include <cstdint>

#include "profiling/instrumentation.h"
#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/kernel.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/platform.h"

#if RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM))

void Kernel8bitAvx2(const KernelParams8bit<8, 8>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void KernelFloatAvx2(const KernelParamsFloat<8, 8>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM)

static constexpr int kAvxFloatBlockSize = 8;
static constexpr int kAvx8bitBlockSize = 8;
static constexpr int kAvx8bitInnerSize = 4;

namespace {
inline float mm256_get1_ps(const __m256 a, int i) {
  __m256i ai = _mm256_castps_si256(a);
  int float_val_as_int;
  switch (i) {
    case 0:
      float_val_as_int = _mm256_extract_epi32(ai, 0);
      break;
    case 1:
      float_val_as_int = _mm256_extract_epi32(ai, 1);
      break;
    case 2:
      float_val_as_int = _mm256_extract_epi32(ai, 2);
      break;
    case 3:
      float_val_as_int = _mm256_extract_epi32(ai, 3);
      break;
    case 4:
      float_val_as_int = _mm256_extract_epi32(ai, 4);
      break;
    case 5:
      float_val_as_int = _mm256_extract_epi32(ai, 5);
      break;
    case 6:
      float_val_as_int = _mm256_extract_epi32(ai, 6);
      break;
    case 7:
      float_val_as_int = _mm256_extract_epi32(ai, 7);
      break;
    default:
      RUY_DCHECK_LT(i, 8);
      return .0f;
  }
  return reinterpret_cast<float&>(float_val_as_int);
}

inline __m256 mm256_n_loadu_ps(int i, const float* src) {
  switch (i) {
    case 0:
      return _mm256_setzero_ps();
    case 1:
      return _mm256_setr_m128(_mm_setr_ps(src[0], .0f, .0f, .0f),
                              _mm_setzero_ps());
    case 2:
      return _mm256_setr_m128(_mm_setr_ps(src[0], src[1], .0f, .0f),
                              _mm_setzero_ps());
    case 3:
      return _mm256_setr_m128(_mm_setr_ps(src[0], src[1], src[2], .0f),
                              _mm_setzero_ps());
    case 4:
      return _mm256_setr_m128(_mm_setr_ps(src[0], src[1], src[2], src[3]),
                              _mm_setzero_ps());
    case 5:
      return _mm256_setr_ps(src[0], src[1], src[2], src[3], src[4], .0f, .0f,
                            .0f);
    case 6:
      return _mm256_setr_ps(src[0], src[1], src[2], src[3], src[4], src[5], .0f,
                            .0f);
    case 7:
      return _mm256_setr_ps(src[0], src[1], src[2], src[3], src[4], src[5],
                            src[6], .0f);
    case 8:
      return _mm256_loadu_ps(src);
    default:
      RUY_DCHECK(i < 9);
      return _mm256_setzero_ps();
  }
}

inline void _mm256_n_storeu_ps(float* dst, int residual_rows, const __m256 v) {
  for (int i = 0; i < residual_rows; ++i) {
    dst[i] = mm256_get1_ps(v, i);
  }
}
}  // namespace

void Kernel8bitAvx2(const KernelParams8bit<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx2");

  std::int32_t accum_data[kAvx8bitBlockSize][kAvx8bitBlockSize];
  int bias_ptr_block_increment =
      params.flags & RUY_ASM_FLAG_HAS_BIAS ? kAvx8bitBlockSize : 0;

  const std::int8_t* rhs_col_ptr = params.rhs_base_ptr;
  void* dst_col_ptr = params.dst_base_ptr;
  const std::int32_t* bias_col_ptr = params.bias;
  if (params.flags & RUY_ASM_FLAG_HAS_BIAS) {
    bias_col_ptr += params.start_row;
  }

  for (int col = params.start_col; col <= params.last_col;
       col += kAvx8bitBlockSize) {
    const std::int8_t* lhs_col_ptr = params.lhs_base_ptr;
    void* dst_ptr = dst_col_ptr;
    const std::int32_t* bias_ptr = bias_col_ptr;

    for (int row = params.start_row; row <= params.last_row;
         row += kAvx8bitBlockSize) {
      const int residual_rows =
          std::min(params.dst_rows - row, kAvx8bitBlockSize);
      const int residual_cols =
          std::min(params.dst_cols - col, kAvx8bitBlockSize);

      // Initialize with bias.
      std::int32_t initial_accum_data[kAvx8bitBlockSize];
      for (int i = 0; i < kAvx8bitBlockSize; ++i) {
        initial_accum_data[i] = 0;
      }
      for (int i = 0; i < residual_rows; ++i) {
        initial_accum_data[i] = bias_ptr[i];
      }
      for (int j = 0; j < kAvx8bitBlockSize; ++j) {
        for (int i = 0; i < kAvx8bitBlockSize; ++i) {
          accum_data[j][i] = initial_accum_data[i];
        }
      }
      bias_ptr += bias_ptr_block_increment;

      std::int8_t lhs_data[kAvx8bitBlockSize][kAvx8bitInnerSize];
      std::int8_t rhs_data[kAvx8bitBlockSize][kAvx8bitInnerSize];
      const std::int8_t* lhs_ptr = lhs_col_ptr;
      const std::int8_t* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; d += kAvx8bitInnerSize) {
        for (int i = 0; i < kAvx8bitBlockSize; ++i) {
          for (int x = 0; x < kAvx8bitInnerSize; ++x) {
            lhs_data[i][x] = lhs_ptr[i * kAvx8bitInnerSize + x];
            rhs_data[i][x] = rhs_ptr[i * kAvx8bitInnerSize + x];
          }
        }
        for (int j = 0; j < kAvx8bitBlockSize; ++j) {
          for (int i = 0; i < kAvx8bitBlockSize; ++i) {
            for (int x = 0; x < kAvx8bitInnerSize; ++x) {
              accum_data[j][i] += lhs_data[i][x] * rhs_data[j][x];
            }
          }
        }
        lhs_ptr += kAvx8bitBlockSize * kAvx8bitInnerSize;
        rhs_ptr += kAvx8bitBlockSize * kAvx8bitInnerSize;
      }

      if ((params.flags & RUY_ASM_FLAG_HAS_LHS_SUMS) && params.rhs_zero_point) {
        for (int j = 0; j < kAvx8bitBlockSize; ++j) {
          for (int i = 0; i < kAvx8bitBlockSize; ++i) {
            accum_data[j][i] -=
                params.rhs_zero_point * params.lhs_sums[row + i];
          }
        }
      }
      if ((params.flags & RUY_ASM_FLAG_HAS_RHS_SUMS) && params.lhs_zero_point) {
        for (int j = 0; j < kAvx8bitBlockSize; ++j) {
          for (int i = 0; i < kAvx8bitBlockSize; ++i) {
            accum_data[j][i] -=
                params.lhs_zero_point * params.rhs_sums[col + j];
          }
        }
      }
      if (params.lhs_zero_point && params.rhs_zero_point) {
        for (int j = 0; j < kAvx8bitBlockSize; ++j) {
          for (int i = 0; i < kAvx8bitBlockSize; ++i) {
            accum_data[j][i] += params.prod_zp_depth;
          }
        }
      }

      if (params.dst_type_id != DstTypeId<std::int32_t>::kValue) {
        std::int32_t m_vector[kAvx8bitBlockSize];
        std::int32_t e_vector[kAvx8bitBlockSize];
        // Does not make use of RUY_ASM_FLAG_NEEDS_LEFT_SHIFT.
        if (params.flags & RUY_ASM_FLAG_HAS_PERCHANNEL) {
          int i = 0;
          for (; i < residual_rows; ++i) {
            m_vector[i] = params.multiplier_fixedpoint[row + i];
            e_vector[i] = params.multiplier_exponent[row + i];
          }
          for (; i < kAvx8bitBlockSize; ++i) {
            m_vector[i] = m_vector[0];
            e_vector[i] = e_vector[0];
          }
        } else {
          // These arrays have size LhsCols, and are pre-filled.
          for (int i = 0; i < kAvx8bitBlockSize; ++i) {
            m_vector[i] = params.multiplier_fixedpoint[i];
            e_vector[i] = params.multiplier_exponent[i];
          }
        }
        for (int j = 0; j < kAvx8bitBlockSize; ++j) {
          for (int i = 0; i < kAvx8bitBlockSize; ++i) {
            accum_data[j][i] = MultiplyByQuantizedMultiplier(
                accum_data[j][i], m_vector[i], e_vector[i]);
          }
        }

        if (params.dst_zero_point) {
          for (int j = 0; j < kAvx8bitBlockSize; ++j) {
            for (int i = 0; i < kAvx8bitBlockSize; ++i) {
              accum_data[j][i] += params.dst_zero_point;
            }
          }
        }

        for (int j = 0; j < kAvx8bitBlockSize; ++j) {
          for (int i = 0; i < kAvx8bitBlockSize; ++i) {
            accum_data[j][i] =
                std::min<std::int32_t>(accum_data[j][i], params.clamp_max);
            accum_data[j][i] =
                std::max<std::int32_t>(accum_data[j][i], params.clamp_min);
          }
        }
      }

      const bool store_full_block = (residual_rows == kAvx8bitBlockSize) &&
                                    (residual_cols == kAvx8bitBlockSize);

      if (params.dst_type_id == DstTypeId<std::int8_t>::kValue) {
        std::int8_t* tmp_ptr =
            store_full_block
                ? static_cast<std::int8_t*>(dst_ptr)
                : const_cast<std::int8_t*>(
                      reinterpret_cast<const std::int8_t*>(params.dst_tmp_buf));
        const int block_col_offset =
            store_full_block ? params.dst_stride / sizeof(std::int8_t)
                             : kAvx8bitBlockSize;
        for (int j = 0; j < kAvx8bitBlockSize; ++j) {
          for (int i = 0; i < kAvx8bitBlockSize; ++i) {
            tmp_ptr[i] = accum_data[j][i];
          }
          tmp_ptr += block_col_offset;
        }

        if (!store_full_block) {
          const std::int8_t* block_ptr =
              reinterpret_cast<const std::int8_t*>(params.dst_tmp_buf);
          for (int j = 0; j < residual_cols; ++j) {
            for (int i = 0; i < residual_rows; ++i) {
              static_cast<std::int8_t*>(
                  dst_ptr)[j * params.dst_stride / sizeof(std::int8_t) + i] =
                  block_ptr[i];
            }
            block_ptr += kAvx8bitBlockSize;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int8_t*>(dst_ptr) +
                                     kAvx8bitBlockSize);
      } else if (params.dst_type_id == DstTypeId<std::uint8_t>::kValue) {
        std::uint8_t* tmp_ptr = store_full_block
                                    ? static_cast<std::uint8_t*>(dst_ptr)
                                    : const_cast<std::uint8_t*>(
                                          reinterpret_cast<const std::uint8_t*>(
                                              params.dst_tmp_buf));
        const int block_col_offset =
            store_full_block ? params.dst_stride : kAvx8bitBlockSize;
        for (int j = 0; j < kAvx8bitBlockSize; ++j) {
          for (int i = 0; i < kAvx8bitBlockSize; ++i) {
            tmp_ptr[i] = accum_data[j][i];
          }
          tmp_ptr += block_col_offset;
        }

        if (!store_full_block) {
          const std::uint8_t* block_ptr =
              reinterpret_cast<const std::uint8_t*>(params.dst_tmp_buf);
          for (int j = 0; j < residual_cols; ++j) {
            for (int i = 0; i < residual_rows; ++i) {
              static_cast<std::uint8_t*>(
                  dst_ptr)[j * params.dst_stride / sizeof(std::uint8_t) + i] =
                  block_ptr[i];
            }
            block_ptr += kAvx8bitBlockSize;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::uint8_t*>(dst_ptr) +
                                     kAvx8bitBlockSize);
      } else if (params.dst_type_id == DstTypeId<std::int16_t>::kValue) {
        if (store_full_block) {
          std::int16_t* tmp_ptr = static_cast<std::int16_t*>(dst_ptr);
          const int block_col_offset = params.dst_stride / sizeof(std::int16_t);
          for (int j = 0; j < kAvx8bitBlockSize; ++j) {
            for (int i = 0; i < kAvx8bitBlockSize; ++i) {
              tmp_ptr[i] = accum_data[j][i];
            }
            tmp_ptr += block_col_offset;
          }
        } else {
          std::int16_t* tmp_ptr = const_cast<std::int16_t*>(
              reinterpret_cast<const std::int16_t*>(params.dst_tmp_buf));
          const int block_col_offset = kAvx8bitBlockSize;
          for (int j = 0; j < kAvx8bitBlockSize; ++j) {
            for (int i = 0; i < kAvx8bitBlockSize; ++i) {
              tmp_ptr[i] = accum_data[j][i];
            }
            tmp_ptr += block_col_offset;
          }
          const std::int16_t* block_ptr =
              reinterpret_cast<const std::int16_t*>(params.dst_tmp_buf);
          std::int16_t* dst_block_ptr = static_cast<std::int16_t*>(dst_ptr);
          for (int j = 0; j < residual_cols; ++j) {
            for (int i = 0; i < residual_rows; ++i) {
              dst_block_ptr[i] = block_ptr[i];
            }
            dst_block_ptr += params.dst_stride / sizeof(std::int16_t);
            block_ptr += kAvx8bitBlockSize;
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int16_t*>(dst_ptr) +
                                     kAvx8bitBlockSize);
      } else if (params.dst_type_id == DstTypeId<std::int32_t>::kValue) {
        if (store_full_block) {
          std::int32_t* tmp_ptr = static_cast<std::int32_t*>(dst_ptr);
          const int block_col_offset = params.dst_stride / sizeof(std::int32_t);
          for (int j = 0; j < kAvx8bitBlockSize; ++j) {
            for (int i = 0; i < kAvx8bitBlockSize; ++i) {
              tmp_ptr[i] = accum_data[j][i];
            }
            tmp_ptr += block_col_offset;
          }
        } else {
          std::int32_t* dst_block_ptr = static_cast<std::int32_t*>(dst_ptr);
          for (int j = 0; j < residual_cols; ++j) {
            for (int i = 0; i < residual_rows; ++i) {
              dst_block_ptr[i] = accum_data[j][i];
            }
            dst_block_ptr += params.dst_stride / sizeof(std::int32_t);
          }
        }
        dst_ptr = static_cast<void*>(static_cast<std::int32_t*>(dst_ptr) +
                                     kAvx8bitBlockSize);
      } else {
        RUY_DCHECK(false);
      }

      lhs_col_ptr += kAvx8bitBlockSize * params.lhs_stride;
    }  // End row-block loop.

    dst_col_ptr = static_cast<void*>(static_cast<char*>(dst_col_ptr) +
                                     kAvx8bitBlockSize * params.dst_stride);
    rhs_col_ptr += kAvx8bitBlockSize * params.rhs_stride;
  }  // End col-block loop.
}

void KernelFloatAvx2(const KernelParamsFloat<8, 8>& params) {
  gemmlowp::ScopedProfilingLabel label("Kernel kAvx2");

  // As parameters are defined, we need to scale by sizeof(float).
  const std::int64_t lhs_stride = params.lhs_stride >> 2;
  const std::int64_t dst_stride = params.dst_stride >> 2;
  const std::int64_t rhs_stride = params.rhs_stride >> 2;
  //
  int bias_ptr_block_increment = params.flags & RUY_ASM_FLAG_HAS_BIAS ? 1 : 0;
  // kAvxFloatBlockSize = 8.
  const int end_row = std::min(params.dst_rows, params.last_row + 8);
  const int end_col = std::min(params.dst_cols, params.last_col + 8);
  //
  const float* adj_rhs_col_ptr =
      params.rhs_base_ptr - params.start_col * rhs_stride;
  float* adj_dst_col_ptr =
      params.dst_base_ptr - params.start_col * dst_stride - params.start_row;
  const float* adj_lhs_col_ptr =
      params.lhs_base_ptr - params.start_row * lhs_stride;
  const float* bias_col_ptr = params.bias;

  const __m256 clamp_max_v = _mm256_set1_ps(params.clamp_max);
  const __m256 clamp_min_v = _mm256_set1_ps(params.clamp_min);

  int col = params.start_col;
  // Loop through cols by kAvxFloatBlockSize, leaving incomplete remainder
  for (; col <= end_col - 8; col += 8) {
    __m256 accum_data_v[8];

    const float* rhs_col_ptr = adj_rhs_col_ptr + col * rhs_stride;
    float* dst_col_ptr = adj_dst_col_ptr + col * dst_stride;

    for (int row = params.start_row; row < end_row; row += 8) {
      const int residual_rows = std::min(end_row - row, 8);

      const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
      float* dst_ptr = dst_col_ptr + row;
      const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

      // Initialize with bias.
      const __m256 initial_accum_data =
          mm256_n_loadu_ps(residual_rows, bias_ptr);

      for (int j = 0; j < 8; ++j) {
        accum_data_v[j] = initial_accum_data;
      }

      const float* lhs_ptr = lhs_col_ptr;
      const float* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; ++d) {
        const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
        const __m256 rhs_data = _mm256_loadu_ps(rhs_ptr);

        for (int j = 0; j < 8; ++j) {
          const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[j]);
          accum_data_v[j] =
              _mm256_fmadd_ps(lhs_data, dup_rhs_element_j, accum_data_v[j]);
        }
        lhs_ptr += 8;
        rhs_ptr += 8;
      }

      if (residual_rows == 8) {
        for (int j = 0; j < 8; ++j) {
          float* block_ptr = dst_ptr + j * dst_stride;
          accum_data_v[j] = _mm256_min_ps(accum_data_v[j], clamp_max_v);
          accum_data_v[j] = _mm256_max_ps(accum_data_v[j], clamp_min_v);
          _mm256_storeu_ps(block_ptr, accum_data_v[j]);
        }
      } else {
        for (int j = 0; j < 8; ++j) {
          float* block_ptr = dst_ptr + j * dst_stride;
          accum_data_v[j] = _mm256_min_ps(accum_data_v[j], clamp_max_v);
          accum_data_v[j] = _mm256_max_ps(accum_data_v[j], clamp_min_v);
          _mm256_n_storeu_ps(block_ptr, residual_rows, accum_data_v[j]);
        }
      }
    }  // End row-block loop.
  }    // End col-block loop.

  if (col < end_col) {
    // Remaining cols in [0, kAvxFloatBlockSize).
    RUY_DCHECK_GE(end_col - col, 0);
    RUY_DCHECK_LT(end_col - col, 8);

    __m256 accum_data_v[8];

    const float* rhs_col_ptr = adj_rhs_col_ptr + col * rhs_stride;
    float* dst_col_ptr = adj_dst_col_ptr + col * dst_stride;
    const int residual_cols = std::min(end_col - col, 8);

    for (int row = params.start_row; row < end_row; row += 8) {
      const int residual_rows = std::min(end_row - row, 8);

      const float* lhs_col_ptr = adj_lhs_col_ptr + row * lhs_stride;
      float* dst_ptr = dst_col_ptr + row;
      const float* bias_ptr = bias_col_ptr + row * bias_ptr_block_increment;

      // Initialize with bias.
      const __m256 initial_accum_data =
          mm256_n_loadu_ps(residual_rows, bias_ptr);

      for (int j = 0; j < 8; ++j) {
        accum_data_v[j] = initial_accum_data;
      }

      const float* lhs_ptr = lhs_col_ptr;
      const float* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; ++d) {
        const __m256 lhs_data = _mm256_loadu_ps(lhs_ptr);
        const __m256 rhs_data = _mm256_loadu_ps(rhs_ptr);

        for (int j = 0; j < 8; ++j) {
          const __m256 dup_rhs_element_j = _mm256_set1_ps(rhs_data[j]);
          accum_data_v[j] =
              _mm256_fmadd_ps(lhs_data, dup_rhs_element_j, accum_data_v[j]);
        }
        lhs_ptr += 8;
        rhs_ptr += 8;
      }

      for (int j = 0; j < residual_cols; ++j) {
        float* block_ptr = dst_ptr + j * dst_stride;
        accum_data_v[j] = _mm256_min_ps(accum_data_v[j], clamp_max_v);
        accum_data_v[j] = _mm256_max_ps(accum_data_v[j], clamp_min_v);
        _mm256_n_storeu_ps(block_ptr, residual_rows, accum_data_v[j]);
      }
    }  // End row-block loop.
  }    // End col-block terminal conditional.
}

#endif  //  RUY_PLATFORM(AVX2) && RUY_OPT_ENABLED(RUY_OPT_ASM)

}  // namespace ruy
