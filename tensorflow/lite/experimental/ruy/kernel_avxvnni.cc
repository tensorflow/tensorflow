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

#include "tensorflow/lite/experimental/ruy/check_macros.h"
#include "tensorflow/lite/experimental/ruy/kernel.h"
#include "tensorflow/lite/experimental/ruy/opt_set.h"
#include "tensorflow/lite/experimental/ruy/platform.h"
#include "tensorflow/lite/experimental/ruy/profiler/instrumentation.h"

#if RUY_PLATFORM(AVX_VNNI) && RUY_OPT_ENABLED(RUY_OPT_ASM)
#include <immintrin.h>  // IWYU pragma: keep
#endif

namespace ruy {

#if !(RUY_PLATFORM(AVX_VNNI) && RUY_OPT_ENABLED(RUY_OPT_ASM))

void Kernel8bitAvxVnni(const KernelParams8bit<16, 16>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

void KernelFloatAvxVnni(const KernelParamsFloat<16, 16>& params) {
  // CPU-ID-based checks should disable the path that would reach this point.
  RUY_DCHECK(false);
}

#else  // RUY_PLATFORM(AVX_VNNI) && RUY_OPT_ENABLED(RUY_OPT_ASM)

static constexpr int kAvxFloatBlockSize = 16;
static constexpr int kAvx8bitBlockSize = 16;
static constexpr int kAvx8bitInnerSize = 4;

// TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
// Optimization is not finished. In particular the dimensions of the kernel
// blocks can be changed as desired.
//
// When removing this comment, update profiling label below.
void Kernel8bitAvxVnni(const KernelParams8bit<16, 16>& params) {
  profiler::ScopeLabel label("Kernel kAvxVnni 8-bit (UNFINISHED)");

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
}  // NOLINT(readability/fn_size)

// TODO(b/147376783): SSE 4.2 and AVX-VNNI support is incomplete / placeholder.
// Optimization is not finished. In particular the dimensions of the kernel
// blocks can be changed as desired.
//
// When removing this comment, update profiling label below.
void KernelFloatAvxVnni(const KernelParamsFloat<16, 16>& params) {
  profiler::ScopeLabel label("Kernel kAvxVnni float (UNFINISHED)");

  float lhs_data[kAvxFloatBlockSize];
  float rhs_data[kAvxFloatBlockSize];
  float accum_data[kAvxFloatBlockSize][kAvxFloatBlockSize];
  int bias_ptr_block_increment =
      params.flags & RUY_ASM_FLAG_HAS_BIAS ? kAvxFloatBlockSize : 0;

  const float* rhs_col_ptr = params.rhs_base_ptr;
  float* dst_col_ptr = params.dst_base_ptr;
  const float* bias_col_ptr = params.bias;
  if (params.flags & RUY_ASM_FLAG_HAS_BIAS) {
    bias_col_ptr += params.start_row;
  }

  for (int col = params.start_col; col <= params.last_col;
       col += kAvxFloatBlockSize) {
    const float* lhs_col_ptr = params.lhs_base_ptr;
    float* dst_ptr = dst_col_ptr;
    const float* bias_ptr = bias_col_ptr;

    for (int row = params.start_row; row <= params.last_row;
         row += kAvxFloatBlockSize) {
      const int residual_rows =
          std::min(params.dst_rows - row, kAvxFloatBlockSize);
      const int residual_cols =
          std::min(params.dst_cols - col, kAvxFloatBlockSize);

      // Initialize with bias.
      float initial_accum_data[kAvxFloatBlockSize];
      for (int i = 0; i < kAvxFloatBlockSize; ++i) {
        initial_accum_data[i] = 0.0f;
      }
      for (int i = 0; i < residual_rows; ++i) {
        initial_accum_data[i] = bias_ptr[i];
      }
      for (int j = 0; j < kAvxFloatBlockSize; ++j) {
        for (int i = 0; i < kAvxFloatBlockSize; ++i) {
          accum_data[j][i] = initial_accum_data[i];
        }
      }
      bias_ptr += bias_ptr_block_increment;

      const float* lhs_ptr = lhs_col_ptr;
      const float* rhs_ptr = rhs_col_ptr;
      for (int d = 0; d < params.depth; ++d) {
        for (int i = 0; i < kAvxFloatBlockSize; ++i) {
          lhs_data[i] = lhs_ptr[i];
          rhs_data[i] = rhs_ptr[i];
        }

        for (int j = 0; j < kAvxFloatBlockSize; ++j) {
          for (int i = 0; i < kAvxFloatBlockSize; ++i) {
            accum_data[j][i] += lhs_data[i] * rhs_data[j];
          }
        }

        lhs_ptr += kAvxFloatBlockSize;
        rhs_ptr += kAvxFloatBlockSize;
      }

      for (int j = 0; j < kAvxFloatBlockSize; ++j) {
        for (int i = 0; i < kAvxFloatBlockSize; ++i) {
          accum_data[j][i] =
              std::min<float>(accum_data[j][i], params.clamp_max);
          accum_data[j][i] =
              std::max<float>(accum_data[j][i], params.clamp_min);
        }
      }

      const bool store_full_block = (residual_rows == kAvxFloatBlockSize) &&
                                    (residual_cols == kAvxFloatBlockSize);

      {
        float* block_ptr =
            store_full_block ? dst_ptr : const_cast<float*>(params.dst_tmp_buf);
        const int block_col_offset = store_full_block
                                         ? params.dst_stride / sizeof(float)
                                         : kAvxFloatBlockSize;
        for (int j = 0; j < kAvxFloatBlockSize; ++j) {
          for (int i = 0; i < kAvxFloatBlockSize; ++i) {
            block_ptr[i] = accum_data[j][i];
          }
          block_ptr += block_col_offset;
        }
      }
      if (!store_full_block) {
        const float* block_ptr = params.dst_tmp_buf;
        for (int j = 0; j < residual_cols; ++j) {
          for (int i = 0; i < residual_rows; ++i) {
            dst_ptr[j * params.dst_stride / sizeof(float) + i] = block_ptr[i];
          }
          block_ptr += kAvxFloatBlockSize;
        }
      }

      lhs_col_ptr += kAvxFloatBlockSize * params.lhs_stride / sizeof(float);
      dst_ptr += kAvxFloatBlockSize;
    }  // End row-block loop.

    dst_col_ptr += kAvxFloatBlockSize * params.dst_stride / sizeof(float);
    rhs_col_ptr += kAvxFloatBlockSize * params.rhs_stride / sizeof(float);
  }  // End col-block loop.
}

#endif  //  RUY_PLATFORM(AVX_VNNI) && RUY_OPT_ENABLED(RUY_OPT_ASM)

}  // namespace ruy
