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
#include <fcntl.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/activation_functor.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/tensor_utils_impl.h"
#include "tensorflow/lite/kernels/internal/round.h"

#ifdef USE_NEON

#define kFloatWeightsPerNeonLane 4

namespace tflite {
namespace tensor_utils {
namespace {

// Allocates, at least, size bytes of uninitialized storage whose alignment is
// specified by alignment. The size parameter must be an integral multiple of
// alignment.
// Caller is responsible by freeing the allocated memory by calling free on
// the passed freeing_buffer pointer.
void* aligned_alloc(size_t alignment, size_t size, void** freeing_buffer) {
  *freeing_buffer = malloc(size + alignment);
  const size_t offset = ((uintptr_t)*freeing_buffer) % alignment;  // NOLINT
  return offset == 0
             ? *freeing_buffer
             : ((char*)*freeing_buffer + (alignment - offset));  // NOLINT
}

// Use /proc/cpuinfo to test whether we have the right processor.
bool HasSdotInstruction() {
  // TODO(strohman): Replace this with a proper API call once we are running
  // on kernels that can tell us about this instruction: (b/119112014)
  // Note that the C++ spec ensures that this variable will be initialized
  // exactly once.
  static bool has_sdot = []() -> bool {
    char text[1024];
    int fd = open("/proc/cpuinfo", O_RDONLY);
    if (fd < 0) {
      return false;
    }

    bool found = false;
    int buffer = 0;
    const char kSM8150[] = "Qualcomm Technologies, Inc SM8150";
    while (true) {
      int count = read(fd, text + buffer, sizeof(text) - buffer);
      if (count <= 0) {
        break;
      }
      int text_end = buffer + count;

      if (memmem(text, text_end, kSM8150, sizeof(kSM8150) - 1) != nullptr) {
        found = true;
        break;
      }

      // Keep up to some bytes of the previous buffer state so that we
      // can find a string match even if it occurs on a buffer boundary.
      buffer = text_end;
      if (text_end > sizeof(kSM8150)) {
        buffer = sizeof(kSM8150);
      }

      memmove(text, text + text_end - buffer, buffer);
    }
    close(fd);
    return found;
  }();
  return has_sdot;
}

}  // namespace

void NeonMatrixBatchVectorMultiplyAccumulate(const float* matrix, int m_rows,
                                             int m_cols, const float* vector,
                                             int n_batch, float* result,
                                             int result_stride) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      m_cols - (m_cols & (kFloatWeightsPerNeonLane - 1));

  for (int b = 0; b < n_batch; b++) {
    float* result_in_batch = result + b * m_rows * result_stride;
    const float* vector_in_batch = vector + b * m_cols;
    const float* matrix_row = matrix;

    // Main matrix by vector multiplication loop
    for (int r = 0; r < m_rows; r++) {
      float32x4_t acc_32x4 = vmovq_n_f32(0.0);
      for (int c = 0; c < postamble_start; c += kFloatWeightsPerNeonLane) {
        // Load 4 float values from vector and matrix row.
        float32x4_t vector_f32x4 = vld1q_f32(vector_in_batch + c);
        float32x4_t matrix_f32x4 = vld1q_f32(matrix_row + c);
        // Multiply the vector and matrix row and add to accumulator.
        acc_32x4 = vmlaq_f32(acc_32x4, matrix_f32x4, vector_f32x4);
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this column.
      *result_in_batch +=
          (vgetq_lane_f32(acc_32x4, 0) + vgetq_lane_f32(acc_32x4, 1) +
           vgetq_lane_f32(acc_32x4, 2) + vgetq_lane_f32(acc_32x4, 3));
      for (int c = postamble_start; c < m_cols; c++) {
        *result_in_batch += matrix_row[c] * vector_in_batch[c];
      }
      matrix_row += m_cols;
      result_in_batch += result_stride;
    }
  }
}

#ifdef __aarch64__

// We interleave vector data to make the dot product logic more efficient.
// Suppose that vectors is:
//     a0 a1 a2 a3 a4 a5 ...
//     b0 b1 b2 b3 b4 b5 ...
//     c0 c1 c2 c3 c4 c5 ...
//     d0 d1 d2 d3 d4 d5 ...
//     e0 e1 e2 e3 e4 e5 ...
// This code interleaves them like this:
//     a0 a1 a2 a3 b0 b1 b2 b3 c0 c1 c2 c3 d0 d1 d2 d3 a4 a5 a6 a7 b4 ...
//     e0 e1 e2 e3 f0 f1 f2 f3 ...
// Once the data is interleaved, each 16-byte read from the vectors pointer
// contains 4 bytes from each of 4 vectors.
const int8_t* ShuffleVectors(const int8_t* vectors, const int n_batch,
                             const int m_cols, void** shuffled_vectors_free) {
  const int kWeightsPerUint32 = 4;

  int8* shuffled_vectors = reinterpret_cast<int8*>(aligned_alloc(
      kWeightsPerUint32, n_batch * m_cols, shuffled_vectors_free));

  for (int i = 0; i < n_batch; i += 4) {
    int8* shuffled_vectors_ptr = shuffled_vectors + (i * m_cols);
    const int8* unshuffled_vec0_ptr =
        reinterpret_cast<const int8*>(vectors) + (i * m_cols);
    const int8* unshuffled_vec1_ptr =
        reinterpret_cast<const int8*>(vectors) + ((i + 1) * m_cols);
    const int8* unshuffled_vec2_ptr =
        reinterpret_cast<const int8*>(vectors) + ((i + 2) * m_cols);
    const int8* unshuffled_vec3_ptr =
        reinterpret_cast<const int8*>(vectors) + ((i + 3) * m_cols);
    const int8* const end_vec0_ptr = unshuffled_vec1_ptr;

    while (unshuffled_vec0_ptr != end_vec0_ptr) {
      asm volatile(
          // This code path requires that (n_cols % 16) == 0 so we can safely
          // read in 16-byte chunks from each row.
          "ld1 {v0.16b}, [%[unshuffled_vec0_ptr]], #16\n"
          "ld1 {v1.16b}, [%[unshuffled_vec1_ptr]], #16\n"
          "ld1 {v2.16b}, [%[unshuffled_vec2_ptr]], #16\n"
          "ld1 {v3.16b}, [%[unshuffled_vec3_ptr]], #16\n"

          "st4 {v0.s, v1.s, v2.s, v3.s}[0], [%[shuffled_vectors_ptr]], #16\n"
          "st4 {v0.s, v1.s, v2.s, v3.s}[1], [%[shuffled_vectors_ptr]], #16\n"
          "st4 {v0.s, v1.s, v2.s, v3.s}[2], [%[shuffled_vectors_ptr]], #16\n"
          "st4 {v0.s, v1.s, v2.s, v3.s}[3], [%[shuffled_vectors_ptr]], #16\n"

          : [ unshuffled_vec0_ptr ] "+r"(unshuffled_vec0_ptr),
            [ unshuffled_vec1_ptr ] "+r"(unshuffled_vec1_ptr),
            [ unshuffled_vec2_ptr ] "+r"(unshuffled_vec2_ptr),
            [ unshuffled_vec3_ptr ] "+r"(unshuffled_vec3_ptr),
            [ shuffled_vectors_ptr ] "+r"(shuffled_vectors_ptr)
          :
          : "v0", "v1", "v2", "v3", "cc", "memory");
    }
  }

  return reinterpret_cast<const int8_t*>(shuffled_vectors);
}

static void DotprodMatrixBatchFourVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* vectors, const float* scaling_factors, int n_batch,
    float* __restrict__ result) {
  void* shuffled_vectors_free;

  const int8_t* shuffled_vectors =
      ShuffleVectors(vectors, n_batch, m_cols, &shuffled_vectors_free);

  for (int row = 0; row < m_rows; row += 2) {
    for (int batch = 0; batch < n_batch; batch += 4) {
      float* result_ptr = result + (batch * m_rows) + row;
      const int8* mat_ptr0 = matrix + (row * m_cols);
      const int8* mat_ptr1 = matrix + ((row + 1) * m_cols);
      const int8* mat_ptr0_end = mat_ptr1;
      const int8* vec_ptr = shuffled_vectors + (batch * m_cols);
      const float* scaling_factors_ptr = scaling_factors + batch;
      const uint64_t wide_rows = m_rows * sizeof(float);

      asm volatile(
          // Zero out the accumulator registers.
          "dup v0.4s, wzr\n"
          "dup v1.4s, wzr\n"
          "dup v2.4s, wzr\n"
          "dup v3.4s, wzr\n"

          "1:\n"  // batch_cols_loop

          // Read 16 more bytes from a pair of matrix rows.
          "ld1 {v12.16b}, [%[mat_ptr0]], #16\n"

          // Read from input vectors 4 times; 64 bytes total.
          // Each 16-byte register contains parts of 4 vectors; see the
          // shuffle logic above.

          // From Benoit, places to look in the future:
          // - Move load instructions further from sdot
          // - Switch loop use-then-reload
          // - Do partial unrolling to use register space better
          "ld1 {v8.16b}, [%[vec_ptr]], #16\n"
          ".word 0x4f8ce100  // sdot v0.4s, v8.16b, v12.4b[0]\n"
          "ld1 {v9.16b}, [%[vec_ptr]], #16\n"
          ".word 0x4face121  // sdot v1.4s, v9.16b, v12.4b[1]\n"
          "ld1 {v10.16b}, [%[vec_ptr]], #16\n"
          ".word 0x4f8ce940  // sdot v0.4s, v10.16b, v12.4b[2]\n"
          "ld1 {v11.16b}, [%[vec_ptr]], #16\n"
          ".word 0x4face961  // sdot v1.4s, v11.16b, v12.4b[3]\n"

          // Re-use those vectors for the next row as well.
          "ld1 {v13.16b}, [%[mat_ptr1]], #16\n"
          ".word 0x4f8de102  // sdot v2.4s, v8.16b, v13.4b[0]\n"
          ".word 0x4fade123  // sdot v3.4s, v9.16b, v13.4b[1]\n"
          ".word 0x4f8de942  // sdot v2.4s, v10.16b, v13.4b[2]\n"
          ".word 0x4fade963  // sdot v3.4s, v11.16b, v13.4b[3]\n"

          // If we're not done with these rows, continue.
          "cmp %[mat_ptr0], %[mat_ptr0_end]\n"
          "bne 1b\n"  // batch_cols_loop

          // Done with the rows, sum the results.
          "add v0.4s, v0.4s, v1.4s\n"
          "add v2.4s, v2.4s, v3.4s\n"

          // Convert the per-vector sums to floating point.
          "scvtf v0.4s, v0.4s\n"
          "scvtf v1.4s, v2.4s\n"

          // Fetch scale factors.
          "ld1 {v4.4s}, [%[scaling_factors_ptr]]\n"

          // Multiply scale factors times sums.
          "fmul v0.4s, v4.4s, v0.4s\n"
          "fmul v1.4s, v4.4s, v1.4s\n"

          // Load previous result values.
          // The result position is:
          //   result[batch * m_rows + row]
          // Here that is factored into:
          //   result_ptr = result + row
          //   *result_ptr = res[0]
          //   (uint8*)result_ptr += (m_rows * sizeof(float))
          //   *result_ptr = res[1]
          //   ...
          // Since we're reading two rows at a time, though, we read both
          //   result[batch * m_rows + row]
          // and
          //   result[batch * m_rows + row + 1]
          "ld2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
          "ld2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
          "ld2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
          "ld2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"

          // Go back to the starting position (subtract wide_rows * 4).
          "sub %[result_ptr], %[result_ptr], %[wide_rows], lsl #2\n"

          // Add previous result values.
          "fadd v9.4s, v9.4s, v0.4s\n"
          "fadd v10.4s, v10.4s, v1.4s\n"

          // Store results.
          "st2 {v9.s, v10.s}[0], [%[result_ptr]], %[wide_rows]\n"
          "st2 {v9.s, v10.s}[1], [%[result_ptr]], %[wide_rows]\n"
          "st2 {v9.s, v10.s}[2], [%[result_ptr]], %[wide_rows]\n"
          "st2 {v9.s, v10.s}[3], [%[result_ptr]], %[wide_rows]\n"
          : [ mat_ptr0 ] "+r"(mat_ptr0), [ mat_ptr1 ] "+r"(mat_ptr1),
            [ vec_ptr ] "+r"(vec_ptr), [ result_ptr ] "+r"(result_ptr)
          : [ mat_ptr0_end ] "r"(mat_ptr0_end),
            [ scaling_factors_ptr ] "r"(scaling_factors_ptr),
            [ wide_rows ] "r"(wide_rows)
          : "x0", "v0", "v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9",
            "v10", "v11", "v12", "v13", "cc", "memory");
    }
  }

  free(shuffled_vectors_free);
}

static void DotprodSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride) {
  const uint8_t* ledger_ptr = ledger;
  const int8* mat_ptr = matrix;

  for (int row = 0; row < m_rows; row++) {
    int num_nonzero_chunks = *ledger_ptr;
    ledger_ptr++;
    const uint8* ledger_start = ledger_ptr;
    const uint8* ledger_end = ledger_ptr + num_nonzero_chunks;
    const int8* mat_start = mat_ptr;

    for (int batch = 0; batch < n_batch; batch++) {
      const int8* vec_ptr = vectors + (batch * m_cols);
      int64_t row_sum = 0;

      mat_ptr = mat_start;
      ledger_ptr = ledger_start;

      if (ledger_ptr != ledger_end) {
        asm volatile(
            "dup v0.4s, wzr\n"
            "dup v1.4s, wzr\n"
            "dup v8.4s, wzr\n"
            "mov x7, 0\n"

            "1:\n"  // chunks_loop

            // Single matrix chunk, 16 bytes
            "ld1 {v8.16b}, [%[mat_ptr]], #16\n"

            // Read the next ledger index and increment.
            "ldrb w7, [%[ledger_ptr]], #1\n"

            // Read 16 bytes of vector data from (vec_ptr + (ledger_index * 16))
            "add x8, %[vec_ptr], x7, lsl #4\n"
            "ld1 {v9.16b}, [x8]\n"

            // Dot product of matrix row and vector.
            ".word 0x4e889520  // sdot v0.4s, v9.16b, v8.16b\n"

            "cmp %[ledger_ptr], %[ledger_end]\n"
            "blt 1b\n"  // chunks_loop

            // Sum the 4 vector components into a 32-bit value.
            "addv s1, v0.4s\n"
            // row_sum is 64-bit, so we copy 64 bits of v1 into it.
            // We have to be careful to cast this value to 32 bits in order
            // to interpret the sign bit properly.
            "mov %[row_sum], v1.d[0]\n"
            : [ row_sum ] "=r"(row_sum), [ ledger_ptr ] "+r"(ledger_ptr),
              [ mat_ptr ] "+r"(mat_ptr), [ vec_ptr ] "+r"(vec_ptr)
            : [ ledger_end ] "r"(ledger_end)
            : "x0", "x1", "x7", "x8", "v0", "v1", "v8", "v9", "cc", "memory");
      }
      result[(batch * m_rows + row) * result_stride] +=
          static_cast<int32>(row_sum) * scaling_factors[batch];
    }
  }
}

#endif  // __aarch64__

void NeonMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const int m_rows, const int m_cols,
    const int8_t* __restrict__ vectors, const float* scaling_factors,
    int n_batch, float* __restrict__ result, int result_stride) {
#ifdef __aarch64__
  if (HasSdotInstruction() && m_cols % 16 == 0 && m_rows % 2 == 0 &&
      m_rows >= n_batch) {
    if (n_batch % 4 == 0 && result_stride == 1) {
      // Benchmarks suggest that it's always better to use the batch code
      // when we can, even on small matrices.
      DotprodMatrixBatchFourVectorMultiplyAccumulate(
          matrix, m_rows, m_cols, vectors, scaling_factors, n_batch, result);
      return;
    }
  }
#endif  // __aarch64__

  const int kWeightsPerUint32 = 4;
  const int kWeightsPerNeonLane = 16;
  // Assuming *matrix is kWeightsPerUint32-byte aligned,
  // every row of the matrix is also
  // kWeightsPerUint32-byte aligned as long as cols is
  // a multiple of kWeightsPerUint32. The assumption
  // is currently satisfied by TFLite's 16-byte memory
  // alignment scheme.
  //
  // Otherwise, we allocate an aligned memory block and set
  // a flag to later copy rows from matrix to the block
  // for aligned multiplication.
  bool unaligned = false;
  int8_t* aligned_row = nullptr;
  void* aligned_row_free = nullptr;
  if ((m_cols & (kWeightsPerUint32 - 1)) != 0) {
    unaligned = true;
    aligned_row = (int8_t*)aligned_alloc(kWeightsPerUint32, m_cols,  // NOLINT
                                         &aligned_row_free);
  }
  void* aligned_vec_free = nullptr;
  int8_t* aligned_vec =
      (int8_t*)aligned_alloc(kWeightsPerUint32, m_cols,  // NOLINT
                             &aligned_vec_free);

  // If m_cols is not at least kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start = m_cols - (m_cols & (kWeightsPerNeonLane - 1));

  int batch, row, col;
  for (batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Copy the vector data to an aligned vector.
    memcpy(aligned_vec, vectors + batch * m_cols, sizeof(int8_t) * m_cols);
    // Compute dot-product for every column.
    for (row = 0; row < m_rows; ++row, result += result_stride) {
      // Get the address of the first element of the row.
      int8_t* row_ptr = (int8_t*)matrix + row * m_cols;  // NOLINT
      if (unaligned) {
        memcpy(aligned_row, row_ptr, sizeof(int8_t) * m_cols);
        row_ptr = aligned_row;
      }

      // Initialize the dot product sum for the row to 0.
      int32x4_t dotprod = vmovq_n_s32(0);

      // Prefetch the row to cache.
      __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                         3 /* temporal locality */);

      // For every block of 16 8-bit elements.
      col = 0;
      for (; col < postamble_start; col += kWeightsPerNeonLane) {
        // Load 16 8-bit values from the row and vector, each, to operate on.
        // Here the assumption is that each buffer is 4-byte aligned. Otherwise,
        // performance may suffer significantly.
        TFLITE_DCHECK_EQ(  // NOLINT
            (uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1), 0);
        const int8x16_t s1_8x16 = vld1q_s8((const int8_t*)(aligned_vec + col));
        const int8x16_t s2_8x16 = vld1q_s8((const int8_t*)(row_ptr + col));
        // Multiply the low bits (i.e. the lower 8 8bit numbers in the
        // registers).
        int16x8_t prod_16x8 =
            vmull_s8(vget_low_s8(s1_8x16), vget_low_s8(s2_8x16));
        // Multiply the high bits (i.e. the higher 8 8bit numbers in the
        // registers), and accumulate with the result of the low bits product.
        // The assumption here is that overflow will not happen as we quantize
        // our values to be in the range [-127, 127]. As such the sum of the 2
        // products is always strictly smaller than 15-bits (32767 in absolute
        // value).
        prod_16x8 =
            vmlal_s8(prod_16x8, vget_high_s8(s1_8x16), vget_high_s8(s2_8x16));

        dotprod = vpadalq_s16(dotprod, prod_16x8);
      }  // for col

      int32 postable_sum = 0;
      // Postamble loop.
      // TODO(raziel): if (ABSL_PREDICT_FALSE(postamble_start < m_rows))
      if (postamble_start < m_cols) {
        col = postamble_start;
        if ((m_cols - postamble_start) >= (kWeightsPerNeonLane >> 1)) {
          // Load 8 8-bit values from the row and column each to operate on.
          // Here the assumption is that each buffer is 4-bytes aligned.
          // Otherwise, performance may suffer significantly.
          TFLITE_DCHECK_EQ(  // NOLINT
              (uintptr_t)(&row_ptr[col]) & (kWeightsPerUint32 - 1), 0);
          const int8x8_t s1_8x8 = vld1_s8((const int8_t*)(aligned_vec + col));
          const int8x8_t s2_8x8 = vld1_s8((const int8_t*)(row_ptr + col));
          const int16x8_t prod_16x8 = vmull_s8(s1_8x8, s2_8x8);
          dotprod = vpadalq_s16(dotprod, prod_16x8);
          col += (kWeightsPerNeonLane >> 1);
        }
        for (; col < m_cols; ++col) {
          postable_sum += row_ptr[col] * aligned_vec[col];
        }  // for col
      }
      // Add the 4 intermediate sum values to get the final dot-prod value for
      // this row.
      int64x2_t pairwiseAdded = vpaddlq_s32(dotprod);
      int32 neon_sum =
          vgetq_lane_s64(pairwiseAdded, 0) + vgetq_lane_s64(pairwiseAdded, 1);

      *result += ((neon_sum + postable_sum) * batch_scaling_factor);
    }  // for row
  }    // for batch

  if (unaligned) {
    free(aligned_row_free);
  }
  free(aligned_vec_free);
}

void NeonSparseMatrixBatchVectorMultiplyAccumulate(
    const float* matrix, const uint8_t* ledger, int m_rows, int m_cols,
    const float* vector, int n_batch, float* result, int result_stride) {
  const int kBlockSize = 16;
  const int kNeonLanesPerBlock = 4;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);

  float* result_in_batch = result;
  for (int b = 0; b < n_batch; b++) {
    const float* matrix_ptr = matrix;
    const uint8_t* ledger_ptr = ledger;
    for (int r = 0; r < m_rows; r++) {
      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        float32x4_t acc_32x4 = vmovq_n_f32(0.0);
        const float* vector_in_batch = vector + b * m_cols;

        for (int i = 0; i < num_nonzero_blocks; i++) {
          const int block_start_index = *ledger_ptr++ * kBlockSize;
          const float* vector_block_in_batch_ptr =
              vector_in_batch + block_start_index;

          for (int c = 0; c < kNeonLanesPerBlock; c++) {
            // Load 4 float values from the vector and matrix row.
            float32x4_t vector_f32x4 = vld1q_f32(vector_block_in_batch_ptr +
                                                 c * kFloatWeightsPerNeonLane);
            float32x4_t matrix_f32x4 =
                vld1q_f32(matrix_ptr + c * kFloatWeightsPerNeonLane);
            // Multiply the vector and matrix row and add to accumulator.
            acc_32x4 = vmlaq_f32(acc_32x4, matrix_f32x4, vector_f32x4);
          }
          matrix_ptr += kBlockSize;
        }
        *result_in_batch +=
            (vgetq_lane_f32(acc_32x4, 0) + vgetq_lane_f32(acc_32x4, 1) +
             vgetq_lane_f32(acc_32x4, 2) + vgetq_lane_f32(acc_32x4, 3));
      }
      result_in_batch += result_stride;
    }
  }
}

void NeonSparseMatrixBatchVectorMultiplyAccumulate(
    const int8_t* __restrict__ matrix, const uint8_t* ledger, const int m_rows,
    const int m_cols, const int8_t* __restrict__ vectors,
    const float* scaling_factors, int n_batch, float* __restrict__ result,
    int result_stride) {
#ifdef __aarch64__
  if (HasSdotInstruction() && m_cols % 16 == 0) {
    DotprodSparseMatrixBatchVectorMultiplyAccumulate(
        matrix, ledger, m_rows, m_cols, vectors, scaling_factors, n_batch,
        result, result_stride);
    return;
  }
#endif  // __aarch64__

  const int kWeightsPerUint32 = 4;
  const int kWeightsPerNeonLane = 16;
  const int kBlockSize = kWeightsPerNeonLane;
  TFLITE_DCHECK_EQ(  // NOLINT
      m_cols % kBlockSize, 0);
  void* aligned_vec_free = nullptr;
  int8_t* aligned_vec =
      (int8_t*)aligned_alloc(kWeightsPerUint32, m_cols,  // NOLINT
                             &aligned_vec_free);

  int batch, row;
  for (batch = 0; batch < n_batch; ++batch) {
    const float batch_scaling_factor = scaling_factors[batch];
    // Copy the vector data to an aligned vector.
    memcpy(aligned_vec, vectors + batch * m_cols, sizeof(int8) * m_cols);

    const uint8_t* ledger_ptr = ledger;
    const int8_t* row_ptr = matrix;
    for (row = 0; row < m_rows; ++row, result += result_stride) {
      // Initialize the dot product sum for the row to 0.
      int32x4_t dotprod = vmovq_n_s32(0);
      int num_nonzero_blocks = *ledger_ptr++;
      if (num_nonzero_blocks > 0) {
        // Prefetch the row to cache.
        __builtin_prefetch(row_ptr, 0 /* prefetch for read */,
                           3 /* temporal locality */);
        for (int i = 0; i < num_nonzero_blocks; i++) {
          const int col_index = *ledger_ptr++ * kBlockSize;
          // Load 16 8-bit values from the row and vector, each, to operate on.
          // Here the assumption is that each buffer is 4-byte aligned.
          // Otherwise, performance may suffer significantly.
          TFLITE_DCHECK_EQ(  // NOLINT
              (uintptr_t)(&row_ptr) & (kWeightsPerUint32 - 1), 0);
          const int8x16_t s1_8x16 =
              vld1q_s8((const int8_t*)(aligned_vec + col_index));
          const int8x16_t s2_8x16 = vld1q_s8((const int8_t*)(row_ptr));
          // Multiply the low bits (i.e. the lower 8 8bit numbers in the
          // registers).
          int16x8_t prod_16x8 =
              vmull_s8(vget_low_s8(s1_8x16), vget_low_s8(s2_8x16));
          // Multiply the high bits (i.e. the lower 8 8bit numbers in the
          // registers), and accumulate with the result of the low bits product.
          // The assumption here is that overflow will not happen as we quantize
          // our values to be in the range [-127, 127]. As such the sum of the 2
          // products is always strictly smaller than 15-bits (32767 in absolute
          // value).
          prod_16x8 =
              vmlal_s8(prod_16x8, vget_high_s8(s1_8x16), vget_high_s8(s2_8x16));

          dotprod = vpadalq_s16(dotprod, prod_16x8);
          row_ptr += kBlockSize;
        }
        // Add the 4 intermediate sum values to get the final dot-prod value for
        // this row.
        int64x2_t pairwiseAdded = vpaddlq_s32(dotprod);
        int32 neon_sum =
            vgetq_lane_s64(pairwiseAdded, 0) + vgetq_lane_s64(pairwiseAdded, 1);
        *result += neon_sum * batch_scaling_factor;
      }
    }  // for row
  }    // for batch
  free(aligned_vec_free);
}

void NeonVectorVectorCwiseProduct(const float* vector1, const float* vector2,
                                  int v_size, float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from vector1 and vector2.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    // Vector multiply 4 float
    float32x4_t mul_32x4 = vmulq_f32(v1_f32x4, v2_f32x4);
    // Save to result array.
    vst1q_f32(&result[v], mul_32x4);
  }
  for (int v = postamble_start; v < v_size; v++) {
    result[v] = vector1[v] * vector2[v];
  }
}

void NeonVectorVectorCwiseProductAccumulate(const float* vector1,
                                            const float* vector2, int v_size,
                                            float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from vector1 and vector2 and accumulator.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    float32x4_t acc_32x4 = vld1q_f32(result + v);
    // Vector multiply-accumulate 4 float
    acc_32x4 = vmlaq_f32(acc_32x4, v1_f32x4, v2_f32x4);
    // Save to result array.
    vst1q_f32(&result[v], acc_32x4);
  }
  for (int v = postamble_start; v < v_size; v++) {
    result[v] += vector1[v] * vector2[v];
  }
}

void NeonVectorBatchVectorCwiseProduct(const float* vector, int v_size,
                                       const float* batch_vector, int n_batch,
                                       float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
      // Load from memory to vectors.
      float32x4_t batch_vector_f32x4 = vld1q_f32(batch_vector + v);
      float32x4_t vector_f32x4 = vld1q_f32(vector + v);
      // Multiply.
      float32x4_t result_f32x4 = vmulq_f32(batch_vector_f32x4, vector_f32x4);
      // Store.
      vst1q_f32(result + v, result_f32x4);
    }
    // Postamble loop
    for (int v = postamble_start; v < v_size; v++) {
      result[v] = vector[v] * batch_vector[v];
    }
    // Update the pointers.
    result += v_size;
    batch_vector += v_size;
  }
}

void NeonVectorBatchVectorCwiseProductAccumulate(const float* vector,
                                                 int v_size,
                                                 const float* batch_vector,
                                                 int n_batch, float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  float* result_ptr = result;
  const float* batch_vector_ptr = batch_vector;
  for (int b = 0; b < n_batch; b++) {
    for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
      // Load from memory to vectors.
      float32x4_t result_f32x4 = vld1q_f32(result_ptr + v);
      float32x4_t batch_vector_f32x4 = vld1q_f32(batch_vector_ptr + v);
      float32x4_t vector_f32x4 = vld1q_f32(vector + v);
      // Multiply-accumulate.
      result_f32x4 = vmlaq_f32(result_f32x4, batch_vector_f32x4, vector_f32x4);
      // Store.
      vst1q_f32(result_ptr + v, result_f32x4);
    }
    // Postamble loop
    for (int v = postamble_start; v < v_size; v++) {
      result_ptr[v] += vector[v] * batch_vector_ptr[v];
    }
    // Update the pointers.
    result_ptr += v_size;
    batch_vector_ptr += v_size;
  }
}

void NeonSub1Vector(const float* vector, int v_size, float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  float32x4_t one_f32x4 = vmovq_n_f32(1.0);
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from the current pointers of the input column and
    // subtract from 1.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    float32x4_t result_f32x4 = vsubq_f32(one_f32x4, v_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  for (int v = postamble_start; v < v_size; v++) {
    result[v] = 1.0f - vector[v];
  }
}

bool NeonIsZeroVector(const float* vector, int v_size) {
  // If v_size is not divisible by kFloatWeightsPerNeonLane, we cannot
  // use the main vectorized loop, and we need to process sequentially.
  // postamble_start shows the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  const float32x4_t zero_x4_float = vmovq_n_f32(0.0f);
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    const float32x4_t i_x4_float = vld1q_f32(vector + v);
    uint32x4_t cmp_result = vceqq_f32(i_x4_float, zero_x4_float);
    if (vgetq_lane_u32(cmp_result, 0) == 0) return false;
    if (vgetq_lane_u32(cmp_result, 1) == 0) return false;
    if (vgetq_lane_u32(cmp_result, 2) == 0) return false;
    if (vgetq_lane_u32(cmp_result, 3) == 0) return false;
  }

  // Postamble loop
  for (int v = postamble_start; v < v_size; ++v) {
    if (vector[v] != 0.0) return false;
  }
  return true;
}

void NeonClipVector(const float* vector, int v_size, float abs_limit,
                    float* result) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));

  // Replicate abs_limit and -abs_limit in two vectors.
  const float32x4_t abs_limit_f32x4 = vmovq_n_f32(abs_limit);
  const float32x4_t neg_abs_limit_f32x4 = vmovq_n_f32(-abs_limit);

  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load from memory to vector.
    float32x4_t v_f32x4 = vld1q_f32(vector + v);
    // Clip between abs_limit and -abs_limit.
    float32x4_t result_f32x4 = vminq_f32(abs_limit_f32x4, v_f32x4);
    result_f32x4 = vmaxq_f32(neg_abs_limit_f32x4, result_f32x4);
    // Save to output.
    vst1q_f32(result + v, result_f32x4);
  }
  // Postamble loop.
  for (int v = postamble_start; v < v_size; v++) {
    result[v] = (abs_limit < vector[v]) ? abs_limit : vector[v];
    result[v] = (-abs_limit > result[v]) ? -abs_limit : result[v];
  }
}

void NeonVectorScalarMultiply(const int8_t* vector, const int v_size,
                              const float scale, float* result) {
  // Here the assumption is that each buffer is 4-byte aligned.
  const int kWeightsPerUint32 = 4;
  TFLITE_CHECK_EQ((intptr_t)(&vector[0]) & (kWeightsPerUint32 - 1), 0);
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int kWeightsPerNeonLane = 16;
  const int postamble_start = v_size - (v_size & (kWeightsPerNeonLane - 1));

  // Create a vector of 4 floats with the scale value.
  const float32x4_t scale_f32x4 = vdupq_n_f32(scale);
  int v = 0;
  for (; v < postamble_start; v += kWeightsPerNeonLane) {
    // Load int8 values, sixteen at a time.
    const int8x16_t v_i8x16 = vld1q_s8(vector + v);
    // Split it into two components of size eight.
    const int8x8_t v0_i8x8 = vget_low_s8(v_i8x16);
    const int8x8_t v1_i8x8 = vget_high_s8(v_i8x16);
    // Convert both components to int16 first.
    const int16x8_t v0_i16x8 = vmovl_s8(v0_i8x8);
    const int16x8_t v1_i16x8 = vmovl_s8(v1_i8x8);
    // Split each of them into two components each.
    const int16x4_t v0_i16x4 = vget_low_s16(v0_i16x8);
    const int16x4_t v1_i16x4 = vget_high_s16(v0_i16x8);
    const int16x4_t v2_i16x4 = vget_low_s16(v1_i16x8);
    const int16x4_t v3_i16x4 = vget_high_s16(v1_i16x8);
    // Convert these to int32 and then to float.
    float32x4_t v0_f32x4 = vcvtq_f32_s32(vmovl_s16(v0_i16x4));
    float32x4_t v1_f32x4 = vcvtq_f32_s32(vmovl_s16(v1_i16x4));
    float32x4_t v2_f32x4 = vcvtq_f32_s32(vmovl_s16(v2_i16x4));
    float32x4_t v3_f32x4 = vcvtq_f32_s32(vmovl_s16(v3_i16x4));
    // Vector multiply four floats at a time.
    v0_f32x4 = vmulq_f32(v0_f32x4, scale_f32x4);
    v1_f32x4 = vmulq_f32(v1_f32x4, scale_f32x4);
    v2_f32x4 = vmulq_f32(v2_f32x4, scale_f32x4);
    v3_f32x4 = vmulq_f32(v3_f32x4, scale_f32x4);
    // Store the results.
    vst1q_f32(result + v, v0_f32x4);
    vst1q_f32(result + v + 4, v1_f32x4);
    vst1q_f32(result + v + 8, v2_f32x4);
    vst1q_f32(result + v + 12, v3_f32x4);
  }

  if (v_size - postamble_start >= (kWeightsPerNeonLane >> 1)) {
    // Load eight int8 values, if there is at least eight remaining.
    const int8x8_t v_i8x8 = vld1_s8(vector + v);
    // Convert them to int16 first.
    const int16x8_t v_i16x8 = vmovl_s8(v_i8x8);
    // Split it into two components.
    const int16x4_t v0_i16x4 = vget_low_s16(v_i16x8);
    const int16x4_t v1_i16x4 = vget_high_s16(v_i16x8);
    // Convert the components two floats.
    float32x4_t v0_f32x4 = vcvtq_f32_s32(vmovl_s16(v0_i16x4));
    float32x4_t v1_f32x4 = vcvtq_f32_s32(vmovl_s16(v1_i16x4));
    // Vector multiply four floats at a time.
    v0_f32x4 = vmulq_f32(v0_f32x4, scale_f32x4);
    v1_f32x4 = vmulq_f32(v1_f32x4, scale_f32x4);
    // Store the results.
    vst1q_f32(result + v, v0_f32x4);
    vst1q_f32(result + v + 4, v1_f32x4);
    v += (kWeightsPerNeonLane >> 1);
  }

  // Postamble loop.
  for (; v < v_size; v++) {
    result[v] = scale * vector[v];
  }
}

void NeonSymmetricQuantizeFloats(const float* values, const int size,
                                 int8_t* quantized_values, float* min,
                                 float* max, float* scaling_factor) {
  // TODO(raziel): vectorize min/max calculation.
  auto minmax = std::minmax_element(values, values + size);
  *min = *minmax.first;
  *max = *minmax.second;
  const int kScale = 127;
  const float range = std::max(std::abs(*min), std::abs(*max));
  if (range == 0) {
    memset(quantized_values, 0, size * sizeof(int8_t));
    *scaling_factor = 1;
    return;
  }
  *scaling_factor = range / kScale;
  const float scaling_factor_inv = kScale / range;

  const int postamble_start =
      size - (size & (2 * kFloatWeightsPerNeonLane - 1));

  // Vectorized constants.
  const float32x4_t q_factor_f32x4 = vmovq_n_f32(scaling_factor_inv);
  const float32x4_t point5_f32x4 = vmovq_n_f32(0.5);
  const float32x4_t zero_f32x4 = vmovq_n_f32(0.0);
  const int32x4_t scale_i32x4 = vmovq_n_s32(kScale);
  const int32x4_t neg_scale_i32x4 = vmovq_n_s32(-kScale);

  for (int i = 0; i < postamble_start; i += 2 * kFloatWeightsPerNeonLane) {
    // Implements the vectorized version of the following:
    // const int32 quantized_value = static_cast<int32>(
    //    std::round(*scaling_factor * values[i]));
    // Since the vectorized round intrinsics (vrndqa_f32) is not supported
    // on all Neon flavors, we use the following method for rounding: if (x
    // < 0) (int)(x - 0.5) if (x >= 0) (int)(x + 0.5)
    float32x4_t value0_f32x4 = vld1q_f32(&values[i]);
    float32x4_t value1_f32x4 = vld1q_f32(&values[i + kFloatWeightsPerNeonLane]);
    float32x4_t mul0_f32x4 = vmulq_f32(value0_f32x4, q_factor_f32x4);
    float32x4_t mul1_f32x4 = vmulq_f32(value1_f32x4, q_factor_f32x4);

    int32x4_t cmp_with_zero0_ui32x4 =
        (int32x4_t)vcltq_f32(mul0_f32x4, zero_f32x4);  // NOLINT
    int32x4_t cmp_with_zero1_ui32x4 =
        (int32x4_t)vcltq_f32(mul1_f32x4, zero_f32x4);  // NOLINT

    float32x4_t cmp_with_zero0_f32x4 = vcvtq_f32_s32(cmp_with_zero0_ui32x4);
    float32x4_t cmp_with_zero1_f32x4 = vcvtq_f32_s32(cmp_with_zero1_ui32x4);
    cmp_with_zero0_f32x4 = vaddq_f32(cmp_with_zero0_f32x4, point5_f32x4);
    cmp_with_zero1_f32x4 = vaddq_f32(cmp_with_zero1_f32x4, point5_f32x4);

    mul0_f32x4 = vaddq_f32(mul0_f32x4, cmp_with_zero0_f32x4);
    mul1_f32x4 = vaddq_f32(mul1_f32x4, cmp_with_zero1_f32x4);

    int32x4_t f2i0_i32x4 = vcvtq_s32_f32(mul0_f32x4);
    int32x4_t f2i1_i32x4 = vcvtq_s32_f32(mul1_f32x4);

    // Implements the vectorized version of the folowing block:
    //  quantized_values[i] = std::min(kScale, std::max(-kScale,
    //  quantized_value));
    int32x4_t max0_i32x4 = vmaxq_s32(f2i0_i32x4, neg_scale_i32x4);
    int32x4_t max1_i32x4 = vmaxq_s32(f2i1_i32x4, neg_scale_i32x4);
    int32x4_t min0_i32x4 = vminq_s32(max0_i32x4, scale_i32x4);
    int32x4_t min1_i32x4 = vminq_s32(max1_i32x4, scale_i32x4);

    int16x4_t min0_16x4 = vmovn_s32(min0_i32x4);
    int16x4_t min1_16x4 = vmovn_s32(min1_i32x4);

    int16x8_t min_16x8 = vcombine_s16(min0_16x4, min1_16x4);
    int8x8_t min_s8x8 = vqmovn_s16(min_16x8);
    vst1_s8(&quantized_values[i], min_s8x8);
  }

  for (int i = postamble_start; i < size; ++i) {
    const int32 quantized_value =
        static_cast<int32>(TfLiteRound(scaling_factor_inv * values[i]));
    quantized_values[i] = std::min(kScale, std::max(-kScale, quantized_value));
  }
}

float NeonVectorVectorDotProduct(const float* vector1, const float* vector2,
                                 int v_size) {
  // If v_size is not divisible by kWeightsPerNeonLane, we cannot use the main
  // vectorized loop, and we need to process sequentially. postamble_start shows
  // the start index where this should happen.
  const int postamble_start =
      v_size - (v_size & (kFloatWeightsPerNeonLane - 1));
  float32x4_t acc_32x4 = vmovq_n_f32(0.0);
  for (int v = 0; v < postamble_start; v += kFloatWeightsPerNeonLane) {
    // Load 4 float values from vector1 and vector2 and accumulator.
    float32x4_t v1_f32x4 = vld1q_f32(vector1 + v);
    float32x4_t v2_f32x4 = vld1q_f32(vector2 + v);
    // Vector multiply-accumulate 4 float
    acc_32x4 = vmlaq_f32(acc_32x4, v1_f32x4, v2_f32x4);
  }

  float result = (vgetq_lane_f32(acc_32x4, 0) + vgetq_lane_f32(acc_32x4, 1) +
                  vgetq_lane_f32(acc_32x4, 2) + vgetq_lane_f32(acc_32x4, 3));
  // Postamble loop.
  for (int v = postamble_start; v < v_size; v++) {
    result += vector1[v] * vector2[v];
  }
  return result;
}

void NeonBatchVectorBatchVectorDotProduct(const float* vector1,
                                          const float* vector2, int v_size,
                                          int n_batch, float* result,
                                          int result_stride) {
  float* result_ptr = result;
  const float* vector1_ptr = vector1;
  const float* vector2_ptr = vector2;
  for (int b = 0; b < n_batch; b++) {
    *result_ptr = NeonVectorVectorDotProduct(vector1_ptr, vector2_ptr, v_size);
    vector1_ptr += v_size;
    vector2_ptr += v_size;
    result_ptr += result_stride;
  }
}

void NeonReductionSumVector(const float* input_vector, float* output_vector,
                            int output_size, int reduction_size) {
  const float* input_vector_ptr = input_vector;
  for (int o = 0; o < output_size; o++) {
    // If reduction_size is not divisible by kWeightsPerNeonLane, we cannot use
    // the main vectorized loop, and we need to process sequentially.
    // postamble_start shows the start index where this should happen.
    const int postamble_start =
        reduction_size - (reduction_size & (kFloatWeightsPerNeonLane - 1));
    float32x4_t sum_f32x4 = vmovq_n_f32(0.0);
    for (int r = 0; r < postamble_start; r += kFloatWeightsPerNeonLane) {
      float32x4_t v1_f32x4 = vld1q_f32(input_vector_ptr + r);
      sum_f32x4 = vaddq_f32(sum_f32x4, v1_f32x4);
    }
    output_vector[o] +=
        (vgetq_lane_f32(sum_f32x4, 0) + vgetq_lane_f32(sum_f32x4, 1) +
         vgetq_lane_f32(sum_f32x4, 2) + vgetq_lane_f32(sum_f32x4, 3));
    input_vector_ptr += postamble_start;

    // Postamble loop.
    for (int r = postamble_start; r < reduction_size; r++) {
      output_vector[o] += *input_vector_ptr++;
    }
  }
}

void NeonVectorShiftLeft(float* vector, int v_size, float shift_value) {
  // This variable keeps track of the next to the last index which is being
  // copied to make sure we are not out of the vector boundary.
  int last_index_copy = kFloatWeightsPerNeonLane;
  int current_index_copy = 0;
  while (last_index_copy < v_size) {
    float32x4_t v_f32x4 = vld1q_f32(vector + current_index_copy + 1);
    vst1q_f32(vector + current_index_copy, v_f32x4);
    current_index_copy += kFloatWeightsPerNeonLane;
    last_index_copy += kFloatWeightsPerNeonLane;
  }
  // Postamble loop.
  for (int i = current_index_copy; i < v_size - 1; i++) {
    vector[i] = vector[i + 1];
  }
  vector[v_size - 1] = shift_value;
}

}  // namespace tensor_utils
}  // namespace tflite

#endif  // USE_NEON
