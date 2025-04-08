#pragma once
#include <stdint.h>

// OpenAI change: Switched from `extern "C"` to `namespace f2reduce`.
namespace f2reduce {

/**
 * Converts a matrix over F_2 into row-reduced echelon form.
 *
 * The matrix should be in row-major format. The stride parameter specifies
 * the offset (in 64-bit words, *not* bytes!) between successive rows of the
 * matrix, and should obey the inequality:
 *
 *     64 |stride| >= cols
 *
 * i.e. that the rows occupy disjoint regions of memory. For best performance
 * the stride should be divisible by 16 words (128 bytes).
 *
 * We adopt 'little-endian' semantics: the element in row i and column j+64*k
 * of the matrix (zero-indexed) is given by (matrix[i * stride + k] >> j) & 1.
 *
 * The matrix is overwritten in place with its row-reduced echelon form.
 */
void inplace_rref_strided(uint64_t *matrix, uint64_t rows, uint64_t cols, uint64_t stride);

uint64_t get_recommended_stride(uint64_t cols);

}  // namespace f2reduce
