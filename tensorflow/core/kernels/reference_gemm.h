#ifndef TENSORFLOW_KERNELS_REFERENCE_GEMM_H_
#define TENSORFLOW_KERNELS_REFERENCE_GEMM_H_

// This is an unoptimized but debuggable implementation of the GEMM matrix
// multiply function, used to compare to faster but more opaque versions, or
// for bit depths or argument combinations that aren't supported by optimized
// code.
// It assumes the row-major convention used by TensorFlow, and implements
// C = A * B, like the standard BLAS GEMM interface. If the tranpose flags are
// true, then the relevant matrix is treated as stored in column-major order.

namespace tensorflow {
template <class T1, class T2, class T3>
void ReferenceGemm(bool transpose_a, bool transpose_b, bool transpose_c,
                   size_t m, size_t n, size_t k, const T1* a, T1 offset_a,
                   size_t lda, const T2* b, T2 offset_b, size_t ldb, T3* c,
                   int32 shift_c, int32 offset_c, int32 mult_c, size_t ldc) {
  int a_i_stride;
  int a_l_stride;
  if (transpose_a) {
    a_i_stride = 1;
    a_l_stride = lda;
  } else {
    a_i_stride = lda;
    a_l_stride = 1;
  }
  int b_j_stride;
  int b_l_stride;
  if (transpose_b) {
    b_j_stride = ldb;
    b_l_stride = 1;
  } else {
    b_j_stride = 1;
    b_l_stride = ldb;
  }
  int c_i_stride;
  int c_j_stride;
  if (transpose_c) {
    c_i_stride = 1;
    c_j_stride = ldc;
  } else {
    c_i_stride = ldc;
    c_j_stride = 1;
  }

  const int32 highest = static_cast<int32>(Eigen::NumTraits<T3>::highest());
  const int32 lowest = static_cast<int32>(Eigen::NumTraits<T3>::lowest());
  const int32 rounding = (shift_c < 1) ? 0 : (1 << (shift_c - 1));

  int i, j, l;
  for (j = 0; j < n; j++) {
    for (i = 0; i < m; i++) {
      int32 total = 0;
      for (l = 0; l < k; l++) {
        const size_t a_index = ((i * a_i_stride) + (l * a_l_stride));
        const int32 a_value = a[a_index] - offset_a;
        const size_t b_index = ((j * b_j_stride) + (l * b_l_stride));
        const int32 b_value = b[b_index] - offset_b;
        total += (a_value * b_value);
      }
      const size_t c_index = ((i * c_i_stride) + (j * c_j_stride));
      int32_t output = ((((total + offset_c) * mult_c) + rounding) >> shift_c);
      if (output > highest) {
        output = highest;
      }
      if (output < lowest) {
        output = lowest;
      }
      c[c_index] = static_cast<T3>(output);
    }
  }
}
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_REFERENCE_GEMM_H_
