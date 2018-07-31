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

// See docs in ../ops/array_ops.cc.

#if defined(INTEL_MKL) && !defined(DO_NOT_USE_ML)
#define EIGEN_USE_THREADS

#include "mkl_trans.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/kernels/transpose_op.h"

namespace tensorflow {

// output = TransposeOp(T<any> input, T<int32> perm) takes a tensor
// of type T and rank N, and a permutation of 0, 1, ..., N-1. It
// shuffles the dimensions of the input tensor according to permutation.
//
// Specifically, the returned tensor output meets the following condition:
// 1) output.dims() == input.dims();
// 2) output.dim_size(i) == input.dim_size(perm[i]);
// 3) output.tensor<T, N>(i_0, i_1, ..., i_N-1) ==
//      input.tensor<T, N>(j_0, j_1, ..., j_N-1),
//    where i_s == j_{perm[s]}
//
// REQUIRES: perm is a vector of int32.
// REQUIRES: input.dims() == perm.size().
// REQUIRES: perm is a permutation.

namespace {
template <typename T>
Status MKLTranspose2D(const char trans, const Tensor& in, Tensor* out);

// Documentation here: https://software.intel.com/en-us/node/520863
// Parameters: (ordering:row-major, operation:transpose, num_rows, num_cols,
//              alpha (for scaling), array, dist_bet_adjacent_cols/rows
//              (source), array, dist_bet_adjacent_cols/rows (dest))

#define INSTANTIATE(T, PREFIX)                                                \
  template <>                                                                 \
  Status MKLTranspose2D<T>(const char trans, const Tensor& in, Tensor* out) { \
    mkl_##PREFIX##omatcopy('R', trans, in.dim_size(0), in.dim_size(1), 1,     \
                           in.flat<T>().data(), in.dim_size(1),               \
                           out->flat<T>().data(), in.dim_size(0));            \
    return Status::OK();                                                      \
  }

INSTANTIATE(float, s)
INSTANTIATE(double, d)

#undef INSTANTIATE

template <>
Status MKLTranspose2D<complex64>(const char trans, const Tensor& in,
                                 Tensor* out) {
  const MKL_Complex8 alpha = {1.0f, 0.0f};
  mkl_comatcopy(
      'R', trans, in.dim_size(0), in.dim_size(1), alpha,
      reinterpret_cast<const MKL_Complex8*>(in.flat<complex64>().data()),
      in.dim_size(1),
      reinterpret_cast<MKL_Complex8*>(
          const_cast<complex64*>(out->flat<complex64>().data())),
      in.dim_size(0));
  return Status::OK();
}

template <>
Status MKLTranspose2D<complex128>(const char trans, const Tensor& in,
                                  Tensor* out) {
  const MKL_Complex16 alpha = {1.0, 0.0};
  mkl_zomatcopy(
      'R', trans, in.dim_size(0), in.dim_size(1), alpha,
      reinterpret_cast<const MKL_Complex16*>(in.flat<complex128>().data()),
      in.dim_size(1),
      reinterpret_cast<MKL_Complex16*>(
          const_cast<complex128*>(out->flat<complex128>().data())),
      in.dim_size(0));
  return Status::OK();
}

static const char kMKLTranspose = 'T';
static const char kMKLConjugateTranspose = 'C';

}  // namespace

Status MklTransposeCpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                      gtl::ArraySlice<int32> perm,
                                      Tensor* out) {
  if (in.dims() == 2) {
    if (perm[0] == 0 && perm[1] == 1) {
      return Status::OK();
    }
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTranspose2D<float>(kMKLTranspose, in, out);
      case DT_DOUBLE:
        return MKLTranspose2D<double>(kMKLTranspose, in, out);
      case DT_COMPLEX64:
        return MKLTranspose2D<complex64>(kMKLTranspose, in, out);
      case DT_COMPLEX128:
        return MKLTranspose2D<complex128>(kMKLTranspose, in, out);
      default:
        break;
    }
  }
  // Fallback to eigen if transpose parameters not supported by MKL
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
}

Status MklConjugateTransposeCpuOp::DoTranspose(OpKernelContext* ctx,
                                               const Tensor& in,
                                               gtl::ArraySlice<int32> perm,
                                               Tensor* out) {
  if (in.dims() == 2 && perm[0] == 1 && perm[1] == 0) {
    // TODO(rmlarsen): By setting lda and ldb, we could use the MKL kernels
    // for any transpose that can be reduced to swapping the last two
    // dimensions in a rank-3 tensor. We can even run each outer dimension in
    // a separate thread.
    switch (in.dtype()) {
      case DT_FLOAT:
        return MKLTranspose2D<float>(kMKLTranspose, in, out);
      case DT_DOUBLE:
        return MKLTranspose2D<double>(kMKLTranspose, in, out);
      case DT_COMPLEX64:
        return MKLTranspose2D<complex64>(kMKLConjugateTranspose, in, out);
      case DT_COMPLEX128:
        return MKLTranspose2D<complex128>(kMKLConjugateTranspose, in, out);
      default:
        break;
    }
  }
  // Fallback to eigen if transpose parameters not supported by MKL
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoConjugateTranspose(ctx->eigen_device<CPUDevice>(), in,
                                            perm, out);
}

}  // namespace tensorflow

#endif  // INTEL_MKL
