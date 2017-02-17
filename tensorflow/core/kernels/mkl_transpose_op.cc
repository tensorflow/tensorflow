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

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include "tensorflow/core/kernels/transpose_op.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "third_party/mkl/include/mkl_trans.h"

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

Status MklTransposeCpuOp::DoTranspose(OpKernelContext* ctx, const Tensor& in,
                                      gtl::ArraySlice<int32> perm,
                                      Tensor* out) {
  if (in.dims() == 2 && in.dtype() == DT_FLOAT) {
    float* user_o = out->flat<float>().data();
    const float* user_i = in.flat<float>().data();

    // Documentation here: https://software.intel.com/en-us/node/520863
    // Parameters: (ordering:row-major, operation:transpose, num_rows, num_cols,
    //              alpha (for scaling), array, dist_bet_adjacent_cols/rows
    //              (source), array, dist_bet_adjacent_cols/rows (dest))
    mkl_somatcopy('R', 'T', in.dim_size(0), in.dim_size(1), 1,
                  user_i, in.dim_size(1),
                  user_o, in.dim_size(0));

    return Status::OK();
  }

  // Fallback to eigen if transpose parameters not supported by MKL
  typedef Eigen::ThreadPoolDevice CPUDevice;
  return ::tensorflow::DoTranspose(ctx->eigen_device<CPUDevice>(), in, perm,
                                   out);
} // MklTransposeCpuOp::DoTranspose
} // namespace tensorflow

#endif  // INTEL_MKL
