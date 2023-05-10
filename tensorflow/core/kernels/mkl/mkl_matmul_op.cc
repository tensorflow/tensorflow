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

// See docs in ../ops/math_ops.cc.

// This file uses MKL CBLAS xGEMM for acceleration of TF Matrix-Matrix
// Multiplication (MatMul) operations.
// We currently register this kernel only for MKL supported data
// types (float, double, complex64, complex128). The macro INTEL_MKL is defined
// by the build system only when MKL is chosen as an option at configure stage
// and when it is undefined at build time, this file becomes an empty
// compilation unit

#if defined(INTEL_MKL) && !defined(ENABLE_ONEDNN_V3)

#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename T, bool USE_CUBLAS>
class MklMatMulOp : public OpKernel {
 public:
  explicit MklMatMulOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_a", &transpose_a_));
    OP_REQUIRES_OK(ctx, ctx->GetAttr("transpose_b", &transpose_b_));
  }

  void Compute(OpKernelContext* ctx) override {
    const Tensor& a = ctx->input(0);
    const Tensor& b = ctx->input(1);

    // Check that the dimensions of the two matrices are valid.
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(a.shape()),
                errors::InvalidArgument("In[0] ndims must be >= 2"));
    OP_REQUIRES(ctx, TensorShapeUtils::IsMatrix(b.shape()),
                errors::InvalidArgument("In[1] ndims must be >= 2"));
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dim_pair;
    dim_pair[0].first = transpose_a_ ? 0 : 1;
    dim_pair[0].second = transpose_b_ ? 1 : 0;

    int d1 = a.dim_size(dim_pair[0].first);
    int d2 = b.dim_size(dim_pair[0].second);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument("Matrix size-incompatible: In[0]: ",
                                        a.shape().DebugString(),
                                        ", In[1]: ", b.shape().DebugString()));
    int a_dim_remaining = 1 - dim_pair[0].first;
    int b_dim_remaining = 1 - dim_pair[0].second;
    TensorShape out_shape(
        {a.dim_size(a_dim_remaining), b.dim_size(b_dim_remaining)});
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));

    if (out->NumElements() == 0) {
      // If a has shape [0, x] or b has shape [x, 0], the output shape
      // is a 0-element matrix, so there is nothing to do.
      return;
    }

    if (a.NumElements() == 0 && b.NumElements() == 0) {
      // If a has shape [x, 0] and b has shape [0, y], the
      // output shape is [x, y] where x and y are non-zero, so we fill
      // the output with zeros.
      functor::SetZeroFunctor<Device, T> f;
      f(ctx->eigen_device<Device>(), out->flat<T>());
      return;
    }

    if (std::is_same<T, float>::value) {
      (void)SetFPMathMode();
    }

    const int64_t m = a.dim_size(1 - dim_pair[0].first);
    const int64_t k = a.dim_size(dim_pair[0].first);
    const int64_t n = b.dim_size(1 - dim_pair[0].second);
    bool transpose_a = dim_pair[0].first == 0;
    bool transpose_b = dim_pair[0].second == 1;

    auto a_ptr = (a.template flat<T>().data());
    auto b_ptr = (b.template flat<T>().data());
    auto c_ptr = (out->template flat<T>().data());

    MklBlasGemm(ctx, transpose_a, transpose_b, m, n, k, a_ptr,
                transpose_a ? m : k, b_ptr, transpose_b ? k : n, c_ptr, n);
  }

 private:
  bool transpose_a_;
  bool transpose_b_;
  // --------------------------------------------------------------------------
  //
  // @brief Matrix-Matrix Multiplication with FP32 tensors, a, b, c using CBLAS
  // interface. c = op(a) * op(b)
  //
  // @param transa  Specifies the form of op(a) used in MatMul. If transa is
  // true, then op(a) = a^T, otherwise op(a) = a
  //
  // @param transb  Specifies the form of op(b) used in MatMul. If transb is
  // true, then op(b) = b^T, otherwise op(b) = b
  //
  // @param m       Specifies the number of rows of the matrix op(a) and of the
  // matrix c. The value of m must be at least zero.
  //
  // @param n       Specifies the number of columns of the matrix op(b) and the
  // number of columns of the matrix c. The value of n must be at least zero.
  //
  // @param k       Specifies the number of columns of the matrix op(a) and the
  // number of rows of the matrix op(b)
  //
  // @param a       Address of matrix a
  //
  // @param lda     Leading dimension of 'a' matrix. This is set at calling site
  // depending on transa parameter. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows
  // lda = max(1,k) when transa is false, otherwise lda = max(1,m)
  //
  // @param b       Address of matrix b
  //
  // @param ldb     Leading dimension of 'b' matrix. This is set at calling site
  // depending on transb parameter. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows
  // ldb = max(1,n) when transb is false, otherwise ldb = max(1,k)
  //
  // @param c       Address of matrix c
  //
  // @param ldc     Leading dimension of 'c' matrix. Since TF uses row-major
  // layout, leading dimension is the stride between consecutive rows, max(1,n)
  //
  // --------------------------------------------------------------------------
  void MklBlasGemm(OpKernelContext* ctx, bool transa, bool transb,
                   const int64_t m, const int64_t n, const int64_t k,
                   const float* a, const int64_t lda, const float* b,
                   const int64_t ldb, float* c, const int64_t ldc) {
    // BLAS GEMM API defines Matrix Multiplication as c = alpha * op(a) * op(b)
    // + beta * c.
    // Since TF MatMul does not have parameters for alpha, beta, we set them to
    // 1.0 and 0.0 respectively.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    char char_transa = transa ? 'T' : 'N';
    char char_transb = transb ? 'T' : 'N';
    VLOG(2) << "MKL DNN SGEMM called";
#ifndef ENABLE_ONEDNN_OPENMP
    MklDnnThreadPool eigen_tp(ctx);
    // With threadpool , the runtime overhead is comparable to the kernel
    // execution for small kernel sizes. For such sizes, it may be better to run
    // the kernel single threaded. Here we are coming up with a cost model based
    // on L1 sizes. If we find that matrices are small enough, we will execute
    // single threaded. This may need tuning.
    if (ExecuteSingleThreadedGemm(m, n, k, sizeof(float))) {
      // For now, call single-threaded gemm.
      dnnl::threadpool_interop::sgemm(char_transa, char_transb, m, n, k, alpha,
                                      a, lda, b, ldb, beta, c, ldc, nullptr);
    } else {
      dnnl::threadpool_interop::sgemm(char_transa, char_transb, m, n, k, alpha,
                                      a, lda, b, ldb, beta, c, ldc, &eigen_tp);
    }
#else
    dnnl_sgemm(char_transa, char_transb, m, n, k, alpha, a, lda, b, ldb, beta,
               c, ldc);
#endif  // !ENABLE_ONEDNN_OPENMP
  }

  void MklBlasGemm(OpKernelContext* ctx, bool transa, bool transb,
                   const int64_t m, const int64_t n, const int64_t k,
                   const bfloat16* a, const int64_t lda, const bfloat16* b,
                   const int64_t ldb, bfloat16* c, const int64_t ldc) {
    const float alpha = 1.0f;
    const float beta = 0.0f;
    const int index_transa = transa ? 1 : 0;
    const int index_transb = transb ? 1 : 0;

    const char ftrans[] = {'N', 'T', 'C'};
    dnnl_gemm<bfloat16>(ftrans[index_transa], ftrans[index_transb], m, n, k,
                        alpha, a, lda, b, ldb, beta, c, ldc, ctx);
  }
};

// We do not want to use this kernel for aarch64 because the
// Arm Compute Library does not provide a BLAS SGEMM
// interface, which is what MklMatMulOp calls by default.
#ifndef DNNL_AARCH64_USE_ACL
#define REGISTER_CPU(T)                                   \
  REGISTER_KERNEL_BUILDER(                                \
      Name("_MklMatMul")                                  \
          .Device(DEVICE_CPU)                             \
          .TypeConstraint<T>("T")                         \
          .Label(mkl_op_registry::kMklNameChangeOpLabel), \
      MklMatMulOp<CPUDevice, T, false /* cublas, ignored for CPU */>);

// TODO(intel-tf): Consider template specialization when adding/removing
// additional types
TF_CALL_float(REGISTER_CPU);
TF_CALL_bfloat16(REGISTER_CPU);
#endif  // !DNNL_AARCH64_USE_ACL

}  // namespace tensorflow
#endif  // INTEL_MKL && !ENABLE_ONEDNN_V3
