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

// This file uses MKL CBLAS batched xGEMM for acceleration of TF Batch
// Matrix-Matrix Multiplication (MatMul) operations.
// We currently register this kernel only for MKL supported data
// types (float, double, complex64, complex128). The macro INTEL_MKL is defined
// by the build system only when MKL is chosen as an option at configure stage
// and when it is undefined at build time, this file becomes an empty
// compilation unit

#define EIGEN_USE_THREADS

#if defined(INTEL_MKL)
#include <vector>
#include "mkl_cblas.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename Device, typename Scalar>
class BatchMatMulMkl : public OpKernel {
 public:
  explicit BatchMatMulMkl(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  virtual ~BatchMatMulMkl() {}

  void Compute(OpKernelContext *ctx) override {
    const Tensor &in0 = ctx->input(0);
    const Tensor &in1 = ctx->input(1);
    OP_REQUIRES(ctx, in0.dims() == in1.dims(),
                errors::InvalidArgument("In[0] and In[1] has different ndims: ",
                                        in0.shape().DebugString(), " vs. ",
                                        in1.shape().DebugString()));
    const int ndims = in0.dims();
    OP_REQUIRES(
        ctx, ndims >= 2,
        errors::InvalidArgument("In[0] and In[1] ndims must be >= 2: ", ndims));
    TensorShape out_shape;
    for (int i = 0; i < ndims - 2; ++i) {
      OP_REQUIRES(ctx, in0.dim_size(i) == in1.dim_size(i),
                  errors::InvalidArgument("In[0].dim(", i, ") and In[1].dim(",
                                          i, ") must be the same: ",
                                          in0.shape().DebugString(), " vs ",
                                          in1.shape().DebugString()));
      out_shape.AddDim(in0.dim_size(i));
    }
    auto n = (ndims == 2) ? 1 : out_shape.num_elements();
    auto d0 = in0.dim_size(ndims - 2);
    auto d1 = in0.dim_size(ndims - 1);
    Tensor in0_reshaped;
    CHECK(in0_reshaped.CopyFrom(in0, TensorShape({n, d0, d1})));
    auto d2 = in1.dim_size(ndims - 2);
    auto d3 = in1.dim_size(ndims - 1);
    Tensor in1_reshaped;
    CHECK(in1_reshaped.CopyFrom(in1, TensorShape({n, d2, d3})));
    if (adj_x_) std::swap(d0, d1);
    if (adj_y_) std::swap(d2, d3);
    OP_REQUIRES(ctx, d1 == d2,
                errors::InvalidArgument(
                    "In[0] mismatch In[1] shape: ", d1, " vs. ", d2, ": ",
                    in0.shape().DebugString(), " ", in1.shape().DebugString(),
                    " ", adj_x_, " ", adj_y_));
    out_shape.AddDim(d0);
    out_shape.AddDim(d3);
    Tensor *out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (in0.NumElements() == 0 || in1.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }

    const uint64 M = in0_reshaped.dim_size(adj_x_ ? 2 : 1);
    const uint64 K = in0_reshaped.dim_size(adj_x_ ? 1 : 2);
    const uint64 N = in1_reshaped.dim_size(adj_y_ ? 1 : 2);
    auto in0_ptr = (in0.template flat<Scalar>().data());
    auto in1_ptr = (in1.template flat<Scalar>().data());
    auto out_ptr = (out->template flat<Scalar>().data());
    std::vector<CBLAS_TRANSPOSE> transa_array;
    std::vector<CBLAS_TRANSPOSE> transb_array;
    std::vector<MKL_INT> m_array;
    std::vector<MKL_INT> n_array;
    std::vector<MKL_INT> k_array;
    std::vector<Scalar> alpha_array;
    std::vector<Scalar> beta_array;
    std::vector<const Scalar *> a_array;
    std::vector<const Scalar *> b_array;
    std::vector<Scalar *> c_array;
    std::vector<MKL_INT> lda_array;
    std::vector<MKL_INT> ldb_array;
    std::vector<MKL_INT> ldc_array;
    std::vector<MKL_INT> group_size;
    for (int64 i = 0; i < n; i++) {
      transa_array.push_back(adj_x_ ? CblasTrans : CblasNoTrans);
      transb_array.push_back(adj_y_ ? CblasTrans : CblasNoTrans);
      m_array.push_back(M);
      n_array.push_back(N);
      k_array.push_back(K);
      alpha_array.push_back(1.0);
      beta_array.push_back(0.0);
      a_array.push_back(in0_ptr + i * M * K);
      b_array.push_back(in1_ptr + i * K * N);
      c_array.push_back(out_ptr + i * M * N);
      lda_array.push_back(adj_x_ ? M : K);
      ldb_array.push_back(adj_y_ ? K : N);
      ldc_array.push_back(N);
    }
    group_size.push_back(n);
    MklCblasGemmBatch(CblasRowMajor, &transa_array[0], &transb_array[0],
                      &m_array[0], &n_array[0], &k_array[0], &alpha_array[0],
                      &a_array[0], &lda_array[0], &b_array[0], &ldb_array[0],
                      &beta_array[0], &c_array[0], &ldc_array[0], 1,
                      &group_size[0]);
  }

 private:
  bool adj_x_;
  bool adj_y_;

  void MklCblasGemmBatch(const CBLAS_LAYOUT Layout,
                         const CBLAS_TRANSPOSE *TransA_Array,
                         const CBLAS_TRANSPOSE *TransB_Array,
                         const MKL_INT *M_Array, const MKL_INT *N_Array,
                         const MKL_INT *K_Array, const float *alpha_Array,
                         const float **A_Array, const MKL_INT *lda_Array,
                         const float **B_Array, const MKL_INT *ldb_Array,
                         const float *beta_Array, float **C_Array,
                         const MKL_INT *ldc_Array, const MKL_INT group_count,
                         const MKL_INT *group_size) {
    cblas_sgemm_batch(Layout, TransA_Array, TransB_Array, M_Array, N_Array,
                      K_Array, alpha_Array, A_Array, lda_Array, B_Array,
                      ldb_Array, beta_Array, C_Array, ldc_Array, group_count,
                      group_size);
  }

  void MklCblasGemmBatch(const CBLAS_LAYOUT Layout,
                         const CBLAS_TRANSPOSE *TransA_Array,
                         const CBLAS_TRANSPOSE *TransB_Array,
                         const MKL_INT *M_Array, const MKL_INT *N_Array,
                         const MKL_INT *K_Array, const double *alpha_Array,
                         const double **A_Array, const MKL_INT *lda_Array,
                         const double **B_Array, const MKL_INT *ldb_Array,
                         const double *beta_Array, double **C_Array,
                         const MKL_INT *ldc_Array, const MKL_INT group_count,
                         const MKL_INT *group_size) {
    cblas_dgemm_batch(Layout, TransA_Array, TransB_Array, M_Array, N_Array,
                      K_Array, alpha_Array, A_Array, lda_Array, B_Array,
                      ldb_Array, beta_Array, C_Array, ldc_Array, group_count,
                      group_size);
  }
};

#define REGISTER_BATCH_MATMUL_MKL(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      BatchMatMulMkl<CPUDevice, TYPE>)

TF_CALL_float(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_double(REGISTER_BATCH_MATMUL_MKL);

}  // end namespace tensorflow
#endif
