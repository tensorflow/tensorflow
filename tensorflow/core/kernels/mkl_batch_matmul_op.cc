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
#include "tensorflow/core/kernels/batch_matmul_op_impl.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl_matmul_ops_common.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/mkl_util.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

//  The third parameter v2_bcast is set to true if we are using V2 otherwise
//  we set it to false.
template <typename Device, typename Scalar, bool v2_bcast>
class BatchMatMulMkl : public OpKernel {
 public:
  explicit BatchMatMulMkl(OpKernelConstruction* context)
      : OpKernel(context), eigen_batch_mm_v2_(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  virtual ~BatchMatMulMkl() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& lhs = ctx->input(0);
    const Tensor& rhs = ctx->input(1);

    if (!v2_bcast) {
      // Using V1, so check to make sure lhs and rhs dimensions are correct and
      // no broadcasting is needed.
      OP_REQUIRES(ctx, lhs.dims() == rhs.dims(),
                  errors::InvalidArgument("lhs and rhs has different ndims: ",
                                          lhs.shape().DebugString(), " vs. ",
                                          rhs.shape().DebugString()));
      const int ndims = lhs.dims();
      OP_REQUIRES(
          ctx, ndims >= 2,
          errors::InvalidArgument("lhs and rhs ndims must be >= 2: ", ndims));
      for (int i = 0; i < ndims - 2; ++i) {
        OP_REQUIRES(ctx, lhs.dim_size(i) == rhs.dim_size(i),
                    errors::InvalidArgument("lhs.dim(", i, ") and rhs.dim(", i,
                                            ") must be the same: ",
                                            lhs.shape().DebugString(), " vs ",
                                            rhs.shape().DebugString()));
      }
    } else {
      OP_REQUIRES(
          ctx, lhs.dims() >= 2,
          errors::InvalidArgument("In[0] ndims must be >= 2: ", lhs.dims()));
      OP_REQUIRES(
          ctx, rhs.dims() >= 2,
          errors::InvalidArgument("In[1] ndims must be >= 2: ", rhs.dims()));
    }

    // lhs and rhs can have different dimensions
    const int ndims_lhs = lhs.dims();
    const int ndims_rhs = rhs.dims();

    // Get broadcast info
    MatMulBCast bcast(lhs.shape().dim_sizes(), rhs.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            lhs.shape().DebugString(), " vs. ", rhs.shape().DebugString()));

#ifdef ENABLE_MKLDNN_THREADPOOL
    if (bcast.IsBroadcastingRequired()) {
      // Calling Eigen Kernel for broadcasting case and return. Eigen does
      // not have BF16 support, so we have to fail graciously in that case.
      eigen_batch_mm_v2_.Compute(ctx);
      return;
    }
#endif  // ENABLE_MKLDNN_THREADPOOL
    TensorShape out_shape = bcast.output_batch_shape();
    auto batch_size = bcast.output_batch_size();

    auto lhs_rows = lhs.dim_size(ndims_lhs - 2);
    auto lhs_cols = lhs.dim_size(ndims_lhs - 1);
    auto rhs_rows = rhs.dim_size(ndims_rhs - 2);
    auto rhs_cols = rhs.dim_size(ndims_rhs - 1);

    if (adj_x_) std::swap(lhs_rows, lhs_cols);
    if (adj_y_) std::swap(rhs_rows, rhs_cols);
    OP_REQUIRES(ctx, lhs_cols == rhs_rows,
                errors::InvalidArgument(
                    "lhs mismatch rhs shape: ", lhs_cols, " vs. ", rhs_rows,
                    ": ", lhs.shape().DebugString(), " ",
                    rhs.shape().DebugString(), " ", adj_x_, " ", adj_y_));

    out_shape.AddDim(lhs_rows);
    out_shape.AddDim(rhs_cols);

    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (lhs.NumElements() == 0 || rhs.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }

    auto rhs_reshaped = rhs.template flat_inner_dims<Scalar, 3>();
    auto lhs_reshaped = lhs.template flat_inner_dims<Scalar, 3>();
    auto out_reshaped = out->template flat_inner_dims<Scalar, 3>();
    const uint64 M = lhs_reshaped.dimension(adj_x_ ? 2 : 1);
    const uint64 K = lhs_reshaped.dimension(adj_x_ ? 1 : 2);
    const uint64 N = rhs_reshaped.dimension(adj_y_ ? 1 : 2);

    std::vector<MKL_INT> m_array(batch_size, M);
    std::vector<MKL_INT> n_array(batch_size, N);
    std::vector<MKL_INT> k_array(batch_size, K);
    std::vector<MKL_INT> lda_array(batch_size, adj_x_ ? M : K);
    std::vector<MKL_INT> ldb_array(batch_size, adj_y_ ? K : N);
    std::vector<MKL_INT> ldc_array(batch_size, N);
    std::vector<MKL_INT> group_size(1, batch_size);
    const Scalar* a = nullptr;
    const Scalar* b = nullptr;
    Scalar* c = nullptr;
    bool threadpool_enabled = false;
#ifdef ENABLE_MKLDNN_THREADPOOL
    threadpool_enabled = true;
#endif  // ENABLE_MKLDNN_THREADPOOL
    if (std::is_same<Scalar, bfloat16>::value || threadpool_enabled) {
      // DNNL bfloat16 API requires a, b, and c as pointers to tensors
      // represented as flat-byte array.
      a = &lhs_reshaped(0, 0, 0);
      b = &rhs_reshaped(0, 0, 0);
      OP_REQUIRES(ctx, !bcast.IsBroadcastingRequired(),
                  errors::Unimplemented("Broadcasting is not supported for "
                                        "_MklBatchMatMul yet."));
      c = &out_reshaped(0, 0, 0);
      // TODO(nhasabni): Use appropriate cast instead of passing addresses of
      // a,b and c.
      MklCblasGemmBatch(CblasRowMajor, adj_x_, adj_y_, m_array, n_array,
                        k_array, &a, lda_array, &b, ldb_array, &c, ldc_array, 1,
                        group_size, ctx);
    } else {
      std::vector<const Scalar*> a_array;
      std::vector<const Scalar*> b_array;
      std::vector<Scalar*> c_array;
      a_array.reserve(batch_size);
      b_array.reserve(batch_size);
      c_array.reserve(batch_size);

      if (!bcast.IsBroadcastingRequired()) {
        for (int64 i = 0; i < batch_size; i++) {
          a_array.push_back(&lhs_reshaped(i, 0, 0));
          b_array.push_back(&rhs_reshaped(i, 0, 0));
          c_array.push_back(&out_reshaped(i, 0, 0));
        }
      } else {
        // Broadcasting is needed, so get the mapping from flattened output
        // batch indices to x's and y's flattened batch indices.
        const std::vector<int64>& a_batch_indices = bcast.x_batch_indices();
        const std::vector<int64>& b_batch_indices = bcast.y_batch_indices();

        for (int64 i = 0; i < batch_size; i++) {
          a_array.push_back(&lhs_reshaped(a_batch_indices[i], 0, 0));
          b_array.push_back(&rhs_reshaped(b_batch_indices[i], 0, 0));
          c_array.push_back(&out_reshaped(i, 0, 0));
        }
      }

      // MKL CBLAS API requires a, b, and c as array of pointers, where each
      // pointer is to 2D matrix.
      MklCblasGemmBatch(CblasRowMajor, adj_x_, adj_y_, m_array, n_array,
                        k_array, &a_array[0], lda_array, &b_array[0], ldb_array,
                        &c_array[0], ldc_array, 1, group_size, ctx);
    }
  }

 private:
  bool adj_x_;
  bool adj_y_;
  BatchMatMulV2Op<CPUDevice, Scalar> eigen_batch_mm_v2_;

  void MklCblasGemmBatch(
      const CBLAS_LAYOUT Layout, const bool TransA, const bool TransB,
      const std::vector<MKL_INT>& M_Array, const std::vector<MKL_INT>& N_Array,
      const std::vector<MKL_INT>& K_Array, const float** A_Array,
      const std::vector<MKL_INT>& lda_Array, const float** B_Array,
      const std::vector<MKL_INT>& ldb_Array, float** C_Array,
      const std::vector<MKL_INT>& ldc_Array, const MKL_INT group_count,
      const std::vector<MKL_INT>& group_size, OpKernelContext* ctx) {
#ifndef ENABLE_MKLDNN_THREADPOOL
    std::vector<CBLAS_TRANSPOSE> TransA_Array(
        group_size[0], TransA ? CblasTrans : CblasNoTrans);
    std::vector<CBLAS_TRANSPOSE> TransB_Array(
        group_size[0], TransB ? CblasTrans : CblasNoTrans);
    std::vector<float> alpha_Array(group_size[0], 1.0);
    std::vector<float> beta_Array(group_size[0], 0.0);
    cblas_sgemm_batch(Layout, &TransA_Array[0], &TransB_Array[0], &M_Array[0],
                      &N_Array[0], &K_Array[0], &alpha_Array[0],
                      reinterpret_cast<const float**>(A_Array), &lda_Array[0],
                      reinterpret_cast<const float**>(B_Array), &ldb_Array[0],
                      &beta_Array[0], reinterpret_cast<float**>(C_Array),
                      &ldc_Array[0], group_count, &group_size[0]);
#else
    DCHECK(Layout == CblasRowMajor);
    std::vector<bool> TransA_Array(group_size[0], TransA);
    std::vector<bool> TransB_Array(group_size[0], TransB);
    std::vector<float> alpha_Array(group_size[0], 1.0);
    std::vector<float> beta_Array(group_size[0], 0.0);
    dnnl_gemm_batch<float>(TransA_Array, TransB_Array, M_Array, N_Array,
                           K_Array, alpha_Array, *A_Array, *B_Array, beta_Array,
                           *C_Array, group_count, group_size, ctx);
#endif  // !ENABLE_MKLDNN_THREADPOOL
  }
// BatchMatMul BFloat16 support only exists in DNNL 1.2 onwards.
#if defined(ENABLE_MKLDNN_V1) && defined(ENABLE_INTEL_MKL_BFLOAT16)
  void MklCblasGemmBatch(
      const CBLAS_LAYOUT Layout, const bool TransA, const bool TransB,
      const std::vector<MKL_INT>& M_Array, const std::vector<MKL_INT>& N_Array,
      const std::vector<MKL_INT>& K_Array, const bfloat16** A_Array,
      const std::vector<MKL_INT>& lda_Array, const bfloat16** B_Array,
      const std::vector<MKL_INT>& ldb_Array, bfloat16** C_Array,
      const std::vector<MKL_INT>& ldc_Array, const MKL_INT group_count,
      const std::vector<MKL_INT>& group_size, OpKernelContext* ctx) {
    DCHECK(Layout == CblasRowMajor);
    std::vector<bool> TransA_Array(group_size[0], TransA);
    std::vector<bool> TransB_Array(group_size[0], TransB);
    std::vector<float> alpha_Array(group_size[0], 1.0);
    std::vector<float> beta_Array(group_size[0], 0.0);
    // TODO(nhasabni): Remove *A when we pass a, b, and c correctly.
    // MKLDNN API does not require lda, ldb, and ldc.
    dnnl_gemm_batch<bfloat16>(
        TransA_Array, TransB_Array, M_Array, N_Array, K_Array, alpha_Array,
        *A_Array, *B_Array, beta_Array, *C_Array, group_count, group_size, ctx);
  }
#endif  // ENABLE_MKLDNN_V1 && ENABLE_INTEL_MKL_BFLOAT16
};

#define REGISTER_BATCH_MATMUL_MKL(TYPE)                                       \
  REGISTER_KERNEL_BUILDER(Name("_MklBatchMatMul")                             \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          BatchMatMulMkl<CPUDevice, TYPE, false>)

#define REGISTER_BATCH_MATMUL_MKL_V2(TYPE)                                    \
  REGISTER_KERNEL_BUILDER(Name("_MklBatchMatMulV2")                           \
                              .Device(DEVICE_CPU)                             \
                              .TypeConstraint<TYPE>("T")                      \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel), \
                          BatchMatMulMkl<CPUDevice, TYPE, true>)

#ifdef ENABLE_MKL
TF_CALL_float(REGISTER_BATCH_MATMUL_MKL_V2);
TF_CALL_float(REGISTER_BATCH_MATMUL_MKL);
#if defined(ENABLE_MKLDNN_V1) && defined(ENABLE_INTEL_MKL_BFLOAT16)
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL);
TF_CALL_bfloat16(REGISTER_BATCH_MATMUL_MKL_V2);
#endif  // ENABLE_MKLDNN_V1 && ENABLE_INTEL_MKL_BFLOAT16
#endif  // ENABLE_MKL

}  // end namespace tensorflow
#endif
