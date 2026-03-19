/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#if defined(INTEL_MKL) && defined(ENABLE_ONEDNN_V3)

#define EIGEN_USE_THREADS

#include "Eigen/Core"
#include "Eigen/SparseCore"
#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/cwise_ops_common.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/kernels/mkl/mkl_matmul_ops_common.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/mat_mul_op.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/kernels/sparse/transpose_op.h"
#include "tensorflow/core/kernels/transpose_functor.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/threadpool.h"
#include "tensorflow/core/util/mkl_util.h"
#include "unsupported/Eigen/CXX11/Tensor"

using dnnl::stream;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

// Implements a kernel which, given a SparseMatrix `a` and dense Tensor `b`,
// computes a dense Tensor `c` satisfying `c = a * b` where * denotes matrix
// multiplication.
//
// The rank of both `a` and `b` must be equal and their shapes must be
// compatible for matrix multiplication. Otherwise, InvalidArgument runtime
// errors will be thrown. Only inputs of rank 2 are supported.
//
template <typename Device, typename T, bool native_format = false>
class MklSparseMatrixMatMulOp : public MklDnnMatMulOpBase<T, void, T> {
 private:
  tensorflow::CSRMatMulCPUOp<T> eigen_sparse_matmul_op_;

 public:
  explicit MklSparseMatrixMatMulOp(OpKernelConstruction* ctx)
      : MklDnnMatMulOpBase<T, void, T>(ctx), eigen_sparse_matmul_op_(ctx) {}

  // Throws errors if there are issues with the input.
  Status ValidateInputs(const CSRSparseMatrix& sparse_matrix_a,
                        const Tensor& dense_tensor_b, int* rank) {
    // Validate datatypes.
    if (sparse_matrix_a.dtype() != dense_tensor_b.dtype()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input types don't match.  a.dtype == ",
          DataTypeString(sparse_matrix_a.dtype()),
          " vs. b.dtype == ", DataTypeString(dense_tensor_b.dtype())));
    }

    // Validate the ranks.
    *rank = sparse_matrix_a.dims();
    if (*rank != dense_tensor_b.dims()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Ranks of a and b must match, saw: ", *rank, " vs. ",
                       dense_tensor_b.dims(), "."));
    }

    // Validate shapes.
    const auto& a_dense_shape = sparse_matrix_a.dense_shape().vec<int64_t>();
    const int64_t a_inner_dim = a_dense_shape(*rank - 1);
    const int64_t b_inner_dim = dense_tensor_b.dim_size(*rank - 2);
    if (a_inner_dim != b_inner_dim) {
      return absl::InvalidArgumentError(
          absl::StrCat("Inner product dimensions of A and B do not agree. ",
                       "Shapes are: ", TensorShape(a_dense_shape).DebugString(),
                       " vs. ", dense_tensor_b.shape().DebugString()));
    }

    Status s = sparse_matrix_a.Validate();
    return s;
  }

  // Determine if we should call the Eigen kernel as a fallback.
  bool ShouldCallEigenFallback(const CSRSparseMatrix& sparse_matrix_a,
                               const Tensor& dense_tensor_b, int rank) {
    if (sparse_matrix_a.dtype() != DT_FLOAT) {
      VLOG(1) << "sparse_matrix_a.dtype() is not DT_FLOAT";
      return true;
    }
    if (rank != 2) {
      VLOG(1) << "rank is not 2, but " << rank << " instead.";
      return true;
    }

    return false;
  }

  void Compute(OpKernelContext* ctx) override {
    // Try to catch any exceptions during the matmul itself.
    try {
      // Handle the input.
      const CSRSparseMatrix* sparse_matrix_a;
      OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &sparse_matrix_a));
      const Tensor& rhs_tensor = ctx->input(1);
      Tensor* output_tensor = nullptr;

      int rank;
      OP_REQUIRES_OK(ctx,
                     this->ValidateInputs(*sparse_matrix_a, rhs_tensor, &rank));

      const auto dense_shape = sparse_matrix_a->dense_shape().vec<int64_t>();
      if (ShouldCallEigenFallback(*sparse_matrix_a, rhs_tensor, rank)) {
        return eigen_sparse_matmul_op_.Compute(ctx);
        return;
      }

      // Dimensions of the matrices.
      int64_t num_lhs_rows = dense_shape(rank - 2);
      int64_t num_lhs_cols = dense_shape(rank - 1);
      int64_t num_rhs_rows = rhs_tensor.dim_size(rank - 2);
      int64_t num_rhs_cols = rhs_tensor.dim_size(rank - 1);
      memory::dims lhs_dims = memory::dims({num_lhs_rows, num_lhs_cols});
      memory::dims rhs_dims = memory::dims({num_rhs_rows, num_rhs_cols});
      memory::dims output_dims = memory::dims({num_lhs_rows, num_rhs_cols});

      // Choose the datatype.
      const float* lhs_data;
      dnnl::memory::data_type lhs_datatype;
      switch (sparse_matrix_a->dtype()) {
        case DT_FLOAT:
          lhs_data = sparse_matrix_a->values().flat<float>().data();
          lhs_datatype = dnnl::memory::data_type::f32;
          break;
        default:
          OP_REQUIRES(ctx, sparse_matrix_a->dtype() == DT_FLOAT,
                      absl::InvalidArgumentError(absl::StrCat(
                          "MklSparseMatrixMatMulOp got an unexpected data ",
                          "type for sparse-matrix input.")));
      }

      // Get the oneDNN primitive.
      string prefix = "sparsecsrmatmul";
      MklMatMulParams matmul_params(prefix, lhs_dims, rhs_dims, output_dims,
                                    dnnl::memory::dims(), dnnl::memory::dims(),
                                    dnnl::memory::dims(),
                                    sparse_matrix_a->total_nnz());
      MklMatMulPrimitive<T, T, T, true>* matmul_prim =
          MklMatMulPrimitiveFactory<T, T, T, T, true>::Get(matmul_params, 0);

      // Threading.
      auto st = ExecuteSingleThreadedGemm(num_lhs_rows, num_rhs_rows,
                                          num_rhs_cols, sizeof(T));
      Eigen::ThreadPoolInterface* eigen_interface =
          EigenThreadPoolFromTfContext(ctx);
      tsl::OneDnnThreadPool eigen_tp(eigen_interface,
                                     ThreadPoolUseCallerThread(), st ? 1 : -1);

      // Get the cached primitive.
      std::shared_ptr<dnnl::matmul::primitive_desc> matmul_pd =
          matmul_prim->GetPrimitiveDesc();

      // Allocate room for the result.
      TensorShape output_tf_shape({num_lhs_rows, num_rhs_cols});
      OP_REQUIRES_OK(ctx,
                     ctx->allocate_output(0, output_tf_shape, &output_tensor));

      T* rhs_data = const_cast<T*>(rhs_tensor.flat<T>().data());
      T* output_data = const_cast<T*>(output_tensor->flat<T>().data());
      MklDnnData<T> lhs_mkl(&(this->cpu_engine_));
      MklDnnData<T> rhs_mkl(&(this->cpu_engine_));

      // CPU stream.
      std::shared_ptr<stream> cpu_stream;
      cpu_stream.reset(CreateStream(&eigen_tp, matmul_prim->GetEngine()));

      // Allocate a scratchpad.
      UserScratchPad<unsigned char> scratch_pad;
      scratch_pad.AllocateSPTensor(matmul_prim, ctx);

      // Execute the actual matmul.
      matmul_prim->Execute(
          cpu_stream, lhs_data, rhs_data, output_data, scratch_pad.Get(),
          nullptr, nullptr,
          sparse_matrix_a->col_indices().flat<int32_t>().data(),
          sparse_matrix_a->row_pointers().flat<int32_t>().data());
    } catch (dnnl::error& e) {
      OP_REQUIRES_OK(
          ctx,
          absl::AbortedError(absl::StrCat(
              "Operation received an exception:", "Status: ",
              std::to_string(e.status), ", message: ", string(e.message),
              ", in file ", string(__FILE__), ":", std::to_string(__LINE__))));
    }
  }
};

#define REGISTER_CPU(T)                                                      \
  REGISTER_KERNEL_BUILDER(Name("_MklNativeSparseMatrixMatMul")               \
                              .Device(DEVICE_CPU)                            \
                              .Label(mkl_op_registry::kMklNameChangeOpLabel) \
                              .TypeConstraint<T>("T"),                       \
                          MklSparseMatrixMatMulOp<CPUDevice, T>);

REGISTER_CPU(float)

#undef REGISTER_CPU

}  // namespace tensorflow

#endif  // INTEL_MKL && ENABLE_ONEDNN_V3
