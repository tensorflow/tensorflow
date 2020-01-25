/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/tensor_util.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/dense_update_functor.h"
#include "tensorflow/core/kernels/sparse/kernels.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"
#include "tensorflow/core/kernels/fill_functor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/cuda_solvers.h"
#include "tensorflow/core/kernels/cuda_sparse.h"
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {
template <typename Device, typename T>
class CSRSparseMatrixAddFunctor {
 public:
  explicit CSRSparseMatrixAddFunctor(OpKernelContext* ctx, const T alpha,
                                     const T beta)
      : ctx_(ctx), alpha_(alpha), beta_(beta) {}

  Status operator()(const CSRSparseMatrix& a, const CSRSparseMatrix& b,
                    CSRSparseMatrix* c) {
    TensorShape a_tensor_shape;
    TensorShape b_tensor_shape;
    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(a.dense_shape().vec<int64>(),
                                                   &a_tensor_shape));
    TF_RETURN_IF_ERROR(TensorShapeUtils::MakeShape(b.dense_shape().vec<int64>(),
                                                   &b_tensor_shape));

    if (a_tensor_shape.dims() == 3) {
      if ((a_tensor_shape.dims() != b_tensor_shape.dims()) ||
          (a_tensor_shape.dim_size(0) != b_tensor_shape.dim_size(0))) {
        return errors::InvalidArgument(
            "Incompatible shapes of a and b, a.shape == ",
            a_tensor_shape.DebugString(),
            ", b.shape == ", b_tensor_shape.DebugString());
      }
    }
    const int rank = a_tensor_shape.dims();
    if ((a_tensor_shape.dim_size(rank - 2) !=
         b_tensor_shape.dim_size(rank - 2)) ||
        (a_tensor_shape.dim_size(rank - 1) !=
         b_tensor_shape.dim_size(rank - 1))) {
      return errors::InvalidArgument(
          "Incompatible shapes of a and b, a.shape == ",
          a_tensor_shape.DebugString(),
          ", b.shape == ", b_tensor_shape.DebugString());
    }

    const int batch_size = a.batch_size();

    // TODO(ebrevdo): Add support for broadcasting at least in the
    // batch dimension.
    auto a_dense_shape = a.dense_shape().vec<int64>();
    auto b_dense_shape = b.dense_shape().vec<int64>();
    Tensor c_dense_shape_t = a.dense_shape();

    const int64 rows = a_dense_shape((rank == 2) ? 0 : 1);

    functor::CSRSparseMatrixAdd<Device, T> csr_geam(ctx_, alpha_, beta_);
    TF_RETURN_IF_ERROR(csr_geam.Initialize());

    Tensor c_batch_ptr_t(cpu_allocator(), DT_INT32,
                         TensorShape({batch_size + 1}));
    auto c_batch_ptr = c_batch_ptr_t.vec<int32>();
    c_batch_ptr(0) = 0;

    Tensor c_row_ptr_t;
    TF_RETURN_IF_ERROR(ctx_->allocate_temp(
        DT_INT32, TensorShape({batch_size * (rows + 1)}), &c_row_ptr_t));
    auto c_row_ptr = c_row_ptr_t.vec<int32>();

    // Set the output row pointers to zero, in case we hit any empty
    // combinations of rows in a and b.
    functor::SetZeroFunctor<Device, int32> set_zero;
    const Device& d = ctx_->eigen_device<Device>();
    set_zero(d, c_row_ptr_t.flat<int32>());

    for (int i = 0; i < batch_size; ++i) {
      // Calculate output sizes for all minibatch entries.
      // Store in c_batch_ptr and update c_row_ptrs.
      if (a.nnz(i) == 0 && b.nnz(i) == 0) {
        c_batch_ptr(i + 1) = c_batch_ptr(i);
        continue;
      }
      ConstCSRComponent<T> a_comp{a.row_pointers_vec(i), a.col_indices_vec(i),
                                  a.values_vec<T>(i), a_dense_shape};
      ConstCSRComponent<T> b_comp{b.row_pointers_vec(i), b.col_indices_vec(i),
                                  b.values_vec<T>(i), b_dense_shape};
      TTypes<int32>::UnalignedVec c_row_ptr_i(&c_row_ptr(i * (rows + 1)),
                                              rows + 1);
      int c_nnz_i;
      TF_RETURN_IF_ERROR(
          csr_geam.GetOutputStructure(a_comp, b_comp, c_row_ptr_i, &c_nnz_i));
      c_batch_ptr(i + 1) = c_batch_ptr(i) + c_nnz_i;
    }

    Tensor c_col_ind_t;
    Tensor c_values_t;

    const int total_nnz = c_batch_ptr(batch_size);

    TF_RETURN_IF_ERROR(
        ctx_->allocate_temp(DT_INT32, TensorShape({total_nnz}), &c_col_ind_t));
    TF_RETURN_IF_ERROR(ctx_->allocate_temp(
        DataTypeToEnum<T>::value, TensorShape({total_nnz}), &c_values_t));
    TF_RETURN_IF_ERROR(CSRSparseMatrix::CreateCSRSparseMatrix(
        DataTypeToEnum<T>::value, c_dense_shape_t, c_batch_ptr_t, c_row_ptr_t,
        c_col_ind_t, c_values_t, c));

    for (int i = 0; i < batch_size; ++i) {
      if (a.nnz(i) == 0 && b.nnz(i) == 0) {
        // Setting of c_row_pointers_vec(i) == 0 is already done.
        continue;
      }
      ConstCSRComponent<T> a_comp{a.row_pointers_vec(i), a.col_indices_vec(i),
                                  a.values_vec<T>(i), a_dense_shape};
      ConstCSRComponent<T> b_comp{b.row_pointers_vec(i), b.col_indices_vec(i),
                                  b.values_vec<T>(i), b_dense_shape};
      CSRComponent<T> c_comp{c->row_pointers_vec(i), c->col_indices_vec(i),
                             c->values_vec<T>(i), c_dense_shape_t.vec<int64>()};

      TF_RETURN_IF_ERROR(csr_geam.Compute(a_comp, b_comp, &c_comp));
    }

    return Status::OK();
  }

 private:
  OpKernelContext* ctx_;
  const T alpha_;
  const T beta_;
};

template <typename Device, typename T>
class CSRSparseMatrixSumFunctor : public CSRSparseMatrixAddFunctor<Device, T> {
 public:
  // Same as above, but with alpha = beta = 1.0, so C = 1.0 * A + 1.0 * B.
  explicit CSRSparseMatrixSumFunctor(OpKernelContext* ctx)
      : CSRSparseMatrixAddFunctor<Device, T>(ctx, 1, 1) {}
};

}  // namespace

template <typename Device, typename T>
class CSRAddOp : public OpKernel {
 public:
  explicit CSRAddOp(OpKernelConstruction* c) : OpKernel(c) {}

  void Compute(OpKernelContext* ctx) final {
    const CSRSparseMatrix* a_matrix;
    const CSRSparseMatrix* b_matrix;
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 0, &a_matrix));
    OP_REQUIRES_OK(ctx, ExtractVariantFromInput(ctx, 1, &b_matrix));

    OP_REQUIRES(
        ctx, a_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument("dtype of a is not equal to 'type': ",
                                DataTypeString(a_matrix->dtype()), " vs. ",
                                DataTypeString(DataTypeToEnum<T>::value)));
    OP_REQUIRES(
        ctx, b_matrix->dtype() == DataTypeToEnum<T>::value,
        errors::InvalidArgument("dtype of b is not equal to 'type': ",
                                DataTypeString(b_matrix->dtype()), " vs. ",
                                DataTypeString(DataTypeToEnum<T>::value)));

    const Tensor& alpha_t = ctx->input(2);
    const Tensor& beta_t = ctx->input(3);
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(alpha_t.shape()),
        errors::InvalidArgument("Expected alpha to be a scalar, saw shape: ",
                                alpha_t.shape().DebugString()));
    OP_REQUIRES(
        ctx, TensorShapeUtils::IsScalar(beta_t.shape()),
        errors::InvalidArgument("Expected beta to be a scalar, saw shape: ",
                                beta_t.shape().DebugString()));

    const T host_alpha = alpha_t.scalar<T>()();
    const T host_beta = beta_t.scalar<T>()();

    Tensor c_t(cpu_allocator(), DT_VARIANT, TensorShape({}));
    CSRSparseMatrix c_matrix;
    CSRSparseMatrixAddFunctor<Device, T> add_functor(ctx, host_alpha,
                                                     host_beta);
    OP_REQUIRES_OK(ctx, add_functor(*a_matrix, *b_matrix, &c_matrix));
    c_t.scalar<Variant>()() = std::move(c_matrix);
    ctx->set_output(0, c_t);
  }
};

#define REGISTER(DEV, T)                              \
  REGISTER_KERNEL_BUILDER(Name("SparseMatrixAdd")     \
                              .Device(DEVICE_##DEV)   \
                              .TypeConstraint<T>("T") \
                              .HostMemory("alpha")    \
                              .HostMemory("beta"),    \
                          CSRAddOp<DEV##Device, T>);

#if GOOGLE_CUDA

#define REGISTER_GPU(T) REGISTER(GPU, T)

REGISTER_GPU(float)
REGISTER_GPU(double)
#if GOOGLE_CUDA
REGISTER_GPU(complex64)
REGISTER_GPU(complex128)
#endif

#undef REGISTER_GPU

REGISTER_UNARY_VARIANT_BINARY_OP_FUNCTION(
    ADD_VARIANT_BINARY_OP, DEVICE_GPU, CSRSparseMatrix,
    (CSRSparseMatrixBinaryHelper<GPUDevice, CSRSparseMatrixSumFunctor>));

#endif  // GOOGLE_CUDA

#undef REGISTER

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
namespace functor {
template <typename T>
struct CSRSparseMatrixAdd<GPUDevice, T>
    : public CSRStructureModifyingFunctor<GPUDevice, T> {
  explicit CSRSparseMatrixAdd(OpKernelContext* ctx, const T alpha, const T beta)
      : ctx_(ctx),
        cuda_sparse_(ctx),
        alpha_(alpha),
        beta_(beta),
        initialized_(false) {}

  Status Initialize() {
    TF_RETURN_IF_ERROR(cuda_sparse_.Initialize());
    TF_RETURN_IF_ERROR(descrA_.Initialize());
    TF_RETURN_IF_ERROR(descrB_.Initialize());
    TF_RETURN_IF_ERROR(descrC_.Initialize());
    initialized_ = true;
    return Status::OK();
  }

  Status GetOutputStructure(const ConstCSRComponent<T>& a,
                            const ConstCSRComponent<T>& b,
                            TTypes<int32>::UnalignedVec c_row_ptr,
                            int* output_nnz) {
    DCHECK(initialized_);

    const int m = a.row_ptr.size() - 1;
    DCHECK_EQ(m, b.row_ptr.size() - 1);
    const int row_dim = a.dense_shape_host.size() == 2 ? 0 : 1;
    DCHECK_EQ(m, a.dense_shape_host(row_dim));
    DCHECK_EQ(m, b.dense_shape_host(row_dim));
    const int nnzA = a.col_ind.size();
    const int nnzB = b.col_ind.size();
    *output_nnz = -1;

    const int n = a.dense_shape_host(row_dim + 1);
    DCHECK_EQ(n, b.dense_shape_host(row_dim + 1));

    TF_RETURN_IF_ERROR(cuda_sparse_.CsrgeamNnz(
        m, n, descrA_.descr(), nnzA, a.row_ptr.data(), a.col_ind.data(),
        descrB_.descr(), nnzB, b.row_ptr.data(), b.col_ind.data(),
        descrC_.descr(), c_row_ptr.data(), output_nnz));

    if (*output_nnz < 0) {
      return errors::Internal(
          "CSRAdd: CsrgeamNnz returned nnzTotalDevHostPtr < 0: ", *output_nnz);
    }
    return Status::OK();
  }

  Status Compute(const ConstCSRComponent<T>& a, const ConstCSRComponent<T>& b,
                 CSRComponent<T>* c) {
    DCHECK(initialized_);

    const int m = a.row_ptr.size() - 1;
    DCHECK_EQ(m, b.row_ptr.size() - 1);
    const int row_dim = a.dense_shape_host.size() == 2 ? 0 : 1;
    DCHECK_EQ(m, a.dense_shape_host(row_dim));
    DCHECK_EQ(m, b.dense_shape_host(row_dim));
    const int nnzA = a.col_ind.size();
    const int nnzB = b.col_ind.size();

    const int n = a.dense_shape_host(row_dim + 1);
    DCHECK_EQ(n, b.dense_shape_host(row_dim + 1));

    // Adding alpha * a + beta * b.
    TF_RETURN_IF_ERROR(cuda_sparse_.Csrgeam(
        m, n, &alpha_, descrA_.descr(), nnzA, a.values.data(), a.row_ptr.data(),
        a.col_ind.data(), &beta_, descrB_.descr(), nnzB, b.values.data(),
        b.row_ptr.data(), b.col_ind.data(), descrC_.descr(), c->values.data(),
        c->row_ptr.data(), c->col_ind.data()));

    return Status::OK();
  }

 private:
  OpKernelContext* ctx_;
  GpuSparse cuda_sparse_;
  GpuSparseMatrixDescriptor descrA_;
  GpuSparseMatrixDescriptor descrB_;
  GpuSparseMatrixDescriptor descrC_;
  const T alpha_;
  const T beta_;
  bool initialized_;

  TF_DISALLOW_COPY_AND_ASSIGN(CSRSparseMatrixAdd);
};

}  // namespace functor

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

}  // namespace tensorflow
