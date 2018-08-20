/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_BATCH_MATMUL_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_BATCH_MATMUL_OP_IMPL_H_

#define EIGEN_USE_THREADS

#include <vector>
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
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
#include "tensorflow/core/util/work_sharder.h"

#if GOOGLE_CUDA
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;
#ifdef TENSORFLOW_USE_SYCL
typedef Eigen::SyclDevice SYCLDevice;
#endif  // TENSORFLOW_USE_SYCL

namespace {

Eigen::IndexPair<Eigen::DenseIndex> ContractionDims(bool adj_x, bool adj_y) {
  if (!adj_x) {
    if (!adj_y) {
      return Eigen::IndexPair<Eigen::DenseIndex>(1, 0);
    } else {
      return Eigen::IndexPair<Eigen::DenseIndex>(1, 1);
    }
  } else {
    if (!adj_y) {
      return Eigen::IndexPair<Eigen::DenseIndex>(0, 0);
    } else {
      return Eigen::IndexPair<Eigen::DenseIndex>(0, 1);
    }
  }
}

// Parallel batch matmul kernel based on the multi-threaded tensor contraction
// in Eigen.
template <typename Scalar, bool IsComplex = true>
struct ParallelMatMulKernel {
  static void Conjugate(const OpKernelContext* context, Tensor* out) {
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();
    auto z = out->tensor<Scalar, 3>();
    z.device(d) = z.conjugate();
  }

  static void Run(const OpKernelContext* context, const Tensor& in_x,
                  const Tensor in_y, bool adj_x, bool adj_y, Tensor* out,
                  int start, int limit) {
    static_assert(IsComplex, "Complex type expected.");
    auto Tx = in_x.tensor<Scalar, 3>();
    auto Ty = in_y.tensor<Scalar, 3>();
    auto Tz = out->tensor<Scalar, 3>();
    // We use the identities
    //   conj(a) * conj(b) = conj(a * b)
    //   conj(a) * b = conj(a * conj(b))
    // to halve the number of cases. The final conjugation of the result is
    // done at the end of LaunchBatchMatMul<CPUDevice, Scalar>::Launch().
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x, adj_y);
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();
    for (int i = start; i < limit; ++i) {
      auto x = Tx.template chip<0>(i);
      auto z = Tz.template chip<0>(i);
      if (adj_x != adj_y) {
        auto y = Ty.template chip<0>(i).conjugate();
        z.device(d) = x.contract(y, contract_pairs);
      } else {
        auto y = Ty.template chip<0>(i);
        z.device(d) = x.contract(y, contract_pairs);
      }
    }
  }
};

// The Eigen contraction kernel used here is very large and slow to compile,
// so we partially specialize ParallelMatMulKernel for real types to avoid all
// but one of the instantiations.
template <typename Scalar>
struct ParallelMatMulKernel<Scalar, false> {
  static void Conjugate(const OpKernelContext* context, Tensor* out) {}

  static void Run(const OpKernelContext* context, const Tensor& in_x,
                  const Tensor& in_y, bool adj_x, bool adj_y, Tensor* out,
                  int start, int limit) {
    auto Tx = in_x.tensor<Scalar, 3>();
    auto Ty = in_y.tensor<Scalar, 3>();
    auto Tz = out->tensor<Scalar, 3>();
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x, adj_y);
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();
    for (int i = start; i < limit; ++i) {
      auto x = Tx.template chip<0>(i);
      auto y = Ty.template chip<0>(i);
      auto z = Tz.template chip<0>(i);
      z.device(d) = x.contract(y, contract_pairs);
    }
  }
};

// TODO(rmlarsen): Get rid of this when we have upstreamed improvements
// for matrix*vector and vector*matrix to Eigen's general matrix product.
template <typename Tx, typename Ty, typename Tz>
static void Multiply(bool adj_x, bool adj_y, Tx x, Ty y, Tz z) {
  if (!adj_x) {
    if (!adj_y) {
      z.noalias() = x * y;
    } else {
      z.noalias() = x * y.adjoint();
    }
  } else {
    if (!adj_y) {
      z.noalias() = x.adjoint() * y;
    } else {
      z.noalias() = x.adjoint() * y.adjoint();
    }
  }
}

// Sequential batch matmul kernel that calls the regular Eigen matmul.
// We prefer this over the tensor contraction because it performs
// better on vector-matrix and matrix-vector products.
template <typename Scalar>
struct SequentialMatMulKernel {
  using Matrix =
      Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  using ConstMatrixMap = Eigen::Map<const Matrix>;
  using MatrixMap = Eigen::Map<Matrix>;

  static ConstMatrixMap ConstTensorSliceToEigenMatrix(const Tensor& t,
                                                      int slice) {
    return ConstMatrixMap(
        t.flat<Scalar>().data() + slice * t.dim_size(1) * t.dim_size(2),
        t.dim_size(1), t.dim_size(2));
  }

  static MatrixMap TensorSliceToEigenMatrix(Tensor* t, int slice) {
    return MatrixMap(
        t->flat<Scalar>().data() + slice * t->dim_size(1) * t->dim_size(2),
        t->dim_size(1), t->dim_size(2));
  }

  static void Run(const Tensor& in_x, const Tensor& in_y, bool adj_x,
                  bool adj_y, Tensor* out, int start, int limit) {
    for (int i = start; i < limit; ++i) {
      auto x = ConstTensorSliceToEigenMatrix(in_x, i);
      auto y = ConstTensorSliceToEigenMatrix(in_y, i);
      auto z = TensorSliceToEigenMatrix(out, i);
      // TODO(rmlarsen): Get rid of the special casing here when we have
      // upstreamed improvements for matrix*vector and vector*matrix to
      // Eigen's general matrix product.
      if (!adj_x && x.rows() == 1) {
        Multiply(adj_x, adj_y, x.row(0), y, z);
      } else if (adj_x && x.cols() == 1) {
        Multiply(adj_x, adj_y, x.col(0), y, z);
      } else if (!adj_y && y.cols() == 1) {
        Multiply(adj_x, adj_y, x, y.col(0), z);
      } else if (adj_y && y.rows() == 1) {
        Multiply(adj_x, adj_y, x, y.row(0), z);
      } else {
        Multiply(adj_x, adj_y, x, y, z);
      }
    }
  }
};

}  // namespace

template <typename Device, typename Scalar>
struct LaunchBatchMatMul;

template <typename Scalar>
struct LaunchBatchMatMul<CPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y, Tensor* out) {
    typedef ParallelMatMulKernel<Scalar, Eigen::NumTraits<Scalar>::IsComplex>
        ParallelMatMulKernel;
    bool conjugate_result = false;

    // Number of matrix multiplies i.e. size of the batch.
    const int64 batch_size = in_x.dim_size(0);
    const int64 cost_per_unit =
        in_x.dim_size(1) * in_x.dim_size(2) * out->dim_size(2);
    const int64 small_dim = std::min(
        std::min(in_x.dim_size(1), in_x.dim_size(2)), out->dim_size(2));
    const int64 kMaxCostOuterParallelism = 128 * 128 * 256;  // heuristic.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    if (small_dim > 1 &&
        (batch_size == 1 || cost_per_unit > kMaxCostOuterParallelism)) {
      // Parallelize over inner dims.
      // For large matrix products it is counter-productive to parallelize
      // over the batch dimension.
      ParallelMatMulKernel::Run(context, in_x, in_y, adj_x, adj_y, out, 0,
                                batch_size);
      conjugate_result = adj_x;
    } else {
      // Parallelize over outer dims. For small matrices and large batches, it
      // is counter-productive to parallelize the inner matrix multiplies.
      Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
            cost_per_unit,
            [&in_x, &in_y, adj_x, adj_y, out](int start, int limit) {
              SequentialMatMulKernel<Scalar>::Run(in_x, in_y, adj_x, adj_y, out,
                                                  start, limit);
            });
    }
    if (conjugate_result) {
      // We used one of the identities
      //   conj(a) * conj(b) = conj(a * b)
      //   conj(a) * b = conj(a * conj(b))
      // above, we need to conjugate the final output. This is a
      // no-op for non-complex types.
      ParallelMatMulKernel::Conjugate(context, out);
    }
  }
};

#if GOOGLE_CUDA

namespace {
template <typename T>
se::DeviceMemory<T> AsDeviceMemory(const T* cuda_memory) {
  se::DeviceMemoryBase wrapped(const_cast<T*>(cuda_memory));
  se::DeviceMemory<T> typed(wrapped);
  return typed;
}

class CublasScratchAllocator : public se::ScratchAllocator {
 public:
  using Stream = se::Stream;
  using DeviceMemoryBytes = se::DeviceMemory<uint8>;

  CublasScratchAllocator(OpKernelContext* context) : context_(context) {}

  int64 GetMemoryLimitInBytes(Stream* stream) override { return -1; }

  se::port::StatusOr<DeviceMemoryBytes> AllocateBytes(
      Stream* stream, int64 byte_size) override {
    Tensor temporary_memory;

    Status allocation_status(context_->allocate_temp(
        DT_UINT8, TensorShape({byte_size}), &temporary_memory));
    if (!allocation_status.ok()) {
      return se::port::StatusOr<DeviceMemoryBytes>(
          DeviceMemoryBytes::MakeFromByteSize(nullptr, 0));
    }
    // Hold the reference of the allocated tensors until the end of the
    // allocator.
    allocated_tensors_.push_back(temporary_memory);
    return se::port::StatusOr<DeviceMemoryBytes>(
        DeviceMemoryBytes::MakeFromByteSize(
            temporary_memory.flat<uint8>().data(),
            temporary_memory.flat<uint8>().size()));
  }

 private:
  OpKernelContext* context_;
  std::vector<Tensor> allocated_tensors_;
};
}  // namespace

template <typename Scalar>
struct LaunchBatchMatMul<GPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y, Tensor* out) {
    constexpr se::blas::Transpose kTranspose =
        is_complex<Scalar>::value ? se::blas::Transpose::kConjugateTranspose
                                  : se::blas::Transpose::kTranspose;
    se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose,
                                   kTranspose};
    const uint64 m = in_x.dim_size(adj_x ? 2 : 1);
    const uint64 k = in_x.dim_size(adj_x ? 1 : 2);
    const uint64 n = in_y.dim_size(adj_y ? 1 : 2);
    const uint64 batch_size = in_x.dim_size(0);
    auto blas_transpose_a = trans[adj_x];
    auto blas_transpose_b = trans[adj_y];

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    typedef se::DeviceMemory<Scalar> DeviceMemoryType;
    std::vector<DeviceMemoryType> a_device_memory;
    std::vector<DeviceMemoryType> b_device_memory;
    std::vector<DeviceMemoryType> c_device_memory;
    std::vector<DeviceMemoryType*> a_ptrs;
    std::vector<DeviceMemoryType*> b_ptrs;
    std::vector<DeviceMemoryType*> c_ptrs;
    a_device_memory.reserve(batch_size);
    b_device_memory.reserve(batch_size);
    c_device_memory.reserve(batch_size);
    a_ptrs.reserve(batch_size);
    b_ptrs.reserve(batch_size);
    c_ptrs.reserve(batch_size);
    auto* a_base_ptr = in_x.template flat<Scalar>().data();
    auto* b_base_ptr = in_y.template flat<Scalar>().data();
    auto* c_base_ptr = out->template flat<Scalar>().data();
    for (int64 i = 0; i < batch_size; ++i) {
      a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
      b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
      c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());
    }

    typedef Scalar Coefficient;

    // Cublas does
    // C = A x B
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A', where ' stands for transpose (not adjoint).
    // TODO(yangzihao): Choose the best of the three strategies using autotune.
    if (batch_size == 1) {
      // This is a regular matrix*matrix or matrix*vector multiply. Avoid the
      // overhead of the scratch allocator and the batch interface.
      if (n == 1 &&
          blas_transpose_b != se::blas::Transpose::kConjugateTranspose &&
          blas_transpose_a != se::blas::Transpose::kConjugateTranspose) {
        // This is a matrix*vector multiply so use GEMV to compute A * b.
        // Here we are multiplying in the natural order, so we have to flip
        // the transposition flag to compensate for the tensor being stored
        // row-major. Since GEMV doesn't provide a way to just conjugate an
        // argument, we have to defer those cases to GEMM below.
        auto gemv_trans_a = blas_transpose_a == se::blas::Transpose::kTranspose
                                ? se::blas::Transpose::kNoTranspose
                                : se::blas::Transpose::kTranspose;
        bool blas_launch_status =
            stream
                ->ThenBlasGemv(gemv_trans_a, adj_x ? m : k, adj_x ? k : m,
                               static_cast<Coefficient>(1.0), *(a_ptrs[0]),
                               adj_x ? m : k, *(b_ptrs[0]), 1,
                               static_cast<Coefficient>(0.0), c_ptrs[0], 1)
                .ok();
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal(
              "Blas xGEMV launch failed : a.shape=", in_x.shape().DebugString(),
              ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
              ", k=", k));
        }
      } else {
        bool blas_launch_status =
            stream
                ->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k,
                               static_cast<Coefficient>(1.0), *(b_ptrs[0]),
                               adj_y ? k : n, *(a_ptrs[0]), adj_x ? m : k,
                               static_cast<Coefficient>(0.0), c_ptrs[0], n)
                .ok();
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal(
              "Blas xGEMM launch failed : a.shape=", in_x.shape().DebugString(),
              ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
              ", k=", k));
        }
      }
    } else {
      CublasScratchAllocator scratch_allocator(context);
      bool blas_launch_status =
          stream
              ->ThenBlasGemmBatchedWithScratch(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), b_ptrs, adj_y ? k : n, a_ptrs,
                  adj_x ? m : k, static_cast<Coefficient>(0.0), c_ptrs, n,
                  batch_size, &scratch_allocator)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMMBatched launch failed : a.shape=",
            in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    }
  }
};

template <>
struct LaunchBatchMatMul<GPUDevice, Eigen::half> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y, Tensor* out) {
    typedef Eigen::half Scalar;
    constexpr perftools::gputools::blas::Transpose kTranspose =
        is_complex<Scalar>::value
            ? perftools::gputools::blas::Transpose::kConjugateTranspose
            : perftools::gputools::blas::Transpose::kTranspose;
    perftools::gputools::blas::Transpose trans[] = {
        perftools::gputools::blas::Transpose::kNoTranspose, kTranspose};
    const uint64 m = in_x.dim_size(adj_x ? 2 : 1);
    const uint64 k = in_x.dim_size(adj_x ? 1 : 2);
    const uint64 n = in_y.dim_size(adj_y ? 1 : 2);
    const uint64 batch_size = in_x.dim_size(0);
    auto blas_transpose_a = trans[adj_x];
    auto blas_transpose_b = trans[adj_y];

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    typedef perftools::gputools::DeviceMemory<Scalar> DeviceMemoryType;
    std::vector<DeviceMemoryType> a_device_memory;
    std::vector<DeviceMemoryType> b_device_memory;
    std::vector<DeviceMemoryType> c_device_memory;
    std::vector<DeviceMemoryType*> a_ptrs;
    std::vector<DeviceMemoryType*> b_ptrs;
    std::vector<DeviceMemoryType*> c_ptrs;
    a_device_memory.reserve(batch_size);
    b_device_memory.reserve(batch_size);
    c_device_memory.reserve(batch_size);
    a_ptrs.reserve(batch_size);
    b_ptrs.reserve(batch_size);
    c_ptrs.reserve(batch_size);
    auto* a_base_ptr = in_x.template flat<Scalar>().data();
    auto* b_base_ptr = in_y.template flat<Scalar>().data();
    auto* c_base_ptr = out->template flat<Scalar>().data();
    for (int64 i = 0; i < batch_size; ++i) {
      a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
      b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
      c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());
    }

    typedef float Coefficient;

    // Cublas does
    // C = A x B
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A', where ' stands for transpose (not adjoint).
    // TODO(yangzihao): Choose the best of the three strategies using autotune.
    if (batch_size == 1) {
      // This is a regular matrix*matrix or matrix*vector multiply. Avoid the
      // overhead of the scratch allocator and the batch interface.
      // TODO(benbarsdell): Use fp16 Gemv if it becomes supported by CUBLAS
      bool blas_launch_status =
          stream
              ->ThenBlasGemm(blas_transpose_b, blas_transpose_a, n, m, k,
                             static_cast<Coefficient>(1.0), *(b_ptrs[0]),
                             adj_y ? k : n, *(a_ptrs[0]), adj_x ? m : k,
                             static_cast<Coefficient>(0.0), c_ptrs[0], n)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMM launch failed : a.shape=", in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k));
      }
    } else {
      CublasScratchAllocator scratch_allocator(context);
      bool blas_launch_status =
          stream
              ->ThenBlasGemmBatchedWithScratch(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), b_ptrs, adj_y ? k : n, a_ptrs,
                  adj_x ? m : k, static_cast<Coefficient>(0.0), c_ptrs, n,
                  batch_size, &scratch_allocator)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(
            errors::Internal("Blas xGEMMBatched launch failed : a.shape=",
                             in_x.shape().DebugString(), ", b.shape=",
                             in_y.shape().DebugString(), ", m=", m, ", n=", n,
                             ", k=", k, ", batch_size=", batch_size));
      }
    }
  }
};

#endif  // GOOGLE_CUDA

#ifdef TENSORFLOW_USE_SYCL
template <typename Scalar>
struct ParallelMatMulKernelSYCL {
  static void Run(const OpKernelContext* context, const Tensor& in_x,
                  const Tensor& in_y, bool adj_x, bool adj_y, Tensor* out,
                  int start, int limit) {
    auto Tx = in_x.tensor<Scalar, 3>();
    auto Ty = in_y.tensor<Scalar, 3>();
    auto Tz = out->tensor<Scalar, 3>();
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x, adj_y);
    auto d = context->eigen_sycl_device();
    for (int i = start; i < limit; ++i) {
      auto x = Tx.template chip<0>(i);
      auto y = Ty.template chip<0>(i);
      auto z = Tz.template chip<0>(i);
      z.device(d) = x.contract(y, contract_pairs);
    }
  }
};

template <typename Scalar>
struct LaunchBatchMatMul<SYCLDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y, Tensor* out) {
    // Number of matrix multiplies i.e. size of the batch.
    const int64 batch_size = in_x.dim_size(0);
    ParallelMatMulKernelSYCL<Scalar>::Run(context, in_x, in_y, adj_x, adj_y,
                                          out, 0, batch_size);
  }
};
#endif  // TENSORFLOW_USE_SYCL

template <typename Device, typename Scalar>
class BatchMatMul : public OpKernel {
 public:
  explicit BatchMatMul(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
    OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
  }

  virtual ~BatchMatMul() {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);
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
                  errors::InvalidArgument(
                      "In[0].dim(", i, ") and In[1].dim(", i,
                      ") must be the same: ", in0.shape().DebugString(), " vs ",
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
    Tensor* out = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, out_shape, &out));
    if (out->NumElements() == 0) {
      return;
    }
    if (in0.NumElements() == 0 || in1.NumElements() == 0) {
      functor::SetZeroFunctor<Device, Scalar> f;
      f(ctx->eigen_device<Device>(), out->flat<Scalar>());
      return;
    }
    Tensor out_reshaped;
    CHECK(out_reshaped.CopyFrom(*out, TensorShape({n, d0, d3})));
    LaunchBatchMatMul<Device, Scalar>::Launch(ctx, in0_reshaped, in1_reshaped,
                                              adj_x_, adj_y_, &out_reshaped);
  }

 private:
  bool adj_x_;
  bool adj_y_;
};

#define REGISTER_BATCH_MATMUL_CPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      BatchMatMul<CPUDevice, TYPE>)

#define REGISTER_BATCH_MATMUL_GPU(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BatchMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      BatchMatMul<GPUDevice, TYPE>)

#ifdef TENSORFLOW_USE_SYCL
#define REGISTER_BATCH_MATMUL_SYCL(TYPE)                                 \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("BatchMatMul").Device(DEVICE_SYCL).TypeConstraint<TYPE>("T"), \
      BatchMatMul<SYCLDevice, TYPE>)
#endif  // TENSORFLOW_USE_SYCL
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCH_MATMUL_OP_IMPL_H_
