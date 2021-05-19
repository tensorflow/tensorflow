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

#ifndef TENSORFLOW_CORE_KERNELS_MATMUL_OP_IMPL_H_
#define TENSORFLOW_CORE_KERNELS_MATMUL_OP_IMPL_H_

#define EIGEN_USE_THREADS

#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/type_traits.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/fill_functor.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/matmul_autotune.h"
#include "tensorflow/core/util/matmul_bcast.h"
#include "tensorflow/core/util/work_sharder.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#if defined(TENSORFLOW_USE_CUSTOM_CONTRACTION_KERNEL)
#include "tensorflow/core/kernels/eigen_contraction_kernel.h"
#endif

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "tensorflow/core/kernels/gpu_utils.h"
#include "tensorflow/core/platform/stream_executor.h"
#include "tensorflow/stream_executor/matmul_util.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"  // For CUDA_VERSION
#endif

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace {

// Returns the pair of dimensions along which to perform Tensor contraction to
// emulate matrix multiplication.
// For matrix multiplication of 2D Tensors X and Y, X is contracted along
// second dimension and Y is contracted along the first dimension (if neither X
// nor Y is adjointed). The dimension to contract along is switched when any
// operand is adjointed.
// See http://en.wikipedia.org/wiki/Tensor_contraction
Eigen::IndexPair<Eigen::DenseIndex> ContractionDims(bool adj_x, bool adj_y) {
  return Eigen::IndexPair<Eigen::DenseIndex>(adj_x ? 0 : 1, adj_y ? 1 : 0);
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
                  const Tensor in_y, bool adj_x, bool adj_y, bool trans_x,
                  bool trans_y, const MatMulBCast& bcast, Tensor* out,
                  int batch_size) {
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
    contract_pairs[0] = ContractionDims(adj_x || trans_x, adj_y || trans_y);
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();

    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    // TODO(rmlarsen): Consider launching these contractions asynchronously.
    for (int64 i = 0; i < batch_size; ++i) {
      const int64 x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64 y_batch_index = should_bcast ? y_batch_indices[i] : i;

      auto x = Tx.template chip<0>(x_batch_index);
      auto z = Tz.template chip<0>(i);
      if (adj_x != adj_y) {
        auto y = Ty.template chip<0>(y_batch_index).conjugate();
        z.device(d) = x.contract(y, contract_pairs);
      } else {
        auto y = Ty.template chip<0>(y_batch_index);
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
                  const Tensor& in_y, bool adj_x, bool adj_y, bool trans_x,
                  bool trans_y, const MatMulBCast& bcast, Tensor* out,
                  int batch_size) {
    const bool should_bcast = bcast.IsBroadcastingRequired();
    const Eigen::ThreadPoolDevice d = context->eigen_cpu_device();
    Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> contract_pairs;
    contract_pairs[0] = ContractionDims(adj_x || trans_x, adj_y || trans_y);
    if (batch_size == 1 && !should_bcast) {
      auto Tx = in_x.flat_inner_dims<Scalar, 2>();
      auto Ty = in_y.flat_inner_dims<Scalar, 2>();
      auto Tz = out->flat_inner_dims<Scalar, 2>();
      Tz.device(d) = Tx.contract(Ty, contract_pairs);
    } else {
      auto Tx = in_x.tensor<Scalar, 3>();
      auto Ty = in_y.tensor<Scalar, 3>();
      auto Tz = out->tensor<Scalar, 3>();
      const auto& x_batch_indices = bcast.x_batch_indices();
      const auto& y_batch_indices = bcast.y_batch_indices();
      // TODO(rmlarsen): Consider launching these contractions asynchronously.
      for (int64 i = 0; i < batch_size; ++i) {
        const int64 x_batch_index = should_bcast ? x_batch_indices[i] : i;
        const int64 y_batch_index = should_bcast ? y_batch_indices[i] : i;
        auto x = Tx.template chip<0>(x_batch_index);
        auto y = Ty.template chip<0>(y_batch_index);
        auto z = Tz.template chip<0>(i);

        z.device(d) = x.contract(y, contract_pairs);
      }
    }
  }
};

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
                  bool adj_y, bool trans_x, bool trans_y,
                  const MatMulBCast& bcast, Tensor* out, int start, int limit) {
    const bool should_bcast = bcast.IsBroadcastingRequired();
    const auto& x_batch_indices = bcast.x_batch_indices();
    const auto& y_batch_indices = bcast.y_batch_indices();
    for (int64 i = start; i < limit; ++i) {
      const int64 x_batch_index = should_bcast ? x_batch_indices[i] : i;
      const int64 y_batch_index = should_bcast ? y_batch_indices[i] : i;
      auto x = ConstTensorSliceToEigenMatrix(in_x, x_batch_index);
      auto y = ConstTensorSliceToEigenMatrix(in_y, y_batch_index);
      auto z = TensorSliceToEigenMatrix(out, i);
      // Assume at most one of adj_x or trans_x is true. Similarly, for adj_y
      // and trans_y.
      if (!adj_x && !trans_x) {
        if (!adj_y && !trans_y) {
          z.noalias() = x * y;
        } else if (adj_y) {
          z.noalias() = x * y.adjoint();
        } else {  // trans_y == true
          z.noalias() = x * y.transpose();
        }
      } else if (adj_x) {
        if (!adj_y && !trans_y) {
          z.noalias() = x.adjoint() * y;
        } else if (adj_y) {
          z.noalias() = x.adjoint() * y.adjoint();
        } else {  // trans_y == true
          z.noalias() = x.adjoint() * y.transpose();
        }
      } else {  // trans_x == true
        if (!adj_y && !trans_y) {
          z.noalias() = x.transpose() * y;
        } else if (adj_y) {
          z.noalias() = x.transpose() * y.adjoint();
        } else {  // trans_y == true
          z.noalias() = x.transpose() * y.transpose();
        }
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
                     const Tensor& in_y, bool adj_x, bool adj_y, bool trans_x,
                     bool trans_y, const MatMulBCast& bcast, bool use_autotune,
                     Tensor* out) {
    typedef ParallelMatMulKernel<Scalar, Eigen::NumTraits<Scalar>::IsComplex>
        ParallelMatMulKernel;
    bool conjugate_result = false;

    // Number of matrix multiplies i.e. size of the batch.
    const int64 batch_size = bcast.output_batch_size();
    const int64 cost_per_unit =
        in_x.dim_size(1) * in_x.dim_size(2) * out->dim_size(2);
    const int64 small_dim = std::min(
        std::min(in_x.dim_size(1), in_x.dim_size(2)), out->dim_size(2));
    // NOTE(nikhilsarda): This heuristic is optimal in benchmarks as of
    // Jan 21, 2020.
    const int64 kMaxCostOuterParallelism = 128 * 128;  // heuristic.
    auto worker_threads = *(context->device()->tensorflow_cpu_worker_threads());
    // TODO(rmlarsen): Reconsider the heuristics now that we have asynchronous
    // evaluation in Eigen Tensor.
    if (small_dim > 1 &&
        (batch_size == 1 || cost_per_unit > kMaxCostOuterParallelism)) {
      // Parallelize over inner dims.
      // For large matrix products it is counter-productive to parallelize
      // over the batch dimension.
      ParallelMatMulKernel::Run(context, in_x, in_y, adj_x, adj_y, trans_x,
                                trans_y, bcast, out, batch_size);
      conjugate_result = adj_x;
    } else {
      // Parallelize over outer dims. For small matrices and large batches, it
      // is counter-productive to parallelize the inner matrix multiplies.
      Shard(worker_threads.num_threads, worker_threads.workers, batch_size,
            cost_per_unit,
            [&in_x, &in_y, adj_x, adj_y, trans_x, trans_y, &bcast, out](
                int start, int limit) {
              SequentialMatMulKernel<Scalar>::Run(in_x, in_y, adj_x, adj_y,
                                                  trans_x, trans_y, bcast, out,
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

template <typename Scalar>
struct LaunchBatchMatMul<GPUDevice, Scalar> {
  static void Launch(OpKernelContext* context, const Tensor& in_x,
                     const Tensor& in_y, bool adj_x, bool adj_y, bool trans_x,
                     bool trans_y, const MatMulBCast& bcast, bool use_autotune,
                     Tensor* out) {
    se::blas::Transpose trans[] = {se::blas::Transpose::kNoTranspose,
                                   se::blas::Transpose::kTranspose,
                                   se::blas::Transpose::kConjugateTranspose};
    const uint64 m = in_x.dim_size(adj_x || trans_x ? 2 : 1);
    const uint64 k = in_x.dim_size(adj_x || trans_x ? 1 : 2);
    const uint64 n = in_y.dim_size(adj_y || trans_y ? 1 : 2);
    const int64 batch_size = bcast.output_batch_size();
    auto blas_transpose_a = trans[adj_x ? 2 : (trans_x ? 1 : 0)];
    auto blas_transpose_b = trans[adj_y ? 2 : (trans_y ? 1 : 0)];

    auto* stream = context->op_device_context()->stream();
    OP_REQUIRES(context, stream, errors::Internal("No GPU stream available."));

    typedef se::DeviceMemory<Scalar> DeviceMemoryType;
    std::vector<DeviceMemoryType> a_device_memory;
    std::vector<DeviceMemoryType> b_device_memory;
    std::vector<DeviceMemoryType> c_device_memory;
    std::vector<DeviceMemoryType*> a_ptrs;
    std::vector<DeviceMemoryType*> b_ptrs;
    std::vector<DeviceMemoryType*> c_ptrs;
    a_device_memory.reserve(bcast.x_batch_size());
    b_device_memory.reserve(bcast.y_batch_size());
    c_device_memory.reserve(batch_size);
    a_ptrs.reserve(batch_size);
    b_ptrs.reserve(batch_size);
    c_ptrs.reserve(batch_size);
    auto* a_base_ptr = in_x.template flat<Scalar>().data();
    auto* b_base_ptr = in_y.template flat<Scalar>().data();
    auto* c_base_ptr = out->template flat<Scalar>().data();
    uint64 a_stride;
    uint64 b_stride;
    uint64 c_stride;

    typedef typename CoefficientType<Scalar>::type Coefficient;

    static const int64 max_scratch_size = GetBlasWorkspaceLimit(
        "TF_CUBLAS_WORKSPACE_LIMIT_IN_MB", 1LL << 32);  // 4GB by default

    // The BlasLtMatmul routines are only supported from CUDA 11.0 onward.
#if GOOGLE_CUDA && CUDA_VERSION >= 11000
    // LOG(INFO) << "x_batch_size() " << bcast.x_batch_size();
    // LOG(INFO) << "y_batch_size() " << bcast.y_batch_size();
    // LOG(INFO) << "out_batch_size() " << bcast.output_batch_size();
    bool is_full_broadcast =
        std::min(bcast.x_batch_size(), bcast.y_batch_size()) == 1;
    bool requires_mixed_broadcasting =
        bcast.IsBroadcastingRequired() && !is_full_broadcast;
    // LOG(INFO) << "is_full_broadcast? " << is_full_broadcast;
    // LOG(INFO) << "IsBroadcastingRequired()? " <<
    // bcast.IsBroadcastingRequired();
    if (!requires_mixed_broadcasting) {
      bool broadcast_a = bcast.x_batch_size() == 1;
      bool broadcast_b = bcast.y_batch_size() == 1;
      a_stride = broadcast_a ? 0 : m * k;
      b_stride = broadcast_b ? 0 : k * n;
      c_stride = m * n;
      a_device_memory.push_back(AsDeviceMemory(a_base_ptr));
      b_device_memory.push_back(AsDeviceMemory(b_base_ptr));
      c_device_memory.push_back(AsDeviceMemory(c_base_ptr));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());

      DataType dtype = DataTypeToEnum<Scalar>::value;
      bool allow_tf32 = tensor_float_32_execution_enabled();
      int device_id = stream->parent()->device_ordinal();
      BatchMatmulParameters matmul_parameters(
          trans_x, trans_y, adj_x, adj_y, m, n, k, batch_size, broadcast_a,
          broadcast_b, dtype, dtype, allow_tf32, device_id);

      static const int64 max_autotune_algorithm_count =
          MatmulMaxAutotuneAlgorithmCount();
      int max_algorithm_count = use_autotune ? max_autotune_algorithm_count : 1;

      const auto* plan_and_algorithms =
          BatchMatmulPlanMapSingleton::GetInstance()->Find(matmul_parameters);
      if (!plan_and_algorithms) {
        se::blas::DataType blas_dtype = se::blas::ToDataType<Scalar>::value;
        se::blas::ComputationType computation_type;
        OP_REQUIRES(
            context,
            GetBlasComputationType(dtype, allow_tf32, &computation_type),
            errors::Internal("Unsupported dtype for batched matmul"));

        se::blas::BlasLtMatmulPlanParams plan_params;
        plan_params.ab_type = blas_dtype;
        plan_params.c_type = blas_dtype;
        plan_params.computation_type = computation_type;
        plan_params.pointer_mode = se::blas::PointerMode::kHost;
        plan_params.epilogue = se::blas::Epilogue::kDefault;
        plan_params.transa = blas_transpose_b;
        plan_params.transb = blas_transpose_a;
        plan_params.m = n;
        plan_params.n = m;
        plan_params.k = k;
        plan_params.lda = in_y.dim_size(2);
        plan_params.ldb = in_x.dim_size(2);
        plan_params.ldc = n;
        plan_params.batch_count = batch_size;
        plan_params.stride_a = b_stride;
        plan_params.stride_b = a_stride;
        plan_params.stride_c = c_stride;

        // VLOG(4) << "plan_params.ab_type " << plan_params.ab_type
        //         << "plan_params.c_type " << plan_params.c_type
        //         << "plan_params.computation_type "
        //         << plan_params.computation_type
        VLOG(4) << "plan_params.transa " << (adj_y || trans_y)
                << "plan_params.transb " << (adj_x || trans_x)
                << "plan_params.m " << plan_params.m << "plan_params.n "
                << plan_params.n << "plan_params.k " << plan_params.k
                << "plan_params.lda " << plan_params.lda << "plan_params.ldb "
                << plan_params.ldb << "plan_params.ldc " << plan_params.ldc
                << "plan_params.batch_count " << plan_params.batch_count
                << "plan_params.stride_a " << plan_params.stride_a
                << "plan_params.stride_b " << plan_params.stride_b
                << "plan_params.stride_c " << plan_params.stride_c;
        auto status_or_plan =
            stream->parent()->CreateBlasLtMatmulPlan(plan_params);
        OP_REQUIRES(context, status_or_plan.ok(),
                    FromExecutorStatus(status_or_plan));
        std::unique_ptr<se::blas::IBlasLtMatmulPlan> plan =
            status_or_plan.ConsumeValueOrDie();

        auto status_or_algorithms = stream->parent()->GetBlasLtMatmulAlgorithms(
            plan.get(), max_scratch_size, max_algorithm_count);
        OP_REQUIRES(context, status_or_algorithms.ok(),
                    FromExecutorStatus(status_or_algorithms));
        auto algorithms = status_or_algorithms.ConsumeValueOrDie();

        plan_and_algorithms =
            BatchMatmulPlanMapSingleton::GetInstance()->Insert(
                matmul_parameters, {std::move(plan), std::move(algorithms)});
      }
      const auto& plan = plan_and_algorithms->plan;
      const auto& algorithms = plan_and_algorithms->algorithms;

      // The BlasLtMatmul routines (unlike BlasGemm, BlasGemmBatched etc.) take
      // alpha and beta with the same type as the matrices.
      Scalar alpha(1.0);
      Scalar beta(0.0);

      // Note that algorithm_config.algorithm() here is used to refer
      // to the index within the algorithms vector, not the algorithm
      // itself.
      se::blas::AlgorithmConfig algorithm_config(se::blas::kNoAlgorithm);
      if (max_algorithm_count == 1) {
        algorithm_config.set_algorithm(0);
      } else if (!AutoTuneBatchMatmul::GetInstance()->FindBasedOnScore(
                     matmul_parameters, &algorithm_config)) {
        VLOG(4) << "Autotuning BlasLtMatmul over " << algorithms.size()
                << " algorithms.";
        se::blas::ProfileResult best_result;
        se::blas::ProfileResult profile_result;
        // for (const auto& profile_algorithm : plan_and_algorithms->algorithms)
        // {
        for (size_t i = 0; i != algorithms.size(); ++i) {
          const auto& profile_algorithm = algorithms[i];
          // Create a new scratch allocator with every autotuning run so that
          // scratch space is deallocated between runs.
          BlasScratchAllocator scratch_allocator(max_scratch_size, context);
          // LOG(INFO) << "Calling BlasLtMatMul autorune->alg " << i;
          bool cublas_launch_status =
              stream
                  ->ThenBlasLtMatmul(plan.get(), alpha, *b_ptrs[0], *a_ptrs[0],
                                     beta, c_ptrs[0], &scratch_allocator,
                                     profile_algorithm.get(), /*bias = */ {},
                                     &profile_result)
                  .ok();

          VLOG(4) << "  Autotune algorithm " << i
                  << " result: " << profile_result.elapsed_time_in_ms()
                  << " ms, valid=" << profile_result.is_valid()
                  << ", workspace_size=" << profile_algorithm->workspace_size();

          if (cublas_launch_status && profile_result.is_valid() &&
              profile_result.elapsed_time_in_ms() <
                  best_result.elapsed_time_in_ms()) {
            best_result = profile_result;
          }
        }

        if (best_result.is_valid()) {
          algorithm_config.set_algorithm(best_result.algorithm());
        }
        // We make sure that each matmul parameter set only gets one pass of
        // autotune. If no algorithms works, we add kNoAlgorithm to the autotune
        // map.
        AutoTuneBatchMatmul::GetInstance()->InsertBasedOnScore(
            matmul_parameters, algorithm_config);
      }
      se::blas::AlgorithmType algorithm_idx = algorithm_config.algorithm();
      OP_REQUIRES(context,
                  0 <= algorithm_idx && algorithm_idx < algorithms.size(),
                  errors::Internal("Missing/invalid BatchMatmul algorithm"));
      const auto& algorithm = algorithms[algorithm_idx];
      BlasScratchAllocator scratch_allocator(max_scratch_size, context);
      // LOG(INFO) << "Calling BlasLtMatMul";
      VLOG(4) << "Calling BlasLtMatMul : a.shape=(" << bcast.x_batch_size()
              << ", " << in_x.dim_size(1) << ", " << in_x.dim_size(2)
              << "), b.shape=(" << bcast.y_batch_size() << ", "
              << in_y.dim_size(1) << ", " << in_y.dim_size(2) << "), m=" << m
              << ", n=" << n << ", k=" << k << ", batch_size=" << batch_size
              << "trans_x = " << trans_x << "trans_y = " << trans_y
              << "adj_x = " << adj_x << "adj_y = " << adj_y;
      bool cublas_launch_status =
          stream
              ->ThenBlasLtMatmul(plan.get(), alpha, *b_ptrs[0], *a_ptrs[0],
                                 beta, c_ptrs[0], &scratch_allocator,
                                 algorithm.get())
              .ok();
      if (!cublas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas batched matmul launch failed : a.shape=(",
            bcast.x_batch_size(), ", ", in_x.dim_size(0), ", ",
            in_x.dim_size(1), "), b.shape=(", bcast.y_batch_size(), ", ",
            in_y.dim_size(0), ", ", in_y.dim_size(1), "), m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    } else {  // requires mixed broadcasting
      const std::vector<int64>& a_batch_indices = bcast.x_batch_indices();
      const std::vector<int64>& b_batch_indices = bcast.y_batch_indices();
      for (int64 i = 0; i < bcast.x_batch_size(); ++i) {
        a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
      }
      for (int64 i = 0; i < bcast.y_batch_size(); ++i) {
        b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
      }
      for (int64 i = 0; i < batch_size; ++i) {
        c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
        a_ptrs.push_back(&a_device_memory[a_batch_indices[i]]);
        b_ptrs.push_back(&b_device_memory[b_batch_indices[i]]);
        c_ptrs.push_back(&c_device_memory.back());
      }

      BlasScratchAllocator scratch_allocator(max_scratch_size, context);
      bool blas_launch_status =
          stream
              ->ThenBlasGemmBatchedWithScratch(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), b_ptrs,
                  adj_y || trans_y ? k : n, a_ptrs, adj_x || trans_x ? m : k,
                  static_cast<Coefficient>(0.0), c_ptrs, n, batch_size,
                  &scratch_allocator)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMMBatched launch failed : a.shape=",
            in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    }
    return;
#else   // if not GOOGLE_CUDA or CUDA_VERSION < 11000
    bool is_full_broadcast =
        std::min(bcast.x_batch_size(), bcast.y_batch_size()) == 1;
    bool use_strided_batched =
        (!bcast.IsBroadcastingRequired() || is_full_broadcast) &&
        batch_size > 1;
    if (use_strided_batched) {
      a_stride = bcast.x_batch_size() != 1 ? m * k : 0;
      b_stride = bcast.y_batch_size() != 1 ? k * n : 0;
      c_stride = m * n;
      a_device_memory.push_back(AsDeviceMemory(a_base_ptr));
      b_device_memory.push_back(AsDeviceMemory(b_base_ptr));
      c_device_memory.push_back(AsDeviceMemory(c_base_ptr));
      a_ptrs.push_back(&a_device_memory.back());
      b_ptrs.push_back(&b_device_memory.back());
      c_ptrs.push_back(&c_device_memory.back());
    } else if (!bcast.IsBroadcastingRequired()) {
      for (int64 i = 0; i < batch_size; ++i) {
        a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
        b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
        c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
        a_ptrs.push_back(&a_device_memory.back());
        b_ptrs.push_back(&b_device_memory.back());
        c_ptrs.push_back(&c_device_memory.back());
      }
    } else {
      const std::vector<int64>& a_batch_indices = bcast.x_batch_indices();
      const std::vector<int64>& b_batch_indices = bcast.y_batch_indices();
      for (int64 i = 0; i < bcast.x_batch_size(); ++i) {
        a_device_memory.push_back(AsDeviceMemory(a_base_ptr + i * m * k));
      }
      for (int64 i = 0; i < bcast.y_batch_size(); ++i) {
        b_device_memory.push_back(AsDeviceMemory(b_base_ptr + i * k * n));
      }
      for (int64 i = 0; i < batch_size; ++i) {
        c_device_memory.push_back(AsDeviceMemory(c_base_ptr + i * m * n));
        a_ptrs.push_back(&a_device_memory[a_batch_indices[i]]);
        b_ptrs.push_back(&b_device_memory[b_batch_indices[i]]);
        c_ptrs.push_back(&c_device_memory.back());
      }
    }

    // Blas does
    // C = A x B
    // where A, B and C are assumed to be in column major.
    // We want the output to be in row-major, so we can compute
    // C' = B' x A', where ' stands for transpose (not adjoint).
    // TODO(yangzihao): Choose the best of the three strategies using autotune.
    if (batch_size == 1) {
      // This is a regular matrix*matrix or matrix*vector multiply. Avoid the
      // overhead of the scratch allocator and the batch interface.
      // Note that the GEMV call here does not support Eigen::half, so we do not
      // use this path in that case. A workaround is applied to the pointers
      // passed to the call itself to avoid compilation errors.
      if (!std::is_same<Scalar, Eigen::half>::value && n == 1 &&
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
        // Cast pointers as a workaround for GEMV not supporting Eigen::half
        // (this will never actually be executed for Eigen::half).
        typedef se::DeviceMemory<Coefficient> NonHalfDeviceMemoryType;
        NonHalfDeviceMemoryType a_ptr(*(a_ptrs[0]));
        NonHalfDeviceMemoryType b_ptr(*(b_ptrs[0]));
        NonHalfDeviceMemoryType c_ptr(*(c_ptrs[0]));
        bool blas_launch_status =
            stream
                ->ThenBlasGemv(gemv_trans_a, adj_x || trans_x ? m : k,
                               adj_x || trans_x ? k : m,
                               static_cast<Coefficient>(1.0), a_ptr,
                               adj_x || trans_x ? m : k, b_ptr, 1,
                               static_cast<Coefficient>(0.0), &c_ptr, 1)
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
                               adj_y || trans_y ? k : n, *(a_ptrs[0]),
                               adj_x || trans_x ? m : k,
                               static_cast<Coefficient>(0.0), c_ptrs[0], n)
                .ok();
        if (!blas_launch_status) {
          context->SetStatus(errors::Internal(
              "Blas xGEMM launch failed : a.shape=", in_x.shape().DebugString(),
              ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
              ", k=", k));
        }
      }
    } else if (use_strided_batched) {
      bool blas_launch_status =
          stream
              ->ThenBlasGemmStridedBatched(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), *b_ptrs[0],
                  adj_y || trans_y ? k : n, b_stride, *a_ptrs[0],
                  adj_x || trans_x ? m : k, a_stride,
                  static_cast<Coefficient>(0.0), c_ptrs[0], n, c_stride,
                  batch_size)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMMStridedBatched launch failed : a.shape=",
            in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    } else {
      BlasScratchAllocator scratch_allocator(max_scratch_size, context);
      bool blas_launch_status =
          stream
              ->ThenBlasGemmBatchedWithScratch(
                  blas_transpose_b, blas_transpose_a, n, m, k,
                  static_cast<Coefficient>(1.0), b_ptrs,
                  adj_y || trans_y ? k : n, a_ptrs, adj_x || trans_x ? m : k,
                  static_cast<Coefficient>(0.0), c_ptrs, n, batch_size,
                  &scratch_allocator)
              .ok();
      if (!blas_launch_status) {
        context->SetStatus(errors::Internal(
            "Blas xGEMMBatched launch failed : a.shape=",
            in_x.shape().DebugString(),
            ", b.shape=", in_y.shape().DebugString(), ", m=", m, ", n=", n,
            ", k=", k, ", batch_size=", batch_size));
      }
    }
#endif  // not GOOGLE_CUDA or CUDA_VERSION < 11000
  }
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM


template <typename Device, typename Scalar>
class BaseBatchMatMulOp : public OpKernel {
 public:
  explicit BaseBatchMatMulOp(OpKernelConstruction* context,
                             bool is_legacy_matmul)
      : OpKernel(context) {
    if (is_legacy_matmul) {
      // The old MatMul kernel has "transpose_a/transpose_b" attributes.
      OP_REQUIRES_OK(context, context->GetAttr("transpose_a", &trans_x_));
      OP_REQUIRES_OK(context, context->GetAttr("transpose_b", &trans_y_));
      adj_x_ = false;
      adj_y_ = false;
    } else {
      OP_REQUIRES_OK(context, context->GetAttr("adj_x", &adj_x_));
      OP_REQUIRES_OK(context, context->GetAttr("adj_y", &adj_y_));
      trans_x_ = false;
      trans_y_ = false;
    }
    use_autotune_ = MatmulAutotuneEnable();
  }

  ~BaseBatchMatMulOp() override {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& in0 = ctx->input(0);
    const Tensor& in1 = ctx->input(1);

    const Status s = ValidateInputTensors(ctx, in0, in1);
    if (!s.ok()) {
      ctx->SetStatus(s);
      return;
    }

    MatMulBCast bcast(in0.shape().dim_sizes(), in1.shape().dim_sizes());
    OP_REQUIRES(
        ctx, bcast.IsValid(),
        errors::InvalidArgument(
            "In[0] and In[1] must have compatible batch dimensions: ",
            in0.shape().DebugString(), " vs. ", in1.shape().DebugString()));

    TensorShape out_shape = bcast.output_batch_shape();
    auto batch_size = bcast.output_batch_size();
    auto d0 = in0.dim_size(in0.dims() - 2);
    auto d1 = in0.dim_size(in0.dims() - 1);
    Tensor in0_reshaped;
    OP_REQUIRES(
        ctx,
        in0_reshaped.CopyFrom(in0, TensorShape({bcast.x_batch_size(), d0, d1})),
        errors::Internal("Failed to reshape In[0] from ",
                         in0.shape().DebugString()));
    auto d2 = in1.dim_size(in1.dims() - 2);
    auto d3 = in1.dim_size(in1.dims() - 1);
    Tensor in1_reshaped;
    OP_REQUIRES(
        ctx,
        in1_reshaped.CopyFrom(in1, TensorShape({bcast.y_batch_size(), d2, d3})),
        errors::Internal("Failed to reshape In[1] from ",
                         in1.shape().DebugString()));
    if (adj_x_ || trans_x_) std::swap(d0, d1);
    if (adj_y_ || trans_y_) std::swap(d2, d3);
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
    OP_REQUIRES(ctx,
                out_reshaped.CopyFrom(*out, TensorShape({batch_size, d0, d3})),
                errors::Internal("Failed to reshape output from ",
                                 out->shape().DebugString()));
    // LOG(INFO) << "in0 " << in0.shape().DebugString();
    // LOG(INFO) << "in1 " << in1.shape().DebugString();
    // LOG(INFO) << "in0_reshaped " << in0_reshaped.shape().DebugString();
    // LOG(INFO) << "in1_reshaped " << in1_reshaped.shape().DebugString();
    // LOG(INFO) << "out_reshaped " << out_reshaped.shape().DebugString();
    if (std::is_same<Scalar, bfloat16>::value) {
      bool is_cpu = std::is_same<Device, CPUDevice>::value;
      OP_REQUIRES(ctx, is_cpu,
                  errors::Internal("bfloat16 matmul is not supported by GPU"));
      Tensor in0_reshaped_float, in1_reshaped_float, out_reshaped_float;
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, in0_reshaped.shape(),
                                             &in0_reshaped_float));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, in1_reshaped.shape(),
                                             &in1_reshaped_float));
      OP_REQUIRES_OK(ctx, ctx->allocate_temp(DT_FLOAT, out_reshaped.shape(),
                                             &out_reshaped_float));

      // TODO: Avoid extra copy to make bfloat16 matmul efficient on CPU.
      BFloat16ToFloat(in0_reshaped.flat<bfloat16>().data(),
                      in0_reshaped_float.flat<float>().data(),
                      in0_reshaped.NumElements());
      BFloat16ToFloat(in1_reshaped.flat<bfloat16>().data(),
                      in1_reshaped_float.flat<float>().data(),
                      in1_reshaped.NumElements());

      LaunchBatchMatMul<Device, float>::Launch(
          ctx, in0_reshaped_float, in1_reshaped_float, adj_x_, adj_y_, trans_x_,
          trans_y_, bcast, use_autotune_, &out_reshaped_float);
      FloatToBFloat16(out_reshaped_float.flat<float>().data(),
                      out_reshaped.flat<bfloat16>().data(), out->NumElements());
    } else {
      LaunchBatchMatMul<Device, Scalar>::Launch(
          ctx, in0_reshaped, in1_reshaped, adj_x_, adj_y_, trans_x_, trans_y_,
          bcast, use_autotune_, &out_reshaped);
    }
  }

 protected:
  virtual Status ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                                      const Tensor& in1) = 0;

 private:
  // TODO(171979567) Make the ops take both adj and transpose attributes.
  bool adj_x_;
  bool adj_y_;
  bool trans_x_;
  bool trans_y_;
  bool use_autotune_;
};

// BatchMatMul Op implementation which disallows broadcasting.
template <typename Device, typename Scalar, bool is_legacy_matmul = false>
class BatchMatMulOp : public BaseBatchMatMulOp<Device, Scalar> {
 public:
  explicit BatchMatMulOp(OpKernelConstruction* context)
      : BaseBatchMatMulOp<Device, Scalar>(context, is_legacy_matmul) {}

  ~BatchMatMulOp() override {}

 private:
  Status ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                              const Tensor& in1) override {
    // Disallow broadcasting support. Ensure that all batch dimensions of the
    // input tensors match.
    if (in0.dims() != in1.dims()) {
      return errors::InvalidArgument(
          "In[0] and In[1] has different ndims: ", in0.shape().DebugString(),
          " vs. ", in1.shape().DebugString());
    }
    const int ndims = in0.dims();
    if (is_legacy_matmul) {
      if (ndims != 2) {
        return errors::InvalidArgument("In[0] and In[1] ndims must be == 2: ",
                                       ndims);
      }
    } else {
      if (ndims < 2) {
        return errors::InvalidArgument("In[0] and In[1] ndims must be >= 2: ",
                                       ndims);
      }
      for (int i = 0; i < ndims - 2; ++i) {
        if (in0.dim_size(i) != in1.dim_size(i)) {
          return errors::InvalidArgument(
              "In[0].dim(", i, ") and In[1].dim(", i,
              ") must be the same: ", in0.shape().DebugString(), " vs ",
              in1.shape().DebugString());
        }
      }
    }
    return Status::OK();
  }
};

// BatchMatMul Op implementation with broadcasting support.
template <typename Device, typename Scalar>
class BatchMatMulV2Op : public BaseBatchMatMulOp<Device, Scalar> {
 public:
  explicit BatchMatMulV2Op(OpKernelConstruction* context)
      : BaseBatchMatMulOp<Device, Scalar>(context,
                                          /* is_legacy_matmul= */ false) {}

  ~BatchMatMulV2Op() override {}

 private:
  Status ValidateInputTensors(OpKernelContext* ctx, const Tensor& in0,
                              const Tensor& in1) override {
    // Enable broadcasting support. Validity of broadcasting is checked in
    // BaseBatchMatMulOp.
    if (in0.dims() < 2) {
      return errors::InvalidArgument("In[0] ndims must be >= 2: ", in0.dims());
    }
    if (in1.dims() < 2) {
      return errors::InvalidArgument("In[1] ndims must be >= 2: ", in1.dims());
    }
    return Status::OK();
  }
};

#define REGISTER_BATCH_MATMUL_CPU(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),   \
      BatchMatMulOp<CPUDevice, TYPE>);                                    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMulV2").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"), \
      BatchMatMulV2Op<CPUDevice, TYPE>);                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MatMul").Device(DEVICE_CPU).TypeConstraint<TYPE>("T"),        \
      BatchMatMulOp<CPUDevice, TYPE, /* is_legacy_matmul=*/true>)

#define REGISTER_BATCH_MATMUL_GPU(TYPE)                                   \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),   \
      BatchMatMulOp<GPUDevice, TYPE>);                                    \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("BatchMatMulV2").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"), \
      BatchMatMulV2Op<GPUDevice, TYPE>);                                  \
  REGISTER_KERNEL_BUILDER(                                                \
      Name("MatMul").Device(DEVICE_GPU).TypeConstraint<TYPE>("T"),        \
      BatchMatMulOp<GPUDevice, TYPE, /* is_legacy_matmul=*/true>)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_MATMUL_OP_IMPL_H_
