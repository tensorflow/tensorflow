/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_BLAS_LT_H_
#define TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_BLAS_LT_H_

#include <algorithm>
#include <memory>
#include <string>

#include "third_party/gpus/cuda/include/cublasLt.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/host_or_device_scalar.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace stream_executor {
namespace gpu {
class GpuExecutor;
}  // namespace gpu

namespace cuda {

class BlasLt {
 public:
  struct MatmulDescDestroyer {
    void operator()(cublasLtMatmulDesc_t matmul_desc) const {
      cublasLtMatmulDescDestroy(matmul_desc);
    }
  };
  struct LayoutDestroyer {
    void operator()(cublasLtMatrixLayout_t layout) const {
      cublasLtMatrixLayoutDestroy(layout);
    }
  };
  struct MatmulPreferenceDestroyer {
    void operator()(cublasLtMatmulPreference_t matmul_pref) const {
      cublasLtMatmulPreferenceDestroy(matmul_pref);
    }
  };
  using UniqueOpDesc =
      std::unique_ptr<std::remove_pointer<cublasLtMatmulDesc_t>::type,
                      MatmulDescDestroyer>;
  using UniqueLayoutDesc =
      std::unique_ptr<std::remove_pointer<cublasLtMatrixLayout_t>::type,
                      LayoutDestroyer>;
  using UniqueMatmulPreference =
      std::unique_ptr<std::remove_pointer<cublasLtMatmulPreference_t>::type,
                      MatmulPreferenceDestroyer>;

  template <typename T>
  using Owned =
      std::unique_ptr<std::remove_pointer_t<T>, cublasStatus_t (*)(T)>;

  enum class Epilogue {
    kDefault = 1,                   // No special postprocessing
    kReLU = 2,                      // Apply point-wise ReLU function
    kBias = 4,                      // Add broadcasted bias vector
    kBiasThenReLU = kBias | kReLU,  // Apply bias and then ReLU transform
  };

  // Describes the location of pointers for the scaling factors alpha and beta.
  enum class PointerMode {
    kHost,
    kDevice,
  };

  // Parameters for the CreateBlasLtMatmulPlan method.
  struct MatmulPlanParams {
    std::string ToString() const;

    blas::DataType ab_type;
    blas::DataType c_type;
    blas::ComputationType computation_type;
    PointerMode pointer_mode;
    Epilogue epilogue;
    blas::Transpose transa;
    blas::Transpose transb;
    uint64_t m;
    uint64_t n;
    uint64_t k;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int batch_count = 1;
    int64_t stride_a = 0;
    int64_t stride_b = 0;
    int64_t stride_c = 0;
  };

  class MatmulAlgorithm {
   public:
    MatmulAlgorithm(blas::AlgorithmType index, cublasLtMatmulAlgo_t algo,
                    size_t workspace_size)
        : index_(index), algo_(algo), workspace_size_(workspace_size) {}

    blas::AlgorithmType index() const { return index_; }

    size_t workspace_size() const { return workspace_size_; }

    const cublasLtMatmulAlgo_t *algo() const { return &algo_; }

    int algo_id() const;

   private:
    blas::AlgorithmType index_;
    cublasLtMatmulAlgo_t algo_;
    size_t workspace_size_;
  };

  class MatmulPlan {
   public:
    port::Status init(const MatmulPlanParams &p);

    cublasLtMatmulDesc_t op_desc() const { return op_desc_.get(); }
    cublasLtMatrixLayout_t a_desc() const { return a_desc_.get(); }
    cublasLtMatrixLayout_t b_desc() const { return b_desc_.get(); }
    cublasLtMatrixLayout_t c_desc() const { return c_desc_.get(); }
    cublasLtMatrixLayout_t d_desc() const { return d_desc_.get(); }
    cublasLtMatrixLayout_t a_remainder_desc() const {
      return a_remainder_desc_.get();
    }
    cublasLtMatrixLayout_t b_remainder_desc() const {
      return b_remainder_desc_.get();
    }
    cublasLtMatrixLayout_t c_remainder_desc() const {
      return c_remainder_desc_.get();
    }
    cublasLtMatrixLayout_t d_remainder_desc() const {
      return d_remainder_desc_.get();
    }

    const MatmulPlanParams &params() const { return params_; }
    blas::DataType scale_type() const { return scale_type_; }
    blas::DataType ab_type() const { return params_.ab_type; }
    blas::DataType c_type() const { return params_.c_type; }
    int capped_batch_count() const {
      return std::min(params_.batch_count, kMaxBatchCount);
    }
    int remainder_batch_count() const { return remainder_batch_count_; }

    // Note: Must be const to satisfy API. This is always called before the plan
    // is executed, so the state change is not observed in subsequent
    // executions.
    bool SetBiasPointer(const void *bias) const;

   private:
    // In some cases cublasLt does not support large batch sizes, so we need to
    // split up such cases into multiple calls.
    static constexpr int kMaxBatchCount = 65535;
    MatmulPlanParams params_;
    blas::DataType scale_type_;
    UniqueOpDesc op_desc_;
    // These have batch count set to capped_batch_count().
    UniqueLayoutDesc a_desc_;
    UniqueLayoutDesc b_desc_;
    UniqueLayoutDesc c_desc_;
    UniqueLayoutDesc d_desc_;
    int remainder_batch_count_;
    // These have batch count set to remainder_batch_count_, and are only
    // created if params_.batch_count > kMaxBatchSize.
    UniqueLayoutDesc a_remainder_desc_;
    UniqueLayoutDesc b_remainder_desc_;
    UniqueLayoutDesc c_remainder_desc_;
    UniqueLayoutDesc d_remainder_desc_;
  };

  // Creates a plan which can be passed to DoMatmul(). When possible, plans
  // should be created once and reused for multiple calls to DoMatmul().
  static port::StatusOr<MatmulPlan> CreateMatmulPlan(
      const MatmulPlanParams &params);

  explicit BlasLt(gpu::GpuExecutor *parent)
      : parent_(parent), blas_lt_(nullptr, cublasLtDestroy) {}

  port::Status Init();

  // Gets a list of supported algorithms for DoMatmul. The algorithms are
  // returned in the order of increasing estimated compute time according to an
  // internal heuristic. The first returned algorithm can be used as the default
  // algorithm if no autotuning is to be performed.
  port::StatusOr<std::vector<MatmulAlgorithm>> GetMatmulAlgorithms(
      const MatmulPlan &plan, size_t max_workspace_size,
      int max_algorithm_count);

  // Executes a blaslt matmul operation on the stream. If output_profile_result
  // is not nullptr, the operation is profiled, error messages are
  // suppressed, and output_profile_result->algorithm() is set to
  // algorithm->index(). If epilogue was set to kBias or kBiasThenReLU when
  // creating the plan, the bias argument here must refer to a valid device
  // vector of length equal to the number of rows in matrix c. If epilogue was
  // set to any other value then the bias argument here must be null. The bias
  // vector is broadcast across the batch dimension.
  // Note that the data types of a and b (c and bias) must match the ab_type
  // (c_type) with which the plan was created, and the data types of alpha and
  // beta must match the data type of c.
  bool DoMatmul(Stream *stream, const MatmulPlan &plan,
                const HostOrDeviceScalar<void> &alpha, DeviceMemoryBase a,
                DeviceMemoryBase b, const HostOrDeviceScalar<void> &beta,
                DeviceMemoryBase c, ScratchAllocator *scratch_allocator,
                const MatmulAlgorithm &algorithm, DeviceMemoryBase bias,
                blas::ProfileResult *output_profile_result);

  template <typename ABType, typename CType>
  bool DoMatmul(Stream *stream, const MatmulPlan &plan,
                const HostOrDeviceScalar<CType> &alpha,
                const DeviceMemory<ABType> &a, const DeviceMemory<ABType> &b,
                const HostOrDeviceScalar<CType> &beta, DeviceMemory<CType> *c,
                ScratchAllocator *scratch_allocator,
                const MatmulAlgorithm &algorithm,
                const DeviceMemory<CType> &bias = {},
                blas::ProfileResult *output_profile_result = nullptr) {
    constexpr blas::DataType ab_type = blas::ToDataType<ABType>::value;
    if (ab_type != plan.ab_type()) {
      VLOG(2) << "DoMatmul returning false because a and b type does "
                 "not match plan: expected "
              << plan.ab_type() << ", got " << ab_type;
      return false;
    }
    constexpr blas::DataType c_type = blas::ToDataType<CType>::value;
    if (c_type != plan.c_type()) {
      VLOG(2) << "DoMatmul returning false because c type does "
                 "not match plan: expected "
              << plan.c_type() << ", got " << c_type;
      return false;
    }
    return DoMatmul(stream, plan, alpha, a, b, beta, *c, scratch_allocator,
                    algorithm, bias, output_profile_result);
  }

 private:
  bool DoMatmulInternal(Stream *stream, bool err_on_failure,
                        const MatmulPlan &plan,
                        const HostOrDeviceScalar<void> &alpha,
                        DeviceMemoryBase a, DeviceMemoryBase b,
                        const HostOrDeviceScalar<void> &beta,
                        DeviceMemoryBase c, DeviceMemoryBase d,
                        ScratchAllocator *scratch_allocator,
                        const MatmulAlgorithm &algorithm,
                        DeviceMemoryBase bias);

  // Helper function for implementing GetBlasLtMatmulAlgorithms.
  port::StatusOr<std::vector<MatmulAlgorithm>> GetMatmulAlgorithmsInternal(
      const MatmulPlan &plan, size_t max_workspace_size,
      int max_algorithm_count, bool for_remainder_batch = false);

  gpu::GpuExecutor *parent_;

  absl::Mutex mu_;
  Owned<cublasLtHandle_t> blas_lt_ TF_GUARDED_BY(mu_);
};

// Returns `BlasLt` implementation for a stream if available, or `nullptr`.
BlasLt *GetBlasLt(Stream *stream);

namespace internal {

inline auto AsTuple(const BlasLt::MatmulPlanParams &p) {
  return std::make_tuple(p.ab_type, p.c_type, p.computation_type,
                         p.pointer_mode, p.epilogue, p.transa, p.transb, p.m,
                         p.n, p.k, p.lda, p.ldb, p.ldc, p.batch_count,
                         p.stride_a, p.stride_b, p.stride_c);
}

}  // namespace internal

bool operator==(const BlasLt::MatmulPlanParams &a,
                const BlasLt::MatmulPlanParams &b);

template <typename H>
H AbslHashValue(H h, const BlasLt::MatmulPlanParams &plan_params) {
  return H::combine(std::move(h), internal::AsTuple(plan_params));
}

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_BLAS_LT_H_
