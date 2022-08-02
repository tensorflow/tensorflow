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
#include <optional>
#include <string>
#include <vector>

#include "third_party/gpus/cuda/include/cublasLt.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/cuda/cuda_blas_utils.h"
#include "tensorflow/stream_executor/host_or_device_scalar.h"
#include "tensorflow/stream_executor/lib/status.h"

namespace stream_executor {
namespace gpu {
class GpuExecutor;
}  // namespace gpu

namespace cuda {

class BlasLt {
  template <typename T>
  using Owned =
      std::unique_ptr<std::remove_pointer_t<T>, cublasStatus_t (*)(T)>;

 public:
  class MatrixLayout {
   public:
    enum class Order { kRowMajor, kColumnMajor };

    // If `leading_dim_stride` is not specified, it defaults to:
    //  - `num_cols` if `order == kRowMajor`,
    //  - `num_rows` if `order == kColumnMajor`.
    // If `batch_stride` is not specified, it defaults to `num_rows * num_cols`
    // if `batch_size > 1`, otherwise `0`.
    static port::StatusOr<MatrixLayout> Create(
        blas::DataType type, size_t num_rows, size_t num_cols, Order order,
        size_t batch_size = 1,
        std::optional<int64_t> leading_dim_stride = std::nullopt,
        std::optional<int64_t> batch_stride = std::nullopt);

    cudaDataType_t type() const;

    cublasLtMatrixLayout_t get() const { return handle_.get(); }

   private:
    explicit MatrixLayout(cublasLtMatrixLayout_t handle)
        : handle_(handle, cublasLtMatrixLayoutDestroy) {}

    Owned<cublasLtMatrixLayout_t> handle_;
  };

  enum class Epilogue {
    kDefault = 1,                   // No special postprocessing
    kReLU = 2,                      // Apply point-wise ReLU function
    kBias = 4,                      // Add broadcasted bias vector
    kBiasThenReLU = kBias | kReLU,  // Apply bias and then ReLU transform
    kGeLU = 32,  // Apply GELU point-wise transform to the results
    kBiasThenGeLUApproximate =
        kBias | kGeLU,  // Apply bias and then GeLU Tanh transform
  };

  // Describes the location of pointers for the scaling factors alpha and beta.
  enum class PointerMode {
    kHost,
    kDevice,
  };

  class MatmulDesc {
   public:
    static port::StatusOr<MatmulDesc> Create(
        blas::ComputationType compute_type, blas::DataType scale_type,
        blas::Transpose trans_a = blas::Transpose::kNoTranspose,
        blas::Transpose trans_b = blas::Transpose::kNoTranspose,
        Epilogue epilogue = Epilogue::kDefault,
        PointerMode pointer_mode = PointerMode::kHost);

    cublasComputeType_t compute_type() const;
    cudaDataType_t scale_type() const;
    cublasLtPointerMode_t pointer_mode() const;

    cublasLtMatmulDesc_t get() const { return handle_.get(); }

   private:
    explicit MatmulDesc(cublasLtMatmulDesc_t handle)
        : handle_(handle, cublasLtMatmulDescDestroy) {}

    Owned<cublasLtMatmulDesc_t> handle_;
  };

  // TODO(cjfj): Add consistency checks for types, shapes, etc.?
  struct MatmulPlan {
    MatmulDesc op_desc;
    MatrixLayout a_desc;
    MatrixLayout b_desc;
    MatrixLayout c_desc;
    MatrixLayout d_desc;
  };

  class MatmulPreference {
   public:
    static port::StatusOr<MatmulPreference> Create(size_t max_workspace_size);

    cublasLtMatmulPreference_t get() const { return handle_.get(); }

   private:
    explicit MatmulPreference(cublasLtMatmulPreference_t handle)
        : handle_(handle, cublasLtMatmulPreferenceDestroy) {}

    Owned<cublasLtMatmulPreference_t> handle_;
  };

  struct MatmulAlgorithm {
    cublasLtMatmulAlgo_t algo;
    size_t workspace_size;
  };

  explicit BlasLt(gpu::GpuExecutor* parent)
      : parent_(parent), blas_lt_(nullptr, cublasLtDestroy) {}

  port::Status Init();

  // Returns the type for the alpha and beta scalars.
  static blas::DataType GetScaleType(blas::DataType c_type,
                                     blas::ComputationType computation_type);

  // Returns a list of supported algorithms for DoMatmul. The algorithms are
  // returned in the order of increasing estimated compute time according to an
  // internal heuristic.
  port::StatusOr<std::vector<MatmulAlgorithm>> GetMatmulAlgorithms(
      const MatmulPlan& plan, const MatmulPreference& preference,
      size_t max_algorithm_count = 128);

  template <typename AB, typename CD, typename Scale>
  port::Status DoMatmul(Stream* stream, const MatmulPlan& plan,
                        const HostOrDeviceScalar<Scale>& alpha,
                        const DeviceMemory<AB>& a, const DeviceMemory<AB>& b,
                        const HostOrDeviceScalar<Scale>& beta,
                        const DeviceMemory<CD>& c, DeviceMemory<CD>& d,
                        const MatmulAlgorithm& algorithm,
                        ScratchAllocator& scratch_allocator,
                        const DeviceMemory<CD>& bias = {},
                        blas::ProfileResult* profile_result = nullptr) {
    if (AsCudaDataType(blas::ToDataType<Scale>::value) !=
        plan.op_desc.scale_type()) {
      return port::InvalidArgumentError("mismatched scale types");
    }

    bool expect_scale_factor_on_device =
        (plan.op_desc.pointer_mode() == CUBLASLT_POINTER_MODE_DEVICE);

    if (alpha.on_device() != expect_scale_factor_on_device) {
      return port::InvalidArgumentError("wrong location for alpha");
    }

    if (beta.on_device() != expect_scale_factor_on_device) {
      return port::InvalidArgumentError("wrong location for beta");
    }

    if (AsCudaDataType(blas::ToDataType<AB>::value) != plan.a_desc.type()) {
      return port::InvalidArgumentError("mismatched A matrix types");
    }

    if (AsCudaDataType(blas::ToDataType<AB>::value) != plan.b_desc.type()) {
      return port::InvalidArgumentError("mismatched B matrix types");
    }

    if (AsCudaDataType(blas::ToDataType<CD>::value) != plan.c_desc.type()) {
      return port::InvalidArgumentError("mismatched C matrix types");
    }

    if (AsCudaDataType(blas::ToDataType<CD>::value) != plan.d_desc.type()) {
      return port::InvalidArgumentError("mismatched D matrix types");
    }

    return DoMatmul(stream, plan, alpha.opaque(), a, b, beta.opaque(), c, d,
                    algorithm, scratch_allocator, bias, profile_result);
  }

 private:
  port::Status DoMatmul(Stream* stream, const MatmulPlan& plan,
                        const void* alpha, DeviceMemoryBase a,
                        DeviceMemoryBase b, const void* beta,
                        DeviceMemoryBase c, DeviceMemoryBase d,
                        const MatmulAlgorithm& algorithm,
                        ScratchAllocator& scratch_allocator,
                        DeviceMemoryBase bias,
                        blas::ProfileResult* profile_result);

  gpu::GpuExecutor* parent_;

  absl::Mutex mu_;
  Owned<cublasLtHandle_t> blas_lt_ ABSL_GUARDED_BY(mu_);
};

// Returns `BlasLt` implementation for a stream if available, or `nullptr`.
BlasLt* GetBlasLt(Stream* stream);

}  // namespace cuda
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_CUDA_CUDA_BLAS_LT_H_
