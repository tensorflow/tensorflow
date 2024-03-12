/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_CUDA_CUDA_BLAS_LT_H_
#define XLA_STREAM_EXECUTOR_CUDA_CUDA_BLAS_LT_H_

#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "third_party/gpus/cuda/include/cublasLt.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/library_types.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/types.h"

namespace stream_executor {
namespace gpu {
class GpuExecutor;
}  // namespace gpu

namespace cuda {

class BlasLt : public gpu::BlasLt {
  template <typename T>
  using Owned =
      std::unique_ptr<std::remove_pointer_t<T>, cublasStatus_t (*)(T)>;

 public:
  struct MatrixLayout {
    // If `leading_dim_stride` is not specified, it defaults to:
    //  - `num_cols` if `order == kRowMajor`,
    //  - `num_rows` if `order == kColumnMajor`.
    // If `batch_stride` is not specified, it defaults to `num_rows * num_cols`
    // if `batch_size > 1`, otherwise `0`.
    static absl::StatusOr<MatrixLayout> Create(const gpu::MatrixLayout& m);

    cudaDataType_t type() const;
    cublasLtMatrixLayout_t get() const { return handle_.get(); }

   private:
    explicit MatrixLayout(cublasLtMatrixLayout_t handle)
        : handle_(handle, cublasLtMatrixLayoutDestroy) {}

    Owned<cublasLtMatrixLayout_t> handle_;
  };

  class MatmulDesc {
   public:
    static absl::StatusOr<MatmulDesc> Create(
        blas::ComputationType compute_type, blas::DataType scale_type,
        blas::Transpose trans_a = blas::Transpose::kNoTranspose,
        blas::Transpose trans_b = blas::Transpose::kNoTranspose,
        Epilogue epilogue = Epilogue::kDefault, bool enable_fast_accum = false,
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

  class MatmulPlan : public gpu::BlasLt::MatmulPlan {
   public:
    MatmulPlan(const BlasLt& blas_lt_ref, MatmulDesc&& op_desc,
               MatrixLayout&& a_desc, MatrixLayout&& b_desc,
               MatrixLayout&& c_desc, MatrixLayout&& d_desc,
               xla::complex128 alpha, double beta, bool must_swap_operands)
        : blas_lt_ref_(blas_lt_ref),
          op_desc_(std::move(op_desc)),
          a_desc_(std::move(a_desc)),
          b_desc_(std::move(b_desc)),
          c_desc_(std::move(c_desc)),
          d_desc_(std::move(d_desc)),
          alpha_(alpha),
          beta_(beta),
          must_swap_operands_(must_swap_operands) {}

    ~MatmulPlan() override = default;

    absl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a_buffer, DeviceMemoryBase b_buffer,
        DeviceMemoryBase c_buffer, DeviceMemoryBase d_buffer,
        DeviceMemoryBase bias_buffer,  // may be null
        DeviceMemoryBase aux_buffer,   // may be null
        DeviceMemoryBase a_scale_buffer, DeviceMemoryBase b_scale_buffer,
        DeviceMemoryBase c_scale_buffer, DeviceMemoryBase d_scale_buffer,
        DeviceMemoryBase d_amax_buffer, const MatmulAlgorithm& algorithm,
        ScratchAllocator& scratch_allocator,
        blas::ProfileResult* profile_result = nullptr) const override;

    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        size_t max_algorithm_count, size_t max_workspace_size) const override;

   protected:
    absl::Status ValidateInputs(blas::DataType scale_type, bool alpha_on_device,
                                bool beta_on_device, blas::DataType A_type,
                                blas::DataType B_type, blas::DataType C_type,
                                blas::DataType D_type) const override;

    absl::Status DoMatmul(Stream* stream, const void* alpha, DeviceMemoryBase a,
                          DeviceMemoryBase b, const void* beta,
                          DeviceMemoryBase c, DeviceMemoryBase d,
                          const MatmulAlgorithm& algorithm,
                          ScratchAllocator& scratch_allocator,
                          DeviceMemoryBase bias, DeviceMemoryBase aux,
                          DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
                          DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
                          DeviceMemoryBase d_amax,
                          blas::ProfileResult* profile_result) const override;

   private:
    const BlasLt& blas_lt_ref_;
    // TODO(cjfj): Add consistency checks for types, shapes, etc.?
    MatmulDesc op_desc_;
    MatrixLayout a_desc_;
    MatrixLayout b_desc_;
    MatrixLayout c_desc_;
    MatrixLayout d_desc_;
    xla::complex128 alpha_;
    double beta_;
    bool must_swap_operands_;
  };  // class MatmulPlan

  explicit BlasLt(gpu::GpuExecutor* parent)
      : parent_(parent), blas_lt_(nullptr, cublasLtDestroy) {}

  absl::Status Init() override;

  absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const gpu::GemmConfig& cfg,
                                              Epilogue epilogue) const override;

  ~BlasLt() override = default;

 private:
  gpu::GpuExecutor* parent_;
  mutable absl::Mutex mu_;
  Owned<cublasLtHandle_t> blas_lt_ ABSL_GUARDED_BY(mu_);
};

}  // namespace cuda
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_CUDA_CUDA_BLAS_LT_H_
