/* Copyright 2023 The OpenXLA Authors.
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

#ifndef XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
#define XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_

#include "absl/status/status.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/host_or_device_scalar.h"
#include "xla/types.h"

#if TF_HIPBLASLT

#include "xla/stream_executor/rocm/hip_blas_utils.h"

namespace stream_executor {

namespace gpu {
class GpuExecutor;
}  // namespace gpu

namespace rocm {

class BlasLt : public gpu::BlasLt {
  template <typename T>
  using Owned =
      std::unique_ptr<std::remove_pointer_t<T>, hipblasStatus_t (*)(T)>;

 public:
  struct MatrixLayout {
    static absl::StatusOr<MatrixLayout> Create(const gpu::MatrixLayout& m);

    hipDataType type() const { return datatype_; }
    hipblasLtMatrixLayout_t get() const { return handle_.get(); }

   private:
    MatrixLayout(hipblasLtMatrixLayout_t handle, hipDataType datatype)
        : handle_(handle, wrap::hipblasLtMatrixLayoutDestroy),
          datatype_(datatype) {}

    Owned<hipblasLtMatrixLayout_t> handle_;
    hipDataType datatype_;
  };

  class MatmulDesc {
   public:
    static absl::StatusOr<MatmulDesc> Create(
        blas::ComputationType compute_type, blas::DataType scale_type,
        blas::Transpose trans_a = blas::Transpose::kNoTranspose,
        blas::Transpose trans_b = blas::Transpose::kNoTranspose,
        Epilogue epilogue = Epilogue::kDefault,
        PointerMode pointer_mode = PointerMode::kHost);

    hipblasComputeType_t compute_type() const { return compute_type_; }
    hipDataType scale_type() const { return datatype_; }
    hipblasPointerMode_t pointer_mode() const {
      return HIPBLAS_POINTER_MODE_HOST;
    }
    hipblasLtMatmulDesc_t get() const { return handle_.get(); }

   private:
    MatmulDesc(hipblasLtMatmulDesc_t handle, hipblasComputeType_t compute_type,
               hipDataType datatype)
        : handle_(handle, wrap::hipblasLtMatmulDescDestroy),
          compute_type_(compute_type),
          datatype_(datatype) {}

    Owned<hipblasLtMatmulDesc_t> handle_;
    hipblasComputeType_t compute_type_;
    hipDataType datatype_;
  };

  struct MatmulPlan : public gpu::BlasLt::MatmulPlan {
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

    absl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a_buffer, DeviceMemoryBase b_buffer,
        DeviceMemoryBase c_buffer, DeviceMemoryBase d_buffer,
        DeviceMemoryBase bias_buffer,  // may be null
        DeviceMemoryBase aux_buffer,   // may be null
        DeviceMemoryBase a_scale_buffer, DeviceMemoryBase b_scale_buffer,
        DeviceMemoryBase c_scale_buffer, DeviceMemoryBase d_scale_buffer,
        DeviceMemoryBase d_amax_buffer, const MatmulAlgorithm& algorithm,
        std::optional<DeviceMemoryBase> workspace,
        blas::ProfileResult* profile_result = nullptr) const override;

    absl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a_buffer, DeviceMemoryBase b_buffer,
        DeviceMemoryBase c_buffer, DeviceMemoryBase d_buffer,
        DeviceMemoryBase bias_buffer,  // may be null
        DeviceMemoryBase aux_buffer,   // may be null
        DeviceMemoryBase a_scale_buffer, DeviceMemoryBase b_scale_buffer,
        DeviceMemoryBase c_scale_buffer, DeviceMemoryBase d_scale_buffer,
        DeviceMemoryBase d_amax_buffer, const MatmulAlgorithm& algorithm,
        std::optional<DeviceMemoryBase> workspace,
        std::optional<ScratchAllocator*> scratch_allocator,
        blas::ProfileResult* profile_result = nullptr) const;

    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        size_t max_algorithm_count, size_t max_workspace_size) const override;

   protected:
    absl::Status ValidateInputs(blas::DataType scale_type, bool alpha_on_device,
                                bool beta_on_device, blas::DataType A_type,
                                blas::DataType B_type, blas::DataType C_type,
                                blas::DataType D_type) const override;

    // API that uses scratch_allocator to allocate workspace
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

    // API that uses pre-allocated buffer as workspace
    absl::Status DoMatmul(Stream* stream, const void* alpha, DeviceMemoryBase a,
                          DeviceMemoryBase b, const void* beta,
                          DeviceMemoryBase c, DeviceMemoryBase d,
                          const MatmulAlgorithm& algorithm,
                          DeviceMemoryBase bias, DeviceMemoryBase aux,
                          DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
                          DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
                          DeviceMemoryBase d_amax,
                          std::optional<DeviceMemoryBase> workspace,
                          blas::ProfileResult* profile_result) const override;

    absl::Status DoMatmul(Stream* stream, const void* alpha, DeviceMemoryBase a,
                          DeviceMemoryBase b, const void* beta,
                          DeviceMemoryBase c, DeviceMemoryBase d,
                          const MatmulAlgorithm& algorithm,
                          DeviceMemoryBase bias, DeviceMemoryBase aux,
                          DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
                          DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
                          DeviceMemoryBase d_amax,
                          std::optional<DeviceMemoryBase> workspace,
                          std::optional<ScratchAllocator*> scratch_allocator,
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
      : parent_(parent), blas_lt_(nullptr, wrap::hipblasLtDestroy) {}

  absl::Status Init() override;

  absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const gpu::GemmConfig& cfg,
                                              Epilogue epilogue) const override;

  ~BlasLt() override = default;

 private:
  gpu::GpuExecutor* parent_;
  mutable absl::Mutex mu_;
  Owned<hipblasLtHandle_t> blas_lt_ ABSL_GUARDED_BY(mu_);
};

}  // namespace rocm
}  // namespace stream_executor

#endif  // TF_HIPBLASLT
#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
