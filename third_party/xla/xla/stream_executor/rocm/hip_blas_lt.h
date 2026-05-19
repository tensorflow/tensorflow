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

#include <any>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "rocm/include/hipblaslt/hipblaslt.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/rocm/hip_blas_utils.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"

namespace hipblaslt_ext {
class GroupedGemm;
struct UserArguments;
}  // namespace hipblaslt_ext

namespace stream_executor {

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
        : handle_(handle, hipblasLtMatrixLayoutDestroy), datatype_(datatype) {}

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
        PointerMode pointer_mode = PointerMode::kHost,
        gpu::ScaleMode scale_mode = gpu::ScaleMode::kNone);

    hipblasComputeType_t compute_type() const { return compute_type_; }
    hipDataType scale_type() const { return datatype_; }
    bool has_bias_epilogue() const { return has_bias_epilogue_; }
    gpu::ScaleMode scale_mode() const { return scale_mode_; }
    hipblasPointerMode_t pointer_mode() const {
      return HIPBLAS_POINTER_MODE_HOST;
    }
    hipblasLtMatmulDesc_t get() const { return handle_.get(); }

   private:
    MatmulDesc(hipblasLtMatmulDesc_t handle, hipblasComputeType_t compute_type,
               hipDataType datatype, bool bias_epilogue,
               gpu::ScaleMode scale_mode)
        : handle_(handle, hipblasLtMatmulDescDestroy),
          compute_type_(compute_type),
          datatype_(datatype),
          has_bias_epilogue_(bias_epilogue),
          scale_mode_(scale_mode) {}

    Owned<hipblasLtMatmulDesc_t> handle_;
    hipblasComputeType_t compute_type_;
    hipDataType datatype_;
    bool has_bias_epilogue_;
    gpu::ScaleMode scale_mode_;
  };

  class RegularMatmulPlan : public gpu::BlasLt::MatmulPlan {
   public:
    friend class BlasLt;
    // We use a fixed-size array to store the alpha and beta values which can
    // fit all supported scale types.
    constexpr static size_t kMaxScaleBytes = 16;

    RegularMatmulPlan(const BlasLt& blas_lt, MatmulDesc&& op_desc,
                      MatrixLayout&& a_desc, MatrixLayout&& b_desc,
                      MatrixLayout&& c_desc, MatrixLayout&& d_desc,
                      bool must_swap_operands)
        : blas_lt_(blas_lt),
          op_desc_(std::move(op_desc)),
          a_desc_(std::move(a_desc)),
          b_desc_(std::move(b_desc)),
          c_desc_(std::move(c_desc)),
          d_desc_(std::move(d_desc)),
          must_swap_operands_(must_swap_operands) {}

    ~RegularMatmulPlan() override = default;

    absl::Status ExecuteOnStream(
        Stream* stream, const gpu::BlasLt::MemoryArgs& args,
        blas::ProfileResult* profile_result) const override;

    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        size_t max_algorithm_count, size_t max_workspace_size) const override;

    absl::Status SetAlgorithm(const MatmulAlgorithm& algorithm) override {
      algorithm_ = algorithm;
      return absl::OkStatus();
    }

   private:
    const BlasLt& blas_lt_;
    MatmulDesc op_desc_;
    MatrixLayout a_desc_;
    MatrixLayout b_desc_;
    MatrixLayout c_desc_;
    MatrixLayout d_desc_;
    alignas(16) std::array<uint8_t, kMaxScaleBytes> alpha_, beta_;
    bool must_swap_operands_;
    mutable std::optional<MatmulAlgorithm> algorithm_;  // selected algorithm
  };  // class RegularMatmulPlan

  class GroupedMatmulPlan : public gpu::BlasLt::MatmulPlan {
   public:
    friend class BlasLt;

    GroupedMatmulPlan(const BlasLt& blas_lt, const gpu::GroupedGemmConfig& cfg)
        : blas_lt_(blas_lt), cfg_(cfg) {}

    ~GroupedMatmulPlan() override = default;

    absl::Status ExecuteOnStream(
        Stream* stream, const gpu::BlasLt::MemoryArgs& args,
        blas::ProfileResult* profile_result) const override;

    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        size_t max_algorithm_count, size_t max_workspace_size) const override;

    absl::Status SetAlgorithm(const MatmulAlgorithm& algorithm) override {
      algorithm_ = algorithm;
      algorithm_dirty_ = true;
      return absl::OkStatus();
    }

   private:
    absl::Status DoInitialize(blas::ComputationType compute_type,
                              Epilogue epilogue);

    const BlasLt& blas_lt_;
    gpu::GroupedGemmConfig cfg_;
    Epilogue epilogue_ = Epilogue::kDefault;
    std::unique_ptr<hipblaslt_ext::GroupedGemm> grouped_gemm_;
    mutable std::optional<MatmulAlgorithm> algorithm_;  // selected algorithm
    mutable bool algorithm_dirty_ = false;
    mutable DeviceAddressBase saved_address_workspace_{};
    // Saved default activation parameters from hipBLASLt
    int32_t activation_type_ = 0;
    int8_t bias_type_ = 0;
  };  // class GroupedMatmulPlan

  // Executes complex (C64/C128) matmuls via rocBLAS (rocblas_cgemm/zgemm),
  // since hipBLASLt has no complex GEMM kernels in current ROCm releases.
  class RocBlasGemmPlan : public gpu::BlasLt::MatmulPlan {
   public:
    friend class BlasLt;

    RocBlasGemmPlan(const BlasLt& blas_lt, const gpu::GemmConfig& cfg)
        : blas_lt_(blas_lt), cfg_(cfg) {}

    ~RocBlasGemmPlan() override = default;

    absl::Status ExecuteOnStream(
        Stream* stream, const gpu::BlasLt::MemoryArgs& args,
        blas::ProfileResult* profile_result) const override;

    // Advertise a single no-workspace pseudo-algorithm so the autotuner /
    // thunk machinery proceeds unchanged (hipBLASLt heuristics are unused).
    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        size_t /*max_algorithm_count*/,
        size_t /*max_workspace_size*/) const override {
      return std::vector<MatmulAlgorithm>{MatmulAlgorithm{std::any{}, 0}};
    }

    absl::Status SetAlgorithm(const MatmulAlgorithm& /*algorithm*/) override {
      return absl::OkStatus();
    }

   private:
    const BlasLt& blas_lt_;
    gpu::GemmConfig cfg_;
  };  // class RocBlasGemmPlan

  explicit BlasLt(StreamExecutor* executor)
      : executor_(executor), handle_(nullptr, hipblasLtDestroy) {}

  absl::Status Init() override;

  absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const gpu::GemmConfig& cfg,
                                              Epilogue epilogue) const override;

  absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const gpu::GroupedGemmConfig& cfg,
                                              Epilogue epilogue) const override;

  ~BlasLt() override = default;

 private:
  StreamExecutor* executor_;
  mutable absl::Mutex mu_;
  Owned<hipblasLtHandle_t> handle_ ABSL_GUARDED_BY(mu_);
};

}  // namespace rocm
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
