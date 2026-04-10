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

#include <cstddef>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "rocm/rocm_config.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"

#if TF_HIPBLASLT

#include "rocm/include/hipblaslt/hipblaslt-ext.hpp"
#include "xla/stream_executor/rocm/hip_blas_utils.h"

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
        : handle_(handle, wrap::hipblasLtMatmulDescDestroy),
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

  struct MatmulPlan : public gpu::BlasLt::MatmulPlan {
    // Constructor for regular matmul
    MatmulPlan(MatmulDesc&& op_desc, MatrixLayout&& a_desc,
               MatrixLayout&& b_desc, MatrixLayout&& c_desc,
               MatrixLayout&& d_desc, xla::complex128 alpha, double beta,
               bool must_swap_operands)
        : op_desc_(std::move(op_desc)),
          a_desc_(std::move(a_desc)),
          b_desc_(std::move(b_desc)),
          c_desc_(std::move(c_desc)),
          d_desc_(std::move(d_desc)),
          alpha_(alpha),
          beta_(beta),
          must_swap_operands_(must_swap_operands),
          grouped_gemm_(nullptr) {}

    // Constructor for grouped matmul
    MatmulPlan(gpu::GroupedGemmConfig&& cfg, bool must_swap_operands,
               hipblasLtHandle_t blas_lt_handle,
               blas::ComputationType compute_type)
        : must_swap_operands_(must_swap_operands),
          cfg_(std::move(cfg)),
          grouped_gemm_(nullptr) {
      InitializeGroupedGemm(blas_lt_handle, compute_type);
    }

    ~MatmulPlan() override = default;

    absl::Status ExecuteOnStream(
        Stream* stream, const gpu::BlasLt::MemoryArgs& args,
        blas::ProfileResult* profile_result) const override;

    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        const Stream* stream, size_t max_algorithm_count,
        size_t max_workspace_size) const override;

    absl::Status SetAlgorithm(const MatmulAlgorithm& algorithm) override {
      algorithm_ = algorithm;
      algorithm_must_be_initialized_ = true;
      return absl::OkStatus();
    }

    bool is_grouped() const { return grouped_gemm_ != nullptr; }

   protected:
    absl::Status DoMatmul(Stream* stream, const void* alpha, const void* beta,
                          const gpu::BlasLt::MemoryArgs& args,
                          blas::ProfileResult* profile_result) const;

   private:
    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithmsForGroupedMatmul(
        const Stream* stream, size_t max_algorithm_count,
        size_t max_workspace_size) const;
    absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithmsForMatmul(
        const Stream* stream, size_t max_algorithm_count,
        size_t max_workspace_size) const;
    absl::Status ExecuteRegularMatmul(
        Stream* stream, const gpu::BlasLt::MemoryArgs& args,
        blas::ProfileResult* profile_result) const;
    absl::Status ExecuteGroupedMatmul(
        Stream* stream, const gpu::BlasLt::MemoryArgs& args,
        blas::ProfileResult* profile_result) const;

    void InitializeGroupedGemm(hipblasLtHandle_t blas_lt_handle,
                               blas::ComputationType compute_type);

    // TODO(cjfj): Add consistency checks for types, shapes, etc.?
    // Regular matmul members (optional for grouped matmul)
    std::optional<MatmulDesc> op_desc_;
    std::optional<MatrixLayout> a_desc_;
    std::optional<MatrixLayout> b_desc_;
    std::optional<MatrixLayout> c_desc_;
    std::optional<MatrixLayout> d_desc_;
    std::optional<xla::complex128> alpha_;
    std::optional<double> beta_;
    bool must_swap_operands_;
    std::optional<MatmulAlgorithm> algorithm_;  // selected algorithm
    // Grouped matmul members
    std::optional<gpu::GroupedGemmConfig> cfg_;
    std::unique_ptr<hipblaslt_ext::GroupedGemm> grouped_gemm_;
    mutable bool algorithm_must_be_initialized_ = false;
    mutable DeviceMemoryBase saved_address_workspace_{};
  };  // class MatmulPlan

  explicit BlasLt(StreamExecutor* parent)
      : parent_(parent), blas_lt_(nullptr, wrap::hipblasLtDestroy) {}

  absl::Status Init() override;

  absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const gpu::GemmConfig& cfg,
                                              Epilogue epilogue) const override;

  absl::StatusOr<MatmulPlanPtr> GetGroupedMatmulPlan(
      gpu::GroupedGemmConfig& config,
      const std::vector<Epilogue>& epilogues) const override;

  ~BlasLt() override = default;

 private:
  StreamExecutor* parent_;
  mutable absl::Mutex mu_;
  Owned<hipblasLtHandle_t> blas_lt_ ABSL_GUARDED_BY(mu_);
};

}  // namespace rocm
}  // namespace stream_executor

#endif  // TF_HIPBLASLT
#endif  // XLA_STREAM_EXECUTOR_ROCM_HIP_BLAS_LT_H_
