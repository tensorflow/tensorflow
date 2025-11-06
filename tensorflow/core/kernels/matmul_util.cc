/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/matmul_util.h"

#if GOOGLE_CUDA || TF_HIPBLASLT

#include <optional>
#include <string>
#include <utility>

#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/matmul_autotune.h"

namespace tensorflow {

int64_t GetWorkspaceLimit(int64_t default_value_in_bytes) {
  const char* workspace_limit_in_mb_str =
      getenv("TF_CUBLAS_WORKSPACE_LIMIT_IN_MB");
  if (workspace_limit_in_mb_str != nullptr &&
      strcmp(workspace_limit_in_mb_str, "") != 0) {
    int64_t scratch_limit_in_mb = -1;
    if (strings::safe_strto64(workspace_limit_in_mb_str,
                              &scratch_limit_in_mb)) {
      return scratch_limit_in_mb * (1 << 20);
    } else {
      LOG(WARNING) << "Invalid value for TF_CUBLAS_WORKSPACE_LIMIT_IN_MB: "
                   << workspace_limit_in_mb_str;
    }
  }
  return default_value_in_bytes;
}

std::string BlasLtMatmulPlanParams::ToString() const {
  return "";  // TODO
}

bool BlasLtMatmulPlanParams::operator==(
    const BlasLtMatmulPlanParams& other) const {
  return internal::AsTuple(*this) == internal::AsTuple(other);
}

namespace {

// Thread-safe map from matmul parameters to their corresponding plan and
// algorithms.
struct BlasLtMatmulPlanMap {
  absl::Mutex mu;

  template <class K, class... Args>
  auto try_emplace(K&& k, Args&&... args) {
    absl::MutexLock lock(&mu);
    return map_.try_emplace(std::forward<K>(k), std::forward<Args>(args)...);
  }

 private:
  absl::flat_hash_map<BlasLtMatmulPlanParams,
                      std::unique_ptr<PlanAndAlgorithms>>
      map_ ABSL_GUARDED_BY(mu);
};

int MatmulMaxAutotuneAlgorithmCount() {
  int64_t value;
  Status status =
      ReadInt64FromEnvVar("TF_MATMUL_AUTOTUNE_MAX_ALGORITHMS", 10, &value);
  if (!status.ok()) {
    LOG(ERROR) << status.message();
  }
  static constexpr const int kMaxValue = std::numeric_limits<int>::max();
  if (value < 1 || value > kMaxValue) {
    LOG(ERROR) << "Invalid value for TF_MATMUL_AUTOTUNE_MAX_ALGORITHMS: "
               << value << " is not in range [1, " << kMaxValue << "]";
  }
  return value;
}

StatusOr<se::blas::ComputationType> GetBlasComputationType(
    se::blas::DataType dtype) {
  using se::blas::ComputationType;
  static bool use_f32_for_f16_computation = MatmulDoFP32ComputationFP16Input();
  switch (dtype) {
    case se::blas::DataType::kHalf:
      return use_f32_for_f16_computation ? ComputationType::kF32
                                         : ComputationType::kF16;
    case se::blas::DataType::kBF16:
      return ComputationType::kF32;
    case se::blas::DataType::kFloat:  // fall-through
    case se::blas::DataType::kComplexFloat:
      return tensor_float_32_execution_enabled() ? ComputationType::kTF32AsF32
                                                 : ComputationType::kF32;
    case se::blas::DataType::kDouble:  // fall-through
    case se::blas::DataType::kComplexDouble:
      return ComputationType::kF64;
    default:
      return errors::Internal("Unsupported dtype for Blas Plans.");
  }
}

}  // namespace

/* static */ StatusOr<const PlanAndAlgorithms*> PlanAndAlgorithms::GetOrCreate(
    se::Stream* stream, const BlasLtMatmulPlanParams& params,
    absl::Mutex** ppmu, std::optional<int> max_algorithm_count) {
  static const int64_t max_scratch_size =
      GetWorkspaceLimit(1LL << 32);  // 4GB by default
  static const int64_t max_autotune_algorithm_count =
      MatmulMaxAutotuneAlgorithmCount();

  if (!max_algorithm_count) max_algorithm_count = max_autotune_algorithm_count;

  static BlasLtMatmulPlanMap plan_map;

  auto [ptr, inserted] =
      plan_map.try_emplace(params, std::make_unique<PlanAndAlgorithms>());
  if (inserted) {
    TF_ASSIGN_OR_RETURN(auto xlatype,
                        se::gpu::AsXlaPrimitiveType(params.dtype));
    TF_ASSIGN_OR_RETURN(auto computation_type,
                        GetBlasComputationType(params.dtype));

    // row-major output is now handled automatically by blas-lt API
    constexpr auto kRowMajor = se::gpu::MatrixLayout::Order::kRowMajor;

    int64_t rows_a = static_cast<int64_t>(params.m),
            cols_a = static_cast<int64_t>(params.k),
            rows_b = static_cast<int64_t>(params.k),
            cols_b = static_cast<int64_t>(params.n),
            rows_c = static_cast<int64_t>(params.m),
            cols_c = static_cast<int64_t>(params.n),
            batch_sz = static_cast<int64_t>(params.batch_count);

    if (params.trans_a != se::blas::Transpose::kNoTranspose) {
      std::swap(rows_a, cols_a);
    }
    if (params.trans_b != se::blas::Transpose::kNoTranspose) {
      std::swap(rows_b, cols_b);
    }
    int64_t batch_stride_a = params.broadcast_a ? 0 : rows_a * cols_a;
    int64_t batch_stride_b = params.broadcast_b ? 0 : rows_b * cols_b;

    // `A` and `B` swapped (see above re. column-major output).
    se::gpu::GemmConfig cfg = {
        .lhs_layout =
            se::gpu::MatrixLayout{xlatype, rows_a, cols_a, kRowMajor, batch_sz,
                                  std::nullopt, batch_stride_a, params.trans_a},
        .rhs_layout =
            se::gpu::MatrixLayout{xlatype, rows_b, cols_b, kRowMajor, batch_sz,
                                  std::nullopt, batch_stride_b, params.trans_b},
        .c_layout =
            se::gpu::MatrixLayout{xlatype, rows_c, cols_c, kRowMajor, batch_sz},
        .output_layout =
            se::gpu::MatrixLayout{xlatype, rows_c, cols_c, kRowMajor, batch_sz},
        .alpha = xla::complex128{1.0, 0.0},
        .beta = 0.0,
        .compute_precision = se::blas::kDefaultComputePrecision,
        .precision_algorithm = xla::PrecisionConfig::ALG_UNSET,
        .algorithm = {},
        .grad_x = false,
        .grad_y = false,
        .compute_type = computation_type,
    };

    TF_ASSIGN_OR_RETURN(auto plan, se::gpu::BlasLt::GetMatmulPlan(
                                       stream, cfg, params.epilogue));

    TF_ASSIGN_OR_RETURN(
        auto algorithms,
        plan->GetAlgorithms(stream, *max_algorithm_count, max_scratch_size));

    *ptr->second = {std::move(plan), std::move(algorithms)};
  }
  *ppmu = &plan_map.mu;
  return ptr->second.get();
}

Status PlanAndAlgorithms::ExecuteOnStream(
    se::Stream* stream, const se::DeviceMemoryBase& a,
    const se::DeviceMemoryBase& b, se::DeviceMemoryBase& c,
    size_t algorithm_idx, se::ScratchAllocator& scratch_allocator,
    const se::DeviceMemoryBase& bias,
    se::blas::ProfileResult* profile_result) const {
  if (!plan || algorithm_idx >= algorithms.size()) {
    return errors::Internal("MatmulPlan or algorithms are not initialized!");
  }
  TF_RETURN_IF_ERROR(plan->SetAlgorithm(algorithms[algorithm_idx]));
  return plan->ExecuteOnStream(stream, a, b, c, c,
                               bias,                    // bias_buffer
                               se::DeviceMemoryBase{},  // aux_buffer
                               se::DeviceMemoryBase{},  // a_scale_buffer
                               se::DeviceMemoryBase{},  // b_scale_buffer
                               se::DeviceMemoryBase{},  // c_scale_buffer
                               se::DeviceMemoryBase{},  // d_scale_buffer
                               se::DeviceMemoryBase{},  // d_amax_buffer
                               scratch_allocator, profile_result);
}

}  // namespace tensorflow

#endif