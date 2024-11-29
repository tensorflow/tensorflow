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
#include <deque>
#include <utility>

#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/tensor_float_32_utils.h"
#include "tensorflow/core/util/env_var.h"
#include "tensorflow/core/util/matmul_autotune.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

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

bool BlasLtMatmulPlanParams::operator==(
    const BlasLtMatmulPlanParams& other) const {
  return internal::AsTuple(*this) == internal::AsTuple(other);
}

namespace {

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

/* static */ BlasLtMatmulPlanCache& BlasLtMatmulPlanCache::i(se::Stream *stream) {
  static absl::Mutex m(absl::kConstInit);
  // Each GPU gets different cache instance
  static std::deque< BlasLtMatmulPlanCache > meta(8);
  absl::MutexLock lock(&m);
  size_t dev_id = stream->parent()->device_ordinal();
  if (dev_id >= meta.size()) meta.resize(dev_id + 1);
  return meta[dev_id];
}

/* static */ auto BlasLtMatmulPlanCache::GetOrCreate(
    se::Stream* stream, const BlasLtMatmulPlanParams& params,
    absl::Mutex** ppmu, std::optional<int> max_algorithm_count) -> StatusOr<const Entry *>{
  static const int64_t max_scratch_size =
      GetWorkspaceLimit(1LL << 32);  // 4GB by default
  static const int64_t max_autotune_algorithm_count =
      MatmulMaxAutotuneAlgorithmCount();

  if (!max_algorithm_count) max_algorithm_count = max_autotune_algorithm_count;

  auto& self = BlasLtMatmulPlanCache::i(stream);

  absl::MutexLock lock(self.mutex_.get());
  auto [ptr, inserted] = self.map_.emplace(params, Entry{});
  auto& entry = ptr->second;
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

    TF_ASSIGN_OR_RETURN(entry.plan, se::gpu::BlasLt::GetMatmulPlan(
                                       stream, cfg, params.epilogue));

    TF_ASSIGN_OR_RETURN(
        entry.algorithms,
        entry.plan->GetAlgorithms(*max_algorithm_count, max_scratch_size));
  }
  *ppmu = self.mutex_.get();
  return &entry;
}

/*static */ Status BlasLtMatmulPlanCache::ExecuteOnStream(se::Stream* stream, 
                      const Entry& entry,
                      const se::DeviceMemoryBase& a,
                      const se::DeviceMemoryBase& b, 
                      se::DeviceMemoryBase& c,
                      size_t algorithm_idx, 
                      se::ScratchAllocator& scratch_allocator,
                      const se::DeviceMemoryBase& bias,
                      se::blas::ProfileResult* profile_result) {

  return entry.plan->ExecuteOnStream(
        stream, a, b, c, c,
        bias,                  // bias_buffer
        se::DeviceMemoryBase{}, // aux_buffer
        se::DeviceMemoryBase{}, // a_scale_buffer
        se::DeviceMemoryBase{}, // b_scale_buffer
        se::DeviceMemoryBase{}, // c_scale_buffer
        se::DeviceMemoryBase{}, // d_scale_buffer
        se::DeviceMemoryBase{}, // d_amax_buffer
        entry.algorithms[algorithm_idx],
        scratch_allocator, 
        profile_result);
}


}  // namespace tensorflow

#endif