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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_H_

#include <any>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/protobuf_util.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.pb.h"
#include "xla/stream_executor/host_or_device_scalar.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"

namespace stream_executor::gpu {

absl::StatusOr<blas::DataType> AsBlasDataType(xla::PrimitiveType dtype);

absl::StatusOr<xla::PrimitiveType> AsXlaPrimitiveType(blas::DataType dtype);

absl::StatusOr<blas::ComputationType> GetBlasComputationType(
    xla::PrecisionConfig::Algorithm algorithm, xla::PrimitiveType lhs_dtype,
    xla::PrimitiveType output_dtype, int64_t compute_precision);

// Returns the type for the alpha and beta scalars.
blas::DataType GetScaleType(blas::DataType c_type,
                            blas::ComputationType computation_type);

struct MatrixLayout {  // plain MatrixLayout which is extended with create
                       // functions in matmul_utils.h
  enum class Order {
    kRowMajor,     // Elements in the same row are contiguous in memory.
    kColumnMajor,  // Elements in the same column are contiguous in memory.
  };

  MatrixLayout(xla::PrimitiveType dtype_, int64_t num_rows_, int64_t num_cols_,
               Order order_, int64_t batch_size_ = 1,
               std::optional<int64_t> leading_dim_stride_ = {},
               std::optional<int64_t> batch_stride_ = {},
               std::optional<blas::Transpose> transpose_ = {});

  void Transpose();

  xla::PrimitiveType dtype;
  // `num_rows` / `num_cols` are for the "logical" matrix shape:
  // i.e. the contracting dim has size `num_cols` for LHS operands and
  // `num_rows` for RHS operands.
  int64_t num_rows;
  int64_t num_cols;
  Order order;
  int64_t batch_size;
  int64_t leading_dim_stride;
  // `batch_stride` is set to `0` when `batch_size == 1`.
  int64_t batch_stride;
  blas::Transpose transpose;

  static absl::StatusOr<MatrixLayout> FromProto(
      const xla::GemmConfigProto::MatrixLayout& proto);
  xla::GemmConfigProto::MatrixLayout ToProto() const;
};

// compact version of the matrix layout to be used to pass matrices
// to underlying blas API
struct MatrixDescriptor {
  DeviceMemoryBase data;
  int64_t leading_dim_stride = 0;
  int64_t batch_stride = 0;
  blas::DataType type{};
  blas::Transpose transpose{};

  template <typename T>
  DeviceMemory<T> cast() const {
    return DeviceMemory<T>(data);
  }
};

struct OutputMatrixDescriptor : public MatrixDescriptor {
  OutputMatrixDescriptor(MatrixDescriptor&& parent) noexcept
      : MatrixDescriptor(std::move(parent)) {}
  int64_t batch_size = 0;
  int64_t m = 0, n = 0, k = 0;
  blas::ComputationType compute_type{};
};

// BLAS GeMM's output is column-major. If we require row-major, use identity:
// C^T = (A @ B)^T = B^T @ A^T.
bool MakeOutputColumnMajor(MatrixLayout& lhs, MatrixLayout& rhs,
                           MatrixLayout& output, MatrixLayout* c = nullptr);

struct GemmConfig {  // plain GemmConfig which is extended with create functions
                     // in matmul_utils.h
  MatrixLayout lhs_layout;
  MatrixLayout rhs_layout;
  MatrixLayout c_layout;
  MatrixLayout output_layout;
  xla::complex128 alpha;
  double beta;
  int64_t compute_precision;
  // PrecisionConfig-level algorithm
  xla::PrecisionConfig::Algorithm precision_algorithm;
  // BLAS-library-level algorithm.
  std::optional<int64_t> algorithm;
  bool grad_x;
  bool grad_y;
  std::optional<blas::ComputationType> compute_type;

  static absl::StatusOr<GemmConfig> FromProto(
      const xla::GemmConfigProto& proto);
  xla::GemmConfigProto ToProto() const;
};

struct BlasLt {
  enum class Epilogue {
    kDefault = 1,                   // No special postprocessing
    kReLU = 2,                      // Apply point-wise ReLU function
    kBias = 4,                      // Add broadcasted bias vector
    kBiasThenReLU = kBias | kReLU,  // Apply bias and then ReLU transform
    kGELU = 32,                // Apply GELU point-wise transform to the results
    kGELUWithAux = 32 | 1024,  // Apply GELU with auxiliary output.
    kBiasThenGELU = kBias | kGELU,  // Apply bias and then approximate GELU.
    kBiasThenGELUWithAux = kBiasThenGELU | 1024,
  };

  // Describes the location of pointers for the scaling factors alpha and beta.
  enum class PointerMode {
    kHost,
    kDevice,
  };

  struct MatmulAlgorithm {
    std::any opaque_algo;
    size_t workspace_size;
  };

  struct MemoryArgs {
    DeviceMemoryBase a, b, c, d;                          // these are mandatory
    DeviceMemoryBase bias, aux;                           // these may be null
    DeviceMemoryBase a_scale, b_scale, c_scale, d_scale;  // these may be null
    DeviceMemoryBase d_amax;                              // this may be null
    DeviceMemoryBase workspace;                           // either workspace or
    ScratchAllocator* scratch_allocator;  // scratch_allocator must not be null
  };

  struct MatmulPlan {
    // This function is to be removed once TF interface is fixed,
    // see tensorflow/core/kernels/matmul_util.cc
    absl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a, DeviceMemoryBase b,
        DeviceMemoryBase c, DeviceMemoryBase d,
        DeviceMemoryBase bias,  // may be null
        DeviceMemoryBase aux,   // may be null
        DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
        DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
        DeviceMemoryBase d_amax, const MatmulAlgorithm& algorithm,
        ScratchAllocator& scratch_allocator,
        blas::ProfileResult* profile_result = nullptr) const {
      // Temporary hack until Tensorflow side is fixed
      TF_RETURN_IF_ERROR(
          const_cast<MatmulPlan*>(this)->SetAlgorithm(algorithm));
      return ExecuteOnStream(
          stream,
          MemoryArgs{a, b, c, d, bias, aux, a_scale, b_scale, c_scale, d_scale,
                     d_amax, DeviceMemoryBase{}, &scratch_allocator},
          profile_result);
    }

    // API that uses scratch_allocator to allocate workspace.
    // This version is used by TF: see tensorflow/core/kernels/matmul_util.cc
    absl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a, DeviceMemoryBase b,
        DeviceMemoryBase c, DeviceMemoryBase d,
        DeviceMemoryBase bias,  // may be null
        DeviceMemoryBase aux,   // may be null
        DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
        DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
        DeviceMemoryBase d_amax, ScratchAllocator& scratch_allocator,
        blas::ProfileResult* profile_result = nullptr) const {
      return ExecuteOnStream(
          stream,
          MemoryArgs{a, b, c, d, bias, aux, a_scale, b_scale, c_scale, d_scale,
                     d_amax, DeviceMemoryBase{}, &scratch_allocator},
          profile_result);
    }

    // API that uses pre-allocated buffer as workspace.
    absl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a, DeviceMemoryBase b,
        DeviceMemoryBase c, DeviceMemoryBase d,
        DeviceMemoryBase bias,  // may be null
        DeviceMemoryBase aux,   // may be null
        DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
        DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
        DeviceMemoryBase d_amax, DeviceMemoryBase workspace,
        blas::ProfileResult* profile_result = nullptr) const {
      return ExecuteOnStream(
          stream,
          MemoryArgs{a, b, c, d, bias, aux, a_scale, b_scale, c_scale, d_scale,
                     d_amax, workspace, nullptr},
          profile_result);
    }

    // The most general form: to be implemented by derived clases.
    virtual absl::Status ExecuteOnStream(
        Stream* stream, const MemoryArgs& args,
        blas::ProfileResult* profile_result) const = 0;

    // Returns a list of supported algorithms for DoMatmul. The algorithms are
    // returned in the order of increasing estimated compute time according to
    // an internal heuristic.
    virtual absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        const Stream* stream, size_t max_algorithm_count = 128,
        size_t max_workspace_size = 1ll << 32) const = 0;

    // Algorithm must to be set before calling ExecuteOnStream function(s).
    // Usually, we call ExecuteOnStream with the same algorithm ID, hence using
    // a separate function here enables BlasLt implementations to do additional
    // optimizations (like preloading matmul kernels) once the algorithm is set.
    virtual absl::Status SetAlgorithm(const MatmulAlgorithm& algorithm) = 0;

    virtual ~MatmulPlan() {}
  };  // class MatmulPlan

  using MatmulPlanPtr = std::unique_ptr<MatmulPlan>;
  using PlanCreateFunc = absl::AnyInvocable<absl::StatusOr<MatmulPlanPtr>()>;

  virtual absl::Status Init() = 0;

  virtual absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(
      const GemmConfig& cfg, Epilogue epilogue) const = 0;

  static BlasLt* Get(const Stream* stream);

  // convenience function to create MatmulPlan directly using stream
  static absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const Stream* stream,
                                                     const GemmConfig& cfg,
                                                     Epilogue epilogue);

  absl::StatusOr<MatmulPlan*> GetOrCreateMatmulPlan(const std::string& key,
                                                    PlanCreateFunc create);

  void ClearMatmulPlanCache();
  size_t GetMatmulPlanCacheSize() const;

  virtual ~BlasLt() {}

 protected:
  mutable absl::Mutex plan_cache_mu_;
  absl::flat_hash_map<std::string, MatmulPlanPtr> plan_cache_
      ABSL_GUARDED_BY(plan_cache_mu_);
};  // class BlasLt

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_H_
