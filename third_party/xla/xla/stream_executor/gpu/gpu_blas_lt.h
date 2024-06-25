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

#include "absl/status/status.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_memory.h"
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

  struct MatmulPlan {
    // DoMatmul provides two sets of API for maintaning compatibility for XLA,
    // and TF. One set API uses scratch_allocator to allocate workspace, and one
    // set API allow uses to provide pre-allocated buffer as workspace.

    // API that uses scratch_allocator to allocate workspace
    // This function is also used by Tensorflow:
    // see tensorflow/core/kernels/matmul_util.h.
    template <typename A, typename B, typename C, typename D, typename Scale>
    absl::Status DoMatmul(Stream* stream,
                          const HostOrDeviceScalar<Scale>& alpha,
                          const DeviceMemory<A>& a, const DeviceMemory<B>& b,
                          const HostOrDeviceScalar<Scale>& beta,
                          const DeviceMemory<C>& c, DeviceMemory<D>& d,
                          const MatmulAlgorithm& algorithm,
                          ScratchAllocator& scratch_allocator,
                          const DeviceMemory<C>& bias = {},
                          const DeviceMemoryBase& aux = DeviceMemory<uint8_t>{},
                          blas::ProfileResult* profile_result = nullptr) const {
      return DoMatmul(stream, alpha, a, b, beta, c, d, algorithm, bias, aux, {},
                      {}, {}, {}, {}, std::nullopt, &scratch_allocator,
                      profile_result);
    }

    // API that uses pre-allocated buffer as workspace
    template <typename A, typename B, typename C, typename D, typename Scale>
    absl::Status DoMatmul(
        Stream* stream, const HostOrDeviceScalar<Scale>& alpha,
        const DeviceMemory<A>& a, const DeviceMemory<B>& b,
        const HostOrDeviceScalar<Scale>& beta, const DeviceMemory<C>& c,
        DeviceMemory<D>& d, const MatmulAlgorithm& algorithm,
        const DeviceMemory<C>& bias = {},
        const DeviceMemoryBase& aux = DeviceMemory<uint8_t>{},
        std::optional<DeviceMemoryBase> workspace = std::nullopt,
        blas::ProfileResult* profile_result = nullptr) const {
      return DoMatmul(stream, alpha, a, b, beta, c, d, algorithm, bias, aux, {},
                      {}, {}, {}, {}, workspace, std::nullopt, profile_result);
    }

    // The most general form: uses pre-allocated buffer workspace or
    // provided scratch allocator
    template <typename A, typename B, typename C, typename D, typename Scale>
    absl::Status DoMatmul(
        Stream* stream, const HostOrDeviceScalar<Scale>& alpha,
        const DeviceMemory<A>& a, const DeviceMemory<B>& b,
        const HostOrDeviceScalar<Scale>& beta, const DeviceMemory<C>& c,
        DeviceMemory<D>& d, const MatmulAlgorithm& algorithm,
        const DeviceMemory<C>& bias = {},
        const DeviceMemoryBase& aux = DeviceMemoryBase{},
        const DeviceMemory<Scale>& a_scale = {},
        const DeviceMemory<Scale>& b_scale = {},
        const DeviceMemory<Scale>& c_scale = {},
        const DeviceMemory<Scale>& d_scale = {},
        const DeviceMemory<Scale>& d_amax = {},
        std::optional<DeviceMemoryBase> workspace = std::nullopt,
        std::optional<ScratchAllocator*> scratch_allocator = std::nullopt,
        blas::ProfileResult* profile_result = nullptr) const {
      TF_RETURN_IF_ERROR(ValidateInputs(
          blas::ToDataType<Scale>::value, alpha.on_device(), beta.on_device(),
          blas::ToDataType<A>::value, blas::ToDataType<B>::value,
          blas::ToDataType<C>::value, blas::ToDataType<D>::value));

      return DoMatmul(stream, alpha.opaque(), a, b, beta.opaque(), c, d,
                      algorithm, bias, aux, a_scale, b_scale, c_scale, d_scale,
                      d_amax, workspace, scratch_allocator, profile_result);
    }

    // API that uses scratch_allocator to allocate workspace
    absl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a_buffer, DeviceMemoryBase b_buffer,
        DeviceMemoryBase c_buffer, DeviceMemoryBase d_buffer,
        DeviceMemoryBase bias_buffer,  // may be null
        DeviceMemoryBase aux_buffer,   // may be null
        DeviceMemoryBase a_scale_buffer, DeviceMemoryBase b_scale_buffer,
        DeviceMemoryBase c_scale_buffer, DeviceMemoryBase d_scale_buffer,
        DeviceMemoryBase d_amax_buffer, const MatmulAlgorithm& algorithm,
        ScratchAllocator& scratch_allocator,
        blas::ProfileResult* profile_result = nullptr) const {
      return ExecuteOnStream(stream, a_buffer, b_buffer, c_buffer, d_buffer,
                             bias_buffer, aux_buffer, a_scale_buffer,
                             b_scale_buffer, c_scale_buffer, d_scale_buffer,
                             d_amax_buffer, algorithm, std::nullopt,
                             &scratch_allocator, profile_result);
    }

    // API that uses pre-allocated buffer as workspace
    absl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a_buffer, DeviceMemoryBase b_buffer,
        DeviceMemoryBase c_buffer, DeviceMemoryBase d_buffer,
        DeviceMemoryBase bias_buffer,  // may be null
        DeviceMemoryBase aux_buffer,   // may be null
        DeviceMemoryBase a_scale_buffer, DeviceMemoryBase b_scale_buffer,
        DeviceMemoryBase c_scale_buffer, DeviceMemoryBase d_scale_buffer,
        DeviceMemoryBase d_amax_buffer, const MatmulAlgorithm& algorithm,
        std::optional<DeviceMemoryBase> workspace,
        blas::ProfileResult* profile_result = nullptr) const {
      return ExecuteOnStream(stream, a_buffer, b_buffer, c_buffer, d_buffer,
                             bias_buffer, aux_buffer, a_scale_buffer,
                             b_scale_buffer, c_scale_buffer, d_scale_buffer,
                             d_amax_buffer, algorithm, workspace, std::nullopt,
                             profile_result);
    }

    // The most general form: to be implemented by derived clases.
    virtual absl::Status ExecuteOnStream(
        Stream* stream, DeviceMemoryBase a_buffer, DeviceMemoryBase b_buffer,
        DeviceMemoryBase c_buffer, DeviceMemoryBase d_buffer,
        DeviceMemoryBase bias_buffer,  // may be null
        DeviceMemoryBase aux_buffer,   // may be null
        DeviceMemoryBase a_scale_buffer, DeviceMemoryBase b_scale_buffer,
        DeviceMemoryBase c_scale_buffer, DeviceMemoryBase d_scale_buffer,
        DeviceMemoryBase d_amax_buffer, const MatmulAlgorithm& algorithm,
        std::optional<DeviceMemoryBase> workspace,
        std::optional<ScratchAllocator*> scratch_allocator,
        blas::ProfileResult* profile_result) const = 0;

    // Returns a list of supported algorithms for DoMatmul. The algorithms are
    // returned in the order of increasing estimated compute time according to
    // an internal heuristic.
    virtual absl::StatusOr<std::vector<MatmulAlgorithm>> GetAlgorithms(
        size_t max_algorithm_count = 128,
        size_t max_workspace_size = 1ll << 32) const = 0;

    virtual ~MatmulPlan() {}

   protected:
    // might be used internally by ExecuteOnStream in derived classes
    template <typename Scale, typename A, typename B = A, typename C = A,
              typename D = A>
    absl::Status DoMatmul(Stream* stream, xla::complex128 alpha,
                          DeviceMemoryBase a, DeviceMemoryBase b, double beta,
                          DeviceMemoryBase c, DeviceMemoryBase d,
                          DeviceMemoryBase bias, DeviceMemoryBase aux,
                          DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
                          DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
                          DeviceMemoryBase d_amax,
                          const MatmulAlgorithm& algorithm,
                          std::optional<DeviceMemoryBase> workspace,
                          std::optional<ScratchAllocator*> scratch_allocator,
                          blas::ProfileResult* profile_result = nullptr) const {
      Scale salpha;
      if constexpr (std::is_same_v<Scale, xla::complex64> ||
                    std::is_same_v<Scale, xla::complex128>) {
        salpha = static_cast<Scale>(alpha);
      } else {
        salpha = static_cast<Scale>(alpha.real());
      }
      Scale sbeta = static_cast<Scale>(beta);
      DeviceMemory<D> output(d);

      return DoMatmul<A, B, C, D, Scale>(
          stream, HostOrDeviceScalar<Scale>(salpha), DeviceMemory<A>(a),
          DeviceMemory<B>(b), HostOrDeviceScalar<Scale>(sbeta),
          DeviceMemory<C>(c), output, algorithm, DeviceMemory<C>(bias), aux,
          DeviceMemory<Scale>(a_scale), DeviceMemory<Scale>(b_scale),
          DeviceMemory<Scale>(c_scale), DeviceMemory<Scale>(d_scale),
          DeviceMemory<Scale>(d_amax), workspace, scratch_allocator,
          profile_result);
    }

    // This is used internally by template DoMatmul function to validate inputs
    virtual absl::Status ValidateInputs(
        blas::DataType scale_type, bool alpha_on_device, bool beta_on_device,
        blas::DataType A_type, blas::DataType B_type, blas::DataType C_type,
        blas::DataType D_type) const = 0;

    // The most general version to be implemented by derived classes
    virtual absl::Status DoMatmul(
        Stream* stream, const void* alpha, DeviceMemoryBase a,
        DeviceMemoryBase b, const void* beta, DeviceMemoryBase c,
        DeviceMemoryBase d, const MatmulAlgorithm& algorithm,
        DeviceMemoryBase bias, DeviceMemoryBase aux, DeviceMemoryBase a_scale,
        DeviceMemoryBase b_scale, DeviceMemoryBase c_scale,
        DeviceMemoryBase d_scale, DeviceMemoryBase d_amax,
        std::optional<DeviceMemoryBase> workspace,
        std::optional<ScratchAllocator*> scratch_allocator,
        blas::ProfileResult* profile_result = nullptr) const = 0;
  };  // class MatmulPlan

  using MatmulPlanPtr = std::unique_ptr<MatmulPlan>;

  virtual absl::Status Init() = 0;

  virtual absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(
      const GemmConfig& cfg, Epilogue epilogue) const = 0;

  static BlasLt* Get(const Stream* stream);

  // convenience function to create MatmulPlan directly using stream
  static absl::StatusOr<MatmulPlanPtr> GetMatmulPlan(const Stream* stream,
                                                     const GemmConfig& cfg,
                                                     Epilogue epilogue);

  virtual ~BlasLt() {}
};  // class BlasLt

}  // namespace stream_executor::gpu

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_BLAS_LT_H_
