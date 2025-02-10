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

#include "xla/stream_executor/cuda/cuda_blas_lt.h"

#include <Eigen/Core>
#include <algorithm>
#include <any>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "third_party/gpus/cuda/include/cublasLt.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/library_types.h"
#include "xla/primitive_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/activate_context.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/cuda/cuda_blas.h"
#include "xla/stream_executor/cuda/cuda_blas_utils.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event_based_timer.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/stream.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/ml_dtypes.h"
#include "tsl/platform/statusor.h"

#define SET_ATTR(setter, handle, attr, value) \
  ToStatus(setter(handle, attr, &value, sizeof(decltype(value))), #setter)

#define GET_ATTR(getter, handle, attr, ValueT)                            \
  [&]() -> absl::StatusOr<ValueT> {                                       \
    ValueT value;                                                         \
    TF_RETURN_IF_ERROR(ToStatus(                                          \
        getter(handle, attr, &value, sizeof(ValueT), nullptr), #getter)); \
    return std::move(value);                                              \
  }()

namespace stream_executor {

namespace cuda {

using ::xla::complex128;
using ::xla::complex64;
namespace {

template <typename T>
absl::Status SetAttr(cublasLtMatrixLayout_t handle,
                     cublasLtMatrixLayoutAttribute_t attr, T value) {
  return SET_ATTR(cublasLtMatrixLayoutSetAttribute, handle, attr, value);
}

template <typename T>
absl::StatusOr<T> GetAttr(cublasLtMatrixLayout_t handle,
                          cublasLtMatrixLayoutAttribute_t attr) {
  return GET_ATTR(cublasLtMatrixLayoutGetAttribute, handle, attr, T);
}

template <typename T>
absl::Status SetAttr(cublasLtMatmulDesc_t handle,
                     cublasLtMatmulDescAttributes_t attr, T value) {
  return SET_ATTR(cublasLtMatmulDescSetAttribute, handle, attr, value);
}

template <typename T>
absl::StatusOr<T> GetAttr(cublasLtMatmulDesc_t handle,
                          cublasLtMatmulDescAttributes_t attr) {
  return GET_ATTR(cublasLtMatmulDescGetAttribute, handle, attr, T);
}

template <typename T>
absl::Status SetAttr(cublasLtMatmulPreference_t handle,
                     cublasLtMatmulPreferenceAttributes_t attr, T value) {
  return SET_ATTR(cublasLtMatmulPreferenceSetAttribute, handle, attr, value);
}

cublasLtPointerMode_t AsCublasLtPointerMode(
    gpu::BlasLt::PointerMode pointer_mode) {
  switch (pointer_mode) {
    case gpu::BlasLt::PointerMode::kHost:
      return CUBLASLT_POINTER_MODE_HOST;
    case gpu::BlasLt::PointerMode::kDevice:
      return CUBLASLT_POINTER_MODE_DEVICE;
  }
}

absl::StatusOr<cublasLtEpilogue_t> AsCublasLtEpilogue(
    gpu::BlasLt::Epilogue epilogue) {
  switch (epilogue) {
    case gpu::BlasLt::Epilogue::kDefault:
      return CUBLASLT_EPILOGUE_DEFAULT;
    case gpu::BlasLt::Epilogue::kReLU:
      return CUBLASLT_EPILOGUE_RELU;
    case gpu::BlasLt::Epilogue::kBias:
      return CUBLASLT_EPILOGUE_BIAS;
    case gpu::BlasLt::Epilogue::kBiasThenReLU:
      return CUBLASLT_EPILOGUE_RELU_BIAS;
#if CUDA_VERSION >= 11040
    case gpu::BlasLt::Epilogue::kGELU:
      return CUBLASLT_EPILOGUE_GELU;
    case gpu::BlasLt::Epilogue::kGELUWithAux:
      return CUBLASLT_EPILOGUE_GELU_AUX;
    case gpu::BlasLt::Epilogue::kBiasThenGELU:
      return CUBLASLT_EPILOGUE_GELU_BIAS;
    case gpu::BlasLt::Epilogue::kBiasThenGELUWithAux:
      return CUBLASLT_EPILOGUE_GELU_AUX_BIAS;
#else
    case gpu::BlasLt::Epilogue::kGELU:
    case gpu::BlasLt::Epilogue::kGELUWithAux:
    case gpu::BlasLt::Epilogue::kBiasThenGELU:
    case gpu::BlasLt::Epilogue::kBiasThenGELUWithAux:
      return absl::InternalError("GELU epilogues require cublasLt >= 11.4");
#endif
  }
}

}  // namespace

absl::Status BlasLt::Init() {
  cublasLtHandle_t blas_lt;
  SE_CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&blas_lt));
  absl::MutexLock lock(&mu_);
  blas_lt_.reset(blas_lt);
  return absl::OkStatus();
}

/*static*/ absl::StatusOr<BlasLt::MatrixLayout> BlasLt::MatrixLayout::Create(
    const gpu::MatrixLayout& m) {
  TF_ASSIGN_OR_RETURN(auto type, gpu::AsBlasDataType(m.dtype));

  cublasLtMatrixLayout_t cu_layout;
  SE_CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&cu_layout, AsCudaDataType(type), m.num_rows,
                                 m.num_cols, m.leading_dim_stride));
  // Wrap cublas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatrixLayout layout(cu_layout);
  TF_RETURN_IF_ERROR(
      SetAttr(cu_layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
              int32_t{(m.order == gpu::MatrixLayout::Order::kRowMajor)
                          ? CUBLASLT_ORDER_ROW
                          : CUBLASLT_ORDER_COL}));
  TF_RETURN_IF_ERROR(SetAttr(cu_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                             static_cast<int32_t>(m.batch_size)));

  VLOG(2) << "MatrixLayout::Create: num_rows: " << m.num_rows
          << " num_cols:" << (int)m.num_cols << ", order: " << (int)m.order
          << "," << " batchsz " << m.batch_size
          << " leaddimstride: " << m.leading_dim_stride
          << " batch_stride: " << m.batch_stride;

  TF_RETURN_IF_ERROR(SetAttr(
      cu_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, m.batch_stride));
  return std::move(layout);
}

cudaDataType_t BlasLt::MatrixLayout::type() const {
  return static_cast<cudaDataType_t>(
      GetAttr<uint32_t>(handle_.get(), CUBLASLT_MATRIX_LAYOUT_TYPE).value());
}

/*static*/ absl::StatusOr<BlasLt::MatmulDesc> BlasLt::MatmulDesc::Create(
    blas::ComputationType compute_type, blas::DataType scale_type,
    blas::Transpose trans_a, blas::Transpose trans_b,
    gpu::BlasLt::Epilogue epilogue, bool enable_fast_accum,
    PointerMode pointer_mode) {
  VLOG(2) << "MatmulDesc::Create: compute_type: " << (int)compute_type
          << " scale:" << (int)scale_type << " trans a/b: " << (int)trans_a
          << "," << (int)trans_b << " epilogue:" << (int)epilogue
          << " pointer: " << (int)pointer_mode;

  cublasLtMatmulDesc_t cu_desc;
  SE_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulDescCreate(
      &cu_desc, AsCublasComputeType(compute_type), AsCudaDataType(scale_type)));
  // Wrap cublas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatmulDesc desc(cu_desc);
  TF_RETURN_IF_ERROR(SetAttr(cu_desc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                             AsCublasLtPointerMode(pointer_mode)));
  TF_RETURN_IF_ERROR(SetAttr(cu_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                             AsCublasOperation(trans_a)));
  TF_RETURN_IF_ERROR(SetAttr(cu_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                             AsCublasOperation(trans_b)));
  TF_ASSIGN_OR_RETURN(cublasLtEpilogue_t epi, AsCublasLtEpilogue(epilogue));
  TF_RETURN_IF_ERROR(SetAttr(cu_desc, CUBLASLT_MATMUL_DESC_EPILOGUE, epi));
  // The CUBLASLT_MATMUL_DESC_FAST_ACCUM flag only impacts FP8 gemms. It speeds
  // up gemms at the expense of accumulation precision. In practice, it is safe
  // to set on the forward pass but not the backward pass.
  TF_RETURN_IF_ERROR(SetAttr(cu_desc, CUBLASLT_MATMUL_DESC_FAST_ACCUM,
                             static_cast<int8_t>(enable_fast_accum)));
  return std::move(desc);
}

cublasComputeType_t BlasLt::MatmulDesc::compute_type() const {
  return static_cast<cublasComputeType_t>(
      GetAttr<int32_t>(handle_.get(), CUBLASLT_MATMUL_DESC_COMPUTE_TYPE)
          .value());
}

cudaDataType_t BlasLt::MatmulDesc::scale_type() const {
  return static_cast<cudaDataType_t>(
      GetAttr<int32_t>(handle_.get(), CUBLASLT_MATMUL_DESC_SCALE_TYPE).value());
}

cublasLtPointerMode_t BlasLt::MatmulDesc::pointer_mode() const {
  return static_cast<cublasLtPointerMode_t>(
      GetAttr<int32_t>(handle_.get(), CUBLASLT_MATMUL_DESC_POINTER_MODE)
          .value());
}

auto BlasLt::MatmulPlan::GetAlgorithms(size_t max_algorithm_count,
                                       size_t max_workspace_size) const
    -> absl::StatusOr<std::vector<MatmulAlgorithm>> {
  max_algorithm_count = std::min(max_algorithm_count, size_t{INT_MAX});
  std::vector<cublasLtMatmulHeuristicResult_t> results(max_algorithm_count);
  {
    absl::MutexLock lock(&blas_lt_ref_.mu_);
    TF_RET_CHECK(blas_lt_ref_.blas_lt_ != nullptr);

    cublasLtMatmulPreference_t cu_preference;
    SE_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulPreferenceCreate(&cu_preference));
    // Wrap cublas handle immediately, so it is cleaned up if an error occurs.
    // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
    Owned<cublasLtMatmulPreference_t> preference(
        cu_preference, cublasLtMatmulPreferenceDestroy);

    TF_RETURN_IF_ERROR(SetAttr<uint64_t>(
        cu_preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        max_workspace_size));

    std::unique_ptr<ActivateContext> activation =
        blas_lt_ref_.parent_->Activate();

    int found_algorithm_count = 0;
    SE_CUBLAS_RETURN_IF_ERROR(cublasLtMatmulAlgoGetHeuristic(
        blas_lt_ref_.blas_lt_.get(), op_desc_.get(), a_desc_.get(),
        b_desc_.get(), c_desc_.get(), d_desc_.get(), preference.get(),
        max_algorithm_count, results.data(), &found_algorithm_count));
    results.resize(found_algorithm_count);
  }

  std::vector<MatmulAlgorithm> algorithms;
  algorithms.reserve(results.size());
  for (const cublasLtMatmulHeuristicResult_t& result : results) {
    if (result.state == CUBLAS_STATUS_SUCCESS) {  // Skip failed algos.
      algorithms.push_back({result.algo, result.workspaceSize});
    }
  }
  return std::move(algorithms);
}

namespace {

bool IsFastAccumEnabled(const xla::PrecisionConfig::Algorithm algorithm,
                        xla::PrimitiveType lhs_type,
                        xla::PrimitiveType rhs_type,
                        int64_t compute_precision) {
  if (algorithm == xla::PrecisionConfig::ALG_UNSET) {
    return (xla::primitive_util::IsF8Type(lhs_type) ||
            xla::primitive_util::IsF8Type(rhs_type)) &&
           compute_precision == 0;
  }

  return algorithm ==
         xla::PrecisionConfig::ALG_DOT_ANY_F8_ANY_F8_F32_FAST_ACCUM;
}

}  // namespace

auto BlasLt::GetMatmulPlan(const gpu::GemmConfig& cfg,
                           gpu::BlasLt::Epilogue epilogue) const
    -> absl::StatusOr<MatmulPlanPtr> {
  auto lhs_layout = cfg.lhs_layout, rhs_layout = cfg.rhs_layout,
       output_layout = cfg.output_layout, c_layout = cfg.c_layout;
  // cublasLt matmul requires batch sizes to be equal. If only one operand has a
  // batch, the other will be broadcast (as its batch_stride == 0).
  size_t batch_size = std::max(lhs_layout.batch_size, rhs_layout.batch_size);
  lhs_layout.batch_size = batch_size;
  rhs_layout.batch_size = batch_size;

  bool must_swap_operands =
      MakeOutputColumnMajor(lhs_layout, rhs_layout, output_layout, &c_layout);

  // Do not transpose either input. Note the cuBLASLt documentation somewhat
  // incorrectly claims "A must be transposed and B non-transposed" when A and B
  // are FP8 (https://docs.nvidia.com/cuda/cublas/#cublasltmatmul). In reality,
  // this is only true if A and B are column-major. If A is row-major, A must
  // *not* be transposed, and if B is row-major, B must be transposed. We never
  // transpose A or B, and expect the caller to ensure A is row-major and B is
  // column when A and B are FP8.
  auto trans_a = lhs_layout.transpose, trans_b = rhs_layout.transpose;

  if (xla::primitive_util::IsF8Type(lhs_layout.dtype) &&
      lhs_layout.order == gpu::MatrixLayout::Order::kColumnMajor) {
    return xla::Internal("The F8 LHS must be row-major");
  }
  if (xla::primitive_util::IsF8Type(rhs_layout.dtype) &&
      rhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    return xla::Internal("The F8 RHS must be column-major");
  }

  TF_ASSIGN_OR_RETURN(auto output_dtype,
                      gpu::AsBlasDataType(output_layout.dtype));

  auto compute_type = cfg.compute_type;
  if (!compute_type) {  // obtain compute_type unless provided by the user
    TF_ASSIGN_OR_RETURN(compute_type,
                        gpu::GetBlasComputationType(
                            cfg.precision_algorithm, lhs_layout.dtype,
                            output_layout.dtype, cfg.compute_precision));
  }

  // FP8 matmuls have a fast accumulation mode that is less precise than the
  // default accumulation mode. Use the fast accumulation mode if the compute
  // precision is DEFAULT.
  bool enable_fast_accum =
      IsFastAccumEnabled(cfg.precision_algorithm, lhs_layout.dtype,
                         rhs_layout.dtype, cfg.compute_precision);
  TF_ASSIGN_OR_RETURN(
      auto op_desc,
      MatmulDesc::Create(*compute_type,
                         gpu::GetScaleType(output_dtype, *compute_type),
                         trans_a, trans_b, epilogue, enable_fast_accum));

  TF_ASSIGN_OR_RETURN(auto a_desc, MatrixLayout::Create(lhs_layout));
  TF_ASSIGN_OR_RETURN(auto b_desc, MatrixLayout::Create(rhs_layout));
  TF_ASSIGN_OR_RETURN(auto c_desc, MatrixLayout::Create(c_layout));
  TF_ASSIGN_OR_RETURN(auto d_desc, MatrixLayout::Create(output_layout));

  return std::make_unique<MatmulPlan>(*this, std::move(op_desc),
                                      std::move(a_desc), std::move(b_desc),
                                      std::move(c_desc), std::move(d_desc),
                                      cfg.alpha, cfg.beta, must_swap_operands);
}

absl::Status BlasLt::MatmulPlan::ValidateInputs(
    blas::DataType scale_type, bool alpha_on_device, bool beta_on_device,
    blas::DataType A_type, blas::DataType B_type, blas::DataType C_type,
    blas::DataType D_type) const {
  if (AsCudaDataType(scale_type) != op_desc_.scale_type()) {
    return absl::InvalidArgumentError("mismatched scale types");
  }

  bool expect_scale_factor_on_device =
      (op_desc_.pointer_mode() == CUBLASLT_POINTER_MODE_DEVICE);

  if (alpha_on_device != expect_scale_factor_on_device) {
    return absl::InvalidArgumentError("wrong location for alpha");
  }

  if (beta_on_device != expect_scale_factor_on_device) {
    return absl::InvalidArgumentError("wrong location for beta");
  }

  if (AsCudaDataType(A_type) != a_desc_.type()) {
    return absl::InvalidArgumentError("mismatched A matrix types");
  }

  if (AsCudaDataType(B_type) != b_desc_.type()) {
    return absl::InvalidArgumentError("mismatched B matrix types");
  }

  if (AsCudaDataType(C_type) != c_desc_.type()) {
    return absl::InvalidArgumentError("mismatched C matrix types");
  }

  if (AsCudaDataType(D_type) != d_desc_.type()) {
    return absl::InvalidArgumentError("mismatched D matrix types");
  }

  return absl::OkStatus();
}

absl::Status BlasLt::MatmulPlan::DoMatmul(
    Stream* stream, const void* alpha, DeviceMemoryBase a, DeviceMemoryBase b,
    const void* beta, DeviceMemoryBase c, DeviceMemoryBase d,
    const MatmulAlgorithm& algorithm, DeviceMemoryBase bias,
    DeviceMemoryBase aux, DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
    DeviceMemoryBase c_scale, DeviceMemoryBase d_scale, DeviceMemoryBase d_amax,
    std::optional<DeviceMemoryBase> workspace,
    std::optional<ScratchAllocator*> scratch_allocator,
    blas::ProfileResult* profile_result = nullptr) const {
  std::unique_ptr<EventBasedTimer> timer;
  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(timer, stream->CreateEventBasedTimer(
                                   profile_result->warmup_run_executed()));
  }

  void* workspace_addr;
  uint64_t workspace_size = 0;
  if (workspace.has_value()) {
    workspace_addr = workspace.value().opaque();
    workspace_size = workspace.value().size();
    TF_RET_CHECK(workspace_size >= algorithm.workspace_size);
  } else if (algorithm.workspace_size > 0) {
    TF_RET_CHECK(scratch_allocator.has_value());
    TF_ASSIGN_OR_RETURN(
        DeviceMemory<uint8_t> alloc,
        scratch_allocator.value()->AllocateBytes(algorithm.workspace_size));
    workspace_addr = gpu::GpuMemoryMutable(&alloc);
    workspace_size = algorithm.workspace_size;
  }

  auto palgo = std::any_cast<cublasLtMatmulAlgo_t>(&algorithm.opaque_algo);
  {
    absl::MutexLock lock(&blas_lt_ref_.mu_);
    TF_RET_CHECK(blas_lt_ref_.blas_lt_ != nullptr);
    // We must set the bias and aux pointers while holding the mutex, to avoid a
    // potential race condition from multiple threads sharing the same plan.
    if (bias != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(
          op_desc_.get(), CUBLASLT_MATMUL_DESC_BIAS_POINTER, bias.opaque()));
    }
#if CUDA_VERSION >= 11080
    if (a_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 CUBLASLT_MATMUL_DESC_A_SCALE_POINTER,
                                 a_scale.opaque()));
    }
    if (b_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 CUBLASLT_MATMUL_DESC_B_SCALE_POINTER,
                                 b_scale.opaque()));
    }
    if (c_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 CUBLASLT_MATMUL_DESC_C_SCALE_POINTER,
                                 c_scale.opaque()));
    }
    if (d_scale != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 CUBLASLT_MATMUL_DESC_D_SCALE_POINTER,
                                 d_scale.opaque()));
    }
    if (d_amax != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 CUBLASLT_MATMUL_DESC_AMAX_D_POINTER,
                                 d_amax.opaque()));
    }
#else
    if (a_scale != nullptr || b_scale != nullptr || c_scale != nullptr ||
        d_scale != nullptr || d_amax != nullptr) {
      return absl::InternalError(
          "A/B/C/D scales and amax require cublasLt >= 11.8");
    }
#endif

    if (aux != nullptr) {
#if CUDA_VERSION >= 11040
      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_POINTER,
                                 aux.opaque()));

      // Set leading dim and batch stride of auxiliary output to match output.
      // TODO(cjfj): Set this once at initialization.
      TF_ASSIGN_OR_RETURN(
          int64_t output_leading_dim,
          GetAttr<int64_t>(d_desc_.get(), CUBLASLT_MATRIX_LAYOUT_LD));

      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_LD,
                                 output_leading_dim));

      TF_ASSIGN_OR_RETURN(
          int64_t output_batch_stride,
          GetAttr<int64_t>(d_desc_.get(),
                           CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET));

      TF_RETURN_IF_ERROR(SetAttr(op_desc_.get(),
                                 CUBLASLT_MATMUL_DESC_EPILOGUE_AUX_BATCH_STRIDE,
                                 output_batch_stride));
#else
      return absl::InternalError(
          "Auxiliary inputs / outputs require cublasLt >= 11.4");
#endif
    }

    std::unique_ptr<ActivateContext> activation =
        blas_lt_ref_.parent_->Activate();

    if (palgo != nullptr) {
      SE_CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(
          blas_lt_ref_.blas_lt_.get(), op_desc_.get(), alpha, a.opaque(),
          a_desc_.get(), b.opaque(), b_desc_.get(), beta, c.opaque(),
          c_desc_.get(), d.opaque(), d_desc_.get(), palgo, workspace_addr,
          workspace_size, gpu::AsGpuStreamValue(stream)));
    } else {
      return absl::InternalError("cublaslt: Invalid algorithm type");
    }
  }

  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed, timer->GetElapsedDuration());
    // set algorithm ID to be unique (otherwise it gets kDefaultAlgorithm ID)
    profile_result->set_algorithm(reinterpret_cast<blas::AlgorithmType>(palgo));
    profile_result->set_is_valid(true);
    profile_result->set_elapsed_time_in_ms(absl::ToDoubleMilliseconds(elapsed));
  }
  return absl::OkStatus();
}

namespace {

template <cudaDataType_t CudaT>
struct CudaToNativeT;

#if CUDA_VERSION >= 11080
template <>
struct CudaToNativeT<CUDA_R_8F_E4M3> {
  using type = tsl::float8_e4m3fn;
};
template <>
struct CudaToNativeT<CUDA_R_8F_E5M2> {
  using type = tsl::float8_e5m2;
};
#endif

template <>
struct CudaToNativeT<CUDA_R_16BF> {
  using type = Eigen::bfloat16;
};
template <>
struct CudaToNativeT<CUDA_R_16F> {
  using type = Eigen::half;
};
template <>
struct CudaToNativeT<CUDA_R_32F> {
  using type = float;
};
template <>
struct CudaToNativeT<CUDA_R_64F> {
  using type = double;
};
template <>
struct CudaToNativeT<CUDA_C_32F> {
  using type = xla::complex64;
};
template <>
struct CudaToNativeT<CUDA_C_64F> {
  using type = xla::complex128;
};

}  // namespace

absl::Status BlasLt::MatmulPlan::ExecuteOnStream(
    Stream* stream, DeviceMemoryBase a, DeviceMemoryBase b, DeviceMemoryBase c,
    DeviceMemoryBase d, DeviceMemoryBase bias, DeviceMemoryBase aux,
    DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
    DeviceMemoryBase c_scale, DeviceMemoryBase d_scale, DeviceMemoryBase d_amax,
    const MatmulAlgorithm& algorithm, std::optional<DeviceMemoryBase> workspace,
    std::optional<ScratchAllocator*> scratch_allocator,
    blas::ProfileResult* profile_result) const {
  if (must_swap_operands_) {
    std::swap(a, b);
  }

  std::tuple operand_types{a_desc_.type(), b_desc_.type(), c_desc_.type(),
                           d_desc_.type()};

#define TYPED_MATMUL(SCALENTYPE, ATYPE, BTYPE, CTYPE, DTYPE)                \
  if (operand_types == std::tuple{ATYPE, BTYPE, CTYPE, DTYPE}) {            \
    return gpu::BlasLt::MatmulPlan::DoMatmul<                               \
        SCALENTYPE, CudaToNativeT<ATYPE>::type, CudaToNativeT<BTYPE>::type, \
        CudaToNativeT<CTYPE>::type, CudaToNativeT<DTYPE>::type>(            \
        stream, alpha_, a, b, beta_, c, d, bias, aux, a_scale, b_scale,     \
        c_scale, d_scale, d_amax, algorithm, workspace, scratch_allocator,  \
        profile_result);                                                    \
  }

#if CUDA_VERSION >= 11080
  // FP8 compatible type combinations (see cuBLASLt documentation):
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_16BF)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F, CUDA_R_16F)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F, CUDA_R_32F)

  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16BF, CUDA_R_16BF)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16BF,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16BF,
               CUDA_R_8F_E5M2)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16F,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16F,
               CUDA_R_8F_E5M2)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16F, CUDA_R_16F)
  TYPED_MATMUL(float, CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_32F, CUDA_R_32F)

  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16BF, CUDA_R_16BF)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16BF,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16BF,
               CUDA_R_8F_E5M2)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16F,
               CUDA_R_8F_E4M3)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16F,
               CUDA_R_8F_E5M2)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16F, CUDA_R_16F)
  TYPED_MATMUL(float, CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_32F, CUDA_R_32F)
#endif

  // Other data types:
  TYPED_MATMUL(float, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF)
  TYPED_MATMUL(float, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_16F)
  TYPED_MATMUL(float, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_32F, CUDA_R_32F)
  TYPED_MATMUL(float, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F)
  TYPED_MATMUL(float, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F)
  TYPED_MATMUL(double, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F)
  TYPED_MATMUL(xla::complex64, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUDA_C_32F)
  TYPED_MATMUL(xla::complex128, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F, CUDA_C_64F)

#undef TYPED_MATMUL

  return xla::Internal("Unexpected dtype");
}

}  // namespace cuda

}  // namespace stream_executor
