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

#include "xla/stream_executor/cuda/cuda_blas_lt.h"

#include <algorithm>
#include <any>
#include <climits>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "third_party/gpus/cuda/include/cublasLt.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "xla/primitive_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/cuda/cuda_blas.h"
#include "xla/stream_executor/cuda/cuda_blas_utils.h"
#include "xla/stream_executor/gpu/gpu_activation.h"
#include "xla/stream_executor/gpu/gpu_blas_lt.h"
#include "xla/stream_executor/gpu/gpu_helpers.h"
#include "xla/stream_executor/gpu/gpu_stream.h"
#include "xla/stream_executor/gpu/gpu_timer.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"

#define SET_ATTR(setter, handle, attr, value) \
  ToStatus(setter(handle, attr, &value, sizeof(decltype(value))), #setter)

#define GET_ATTR(getter, handle, attr, ValueT)                            \
  [&]() -> tsl::StatusOr<ValueT> {                                        \
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
tsl::Status SetAttr(cublasLtMatrixLayout_t handle,
                    cublasLtMatrixLayoutAttribute_t attr, T value) {
  return SET_ATTR(cublasLtMatrixLayoutSetAttribute, handle, attr, value);
}

template <typename T>
tsl::StatusOr<T> GetAttr(cublasLtMatrixLayout_t handle,
                         cublasLtMatrixLayoutAttribute_t attr) {
  return GET_ATTR(cublasLtMatrixLayoutGetAttribute, handle, attr, T);
}

template <typename T>
tsl::Status SetAttr(cublasLtMatmulDesc_t handle,
                    cublasLtMatmulDescAttributes_t attr, T value) {
  return SET_ATTR(cublasLtMatmulDescSetAttribute, handle, attr, value);
}

template <typename T>
tsl::StatusOr<T> GetAttr(cublasLtMatmulDesc_t handle,
                         cublasLtMatmulDescAttributes_t attr) {
  return GET_ATTR(cublasLtMatmulDescGetAttribute, handle, attr, T);
}

template <typename T>
tsl::Status SetAttr(cublasLtMatmulPreference_t handle,
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

tsl::StatusOr<cublasLtEpilogue_t> AsCublasLtEpilogue(
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
      return tsl::errors::Internal("GELU epilogues require cublasLt >= 11.4");
#endif
  }
}

}  // namespace

tsl::Status BlasLt::Init() {
  cublasLtHandle_t blas_lt;
  SE_CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&blas_lt));
  absl::MutexLock lock(&mu_);
  blas_lt_.reset(blas_lt);
  return tsl::OkStatus();
}

/*static*/ tsl::StatusOr<BlasLt::MatrixLayout> BlasLt::MatrixLayout::Create(
    const gpu::MatrixLayout& m) {
  TF_ASSIGN_OR_RETURN(auto type, gpu::AsBlasDataType(m.dtype));

  auto leading_dim_stride = m.leading_dim_stride;
  if (!leading_dim_stride) {
    leading_dim_stride = (m.order == gpu::MatrixLayout::Order::kRowMajor)
                             ? m.num_cols
                             : m.num_rows;
  }

  cublasLtMatrixLayout_t cu_layout;
  SE_CUBLAS_RETURN_IF_ERROR(
      cublasLtMatrixLayoutCreate(&cu_layout, AsCudaDataType(type), m.num_rows,
                                 m.num_cols, *leading_dim_stride));
  // Wrap cublas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatrixLayout layout(cu_layout);
  TF_RETURN_IF_ERROR(
      SetAttr(cu_layout, CUBLASLT_MATRIX_LAYOUT_ORDER,
              int32_t{(m.order == gpu::MatrixLayout::Order::kRowMajor)
                          ? CUBLASLT_ORDER_ROW
                          : CUBLASLT_ORDER_COL}));
  TF_RETURN_IF_ERROR(SetAttr(cu_layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                             static_cast<int32_t>(m.batch_size)));

  auto batch_stride = m.batch_stride;
  if (!batch_stride) {
    batch_stride = (m.batch_size > 1) ? m.num_rows * m.num_cols : 0;
  }

  VLOG(2) << "MatrixLayout::Create: num_rows: " << m.num_rows
          << " num_cols:" << (int)m.num_cols << ", order: " << (int)m.order
          << ","
          << " batchsz " << m.batch_size
          << " leaddimstride: " << *leading_dim_stride
          << " batch_stride: " << *batch_stride;

  TF_RETURN_IF_ERROR(SetAttr(
      cu_layout, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, *batch_stride));
  return std::move(layout);
}

cudaDataType_t BlasLt::MatrixLayout::type() const {
  return static_cast<cudaDataType_t>(
      GetAttr<uint32_t>(handle_.get(), CUBLASLT_MATRIX_LAYOUT_TYPE).value());
}

/*static*/ tsl::StatusOr<BlasLt::MatmulDesc> BlasLt::MatmulDesc::Create(
    blas::ComputationType compute_type, blas::DataType scale_type,
    blas::Transpose trans_a, blas::Transpose trans_b,
    gpu::BlasLt::Epilogue epilogue, PointerMode pointer_mode) {
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
    -> tsl::StatusOr<std::vector<MatmulAlgorithm>> {
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

    gpu::ScopedActivateExecutorContext sac{blas_lt_ref_.parent_};

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

auto BlasLt::GetMatmulPlan(const gpu::GemmConfig& cfg,
                           gpu::BlasLt::Epilogue epilogue) const
    -> tsl::StatusOr<MatmulPlanPtr> {
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
  auto trans_a = lhs_layout.transpose ? *lhs_layout.transpose
                                      : blas::Transpose::kNoTranspose;
  auto trans_b = rhs_layout.transpose ? *rhs_layout.transpose
                                      : blas::Transpose::kNoTranspose;

  if (xla::primitive_util::IsF8Type(lhs_layout.dtype) &&
      lhs_layout.order == gpu::MatrixLayout::Order::kColumnMajor) {
    return xla::InternalError("The F8 LHS must be column-major");
  }
  if (xla::primitive_util::IsF8Type(rhs_layout.dtype) &&
      rhs_layout.order == gpu::MatrixLayout::Order::kRowMajor) {
    return xla::InternalError("The F8 RHS must be row-major");
  }

  TF_ASSIGN_OR_RETURN(auto output_dtype,
                      gpu::AsBlasDataType(output_layout.dtype));

  auto compute_type = cfg.compute_type;
  if (!compute_type) {  // obtain compute_type unless provided by the user
    TF_ASSIGN_OR_RETURN(compute_type, gpu::GetBlasComputationType(
                                          lhs_layout.dtype, output_layout.dtype,
                                          cfg.compute_precision));
  }

  TF_ASSIGN_OR_RETURN(
      auto op_desc,
      MatmulDesc::Create(*compute_type,
                         gpu::GetScaleType(output_dtype, *compute_type),
                         trans_a, trans_b, epilogue));

  TF_ASSIGN_OR_RETURN(auto a_desc, MatrixLayout::Create(lhs_layout));
  TF_ASSIGN_OR_RETURN(auto b_desc, MatrixLayout::Create(rhs_layout));
  TF_ASSIGN_OR_RETURN(auto c_desc, MatrixLayout::Create(c_layout));
  TF_ASSIGN_OR_RETURN(auto d_desc, MatrixLayout::Create(output_layout));

  return std::make_unique<MatmulPlan>(*this, std::move(op_desc),
                                      std::move(a_desc), std::move(b_desc),
                                      std::move(c_desc), std::move(d_desc),
                                      cfg.alpha, cfg.beta, must_swap_operands);
}

tsl::Status BlasLt::MatmulPlan::ValidateInputs(
    blas::DataType scale_type, bool alpha_on_device, bool beta_on_device,
    blas::DataType A_type, blas::DataType B_type, blas::DataType C_type,
    blas::DataType D_type) const {
  if (AsCudaDataType(scale_type) != op_desc_.scale_type()) {
    return tsl::errors::InvalidArgument("mismatched scale types");
  }

  bool expect_scale_factor_on_device =
      (op_desc_.pointer_mode() == CUBLASLT_POINTER_MODE_DEVICE);

  if (alpha_on_device != expect_scale_factor_on_device) {
    return tsl::errors::InvalidArgument("wrong location for alpha");
  }

  if (beta_on_device != expect_scale_factor_on_device) {
    return tsl::errors::InvalidArgument("wrong location for beta");
  }

  if (AsCudaDataType(A_type) != a_desc_.type()) {
    return tsl::errors::InvalidArgument("mismatched A matrix types");
  }

  if (AsCudaDataType(B_type) != b_desc_.type()) {
    return tsl::errors::InvalidArgument("mismatched B matrix types");
  }

  if (AsCudaDataType(C_type) != c_desc_.type()) {
    return tsl::errors::InvalidArgument("mismatched C matrix types");
  }

  if (AsCudaDataType(D_type) != d_desc_.type()) {
    return tsl::errors::InvalidArgument("mismatched D matrix types");
  }

  return tsl::OkStatus();
}

tsl::Status BlasLt::MatmulPlan::DoMatmul(
    Stream* stream, const void* alpha, DeviceMemoryBase a, DeviceMemoryBase b,
    const void* beta, DeviceMemoryBase c, DeviceMemoryBase d,
    const MatmulAlgorithm& algorithm, ScratchAllocator& scratch_allocator,
    DeviceMemoryBase bias, DeviceMemoryBase aux, DeviceMemoryBase a_scale,
    DeviceMemoryBase b_scale, DeviceMemoryBase c_scale,
    DeviceMemoryBase d_scale, DeviceMemoryBase d_amax,
    blas::ProfileResult* profile_result) const {
  TF_ASSIGN_OR_RETURN(
      std::optional<gpu::GpuTimer> timer,
      gpu::GpuTimer::CreateIfNeeded(gpu::AsGpuStream(stream), profile_result));

  void* workspace = nullptr;
  if (algorithm.workspace_size > 0) {
    TF_ASSIGN_OR_RETURN(
        DeviceMemory<uint8_t> alloc,
        scratch_allocator.AllocateBytes(algorithm.workspace_size));
    workspace = gpu::GpuMemoryMutable(&alloc);
  }

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
      return tsl::errors::Internal(
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
      return tsl::errors::Internal(
          "Auxiliary inputs / outputs require cublasLt >= 11.4");
#endif
    }

    gpu::ScopedActivateExecutorContext sac{blas_lt_ref_.parent_};

    if (auto palgo =
            std::any_cast<cublasLtMatmulAlgo_t>(&algorithm.opaque_algo)) {
      SE_CUBLAS_RETURN_IF_ERROR(cublasLtMatmul(
          blas_lt_ref_.blas_lt_.get(), op_desc_.get(), alpha, a.opaque(),
          a_desc_.get(), b.opaque(), b_desc_.get(), beta, c.opaque(),
          c_desc_.get(), d.opaque(), d_desc_.get(), palgo, workspace,
          algorithm.workspace_size, gpu::AsGpuStreamValue(stream)));
    } else {
      return tsl::errors::Internal("cublaslt: Invalid algorithm type");
    }
  }

  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed, timer->GetElapsedDuration());
    profile_result->set_is_valid(true);
    profile_result->set_elapsed_time_in_ms(absl::ToDoubleMilliseconds(elapsed));
  }
  return tsl::OkStatus();
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

tsl::Status BlasLt::MatmulPlan::ExecuteOnStream(
    Stream* stream, DeviceMemoryBase a, DeviceMemoryBase b, DeviceMemoryBase c,
    DeviceMemoryBase d, DeviceMemoryBase bias, DeviceMemoryBase aux,
    DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
    DeviceMemoryBase c_scale, DeviceMemoryBase d_scale, DeviceMemoryBase d_amax,
    const MatmulAlgorithm& algorithm, ScratchAllocator& scratch_allocator,
    blas::ProfileResult* profile_result) const {
  if (must_swap_operands_) {
    std::swap(a, b);
  }

  std::tuple operand_types{a_desc_.type(), b_desc_.type(), c_desc_.type(),
                           d_desc_.type()};

#define TYPED_MATMUL(SCALENTYPE, ATYPE, BTYPE, CTYPE, DTYPE)                \
  if (operand_types == std::make_tuple(ATYPE, BTYPE, CTYPE, DTYPE)) {       \
    return gpu::BlasLt::MatmulPlan::DoMatmul<                               \
        SCALENTYPE, CudaToNativeT<ATYPE>::type, CudaToNativeT<BTYPE>::type, \
        CudaToNativeT<CTYPE>::type, CudaToNativeT<DTYPE>::type>(            \
        stream, alpha_, a, b, beta_, c, d, bias, aux, a_scale, b_scale,     \
        c_scale, d_scale, d_amax, algorithm, scratch_allocator,             \
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

  return xla::InternalError("Unexpected dtype");
}

}  // namespace cuda

}  // namespace stream_executor
