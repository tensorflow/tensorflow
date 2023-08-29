/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.
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

#include <algorithm>
#include <climits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "rocm/rocm_config.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/hip_blas_utils.h"

#if TF_HIPBLASLT
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/hip_blas_lt.h"
#include "tensorflow/compiler/xla/stream_executor/rocm/rocm_blas.h"
#include "tensorflow/compiler/xla/stream_executor/scratch_allocator.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"

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
namespace rocm {
namespace {

template <typename T>
tsl::Status SetAttr(hipblasLtMatrixLayout_t handle,
                    hipblasLtMatrixLayoutAttribute_t attr, T value) {
  return SET_ATTR(wrap::hipblasLtMatrixLayoutSetAttribute, handle, attr, value);
}

template <typename T>
tsl::StatusOr<T> GetAttr(hipblasLtMatrixLayout_t handle,
                         hipblasLtMatrixLayoutAttribute_t attr) {
  return GET_ATTR(hipblasLtMatrixLayoutGetAttribute, handle, attr, T);
}

template <typename T>
tsl::Status SetAttr(hipblasLtMatmulDesc_t handle,
                    hipblasLtMatmulDescAttributes_t attr, T value) {
  return SET_ATTR(wrap::hipblasLtMatmulDescSetAttribute, handle, attr, value);
}

template <typename T>
tsl::StatusOr<T> GetAttr(hipblasLtMatmulDesc_t handle,
                         hipblasLtMatmulDescAttributes_t attr) {
  return GET_ATTR(hipblasLtMatmulDescGetAttribute, handle, attr, T);
}

template <typename T>
tsl::Status SetAttr(hipblasLtMatmulPreference_t handle,
                    hipblasLtMatmulPreferenceAttributes_t attr, T value) {
  return SET_ATTR(wrap::hipblasLtMatmulPreferenceSetAttribute, handle, attr,
                  value);
}

hipblasPointerMode_t AsHipblasLtPointerMode(BlasLt::PointerMode pointer_mode) {
  switch (pointer_mode) {
    case BlasLt::PointerMode::kHost:
      return HIPBLAS_POINTER_MODE_HOST;
    case BlasLt::PointerMode::kDevice:
      return HIPBLAS_POINTER_MODE_DEVICE;
  }
}

tsl::StatusOr<hipblasLtEpilogue_t> AsHipblasLtEpilogue(
    BlasLt::Epilogue epilogue) {
  switch (epilogue) {
    case BlasLt::Epilogue::kDefault:
      return HIPBLASLT_EPILOGUE_DEFAULT;
    case BlasLt::Epilogue::kReLU:
      return HIPBLASLT_EPILOGUE_RELU;
    case BlasLt::Epilogue::kBias:
      return HIPBLASLT_EPILOGUE_BIAS;
    case BlasLt::Epilogue::kBiasThenReLU:
      return HIPBLASLT_EPILOGUE_RELU_BIAS;
    case BlasLt::Epilogue::kGELU:
      return HIPBLASLT_EPILOGUE_GELU;
    default:
      return tsl::errors::Internal("Unsupported epilogue");
  }
}

}  // namespace

tsl::Status BlasLt::Init() {
  hipblasLtHandle_t blas_lt;
  SE_HIPBLAS_RETURN_IF_ERROR(wrap::hipblasLtCreate(&blas_lt));
  absl::MutexLock lock(&mu_);
  blas_lt_.reset(blas_lt);
  return tsl::OkStatus();
}

/*static*/ tsl::StatusOr<BlasLt::MatrixLayout> BlasLt::MatrixLayout::Create(
    blas::DataType type, size_t num_rows, size_t num_cols,
    BlasLt::MatrixLayout::Order order, size_t batch_size,
    std::optional<int64_t> leading_dim_stride,
    std::optional<int64_t> batch_stride) {
  if (!leading_dim_stride) {
    leading_dim_stride = (order == Order::kRowMajor) ? num_cols : num_rows;
  }

  hipblasLtMatrixLayout_t hip_layout;
  SE_HIPBLAS_RETURN_IF_ERROR(wrap::hipblasLtMatrixLayoutCreate(
      &hip_layout, AsHipblasDataType(type), num_rows, num_cols,
      *leading_dim_stride));
  // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatrixLayout layout(hip_layout);
  if (order != Order::kColumnMajor)
    return tsl::errors::Internal(
        "HipblasLT does not support row-major matrices");
  TF_RETURN_IF_ERROR(SetAttr(hip_layout, HIPBLASLT_MATRIX_LAYOUT_BATCH_COUNT,
                             static_cast<int32_t>(batch_size)));

  if (!batch_stride) {
    batch_stride = (batch_size > 1) ? num_rows * num_cols : 0;
  }

  TF_RETURN_IF_ERROR(SetAttr(
      hip_layout, HIPBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, *batch_stride));
  return std::move(layout);
}

hipblasDatatype_t BlasLt::MatrixLayout::type() const { return HIPBLAS_R_32F; }

/*static*/ tsl::StatusOr<BlasLt::MatmulDesc> BlasLt::MatmulDesc::Create(
    blas::ComputationType compute_type, blas::DataType scale_type,
    blas::Transpose trans_a, blas::Transpose trans_b, BlasLt::Epilogue epilogue,
    BlasLt::PointerMode pointer_mode) {
  hipblasLtMatmulDesc_t hip_desc;
  VLOG(2) << "BlasLt::MatmulDesc::Create compute_type" << int(compute_type)
          << " scale_type " << int(scale_type) << " epilogue " << int(epilogue)
          << " pointer_mode " << int(pointer_mode);
  SE_HIPBLAS_RETURN_IF_ERROR(wrap::hipblasLtMatmulDescCreate(
      &hip_desc, AsHipblasComputeType(compute_type),
      AsHipblasDataType(scale_type)));
  // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatmulDesc desc(hip_desc);
  if (pointer_mode != BlasLt::PointerMode::kHost)
    return tsl::errors::Internal("hipblaslt does not support device pointers");
  TF_RETURN_IF_ERROR(SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_TRANSA,
                             AsHipblasOperation(trans_a)));
  TF_RETURN_IF_ERROR(SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_TRANSB,
                             AsHipblasOperation(trans_b)));
  TF_ASSIGN_OR_RETURN(hipblasLtEpilogue_t epi, AsHipblasLtEpilogue(epilogue));
  TF_RETURN_IF_ERROR(SetAttr(hip_desc, HIPBLASLT_MATMUL_DESC_EPILOGUE, epi));
  return std::move(desc);
}

hipblasLtComputeType_t BlasLt::MatmulDesc::compute_type() const {
  return HIPBLASLT_COMPUTE_F32;
}

hipblasDatatype_t BlasLt::MatmulDesc::scale_type() const {
  return HIPBLAS_R_32F;
}

hipblasPointerMode_t BlasLt::MatmulDesc::pointer_mode() const {
  return HIPBLAS_POINTER_MODE_HOST;
}

/*static*/ tsl::StatusOr<BlasLt::MatmulPreference>
BlasLt::MatmulPreference::Create(size_t max_workspace_size) {
  hipblasLtMatmulPreference_t hip_preference;
  SE_HIPBLAS_RETURN_IF_ERROR(
      wrap::hipblasLtMatmulPreferenceCreate(&hip_preference));
  // Wrap hipblas handle immediately, so it is cleaned up if an error occurs.
  BlasLt::MatmulPreference preference(hip_preference);
  TF_RETURN_IF_ERROR(SetAttr<uint64_t>(
      hip_preference, HIPBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
      max_workspace_size));
  return std::move(preference);
}

tsl::StatusOr<std::vector<BlasLt::MatmulAlgorithm>> BlasLt::GetMatmulAlgorithms(
    const BlasLt::MatmulPlan& plan, const BlasLt::MatmulPreference& preference,
    size_t max_algorithm_count) {
  max_algorithm_count = std::min(max_algorithm_count, size_t{INT_MAX});
  std::vector<hipblasLtMatmulHeuristicResult_t> results(max_algorithm_count);
  {
    absl::MutexLock lock(&mu_);
    TF_RET_CHECK(blas_lt_ != nullptr);

    gpu::ScopedActivateExecutorContext sac{parent_};

    int found_algorithm_count = 0;
    auto error = wrap::hipblasLtMatmulAlgoGetHeuristic(
        blas_lt_.get(), plan.op_desc.get(), plan.a_desc.get(),
        plan.b_desc.get(), plan.c_desc.get(), plan.d_desc.get(),
        preference.get(), max_algorithm_count, results.data(),
        &found_algorithm_count);
    if (error != 0) {
      printf("hipblasLtMatmulAlgoGetHeuristic return %d\n",
             static_cast<int>(error));
      fflush(stdout);
      SE_HIPBLAS_RETURN_IF_ERROR(error);
    }
    results.resize(found_algorithm_count);
  }

  std::vector<BlasLt::MatmulAlgorithm> algorithms;
  algorithms.reserve(results.size());
  for (const hipblasLtMatmulHeuristicResult_t& result : results) {
    if (result.state == HIPBLAS_STATUS_SUCCESS) {  // Skip failed algos.
      algorithms.push_back({result.algo, result.workspaceSize});
    }
  }
  return std::move(algorithms);
}

tsl::Status BlasLt::DoMatmul(Stream* stream, const BlasLt::MatmulPlan& plan,
                             const void* alpha, DeviceMemoryBase a,
                             DeviceMemoryBase b, const void* beta,
                             DeviceMemoryBase c, DeviceMemoryBase d,
                             const BlasLt::MatmulAlgorithm& algorithm,
                             ScratchAllocator& scratch_allocator,
                             DeviceMemoryBase bias, DeviceMemoryBase aux,
                             DeviceMemoryBase a_scale, DeviceMemoryBase b_scale,
                             DeviceMemoryBase c_scale, DeviceMemoryBase d_scale,
                             DeviceMemoryBase d_amax,
                             blas::ProfileResult* profile_result) {
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
    absl::MutexLock lock(&mu_);
    TF_RET_CHECK(blas_lt_ != nullptr);
    // We must set the bias and aux pointers while holding the mutex, to avoid a
    // potential race condition from multiple threads sharing the same plan.
    if (bias != nullptr) {
      TF_RETURN_IF_ERROR(SetAttr(plan.op_desc.get(),
                                 HIPBLASLT_MATMUL_DESC_BIAS_POINTER,
                                 bias.opaque()));
    }

    if ((a_scale != nullptr) || (b_scale != nullptr) || (c_scale != nullptr) ||
        (d_scale != nullptr))
      return tsl::errors::Internal("hipblaslt does not support scale");

    if (d_amax != nullptr)
      return tsl::errors::Internal("hipblaslt does not support amax");

    if (aux != nullptr)
      return tsl::errors::Internal(
          "hipblaslt does not support auxiliary inputs / outputs");

    gpu::ScopedActivateExecutorContext sac{parent_};

    SE_HIPBLAS_RETURN_IF_ERROR(wrap::hipblasLtMatmul(
        blas_lt_.get(), plan.op_desc.get(), alpha, a.opaque(),
        plan.a_desc.get(), b.opaque(), plan.b_desc.get(), beta, c.opaque(),
        plan.c_desc.get(), d.opaque(), plan.d_desc.get(), &algorithm.algo,
        workspace, algorithm.workspace_size, gpu::AsGpuStreamValue(stream)));
  }

  if (profile_result != nullptr) {
    TF_ASSIGN_OR_RETURN(absl::Duration elapsed, timer->GetElapsedDuration());
    profile_result->set_is_valid(true);
    profile_result->set_elapsed_time_in_ms(absl::ToDoubleMilliseconds(elapsed));
  }
  return tsl::OkStatus();
}

BlasLt* GetBlasLt(Stream* stream) {
  gpu::ROCMBlas* blas =
      dynamic_cast<gpu::ROCMBlas*>(stream->parent()->AsBlas());
  return (blas != nullptr) ? &blas->blas_lt() : nullptr;
}

}  // namespace rocm
}  // namespace stream_executor

#endif  // TF_HIPBLASLT
