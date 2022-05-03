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

#include "tensorflow/stream_executor/cuda/cuda_blas_lt.h"

#include <optional>
#include <string>
#include <utility>

#include "third_party/gpus/cuda/include/cublasLt.h"
#include "third_party/gpus/cuda/include/cublas_v2.h"
#include "third_party/gpus/cuda/include/cuda.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/cuda/cuda_blas.h"
#include "tensorflow/stream_executor/cuda/cuda_blas_utils.h"
#include "tensorflow/stream_executor/gpu/gpu_activation.h"
#include "tensorflow/stream_executor/gpu/gpu_helpers.h"
#include "tensorflow/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/stream_executor/scratch_allocator.h"
#include "tensorflow/stream_executor/stream.h"

namespace stream_executor {
namespace cuda {
namespace {

blas::DataType GetScaleType(blas::DataType c_type,
                            blas::ComputationType compute_type) {
  return ((compute_type == blas::ComputationType::kF32) &&
          (c_type != blas::DataType::kComplexFloat))
             ? blas::DataType::kFloat
             : c_type;
}

cublasLtPointerMode_t AsCublasLtPointerMode(BlasLt::PointerMode pointer_mode) {
  switch (pointer_mode) {
    case BlasLt::PointerMode::kHost:
      return CUBLASLT_POINTER_MODE_HOST;
    case BlasLt::PointerMode::kDevice:
      return CUBLASLT_POINTER_MODE_DEVICE;
  }
}

cublasLtEpilogue_t AsCublasLtEpilogue(BlasLt::Epilogue epilogue) {
  switch (epilogue) {
    case BlasLt::Epilogue::kDefault:
      return CUBLASLT_EPILOGUE_DEFAULT;
    case BlasLt::Epilogue::kReLU:
      return CUBLASLT_EPILOGUE_RELU;
    case BlasLt::Epilogue::kBias:
      return CUBLASLT_EPILOGUE_BIAS;
    case BlasLt::Epilogue::kBiasThenReLU:
      return CUBLASLT_EPILOGUE_RELU_BIAS;
    case BlasLt::Epilogue::kBiasThenGeLUApproximate:
      return CUBLASLT_EPILOGUE_GELU_BIAS;
  }
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatrixLayout_t handle,
                                    cublasLtMatrixLayoutAttribute_t attr,
                                    const T &value) {
  cublasStatus_t status =
      cublasLtMatrixLayoutSetAttribute(handle, attr, &value, sizeof(T));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatrixLayoutSetAttribute(attr=", attr,
                     ", value=", value, ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatmulAlgo_t *handle,
                                    cublasLtMatmulAlgoConfigAttributes_t attr,
                                    const T &value) {
  cublasStatus_t status =
      cublasLtMatmulAlgoConfigSetAttribute(handle, attr, &value, sizeof(T));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulAlgoConfigSetAttribute(attr=", attr,
                     ", value=", value, ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatmulPreference_t handle,
                                    cublasLtMatmulPreferenceAttributes_t attr,
                                    const T &value) {
  cublasStatus_t status =
      cublasLtMatmulPreferenceSetAttribute(handle, attr, &value, sizeof(value));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulPreferenceSetAttribute(attr=", attr,
                     ", value=", value, ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

template <typename T>
inline bool GetCublasLtAttr(const cublasLtMatmulAlgo_t *handle,
                            cublasLtMatmulAlgoConfigAttributes_t attr,
                            T *value) {
  auto mutable_handle = const_cast<cublasLtMatmulAlgo_t *>(handle);
  size_t bytes_written = 0;
  return cublasLtMatmulAlgoConfigGetAttribute(mutable_handle, attr, value,
                                              sizeof(T), &bytes_written) ==
             CUBLAS_STATUS_SUCCESS &&
         bytes_written == sizeof(T);
}

template <typename T>
inline const T &ValueForStrCat(const T &value) {
  return value;
}
template <typename T>
inline absl::Hex ValueForStrCat(T *ptr) {
  return absl::Hex(reinterpret_cast<uintptr_t>(ptr));
}

template <typename T>
inline port::Status SetCublasLtAttr(cublasLtMatmulDesc_t handle,
                                    cublasLtMatmulDescAttributes_t attr,
                                    const T &value) {
  cublasStatus_t status =
      cublasLtMatmulDescSetAttribute(handle, attr, &value, sizeof(value));
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulDescSetAttribute(attr=", attr, ", value=",
                     ValueForStrCat(value), ") failed: ", ToString(status)));
  }
  return port::Status::OK();
}

port::StatusOr<BlasLt::UniqueOpDesc> CreateCublasLtOperationDesc(
    blas::ComputationType computation_type, blas::DataType scale_type,
    BlasLt::PointerMode pointer_mode, BlasLt::Epilogue epilogue,
    blas::Transpose transa, blas::Transpose transb) {
  cublasLtMatmulDesc_t desc;
  cublasComputeType_t cublas_compute_type =
      AsCublasComputeType(computation_type);
  cudaDataType_t cuda_scale_type = AsCudaDataType(scale_type);
  cublasStatus_t status =
      cublasLtMatmulDescCreate(&desc, cublas_compute_type, cuda_scale_type);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatmulDescCreate(computation_type=",
                     computation_type, ") failed: ", ToString(status)));
  }
  BlasLt::UniqueOpDesc unique_desc(desc);
  TF_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_POINTER_MODE,
                                     AsCublasLtPointerMode(pointer_mode)));
  TF_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                     AsCublasLtEpilogue(epilogue)));
  TF_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                     AsCublasOperation(transa)));
  TF_RETURN_IF_ERROR(SetCublasLtAttr(desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                     AsCublasOperation(transb)));
  return unique_desc;
}

port::StatusOr<BlasLt::UniqueLayoutDesc> CreateCublasLtLayoutDesc(
    blas::DataType data_type, uint64_t rows, uint64 cols, int64_t ld,
    int64_t stride, int batch_count) {
  cublasLtMatrixLayout_t desc;
  cublasStatus_t status = cublasLtMatrixLayoutCreate(
      &desc, AsCudaDataType(data_type), rows, cols, ld);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(
        port::error::INTERNAL,
        absl::StrCat("cublasLtMatrixLayoutCreate failed: ", ToString(status)));
  }
  BlasLt::UniqueLayoutDesc unique_desc(desc);
  TF_RETURN_IF_ERROR(
      SetCublasLtAttr(desc, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, batch_count));
  TF_RETURN_IF_ERROR(SetCublasLtAttr(
      desc, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, stride));
  return unique_desc;
}

size_t GetDataTypeSizeBytes(blas::DataType ty) {
  switch (ty) {
    case blas::DataType::kHalf:
      return 2;
    case blas::DataType::kFloat:
      return 4;
    case blas::DataType::kDouble:
      return 8;
    case blas::DataType::kInt8:
      return 1;
    case blas::DataType::kInt32:
      return 4;
    case blas::DataType::kComplexFloat:
      return 8;
    case blas::DataType::kComplexDouble:
      return 16;
    default:
      LOG(FATAL) << "Invalid value of blas::DataType in GetDataTypeSizeBytes";
  }
}

port::StatusOr<BlasLt::UniqueMatmulPreference> CreateCublasLtMatmulPreference(
    const BlasLt::MatmulPlan &plan, size_t max_workspace_bytes) {
  cublasLtMatmulPreference_t preference;
  cublasStatus_t status = cublasLtMatmulPreferenceCreate(&preference);
  if (status != CUBLAS_STATUS_SUCCESS) {
    return port::Status(port::error::INTERNAL,
                        absl::StrCat("cublasLtMatmulPreferenceCreate failed: ",
                                     ToString(status)));
  }
  BlasLt::UniqueMatmulPreference unique_preference(preference);
  TF_RETURN_IF_ERROR(SetCublasLtAttr(preference,
                                     CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                     max_workspace_bytes));

  if (plan.params().batch_count == 0) {
    return unique_preference;
  }
  // This is a workaround for a known issue in cuBlasLt where the heuristic may
  // in rare cases select an algo that does not support the specified stride.
  // Specifying the alignment requirements manually like this avoids the issue.
  auto get_alignment_bytes = [](int64_t stride, blas::DataType dtype) {
    return (stride & -stride) * GetDataTypeSizeBytes(dtype);
  };
  if (plan.params().stride_a) {
    TF_RETURN_IF_ERROR(
        SetCublasLtAttr(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
                        (uint32)get_alignment_bytes(plan.params().stride_a,
                                                    plan.params().ab_type)));
  }
  if (plan.params().stride_b) {
    TF_RETURN_IF_ERROR(
        SetCublasLtAttr(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
                        (uint32)get_alignment_bytes(plan.params().stride_b,
                                                    plan.params().ab_type)));
  }
  if (plan.params().stride_c) {
    TF_RETURN_IF_ERROR(
        SetCublasLtAttr(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
                        (uint32)get_alignment_bytes(plan.params().stride_c,
                                                    plan.params().c_type)));
  }
  if (plan.params().stride_c) {
    TF_RETURN_IF_ERROR(
        SetCublasLtAttr(preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
                        (uint32)get_alignment_bytes(plan.params().stride_c,
                                                    plan.params().c_type)));
  }
  return unique_preference;
}

port::Status AllocateWorkspace(void **workspace,
                               ScratchAllocator *scratch_allocator,
                               size_t num_bytes) {
  TF_ASSIGN_OR_RETURN(DeviceMemory<uint8_t> workspace_bytes,
                      scratch_allocator->AllocateBytes(num_bytes));
  *workspace = (void *)gpu::GpuMemoryMutable(&workspace_bytes);
  return port::Status::OK();
}

}  // namespace

port::Status BlasLt::Init() {
  cublasLtHandle_t blas_lt;
  SE_CUBLAS_RETURN_IF_ERROR(cublasLtCreate(&blas_lt));
  absl::MutexLock lock(&mu_);
  blas_lt_.reset(blas_lt);
  return port::Status::OK();
}

std::string BlasLt::MatmulPlanParams::ToString() const {
  return absl::StrCat(transa, ", ", transb, ", ", m, ", ", n, ", ", k, ", ",
                      batch_count, ", ", ab_type, ", ", epilogue);
}

int BlasLt::MatmulAlgorithm::algo_id() const {
  int id;
  GetCublasLtAttr(&algo_, CUBLASLT_ALGO_CONFIG_ID, &id);
  return id;
}

port::Status BlasLt::MatmulPlan::init(const MatmulPlanParams &p) {
  params_ = p;
  scale_type_ = GetScaleType(p.c_type, p.computation_type);
  TF_ASSIGN_OR_RETURN(
      op_desc_,
      CreateCublasLtOperationDesc(
          p.computation_type, GetScaleType(p.c_type, p.computation_type),
          p.pointer_mode, p.epilogue, p.transa, p.transb));
  uint64_t rows_a = p.transa == blas::Transpose::kNoTranspose ? p.m : p.k;
  uint64_t cols_a = p.transa == blas::Transpose::kNoTranspose ? p.k : p.m;
  uint64_t rows_b = p.transb == blas::Transpose::kNoTranspose ? p.k : p.n;
  uint64_t cols_b = p.transb == blas::Transpose::kNoTranspose ? p.n : p.k;
  TF_ASSIGN_OR_RETURN(
      a_desc_, CreateCublasLtLayoutDesc(p.ab_type, rows_a, cols_a, p.lda,
                                        p.stride_a, capped_batch_count()));
  TF_ASSIGN_OR_RETURN(
      b_desc_, CreateCublasLtLayoutDesc(p.ab_type, rows_b, cols_b, p.ldb,
                                        p.stride_b, capped_batch_count()));
  TF_ASSIGN_OR_RETURN(
      c_desc_, CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                        capped_batch_count()));
  TF_ASSIGN_OR_RETURN(
      d_desc_, CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                        capped_batch_count()));
  remainder_batch_count_ =
      p.batch_count > kMaxBatchCount ? p.batch_count % kMaxBatchCount : 0;
  if (remainder_batch_count_) {
    TF_ASSIGN_OR_RETURN(
        a_remainder_desc_,
        CreateCublasLtLayoutDesc(p.ab_type, rows_a, cols_a, p.lda, p.stride_a,
                                 remainder_batch_count_));
    TF_ASSIGN_OR_RETURN(
        b_remainder_desc_,
        CreateCublasLtLayoutDesc(p.ab_type, rows_b, cols_b, p.ldb, p.stride_b,
                                 remainder_batch_count_));
    TF_ASSIGN_OR_RETURN(
        c_remainder_desc_,
        CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                 remainder_batch_count_));
    TF_ASSIGN_OR_RETURN(
        d_remainder_desc_,
        CreateCublasLtLayoutDesc(p.c_type, p.m, p.n, p.ldc, p.stride_c,
                                 remainder_batch_count_));
  }
  return port::Status::OK();
}

bool BlasLt::MatmulPlan::SetBiasPointer(const void *bias) const {
  return SetCublasLtAttr(op_desc_.get(), CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                         bias)
      .ok();
}

/*static*/ port::StatusOr<BlasLt::MatmulPlan> BlasLt::CreateMatmulPlan(
    const BlasLt::MatmulPlanParams &p) {
  MatmulPlan cuda_plan;
  TF_RETURN_IF_ERROR(cuda_plan.init(p));
  return std::move(cuda_plan);
}

port::StatusOr<std::vector<BlasLt::MatmulAlgorithm>>
BlasLt::GetMatmulAlgorithmsInternal(const BlasLt::MatmulPlan &plan,
                                    size_t max_workspace_size,
                                    int max_algorithm_count,
                                    bool for_remainder_batch) {
  TF_ASSIGN_OR_RETURN(UniqueMatmulPreference preference,
                      CreateCublasLtMatmulPreference(plan, max_workspace_size));

  std::vector<cublasLtMatmulHeuristicResult_t> results(max_algorithm_count);
  {
    absl::MutexLock lock(&mu_);

    CHECK(blas_lt_ != nullptr);

    gpu::ScopedActivateExecutorContext sac{parent_};

    int found_algorithm_count = 0;
    const auto &a_desc =
        for_remainder_batch ? plan.a_remainder_desc() : plan.a_desc();
    const auto &b_desc =
        for_remainder_batch ? plan.b_remainder_desc() : plan.b_desc();
    const auto &c_desc =
        for_remainder_batch ? plan.c_remainder_desc() : plan.c_desc();
    const auto &d_desc =
        for_remainder_batch ? plan.d_remainder_desc() : plan.d_desc();
    cublasStatus_t status = cublasLtMatmulAlgoGetHeuristic(
        blas_lt_.get(), plan.op_desc(), a_desc, b_desc, c_desc, d_desc,
        preference.get(), max_algorithm_count, results.data(),
        &found_algorithm_count);
    if (status != CUBLAS_STATUS_SUCCESS) {
      return port::Status(
          port::error::INTERNAL,
          absl::StrCat("cublasLtMatmulAlgoGetHeuristic failed: ",
                       ToString(status)));
    }
    results.resize(found_algorithm_count);
  }

  std::vector<MatmulAlgorithm> out_algorithms;
  out_algorithms.reserve(results.size());
  for (size_t i = 0; i < results.size(); ++i) {
    const auto &result = results[i];
    if (result.state != CUBLAS_STATUS_SUCCESS) continue;  // Skip failed algos
    out_algorithms.emplace_back(i, result.algo, result.workspaceSize);
  }
  return out_algorithms;
}

port::StatusOr<std::vector<BlasLt::MatmulAlgorithm>>
BlasLt::GetMatmulAlgorithms(const BlasLt::MatmulPlan &plan,
                            size_t max_workspace_size,
                            int max_algorithm_count) {
  return GetMatmulAlgorithmsInternal(plan, max_workspace_size,
                                     max_algorithm_count);
}

bool BlasLt::DoMatmulInternal(
    Stream *stream, bool err_on_failure, const BlasLt::MatmulPlan &plan,
    const HostOrDeviceScalar<void> &alpha, DeviceMemoryBase a,
    DeviceMemoryBase b, const HostOrDeviceScalar<void> &beta,
    DeviceMemoryBase c, DeviceMemoryBase d, ScratchAllocator *scratch_allocator,
    const BlasLt::MatmulAlgorithm &algorithm, DeviceMemoryBase bias) {
  if (alpha.data_type() != plan.scale_type() ||
      beta.data_type() != plan.scale_type()) {
    VLOG(2) << "DoBlasLtMatmul returning false because alpha and beta types do "
               "not match plan: expected "
            << plan.c_type() << ", got alpha=" << alpha.data_type()
            << " beta=" << beta.data_type();
    return false;
  }
  if (alpha.is_pointer() != beta.is_pointer()) {
    VLOG(2) << "DoBlasLtMatmul returning false because one of `alpha` "
               "and `beta` is a pointer, but the other is not.";
    return false;
  }
  bool is_pointer_mode_host = !alpha.is_pointer();
  if ((plan.params().pointer_mode == PointerMode::kHost) !=
      is_pointer_mode_host) {
    VLOG(2) << "DoBlasLtMatmul returning false because plan has wrong "
               "pointer_mode for the given alpha/beta.";
    return false;
  }
  if ((plan.params().epilogue == Epilogue::kBias ||
       plan.params().epilogue == Epilogue::kBiasThenReLU ||
       plan.params().epilogue ==
           Epilogue::kBiasThenGeLUApproximate) != (bias != nullptr)) {
    VLOG(2) << "DoBlasLtMatmul returning false because plan has wrong "
               "epilogue for the given bias pointer.";
    return false;
  }
  const void *alpha_ptr = alpha.is_pointer() ? alpha.opaque_pointer().opaque()
                                             : alpha.opaque_value();
  const void *beta_ptr =
      beta.is_pointer() ? beta.opaque_pointer().opaque() : beta.opaque_value();

  void *workspace = nullptr;
  if (algorithm.workspace_size()) {
    port::Status allocation_status = AllocateWorkspace(
        &workspace, scratch_allocator, algorithm.workspace_size());
    if (!allocation_status.ok()) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR)
            << "Failed to allocate workspace for cublasLtMatmul algo with id: "
            << algorithm.algo_id() << " requiring "
            << algorithm.workspace_size() << " bytes of workspace";
      }
      return false;
    }
  }

  // This is only used when batch_count > kMaxBatchCount.
  std::optional<MatmulAlgorithm> remainder_algo;
  if (plan.remainder_batch_count()) {
    // There is no easy way to get the user-specified max workspace size here,
    // so we just allow a very small amount and don't worry too much about
    // performance because this is only used in rare cases. The same reasoning
    // applies to selection of the algorithm.
    size_t max_workspace_size = 4 * 1024 * 1024;  // 4 MiB
    auto status_or_algorithms =
        GetMatmulAlgorithmsInternal(plan, max_workspace_size,
                                    /* max_algorithm_count = */ 1,
                                    /* for_remainder_batch = */ true);
    if (!status_or_algorithms.ok()) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR) << "Failed to get algorithms for blasLt remainder batch.";
      }
      return false;
    }
    auto algorithms = status_or_algorithms.ConsumeValueOrDie();
    remainder_algo = algorithms.front();
  }

  cudaStream_t cuda_stream = gpu::AsGpuStreamValue(stream);

  absl::MutexLock lock(&mu_);

  if (bias != nullptr) {
    if (!plan.SetBiasPointer(bias.opaque())) {
      VLOG(2) << "DoBlasLtMatmul returning false because setting the bias "
                 "pointer failed.";
      return false;
    }
  }

  CHECK(blas_lt_ != nullptr);

  gpu::ScopedActivateExecutorContext sac{parent_};

  // Plan execution is broken down into repeat calls with capped_batch_count,
  // followed by a final call with remainder_batch_count.
  // Cases where batch_count <= kMaxBatchCount require only a single call (a
  // single loop iteration and no remainder).
  int ab_type_size = GetDataTypeSizeBytes(plan.params().ab_type);
  int c_type_size = GetDataTypeSizeBytes(plan.params().c_type);
  const char *a_ptr = static_cast<const char *>(a.opaque());
  const char *b_ptr = static_cast<const char *>(b.opaque());
  const char *c_ptr = static_cast<const char *>(c.opaque());
  char *d_ptr = static_cast<char *>(d.opaque());
  int capped_batch_count = plan.capped_batch_count();
  for (int batch = 0; batch + capped_batch_count <= plan.params().batch_count;
       batch += capped_batch_count) {
    cublasStatus_t ret = cublasLtMatmul(
        blas_lt_.get(), plan.op_desc(), alpha_ptr, a_ptr, plan.a_desc(), b_ptr,
        plan.b_desc(), beta_ptr, c_ptr, plan.c_desc(), d_ptr, plan.d_desc(),
        algorithm.algo(), workspace, algorithm.workspace_size(), cuda_stream);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR) << "failed to run cublasLtMatmul routine: " << ToString(ret);
      }
      return false;
    }
    a_ptr += capped_batch_count * plan.params().stride_a * ab_type_size;
    b_ptr += capped_batch_count * plan.params().stride_b * ab_type_size;
    c_ptr += capped_batch_count * plan.params().stride_c * c_type_size;
    d_ptr += capped_batch_count * plan.params().stride_c * c_type_size;
  }
  // This is only used when batch_count > kMaxBatchCount.
  if (plan.remainder_batch_count()) {
    if (remainder_algo->workspace_size()) {
      port::Status allocation_status = AllocateWorkspace(
          &workspace, scratch_allocator, remainder_algo->workspace_size());
      if (!allocation_status.ok()) {
        if (err_on_failure || VLOG_IS_ON(3)) {
          LOG(ERROR) << "Failed to allocate workspace for cublasLtMatmul algo "
                        "with id: "
                     << remainder_algo->algo_id() << " requiring "
                     << remainder_algo->workspace_size()
                     << " bytes of workspace";
        }
        return false;
      }
    }
    cublasStatus_t ret = cublasLtMatmul(
        blas_lt_.get(), plan.op_desc(), alpha_ptr, a_ptr,
        plan.a_remainder_desc(), b_ptr, plan.b_remainder_desc(), beta_ptr,
        c_ptr, plan.c_remainder_desc(), d_ptr, plan.d_remainder_desc(),
        remainder_algo->algo(), workspace, remainder_algo->workspace_size(),
        cuda_stream);
    if (ret != CUBLAS_STATUS_SUCCESS) {
      if (err_on_failure || VLOG_IS_ON(3)) {
        LOG(ERROR) << "failed to run remainder cublasLtMatmul routine: "
                   << ToString(ret);
      }
      return false;
    }
  }
  return true;
}

bool BlasLt::DoMatmul(Stream *stream, const BlasLt::MatmulPlan &plan,
                      const HostOrDeviceScalar<void> &alpha, DeviceMemoryBase a,
                      DeviceMemoryBase b, const HostOrDeviceScalar<void> &beta,
                      DeviceMemoryBase c, ScratchAllocator *scratch_allocator,
                      const BlasLt::MatmulAlgorithm &algorithm,
                      DeviceMemoryBase bias,
                      blas::ProfileResult *output_profile_result) {
  HostOrDeviceScalar<void> alpha_cast = alpha;
  HostOrDeviceScalar<void> beta_cast = beta;
  if (plan.c_type() == blas::DataType::kHalf &&
      plan.scale_type() == blas::DataType::kFloat) {
    // The given alpha and beta types are F16 (they always match c), but F32*
    // computation type requires that they be F32, so we must cast them.
    if (alpha.is_pointer() || beta.is_pointer()) {
      // We cannot easily convert a pointer to f16 memory to a pointer to f32
      // memory from here, so we don't support this for now.
      return false;
    }
    alpha_cast = HostOrDeviceScalar<void>(
        static_cast<float>(alpha.value<Eigen::half>()));
    beta_cast =
        HostOrDeviceScalar<void>(static_cast<float>(beta.value<Eigen::half>()));
  }

  std::unique_ptr<gpu::GpuTimer, gpu::GpuTimerDeleter> timer;
  if (output_profile_result) {
    timer.reset(new gpu::GpuTimer(parent_));
    if (!timer->Init() || !timer->Start(gpu::AsGpuStream(stream))) {
      return false;
    }
  }

  bool err_on_failure = timer != nullptr;
  bool result =
      DoMatmulInternal(stream, err_on_failure, plan, alpha_cast, a, b,
                       beta_cast, c, c, scratch_allocator, algorithm, bias);

  if (timer && result) {
    // GpuTimer will CHECK-fail if we Stop() it while the stream is in an error
    // state.
    if (!timer->Stop(gpu::AsGpuStream(stream))) {
      return false;
    }
    output_profile_result->set_is_valid(true);
    output_profile_result->set_algorithm(algorithm.index());
    output_profile_result->set_elapsed_time_in_ms(
        timer->GetElapsedMilliseconds());
  }
  return result;
}

BlasLt *GetBlasLt(Stream *stream) {
  CUDABlas *blas = dynamic_cast<CUDABlas *>(stream->parent()->AsBlas());
  return (blas != nullptr) ? &blas->blas_lt() : nullptr;
}

bool operator==(const BlasLt::MatmulPlanParams &a,
                const BlasLt::MatmulPlanParams &b) {
  return internal::AsTuple(a) == internal::AsTuple(b);
}

}  // namespace cuda
}  // namespace stream_executor
