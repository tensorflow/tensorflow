/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"

#include <cstdint>
#include <functional>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/scratch_allocator.h"

namespace xla {
namespace gpu {

GemmThunk::GemmThunk(ThunkInfo thunk_info, GemmConfig config,
                     const BufferAllocation::Slice &lhs_buffer,
                     const BufferAllocation::Slice &rhs_buffer,
                     const BufferAllocation::Slice &output_buffer)
    : Thunk(Kind::kGemm, thunk_info),
      config_(std::move(config)),
      lhs_buffer_(lhs_buffer),
      rhs_buffer_(rhs_buffer),
      output_buffer_(output_buffer) {}

Status GemmThunk::ExecuteOnStream(const ExecuteParams &params) {
  auto get_device_address = [&](const BufferAllocation::Slice &slice) {
    return params.buffer_allocations->GetDeviceAddress(slice);
  };

  se::DeviceMemoryBase lhs_data = get_device_address(lhs_buffer_);
  se::DeviceMemoryBase rhs_data = get_device_address(rhs_buffer_);
  se::DeviceMemoryBase output_data = get_device_address(output_buffer_);

  auto &buffer_allocations = *params.buffer_allocations;
  se::OwningScratchAllocator scratch_allocator(
      buffer_allocations.device_ordinal(),
      buffer_allocations.memory_allocator());

  VLOG(3) << "Running GEMM thunk";
  return RunGemm(config_, lhs_data, rhs_data, output_data, params.stream,
                 &scratch_allocator, nullptr);
}

bool BlasPlansAutotuneCache::Find(const se::BatchMatmulParameters &params,
                                  se::blas::AlgorithmConfig *config) const {
  absl::MutexLock lock(&mu_);
  auto iter = blas_plans_algorithms_map_.find(params);
  if (iter == blas_plans_algorithms_map_.end()) {
    return false;
  }
  *config = iter->second;
  return true;
}

void BlasPlansAutotuneCache::Insert(const se::BatchMatmulParameters &params,
                                    const se::blas::AlgorithmConfig &config) {
  absl::MutexLock lock(&mu_);
  if (!blas_plans_algorithms_map_.contains(params)) {
    blas_plans_algorithms_map_.insert({params, config});
  }
}

// Converts from an XLA PrimitiveType to a blas::ComputationType, which is
// used to specify the precision with which matmul computations should be
// performed, separately from the precision of the inputs and result.
static absl::optional<se::blas::ComputationType> ComputationTypeFromPrimitive(
    PrimitiveType type) {
  switch (type) {
    case F16:  // Use F32 computation for higher precision.
    case BF16:
    case F32:
      return se::blas::ComputationType::kF32;
    case F64:
      return se::blas::ComputationType::kF64;
    case C64:
      return se::blas::ComputationType::kComplexF32;
    case C128:
      return se::blas::ComputationType::kComplexF64;
    case S32:
      return se::blas::ComputationType::kI32;
    default:
      return absl::nullopt;
  }
}

template <typename Input, typename Output>
static Status DoGemmWithAlgorithm(
    int64_t batch_size, const se::blas::MatrixDescriptor &lhs,
    const se::blas::MatrixDescriptor &rhs,
    const se::blas::MatrixDescriptor &output, Output alpha, Output beta,
    se::Stream *stream, se::blas::AlgorithmType algorithm,
    se::blas::ProfileResult *output_profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  PrimitiveType output_type = primitive_util::NativeToPrimitiveType<Output>();
  se::blas::ComputationType computation_type =
      *ComputationTypeFromPrimitive(output_type);
  se::DeviceMemory<Output> output_data(output.data);

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatchedWithAlgorithm(
        lhs.transpose, rhs.transpose, output.num_rows, output.num_cols,
        /*size of reduce dim=*/lhs.reduced_dim(),
        /*alpha=*/alpha, lhs.cast<Input>(),
        /*leading dim of LHS=*/lhs.num_rows, lhs.stride, rhs.cast<Input>(),
        /*leading dim of RHS=*/rhs.num_rows, rhs.stride,
        /*beta=*/beta, &output_data,
        /*leading dim of output=*/output.num_rows, output.stride, batch_size,
        computation_type, algorithm, output_profile_result);
  } else {
    return stream->ThenBlasGemmWithAlgorithm(
        lhs.transpose, rhs.transpose, output.num_rows, output.num_cols,
        /*size of reduce dim=*/lhs.reduced_dim(),
        /*alpha=*/alpha, lhs.cast<Input>(),
        /*lda=*/lhs.num_rows, rhs.cast<Input>(),
        /*ldb=*/rhs.num_rows,
        /*beta=*/beta, &output_data,
        /*ldc=*/output.num_rows, computation_type, algorithm,
        output_profile_result);
  }
}

template <typename Input>
static Status DoGemm(int64_t batch_size, const se::blas::MatrixDescriptor &lhs,
                     const se::blas::MatrixDescriptor &rhs,
                     const se::blas::MatrixDescriptor &output, Input alpha,
                     Input beta, se::Stream *stream,
                     absl::optional<se::blas::AlgorithmType> algorithm,
                     se::blas::ProfileResult *output_profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  se::DeviceMemory<Input> output_data(output.data);

  if (algorithm) {
    return DoGemmWithAlgorithm<Input, Input>(batch_size, lhs, rhs, output,
                                             alpha, beta, stream, *algorithm,
                                             output_profile_result);
  }

  if (batch_size != 1) {
    return stream->ThenBlasGemmStridedBatched(
        lhs.transpose, rhs.transpose, output.num_rows, output.num_cols,
        /*size of reduce dim=*/lhs.reduced_dim(),
        /*alpha=*/alpha, lhs.cast<Input>(),
        /*leading dim of LHS=*/lhs.num_rows, lhs.stride, rhs.cast<Input>(),
        /*leading dim of RHS=*/rhs.num_rows, rhs.stride,
        /*beta=*/beta, &output_data,
        /*leading dim of output=*/output.num_rows, output.stride, batch_size);
  }

  return stream->ThenBlasGemm(
      lhs.transpose, rhs.transpose, output.num_rows, output.num_cols,
      /*size of reduce dim=*/lhs.reduced_dim(),
      /*alpha=*/alpha, lhs.cast<Input>(),
      /*leading dim of LHS=*/lhs.num_rows, rhs.cast<Input>(),
      /*leading dim of RHS=*/rhs.num_rows,
      /*beta=*/beta, &output_data,
      /*leading dim of output=*/output.num_rows);
}

template <typename Input>
static Status DoGemmLt(
    int64_t batch_size, const se::blas::MatrixDescriptor &lhs,
    const se::blas::MatrixDescriptor &rhs,
    const se::blas::MatrixDescriptor &output, se::Stream *stream, Input alpha,
    Input beta, se::ScratchAllocator *scratch_allocator,
    se::blas::IBlasLtMatmulAlgorithm *const algorithm_being_profiled,
    se::blas::ProfileResult *output_profile_result) {
  CHECK(output.transpose == se::blas::Transpose::kNoTranspose);
  tensorflow::DataType dtype = tensorflow::DataTypeToEnum<Input>::value;

  int device_id = stream->parent()->device_ordinal();

  bool trans_x = lhs.transpose == se::blas::Transpose::kTranspose;
  bool trans_y = rhs.transpose == se::blas::Transpose::kTranspose;

  int64_t m = output.num_rows;
  int64_t n = output.num_cols;
  auto k = lhs.reduced_dim();
  bool broadcast = batch_size == 1;
  VLOG(2) << "matmul params: trans_x " << trans_x << " trans_y " << trans_y
          << " adj_x " << false << " adj_y " << false << " m " << m << " n "
          << n << " k " << k << " batch_size " << batch_size << " broadcast "
          << broadcast << " broadcast " << broadcast << " dtype " << dtype
          << " device_id " << device_id;
  se::BatchMatmulParameters matmul_parameters(
      trans_x, trans_y, false, false, m, n, k, batch_size, broadcast, broadcast,
      dtype, dtype, device_id);

  TF_ASSIGN_OR_RETURN(
      const se::blas::PlanAndAlgorithms *plan_and_algorithms,
      GetPlanAndAlgorithms(stream, matmul_parameters, batch_size, dtype, lhs,
                           rhs, output));

  const std::unique_ptr<se::blas::IBlasLtMatmulPlan> &plan =
      plan_and_algorithms->plan;
  const std::vector<std::unique_ptr<se::blas::IBlasLtMatmulAlgorithm>>
      &algorithms = plan_and_algorithms->algorithms;

  // 'algorithm_being_profiled' is the algorithm that is being profiled (as the
  // name suggests). RunGemm can have two callers - gemm_algorithm_picker (at
  // compile time) & gpu_executable (at runtime). 'algorithm_being_profiled' is
  // NULL when RunGemm is called during runtime (i.e, via ExecuteOnStream from
  // gpu_executable). In that case, 'ThenBlasLtMatmul' is called with the
  // algorithm selected by the autotuner (from the AutotuneCache) or default
  // algorithm (when autotuner is disabled). When called from gemm_algorithm
  // picker, 'ThenBlasLtMatmul' is called with the 'algorithm_being_profiled'.
  se::blas::IBlasLtMatmulAlgorithm *algorithm_ptr = algorithm_being_profiled;
  if (!algorithm_being_profiled) {
    se::blas::AlgorithmConfig algorithm_config(se::blas::kNoAlgorithm);

    // When autotuner is disabled, BlasPlansAutotuneCache singleton is empty.
    if (!BlasPlansAutotuneCacheSingleton::GetInstance()->Find(
            matmul_parameters, &algorithm_config)) {
      algorithm_config.set_algorithm(0);
      VLOG(4) << "Autotuner disabled: Inserting algorithm id "
              << algorithm_config.algorithm() << " for " << trans_x << " "
              << trans_y << " " << m << " " << n << " " << k << " "
              << batch_size << " " << broadcast << " " << broadcast << " "
              << dtype << " " << device_id;
      BlasPlansAutotuneCacheSingleton::GetInstance()->Insert(matmul_parameters,
                                                             algorithm_config);
    }
    se::blas::AlgorithmType algorithm_idx = algorithm_config.algorithm();
    algorithm_ptr = algorithms[algorithm_idx].get();
  }

  se::DeviceMemory<Input> output_data(output.data);
  // NOLINTBEGIN: (b/223663260) ClangTidy mistakenly reports .get() as a
  // redundant call
  if (stream
          ->ThenBlasLtMatmul(plan.get(), alpha, lhs.cast<Input>(),
                             rhs.cast<Input>(), beta, &output_data,
                             scratch_allocator, algorithm_ptr, {},
                             output_profile_result)
          .ok()) {
    return Status::OK();
  }
  // NOLINTEND
  return InternalError("BlasLtMatmul failed.");
}

Status RunGemm(const GemmConfig &config, se::DeviceMemoryBase lhs_buffer,
               se::DeviceMemoryBase rhs_buffer,
               se::DeviceMemoryBase output_buffer, se::Stream *stream,
               se::ScratchAllocator *scratch_allocator,
               se::blas::IBlasLtMatmulAlgorithm *const algorithm_being_profiled,
               se::blas::ProfileResult *profile_result,
               absl::optional<se::blas::AlgorithmType> algorithm) {
  VLOG(2) << "Executing a GemmThunk";
  se::blas::MatrixDescriptor lhs = GetMatrixDesc(config.lhs_layout, lhs_buffer);
  se::blas::MatrixDescriptor rhs = GetMatrixDesc(config.rhs_layout, rhs_buffer);
  se::blas::MatrixDescriptor output =
      GetMatrixDesc(config.output_layout, output_buffer);
  int64_t batch_size = config.output_layout.batch_size;

  // TODO(cjfj): Support transposed output when using cuBLASLt.
  MakeBlasGemmCompatible(lhs, rhs, output);

  // The BlasLtMatmul routines are only supported from CUDA 11.0 onward.
  if (config.use_cublaslt && stream->parent()->SupportsBlasPlans()) {
    se::blas::IBlasLtMatmulAlgorithm *algorithm = nullptr;
    if (algorithm_being_profiled) {
      algorithm = algorithm_being_profiled;
    }
    switch (config.output_layout.dtype) {
      case F16:
        return DoGemmLt<Eigen::half>(
            batch_size, lhs, rhs, output, stream,
            static_cast<Eigen::half>(config.alpha.real()),
            static_cast<Eigen::half>(config.beta), scratch_allocator, algorithm,
            /*output_profile_result=*/profile_result);
      case F32:
        return DoGemmLt<float>(batch_size, lhs, rhs, output, stream,
                               static_cast<float>(config.alpha.real()),
                               static_cast<float>(config.beta),
                               scratch_allocator, algorithm,
                               /*output_profile_result=*/profile_result);
      case F64:
        return DoGemmLt<double>(batch_size, lhs, rhs, output, stream,
                                static_cast<double>(config.alpha.real()),
                                config.beta, scratch_allocator, algorithm,
                                /*output_profile_result=*/profile_result);
      case C64:
        return DoGemmLt<complex64>(batch_size, lhs, rhs, output, stream,
                                   static_cast<complex64>(config.alpha),
                                   static_cast<complex64>(config.beta),
                                   scratch_allocator, algorithm,
                                   /*output_profile_result=*/profile_result);
      case C128:
        return DoGemmLt<complex128>(
            batch_size, lhs, rhs, output, stream, config.alpha,
            static_cast<complex64>(config.beta), scratch_allocator, algorithm,
            /*output_profile_result=*/profile_result);
      default:
        return InternalError("Unexpected GEMMLt dtype: %s",
                             primitive_util::LowercasePrimitiveTypeName(
                                 config.output_layout.dtype));
    }
  } else {
    if (!algorithm) algorithm = config.algorithm;

    switch (config.output_layout.dtype) {
      case S32:
        if (!algorithm) algorithm = se::blas::kDefaultGemmAlgo;
        return DoGemmWithAlgorithm<int8_t, int32_t>(
            batch_size, lhs, rhs, output,
            static_cast<int32_t>(config.alpha.real()),
            static_cast<int32_t>(config.beta), stream, *algorithm,
            /*output_profile_result=*/profile_result);
      case F16:
        return DoGemm<Eigen::half>(
            batch_size, lhs, rhs, output,
            static_cast<Eigen::half>(config.alpha.real()),
            static_cast<Eigen::half>(config.beta), stream, algorithm,
            /*output_profile_result=*/profile_result);
      case BF16:
        return DoGemm<Eigen::bfloat16>(
            batch_size, lhs, rhs, output,
            static_cast<Eigen::bfloat16>(config.alpha.real()),
            static_cast<Eigen::bfloat16>(config.beta), stream, algorithm,
            /*output_profile_result=*/profile_result);
      case F32:
        return DoGemm<float>(batch_size, lhs, rhs, output, config.alpha.real(),
                             config.beta, stream, algorithm,
                             /*output_profile_result=*/profile_result);
      case F64:
        return DoGemm<double>(batch_size, lhs, rhs, output, config.alpha.real(),
                              config.beta, stream, algorithm,
                              /*output_profile_result=*/profile_result);
      case C64:
        return DoGemm<complex64>(
            batch_size, lhs, rhs, output, static_cast<complex64>(config.alpha),
            static_cast<complex64>(config.beta), stream, algorithm,
            /*output_profile_result=*/profile_result);
      case C128:
        return DoGemm<complex128>(batch_size, lhs, rhs, output, config.alpha,
                                  static_cast<complex128>(config.beta), stream,
                                  algorithm,
                                  /*output_profile_result=*/profile_result);
      default:
        return InternalError("Unexpected GEMM dtype: %s",
                             primitive_util::LowercasePrimitiveTypeName(
                                 config.output_layout.dtype));
    }
  }
}

}  // namespace gpu
}  // namespace xla
