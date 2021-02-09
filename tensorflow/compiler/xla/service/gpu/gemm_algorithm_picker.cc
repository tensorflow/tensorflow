/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gemm_algorithm_picker.h"

#include <limits>

#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logger.h"
#include "tensorflow/core/protobuf/autotuning.pb.h"
#include "tensorflow/core/util/proto/proto_utils.h"
#include "tensorflow/stream_executor/blas.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/gpu/redzone_allocator.h"

namespace xla {
namespace gpu {

using tensorflow::AutotuneResult;

using GemmCacheKey =
    std::tuple<se::StreamExecutor*, Shape, Shape, Shape, std::string>;

static tensorflow::mutex autotune_cache_mu(tensorflow::LINKER_INITIALIZED);
static auto& autotune_cache TF_GUARDED_BY(autotune_cache_mu) =
    *new absl::flat_hash_map<GemmCacheKey,
                             absl::optional<se::blas::AlgorithmType>>();
static int64 cache_hits TF_GUARDED_BY(autotune_cache_mu) = 0;
static int64 cache_misses TF_GUARDED_BY(autotune_cache_mu) = 0;

// Experimentally tries to pick the best algorithm for the given gemm.
//
// This may fail under perfectly normal circumstances.  In particular, it will
// fail if the program was built with < CUDA 8 or if we're using a gpu older
// than sm_50 -- in both cases, cublas doesn't support gemm-with-algorithm at
// all.
static StatusOr<absl::optional<se::blas::AlgorithmType>> DoUncachedGemmAutotune(
    const HloInstruction* gemm, se::Stream* stream,
    se::DeviceMemoryAllocator* allocator) {
  if (!stream->parent()->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  const HloModuleConfig& hlo_module_config = gemm->GetModule()->config();
  const bool init_cublas_data =
      hlo_module_config.debug_options().xla_gpu_autotune_level() > 1;
  se::RedzoneAllocator input_output_allocator(
      stream, allocator, PtxOptsFromConfig(hlo_module_config),
      /*memory_limit=*/std::numeric_limits<int64>::max());

  BufferComparator comparator(gemm->shape(), hlo_module_config);

  int64 rng_state = 0;
  auto get_initialized_buffer =
      [&](const HloInstruction* op) -> StatusOr<se::DeviceMemoryBase> {
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                        input_output_allocator.AllocateBytes(
                            ShapeUtil::ByteSizeOf(op->shape())));
    if (init_cublas_data) {
      InitializeBuffer(stream, op->shape().element_type(), &rng_state, buffer);
    }
    return buffer;
  };

  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase lhs_buffer,
                      get_initialized_buffer(gemm->operand(0)));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase rhs_buffer,
                      get_initialized_buffer(gemm->operand(1)));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase output_buffer,
                      get_initialized_buffer(gemm));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase reference_result_buffer,
                      get_initialized_buffer(gemm));

  const DebugOptions& debug_options =
      gemm->GetModule()->config().debug_options();

  const bool crash_on_checking_failure =
      debug_options.xla_gpu_crash_on_verification_failures();

  GemmBackendConfig backend_config =
      gemm->backend_config<GemmBackendConfig>().ValueOrDie();
  const int32 cublas_autotune_level =
      gemm->GetModule()->config().debug_options().xla_gpu_autotune_level();
  const bool reinit_cublas_data = cublas_autotune_level > 2;
  const bool check_cublas = cublas_autotune_level > 3;

  VLOG(3) << "Starting autotune of GemmThunk " << gemm->ToString();

  std::vector<se::blas::AlgorithmType> algorithms;
  CHECK(stream->parent()->GetBlasGemmAlgorithms(&algorithms));

  absl::optional<se::blas::AlgorithmType> first_algorithm;
  std::vector<AutotuneResult> profile_results;

  GpuGemmConfig config = GetGpuGemmConfig(gemm);

  for (se::blas::AlgorithmType algorithm : algorithms) {
    // Make sure the output buffer always has the same value if we use
    // the bias parameter.
    if (reinit_cublas_data && backend_config.beta() != 0) {
      int64 rng_state = 0;
      InitializeBuffer(stream, gemm->shape().element_type(), &rng_state,
                       output_buffer);
    }
    se::blas::ProfileResult profile_result;

    // We expect GemmWithAlgorithm to fail sometimes -- in fact, it will fail
    // for all algorithms if we're targeting < sm_50.  But because we pass a
    // non-null ProfileResult, DoGemmWithAlgorithm should always return true,
    // and the actual success-ness is returned in ProfileResult::is_valid.
    CHECK(RunGemm(config, lhs_buffer, rhs_buffer, output_buffer, stream,
                  /*implements_whole_instruction=*/true,
                  /*profile_index=*/-1,
                  /*profiler=*/nullptr,
                  /*profile_result=*/&profile_result, algorithm)
              .ok());

    if (!profile_result.is_valid()) {
      // Unsupported algorithm.
      continue;
    }

    profile_results.emplace_back();
    AutotuneResult& result = profile_results.back();
    result.mutable_gemm()->set_algorithm(algorithm);

    VLOG(2) << "cublas gemm algorithm " << algorithm << " took "
            << profile_result.elapsed_time_in_ms() << "ms" << std::endl;

    *result.mutable_run_time() = tensorflow::proto_utils::ToDurationProto(
        absl::Milliseconds(profile_result.elapsed_time_in_ms()));

    if (!check_cublas) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(
        se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
        input_output_allocator.CheckRedzones());
    if (!rz_check_status.ok()) {
      result.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
      *result.mutable_failure()->mutable_msg() =
          rz_check_status.RedzoneFailureMsg();
      LOG(ERROR) << "Detected cuBLAS out-of-bounds write in gemm buffer";
      CHECK(!crash_on_checking_failure);
      continue;
    }

    if (!first_algorithm) {
      // First run: set the reference result buffer.
      CHECK(reference_result_buffer.size() == output_buffer.size());
      stream->ThenMemcpy(&reference_result_buffer, output_buffer,
                         output_buffer.size());
      first_algorithm.emplace(algorithm);
    } else {
      // Perform the comparison.
      TF_ASSIGN_OR_RETURN(bool compare_result,
                          comparator.CompareEqual(stream, output_buffer,
                                                  reference_result_buffer));
      if (!compare_result) {
        LOG(ERROR) << "Results mismatch between different GEMM algorithms. "
                   << "This is likely a bug/unexpected loss of precision "
                   << "in cuBLAS.";
        CHECK(!crash_on_checking_failure);

        result.mutable_failure()->set_kind(AutotuneResult::WRONG_RESULT);
        result.mutable_failure()->mutable_reference_gemm()->set_algorithm(
            *first_algorithm);
      }
    }
  }

  tensorflow::AutotuningLog log;
  for (const AutotuneResult& profile : profile_results) {
    *log.add_results() = profile;
  }
  if (!crash_on_checking_failure) {
    tensorflow::Logger::GetSingleton()->LogProto(log);
  }

  // Choose fastest correct GEMM, but allow for incorrect results (since the
  // reference result is chosen arbitrary).
  auto has_failure = [](const AutotuneResult& r) {
    return r.has_failure() &&
           r.failure().kind() != AutotuneResult::WRONG_RESULT;
  };

  auto result_comparison_key = [&has_failure](const AutotuneResult& r) {
    return std::make_tuple(
        has_failure(r),
        tensorflow::proto_utils::FromDurationProto(r.run_time()));
  };
  const auto& best_result = absl::c_min_element(
      profile_results,
      [&](const AutotuneResult& lhs, const AutotuneResult& rhs) {
        return result_comparison_key(lhs) < result_comparison_key(rhs);
      });

  if (best_result != profile_results.end() && !has_failure(*best_result)) {
    return {best_result->gemm().algorithm()};
  }

  VLOG(1) << "Unable to autotune cuBLAS gemm on stream " << stream
          << " none of the " << algorithms.size() << " ran successfully";
  return {absl::nullopt};
}

static StatusOr<absl::optional<se::blas::AlgorithmType>> DoGemmAutotune(
    const HloInstruction* instr, const GemmBackendConfig& gemm_config,
    se::DeviceMemoryAllocator* allocator, se::Stream* stream) {
  const HloInstruction* lhs = instr->operand(0);
  const HloInstruction* rhs = instr->operand(1);

  // Don't run autotuning concurrently on the same GPU.
  tensorflow::mutex_lock gpu_lock = LockGpu(stream->parent());

  GemmCacheKey key =
      std::make_tuple(stream->parent(), lhs->shape(), rhs->shape(),
                      instr->shape(), gemm_config.SerializeAsString());

  tensorflow::mutex_lock cache_lock(autotune_cache_mu);
  auto it = autotune_cache.find(key);
  int64 autotuning_requests = cache_hits + cache_misses;
  if (autotuning_requests && autotuning_requests % 10 == 0) {
    VLOG(2) << "Autotuning cache hits/(hits + misses): " << cache_hits << "/"
            << autotuning_requests;
  }

  if (it != autotune_cache.end()) {
    cache_hits++;
    VLOG(4) << "Autotuning cache hit, using algorithm: "
            << (it->second.has_value() ? absl::StrCat(*(it->second))
                                       : "<generic>");
    return it->second;
  }
  cache_misses++;
  VLOG(4) << "Autotuning cache miss";

  int64 batch_size = gemm_config.batch_size();
  absl::optional<se::blas::AlgorithmType> result;
  if (batch_size != 1) {
    // TODO(b/112111608): Implement auto tune for batched gemm.
    VLOG(2) << "Batch size is non-singular, using generic algorithm";
    result = absl::nullopt;
  } else {
    TF_ASSIGN_OR_RETURN(result,
                        DoUncachedGemmAutotune(instr, stream, allocator));
  }

  CHECK(autotune_cache.emplace(key, result).second);
  return result;
}

static StatusOr<bool> RunOnInstruction(HloInstruction* instr,
                                       se::StreamExecutor* executor,
                                       se::DeviceMemoryAllocator* allocator) {
  if (allocator == nullptr) {
    allocator = executor->GetAllocator();
  }
  TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                      allocator->GetStream(executor->device_ordinal()));

  GemmBackendConfig gemm_config =
      instr->backend_config<GemmBackendConfig>().ValueOrDie();

  TF_ASSIGN_OR_RETURN(absl::optional<se::blas::AlgorithmType> gemm_algorithm,
                      DoGemmAutotune(instr, gemm_config, allocator, stream));

  // We update instruction->backend_config(); if no algorithms are supported,
  // a different API is used, which does not require specifying an algorithm.
  GemmBackendConfig updated_config = gemm_config;
  if (gemm_algorithm) {
    updated_config.set_selected_algorithm(*gemm_algorithm);
  }
  TF_RETURN_IF_ERROR(instr->set_backend_config(updated_config));
  return updated_config.SerializeAsString() != gemm_config.SerializeAsString();
}

static StatusOr<bool> RunOnComputation(HloComputation* computation,
                                       se::StreamExecutor* se,
                                       se::DeviceMemoryAllocator* allocator) {
  bool changed = false;
  for (HloInstruction* instr : computation->instructions()) {
    if (IsCublasGemm(*instr)) {
      TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr, se, allocator));
      changed |= result;
    }
  }
  return changed;
}

StatusOr<bool> GemmAlgorithmPicker::Run(HloModule* module) {
  XLA_SCOPED_LOGGING_TIMER("GemmAlgorithmPicker");

  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    VLOG(2) << "GEMM auto-tuning disabled, GemmAlgorithmPicker returning early";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation : module->MakeNonfusionComputations()) {
    TF_ASSIGN_OR_RETURN(
        bool result, RunOnComputation(computation, stream_exec_, allocator_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
