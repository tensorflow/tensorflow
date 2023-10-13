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

#include "xla/service/gpu/gemm_algorithm_picker.h"

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_asm_opts_util.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logger.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"
#include "tsl/util/proto/proto_utils.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
#include "xla/service/gpu/buffer_comparator.h"
#include "xla/stream_executor/cuda/cuda_blas_lt.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#endif

namespace xla {
namespace gpu {

// Returns the index (into `algorithms`) of the fastest algorithm.
template <typename AlgoT>
StatusOr<AutotuneResult> GetBestAlgorithm(
    se::Stream* stream, se::RedzoneAllocator& allocator,
    std::optional<std::string_view> gemm_str,
    const AutotuneConfig& autotune_config, se::DeviceMemoryBase lhs_buffer,
    se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase output_buffer,
    absl::Span<const AlgoT> algorithms, const Shape& output_shape,
    const HloModuleConfig& hlo_module_config, double beta,
    const std::function<StatusOr<se::blas::ProfileResult>(const AlgoT&)>&
        run_benchmark) {
  if (!stream->parent()->SynchronizeAllActivity()) {
    return InternalError("Failed to synchronize GPU for autotuning.");
  }

  se::DeviceMemoryBase reference_buffer;
  if (autotune_config.should_check_correctness()) {
    TF_ASSIGN_OR_RETURN(
        reference_buffer,
        allocator.AllocateBytes(ShapeUtil::ByteSizeOf(output_shape)));
  }

  BufferComparator comparator(output_shape, hlo_module_config);

  std::vector<AutotuneResult> results;
  std::optional<int64_t> reference_algorithm;

  for (const AlgoT& algorithm : algorithms) {
    // Make sure the output buffer always has the same value if we use
    // the bias parameter.
    if (autotune_config.should_reinit_output_buffer() && beta != 0) {
      int64_t rng_state = 0;
      InitializeBuffer(stream, output_shape.element_type(), &rng_state,
                       output_buffer);
    }

    TF_ASSIGN_OR_RETURN(se::blas::ProfileResult profile_result,
                        run_benchmark(algorithm));

    results.emplace_back();
    AutotuneResult& result = results.back();
    result.mutable_gemm()->set_algorithm(profile_result.algorithm());

    if (!profile_result.is_valid()) {  // Unsupported algorithm.
      result.mutable_failure()->set_kind(AutotuneResult::DISQUALIFIED);
      continue;
    }

    VLOG(2) << "gemm algorithm " << profile_result.algorithm() << " took "
            << profile_result.elapsed_time_in_ms() << "ms";

    *result.mutable_run_time() = tsl::proto_utils::ToDurationProto(
        absl::Milliseconds(profile_result.elapsed_time_in_ms()));

    if (!autotune_config.should_check_correctness()) {
      continue;
    }

    TF_ASSIGN_OR_RETURN(
        se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
        allocator.CheckRedzones());

    if (!rz_check_status.ok()) {
      result.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
      *result.mutable_failure()->mutable_msg() =
          rz_check_status.RedzoneFailureMsg();
      LOG(ERROR) << "Detected out-of-bounds write in gemm buffer";
      CHECK(!autotune_config.should_crash_on_check_failure());
      continue;
    }

    if (!reference_algorithm) {
      stream->ThenMemcpy(&reference_buffer, output_buffer,
                         output_buffer.size());
      reference_algorithm = profile_result.algorithm();
    } else {
      // Perform the comparison.
      TF_ASSIGN_OR_RETURN(
          bool outputs_match,
          comparator.CompareEqual(stream, /*current=*/output_buffer,
                                  /*expected=*/reference_buffer));
      if (!outputs_match) {
        LOG(ERROR) << "Results mismatch between different GEMM algorithms. "
                   << "This is likely a bug/unexpected loss of precision.";
        CHECK(!autotune_config.should_crash_on_check_failure());

        result.mutable_failure()->set_kind(AutotuneResult::WRONG_RESULT);
        result.mutable_failure()->mutable_reference_gemm()->set_algorithm(
            *reference_algorithm);
      }
    }
  }

  if (!autotune_config.should_crash_on_check_failure()) {
    AutotuningLog log;
    for (const AutotuneResult& result : results) {
      *log.add_results() = result;
    }
    tsl::Logger::GetSingleton()->LogProto(log);
  }

  StatusOr<AutotuneResult> best =
      PickBestResult(results, gemm_str, hlo_module_config);
  if (best.ok()) {
    for (size_t i = 0; i < results.size(); ++i) {
      if (best->gemm().algorithm() == results[i].gemm().algorithm()) {
        best->mutable_gemm()->set_algorithm(i);
        return best;
      }
    }
    return InternalError("unknown best algorithm");
  }

  LOG(WARNING) << "Failed to find best cuBLAS algorithm, GEMM performance "
                  "might be suboptimal: "
               << best.status();
  return AutotuneResult{};
}

// Select the best algorithm using information from a Blas instruction.
// Returns the index (into `algorithms`) of the fastest algorithm.
StatusOr<AutotuneResult> GetBestBlasAlgorithm(
    se::Stream* stream, se::RedzoneAllocator& allocator,
    std::optional<std::string_view> gemm_str,
    const AutotuneConfig& autotune_config, se::DeviceMemoryBase lhs_buffer,
    se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase output_buffer,
    absl::Span<const se::blas::AlgorithmType> algorithms,
    const Shape& output_shape, const HloModuleConfig& hlo_module_config,
    double beta,
    const std::function<StatusOr<se::blas::ProfileResult>(
        const se::blas::AlgorithmType&)>& run_benchmark) {
  return GetBestAlgorithm<se::blas::AlgorithmType>(
      stream, allocator, gemm_str, autotune_config, lhs_buffer, rhs_buffer,
      output_buffer, algorithms, output_shape, hlo_module_config, beta,
      run_benchmark);
}

namespace {

StatusOr<se::gpu::BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return se::gpu::BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return se::gpu::BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return se::gpu::BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return se::gpu::BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return se::gpu::BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return se::gpu::BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return se::gpu::BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return se::gpu::BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return InternalError("Unsupported Epilogue.");
  }
}

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)

StatusOr<AutotuneResult> DoGemmAutotuneNoCache(
    const HloInstruction* gemm, const AutotuneCacheKey& key,
    const AutotuneConfig& autotune_config) {
  if (autotune_config.IsDeviceless()) {
    // Return empty result, will tune at runtime.
    return AutotuneResult{};
  }

  VLOG(3) << "Starting autotune of GemmThunk " << gemm->ToString();
  se::DeviceMemoryAllocator* allocator = autotune_config.GetAllocator();
  TF_ASSIGN_OR_RETURN(se::Stream* const stream, autotune_config.GetStream());
  GemmBackendConfig gemm_config =
      gemm->backend_config<GemmBackendConfig>().value();
  const DebugOptions& debug_options =
      gemm->GetModule()->config().debug_options();
  const bool deterministic_ops = debug_options.xla_gpu_deterministic_ops();

  TF_ASSIGN_OR_RETURN(GemmConfig config, GemmConfig::For(gemm));
  // Don't run autotuning concurrently on the same GPU.
  absl::MutexLock gpu_lock(&GetGpuMutex(stream->parent()));

  TF_ASSIGN_OR_RETURN(
      se::RedzoneAllocator buffer_allocator,
      AutotunerUtil::CreateRedzoneAllocator(autotune_config, debug_options));

  int64_t rng_state = 0;
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase lhs_buffer,
      AutotunerUtil::CreateBuffer(buffer_allocator, gemm->operand(0)->shape(),
                                  autotune_config, rng_state));
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase rhs_buffer,
      AutotunerUtil::CreateBuffer(buffer_allocator, gemm->operand(1)->shape(),
                                  autotune_config, rng_state));

  const Shape& output_shape =
      gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0) : gemm->shape();

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase output_buffer,
      AutotunerUtil::CreateBuffer(buffer_allocator, output_shape,
                                  autotune_config, rng_state));

  HloModuleConfig& hlo_module_config = gemm->GetModule()->mutable_config();
  AutotuneResult best_algorithm;
  if (IsCublasLtMatmul(*gemm)) {
    bool has_matrix_bias = config.beta != 0.;

    TF_ASSIGN_OR_RETURN(
        bool has_vector_bias,
        xla::gpu::gpublas_lt::EpilogueAddsVectorBias(gemm_config.epilogue()));

    TF_ASSIGN_OR_RETURN(bool has_aux_output,
                        xla::gpu::gpublas_lt::EpilogueHasAuxiliaryOutput(
                            gemm_config.epilogue()));

    TF_ASSIGN_OR_RETURN(auto epilogue,
                        AsBlasLtEpilogue(gemm_config.epilogue()));

    se::DeviceMemoryBase bias_buffer;
    if (has_vector_bias) {
      TF_ASSIGN_OR_RETURN(
          bias_buffer,
          AutotunerUtil::CreateBuffer(
              buffer_allocator, gemm->operand(has_matrix_bias ? 3 : 2)->shape(),
              autotune_config, rng_state));
    }
    se::DeviceMemoryBase a_scale_buffer, b_scale_buffer, c_scale_buffer,
        d_scale_buffer, d_amax_buffer;

    se::DeviceMemoryBase aux_buffer;
    if (has_aux_output) {
      TF_ASSIGN_OR_RETURN(
          aux_buffer, AutotunerUtil::CreateBuffer(buffer_allocator,
                                                  gemm->shape().tuple_shapes(1),
                                                  autotune_config, rng_state));
    }

    TF_ASSIGN_OR_RETURN(
        auto plan, se::gpu::BlasLt::GetMatmulPlan(stream, config, epilogue));

    TF_ASSIGN_OR_RETURN(auto algorithms, plan->GetAlgorithms());

    TF_ASSIGN_OR_RETURN(
        best_algorithm,
        GetBestAlgorithm<se::gpu::BlasLt::MatmulAlgorithm>(
            stream, buffer_allocator, gemm->ToString(), autotune_config,
            lhs_buffer, rhs_buffer, output_buffer, algorithms, output_shape,
            hlo_module_config, gemm_config.beta(),
            [&](const se::gpu::BlasLt::MatmulAlgorithm& algorithm)
                -> StatusOr<se::blas::ProfileResult> {
              se::OwningScratchAllocator<> scratch_allocator(
                  stream->parent()->device_ordinal(), allocator);
              se::blas::ProfileResult profile_result;
              TF_RETURN_IF_ERROR(plan->ExecuteOnStream(
                  stream, lhs_buffer, rhs_buffer, output_buffer, output_buffer,
                  bias_buffer, aux_buffer, a_scale_buffer, b_scale_buffer,
                  c_scale_buffer, d_scale_buffer, d_amax_buffer, algorithm,
                  scratch_allocator, &profile_result));
              return std::move(profile_result);
            }));
  } else {
    std::vector<se::blas::AlgorithmType> algorithms;
    TF_RET_CHECK(stream->parent()->GetBlasGemmAlgorithms(stream, &algorithms));

    TF_ASSIGN_OR_RETURN(
        best_algorithm,
        GetBestBlasAlgorithm(
            stream, buffer_allocator, gemm->ToString(), autotune_config,
            lhs_buffer, rhs_buffer, output_buffer, algorithms, output_shape,
            hlo_module_config, gemm_config.beta(),
            [&](const se::blas::AlgorithmType& algorithm)
                -> StatusOr<se::blas::ProfileResult> {
              se::blas::ProfileResult profile_result;
              // We expect GemmWithAlgorithm to fail sometimes
              // -- in fact, it will fail for all algorithms if
              // we're targeting < sm_50.  But because we pass a
              // non-null ProfileResult, DoGemmWithAlgorithm
              // should always return true, and the actual
              // success-ness is returned in
              // ProfileResult::is_valid.
              TF_RETURN_IF_ERROR(RunGemm(config, lhs_buffer, rhs_buffer,
                                         output_buffer, deterministic_ops,
                                         stream, algorithm, &profile_result));
              return std::move(profile_result);
            }));
    if (best_algorithm.has_gemm()) {
      int alg_idx = best_algorithm.gemm().algorithm();
      best_algorithm.mutable_gemm()->set_algorithm(algorithms[alg_idx]);
    }
  }
  return best_algorithm;
}

#endif  // (defined(GOOGLE_CUDA) && GOOGLE_CUDA)

// Do Gemm Autotune without stream executor. Use results from autotune cache
// only.
StatusOr<bool> RunOnInstruction(HloInstruction* gemm,
                                const AutotuneConfig& config) {
  VLOG(3) << "Loading the autotune result of GemmThunk " << gemm->ToString();

  AutotuneCacheKey key(config.GetModelStr(), *gemm);

  TF_ASSIGN_OR_RETURN(AutotuneResult algorithm,
                      AutotunerUtil::Autotune(gemm, config, [&] {
                        return DoGemmAutotuneNoCache(gemm, key, config);
                      }));

  se::CudaComputeCapability capability = config.GetCudaComputeCapability();
  GemmBackendConfig gemm_config =
      gemm->backend_config<GemmBackendConfig>().value();
  GemmBackendConfig updated_config = gemm_config;

  // We only set the 'algorithm' field on non-Ampere architectures, as for
  // Ampere it's ignored in any case.
  if (!capability.IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    if (algorithm.has_gemm()) {
      updated_config.set_selected_algorithm(algorithm.gemm().algorithm());
    } else {
      updated_config.set_selected_algorithm(se::blas::kRuntimeAutotuning);
    }
  }
  TF_RETURN_IF_ERROR(gemm->set_backend_config(updated_config));
  return updated_config.SerializeAsString() != gemm_config.SerializeAsString();
}

StatusOr<bool> RunOnComputation(HloComputation* computation,
                                AutotuneConfig config) {
  bool changed = false;
  for (HloInstruction* instr : computation->instructions()) {
    if (IsCublasGemm(*instr)) {
      TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr, config));
      changed |= result;
    }
  }
  return changed;
}

}  // namespace

StatusOr<bool> GemmAlgorithmPicker::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("GemmAlgorithmPicker for ", module->name()));

  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    VLOG(2) << "GEMM auto-tuning disabled, GemmAlgorithmPicker returning early";
    return false;
  }

  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation, config_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
