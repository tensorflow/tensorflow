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

#include <algorithm>
#include <functional>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <variant>

#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_serializable_autotuner.h"
#include "tensorflow/compiler/xla/service/gpu/matmul_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/stream_executor/blas.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory_allocator.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/logger.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/protobuf/autotuning.pb.h"
#include "tensorflow/tsl/util/proto/proto_utils.h"

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/stream_executor/cuda/cuda_blas_lt.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#endif

namespace xla {
namespace gpu {

using tensorflow::AutotuneResult;

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
se::RedzoneAllocator CreateRedzoneAllocator(
    se::Stream* stream, se::DeviceMemoryAllocator* allocator,
    const DebugOptions& debug_options, const AutotuneConfig& config) {
  int64_t redzone_size = config.should_check_correctness()
                             ? se::RedzoneAllocator::kDefaultRedzoneSize
                             : 0;

  return se::RedzoneAllocator(
      stream, allocator, PtxOptsFromDebugOptions(debug_options),
      /*memory_limit=*/std::numeric_limits<int64_t>::max(),
      /*redzone_size=*/redzone_size);
}
#endif

// Returns the index (into `algorithms`) of the fastest algorithm.
template <typename AlgoT>
StatusOr<std::optional<size_t>> GetBestAlgorithm(
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
      CHECK(!autotune_config.should_crash_on_check_failure);
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
          comparator.CompareEqual(stream, output_buffer, reference_buffer));
      if (!outputs_match) {
        LOG(ERROR) << "Results mismatch between different GEMM algorithms. "
                   << "This is likely a bug/unexpected loss of precision.";
        CHECK(!autotune_config.should_crash_on_check_failure);

        result.mutable_failure()->set_kind(AutotuneResult::WRONG_RESULT);
        result.mutable_failure()->mutable_reference_gemm()->set_algorithm(
            *reference_algorithm);
      }
    }
  }

  if (!autotune_config.should_crash_on_check_failure) {
    tensorflow::AutotuningLog log;
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
        return {i};
      }
    }
    return InternalError("unknown best algorithm");
  }

  LOG(WARNING) << "Failed to find best cuBLAS algorithm, GEMM performance "
                  "might be suboptimal: "
               << best.status();
  return {std::nullopt};
}

StatusOr<std::optional<size_t>> GetBestBlasAlgorithm(
    se::Stream* stream, se::RedzoneAllocator& allocator,
    std::optional<std::string_view> gemm_str,
    const AutotuneConfig& autotune_config, se::DeviceMemoryBase lhs_buffer,
    se::DeviceMemoryBase rhs_buffer, se::DeviceMemoryBase output_buffer,
    absl::Span<const se::blas::AlgorithmType> algorithms,
    const Shape& output_shape, const HloModuleConfig& hlo_module_config,
    double beta,
    const std::function<StatusOr<se::blas::ProfileResult>(
        const se::blas::AlgorithmType&)>& run_benchmark) {
  TF_ASSIGN_OR_RETURN(
      std::optional<size_t> result,
      GetBestAlgorithm<se::blas::AlgorithmType>(
          stream, allocator, gemm_str, autotune_config, lhs_buffer, rhs_buffer,
          output_buffer, algorithms, output_shape, hlo_module_config, beta,
          run_benchmark));
  return result;
}

namespace {

StatusOr<se::cuda::BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return se::cuda::BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return se::cuda::BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return se::cuda::BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return se::cuda::BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return se::cuda::BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return se::cuda::BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return se::cuda::BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return se::cuda::BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return InternalError("Unsupported Epilogue.");
  }
}

StatusOr<se::DeviceMemoryBase> CreateBuffer(se::RedzoneAllocator& allocator,
                                            const Shape& shape,
                                            const AutotuneConfig& config,
                                            int64_t& rng_state) {
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                      allocator.AllocateBytes(ShapeUtil::ByteSizeOf(shape)));
  if (config.should_init_buffers()) {
    InitializeBuffer(allocator.stream(), shape.element_type(), &rng_state,
                     buffer);
  }
  return buffer;
}

StatusOr<se::DeviceMemoryBase> CreateBuffer(se::RedzoneAllocator& allocator,
                                            const HloInstruction& op,
                                            const AutotuneConfig& config,
                                            int64_t& rng_state) {
  return CreateBuffer(allocator, op.shape(), config, rng_state);
}

static absl::Mutex autotune_cache_mu(absl::kConstInit);
static auto& autotune_cache ABSL_GUARDED_BY(autotune_cache_mu) =
    *new absl::flat_hash_map<AutotuneCacheKey,
                             std::optional<se::blas::AlgorithmType>>();
static int64_t autotune_cache_hits ABSL_GUARDED_BY(autotune_cache_mu) = 0;
static int64_t autotune_cache_misses ABSL_GUARDED_BY(autotune_cache_mu) = 0;

#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)

StatusOr<std::optional<se::blas::AlgorithmType>> DoGemmAutotune(
    const HloInstruction* gemm, const GemmBackendConfig& gemm_config,
    se::DeviceMemoryAllocator* allocator, se::Stream* stream) {
  VLOG(3) << "Starting autotune of GemmThunk " << gemm->ToString();

  auto key = AutotuneCacheKeyFromInstruction(
      gemm, stream->parent()->GetDeviceDescription().model_str());

  {
    absl::MutexLock lock(&autotune_cache_mu);
    auto it = autotune_cache.find(key);
    int64_t requests = autotune_cache_hits + autotune_cache_misses;
    if (requests && requests % 10 == 0) {
      VLOG(2) << "Autotuning cache hits/(hits + misses): "
              << autotune_cache_hits << "/" << requests;
    }

    if (it != autotune_cache.end()) {
      autotune_cache_hits++;
      VLOG(4) << "Autotuning cache hit, using algorithm: "
              << (it->second.has_value() ? absl::StrCat(*(it->second))
                                         : "<generic>");
      return it->second;
    }
    VLOG(4) << "Autotuning cache miss";
    autotune_cache_misses++;
  }

  const DebugOptions& debug_options =
      gemm->GetModule()->config().debug_options();
  AutotuneConfig autotune_config = GetConfig(debug_options);

  TF_ASSIGN_OR_RETURN(GemmConfig config, GemmConfig::For(gemm));
  // Don't run autotuning concurrently on the same GPU.
  absl::MutexLock gpu_lock(&GetGpuMutex(stream->parent()));

  se::RedzoneAllocator buffer_allocator =
      CreateRedzoneAllocator(stream, allocator, debug_options, autotune_config);

  int64_t rng_state = 0;
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase lhs_buffer,
                      CreateBuffer(buffer_allocator, *gemm->operand(0),
                                   autotune_config, rng_state));
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase rhs_buffer,
                      CreateBuffer(buffer_allocator, *gemm->operand(1),
                                   autotune_config, rng_state));

  const Shape& output_shape =
      gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0) : gemm->shape();

  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase output_buffer,
      CreateBuffer(buffer_allocator, output_shape, autotune_config, rng_state));

  HloModuleConfig& hlo_module_config = gemm->GetModule()->config();
  std::optional<se::blas::AlgorithmType> best_algorithm;
  if (IsCublasLtMatmul(*gemm)) {
    bool has_matrix_bias = config.beta != 0.;

    TF_ASSIGN_OR_RETURN(bool has_vector_bias, cublas_lt::EpilogueAddsVectorBias(
                                                  gemm_config.epilogue()));

    TF_ASSIGN_OR_RETURN(
        bool has_aux_output,
        cublas_lt::EpilogueHasAuxiliaryOutput(gemm_config.epilogue()));

    TF_ASSIGN_OR_RETURN(auto epilogue,
                        AsBlasLtEpilogue(gemm_config.epilogue()));

    se::DeviceMemoryBase bias_buffer;
    if (has_vector_bias) {
      TF_ASSIGN_OR_RETURN(bias_buffer,
                          CreateBuffer(buffer_allocator,
                                       *gemm->operand(has_matrix_bias ? 3 : 2),
                                       autotune_config, rng_state));
    }
    se::DeviceMemoryBase a_scale_buffer, b_scale_buffer, c_scale_buffer,
        d_scale_buffer, d_amax_buffer;

    se::DeviceMemoryBase aux_buffer;
    if (has_aux_output) {
      TF_ASSIGN_OR_RETURN(
          aux_buffer,
          CreateBuffer(buffer_allocator, gemm->shape().tuple_shapes(1),
                       autotune_config, rng_state));
    }

    TF_ASSIGN_OR_RETURN(auto plan,
                        cublas_lt::MatmulPlan::From(config, epilogue));
    TF_ASSIGN_OR_RETURN(
        std::vector<se::cuda::BlasLt::MatmulAlgorithm> algorithms,
        plan.GetAlgorithms(stream));

    TF_ASSIGN_OR_RETURN(
        std::optional<size_t> best_algorithm_idx,
        GetBestAlgorithm<se::cuda::BlasLt::MatmulAlgorithm>(
            stream, buffer_allocator, gemm->ToString(), autotune_config,
            lhs_buffer, rhs_buffer, output_buffer, algorithms, output_shape,
            hlo_module_config, gemm_config.beta(),
            [&](const se::cuda::BlasLt::MatmulAlgorithm& algorithm)
                -> StatusOr<se::blas::ProfileResult> {
              se::OwningScratchAllocator<> scratch_allocator(
                  stream->parent()->device_ordinal(), allocator);
              se::blas::ProfileResult profile_result;
              TF_RETURN_IF_ERROR(plan.ExecuteOnStream(
                  stream, lhs_buffer, rhs_buffer, output_buffer, output_buffer,
                  bias_buffer, aux_buffer, a_scale_buffer, b_scale_buffer,
                  c_scale_buffer, d_scale_buffer, d_amax_buffer, algorithm,
                  scratch_allocator, &profile_result));
              return std::move(profile_result);
            }));

    TF_RET_CHECK(best_algorithm_idx) << "failed to auto-tune cublas_lt matmul";
    best_algorithm = *best_algorithm_idx;
  } else {
    std::vector<se::blas::AlgorithmType> algorithms;
    TF_RET_CHECK(stream->parent()->GetBlasGemmAlgorithms(stream, &algorithms));

    TF_ASSIGN_OR_RETURN(std::optional<size_t> best_algorithm_idx,
                        GetBestBlasAlgorithm(
                            stream, buffer_allocator, gemm->ToString(),
                            autotune_config, lhs_buffer, rhs_buffer,
                            output_buffer, algorithms, output_shape,
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
                              TF_RETURN_IF_ERROR(RunGemm(
                                  config, lhs_buffer, rhs_buffer, output_buffer,
                                  stream, algorithm, &profile_result));
                              return std::move(profile_result);
                            }));

    if (best_algorithm_idx) best_algorithm = algorithms[*best_algorithm_idx];
  }

  // Insert our result into the cache.  After we released the lock on
  // autotune_cache_mu, another autotuning job may have run for this same key on
  // another GPU on the machine.  If so, use its result.
  absl::MutexLock lock(&autotune_cache_mu);
  auto [it, inserted] = autotune_cache.emplace(key, best_algorithm);
  return it->second;
}

StatusOr<bool> RunOnInstruction(HloInstruction* instr, DeviceConfig config) {
  se::StreamExecutor* executor = config.stream_exec;
  se::DeviceMemoryAllocator* allocator = config.allocator;
  if (allocator == nullptr) {
    allocator = executor->GetAllocator();
  }
  TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                      allocator->GetStream(executor->device_ordinal()));

  GemmBackendConfig gemm_config =
      instr->backend_config<GemmBackendConfig>().value();

  TF_ASSIGN_OR_RETURN(std::optional<se::blas::AlgorithmType> gemm_algorithm,
                      DoGemmAutotune(instr, gemm_config, allocator, stream));

  // We update instruction->backend_config(); if no algorithms are supported,
  // a different API is used, which does not require specifying an algorithm.
  GemmBackendConfig updated_config = gemm_config;

  // We only set the 'algorithm' field on non-Ampere architectures, as for
  // Ampere it's ignored in any case.
  if (gemm_algorithm &&
      !executor->GetDeviceDescription().cuda_compute_capability().IsAtLeast(
          se::CudaComputeCapability::AMPERE)) {
    VLOG(4) << "GEMM autotuning picked algorithm " << *gemm_algorithm << " for "
            << instr->name();
    updated_config.set_selected_algorithm(*gemm_algorithm);
  }
  TF_RETURN_IF_ERROR(instr->set_backend_config(updated_config));
  return updated_config.SerializeAsString() != gemm_config.SerializeAsString();
}

#endif

// Do Gemm Autotune without stream executor. Use results from autotune cache
// only.
StatusOr<bool> RunOnInstruction(HloInstruction* gemm, DevicelessConfig config) {
  VLOG(3) << "Loading the autotune result of GemmThunk " << gemm->ToString();

  auto key = AutotuneCacheKeyFromInstruction(gemm, config.model_str);

  // Load selected algorithm from the autotune cache.
  std::optional<se::blas::AlgorithmType> algorithm;
  {
    absl::MutexLock lock(&autotune_cache_mu);
    if (auto it = autotune_cache.find(key); it != autotune_cache.end()) {
      VLOG(4) << "AOT autotuning cache hit, using algorithm: "
              << (it->second.has_value() ? absl::StrCat(*(it->second))
                                         : "<generic>");
      algorithm = it->second;
    }
    VLOG(4) << "AOT autotuning cache miss";
  }

  se::CudaComputeCapability capability = config.cuda_compute_capability;
  GemmBackendConfig gemm_config =
      gemm->backend_config<GemmBackendConfig>().value();
  GemmBackendConfig updated_config = gemm_config;

  // We only set the 'algorithm' field on non-Ampere architectures, as for
  // Ampere it's ignored in any case.
  if (!capability.IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    if (algorithm) {
      updated_config.set_selected_algorithm(*algorithm);
    } else {
      updated_config.set_selected_algorithm(se::blas::kRuntimeAutotuning);
    }
  }
  TF_RETURN_IF_ERROR(gemm->set_backend_config(updated_config));
  return updated_config.SerializeAsString() != gemm_config.SerializeAsString();
}

StatusOr<bool> RunOnComputation(HloComputation* computation,
                                AutotuningConfig config) {
  bool changed = false;
  for (HloInstruction* instr : computation->instructions()) {
    if (IsCublasGemm(*instr)) {
      bool result;
      if (std::holds_alternative<DeviceConfig>(config)) {
#if (defined(GOOGLE_CUDA) && GOOGLE_CUDA)
        TF_ASSIGN_OR_RETURN(
            result, RunOnInstruction(instr, std::get<DeviceConfig>(config)));
#else
        LOG(FATAL) << "GPU-enabled build is required to run autotuning";
#endif
      } else {
        TF_ASSIGN_OR_RETURN(
            result,
            RunOnInstruction(instr, std::get<DevicelessConfig>(config)));
      }
      changed |= result;
    }
  }
  return changed;
}

}  // namespace

void GemmAlgorithmPicker::ClearAutotuneResults() {
  absl::MutexLock lock(&autotune_cache_mu);
  autotune_cache.clear();
}

Status GemmAlgorithmPicker::WriteAutotuneResults(AutotuneResults* results) {
  absl::MutexLock lock(&autotune_cache_mu);

  for (const auto& [k, result] : autotune_cache) {
    // For now, we don't cache "failed to autotune" results, because we don't
    // have a good way to represent them in the proto.
    if (!result.has_value()) continue;

    const auto& [model_str, hlo] = k;
    auto& entry = *results->add_dots();
    entry.set_device(model_str);
    entry.set_hlo(hlo);
    entry.mutable_result()->mutable_gemm()->set_algorithm(*result);
  }

  // Sort the results so they're deterministic.
  std::sort(results->mutable_dots()->pointer_begin(),
            results->mutable_dots()->pointer_end(),
            [](const auto* a, const auto* b) {
              return std::make_pair(absl::string_view(a->device()),
                                    absl::string_view(a->hlo())) <
                     std::make_pair(absl::string_view(b->device()),
                                    absl::string_view(b->hlo()));
            });
  return OkStatus();
}

Status GemmAlgorithmPicker::LoadAutotuneResults(
    const AutotuneResults& results) {
  absl::MutexLock lock(&autotune_cache_mu);
  for (const auto& result : results.dots()) {
    autotune_cache[std::make_tuple(result.device(), result.hlo())] =
        result.result().gemm().algorithm();
  }
  return OkStatus();
}

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
