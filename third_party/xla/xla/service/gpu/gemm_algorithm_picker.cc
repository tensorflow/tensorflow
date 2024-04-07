/* Copyright 2019 The OpenXLA Authors.

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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/matmul_utils.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/service/gpu/variant_visitor.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/blas.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/gpu/redzone_allocator.h"
#include "xla/stream_executor/scratch_allocator.h"
#include "xla/tsl/util/proto/proto_utils.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/gpu/buffer_comparator.h"
#endif

namespace xla {
namespace gpu {
namespace {

using se::gpu::BlasLt;

absl::StatusOr<BlasLt::Epilogue> AsBlasLtEpilogue(
    GemmBackendConfig_Epilogue epilogue) {
  switch (epilogue) {
    case GemmBackendConfig::DEFAULT:
      return BlasLt::Epilogue::kDefault;
    case GemmBackendConfig::RELU:
      return BlasLt::Epilogue::kReLU;
    case GemmBackendConfig::GELU:
      return BlasLt::Epilogue::kGELU;
    case GemmBackendConfig::GELU_AUX:
      return BlasLt::Epilogue::kGELUWithAux;
    case GemmBackendConfig::BIAS:
      return BlasLt::Epilogue::kBias;
    case GemmBackendConfig::BIAS_RELU:
      return BlasLt::Epilogue::kBiasThenReLU;
    case GemmBackendConfig::BIAS_GELU:
      return BlasLt::Epilogue::kBiasThenGELU;
    case GemmBackendConfig::BIAS_GELU_AUX:
      return BlasLt::Epilogue::kBiasThenGELUWithAux;
    default:
      return Internal("Unsupported Epilogue.");
  }
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

class GemmAutotuner {
  const AutotuneConfig& autotune_config_;
  se::DeviceMemoryBase lhs_buffer_, rhs_buffer_, output_buffer_;
  std::unique_ptr<se::RedzoneAllocator> redzone_allocator_;
  se::Stream* stream_ = nullptr;
  bool deterministic_ops_ = false;
  int64_t rng_state_ = 0;

 public:
  explicit GemmAutotuner(const AutotuneConfig& autotune_config)
      : autotune_config_(autotune_config) {}

  absl::StatusOr<AutotuneResult> operator()(const HloInstruction* gemm,
                                            const AutotuneCacheKey& key) {
    if (autotune_config_.IsDeviceless()) {
      // Return empty result, will tune at runtime.
      return AutotuneResult{};
    }
    VLOG(3) << "Starting autotune of GemmThunk " << gemm->ToString();

    TF_ASSIGN_OR_RETURN(stream_, autotune_config_.GetStream());
    const DebugOptions& debug_options =
        gemm->GetModule()->config().debug_options();
    deterministic_ops_ = debug_options.xla_gpu_deterministic_ops();

    TF_ASSIGN_OR_RETURN(auto gemm_config, GemmConfig::For(gemm));

    // Don't run autotuning concurrently on the same GPU.
    absl::MutexLock gpu_lock(&GetGpuMutex(stream_->parent()));

    TF_ASSIGN_OR_RETURN(auto buf_alloc, AutotunerUtil::CreateRedzoneAllocator(
                                            autotune_config_, debug_options));
    redzone_allocator_ =
        std::make_unique<se::RedzoneAllocator>(std::move(buf_alloc));

    TF_ASSIGN_OR_RETURN(lhs_buffer_, CreateBuffer(gemm->operand(0)->shape()));
    TF_ASSIGN_OR_RETURN(rhs_buffer_, CreateBuffer(gemm->operand(1)->shape()));
    TF_ASSIGN_OR_RETURN(output_buffer_, CreateBuffer(GetOutputShape(gemm)));

    return IsCublasLtMatmul(*gemm) || IsCublasLtMatmulF8(*gemm)
               ? TuneGpuBlasLt(gemm, gemm_config)
               : TuneGpuBlas(gemm, gemm_config);
  }

 private:
  const Shape& GetOutputShape(const HloInstruction* gemm) {
    return gemm->shape().IsTuple() ? gemm->shape().tuple_shapes(0)
                                   : gemm->shape();
  }

  absl::StatusOr<se::DeviceMemoryBase> CreateBuffer(const Shape& shape) {
    return AutotunerUtil::CreateBuffer(*redzone_allocator_, shape,
                                       autotune_config_, rng_state_);
  }

  absl::StatusOr<AutotuneResult> TuneGpuBlasLt(const HloInstruction* gemm,
                                               const GemmConfig& gemm_config) {
    GpuBackendConfig gpu_config =
        gemm->backend_config<GpuBackendConfig>().value();
    const GemmBackendConfig& backend_config = gpu_config.gemm_backend_config();

    bool has_matrix_bias = gemm_config.beta != 0.;

    TF_ASSIGN_OR_RETURN(
        bool has_vector_bias,
        gpublas_lt::EpilogueAddsVectorBias(backend_config.epilogue()));

    TF_ASSIGN_OR_RETURN(
        bool has_aux_output,
        gpublas_lt::EpilogueHasAuxiliaryOutput(backend_config.epilogue()));

    TF_ASSIGN_OR_RETURN(auto epilogue,
                        AsBlasLtEpilogue(backend_config.epilogue()));

    se::DeviceMemoryBase a_scale_buffer, b_scale_buffer, c_scale_buffer,
        d_scale_buffer, d_amax_buffer, bias_buffer, aux_buffer;

    if (has_vector_bias) {
      TF_ASSIGN_OR_RETURN(
          bias_buffer,
          CreateBuffer(gemm->operand(has_matrix_bias ? 3 : 2)->shape()));
    }
    if (has_aux_output) {
      TF_ASSIGN_OR_RETURN(aux_buffer,
                          CreateBuffer(gemm->shape().tuple_shapes(1)));
    }

    TF_ASSIGN_OR_RETURN(auto plan,
                        BlasLt::GetMatmulPlan(stream_, gemm_config, epilogue));

    TF_ASSIGN_OR_RETURN(auto algorithms, plan->GetAlgorithms());

    auto tuned_func = [&](const BlasLt::MatmulAlgorithm& algorithm)
        -> absl::StatusOr<se::blas::ProfileResult> {
      se::OwningScratchAllocator<> scratch_allocator(
          stream_->parent()->device_ordinal(), autotune_config_.GetAllocator());
      // Run a warmup iteration without the profiler active.
      TF_RETURN_IF_ERROR(plan->ExecuteOnStream(
          stream_, lhs_buffer_, rhs_buffer_, output_buffer_, output_buffer_,
          bias_buffer, aux_buffer, a_scale_buffer, b_scale_buffer,
          c_scale_buffer, d_scale_buffer, d_amax_buffer, algorithm,
          scratch_allocator));
      se::blas::ProfileResult profile_result;
      profile_result.set_warmup_run_executed(true);
      TF_RETURN_IF_ERROR(plan->ExecuteOnStream(
          stream_, lhs_buffer_, rhs_buffer_, output_buffer_, output_buffer_,
          bias_buffer, aux_buffer, a_scale_buffer, b_scale_buffer,
          c_scale_buffer, d_scale_buffer, d_amax_buffer, algorithm,
          scratch_allocator, &profile_result));
      return std::move(profile_result);
    };

    return GetBestAlgorithm<BlasLt::MatmulAlgorithm>(
        gemm, algorithms, gemm_config.beta, tuned_func);
  }

  absl::StatusOr<AutotuneResult> TuneGpuBlas(const HloInstruction* gemm,
                                             const GemmConfig& gemm_config) {
    int64_t workspace_size =
        std::visit(VariantVisitor{[](const se::CudaComputeCapability& cc) {
                                    return cc.IsAtLeastHopper()
                                               ? GemmConfig::kHopperWorkspace
                                               : GemmConfig::kDefaultWorkspace;
                                  },
                                  [](const se::RocmComputeCapability&) {
                                    return GemmConfig::kDefaultWorkspace;
                                  }},
                   autotune_config_.GetGpuComputeCapability());

    TF_ASSIGN_OR_RETURN(
        auto workspace_buffer,
        CreateBuffer(ShapeUtil::MakeShape(S8, {workspace_size})));

    std::vector<se::blas::AlgorithmType> algorithms;
    TF_ASSIGN_OR_RETURN(GemmConfig::DescriptorsTuple desc,
                        gemm_config.GetMatrixDescriptors(
                            lhs_buffer_, rhs_buffer_, output_buffer_));

    auto blas = stream_->parent()->AsBlas();
    if (blas == nullptr) {
      return absl::InternalError("No BLAS support for stream");
    }
    blas->GetBlasGemmAlgorithms(stream_, desc.lhs, desc.rhs, &desc.output,
                                &gemm_config.alpha, &gemm_config.beta,
                                &algorithms);

    AutotuneResult best_algorithm;
#if TENSORFLOW_USE_ROCM        // Blas gemm algorithms can be empty for ROCM
    if (algorithms.empty()) {  // nothing to autotune
      LOG(WARNING) << "No solutions found: skipping autotuning for ROCM..";
      best_algorithm.mutable_gemm()->set_algorithm(se::blas::kDefaultAlgorithm);
      return best_algorithm;
    }
#endif

    auto tuned_func = [&](const se::blas::AlgorithmType& algorithm)
        -> absl::StatusOr<se::blas::ProfileResult> {
      // Do a warm-up run first, without a profile result. RunGemm swallows
      // error codes when profile_result is passed, as it is in the measurement
      // below, but not otherwise. It is, therefore, consistent to ignore the
      // error code here.
      static_cast<void>(RunGemm(gemm_config, lhs_buffer_, rhs_buffer_,
                                output_buffer_, workspace_buffer,
                                deterministic_ops_, stream_, algorithm));
      se::blas::ProfileResult profile_result;
      // Allow GpuTimer to use its delay kernel implementation to improve
      // accuracy.
      profile_result.set_warmup_run_executed(true);
      // We expect GemmWithAlgorithm to fail sometimes -- in fact, it will fail
      // for all algorithms if we're targeting < sm_50. But because we pass a
      // non-null ProfileResult, DoGemmWithAlgorithm should always return true,
      // and the actual success-ness is returned in ProfileResult::is_valid.
      TF_RETURN_IF_ERROR(RunGemm(gemm_config, lhs_buffer_, rhs_buffer_,
                                 output_buffer_, workspace_buffer,
                                 deterministic_ops_, stream_, algorithm,
                                 &profile_result));
      return std::move(profile_result);
    };

    TF_ASSIGN_OR_RETURN(best_algorithm,
                        GetBestAlgorithm<se::blas::AlgorithmType>(
                            gemm, algorithms, gemm_config.beta, tuned_func));
    if (best_algorithm.has_gemm()) {
      int alg_idx = best_algorithm.gemm().algorithm();
      best_algorithm.mutable_gemm()->set_algorithm(algorithms[alg_idx]);
    }
    return best_algorithm;
  }

  // Returns the index (into `algorithms`) of the fastest algorithm.
  template <typename AlgoT, typename TunedFunc>
  absl::StatusOr<AutotuneResult> GetBestAlgorithm(
      const HloInstruction* gemm, absl::Span<const AlgoT> algorithms,
      double beta, TunedFunc&& run_benchmark) {
    static_assert(std::is_invocable_r_v<absl::StatusOr<se::blas::ProfileResult>,
                                        TunedFunc, const AlgoT&>,
                  "Tuned function has incorrect prototype!");

    if (!stream_->parent()->SynchronizeAllActivity()) {
      return Internal("Failed to synchronize GPU for autotuning.");
    }

    auto& hlo_module_config = gemm->GetModule()->mutable_config();
    const auto& output_shape = GetOutputShape(gemm);

    se::DeviceMemoryBase reference_buffer;
    if (autotune_config_.should_check_correctness()) {
      TF_ASSIGN_OR_RETURN(reference_buffer,
                          redzone_allocator_->AllocateBytes(
                              ShapeUtil::ByteSizeOf(output_shape)));
    }

    BufferComparator comparator(output_shape, hlo_module_config);
    std::vector<AutotuneResult> results;
    results.reserve(algorithms.size());
    std::optional<int64_t> reference_algorithm;

    for (const AlgoT& algorithm : algorithms) {
      // Make sure the output buffer always has the same value if we use
      // the bias parameter.
      if (autotune_config_.should_reinit_output_buffer() && beta != 0) {
        int64_t rng_state = 0;
        InitializeBuffer(stream_, output_shape.element_type(), &rng_state,
                         output_buffer_);
      }
      TF_ASSIGN_OR_RETURN(auto profile_result, run_benchmark(algorithm));

      AutotuneResult& result = results.emplace_back();
      result.mutable_gemm()->set_algorithm(profile_result.algorithm());

      if (!profile_result.is_valid()) {  // Unsupported algorithm.
        result.mutable_failure()->set_kind(AutotuneResult::DISQUALIFIED);
        continue;
      }

      VLOG(2) << "gemm algorithm " << profile_result.algorithm() << " took "
              << profile_result.elapsed_time_in_ms() << "ms";

      *result.mutable_run_time() = tsl::proto_utils::ToDurationProto(
          absl::Milliseconds(profile_result.elapsed_time_in_ms()));

      if (!autotune_config_.should_check_correctness()) {
        continue;
      }
      TF_ASSIGN_OR_RETURN(
          se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
          redzone_allocator_->CheckRedzones());

      if (!rz_check_status.ok()) {
        result.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
        *result.mutable_failure()->mutable_msg() =
            rz_check_status.RedzoneFailureMsg();
        LOG(ERROR) << "Detected out-of-bounds write in gemm buffer";
        CHECK(!autotune_config_.should_crash_on_check_failure());
        continue;
      }

      if (!reference_algorithm) {
        TF_RETURN_IF_ERROR(stream_->Memcpy(&reference_buffer, output_buffer_,
                                           output_buffer_.size()));
        reference_algorithm = profile_result.algorithm();
      } else {
        // Perform the comparison.
        TF_ASSIGN_OR_RETURN(
            bool outputs_match,
            comparator.CompareEqual(stream_, /*current=*/output_buffer_,
                                    /*expected=*/reference_buffer));
        if (!outputs_match) {
          LOG(ERROR) << "Results mismatch between different GEMM algorithms. "
                     << "This is likely a bug/unexpected loss of precision.";
          CHECK(!autotune_config_.should_crash_on_check_failure());

          result.mutable_failure()->set_kind(AutotuneResult::WRONG_RESULT);
          result.mutable_failure()->mutable_reference_gemm()->set_algorithm(
              *reference_algorithm);
        }
      }
    }  // for algorithms

    absl::StatusOr<AutotuneResult> best =
        PickBestResult(results, gemm->ToString(), hlo_module_config);
    if (best.ok()) {
      for (size_t i = 0; i < results.size(); ++i) {
        if (best->gemm().algorithm() == results[i].gemm().algorithm()) {
          best->mutable_gemm()->set_algorithm(i);
          return best;
        }
      }
      return Internal("unknown best algorithm");
    }
    LOG(WARNING) << "Failed to find best cuBLAS algorithm, GEMM performance "
                    "might be suboptimal: "
                 << best.status();
    return AutotuneResult{};
  }  // GetBestAlgorithm
};  // GemmAutotuner

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

// Do Gemm Autotune without stream executor. Use results from autotune cache
// only.
absl::StatusOr<bool> RunOnInstruction(HloInstruction* gemm,
                                      const AutotuneConfig& config) {
  VLOG(3) << "Loading the autotune result of GemmThunk " << gemm->ToString();

  GpuBackendConfig gpu_config =
      gemm->backend_config<GpuBackendConfig>().value();
  GemmBackendConfig& backend_config = *gpu_config.mutable_gemm_backend_config();

  // Degenerate gemms replaced with memzero operation, no need to auto tune it.
  if (backend_config.alpha_real() == 0.0 &&
      backend_config.alpha_imag() == 0.0 && backend_config.beta() == 0.0) {
    VLOG(3) << "Skip degenerate gemm instruction auto tuning";
    return false;
  }

  AutotuneCacheKey key(config.GetModelStr(), *gemm);
  GemmAutotuner autotuner(config);
  TF_ASSIGN_OR_RETURN(AutotuneResult algorithm,
                      AutotunerUtil::Autotune(
                          gemm, config, [&] { return autotuner(gemm, key); }));

  auto old_algorithm = backend_config.selected_algorithm();
  bool update_algorithm =
      IsCublasLtMatmulF8(*gemm) ||
      std::visit(VariantVisitor{[](const se::CudaComputeCapability& cc) {
                                  // We only set the 'algorithm' field on
                                  // non-Ampere architectures, as for Ampere
                                  // it's ignored in any case.
                                  return !cc.IsAtLeast(
                                      se::CudaComputeCapability::AMPERE);
                                },
                                [](const se::RocmComputeCapability&) {
                                  return true;  // TODO: not decided yet
                                }},
                 config.GetGpuComputeCapability());

  if (update_algorithm) {
    int64_t new_algorithm{};
    if (algorithm.has_gemm()) {
      new_algorithm = algorithm.gemm().algorithm();
    } else {
      // NOTE: runtime autotuning is no longer available => set to default
      new_algorithm = se::blas::kDefaultAlgorithm;
    }

    if (new_algorithm == old_algorithm &&
        backend_config.has_selected_algorithm()) {
      // We don't need to update the backend config if
      // the algorithm hasn't changed unless previously
      // the algorithm wasn't set explicitly.
      return false;
    }

    backend_config.set_selected_algorithm(new_algorithm);
    TF_RETURN_IF_ERROR(gemm->set_backend_config(gpu_config));
    return true;  // We changed `gemm`
  }

  return false;  // No change to `gemm`
}

absl::StatusOr<bool> RunOnComputation(HloComputation* computation,
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

absl::StatusOr<bool> GemmAlgorithmPicker::Run(
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
