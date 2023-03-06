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

#include "tensorflow/compiler/xla/service/gpu/triton_autotuner.h"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/node_hash_map.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/xla/autotune_results.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/gpu/target_util.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/util/proto/proto_utils.h"

namespace xla {
namespace gpu {

namespace {

using tensorflow::AutotuneResult;

// Constructs an autotuning key for a gemm performed in Triton.
static AutotuneResult::TritonGemmKey GemmKey(int64_t block_m, int64_t block_n,
                                             int64_t block_k, int64_t split_k,
                                             int64_t num_stages,
                                             int64_t num_warps) {
  AutotuneResult::TritonGemmKey key;
  key.set_block_m(block_m);
  key.set_block_n(block_n);
  key.set_block_k(block_k);
  key.set_split_k(split_k);
  key.set_num_stages(num_stages);
  key.set_num_warps(num_warps);
  return key;
}

// TODO(b/266210099): have a way to generate/load these dynamically.
// Returns a list of possible tilings for a gemm performed in Triton.
static std::vector<AutotuneResult::TritonGemmKey>
GetPossibleMatmulAutotuneConfigs() {
  return {GemmKey(128, 256, 32, 1, 3, 8),  GemmKey(256, 128, 32, 1, 3, 8),
          GemmKey(256, 64, 32, 1, 4, 4),   GemmKey(64, 256, 32, 1, 4, 4),
          GemmKey(128, 64, 32, 1, 4, 4),   GemmKey(64, 128, 32, 1, 4, 4),
          GemmKey(128, 256, 32, 1, 3, 8),  GemmKey(256, 128, 128, 1, 3, 8),
          GemmKey(256, 64, 128, 1, 4, 4),  GemmKey(64, 256, 128, 1, 4, 4),
          GemmKey(128, 128, 128, 1, 4, 4), GemmKey(128, 64, 64, 1, 4, 4),
          GemmKey(64, 128, 64, 1, 4, 4),   GemmKey(128, 32, 64, 1, 4, 4),
          GemmKey(64, 32, 64, 1, 4, 4),    GemmKey(32, 128, 32, 1, 4, 4),
          GemmKey(64, 32, 64, 1, 2, 8),    GemmKey(128, 128, 32, 1, 4, 4),
          GemmKey(32, 32, 256, 1, 1, 4)};
}

// We assume that the string representation is general enough for caching
// purposes.
// TODO(b/266210099): This is unsound. We should probably do the fingerprint of
// the HLO computation proto instead.
std::string ToCanonicalString(const HloComputation* key) {
  HloPrintOptions options = HloPrintOptions::Canonical();
  options.set_print_subcomputation_mode(
      HloPrintOptions::PrintSubcomputationMode::kOff);
  options.set_print_infeed_outfeed_config(false);
  options.set_print_only_essential_constants(true);
  options.set_print_operand_shape(true);
  options.set_print_ids(false);
  options.set_canonicalize_computations(true);
  return key->ToString(options);
}

static absl::Mutex autotune_cache_mu(absl::kConstInit);

static auto& autotune_cache ABSL_GUARDED_BY(autotune_cache_mu) =
    *new AutotuneCacheMap();

struct TritonTilingWrapper {
  AutotuneResult::TritonGemmKey key;

  template <typename H>
  friend H AbslHashValue(H h, const TritonTilingWrapper& w) {
    return H::combine(std::move(h), w.key.SerializeAsString());
  }

  bool operator==(const TritonTilingWrapper& w) const {
    return key.SerializeAsString() == w.key.SerializeAsString();
  }
};

// TODO(b/266210099): Do not duplicate vs. gemm_algorithm_picker.
struct AutotuneConfig {
  bool should_init_buffers() const { return autotune_level >= 2; }
  bool should_reinit_output_buffer() const { return autotune_level >= 3; }
  bool should_check_correctness() const { return autotune_level >= 4; }

  int32_t autotune_level;
  bool should_crash_on_check_failure;
};

struct CompilationResult {
  std::string ptx;
  std::vector<uint8_t> cubin;
  LaunchDimensions launch_dimensions;
};

// TODO(b/266210099): Do not duplicate this functionality with
// gemm_algorithm_picker.
static AutotuneConfig GetConfig(const DebugOptions& debug_options) {
  return {debug_options.xla_gpu_autotune_level(),
          debug_options.xla_gpu_crash_on_verification_failures()};
}

// Create a buffer for a given operation using redzone checker, initialize based
// on a given rng state.
// TODO(b/266210099): Do not duplicate this functionality with
// gemm_algorithm_picker.
static StatusOr<se::DeviceMemoryBase> CreateBuffer(
    se::RedzoneAllocator& allocator, const HloInstruction& op,
    const AutotuneConfig& config, int64_t& rng_state) {
  TF_ASSIGN_OR_RETURN(
      se::DeviceMemoryBase buffer,
      allocator.AllocateBytes(ShapeUtil::ByteSizeOf(op.shape())));
  if (config.should_init_buffers()) {
    InitializeBuffer(allocator.stream(), op.shape().element_type(), &rng_state,
                     buffer);
  }
  return buffer;
}

class TritonAutotunerVisitor : public DfsHloRewriteVisitor {
 public:
  TritonAutotunerVisitor(const AutotuningConfig& config, int num_extra_threads)
      : config_(config), num_extra_threads_(num_extra_threads) {}

  Status HandleFusion(HloInstruction* hlo) override {
    if (hlo->raw_backend_config_string() != kTritonGemmBackendConfig) {
      return OkStatus();
    }

    TF_ASSIGN_OR_RETURN(AutotuneResult autotune_result,
                        AutotuneMatmul(hlo->called_computations()[0]));

    TF_RET_CHECK(autotune_result.has_triton());
    AutotuneResult::TritonGemmKey tiling = autotune_result.triton();
    TF_RETURN_IF_ERROR(hlo->set_backend_config(tiling));
    MarkAsChanged();
    return OkStatus();
  }

 private:
  // Autotune a tiling for a given matmul fusion.
  StatusOr<AutotuneResult> AutotuneMatmul(HloComputation* fusion) {
    if (auto deviceless_config = std::get_if<DevicelessConfig>(&config_)) {
      const std::string& device_description = deviceless_config->model_str;
      AutotuneCacheKey key =
          std::make_tuple(ToCanonicalString(fusion), device_description);
      if (AutotuneResult* autotune_result = TryFindInCache(key)) {
        return *autotune_result;
      }

      return InternalError("Not found");
    }

    const auto& device_config = std::get<DeviceConfig>(config_);
    const std::string& device_description =
        device_config.stream_exec->GetDeviceDescription().model_str();

    AutotuneCacheKey key =
        std::make_tuple(ToCanonicalString(fusion), device_description);
    if (AutotuneResult* autotune_result = TryFindInCache(key)) {
      return *autotune_result;
    }

    TF_ASSIGN_OR_RETURN(AutotuneResult autotune_result,
                        AutotuneMatmulNoCache(fusion, device_config));

    absl::MutexLock lock(&autotune_cache_mu);
    auto [it, inserted] = autotune_cache.emplace(key, autotune_result);
    return it->second;
  }

  AutotuneResult* TryFindInCache(const AutotuneCacheKey& key) {
    absl::MutexLock lock(&autotune_cache_mu);
    auto it = autotune_cache.find(key);
    if (it != autotune_cache.end()) {
      VLOG(1) << "Autotune cache hit";
      return &it->second;
    }
    return nullptr;
  }

  StatusOr<AutotuneResult> AutotuneMatmulNoCache(
      HloComputation* fusion, const DeviceConfig& device_config) {
    se::StreamExecutor* stream_exec = device_config.stream_exec;
    if (!stream_exec->SynchronizeAllActivity()) {
      return InternalError("Failed to synchronize GPU for autotuning.");
    }

    HloInstruction* root = fusion->root_instruction();
    CHECK(!root->shape().IsTuple())
        << "Can only autotune single-output fusions";
    TF_ASSIGN_OR_RETURN(
        se::Stream* const stream,
        device_config.allocator->GetStream(stream_exec->device_ordinal()));

    DebugOptions debug_opts = fusion->parent()->config().debug_options();
    auto autotune_cfg = GetConfig(debug_opts);

    std::vector<AutotuneResult> results;
    se::RedzoneAllocator rz_allocator(
        stream, device_config.allocator, PtxOptsFromDebugOptions(debug_opts),
        /*memory_limit=*/std::numeric_limits<int64_t>::max(),
        /*redzone_size=*/autotune_cfg.should_check_correctness()
            ? se::RedzoneAllocator::kDefaultRedzoneSize
            : 0);

    std::optional<AutotuneResult::TritonGemmKey> reference_tiling;
    se::DeviceMemoryBase reference_buffer;
    if (autotune_cfg.should_check_correctness()) {
      TF_ASSIGN_OR_RETURN(
          reference_buffer,
          rz_allocator.AllocateBytes(ShapeUtil::ByteSizeOf(root->shape())));
    }

    BufferComparator comparator(root->shape(), fusion->parent()->config());

    // Pre-compile all versions first using the thread pool.
    if (num_extra_threads_ > 0) {
      tsl::thread::ThreadPool thread_pool(
          tsl::Env::Default(), "compilation_pool", num_extra_threads_);
      for (const AutotuneResult::TritonGemmKey& conf :
           GetPossibleMatmulAutotuneConfigs()) {
        thread_pool.Schedule([=] {
          StatusOr<CompilationResult*> res =
              Compile(fusion, device_config, conf);
          if (!res.ok()) {
            LOG(ERROR) << "Failure: " << res.status().ToString();
          }
        });
      }
    }

    std::vector<se::DeviceMemoryBase> args;
    int64_t rng_state = 0;
    for (const HloInstruction* param : fusion->parameter_instructions()) {
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryBase param_buffer,
          CreateBuffer(rz_allocator, *param, autotune_cfg, rng_state));
      args.push_back(param_buffer);
    }

    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase output_buffer,
        CreateBuffer(rz_allocator, *root, autotune_cfg, rng_state));
    args.push_back(output_buffer);

    for (AutotuneResult::TritonGemmKey& conf :
         GetPossibleMatmulAutotuneConfigs()) {
      VLOG(1) << "Trying triton tiling: " << conf.DebugString();

      AutotuneResult res;
      *res.mutable_triton() = conf;

      TF_ASSIGN_OR_RETURN(
          std::optional<absl::Duration> duration,
          RunMatmulWithConfig(fusion, conf, device_config, stream, args));

      if (!duration) {
        VLOG(1) << "Skipping tiling " << conf.DebugString();
        continue;
      }

      *res.mutable_run_time() = tsl::proto_utils::ToDurationProto(*duration);
      VLOG(1) << "Running kernel took: " << *duration;

      TF_ASSIGN_OR_RETURN(
          se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
          rz_allocator.CheckRedzones());

      if (!rz_check_status.ok()) {
        res.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
        *res.mutable_failure()->mutable_msg() =
            rz_check_status.RedzoneFailureMsg();
        CHECK(!autotune_cfg.should_crash_on_check_failure);
        continue;
      }

      if (!reference_tiling && autotune_cfg.should_check_correctness()) {
        stream->ThenMemcpy(&reference_buffer, output_buffer,
                           output_buffer.size());
        reference_tiling = res.triton();
      } else {
        TF_ASSIGN_OR_RETURN(
            bool outputs_match,
            comparator.CompareEqual(stream, output_buffer, reference_buffer));
        if (!outputs_match) {
          LOG(ERROR) << "Results mismatch between different tilings. "
                     << "This is likely a bug/unexpected loss of precision.";
          CHECK(!autotune_cfg.should_crash_on_check_failure);
          res.mutable_failure()->set_kind(AutotuneResult::WRONG_RESULT);
        }
      }
      results.push_back(res);
    }

    TF_ASSIGN_OR_RETURN(
        AutotuneResult best,
        PickBestResult(results, root->ToString(), root->GetModule()->config()));
    return best;
  }

  // Run a fusion with a given tiling on given buffers.
  // Returns `true` if run successfully, `false` if the tiling has to be
  // skipped.
  StatusOr<std::optional<absl::Duration>> RunMatmulWithConfig(
      HloComputation* hlo_computation,
      const AutotuneResult::TritonGemmKey& autotune_config,
      const DeviceConfig& device_config, se::Stream* stream,
      absl::Span<se::DeviceMemoryBase const> device_buffers) {
    TF_ASSIGN_OR_RETURN(
        CompilationResult * res,
        Compile(hlo_computation, device_config, autotune_config));
    if (!res) {
      // Out of shmem budget.
      return {std::nullopt};
    }

    // Don't run autotuning concurrently on the same GPU.
    absl::MutexLock gpu_lock(&GetGpuMutex(stream->parent()));

    auto& [ptx, cubin, launch_dimensions] = *res;

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::KernelBase> kernel,
        // TODO(cheshire): Where is "1" coming from?
        CreateKernel(absl::StrCat(triton_fn_name_, 1), device_buffers.size(),
                     ptx, cubin, stream->parent(),
                     launch_dimensions.SharedMemBytes()));

    se::gpu::GpuExecutor* cuda_executor =
        dynamic_cast<se::gpu::GpuExecutor*>(stream->parent()->implementation());
    std::unique_ptr<se::gpu::GpuTimer, se::gpu::GpuTimerDeleter> timer(
        new se::gpu::GpuTimer(cuda_executor));
    // Warmup: in and out buffers are reused while probing different configs, so
    // GPU caches should be in some comparable states during measurements.
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    if (!timer->Init() || !timer->Start(se::gpu::AsGpuStream(stream))) {
      return Status(tsl::error::INTERNAL, "Failed to start timer");
    }
    TF_RETURN_IF_ERROR(ExecuteKernelOnStream(*kernel, device_buffers,
                                             launch_dimensions, stream));

    if (!timer->Stop(se::gpu::AsGpuStream(stream))) {
      return Status(tsl::error::INTERNAL, "Failed to stop timer");
    }
    return std::make_optional(absl::Nanoseconds(timer->Nanoseconds()));
  }

  // Compile a given computation with a given autotuning config, utilizing
  // computation cache. Returns a raw pointer into the map to avoid copying the
  // values. Returning `nullptr` means that the kernel could not be generated.
  StatusOr<CompilationResult*> Compile(
      HloComputation* hlo_computation, const DeviceConfig& device_config,
      const AutotuneResult::TritonGemmKey& autotune_config) {
    using CompilationKey = std::pair<std::string, TritonTilingWrapper>;
    static absl::Mutex mutex(absl::kConstInit);
    static auto& cache ABSL_GUARDED_BY(mutex) =
        *new absl::node_hash_map<CompilationKey,
                                 std::optional<CompilationResult>>();
    CompilationKey key = std::make_pair(ToCanonicalString(hlo_computation),
                                        TritonTilingWrapper{autotune_config});

    // TODO(b/266210099): Avoid duplication.
    {
      absl::MutexLock lock(&mutex);
      auto it = cache.find(key);
      if (it != cache.end()) {
        std::optional<CompilationResult>& res = it->second;
        if (res.has_value()) {
          VLOG(1) << "Compilation cache hit";
          return &*res;
        }
        return nullptr;
      }
    }

    TF_ASSIGN_OR_RETURN(
        std::optional<CompilationResult> res,
        CompileNoCache(hlo_computation, device_config, autotune_config));
    {
      absl::MutexLock lock(&mutex);
      auto [it2, inserted] = cache.emplace(key, res);
      std::optional<CompilationResult>& res_inserted = it2->second;
      if (res_inserted.has_value()) {
        return &*res_inserted;
      }
      return nullptr;
    }
  }

  StatusOr<std::optional<CompilationResult>> CompileNoCache(
      HloComputation* hlo_computation, const DeviceConfig& device_config,
      const AutotuneResult::TritonGemmKey& autotune_config) {
    llvm::LLVMContext llvm_ctx;
    std::vector<uint64_t> arg_sizes;
    for (HloInstruction* param : hlo_computation->parameter_instructions()) {
      arg_sizes.push_back(ShapeUtil::ByteSizeOf(param->shape()));
    }
    CHECK(!hlo_computation->root_instruction()->shape().IsTuple());
    arg_sizes.push_back(
        ShapeUtil::ByteSizeOf(hlo_computation->root_instruction()->shape()));

    const HloModuleConfig& module_config = hlo_computation->parent()->config();
    const se::CudaComputeCapability& cc =
        device_config.stream_exec->GetDeviceDescription()
            .cuda_compute_capability();

    uint64_t start_compilation_nanos = tsl::Env::Default()->NowNanos();

    llvm::Module module("module", llvm_ctx);
    // TODO(b/266210099): Duplication against nvptx_compiler.cc
    module.setTargetTriple(nvptx::TargetTriple());
    module.setDataLayout(nvptx::DataLayout());

    const GpuDeviceInfo dev_info = GetGpuDeviceInfo(device_config.stream_exec);
    std::optional<LaunchDimensions> launch_dimensions =
        TritonWrapper(triton_fn_name_, hlo_computation, cc, dev_info,
                      autotune_config, &module, &MatMul);
    if (!launch_dimensions.has_value()) {
      // Out of shmem budget.
      return {std::nullopt};
    }

    llvm::IRBuilder<> b_(llvm_ctx);
    llvm::Function* kernel_prototype = BuildKernelPrototype(
        triton_fn_name_.c_str(), arg_sizes, b_, module, llvm_ctx);

    // Move function body into kernel prototype.
    // Device kernel we are building.
    llvm::Function* prototype_func = b_.GetInsertBlock()->getParent();

    // Function as created by Triton.
    llvm::Function* implementation_fn = module.getFunction(triton_fn_name_);
    QCHECK(implementation_fn);
    prototype_func->splice(prototype_func->end(), implementation_fn);
    for (const auto& [arg, prototype_arg] :
         llvm::zip_first(implementation_fn->args(), kernel_prototype->args())) {
      arg.replaceAllUsesWith(&prototype_arg);
    }
    implementation_fn->eraseFromParent();

    // Replace pre-existing return with unconditional branch to next block.
    llvm::Instruction* terminator =
        prototype_func->getEntryBlock().getTerminator();
    llvm::BranchInst::Create(&*std::next(prototype_func->begin()), terminator);
    terminator->eraseFromParent();

    LogAndVerify(&module);

    TF_ASSIGN_OR_RETURN(
        std::string ptx,
        nvptx::CompileToPtx(&module,
                            device_config.stream_exec->GetDeviceDescription()
                                .cuda_compute_capability(),
                            module_config));

    se::GpuAsmOpts ptxas_config =
        PtxOptsFromDebugOptions(module_config.debug_options());
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> cubin,
        se::CompileGpuAsm(device_config.stream_exec->device_ordinal(),
                          ptx.c_str(), ptxas_config));

    uint64_t end_compilation_nanos = tsl::Env::Default()->NowNanos();
    absl::Duration compilation_time_span =
        absl::Nanoseconds(end_compilation_nanos - start_compilation_nanos);
    VLOG(1) << "Compilation took: " << compilation_time_span;

    return std::make_optional(
        CompilationResult{ptx, cubin, *launch_dimensions});
  }

  // TODO(b/266210099): Refactor, do not duplicate code vs. ir_emitter_unnested.
  // Builds a prototype for a function with given arguments.
  llvm::Function* BuildKernelPrototype(const char* kernel_name,
                                       std::vector<uint64_t> arg_sizes,
                                       llvm::IRBuilder<>& b_,
                                       llvm::Module& module,
                                       llvm::LLVMContext& llvm_ctx) {
    llvm::FunctionType* kernel_type = llvm::FunctionType::get(
        /*Result=*/llvm::Type::getVoidTy(llvm_ctx),
        std::vector<llvm::Type*>(arg_sizes.size(), b_.getInt8PtrTy()),
        /*isVarArg=*/false);
    llvm::Function* kernel = llvm::Function::Create(
        kernel_type, llvm::GlobalValue::ExternalLinkage, kernel_name, module);

    // Add dereferenceable and alignment information to each of the kernel's
    // parameters.
    auto arg_it = kernel->arg_begin();
    for (size_t arg_no = 0; arg_no < arg_sizes.size(); ++arg_no) {
      uint64_t arg_size = arg_sizes[arg_no];
      llvm::Argument& fn_arg = *arg_it;
      ++arg_it;

      kernel->addDereferenceableParamAttr(arg_no, arg_size);
      kernel->addParamAttr(
          arg_no,
          llvm::Attribute::get(llvm_ctx, llvm::Attribute::Alignment, 128));
      fn_arg.setName(absl::StrCat("alloc", arg_no));
    }

    AnnotateFunctionAsGpuKernel(&module, kernel, &b_);
    // Update the insert point to the entry basic block.
    llvm::BasicBlock* entry_bb =
        llvm::BasicBlock::Create(llvm_ctx, /*Name=*/"entry", /*Parent=*/kernel);

    // Emit a "return void" at entry_bb's end, and set the insert point before
    // that return instruction.
    b_.SetInsertPoint(llvm::ReturnInst::Create(llvm_ctx, entry_bb));
    return kernel;
  }

  AutotuningConfig config_;
  int num_extra_threads_;

  std::string triton_fn_name_ = "matmul_autotune";
};

}  // anonymous namespace

StatusOr<bool> TritonAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return TritonAutotunerVisitor{config_, num_extra_threads_}.RunOnModule(
      module, execution_threads);
}

Status TritonAutotuner::WriteAutotuneResults(AutotuneResults* results) {
  // TODO(anlunx): Remove duplication with gpu_conv_algorithm_picker.
  absl::MutexLock lock(&autotune_cache_mu);

  for (const auto& [k, result] : autotune_cache) {
    const auto& [model_str, hlo] = k;
    auto& entry = *results->add_dots();
    entry.set_device(model_str);
    entry.set_hlo(hlo);
    *entry.mutable_result() = result;
  }

  // Sort the results so that they're deterministic.
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

Status TritonAutotuner::LoadAutotuneResults(const AutotuneResults& results) {
  absl::MutexLock lock(&autotune_cache_mu);
  for (const auto& result : results.convs()) {
    autotune_cache[std::make_tuple(result.device(), result.hlo())] =
        result.result();
  }
  return OkStatus();
}

void TritonAutotuner::ClearAutotuneResults() {
  absl::MutexLock lock(&autotune_cache_mu);
  autotune_cache.clear();
}

}  // namespace gpu
}  // namespace xla
