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
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/IR/LLVMContext.h"
#include "tensorflow/compiler/xla/autotune_results.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/float_normalization.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/compile_module_to_llvm_ir.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_float_support.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_serializable_autotuner.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/compiler/xla/tools/hlo_extractor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/blocking_counter.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/threadpool.h"
#include "tensorflow/tsl/protobuf/autotuning.pb.h"
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

// Maximum number of independent thread blocks along K dimension.
// The actual value is split_k in the tiling configuration
// and has to be <= kMaxSplitK.
// Requires a separate temporary output buffer for each block, so should
// be limited reasonably. The current maximum value was chosen based on
// some matmul configurations benchmarked so far and can be increased further.
constexpr int kMaxSplitK = 16;

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
          GemmKey(32, 32, 256, 1, 1, 4),   GemmKey(64, 32, 32, 16, 1, 4),
          GemmKey(32, 64, 64, 4, 1, 4),    GemmKey(128, 128, 64, 4, 1, 4)};
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
  std::vector<std::string> kernel_names;
  std::vector<LaunchDimensions> launch_dimensions;
};

using CompilationKey = std::pair<std::string, TritonTilingWrapper>;
static absl::Mutex compilation_cache_mutex(absl::kConstInit);
static auto& compilation_cache ABSL_GUARDED_BY(compilation_cache_mutex) =
    *new absl::node_hash_map<CompilationKey,
                             std::optional<CompilationResult>>();

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
    se::RedzoneAllocator& allocator, int64_t byte_size,
    PrimitiveType element_type, const AutotuneConfig& config,
    int64_t& rng_state) {
  TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase buffer,
                      allocator.AllocateBytes(byte_size));
  if (config.should_init_buffers()) {
    InitializeBuffer(allocator.stream(), element_type, &rng_state, buffer);
  }
  return buffer;
}

class TritonAutotunerVisitor : public DfsHloRewriteVisitor {
 public:
  TritonAutotunerVisitor(const AutotuningConfig& config,
                         tsl::thread::ThreadPool* thread_pool)
      : config_(config), thread_pool_(thread_pool) {}

  Status HandleFusion(HloInstruction* hlo) override {
    if (hlo->raw_backend_config_string() != kTritonGemmBackendConfig) {
      return OkStatus();
    }

    TF_ASSIGN_OR_RETURN(AutotuneResult autotune_result,
                        AutotuneMatmul(hlo->called_computations()[0]));

    TF_RET_CHECK(autotune_result.has_triton());
    AutotuneResult::TritonGemmKey tiling = autotune_result.triton();

    if (tiling.split_k() > 1) {
      TF_RETURN_IF_ERROR(MakeDotSplitKBatch(hlo, tiling));
    }

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
    TF_ASSIGN_OR_RETURN(
        se::Stream* const stream,
        device_config.allocator->GetStream(stream_exec->device_ordinal()));

    const DebugOptions debug_opts = fusion->parent()->config().debug_options();
    const AutotuneConfig autotune_cfg = GetConfig(debug_opts);

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

    const std::vector<AutotuneResult::TritonGemmKey> configurations =
        GetPossibleMatmulAutotuneConfigs();

    // Pre-compile all versions first using the thread pool.
    if (thread_pool_) {
      tsl::BlockingCounter counter(configurations.size());
      for (const AutotuneResult::TritonGemmKey& conf : configurations) {
        thread_pool_->Schedule([&] {
          StatusOr<CompilationResult*> res =
              Compile(fusion, device_config, conf);
          if (!res.ok()) {
            LOG(ERROR) << "Failure: " << res.status().ToString();
          }
          counter.DecrementCount();
        });
      }
      counter.Wait();
    }

    std::vector<se::DeviceMemoryBase> inputs;
    int64_t rng_state = 0;
    for (const HloInstruction* param : fusion->parameter_instructions()) {
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryBase param_buffer,
          CreateBuffer(rz_allocator, ShapeUtil::ByteSizeOf(param->shape()),
                       param->shape().element_type(), autotune_cfg, rng_state));
      inputs.push_back(param_buffer);
    }

    // The intermediate one does not need to be initialized.
    TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase intermediate_buffer,
                        rz_allocator.AllocateBytes(
                            ShapeUtil::ByteSizeOf(root->shape()) * kMaxSplitK));

    TF_ASSIGN_OR_RETURN(
        se::DeviceMemoryBase output_buffer,
        CreateBuffer(rz_allocator, ShapeUtil::ByteSizeOf(root->shape()),
                     root->shape().element_type(), autotune_cfg, rng_state));

    for (const AutotuneResult::TritonGemmKey& conf : configurations) {
      VLOG(1) << "Trying triton tiling: " << conf.DebugString();

      AutotuneResult res;
      *res.mutable_triton() = conf;

      TF_ASSIGN_OR_RETURN(
          std::optional<absl::Duration> duration,
          RunMatmulWithConfig(fusion, conf, device_config, stream, inputs,
                              intermediate_buffer, output_buffer));

      if (!duration) {
        VLOG(1) << "Skipping this tiling.";
        continue;
      }

      VLOG(1) << "Running the kernel took: " << *duration;
      *res.mutable_run_time() = tsl::proto_utils::ToDurationProto(*duration);

      if (autotune_cfg.should_check_correctness()) {
        TF_ASSIGN_OR_RETURN(
            se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
            rz_allocator.CheckRedzones());
        if (!rz_check_status.ok()) {
          LOG(ERROR) << "Red zone modified";
          res.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
          *res.mutable_failure()->mutable_msg() =
              rz_check_status.RedzoneFailureMsg();
          CHECK(!autotune_cfg.should_crash_on_check_failure);
          continue;
        }

        if (!reference_tiling) {
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
      }
      results.push_back(res);

      if (autotune_cfg.should_reinit_output_buffer()) {
        InitializeBuffer(stream, root->shape().element_type(), &rng_state,
                         output_buffer);
      }
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
      absl::Span<se::DeviceMemoryBase const> input_buffers,
      se::DeviceMemoryBase intermediate_buffer,
      se::DeviceMemoryBase output_buffer) {
    TF_ASSIGN_OR_RETURN(
        CompilationResult * res,
        Compile(hlo_computation, device_config, autotune_config));
    if (!res) {
      // Out of shared memory budget.
      return {std::nullopt};
    }

    // Don't run autotuning concurrently on the same GPU.
    absl::MutexLock gpu_lock(&GetGpuMutex(stream->parent()));

    auto& [ptx, cubin, kernel_names, launch_dimensions] = *res;
    const bool have_reduction = kernel_names.size() > 1;

    std::vector<se::DeviceMemoryBase> matmul_args;
    for (const se::DeviceMemoryBase& buffer : input_buffers) {
      matmul_args.push_back(buffer);
    }
    matmul_args.push_back(have_reduction ? intermediate_buffer : output_buffer);

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<se::KernelBase> matmul_kernel,
        CreateKernel(kernel_names[0], matmul_args.size(), ptx, cubin,
                     stream->parent(), launch_dimensions[0].SharedMemBytes()));
    std::unique_ptr<se::KernelBase> reduce_kernel;
    std::vector<se::DeviceMemoryBase> reduce_args = {intermediate_buffer,
                                                     output_buffer};
    if (have_reduction) {
      TF_ASSIGN_OR_RETURN(reduce_kernel,
                          CreateKernel(kernel_names[1], reduce_args.size(), ptx,
                                       cubin, stream->parent(),
                                       launch_dimensions[1].SharedMemBytes()));
    }

    se::gpu::GpuExecutor* cuda_executor =
        dynamic_cast<se::gpu::GpuExecutor*>(stream->parent()->implementation());
    std::unique_ptr<se::gpu::GpuTimer, se::gpu::GpuTimerDeleter> timer(
        new se::gpu::GpuTimer(cuda_executor));

    // Warmup: in and out buffers are reused while probing different configs, so
    // GPU caches should be in some comparable states during measurements.
    TF_RETURN_IF_ERROR(ExecuteKernelOnStream(*matmul_kernel, matmul_args,
                                             launch_dimensions[0], stream));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    if (!timer->Init() || !timer->Start(se::gpu::AsGpuStream(stream))) {
      return Status(absl::StatusCode::kInternal, "Failed to start timer");
    }
    TF_RETURN_IF_ERROR(ExecuteKernelOnStream(*matmul_kernel, matmul_args,
                                             launch_dimensions[0], stream));
    if (have_reduction) {
      TF_RETURN_IF_ERROR(ExecuteKernelOnStream(*reduce_kernel, reduce_args,
                                               launch_dimensions[1], stream));
    }
    if (!timer->Stop(se::gpu::AsGpuStream(stream))) {
      return Status(absl::StatusCode::kInternal, "Failed to stop timer");
    }
    return std::make_optional(absl::Nanoseconds(timer->Nanoseconds()));
  }

  // Compile a given computation with a given autotuning config, utilizing
  // computation cache. Returns a raw pointer into the map to avoid copying the
  // values. Returning `nullptr` means that the kernel could not be generated.
  StatusOr<CompilationResult*> Compile(
      HloComputation* hlo_computation, const DeviceConfig& device_config,
      const AutotuneResult::TritonGemmKey& autotune_config) {
    CompilationKey key = std::make_pair(ToCanonicalString(hlo_computation),
                                        TritonTilingWrapper{autotune_config});

    // TODO(b/266210099): Avoid duplication.
    {
      absl::MutexLock lock(&compilation_cache_mutex);
      auto it = compilation_cache.find(key);
      if (it != compilation_cache.end()) {
        VLOG(4) << "Compilation cache hit";
        std::optional<CompilationResult>& res = it->second;
        if (res.has_value()) {
          return &*res;
        }
        return nullptr;
      }
    }

    TF_ASSIGN_OR_RETURN(
        std::optional<CompilationResult> res,
        CompileNoCache(*hlo_computation, device_config, autotune_config));
    {
      absl::MutexLock lock(&compilation_cache_mutex);
      auto [it2, inserted] = compilation_cache.emplace(key, res);
      std::optional<CompilationResult>& res_inserted = it2->second;
      if (res_inserted.has_value()) {
        return &*res_inserted;
      }
      return nullptr;
    }
  }

  StatusOr<std::optional<CompilationResult>> CompileNoCache(
      const HloComputation& original_computation,
      const DeviceConfig& device_config,
      const AutotuneResult::TritonGemmKey& autotune_config) {
    uint64_t start_compilation_nanos = tsl::Env::Default()->NowNanos();

    const se::DeviceDescription& device_description =
        device_config.stream_exec->GetDeviceDescription();
    const GpuDeviceInfo gpu_device_info =
        GetGpuDeviceInfo(device_config.stream_exec);

    std::unique_ptr<HloModule> new_hlo_module =
        ExtractModule(original_computation.FusionInstruction(), /*height=*/0);
    new_hlo_module->set_config(original_computation.parent()->config());
    DebugOptions options =
        original_computation.parent()->config().debug_options();
    // Require thunks because so far we are relying on them for execution here.
    // TODO(b/277066525): stop using thunks.
    options.set_xla_gpu_enable_xla_runtime_executable(false);
    // Avoid dumping compilation steps of every autotuning variant.
    options.set_xla_dump_to("");
    options.set_xla_gpu_dump_llvmir(false);
    new_hlo_module->config().set_debug_options(options);
    HloComputation* entry_computation = new_hlo_module->entry_computation();
    HloInstruction* cloned_dot_fusion = entry_computation->root_instruction();
    TF_RETURN_IF_ERROR(cloned_dot_fusion->set_backend_config(autotune_config));
    if (autotune_config.split_k() > 1) {
      if (!MakeDotSplitKBatch(cloned_dot_fusion, autotune_config).ok()) {
        return {std::nullopt};
      }
      GpuFloatSupport bf16_support(BF16);
      FloatNormalization float_normalization(&bf16_support);
      TF_RETURN_IF_ERROR(
          float_normalization.Run(new_hlo_module.get()).status());
      GpuInstructionFusion instruction_fusion(/*may_duplicate=*/false,
                                              gpu_device_info);
      TF_RETURN_IF_ERROR(instruction_fusion.Run(new_hlo_module.get()).status());
      HloInstruction* root = entry_computation->root_instruction();
      // If the instruction fusion pass above skipped the reduction, turn it
      // into a fusion for a universal set of arguments for execution.
      if (root->opcode() == HloOpcode::kReduce) {
        HloInstruction* fusion_instruction =
            entry_computation->AddInstruction(HloInstruction::CreateFusion(
                root->shape(), ChooseFusionKind(*root->operand(0), *root),
                root));
        HloInstruction* init_value = root->mutable_operand(1);
        TF_CHECK_OK(
            entry_computation->ReplaceInstruction(root, fusion_instruction));
        fusion_instruction->FuseInstruction(init_value);
        TF_CHECK_OK(entry_computation->RemoveInstruction(init_value));
      }
    }

    llvm::LLVMContext llvm_context;
    CompileModuleResults compile_module_results;
    Status compilation_status = xla::gpu::CompileModuleToLlvmIrImpl(
        new_hlo_module.get(), &llvm_context,
        /*target_triple=*/nvptx::TargetTriple(),
        /*data_layout=*/nvptx::DataLayout(),
        /*platform_name=*/device_config.stream_exec->platform()->Name(),
        /*platform_id=*/device_config.stream_exec->platform()->id(),
        gpu_device_info, device_description.cuda_compute_capability(),
        device_description.rocm_compute_capability(),
        DummyCanShareBufferFunction,
        /*pointer_size=*/8, &compile_module_results);
    if (!compilation_status.ok()) {
      VLOG(2) << "Compilation of autotuning variant failed: "
              << compilation_status;
      return {std::nullopt};
    }

    std::vector<std::string> kernel_names;
    std::vector<LaunchDimensions> launch_dimensions;
    CHECK(std::holds_alternative<GpuExecutable::OwnedThunkSequence>(
        compile_module_results.executable));
    const ThunkSequence& thunk_sequence =
        *std::get<GpuExecutable::OwnedThunkSequence>(
            compile_module_results.executable);
    // Expect at maximum two kernels: matmul and an optional reduction.
    CHECK_LE(thunk_sequence.size(), 2);
    for (const std::unique_ptr<Thunk>& thunk : thunk_sequence) {
      CHECK_EQ(thunk->kind(), Thunk::kKernel);
      KernelThunk* kernel_thunk = static_cast<KernelThunk*>(thunk.get());
      kernel_names.push_back(kernel_thunk->kernel_name());
      launch_dimensions.push_back(kernel_thunk->launch_dimensions());
    }

    TF_ASSIGN_OR_RETURN(
        std::string ptx,
        nvptx::CompileToPtx(compile_module_results.llvm_module.get(),
                            device_description.cuda_compute_capability(),
                            new_hlo_module->config()));

    se::GpuAsmOpts ptxas_config =
        PtxOptsFromDebugOptions(new_hlo_module->config().debug_options());
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> cubin,
        se::CompileGpuAsm(device_config.stream_exec->device_ordinal(),
                          ptx.c_str(), ptxas_config));

    uint64_t end_compilation_nanos = tsl::Env::Default()->NowNanos();
    absl::Duration compilation_time_span =
        absl::Nanoseconds(end_compilation_nanos - start_compilation_nanos);
    VLOG(1) << "Compilation took: " << compilation_time_span;

    return std::make_optional(
        CompilationResult{ptx, cubin, kernel_names, launch_dimensions});
  }

  AutotuningConfig config_;
  tsl::thread::ThreadPool* thread_pool_;
};

}  // anonymous namespace

StatusOr<bool> TritonAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return TritonAutotunerVisitor{config_, thread_pool_}.RunOnModule(
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

void TritonAutotuner::ClearCompilationCache() {
  absl::MutexLock lock(&compilation_cache_mutex);
  compilation_cache.clear();
}

}  // namespace gpu
}  // namespace xla
