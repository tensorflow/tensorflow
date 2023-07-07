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
#include <array>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/const_init.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/node_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "llvm/IR/LLVMContext.h"
#include "tensorflow/compiler/xla/autotune_results.pb.h"
#include "tensorflow/compiler/xla/autotuning.pb.h"
#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_clone_context.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/backend.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/float_normalization.h"
#include "tensorflow/compiler/xla/service/gpu/autotuner_util.h"
#include "tensorflow/compiler/xla/service/gpu/bitcast_remover.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_comparator.h"
#include "tensorflow/compiler/xla/service/gpu/compile_module_to_llvm_ir.h"
#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_asm_opts_util.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_device_info.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_float_support.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_fusible.h"
#include "tensorflow/compiler/xla/service/gpu/instruction_fusion.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_thunk.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"
#include "tensorflow/compiler/xla/service/gpu/target_constants.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/hlo_module_config.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/service/platform_util.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/stream_executor/device_description.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/asm_compiler.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_asm_opts.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_stream.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/gpu_timer.h"
#include "tensorflow/compiler/xla/stream_executor/gpu/redzone_allocator.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla.pb.h"
#include "tensorflow/tsl/platform/blocking_counter.h"
#include "tensorflow/tsl/platform/env.h"
#include "tensorflow/tsl/platform/errors.h"
#include "tensorflow/tsl/platform/status.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/platform/threadpool.h"
#include "tensorflow/tsl/util/proto/proto_utils.h"

namespace xla {
namespace gpu {

namespace {

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

struct TritonTilingWrapper {
  const AutotuneResult::TritonGemmKey key;

  template <typename H>
  friend H AbslHashValue(H h, const TritonTilingWrapper& w) {
    return H::combine(std::move(h), w.key.SerializeAsString());
  }

  bool operator==(const TritonTilingWrapper& w) const {
    return key.SerializeAsString() == w.key.SerializeAsString();
  }
};

struct CompilationResult {
  std::string ptx;
  std::vector<uint8_t> cubin;
  std::vector<std::string> kernel_names;
  std::vector<LaunchDimensions> launch_dimensions;
};

// Using the "standard" AutotuneCacheKey in CompilationKey, which encompasses
// both the device type and the code of the HLO. We need this because
// TritonAutotuner may be called with different device types, and a binary
// compiled for one device type may not run on another.
using CompilationKey = std::pair<AutotuneCacheKey, TritonTilingWrapper>;
static absl::Mutex compilation_cache_mutex(absl::kConstInit);
static auto& compilation_cache ABSL_GUARDED_BY(compilation_cache_mutex) =
    *new absl::node_hash_map<CompilationKey,
                             std::optional<CompilationResult>>();

// Here we have a cache for the executables that we use to generate the
// reference values for the GEMM fusions. We only compile the same HLO once.
static absl::Mutex non_triton_executable_cache_mutex(absl::kConstInit);
// The key is the "standard" AutotuneCacheKey, which encompasses both the device
// type and the code of the HLO. We need this because TritonAutotuner may be
// called with different device types, and an executable compiled for one device
// type may not run on another.
static auto& ABSL_GUARDED_BY(
    non_triton_executable_cache_mutex) non_triton_executable_cache =
    *new absl::node_hash_map<AutotuneCacheKey, std::unique_ptr<Executable>>();

// This is like HloRunner, but allows using a custom stream and allocator for
// all operations.
class CustomHloRunner {
 public:
  // Create a CustomHloRunner.
  static StatusOr<std::unique_ptr<CustomHloRunner>> Create(
      se::Stream& stream, se::DeviceMemoryAllocator& allocator);

  // Compile the module, running the HLO passes as well.
  StatusOr<std::unique_ptr<Executable>> CompileRunningHloPasses(
      std::unique_ptr<HloModule> module);

  // Execute the executable using the arguments.
  StatusOr<ExecutionOutput> Execute(Executable& executable,
                                    std::vector<ExecutionInput> arguments);

 private:
  CustomHloRunner(std::unique_ptr<Backend> backend,
                  se::StreamExecutor& stream_executor, se::Stream& stream,
                  se::DeviceMemoryAllocator& allocator);

  std::unique_ptr<Backend> backend_;
  se::StreamExecutor& stream_executor_;
  se::Stream& stream_;
  se::DeviceMemoryAllocator& allocator_;
};

CustomHloRunner::CustomHloRunner(std::unique_ptr<Backend> backend,
                                 se::StreamExecutor& stream_executor,
                                 se::Stream& stream,
                                 se::DeviceMemoryAllocator& allocator)
    : backend_(std::move(backend)),
      stream_executor_(stream_executor),
      stream_(stream),
      allocator_(allocator) {}

StatusOr<std::unique_ptr<CustomHloRunner>> CustomHloRunner::Create(
    se::Stream& stream, se::DeviceMemoryAllocator& allocator) {
  se::StreamExecutor& stream_executor = *stream.parent();

  TF_ASSIGN_OR_RETURN(
      se::Platform * platform,
      PlatformUtil::GetPlatform(stream_executor.platform()->Name()));

  BackendOptions backend_options;
  backend_options.set_platform(platform);
  TF_ASSIGN_OR_RETURN(std::unique_ptr<Backend> backend,
                      Backend::CreateBackend(backend_options));

  return absl::WrapUnique(new CustomHloRunner(
      std::move(backend), stream_executor, stream, allocator));
}

StatusOr<std::unique_ptr<Executable>> CustomHloRunner::CompileRunningHloPasses(
    std::unique_ptr<HloModule> module) {
  auto module_group = std::make_unique<HloModuleGroup>(std::move(module));
  TF_ASSIGN_OR_RETURN(
      auto executables,
      backend_->compiler()->Compile(std::move(module_group),
                                    {{&stream_executor_}}, &allocator_));
  return std::move(executables[0]);
}

StatusOr<ExecutionOutput> CustomHloRunner::Execute(
    Executable& executable, std::vector<ExecutionInput> arguments) {
  ExecutableRunOptions run_options;
  run_options.set_device_ordinal(stream_executor_.device_ordinal());
  run_options.set_stream(&stream_);
  run_options.set_allocator(&allocator_);
  run_options.set_intra_op_thread_pool(
      backend_->eigen_intra_op_thread_pool_device());
  run_options.set_run_id(RunId());

  ServiceExecutableRunOptions service_run_options(
      run_options, backend_->StreamBorrowerWithPriority());

  TF_ASSIGN_OR_RETURN(ExecutionOutput output,
                      executable.ExecuteOnStreamWrapper(&service_run_options,
                                                        std::move(arguments)));

  return std::move(output);
}

class TritonAutotunerVisitor : public DfsHloRewriteVisitor {
 public:
  TritonAutotunerVisitor(const AutotuneConfig& config,
                         tsl::thread::ThreadPool* thread_pool)
      : config_(config), thread_pool_(thread_pool) {}

  Status HandleFusion(HloInstruction* hlo) override {
    TF_ASSIGN_OR_RETURN(auto backend_config,
                        hlo->backend_config<FusionBackendConfig>());
    if (backend_config.kind() != kTritonGemmFusionKind) {
      return OkStatus();
    }

    VLOG(1) << "Tuning " << hlo->ToString();
    TF_ASSIGN_OR_RETURN(AutotuneResult autotune_result,
                        AutotunerUtil::Autotune(hlo, config_, [&] {
                          return AutotuneMatmulNoCache(
                              hlo,
                              AutotuneCacheKey(config_.GetModelStr(), *hlo));
                        }));
    VLOG(1) << "Result: " << autotune_result.ShortDebugString();

    TF_RET_CHECK(autotune_result.has_triton());
    AutotuneResult::TritonGemmKey tiling = autotune_result.triton();

    if (tiling.split_k() > 1) {
      TF_RETURN_IF_ERROR(MakeDotSplitKBatch(hlo, tiling));
    }

    *backend_config.mutable_triton_gemm_config() = tiling;
    TF_RETURN_IF_ERROR(hlo->set_backend_config(backend_config));
    MarkAsChanged();
    return OkStatus();
  }

 private:
  // Autotunes a matmul without using the autotuning cache.
  //
  // `cache_key`: The cache key corresponding to the code of the fusion and the
  // device type. Passing it to avoid recalculating it everywhere it's needed.
  StatusOr<AutotuneResult> AutotuneMatmulNoCache(
      const HloInstruction* instr, const AutotuneCacheKey& cache_key) {
    const HloComputation& fusion = *instr->called_computations()[0];
    se::StreamExecutor* stream_exec = config_.GetExecutor();
    if (!stream_exec->SynchronizeAllActivity()) {
      return InternalError("Failed to synchronize GPU for autotuning.");
    }
    se::DeviceMemoryAllocator* allocator = config_.GetAllocator();
    if (allocator == nullptr) {
      allocator = stream_exec->GetAllocator();
    }

    HloInstruction* root = fusion.root_instruction();
    TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                        allocator->GetStream(stream_exec->device_ordinal()));

    const DebugOptions debug_opts = fusion.parent()->config().debug_options();

    std::vector<AutotuneResult> results;
    // This allocator is used for input and reference buffers that are
    // common for all configurations.
    se::RedzoneAllocator rz_allocator_common(
        stream, allocator, PtxOptsFromDebugOptions(debug_opts),
        /*memory_limit=*/std::numeric_limits<int64_t>::max(),
        /*redzone_size=*/config_.should_check_correctness()
            ? se::RedzoneAllocator::kDefaultRedzoneSize
            : 0);

    se::DeviceMemoryBase reference_buffer;
    if (config_.should_check_correctness()) {
      TF_ASSIGN_OR_RETURN(reference_buffer,
                          rz_allocator_common.AllocateBytes(
                              ShapeUtil::ByteSizeOf(root->shape())));
    }

    BufferComparator comparator(root->shape(), fusion.parent()->config());

    const std::vector<AutotuneResult::TritonGemmKey> configurations =
        GetPossibleMatmulAutotuneConfigs(
            stream_exec->GetDeviceDescription().cuda_compute_capability(),
            config_.ExhaustiveTilingSearch());

    // Pre-compile all versions first using the thread pool.
    if (thread_pool_) {
      tsl::BlockingCounter counter(configurations.size());
      for (const AutotuneResult::TritonGemmKey& conf : configurations) {
        thread_pool_->Schedule([&] {
          StatusOr<CompilationResult*> res = Compile(fusion, conf, cache_key);
          if (!res.ok()) {
            LOG(ERROR) << "Failure: " << res.status();
          }
          counter.DecrementCount();
        });
      }
      counter.Wait();
    }

    std::vector<se::DeviceMemoryBase> inputs;
    int64_t rng_state = 0;
    for (const HloInstruction* param : fusion.parameter_instructions()) {
      TF_ASSIGN_OR_RETURN(
          se::DeviceMemoryBase param_buffer,
          AutotunerUtil::CreateBuffer(rz_allocator_common, param->shape(),
                                      config_, rng_state));
      inputs.push_back(param_buffer);
    }

    const bool disable_reduced_precision_reduction =
        instr->GetModule()
            ->config()
            .debug_options()
            .xla_gpu_triton_gemm_disable_reduced_precision_reduction();

    PrimitiveType output_type = root->shape().element_type();
    PrimitiveType accumulator_type = output_type == PrimitiveType::F64
                                         ? PrimitiveType::F64
                                         : PrimitiveType::F32;

    if (config_.should_check_correctness()) {
      TF_RETURN_IF_ERROR(RunMatmulWithCublas(fusion, stream, allocator, inputs,
                                             reference_buffer, cache_key));
    }

    for (const AutotuneResult::TritonGemmKey& conf : configurations) {
      VLOG(1) << "Trying triton tiling: " << conf.ShortDebugString();

      // This allocator is used for intermediate buffers and output that are
      // unique for each configuration.
      se::RedzoneAllocator rz_allocator(
          stream, allocator, PtxOptsFromDebugOptions(debug_opts),
          /*memory_limit=*/std::numeric_limits<int64_t>::max(),
          /*redzone_size=*/config_.should_check_correctness()
              ? se::RedzoneAllocator::kDefaultRedzoneSize
              : 0);

      AutotuneResult res;
      *res.mutable_triton() = conf;

      // Failing on allocating an intermediate buffer is OK because other
      // less memory-hungry configurations do not need it at all.
      se::DeviceMemoryBase intermediate_buffer;
      if (conf.split_k() > 1) {
        // The intermediate one does not need to be initialized.
        StatusOr<se::DeviceMemoryBase> result = rz_allocator.AllocateBytes(
            ShapeUtil::ElementsIn(root->shape()) *
            ShapeUtil::ByteSizeOfPrimitiveType(
                disable_reduced_precision_reduction ? accumulator_type
                                                    : output_type) *
            conf.split_k());
        if (!result.ok()) {
          // The allocator will log a warning.
          // Proceed to trying next configuration.
          continue;
        }
        intermediate_buffer = *result;
      }

      TF_ASSIGN_OR_RETURN(se::DeviceMemoryBase output_buffer,
                          AutotunerUtil::CreateBuffer(
                              rz_allocator, root->shape(), config_, rng_state));

      TF_ASSIGN_OR_RETURN(
          std::optional<absl::Duration> duration,
          RunMatmulWithConfig(fusion, conf, stream, inputs, intermediate_buffer,
                              output_buffer, cache_key));

      if (!duration) {
        VLOG(1) << "Skipping this tiling.";
        continue;
      }

      VLOG(1) << "Running the kernel took: " << *duration;
      *res.mutable_run_time() = tsl::proto_utils::ToDurationProto(*duration);

      if (config_.should_check_correctness()) {
        TF_ASSIGN_OR_RETURN(
            se::RedzoneAllocator::RedzoneCheckStatus rz_check_status,
            rz_allocator.CheckRedzones());
        if (!rz_check_status.ok()) {
          LOG(ERROR) << "Red zone modified";
          res.mutable_failure()->set_kind(AutotuneResult::REDZONE_MODIFIED);
          *res.mutable_failure()->mutable_msg() =
              rz_check_status.RedzoneFailureMsg();
          CHECK(!config_.should_crash_on_check_failure());
          continue;
        }

        TF_ASSIGN_OR_RETURN(
            bool outputs_match,
            comparator.CompareEqual(stream, output_buffer, reference_buffer));
        if (!outputs_match) {
          LOG(ERROR) << "Results do not match the reference. "
                     << "This is likely a bug/unexpected loss of precision.";
          CHECK(!config_.should_crash_on_check_failure());
          // WRONG_RESULT is not taken seriously by PickBestResult(), so
          // use DISQUALIFIED.
          res.mutable_failure()->set_kind(AutotuneResult::DISQUALIFIED);
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
  //
  // `cache_key`: The cache key corresponding to the code of the fusion and the
  // device type. Passing it to avoid recalculating it everywhere it's needed.
  StatusOr<std::optional<absl::Duration>> RunMatmulWithConfig(
      const HloComputation& hlo_computation,
      const AutotuneResult::TritonGemmKey& autotune_config, se::Stream* stream,
      absl::Span<se::DeviceMemoryBase const> input_buffers,
      se::DeviceMemoryBase intermediate_buffer,
      se::DeviceMemoryBase output_buffer, const AutotuneCacheKey& cache_key) {
    TF_ASSIGN_OR_RETURN(CompilationResult * res,
                        Compile(hlo_computation, autotune_config, cache_key));
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

    // Warmup: in and out buffers are reused while probing different configs, so
    // GPU caches should be in some comparable states during measurements.
    TF_RETURN_IF_ERROR(ExecuteKernelOnStream(*matmul_kernel, matmul_args,
                                             launch_dimensions[0], stream));
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

    TF_ASSIGN_OR_RETURN(
        auto timer, se::gpu::GpuTimer::Create(se::gpu::AsGpuStream(stream)));
    TF_RETURN_IF_ERROR(ExecuteKernelOnStream(*matmul_kernel, matmul_args,
                                             launch_dimensions[0], stream));
    if (have_reduction) {
      TF_RETURN_IF_ERROR(ExecuteKernelOnStream(*reduce_kernel, reduce_args,
                                               launch_dimensions[1], stream));
    }
    TF_ASSIGN_OR_RETURN(absl::Duration timer_duration,
                        timer.GetElapsedDuration());
    return std::make_optional(timer_duration);
  }

  StatusOr<std::unique_ptr<Executable>> CompileMatmulWithCublas(
      const HloComputation& original_computation,
      CustomHloRunner& custom_hlo_runner) {
    // Create an unoptimized HLO module which does the same as
    // `original_computation`, but with CuBLAS.
    std::unique_ptr<HloModule> module =
        ExtractComputationIntoNewModule(original_computation);
    VLOG(3) << "Extracted module: " << module->ToString();
    BitcastRemover bitcast_remover;
    TF_RETURN_IF_ERROR(bitcast_remover.Run(module.get()).status());
    VLOG(3) << "Deoptimized module: " << module->ToString();

    DebugOptions options =
        original_computation.parent()->config().debug_options();
    // Use cuBLAS gemm.
    options.set_xla_gpu_enable_triton_gemm(false);
    // Avoid any autotuning: the result is only used to check numerics,
    // use default algorithms to save compilation time and memory.
    options.set_xla_gpu_autotune_level(0);
    // Avoid dumping compilation steps.
    options.set_xla_dump_to("");
    options.set_xla_gpu_dump_autotune_results_to("");
    options.set_xla_gpu_load_autotune_results_from("");
    options.set_xla_gpu_dump_llvmir(false);
    // Avoid using another thread pool.
    options.set_xla_gpu_force_compilation_parallelism(1);
    module->config().set_debug_options(options);

    return custom_hlo_runner.CompileRunningHloPasses(std::move(module));
  }

  // Runs a matmul fusion without Triton - with cuBLAS, to generate a reference
  // output.
  //
  // `cache_key`: The cache key corresponding to the code of the fusion and the
  // device type. Passing it to avoid recalculating it everywhere it's needed.
  Status RunMatmulWithCublas(
      const HloComputation& original_computation, se::Stream* stream,
      se::DeviceMemoryAllocator* allocator,
      absl::Span<se::DeviceMemoryBase const> input_buffers,
      se::DeviceMemoryBase output_buffer, const AutotuneCacheKey& cache_key) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<CustomHloRunner> custom_hlo_runner,
                        CustomHloRunner::Create(*stream, *allocator));

    Executable* executable = nullptr;
    {
      absl::MutexLock lock(&non_triton_executable_cache_mutex);

      auto it = non_triton_executable_cache.find(cache_key);
      if (it != non_triton_executable_cache.end()) {
        VLOG(4) << "Non-Triton executable cache hit";
        executable = it->second.get();
      }
    }
    if (executable == nullptr) {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<Executable> new_executable,
          CompileMatmulWithCublas(original_computation, *custom_hlo_runner));

      absl::MutexLock lock(&non_triton_executable_cache_mutex);
      auto [it, inserted] = non_triton_executable_cache.emplace(
          cache_key, std::move(new_executable));
      executable = it->second.get();
    }
    TF_RET_CHECK(executable != nullptr);

    // Construct the parameters from the existing input buffers.
    std::vector<ExecutionInput> execution_inputs;
    const HloInstruction::InstructionVector& params =
        original_computation.parameter_instructions();
    TF_RET_CHECK(input_buffers.size() == params.size());
    execution_inputs.reserve(params.size());
    for (int i = 0; i < params.size(); ++i) {
      execution_inputs.emplace_back(params.at(i)->shape());
      // Our executable doesn't have input-output aliasing, so we can pass
      // unowned input buffers.
      execution_inputs.back().SetUnownedBuffer(
          /*index=*/{},
          MaybeOwningDeviceMemory(/*unowned=*/input_buffers.at(i)));
    }

    // Not locking a GPU mutex here, because
    // `GpuExecutable::ExecuteAsyncOnStreamImpl` will do that.
    TF_ASSIGN_OR_RETURN(
        ExecutionOutput execution_output,
        custom_hlo_runner->Execute(*executable, std::move(execution_inputs)));
    ScopedShapedBuffer result = execution_output.ConsumeResult();
    // Copy back the output.
    TF_RET_CHECK(output_buffer.size() == result.root_buffer().size());
    stream->ThenMemcpy(&output_buffer, result.root_buffer(),
                       result.root_buffer().size());
    return OkStatus();
  }

  // Compile a given computation with a given autotuning config, utilizing
  // computation cache. Returns a raw pointer into the map to avoid copying the
  // values. Returning `nullptr` means that the kernel could not be generated.
  //
  // `cache_key`: The cache key corresponding to the code of the fusion and the
  // device type. Passing it to avoid recalculating it everywhere it's needed.
  StatusOr<CompilationResult*> Compile(
      const HloComputation& hlo_computation,
      const AutotuneResult::TritonGemmKey& autotune_config,
      const AutotuneCacheKey& cache_key) {
    CompilationKey key =
        std::make_pair(cache_key, TritonTilingWrapper{autotune_config});

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

    TF_ASSIGN_OR_RETURN(std::optional<CompilationResult> res,
                        CompileNoCache(hlo_computation, autotune_config));
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
      const AutotuneResult::TritonGemmKey& autotune_config) {
    uint64_t start_compilation_nanos = tsl::Env::Default()->NowNanos();

    const se::DeviceDescription& device_description =
        config_.GetExecutor()->GetDeviceDescription();
    const GpuDeviceInfo gpu_device_info =
        GetGpuDeviceInfo(config_.GetExecutor());

    std::unique_ptr<HloModule> new_hlo_module = ExtractInstructionIntoNewModule(
        *original_computation.FusionInstruction());

    // Copy the config from the original computations's module, but use the new
    // entry computation layout. If we extract an instruction into a new
    // module, then its entry computation layout can be different from that of
    // the original module.
    ComputationLayout new_entry_computation_layout =
        new_hlo_module->config().entry_computation_layout();
    new_hlo_module->set_config(original_computation.parent()->config());
    *new_hlo_module->config().mutable_entry_computation_layout() =
        new_entry_computation_layout;

    DebugOptions options =
        original_computation.parent()->config().debug_options();
    // Require thunks because so far we are relying on them for execution here.
    // TODO(b/277066525): stop using thunks.
    options.set_xla_gpu_enable_xla_runtime_executable(false);
    // Avoid dumping compilation steps of every autotuning variant.
    options.set_xla_dump_to("");
    options.set_xla_gpu_dump_autotune_results_to("");
    options.set_xla_gpu_load_autotune_results_from("");
    options.set_xla_gpu_dump_llvmir(false);
    // Avoid using another thread pool for PTX compilation - there are maximum
    // two functions to compile here.
    options.set_xla_gpu_force_compilation_parallelism(1);
    new_hlo_module->config().set_debug_options(options);
    HloComputation* entry_computation = new_hlo_module->entry_computation();
    HloInstruction* cloned_dot_fusion = entry_computation->root_instruction();

    TF_ASSIGN_OR_RETURN(
        auto backend_config,
        cloned_dot_fusion->backend_config<FusionBackendConfig>());
    *backend_config.mutable_triton_gemm_config() = autotune_config;
    TF_RETURN_IF_ERROR(cloned_dot_fusion->set_backend_config(backend_config));

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

    // Verify the HLO here to catch potential rewrite errors.
    TF_RETURN_IF_ERROR(HloVerifier(/*layout_sensitive=*/true,
                                   /*allow_mixed_precision=*/false)
                           .Run(new_hlo_module.get())
                           .status());

    Status compilation_status = xla::gpu::CompileModuleToLlvmIrImpl(
        new_hlo_module.get(), &llvm_context,
        /*target_triple=*/nvptx::TargetTriple(),
        /*data_layout=*/nvptx::DataLayout(),
        /*platform_name=*/config_.GetExecutor()->platform()->Name(),
        /*platform_id=*/config_.GetExecutor()->platform()->id(),
        gpu_device_info, device_description.cuda_compute_capability(),
        device_description.rocm_compute_capability(),
        DummyCanShareBufferFunction,
        /*pointer_size=*/8, &compile_module_results);
    if (compilation_status.code() == absl::StatusCode::kResourceExhausted) {
      VLOG(2) << "Compilation of autotuning variant failed: "
              << compilation_status;
      return {std::nullopt};
    } else if (!compilation_status.ok()) {
      return compilation_status;
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
                            new_hlo_module->config().debug_options()));

    se::GpuAsmOpts ptxas_config =
        PtxOptsFromDebugOptions(new_hlo_module->config().debug_options());
    TF_ASSIGN_OR_RETURN(
        std::vector<uint8_t> cubin,
        se::CompileGpuAsm(config_.GetExecutor()->device_ordinal(), ptx.c_str(),
                          ptxas_config));

    uint64_t end_compilation_nanos = tsl::Env::Default()->NowNanos();
    absl::Duration compilation_time_span =
        absl::Nanoseconds(end_compilation_nanos - start_compilation_nanos);
    VLOG(1) << "Compilation took: " << compilation_time_span;

    return std::make_optional(
        CompilationResult{ptx, cubin, kernel_names, launch_dimensions});
  }

  AutotuneConfig config_;
  tsl::thread::ThreadPool* thread_pool_;
};

// Search space for exhaustive matmul autotuning.
constexpr std::array<int, 6> BLOCK_SIZES = {16, 32, 64, 128, 256, 512};
constexpr std::array<int, 4> NUM_STAGES = {1, 2, 3, 4};
constexpr std::array<int, 4> NUM_WARPS = {2, 4, 8, 16};
constexpr std::array<int, 5> SPLIT_K = {1, 2, 4, 8, 16};

std::vector<AutotuneResult::TritonGemmKey> GetExhaustiveMatmulAutotuneConfigs(
    const se::CudaComputeCapability compute_capability) {
  std::vector<AutotuneResult::TritonGemmKey> configs;
  bool mma_layout_v2 =
      compute_capability.IsAtLeast(se::CudaComputeCapability::AMPERE);
  for (int num_warps : NUM_WARPS) {
    for (int num_stages : NUM_STAGES) {
      // Volta doesn't support num_stages > 2.
      if (!mma_layout_v2 && num_stages > 2) {
        continue;
      }
      for (int block_m : BLOCK_SIZES) {
        for (int block_n : BLOCK_SIZES) {
          // Exclude configs not supported by MMA layout v2.
          if (mma_layout_v2 && (block_m * block_n / 256) % num_warps != 0) {
            continue;
          }
          for (int block_k : BLOCK_SIZES) {
            for (int split_k : SPLIT_K) {
              auto config = GemmKey(block_m, block_n, block_k, split_k,
                                    num_stages, num_warps);
              configs.push_back(std::move(config));
            }
          }
        }
      }
    }
  }
  return configs;
}

std::vector<AutotuneResult::TritonGemmKey> GetFixedMatmulAutotuneConfigs(
    const se::CudaComputeCapability compute_capability) {
  std::vector<AutotuneResult::TritonGemmKey> configs = {
      GemmKey(32, 32, 256, 1, 1, 4), GemmKey(64, 32, 32, 16, 1, 4),
      GemmKey(32, 64, 64, 4, 1, 4),  GemmKey(128, 128, 64, 4, 1, 4),
      GemmKey(16, 16, 256, 1, 1, 4), GemmKey(16, 128, 32, 16, 1, 4),
      GemmKey(16, 64, 128, 1, 1, 4), GemmKey(16, 128, 32, 8, 1, 4),
      GemmKey(16, 16, 512, 1, 1, 4), GemmKey(32, 16, 512, 1, 1, 4),
      GemmKey(64, 32, 64, 1, 2, 8)};
  if (compute_capability.IsAtLeast(se::CudaComputeCapability::AMPERE)) {
    absl::c_copy(
        std::vector<AutotuneResult::TritonGemmKey>{
            GemmKey(128, 256, 32, 1, 3, 8),  GemmKey(256, 128, 32, 1, 3, 8),
            GemmKey(256, 64, 32, 1, 4, 4),   GemmKey(64, 256, 32, 1, 4, 4),
            GemmKey(128, 64, 32, 1, 4, 4),   GemmKey(64, 128, 32, 1, 4, 4),
            GemmKey(128, 256, 32, 1, 3, 8),  GemmKey(256, 128, 128, 1, 3, 8),
            GemmKey(256, 64, 128, 1, 4, 4),  GemmKey(64, 256, 128, 1, 4, 4),
            GemmKey(128, 128, 128, 1, 4, 4), GemmKey(128, 64, 64, 1, 4, 4),
            GemmKey(64, 128, 64, 1, 4, 4),   GemmKey(128, 32, 64, 1, 4, 4),
            GemmKey(64, 32, 64, 1, 4, 4),    GemmKey(32, 128, 32, 1, 4, 4),
            GemmKey(128, 128, 32, 1, 4, 4),  GemmKey(16, 16, 256, 1, 3, 4),
            GemmKey(128, 128, 64, 2, 1, 8),  GemmKey(64, 64, 64, 1, 2, 4),
            GemmKey(16, 64, 256, 8, 1, 4),   GemmKey(256, 256, 128, 1, 3, 8)},
        std::back_inserter(configs));
  }
  if (compute_capability.IsAtLeast(se::CudaComputeCapability::HOPPER)) {
    configs.erase(
        std::remove_if(configs.begin(), configs.end(),
                       [](const AutotuneResult::TritonGemmKey& config) {
                         return (config.block_m() * config.block_n() / 256) %
                                    config.num_warps() !=
                                0;
                       }),
        configs.end());
  }
  return configs;
}

}  // anonymous namespace

std::vector<AutotuneResult::TritonGemmKey> GetPossibleMatmulAutotuneConfigs(
    const se::CudaComputeCapability compute_capability,
    bool exhaustive_tiling_search) {
  return exhaustive_tiling_search
             ? GetExhaustiveMatmulAutotuneConfigs(compute_capability)
             : GetFixedMatmulAutotuneConfigs(compute_capability);
}

std::unique_ptr<HloModule> ExtractInstructionIntoNewModule(
    const HloInstruction& hlo) {
  auto new_hlo_module = std::make_unique<HloModule>(
      "extracted", HloModuleConfig{},
      std::make_unique<CompilationEnvironments>(hlo.GetModule()->comp_envs()));
  int parameter_number = 0;
  HloComputation::Builder builder("entry_computation");
  HloCloneContext clone_context(new_hlo_module.get());
  std::vector<HloInstruction*> new_operands;
  for (const HloInstruction* operand : hlo.operands()) {
    std::unique_ptr<HloInstruction> new_parameter =
        HloInstruction::CreateParameter(parameter_number, operand->shape(),
                                        operand->name());
    ++parameter_number;
    new_operands.push_back(builder.AddInstruction(std::move(new_parameter)));
  }
  std::unique_ptr<HloInstruction> new_instruction =
      hlo.CloneWithNewOperands(hlo.shape(), new_operands, &clone_context);
  builder.AddInstruction(std::move(new_instruction));
  new_hlo_module->AddEntryComputationWithLayouts(builder.Build());
  return new_hlo_module;
}

std::unique_ptr<HloModule> ExtractComputationIntoNewModule(
    const HloComputation& computation) {
  auto new_hlo_module =
      std::make_unique<HloModule>("extracted", HloModuleConfig{},
                                  std::make_unique<CompilationEnvironments>(
                                      computation.parent()->comp_envs()));
  HloCloneContext clone_context(new_hlo_module.get());
  new_hlo_module->AddEntryComputationWithLayouts(
      computation.CloneInContext(clone_context));
  return new_hlo_module;
}

StatusOr<bool> TritonAutotuner::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  if (module->config().debug_options().xla_gpu_autotune_level() == 0) {
    return false;
  }
  return TritonAutotunerVisitor{config_, thread_pool_}.RunOnModule(
      module, execution_threads);
}

void TritonAutotuner::ClearCompilationCache() {
  absl::MutexLock lock(&compilation_cache_mutex);
  compilation_cache.clear();
}

}  // namespace gpu
}  // namespace xla
