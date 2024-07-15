/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/tools/xla_compile_lib.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "xla/client/xla_computation.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/service/export_hlo.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/symbol_repository.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_memory_allocator.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/status_to_from_proto.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/gpu_symbol_repository.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#endif
#if GOOGLE_CUDA
#include "xla/service/gpu/nvptx_compiler.h"
#elif TENSORFLOW_USE_ROCM
#include "xla/service/gpu/amdgpu_compiler.h"
#endif

namespace xla {

static absl::StatusOr<std::string> AotCompileCpuExecutable(
    std::unique_ptr<HloModule> hlo_module) {
  cpu::CpuCompiler cpu_compiler;
  auto module_group = std::make_unique<HloModuleGroup>(std::move(hlo_module));
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      cpu_compiler.Compile(std::move(module_group), {{nullptr}}, {nullptr}));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      cpu_compiler.Export(executables[0].get()));
  return aot_result->SerializeAsString();
}

static absl::StatusOr<std::string> CompileGpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::optional<Compiler::TargetConfig> target_config,
    CompilationResult& result) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  const bool aot = target_config.has_value();

#if GOOGLE_CUDA
  auto gpu_compiler = gpu::NVPTXCompiler();
#elif TENSORFLOW_USE_ROCM
  auto gpu_compiler = gpu::AMDGPUCompiler();
#endif

  auto module_group = std::make_unique<HloModuleGroup>(std::move(hlo_module));

  if (aot) {
    AotCompilationOptions aot_options(gpu_compiler.PlatformId());
    aot_options.set_target_config(*target_config);
    // We need the optimized module, so we call RunHloPasses ourselves above.
    aot_options.set_run_backend_only(true);

    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
        gpu_compiler.CompileAheadOfTime(std::move(module_group), aot_options));
    TF_ASSIGN_OR_RETURN(std::string compile_result,
                        aot_results[0]->SerializeAsString());
    *result.mutable_hlo_module() =
        aot_results[0]->optimized_module()->ToProto();
    return compile_result;
  }

  Compiler::CompileOptions compile_options;
  TF_RETURN_IF_ERROR(stream_executor::ValidateGPUMachineManager());
  TF_ASSIGN_OR_RETURN(
      stream_executor::StreamExecutor * stream_executor,
      stream_executor::GPUMachineManager()->ExecutorForDevice(0));
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorMemoryAllocator>(
          stream_executor);
  compile_options.device_allocator = allocator.get();

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      gpu_compiler.Compile(std::move(module_group), {{stream_executor}},
                           compile_options));
  *result.mutable_hlo_module() = executables[0]->module().ToProto();
  return executables[0]->module().ToString();
#else
  LOG(ERROR) << "Neither ROCm nor CUDA present; returning empty.";
  return "";
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

absl::StatusOr<std::string> CompileExecutable(
    std::unique_ptr<HloModule> hlo_module, BackendType backend,
    std::optional<Compiler::TargetConfig> target_config,
    CompilationResult& result) {
  if (backend == BackendType::kCpu) {
    return AotCompileCpuExecutable(std::move(hlo_module));
  }
  return CompileGpuExecutable(std::move(hlo_module), std::move(target_config),
                              result);
}

absl::Status WriteResultFile(const absl::string_view result_output_file,
                             TimerStats& stats,
                             CompilationResult& compilation_result) {
  if (result_output_file.empty()) {
    return absl::OkStatus();
  }
  absl::MutexLock ml(&stats.stats_mutex);
  const double secs = std::floor(stats.cumulative_secs);
  const double nanos =
      (stats.cumulative_secs - secs) * tsl::EnvTime::kSecondsToNanos;
  google::protobuf::Duration duration;
  duration.set_seconds(secs);
  duration.set_nanos(nanos);

  *compilation_result.mutable_perf_stats()->mutable_compilation_duration() =
      duration;
  *compilation_result.mutable_perf_stats()->mutable_total_duration() = duration;

  return tsl::WriteBinaryProto(
      tsl::Env::Default(), std::string(result_output_file), compilation_result);
}

absl::StatusOr<std::unique_ptr<HloModule>> LoadModule(
    const absl::string_view module_path) {
  auto format = std::string(tsl::io::Extension(module_path));
  if (format == "hlo" || format == "txt" || format == "pb") {
    return LoadModuleFromFile(
        std::string(module_path), format, hlo_module_loader_details::Config(),
        [&](HloModuleConfig* c) {}, nullptr);
  }
  std::string module_string;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), std::string(module_path), &module_string));

  mlir::DialectRegistry dialects;
  // TODO(b/248362914): Register all required dialects.
  dialects.insert<mlir::arith::ArithDialect>();
  dialects.insert<mlir::mhlo::MhloDialect>();
  dialects.insert<mlir::func::FuncDialect>();
  mlir::stablehlo::registerAllDialects(dialects);

  // Parse MHLO module.
  auto threading = mlir::MLIRContext::Threading::DISABLED;
  auto ctx = std::make_unique<mlir::MLIRContext>(dialects, threading);
  mlir::OwningOpRef<mlir::ModuleOp> module =
      mlir::parseSourceString<mlir::ModuleOp>(module_string, ctx.get());

  // Convert Mhlo to Hlo Module.
  XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(
      MlirToXlaComputation(*module, xla_computation, false, false));
  HloModuleProto hlo_module_proto = xla_computation.proto();

  TF_ASSIGN_OR_RETURN(ProgramShape shape, xla_computation.GetProgramShape());
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  HloModuleConfig config(shape);
  config.set_debug_options(debug_options);
  return HloModule::CreateFromProto(hlo_module_proto, config);
}

static absl::StatusOr<std::unique_ptr<HloModuleAndMetadata>>
ReadModuleFromSymbolRepo(absl::string_view symbol_repo,
                         absl::string_view symbol_reference,
                         BackendType backend) {
  std::unique_ptr<HloModuleAndMetadata> mod;
  TF_ASSIGN_OR_RETURN(
      mod, LookupSymbolInRepository(symbol_repo, symbol_reference, backend));
  if (mod == nullptr) {
    return absl::NotFoundError(
        absl::StrCat("Could not find ", symbol_reference, " in ", symbol_repo));
  }
  return mod;
}

static absl::StatusOr<bool> LoadAutotuneDataFromModule(
    HloModuleAndMetadata* mod, BackendType backend) {
  if (backend == BackendType::kGpu) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (auto* data = static_cast<gpu::GpuBackendSpecificData*>(
            mod->backend_specific_data.get());
        data != nullptr && data->autotune_results.has_value()) {
      TF_RETURN_IF_ERROR(
          gpu::AutotunerUtil::LoadAutotuneResults(*data->autotune_results));
      return true;
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  }
  return false;
}

static std::unique_ptr<Compiler::TargetConfig> ReadTargetConfigFromModule(
    HloModuleAndMetadata* mod, BackendType backend) {
  if (backend == BackendType::kGpu) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (auto* data = static_cast<gpu::GpuBackendSpecificData*>(
            mod->backend_specific_data.get());
        data != nullptr) {
      return std::move(mod->target_config);
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  }

  return nullptr;
}

absl::Status XlaCompileMain(const XlaCompileOptions& options) {
  std::unique_ptr<HloModule> hlo_module;
  std::unique_ptr<Compiler::TargetConfig> target_config;
  if (options.platform != "cpu" && options.platform != "gpu") {
    return absl::UnimplementedError(
        absl::StrCat("platform", options.platform, " is not supported"));
  }

  const BackendType backend =
      (options.platform == "gpu" ? BackendType::kGpu : BackendType::kCpu);

  absl::string_view symbol_repo = options.repo_options.symbol_repo;
  if (absl::string_view symbol_id = options.repo_options.symbol_id;
      !symbol_id.empty()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleAndMetadata> mod,
        ReadModuleFromSymbolRepo(symbol_repo, symbol_id, backend));

    hlo_module = std::move(mod->hlo_module);
    target_config = ReadTargetConfigFromModule(mod.get(), backend);
  } else {
    TF_ASSIGN_OR_RETURN(hlo_module, LoadModule(options.module_path));
  }

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  bool found_autotune = false;
#endif

  if (absl::string_view optimized_symbol_id =
          options.repo_options.optimized_symbol_id;
      !optimized_symbol_id.empty()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleAndMetadata> optimized_mod,
        ReadModuleFromSymbolRepo(symbol_repo, optimized_symbol_id, backend));

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    TF_ASSIGN_OR_RETURN(found_autotune, LoadAutotuneDataFromModule(
                                            optimized_mod.get(), backend));
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  }

  xla::TimerStats stats;
  xla::ScopedLoggingTimer timer("compilation", true, "xla_compile_main.cc", 1,
                                &stats);
  CompilationResult compilation_result;
  absl::Cleanup cleanup([&] {
    // Make sure we stop the timer if compilation failed.
    timer.StopAndLog();
    if (!options.result_output_file.empty()) {
      TF_QCHECK_OK(WriteResultFile(options.result_output_file, stats,
                                   compilation_result));
    }
  });
  // Run AOT compilation.
  std::optional<Compiler::TargetConfig> cfg = std::nullopt;
  if (backend == BackendType::kGpu) {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (absl::string_view gpu_target_config_path =
            options.gpu_options.gpu_target_config_path;
        !gpu_target_config_path.empty()) {
      // Parse GpuTargetConfig.
      std::string gpu_target_config_string;
      TF_RETURN_IF_ERROR(tsl::ReadFileToString(
          tsl::Env::Default(), std::string(gpu_target_config_path),
          &gpu_target_config_string));
      stream_executor::GpuTargetConfigProto gpu_target_config_proto;

      if (!tsl::protobuf::TextFormat::ParseFromString(
              gpu_target_config_string, &gpu_target_config_proto)) {
        return FailedPrecondition("Failed to parse GpuTargetConfigProto");
      }

      target_config =
          std::make_unique<Compiler::TargetConfig>(gpu_target_config_proto);

      if (absl::string_view autotune_results_path =
              options.gpu_options.autotune_results_path;
          !found_autotune && !autotune_results_path.empty()) {
        TF_RETURN_IF_ERROR(gpu::AutotunerUtil::LoadAutotuneResultsFromFile(
            autotune_results_path));
      }
    }

    cfg = (options.gpu_options.use_attached_device)
              ? std::nullopt
              : std::make_optional(*std::move(target_config));
#endif
  }
  auto result = CompileExecutable(std::move(hlo_module), backend,
                                  std::move(cfg), compilation_result);
  *compilation_result.mutable_status() = tsl::StatusToProto(result.status());
  if (!result.ok()) {
    return result.status();
  }

  TF_RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(),
                                            options.output_path, *result));

  if (options.repo_options.wait_for_uploads) {
    MaybeWaitForUploads();
  }
  return absl::OkStatus();
}

}  // namespace xla
