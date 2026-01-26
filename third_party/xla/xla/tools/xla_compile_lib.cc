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
#include "absl/base/call_once.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "llvm-c/Target.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OwningOpRef.h"
#include "google/protobuf/text_format.h"
#include "xla/backends/cpu/target_machine_options.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/executable.h"
#include "xla/service/export_hlo.h"
#include "xla/service/gpu/autotuning/autotuner_util.h"
#include "xla/service/gpu/gpu_symbol_repository.h"
#include "xla/service/gpu_topology.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/platform_util.h"
#include "xla/service/symbol_repository.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/shape.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/env_time.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/status_to_from_proto.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla.pb.h"
#include "tsl/platform/path.h"

namespace xla {

static absl::StatusOr<std::string> AotCompileCpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::optional<Compiler::CpuTargetConfig> target_config) {
  cpu::CpuCompiler cpu_compiler;
  Compiler::CompileOptions compile_options;
  compile_options.cpu_target_config = std::move(target_config);
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      cpu_compiler.Compile(std::move(hlo_module), {nullptr}, compile_options));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      cpu_compiler.Export(executables[0].get()));
  return aot_result->SerializeAsString();
}

static absl::StatusOr<std::string> CompileGpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    std::optional<Compiler::GpuTargetConfig> target_config,
    CompilationResult& result) {
  TF_ASSIGN_OR_RETURN(std::string platform_name,
                      xla::PlatformUtil::CanonicalPlatformName("gpu"));
  platform_name = absl::AsciiStrToUpper(platform_name);
  const bool aot = target_config.has_value();

  TF_ASSIGN_OR_RETURN(
      se::Platform::Id platform_id,
      xla::PlatformUtil::GetPlatformIdFromCanonicalName(platform_name));
  TF_ASSIGN_OR_RETURN(auto gpu_compiler, Compiler::GetForPlatform(platform_id));

  if (aot) {
    AotCompilationOptions aot_options(platform_id);
    GpuTopology topology =
        GetSingleDeviceGpuTopology(/*platform_version=*/"", *target_config);
    aot_options.set_gpu_topology(topology);
    // We need the optimized module, so we call RunHloPasses ourselves above.
    aot_options.set_run_backend_only(true);

    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
        gpu_compiler->CompileAheadOfTime(std::move(hlo_module), aot_options));
    TF_ASSIGN_OR_RETURN(std::string compile_result,
                        aot_results[0]->SerializeAsString());
    *result.mutable_hlo_module() =
        aot_results[0]->optimized_module()->ToProto();
    return compile_result;
  }
  TF_ASSIGN_OR_RETURN(
      auto platform,
      stream_executor::PlatformManager::PlatformWithName(platform_name));
  Compiler::CompileOptions compile_options;
  TF_ASSIGN_OR_RETURN(stream_executor::StreamExecutor * stream_executor,
                      platform->ExecutorForDevice(0));
  auto allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_executor);
  compile_options.device_allocator = allocator.get();

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<Executable>> executables,
      gpu_compiler->Compile(std::move(hlo_module), {stream_executor},
                            compile_options));
  *result.mutable_hlo_module() = executables[0]->module().ToProto();
  return executables[0]->module().ToString();
}

absl::StatusOr<std::string> CompileExecutable(
    std::unique_ptr<HloModule> hlo_module, BackendType backend,
    std::optional<Compiler::GpuTargetConfig> gpu_target_config,
    std::optional<Compiler::CpuTargetConfig> cpu_target_config,
    CompilationResult& result) {
  if (backend == BackendType::kCpu) {
    return AotCompileCpuExecutable(std::move(hlo_module),
                                   std::move(cpu_target_config));
  }
  return CompileGpuExecutable(std::move(hlo_module),
                              std::move(gpu_target_config), result);
}

absl::Status WriteResultFile(const absl::string_view result_output_file,
                             TimerStats& stats,
                             CompilationResult& compilation_result) {
  if (result_output_file.empty()) {
    return absl::OkStatus();
  }
  absl::MutexLock ml(stats.stats_mutex);
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

  // Parse MHLO module.
  auto threading = mlir::MLIRContext::Threading::DISABLED;
  auto ctx = std::make_unique<mlir::MLIRContext>(threading);

  TF_ASSIGN_OR_RETURN(mlir::OwningOpRef<mlir::ModuleOp> module,
                      ParseMlirModuleString(module_string, *ctx));

  // Convert Mhlo to Hlo Module.
  XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(*module, xla_computation,
                                          /*use_tuple_args=*/false,
                                          /*return_tuple=*/false,
                                          /*exec_build_options=*/nullptr));
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

static std::unique_ptr<Compiler::GpuTargetConfig> ReadTargetConfigFromModule(
    HloModuleAndMetadata* mod, BackendType backend) {
  if (backend == BackendType::kGpu) {
    if (auto* data = static_cast<gpu::GpuBackendSpecificData*>(
            mod->backend_specific_data.get());
        data != nullptr) {
      return std::move(mod->target_config);
    }
  }

  return nullptr;
}

namespace internal {

absl::StatusOr<bool> LoadAutotuneDataFromModule(HloModuleAndMetadata* mod,
                                                BackendType backend) {
  if (backend == BackendType::kGpu) {
    if (auto* data = static_cast<gpu::GpuBackendSpecificData*>(
            mod->backend_specific_data.get());
        data != nullptr && data->autotune_results.has_value() &&
        mod->hlo_module->config().debug_options().xla_gpu_autotune_level() >
            0) {
      TF_RETURN_IF_ERROR(
          gpu::AutotunerUtil::LoadAutotuneResults(*data->autotune_results));
      return true;
    }
  }
  return false;
}

static absl::once_flag targets_init;

static void InitializeTargets() {
  // Initialize all LLVM targets so we can cross compile.
#if XLA_LLVM_AARCH32_AVAILABLE
  LLVMInitializeARMTarget();
  LLVMInitializeARMTargetInfo();
  LLVMInitializeARMTargetMC();
  LLVMInitializeARMAsmParser();
  LLVMInitializeARMAsmPrinter();
#endif
#if XLA_LLVM_AARCH64_AVAILABLE
  LLVMInitializeAArch64Target();
  LLVMInitializeAArch64TargetInfo();
  LLVMInitializeAArch64TargetMC();
  LLVMInitializeAArch64AsmParser();
  LLVMInitializeAArch64AsmPrinter();
#endif
#if XLA_LLVM_HEXAGON_AVAILABLE
  LLVMInitializeHexagonTarget();
  LLVMInitializeHexagonTargetInfo();
  LLVMInitializeHexagonTargetMC();
  LLVMInitializeHexagonAsmParser();
  LLVMInitializeHexagonAsmPrinter();
#endif
#if XLA_LLVM_POWERPC_AVAILABLE
  LLVMInitializePowerPCTarget();
  LLVMInitializePowerPCTargetInfo();
  LLVMInitializePowerPCTargetMC();
  LLVMInitializePowerPCAsmParser();
  LLVMInitializePowerPCAsmPrinter();
#endif
#if XLA_LLVM_S390X_AVAILABLE
  LLVMInitializeSystemZTarget();
  LLVMInitializeSystemZTargetInfo();
  LLVMInitializeSystemZTargetMC();
  LLVMInitializeSystemZAsmParser();
  LLVMInitializeSystemZAsmPrinter();
#endif
#if XLA_LLVM_X86_AVAILABLE
  LLVMInitializeX86Target();
  LLVMInitializeX86TargetInfo();
  LLVMInitializeX86TargetMC();
  LLVMInitializeX86AsmParser();
  LLVMInitializeX86AsmPrinter();
#endif
}

}  // namespace internal

absl::Status XlaCompileMain(const XlaCompileOptions& options) {
  absl::call_once(internal::targets_init, &internal::InitializeTargets);
  std::unique_ptr<HloModule> hlo_module;
  if (options.platform != "cpu" && options.platform != "gpu") {
    return absl::UnimplementedError(
        absl::StrCat("platform", options.platform, " is not supported"));
  }

  if (options.output_file.empty() && options.result_output_file.empty()) {
    return absl::InvalidArgumentError(
        "At least one of output_file and result_output_file is required");
  }

  const BackendType backend =
      (options.platform == "gpu" ? BackendType::kGpu : BackendType::kCpu);

  std::optional<Compiler::GpuTargetConfig> gpu_cfg = std::nullopt;
  absl::string_view symbol_repo = options.repo_options.symbol_repo;
  if (absl::string_view symbol_id = options.repo_options.symbol_id;
      !symbol_id.empty()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleAndMetadata> mod,
        ReadModuleFromSymbolRepo(symbol_repo, symbol_id, backend));

    hlo_module = std::move(mod->hlo_module);
    gpu_cfg = std::move(*ReadTargetConfigFromModule(mod.get(), backend));
  } else {
    TF_ASSIGN_OR_RETURN(hlo_module, LoadModule(options.module_path));
  }

  bool found_autotune = false;

  if (absl::string_view optimized_symbol_id =
          options.repo_options.optimized_symbol_id;
      !optimized_symbol_id.empty()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleAndMetadata> optimized_mod,
        ReadModuleFromSymbolRepo(symbol_repo, optimized_symbol_id, backend));

    TF_ASSIGN_OR_RETURN(found_autotune, internal::LoadAutotuneDataFromModule(
                                            optimized_mod.get(), backend));
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
  std::optional<Compiler::CpuTargetConfig> cpu_cfg = std::nullopt;

  if (backend == BackendType::kGpu) {
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

      TF_ASSIGN_OR_RETURN(gpu_cfg, Compiler::GpuTargetConfig::FromProto(
                                       gpu_target_config_proto));

      if (absl::string_view autotune_results_path =
              options.gpu_options.autotune_results_path;
          !found_autotune && !autotune_results_path.empty() &&
          hlo_module->config().debug_options().xla_gpu_autotune_level() > 0) {
        TF_RETURN_IF_ERROR(gpu::AutotunerUtil::LoadAutotuneResultsFromFile(
            autotune_results_path));
      }
    }
    if (options.gpu_options.use_attached_device) {
      gpu_cfg = std::nullopt;
    }
  } else if (backend == BackendType::kCpu) {
    cpu::TargetMachineOptions target_machine_options(
        options.cpu_options.target_triple, options.cpu_options.target_cpu,
        options.cpu_options.target_features);
    cpu_cfg =
        std::make_optional<Compiler::CpuTargetConfig>(target_machine_options);
  }
  auto result =
      CompileExecutable(std::move(hlo_module), backend, std::move(gpu_cfg),
                        std::move(cpu_cfg), compilation_result);
  *compilation_result.mutable_status() = tsl::StatusToProto(result.status());
  if (!result.ok()) {
    return result.status();
  }

  if (!options.output_file.empty()) {
    TF_RETURN_IF_ERROR(tsl::WriteStringToFile(tsl::Env::Default(),
                                              options.output_file, *result));
  }

  if (options.repo_options.wait_for_uploads) {
    MaybeWaitForUploads();
  }
  return absl::OkStatus();
}

}  // namespace xla
