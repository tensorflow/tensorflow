/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"  // from @llvm-project
#include "mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/IR/DialectRegistry.h"  // from @llvm-project
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "mlir/Parser/Parser.h"  // from @llvm-project
#include "stablehlo/dialect/Register.h"  // from @stablehlo
#include "xla/autotune_results.pb.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/symbol_repository.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor_pimpl.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/util/command_line_flags.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/gpu_symbol_repository.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#endif
#if GOOGLE_CUDA
#include "xla/service/gpu/nvptx_compiler.h"
#elif TENSORFLOW_USE_ROCM
#include "xla/service/gpu/amdgpu_compiler.h"
#endif

namespace xla {
namespace xla_compile {

const char kUsageHeader[] =
    "xla_compile performs ahead-of-time compilation of an MHLO, StableHLO or "
    "HLO module,\nresulting in an AotCompilationResult compiled for CPU or GPU."
    "\n"
    "A typical invocation looks like this:\n"
    "\n"
    "   $ xla_compile --module_file=mymodule.mlir --output_file=output "
    "--platform=cpu"
    "\n"
    "For GPU, either the attached GPU or a simulated one may be used. To use "
    "a simulated device, set --gpu_target_config to a textproto file "
    "containing a GpuTargetConfigProto forthe device you wish to simulate. To "
    "use the attached GPU, do not set this flag. When compiling with the "
    "attached device, --output_file will contain a text-format HLO module "
    "instead of an AotCompilationResult."
    "\n"
    "HLO may also be looked up in a symbol repository (see symbol_repository.h"
    ") by passing --symbol_repository to a linked-in symbol repository "
    "implementation and setting --symbol_reference to a reference of a symbol "
    "understood by that repository."
    "\n";

StatusOr<std::string> AotCompileCpuExecutable(
    std::unique_ptr<HloModule> hlo_module) {
  cpu::CpuCompiler cpu_compiler;
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<cpu::CpuExecutable> cpu_executable,
      cpu_compiler.CompileXlaRuntimeCpuExecutable(std::move(hlo_module)));
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      cpu_compiler.Export(cpu_executable.get()));
  TF_ASSIGN_OR_RETURN(std::string result, aot_result->SerializeAsString());
  return result;
}

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
StatusOr<std::string> CompileGpuExecutable(
    std::unique_ptr<HloModule> hlo_module,
    const std::optional<Compiler::TargetConfig> target_config) {
  const bool aot = target_config.has_value();

#if GOOGLE_CUDA
  auto gpu_compiler = gpu::NVPTXCompiler();
#elif TENSORFLOW_USE_ROCM
  auto gpu_compiler = gpu::AMDGPUCompiler();
#endif
  Compiler::CompileOptions compile_options;

  stream_executor::StreamExecutor* stream_executor = nullptr;
  std::unique_ptr<stream_executor::StreamExecutorMemoryAllocator> allocator;
  if (aot) {
    compile_options.target_config = *target_config;
  } else {
    TF_RETURN_IF_ERROR(stream_executor::ValidateGPUMachineManager());
    TF_ASSIGN_OR_RETURN(
        stream_executor,
        stream_executor::GPUMachineManager()->ExecutorForDevice(0));
    allocator =
        std::make_unique<stream_executor::StreamExecutorMemoryAllocator>(
            stream_executor);
    compile_options.device_allocator = allocator.get();
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> module_after_opt,
      gpu_compiler.RunHloPasses(std::move(hlo_module), stream_executor,
                                compile_options));

  if (aot) {
    auto module_group =
        std::make_unique<HloModuleGroup>(std::move(module_after_opt));

    AotCompilationOptions aot_options(gpu_compiler.PlatformId());
    aot_options.set_target_config(*target_config);

    TF_ASSIGN_OR_RETURN(
        std::vector<std::unique_ptr<AotCompilationResult>> aot_results,
        gpu_compiler.CompileAheadOfTime(std::move(module_group), aot_options));
    TF_ASSIGN_OR_RETURN(std::string result,
                        aot_results[0]->SerializeAsString());
    return result;
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      gpu_compiler.RunBackend(std::move(module_after_opt), stream_executor,
                              compile_options));
  return executable->module().ToString();
}

#endif

xla::StatusOr<std::unique_ptr<HloModule>> LoadModule(
    const std::string& module_path) {
  auto format = std::string(tsl::io::Extension(module_path));
  if (format == "hlo" || format == "txt") {
    return LoadModuleFromFile(
        module_path, hlo_module_loader_details::Config(),
        /*format=*/"hlo", [&](HloModuleConfig* c) {}, nullptr);
  }
  std::string module_string;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), module_path, &module_string));

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
  DebugOptions debug_options = DefaultDebugOptionsIgnoringFlags();
  HloModuleConfig config(shape);
  config.set_debug_options(debug_options);
  return HloModule::CreateFromProto(hlo_module_proto, config);
}

Status XlaCompileMain(
    const std::string& module_path, const std::string& output_path,
    const std::string& platform, const std::string& gpu_target_config_path,
    const std::string& autotune_results_path, const std::string& symbol_repo,
    const std::string& symbol_id, const bool use_attached_device) {
  std::unique_ptr<HloModule> hlo_module;
  std::unique_ptr<Compiler::TargetConfig> target_config;
  if (!symbol_id.empty()) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<HloModuleAndMetadata> mod,
        LookupSymbolInRepository(symbol_repo, symbol_id, BackendType::kGpu));
    if (mod == nullptr) {
      return absl::NotFoundError(
          absl::StrCat("Could not find ", symbol_id, " in ", symbol_repo));
    }
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    if (auto* data = static_cast<gpu::GpuBackendSpecificData*>(
            mod->backend_specific_data.get());
        data != nullptr) {
      target_config = std::move(mod->target_config);
    }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
    hlo_module = std::move(mod->hlo_module);
  } else {
    TF_ASSIGN_OR_RETURN(hlo_module, LoadModule(module_path));
  }

  // Run AOT compilation.
  std::string result;
  if (platform == "cpu") {
    TF_ASSIGN_OR_RETURN(result, AotCompileCpuExecutable(std::move(hlo_module)));
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  } else if (platform == "gpu") {
    if (!gpu_target_config_path.empty()) {
      // Parse GpuTargetConfig.
      std::string gpu_target_config_string;
      TF_RETURN_IF_ERROR(tsl::ReadFileToString(tsl::Env::Default(),
                                               gpu_target_config_path,
                                               &gpu_target_config_string));
      stream_executor::GpuTargetConfigProto gpu_target_config_proto;

      if (!tsl::protobuf::TextFormat::ParseFromString(
              gpu_target_config_string, &gpu_target_config_proto)) {
        return FailedPrecondition("Failed to parse GpuTargetConfigProto");
      }

      target_config =
          std::make_unique<Compiler::TargetConfig>(gpu_target_config_proto);

      if (!autotune_results_path.empty()) {
        TF_RETURN_IF_ERROR(gpu::AutotunerUtil::LoadAutotuneResultsFromFile(
            autotune_results_path));
      }
    }

    std::optional<Compiler::TargetConfig> cfg =
        (use_attached_device) ? std::nullopt
                              : std::make_optional(*std::move(target_config));
    TF_ASSIGN_OR_RETURN(result,
                        CompileGpuExecutable(std::move(hlo_module), cfg));
#endif
  } else {
    return Unimplemented("platform %s not supported", platform);
  }

  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(tsl::Env::Default(), output_path, result));
  return OkStatus();
}

}  // end namespace xla_compile
}  // end namespace xla

// Read the input file containing the MHLO module, and write a Serialized
// AotCompilationResult or Executable to the output file.
int main(int argc, char* argv[]) {
  std::string module_path;
  std::string output_path;
  std::string platform;
  std::string gpu_target_config_path;
  std::string autotune_results_path;
  std::string symbol_repository;
  std::string symbol_id;
  bool use_attached_device = false;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("module_file", &module_path,
                "The path to the HLO, MHLO or StableHLO file"),
      tsl::Flag("output_file", &output_path, "The path to the output file"),
      tsl::Flag("platform", &platform,
                "The platform on which the built executable runs"),
      tsl::Flag("gpu_target_config", &gpu_target_config_path,
                "The path to a text-format GpuTargetConfig. If not provided, "
                "an attached GPU will be used."),
      tsl::Flag("autotune_results", &autotune_results_path,
                "The path to AutotuneResults, optional when compiling for"
                " GPU"),
      tsl::Flag("symbol_repo", &symbol_repository,
                "Which SymbolRepository to look up --symbol_reference in. If "
                "the repository contains a GpuTargetConfig, "
                "--gpu_target_config will take precedence if it is also set."),
      tsl::Flag("symbol_reference", &symbol_id,
                "Symbol ID to look up in a SymbolRepository. Overrides "
                "--module_file."),
      tsl::Flag("use_attached_device", &use_attached_device,
                "Whether to use the attached GPU or not. Overrides the "
                "AOT-vs-device-backed inference based on the presence of "
                "--gpu_target_config, which is relevant when a GpuTargetConfig "
                "can be found in the symbol repository."),
  };

  tsl::string usage = xla::xla_compile::kUsageHeader;
  usage += tsl::Flags::Usage(argv[0], flag_list);
  if (argc > 1 && absl::string_view(argv[1]) == "--help") {
    std::cerr << usage << "\n";
    return 0;
  }

  bool parsed_flags_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  QCHECK(parsed_flags_ok) << "\n" << usage;

  tsl::port::InitMain(usage.c_str(), &argc, &argv);

  xla::Status result = xla::xla_compile::XlaCompileMain(
      module_path, output_path, platform, gpu_target_config_path,
      autotune_results_path, symbol_repository, symbol_id, use_attached_device);
  if (!result.ok()) {
    LOG(ERROR) << "Compilation failed: " << result;
    return 1;
  }

  LOG(INFO) << "Compilation succeeded";
  return 0;
}
