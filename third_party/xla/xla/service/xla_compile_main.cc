/* Copyright 2022 The OpenXLA Authors.

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

#include "absl/cleanup/cleanup.h"
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
#include "xla/mlir_hlo/mhlo/IR/hlo_ops.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/service/compiler.h"
#include "xla/service/export_hlo.h"
#include "xla/service/hlo_module_config.h"
#include "xla/service/symbol_repository.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/statusor.h"
#include "xla/tools/hlo_module_loader.h"
#include "xla/tools/xla_compile_lib.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status_to_from_proto.h"
#include "tsl/platform/types.h"
#include "tsl/util/command_line_flags.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/gpu/gpu_symbol_repository.h"
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
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  HloModuleConfig config(shape);
  config.set_debug_options(debug_options);
  return HloModule::CreateFromProto(hlo_module_proto, config);
}

Status XlaCompileMain(
    const std::string& module_path, const std::string& output_path,
    const std::string& platform, const std::string& gpu_target_config_path,
    const std::string& autotune_results_path, const std::string& symbol_repo,
    const std::string& symbol_id, const bool use_attached_device,
    const bool wait_for_uploads, const std::string& result_output_file) {
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

  xla::TimerStats stats;
  xla::ScopedLoggingTimer timer("compilation", true, "xla_compile_main.cc", 1,
                                &stats);
  CompilationResult compilation_result;
  absl::Cleanup cleanup([&] {
    // Make sure we stop the timer if compilation failed.
    timer.StopAndLog();
    if (!result_output_file.empty()) {
      TF_QCHECK_OK(
          WriteResultFile(result_output_file, stats, compilation_result));
    }
  });
  // Run AOT compilation.
  std::optional<Compiler::TargetConfig> cfg = std::nullopt;
  if (platform == "gpu") {
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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
      if (!autotune_results_path.empty()) {
        TF_RETURN_IF_ERROR(gpu::AutotunerUtil::LoadAutotuneResultsFromFile(
            autotune_results_path));
      }
#endif
    }

    cfg = (use_attached_device) ? std::nullopt
                                : std::make_optional(*std::move(target_config));
  }
  auto result = CompileExecutable(std::move(hlo_module), platform, cfg,
                                  compilation_result);
  if (!result.ok()) {
    *compilation_result.mutable_status() = tsl::StatusToProto(result.status());
    return result.status();
  }

  TF_RETURN_IF_ERROR(
      tsl::WriteStringToFile(tsl::Env::Default(), output_path, *result));

  if (wait_for_uploads) {
    MaybeWaitForUploads();
  }
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
  bool wait_for_uploads = false;
  std::string result_output_file;
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
      tsl::Flag("wait_for_uploads", &wait_for_uploads,
                "Whether to wait for uploads to a symbol repository to "
                "complete. See export_hlo.h for more on uploads."),
      tsl::Flag("result_output_file", &result_output_file,
                "File to write a serialized xla.CompilationResult proto to."),
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
      autotune_results_path, symbol_repository, symbol_id, use_attached_device,
      wait_for_uploads, result_output_file);
  if (!result.ok()) {
    LOG(ERROR) << "Compilation failed: " << result;
    return 1;
  }

  LOG(INFO) << "Compilation succeeded";
  return 0;
}
