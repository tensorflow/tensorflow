/* Copyright 2025 The OpenXLA Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http:www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/gpu/autotuner/gpu_profiler.h"
#include "xla/backends/gpu/autotuner/legacy_cache.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/autotuning/autotuner_pass.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/init_main.h"

namespace {

const char* const kUsage = R"(
This tool autotunes an HLO module from a given HLO file and prints the
autotuned module to stdout. Honour XLA_FLAGS.

Usage:

  bazel run autotuner_main -- --hlo_file=path/to/hlo_module
)";
}  // namespace

namespace xla {
namespace gpu {
namespace {

absl::StatusOr<std::unique_ptr<HloModule>> GetModule(
    const std::string& hlo_file) {
  std::string hlo_text;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), hlo_file, &hlo_text));
  return ParseAndReturnUnverifiedModule(hlo_text);
}

absl::Status Autotune(HloModule& module) {
  TF_ASSIGN_OR_RETURN(std::string platform_name,
                      PlatformUtil::CanonicalPlatformName("gpu"));

  TF_ASSIGN_OR_RETURN(se::Platform * platform,
                      se::PlatformManager::PlatformWithName(
                          absl::AsciiStrToUpper(platform_name)));
  if (platform->VisibleDeviceCount() == 0) {
    return absl::InternalError("No devices found");
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler,
                      xla::Compiler::GetForPlatform(platform->id()));
  se::StreamExecutor* stream_executor = platform->ExecutorForDevice(0).value();
  auto* gpu_compiler = tensorflow::down_cast<GpuCompiler*>(compiler.get());
  auto alias_info =
      gpu_compiler->GetAliasInfo(stream_executor->GetDeviceDescription());
  DebugOptions debug_options = GetDebugOptionsFromFlags();
  Compiler::GpuTargetConfig target_config(stream_executor);

  mlir::MLIRContext mlir_context;
  xla::RegisterSymbolicExprStorage(&mlir_context);
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<CodegenBackend>> backends,
                      gpu_compiler->GetAutotunerBackends(
                          stream_executor, &target_config, alias_info.get(),
                          debug_options, &mlir_context));

  std::unique_ptr<se::DeviceAddressAllocator> allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_executor);

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "autotuner",
                                      tsl::port::MaxParallelism());

  xla::AutotuneConfig autotune_config = GetAutotuneConfig(debug_options);
  auto profiler = GpuProfiler::Create(
      stream_executor, GetProfileOptions(debug_options, autotune_config),
      allocator.get());

  if (profiler == nullptr) {
    return absl::InternalError("Failed to create profiler to autotune.");
  }
  std::unique_ptr<AutotunerCacheInterface> cache =
      std::make_unique<LegacyCache>(
          debug_options.xla_gpu_per_fusion_autotune_cache_dir(),
          debug_options.xla_gpu_experimental_autotune_cache_mode(),
          target_config.device_description);
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Autotuner> autotuner,
      Autotuner::Create(std::move(backends), std::move(profiler),
                        autotune_config, std::move(cache), &thread_pool));

  bool do_not_autotune_cublas_and_cudnn =
      debug_options.xla_gpu_experimental_disable_binary_libraries() ||
      debug_options.xla_gpu_autotune_level() == 0 ||
      debug_options.xla_gpu_exclude_nondeterministic_ops();
  auto should_autotune = [do_not_autotune_cublas_and_cudnn](
                             const HloInstruction& instruction) -> bool {
    if (!do_not_autotune_cublas_and_cudnn &&
        (instruction.opcode() == HloOpcode::kCustomCall &&
         (IsCublasGemm(instruction) ||
          IsCustomCallToDnnConvolution(instruction)))) {
      return true;
    }
    if (instruction.opcode() != HloOpcode::kFusion) {
      return false;
    }
    auto gpu_config = instruction.backend_config<GpuBackendConfig>();
    const FusionBackendConfig& backend_config =
        gpu_config->fusion_backend_config();
    if (backend_config.kind() == kTritonGemmFusionKind) {
      return !backend_config.has_triton_gemm_config();
    }
    if (backend_config.kind() == kCuDnnFusionKind) {
      return !backend_config.has_cudnn_fusion_config();
    }
    if (backend_config.kind() == kCustomFusionKind) {
      return !backend_config.has_custom_fusion_config();
    }
    return false;
  };

  return autotuner->Autotune(&module, should_autotune);
}

}  // namespace
}  // namespace gpu
}  // namespace xla

int main(int argc, char* argv[]) {
  std::string hlo_file;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo_file", &hlo_file, "Path to the HLO file to autotune."),
  };

  const std::string usage_string =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    LOG(QFATAL) << usage_string;
  }
  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);
  auto module = xla::gpu::GetModule(hlo_file);
  CHECK_OK(module.status());
  CHECK_OK(xla::gpu::Autotune(*module.value()));
  std::cout << module.value()->ToString() << std::endl;
  return 0;
}
