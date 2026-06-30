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
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/ascii.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/autotuner_cache_interface.h"
#include "xla/backends/autotuner/backends.pb.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/codegen_orchestrator.h"
#include "xla/backends/autotuner/config_assigner.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/backends/gpu/autotuner/gpu_profiler.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/analysis/symbolic_expr.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/parser/hlo_parser.h"
#include "xla/service/compiler.h"
#include "xla/service/gpu/autotuning/autotuner_pass.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/platform_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor_address_allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/tsl/util/command_line_flags.h"
#include "xla/xla.pb.h"
#include "tsl/platform/cpu_info.h"
#include "tsl/platform/init_main.h"

namespace {

const char* const kUsage = R"(
This tool autotunes a list of HLO modules and prints the results to stdout.

Usage:

  bazel run autotuner_main -- --hlo_files=path/to/hlo_module1,path/to/hlo_module2
)";
}  // namespace

namespace xla {
namespace gpu {
namespace {

// An AutotunerCache that prints the cached configs to stdout. Used for
// debugging and testing.
class PrintingAutotunerCache : public AutotunerCacheInterface {
 public:
  std::optional<Config> Lookup(const HloInstruction* instr) override {
    return std::nullopt;
  }

  absl::Status Insert(const HloInstruction* instr,
                      const Config& best_config) override {
    std::cout << "PrintingAutotunerCache:\n"
              << "  Instruction: " << instr->ToString() << "\n"
              << "  Backend:     "
              << autotuner::Backend_Name(best_config.codegen_backend) << "\n"
              << "  Config:      "
              << best_config.backend_config.ShortDebugString() << std::endl;
    return absl::OkStatus();
  }

  CacheStats GetCacheStats() const override { return {}; }

  absl::StatusOr<std::string> Serialize(
      absl::Span<const HloInstruction* const> instructions_to_serialize)
      override {
    return "";
  }

  absl::Status Deserialize(absl::string_view serialized_cache) override {
    return absl::OkStatus();
  }
};

absl::StatusOr<std::unique_ptr<HloModule>> GetModule(
    absl::string_view hlo_file) {
  std::string hlo_text;
  RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), hlo_file, &hlo_text));
  return ParseAndReturnUnverifiedModule(hlo_text);
}

struct AutotunerEnvironment {
  // For codegen backends.
  std::unique_ptr<Compiler> compiler;
  std::unique_ptr<mlir::MLIRContext> mlir_context;
  std::unique_ptr<AliasInfo> alias_info;
  std::unique_ptr<Compiler::GpuTargetConfig> target_config;
  std::unique_ptr<se::DeviceAddressAllocator> allocator;
  // For parallel codegen and autotuning.
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool;
  // The autotuner.
  std::unique_ptr<Autotuner> autotuner;
};

absl::StatusOr<AutotunerEnvironment> CreateAutotunerEnvironment(
    const DebugOptions& debug_options) {
  ConfigAssigner::Options assigner_options =
      GetConfigAssignerOptions(debug_options);
  CodegenOrchestrator::Options orchestrator_options =
      GetCodegenOrchestratorOptions(debug_options);
  ASSIGN_OR_RETURN(std::string platform_name,
                   PlatformUtil::CanonicalPlatformName("gpu"));

  ASSIGN_OR_RETURN(se::Platform * platform,
                   se::PlatformManager::PlatformWithName(
                       absl::AsciiStrToUpper(platform_name)));
  if (platform->VisibleDeviceCount() == 0) {
    return absl::InternalError("No devices found");
  }

  ASSIGN_OR_RETURN(std::unique_ptr<Compiler> compiler,
                   xla::Compiler::GetForPlatform(platform->id()));
  ASSIGN_OR_RETURN(se::StreamExecutor * stream_executor_0,
                   platform->ExecutorForDevice(0));
  auto* gpu_compiler = absl::down_cast<GpuCompiler*>(compiler.get());
  auto alias_info =
      gpu_compiler->GetAliasInfo(stream_executor_0->GetDeviceDescription());
  auto target_config =
      std::make_unique<Compiler::GpuTargetConfig>(stream_executor_0);

  std::unique_ptr<se::DeviceAddressAllocator> allocator =
      std::make_unique<stream_executor::StreamExecutorAddressAllocator>(
          stream_executor_0);

  auto mlir_context = std::make_unique<mlir::MLIRContext>();
  xla::RegisterSymbolicExprStorage(mlir_context.get());

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), "autotuner", tsl::port::MaxParallelism());

  std::vector<std::unique_ptr<Profiler>> autotuner_profilers;

  int device_count = platform->VisibleDeviceCount();
  autotuner_profilers.reserve(device_count);

  for (int i = 0; i < device_count; ++i) {
    ASSIGN_OR_RETURN(se::StreamExecutor * stream_executor,
                     platform->ExecutorForDevice(i));
    TF_RET_CHECK(stream_executor->GetDeviceDescription().name() ==
                 stream_executor_0->GetDeviceDescription().name())
        << "Devices are not the same: device 0 is "
        << stream_executor_0->GetDeviceDescription().name() << ", device " << i
        << " is " << stream_executor->GetDeviceDescription().name();
    auto profiler = GpuProfiler::Create(
        stream_executor, GetProfileOptions(debug_options, assigner_options));
    TF_RET_CHECK(profiler != nullptr)
        << "Failed to create profiler for device " << i;

    autotuner_profilers.push_back(std::move(profiler));
  }

  ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<CodegenBackend>> autotuner_backends,
      AutotunerPass::GetGpuAutotunerBackends(
          stream_executor_0, allocator.get(), target_config.get(),
          alias_info.get(), debug_options, mlir_context.get(),
          gpu_compiler->ShapeSizeBytesFunction(), gpu_compiler,
          platform->id()));

  ASSIGN_OR_RETURN(
      auto autotuner_orchestrator,
      CodegenOrchestrator::Create(std::move(autotuner_backends),
                                  orchestrator_options, thread_pool.get()));

  Autotuner::Options autotuner_options;
  autotuner_options.scratch_bytes_window_size_us =
      assigner_options.scratch_bytes_window_size_us;
  autotuner_options.correctness_check_options.enable_correctness_check =
      assigner_options.check_buffers;
  autotuner_options.correctness_check_options.relative_tolerance =
      assigner_options.relative_tolerance;
  autotuner_options.correctness_check_options.crash_on_failure =
      assigner_options.crash_on_check_failure;

  ASSIGN_OR_RETURN(auto autotuner,
                   Autotuner::Create(std::move(autotuner_orchestrator),
                                     std::move(autotuner_profilers),
                                     autotuner_options, thread_pool.get()));

  return AutotunerEnvironment{std::move(compiler),   std::move(mlir_context),
                              std::move(alias_info), std::move(target_config),
                              std::move(allocator),  std::move(thread_pool),
                              std::move(autotuner)};
}

}  // namespace

absl::Status RunAutotuning(const std::vector<std::string>& hlo_files,
                           const DebugOptions& debug_options) {
  ASSIGN_OR_RETURN(AutotunerEnvironment env,
                   CreateAutotunerEnvironment(debug_options));

  auto autotuner_cache = std::make_unique<PrintingAutotunerCache>();

  auto should_autotune = [](const xla::HloInstruction&) { return true; };

  for (const auto& hlo_file : hlo_files) {
    LOG(INFO) << "Autotuning " << hlo_file;
    ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module, GetModule(hlo_file));
    ASSIGN_OR_RETURN(
        std::vector<Autotuner::TuningResult> results,
        env.autotuner->TuneConfigs(*module, should_autotune,
                                   /*tolerate_no_supported_configs=*/true));
    for (const auto& result : results) {
      AutotunerCacheInterface::Config cached_config;
      cached_config.codegen_backend = result.config.codegen_backend->backend();
      cached_config.backend_config = *result.config.backend_config;
      RETURN_IF_ERROR(
          autotuner_cache->Insert(result.instruction, cached_config));
    }
  }
  return absl::OkStatus();
}

}  // namespace gpu
}  // namespace xla

int main(int argc, char* argv[]) {
  std::string hlo_files_str;
  std::vector<tsl::Flag> flag_list = {
      tsl::Flag("hlo_files", &hlo_files_str,
                "Comma-separated list of paths to the HLO files to autotune."),
  };

  const std::string usage_string =
      absl::StrCat(kUsage, "\n\n", tsl::Flags::Usage(argv[0], flag_list));
  bool parse_ok = tsl::Flags::Parse(&argc, argv, flag_list);
  if (!parse_ok) {
    LOG(QFATAL) << usage_string;
  }
  tsl::port::InitMain(usage_string.c_str(), &argc, &argv);

  std::vector<std::string> hlo_files = absl::StrSplit(hlo_files_str, ',');
  if (hlo_files.empty() || (hlo_files.size() == 1 && hlo_files[0].empty())) {
    LOG(QFATAL) << "No HLO files specified.";
  }

  xla::DebugOptions debug_options = xla::GetDebugOptionsFromFlags();
  absl::Status status = xla::gpu::RunAutotuning(hlo_files, debug_options);
  if (!status.ok()) {
    std::cerr << "Failed to autotune: " << status.ToString() << std::endl;
    return 1;
  }

  return 0;
}
