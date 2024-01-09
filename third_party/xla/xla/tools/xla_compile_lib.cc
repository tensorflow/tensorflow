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

#include "xla/tools/xla_compile_lib.h"

#include <cmath>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "google/protobuf/duration.pb.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_module_group.h"
#include "xla/service/compiler.h"
#include "xla/service/cpu/cpu_compiler.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/executable.h"
#include "xla/service/xla_compile_result.pb.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/util.h"
#include "tsl/platform/env.h"
#include "tsl/platform/env_time.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#include "xla/service/gpu/executable.pb.h"
#include "xla/stream_executor/gpu/gpu_init.h"
#endif
#if GOOGLE_CUDA
#include "xla/service/gpu/nvptx_compiler.h"
#elif TENSORFLOW_USE_ROCM
#include "xla/service/gpu/amdgpu_compiler.h"
#endif

namespace xla {

static StatusOr<std::string> AotCompileCpuExecutable(
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

static StatusOr<std::string> CompileGpuExecutable(
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

  *result.mutable_hlo_module() = module_after_opt->ToProto();
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
#else
  LOG(ERROR) << "Neither ROCm nor CUDA present; returning empty.";
  return "";
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

StatusOr<std::string> CompileExecutable(
    std::unique_ptr<HloModule> hlo_module, absl::string_view platform,
    std::optional<Compiler::TargetConfig> target_config,
    CompilationResult& result) {
  if (platform == "cpu") {
    return AotCompileCpuExecutable(std::move(hlo_module));
  } else if (platform == "gpu") {
    return CompileGpuExecutable(std::move(hlo_module), target_config, result);
  }

  return absl::UnimplementedError(
      absl::StrCat("platform", platform, " is not supported"));
}

Status WriteResultFile(const std::string& result_output_file, TimerStats& stats,
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

  return tsl::WriteBinaryProto(tsl::Env::Default(), result_output_file,
                               compilation_result);
}

}  // namespace xla
