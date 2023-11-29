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

#include <optional>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "xla/debug_options_flags.h"
#include "xla/service/compiler.h"
#include "xla/service/dump.h"
#include "xla/service/gpu/executable.pb.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/platform_util.h"
#include "xla/statusor.h"
#include "xla/stream_executor/cuda/cuda_platform_id.h"
#include "xla/stream_executor/platform/initialize.h"
#include "xla/tools/hlo_opt/opt_lib.h"
#include "xla/types.h"

namespace xla {

namespace {

// TODO(cheshire): Switch CUDA/ROCM
static auto kGpuPlatformId = se::cuda::kCudaPlatformId;

static StatusOr<std::unique_ptr<Executable>> ToGpuExecutable(
    std::unique_ptr<HloModule> module, Compiler* compiler,
    se::StreamExecutor* executor, const Compiler::CompileOptions& opts) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<HloModule> optimized_module,
      compiler->RunHloPasses(std::move(module), executor, opts));
  DebugOptions d = optimized_module->config().debug_options();
  d.set_xla_embed_ir_in_executable(true);
  optimized_module->mutable_config().set_debug_options(d);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<Executable> executable,
      compiler->RunBackend(std::move(optimized_module), executor, opts));
  return executable;
}

struct GpuOptProvider : public OptProvider {
  StatusOr<std::optional<std::string>> GenerateStage(
      std::unique_ptr<HloModule> module, absl::string_view s) override {
    TF_ASSIGN_OR_RETURN(
        se::Platform * platform,
        se::MultiPlatformManager::PlatformWithId(kGpuPlatformId));

    TF_ASSIGN_OR_RETURN(Compiler * compiler,
                        Compiler::GetForPlatform(platform));
    DebugOptions debug_opts = GetDebugOptionsFromFlags();

    Compiler::CompileOptions opts;

    se::StreamExecutor* executor = nullptr;
    if (debug_opts.xla_gpu_target_config_filename().empty()) {
      TF_ASSIGN_OR_RETURN(std::vector<se::StreamExecutor*> stream_executors,
                          PlatformUtil::GetStreamExecutors(
                              platform, /*allowed_devices=*/std::nullopt));
      executor = stream_executors[0];
    }

    if (s == "hlo") {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<HloModule> optimized_module,
          compiler->RunHloPasses(std::move(module), executor, opts));
      return optimized_module->ToString();
    } else if (s == "llvm") {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<Executable> executable,
          ToGpuExecutable(std::move(module), compiler, executor, opts));
      return static_cast<gpu::GpuExecutable*>(executable.get())
          ->ir_module_string();
    } else if (s == "ptx") {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<Executable> executable,
          ToGpuExecutable(std::move(module), compiler, executor, opts));
      return static_cast<gpu::GpuExecutable*>(executable.get())->text();
    } else if (s == "buffer-assignment") {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<Executable> executable,
          ToGpuExecutable(std::move(module), compiler, executor, opts));
      return static_cast<gpu::GpuExecutable*>(executable.get())
          ->buffer_assignment()
          ->ToVerboseString(9999);
    } else if (s == "html") {
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<HloModule> optimized_module,
          compiler->RunHloPasses(std::move(module), executor, opts));
      return RenderGraph(optimized_module->name(), *optimized_module,
                         RenderedGraphFormat::kHtml,
                         /*show_fusion_subcomputations=*/false);
    }

    // Unimplemented stage.
    return std::nullopt;
  }

  std::vector<std::string> SupportedStages() override {
    return {"hlo", "llvm", "ptx", "buffer-assignment", "html"};
  }
};

}  // namespace
}  // namespace xla

REGISTER_MODULE_INITIALIZER(gpu_opt_provider, {
  xla::OptProvider::RegisterForPlatform(
      xla::kGpuPlatformId, std::make_unique<xla::GpuOptProvider>());
});
