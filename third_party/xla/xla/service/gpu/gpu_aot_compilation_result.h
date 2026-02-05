/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_GPU_AOT_COMPILATION_RESULT_H_
#define XLA_SERVICE_GPU_GPU_AOT_COMPILATION_RESULT_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "riegeli/bytes/string_writer.h"
#include "xla/debug_options_flags.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiled_module.h"
#include "xla/service/executable.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_executable.pb.h"
#include "xla/stream_executor/kernel_symbol_registry.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util/split_proto/split_gpu_executable_writer.h"

namespace xla::gpu {

// `AotCompilationResult` implementation for GPU, containing a serialized
// `GpuExecutable`.
//
// Unlike `LegacyGpuAotCompilationResult`, this result contains the entire
// optimized executable, including the Thunks, as opposed to just the optimized
// HLO.
class GpuAotCompilationResult : public CompiledModule {
 public:
  static absl::StatusOr<std::unique_ptr<GpuAotCompilationResult>> FromProto(
      GpuExecutableProto executable) {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<HloModule> module,
                        HloModule::CreateFromProtoWithConfig(
                            executable.hlo_module_with_config()));

    return absl::WrapUnique(
        new GpuAotCompilationResult(std::move(executable), std::move(module)));
  }

  absl::StatusOr<std::string> SerializeAsString() const final {
    std::string serialized;
    TF_RETURN_IF_ERROR(WriteSplitGpuExecutable(
        executable_, std::make_unique<riegeli::StringWriter<>>(&serialized)));
    return serialized;
  }

  absl::StatusOr<std::unique_ptr<Executable>>
      LoadExecutable(const se::StreamExecutor* stream_exec) && final {
    stream_executor::Platform::Id platform_id =
        stream_exec->GetPlatform()->id();
    const auto symbol_resolver = [&](absl::string_view symbol_name) {
      stream_executor::KernelSymbolRegistry& registry =
          stream_executor::KernelSymbolRegistry::GetGlobalInstance();
      return registry.FindSymbol(symbol_name, platform_id);
    };
    return GpuExecutable::FromProto(
        executable_, stream_exec->GetDeviceDescription(),
        stream_exec->GetPlatform()->Name(), GetDebugOptionsFromFlags(),
        symbol_resolver);
  }

  const HloModule* optimized_module() const final { return hlo_module_.get(); };

  std::shared_ptr<HloModule> shared_optimized_module() final {
    return hlo_module_;
  };

 private:
  explicit GpuAotCompilationResult(GpuExecutableProto executable,
                                   std::unique_ptr<HloModule> hlo_module)
      : executable_(std::move(executable)),
        hlo_module_(std::move(hlo_module)) {}

  GpuExecutableProto executable_;
  std::shared_ptr<HloModule> hlo_module_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_GPU_AOT_COMPILATION_RESULT_H_
