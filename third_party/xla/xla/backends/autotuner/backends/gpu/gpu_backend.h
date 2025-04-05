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

#ifndef XLA_BACKENDS_AUTOTUNER_BACKENDS_GPU_GPU_BACKEND_H_
#define XLA_BACKENDS_AUTOTUNER_BACKENDS_GPU_GPU_BACKEND_H_

#include <memory>
#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/backend.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/compiler.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

class GpuBackend : public Backend {
 public:
  GpuBackend(absl::string_view name,
             const Compiler::TargetConfig& target_config,
             se::StreamExecutor* stream_executor)
      : Backend(name, target_config), stream_executor_(stream_executor) {
    auto compiler = Compiler::GetForPlatform(stream_executor_->GetPlatform());
    TF_CHECK_OK(compiler.status());
    compiler_ = std::move(*compiler);
  }

  absl::StatusOr<std::unique_ptr<Executable>> Compile(
      const HloInstruction& hlo_instruction,
      const BackendConfig& config) override {
    TF_ASSIGN_OR_RETURN(auto hlo_module, WrapInModule(hlo_instruction, config));

    Compiler::CompileOptions options;
    options.target_config = target_config_;
    options.is_autotuning_compilation = true;

    TF_ASSIGN_OR_RETURN(auto optimized_module,
                        RunHloPasses(std::move(hlo_module), options));
    return compiler_->RunBackend(std::move(optimized_module), stream_executor_,
                                 options);
  }

 protected:
  // TODO Provide a default implementation.
  virtual absl::StatusOr<std::unique_ptr<HloModule>> WrapInModule(
      const HloInstruction& hlo_instruction, const BackendConfig& config) = 0;

  // Optimize the HLO module.
  // TODO Remove this when XLA pipelines is fixed and we autotune only optimized
  // and fused HLOs.
  virtual absl::StatusOr<std::unique_ptr<HloModule>> RunHloPasses(
      std::unique_ptr<HloModule> hlo_module,
      const Compiler::CompileOptions& options) = 0;

  // TODO remove stream executor and compiler when backend can produce
  // executable without it.
  se::StreamExecutor* stream_executor_;
  std::unique_ptr<Compiler> compiler_;
};

}  // namespace xla

#endif  // XLA_BACKENDS_AUTOTUNER_BACKENDS_GPU_GPU_BACKEND_H_
