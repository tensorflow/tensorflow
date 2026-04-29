/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_EARLY_EXIT_COMPILATION_RESULT_H_
#define XLA_SERVICE_GPU_EARLY_EXIT_COMPILATION_RESULT_H_

#include <memory>
#include <string>
#include <utility>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/pjrt/compiled_memory_stats.h"
#include "xla/service/compiled_module.h"
#include "xla/service/executable.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/platform.h"

namespace xla {
namespace gpu {

class EarlyExitCompilationResult : public CompiledModule {
 public:
  explicit EarlyExitCompilationResult(std::unique_ptr<HloModule> module)
      : module_(std::move(module)) {}

  absl::StatusOr<std::string> SerializeAsString() const override;

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable() && final {
    return absl::UnimplementedError(
        "LoadExecutable without parameters not supported");
  }

  absl::StatusOr<std::unique_ptr<Executable>> LoadExecutable(
      se::Platform::Id platform_id,
      const se::DeviceDescription& device_description) &&
      override;

  const HloModule* optimized_module() const override { return module_.get(); }
  std::shared_ptr<HloModule> shared_optimized_module() override {
    return module_;
  }

  absl::StatusOr<CompiledMemoryStats> GetCompiledMemoryStats() const override;

 private:
  std::shared_ptr<HloModule> module_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_EARLY_EXIT_COMPILATION_RESULT_H_
