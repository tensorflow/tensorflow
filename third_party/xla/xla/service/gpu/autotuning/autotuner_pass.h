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

#ifndef XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_PASS_H_
#define XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_PASS_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

class AutotunerPass : public HloModulePass {
 public:
  static absl::StatusOr<std::unique_ptr<AutotunerPass>> Create(
      std::vector<std::unique_ptr<CodegenBackend>> backends,
      const DebugOptions& debug_options, se::DeviceMemoryAllocator* allocator,
      se::StreamExecutor* stream_executor, tsl::thread::ThreadPool* thread_pool,
      InstructionFilterFn should_autotune);

  absl::string_view name() const override { return "autotuner"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  explicit AutotunerPass(std::unique_ptr<Autotuner> autotuner,
                         InstructionFilterFn should_autotune)
      : autotuner_(std::move(autotuner)), should_autotune_(should_autotune) {}

  std::unique_ptr<Autotuner> autotuner_;
  InstructionFilterFn should_autotune_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_PASS_H_
