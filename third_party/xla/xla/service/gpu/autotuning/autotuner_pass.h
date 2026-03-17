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

#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/autotuner/autotuner.h"
#include "xla/backends/autotuner/codegen_backend.h"
#include "xla/backends/autotuner/profiler.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/service/compiler.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

AutotuneConfig GetAutotuneConfig(const DebugOptions& debug_options,
                                 bool is_deviceless = false,
                                 bool optimize_scratch_bytes = true,
                                 bool allow_reg_spills = true);

ProfileOptions GetProfileOptions(const DebugOptions& debug_options,
                                 const AutotuneConfig& autotune_config);

// HloModulePass that runs the autotuner.
class AutotunerPass : public HloModulePass {
 public:
  // Note: the target_config must outlive the pass.
  static absl::StatusOr<std::unique_ptr<AutotunerPass>> Create(
      std::vector<std::unique_ptr<CodegenBackend>> backends,
      const DebugOptions& debug_options, se::StreamExecutor* stream_executor,
      tsl::thread::ThreadPool* thread_pool, InstructionFilterFn should_autotune,
      const Compiler::GpuTargetConfig* target_config,
      se::DeviceAddressAllocator* allocator = nullptr,
      bool optimize_scratch_bytes = true,
      MultiProcessKeyValueStore key_value_store = MultiProcessKeyValueStore(),
      bool allow_reg_spills = false);

  absl::string_view name() const override { return "autotuner"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  explicit AutotunerPass(std::unique_ptr<Autotuner> autotuner,
                         InstructionFilterFn should_autotune,
                         MultiProcessKeyValueStore key_value_store,
                         bool enable_sharding)
      : autotuner_(std::move(autotuner)),
        should_autotune_(std::move(should_autotune)),
        key_value_store_(std::move(key_value_store)),
        enable_sharding_(enable_sharding) {}

  std::unique_ptr<Autotuner> autotuner_;
  InstructionFilterFn should_autotune_;
  MultiProcessKeyValueStore key_value_store_;
  bool enable_sharding_ = false;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_AUTOTUNING_AUTOTUNER_PASS_H_
