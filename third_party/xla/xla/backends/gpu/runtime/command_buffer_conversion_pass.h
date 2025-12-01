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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CONVERSION_PASS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CONVERSION_PASS_H_

#include <string>

#include "absl/base/nullability.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/backends/gpu/runtime/thunk_pass_pipeline.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/stream_executor/device_description.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Converts compatible sequences of Thunks into CommandBufferThunks.
class CommandBufferConversionPass : public ThunkPassInterface {
 public:
  explicit CommandBufferConversionPass(absl::string_view module_name = "")
      : module_name_(module_name) {}

  absl::string_view name() const override {
    return "command-buffer-conversion";
  }

  absl::StatusOr<bool> Run(SequentialThunk* root_thunk,
                           const DebugOptions& debug_options,
                           const HloModule* absl_nullable hlo_module,
                           const se::DeviceDescription& device_info,
                           ThunkPassBufferAllocator& allocator) override;
  struct CommandBufferConfig {
    // DebugOptions control which commands are enabled. Long term we want to
    // remove that flag and enable all supported commands by default.
    absl::flat_hash_set<DebugOptions::CommandBufferCmdType> enabled_commands;
    const se::DeviceDescription& device_description;
    std::string ToString() const;
  };

 private:
  std::string module_name_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_CONVERSION_PASS_H_
