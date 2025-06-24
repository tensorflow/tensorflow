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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_COMMAND_BUFFER_CONVERSION_PASS_H_
#define XLA_SERVICE_GPU_TRANSFORMS_COMMAND_BUFFER_CONVERSION_PASS_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/backends/gpu/runtime/sequential_thunk.h"
#include "xla/service/gpu/transforms/thunk_pass_pipeline.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Converts compatible sequences of Thunks into CommandBufferThunks.
class CommandBufferConversionPass : public ThunkPassInterface {
 public:
  CommandBufferConversionPass() = default;

  absl::string_view name() const override {
    return "command-buffer-conversion";
  }

  absl::StatusOr<bool> Run(SequentialThunk* root_thunk,
                           const DebugOptions& debug_options,
                           const se::DeviceDescription& device_info) override;
  struct CommandBufferConfig {
    // DebugOptions control which commands are enabled. Long term we want to
    // remove that flag and enable all supported commands by default.
    absl::flat_hash_set<DebugOptions::CommandBufferCmdType> enabled_commands;
    absl::flat_hash_set<std::string> enabled_legacy_custom_call_targets;
    const se::DeviceDescription& device_description;
  };
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_COMMAND_BUFFER_CONVERSION_PASS_H_
