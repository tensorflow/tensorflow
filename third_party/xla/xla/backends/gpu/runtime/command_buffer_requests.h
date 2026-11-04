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

#ifndef XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_REQUESTS_H_
#define XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_REQUESTS_H_

#include <optional>

#include "absl/container/node_hash_map.h"

namespace xla::gpu {

class Command;

// Collects command buffer execution requirements requested dynamically by
// thunks (including FFI prepare handlers) during the prepare stage.
class CommandBufferRequests {
 public:
  // Sets whether the command requires an unrecorded warmup execution of the
  // enclosing command buffer before it is recorded for the first time on the
  // target device (e.g., true when a kernel needs to run autotuning on real
  // device buffers before graph capture, or false when the kernel is ready to
  // be recorded directly).
  void SetRequiresWarmup(bool requires_warmup) {
    requires_warmup_ = requires_warmup;
  }
  std::optional<bool> requires_warmup() const { return requires_warmup_; }

  CommandBufferRequests& ForCommand(const Command* command) {
    return command_requests_[command];
  }

  std::optional<bool> requires_warmup(const Command* command) const {
    if (auto it = command_requests_.find(command);
        it != command_requests_.end()) {
      return it->second.requires_warmup();
    }
    return requires_warmup_;
  }

 private:
  std::optional<bool> requires_warmup_;
  absl::node_hash_map<const Command*, CommandBufferRequests> command_requests_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_REQUESTS_H_
