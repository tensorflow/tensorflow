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

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"

namespace xla::gpu {

class Command;

// Collects command buffer execution requirements requested dynamically by
// thunks (including FFI prepare handlers) during the prepare stage.
class CommandBufferRequests {
 public:
  // Scopes the active command during synchronous FFI prepare execution so
  // `SetRequiresWarmup(bool)` knows which command is making the request.
  class ScopedCommand {
   public:
    ScopedCommand(CommandBufferRequests& requests, const Command* command)
        : requests_(requests), prev_(requests.current_command_) {
      requests_.current_command_ = command;
    }
    ~ScopedCommand() { requests_.current_command_ = prev_; }

    ScopedCommand(const ScopedCommand&) = delete;
    ScopedCommand& operator=(const ScopedCommand&) = delete;

   private:
    CommandBufferRequests& requests_;
    const Command* prev_;
  };

  // Sets whether the command requires an unrecorded warmup execution of the
  // enclosing command buffer before it is recorded for the first time on the
  // target device (e.g., true when a kernel needs to run autotuning on real
  // device buffers before graph capture, or false when the kernel is ready to
  // be recorded directly).
  void SetRequiresWarmup(bool requires_warmup) {
    DCHECK(current_command_ != nullptr);
    SetRequiresWarmup(current_command_, requires_warmup);
  }

  void SetRequiresWarmup(const Command* command, bool requires_warmup) {
    DCHECK(command != nullptr);
    command_requests_[command].requires_warmup = requires_warmup;
  }

  std::optional<bool> requires_warmup(const Command* command) const {
    if (auto it = command_requests_.find(command);
        it != command_requests_.end()) {
      return it->second.requires_warmup;
    }
    return std::nullopt;
  }

 private:
  struct Request {
    std::optional<bool> requires_warmup;
  };

  const Command* current_command_ = nullptr;
  absl::flat_hash_map<const Command*, Request> command_requests_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_RUNTIME_COMMAND_BUFFER_REQUESTS_H_
