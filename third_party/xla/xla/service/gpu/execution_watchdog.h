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

#ifndef XLA_SERVICE_GPU_EXECUTION_WATCHDOG_H_
#define XLA_SERVICE_GPU_EXECUTION_WATCHDOG_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

#include "absl/base/nullability.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/runtime/hang_watchdog.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

// RAII scope that owns a HangWatchdog guard for the lifetime of a GPU
// execution. The guard is armed when thunk dispatch begins (after optional
// progress tracking is installed) and is released only after GPU work
// completes.
//
// For async allocators (block_host_until_done=false), the destructor blocks on
// the execution stream before releasing the guard so that the watchdog remains
// valid after thunk enqueue returns.
class ExecutionWatchdogScope {
 public:
  // Returns std::nullopt when execution terminate timeout is not configured.
  static absl::StatusOr<std::optional<ExecutionWatchdogScope>> Create(
      const DebugOptions* absl_nullable debug_options,
      absl::string_view module_name, int32_t device_ordinal,
      const GpuExecutableRunOptions* absl_nullable gpu_run_options,
      se::Stream* absl_nullable stream, bool block_host_until_done);

  ExecutionWatchdogScope(ExecutionWatchdogScope&&) = default;
  ExecutionWatchdogScope& operator=(ExecutionWatchdogScope&&) = default;
  ExecutionWatchdogScope(const ExecutionWatchdogScope&) = delete;
  ExecutionWatchdogScope& operator=(const ExecutionWatchdogScope&) = delete;

  ~ExecutionWatchdogScope();

  // Arms the watchdog. Must be called at most once per scope, typically from
  // GpuExecutable::ExecuteThunksImpl once progress tracking is set up.
  void Arm(HangWatchdog::CancelCallback pre_abort = nullptr);

  bool IsArmed() const { return armed_; }

 private:
  ExecutionWatchdogScope(absl::Duration watchdog_timeout,
                         std::string watchdog_name,
                         const GpuExecutableRunOptions* gpu_run_options,
                         se::Stream* stream, bool block_host_until_done);

  absl::Duration watchdog_timeout_;
  std::string watchdog_name_;
  const GpuExecutableRunOptions* gpu_run_options_ = nullptr;
  se::Stream* stream_ = nullptr;
  bool block_host_until_done_ = false;
  bool armed_ = false;

  std::shared_ptr<std::shared_ptr<HangWatchdog::Guard>> guard_holder_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_EXECUTION_WATCHDOG_H_
