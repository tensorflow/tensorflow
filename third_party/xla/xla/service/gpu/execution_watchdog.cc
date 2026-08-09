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

#include "xla/service/gpu/execution_watchdog.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "absl/base/nullability.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "xla/runtime/hang_watchdog.h"
#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/stream.h"
#include "xla/xla.pb.h"

namespace xla::gpu {

ExecutionWatchdogScope::ExecutionWatchdogScope(
    absl::Duration watchdog_timeout, std::string watchdog_name,
    const GpuExecutableRunOptions* gpu_run_options, se::Stream* stream,
    bool block_host_until_done)
    : watchdog_timeout_(watchdog_timeout),
      watchdog_name_(std::move(watchdog_name)),
      gpu_run_options_(gpu_run_options),
      stream_(stream),
      block_host_until_done_(block_host_until_done) {}

absl::StatusOr<std::optional<ExecutionWatchdogScope>>
ExecutionWatchdogScope::Create(
    const DebugOptions* absl_nullable debug_options,
    absl::string_view module_name, int32_t device_ordinal,
    const GpuExecutableRunOptions* absl_nullable gpu_run_options,
    se::Stream* absl_nullable stream, bool block_host_until_done) {
  absl::Duration watchdog_timeout = absl::InfiniteDuration();
  if (debug_options &&
      !debug_options->xla_gpu_execution_terminate_timeout().empty()) {
    TF_RET_CHECK(absl::ParseDuration(
        debug_options->xla_gpu_execution_terminate_timeout(),
        &watchdog_timeout))
        << "Failed to parse XLA execution terminate timeout";
  }

  if (watchdog_timeout >= absl::InfiniteDuration()) {
    return std::nullopt;
  }

  std::string watchdog_name = absl::StrFormat("[%d] XLA GPU execution `%s`",
                                              device_ordinal, module_name);
  return ExecutionWatchdogScope(watchdog_timeout, std::move(watchdog_name),
                                gpu_run_options, stream, block_host_until_done);
}

void ExecutionWatchdogScope::Arm(HangWatchdog::CancelCallback pre_abort) {
  if (armed_) {
    return;
  }
  armed_ = true;
  guard_holder_ = std::make_shared<std::shared_ptr<HangWatchdog::Guard>>();

  HangWatchdog::CancelCallback on_timeout;
  if (gpu_run_options_ && gpu_run_options_->execution_timeout_handler()) {
    // Capture a weak_ptr so the timeout callback does not keep the Guard alive
    // after this scope is destroyed (otherwise scope.reset() cannot cancel the
    // watchdog due to a shared_ptr cycle through the callback).
    std::weak_ptr<std::shared_ptr<HangWatchdog::Guard>> weak_guard_holder =
        guard_holder_;
    auto watchdog_name = watchdog_name_;
    auto watchdog_timeout = watchdog_timeout_;
    auto* gpu_run_options = gpu_run_options_;
    on_timeout = [watchdog_name, watchdog_timeout,
                  pre_abort = std::move(pre_abort), gpu_run_options,
                  weak_guard_holder]() mutable {
      if (pre_abort) {
        std::move(pre_abort)();
      }

      if (std::shared_ptr<std::shared_ptr<HangWatchdog::Guard>> guard_holder =
              weak_guard_holder.lock()) {
        *guard_holder = HangWatchdog::Global().Watch(
            "post-abort ...", absl::Minutes(1),
            HangWatchdog::Abort("post-abort ...", absl::Minutes(1)));
      }

      gpu_run_options->execution_timeout_handler()(watchdog_name,
                                                   watchdog_timeout);
    };
  } else {
    on_timeout = HangWatchdog::Abort(watchdog_name_, watchdog_timeout_,
                                     std::move(pre_abort));
  }

  *guard_holder_ = HangWatchdog::Global().Watch(
      watchdog_name_, watchdog_timeout_, std::move(on_timeout));
}

ExecutionWatchdogScope::~ExecutionWatchdogScope() {
  if (!armed_) {
    return;
  }

  // When using an async allocator, thunk dispatch returns before GPU work
  // completes. Keep the watchdog alive until the execution stream finishes.
  if (!block_host_until_done_ && stream_ != nullptr) {
    absl::Status block_status = stream_->BlockHostUntilDone();
    if (!block_status.ok()) {
      LOG(ERROR) << "Failed to sync execution stream before releasing "
                    "execution watchdog: "
                 << block_status;
    }
  }

  // Drop the HangWatchdog guard now that execution is done (or abandoned).
  if (guard_holder_ != nullptr) {
    *guard_holder_ = nullptr;
  }
}

}  // namespace xla::gpu
