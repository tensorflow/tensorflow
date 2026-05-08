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

#include "xla/pjrt/device_event_utils.h"

#include <utility>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "absl/types/span.h"
#include "xla/pjrt/device_event.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/concurrency/ref_count.h"

namespace xla {

class ScopedLauncher::Callee
    : public tsl::ReferenceCounted<ScopedLauncher::Callee> {
 public:
  Callee(tsl::Executor::Task task, tsl::Executor* executor)
      : task_(std::move(task)), executor_(executor) {}
  ~Callee() {
    if (executor_) {
      executor_->Execute(std::move(task_));
    } else {
      std::move(task_)();
    }
  }
  tsl::Executor::Task task_;
  tsl::Executor* executor_ = nullptr;
};

ScopedLauncher::ScopedLauncher(tsl::Executor::Task task,
                               tsl::Executor* executor)
    : task_(std::move(task)), executor_(executor) {}

ScopedLauncher::~ScopedLauncher() {
  if (!state_) {
    if (executor_) {
      executor_->Execute(std::move(task_));
    } else {
      std::move(task_)();
    }
  }
}

void ScopedLauncher::AddDependency(PjRtDeviceEventPtr dependency) {
  if (!dependency) {
    return;
  }
  auto state = dependency.state();
  if (state == PJRT_DeviceEvent_State_Error ||
      state == PJRT_DeviceEvent_State_Ready) {
    return;
  }
  if (state_ == nullptr) {
    state_ = tsl::MakeRef<Callee>(std::move(task_), executor_);
  }
  dependency.DeleteWhenReady(state_);
}

void ScopedLauncher::AddDependency(tsl::AsyncValue* dependency) {
  if (!dependency) {
    return;
  }
  if (dependency->IsAvailable()) {
    return;
  }
  if (state_ == nullptr) {
    state_ = tsl::MakeRef<Callee>(std::move(task_), executor_);
  }
  dependency->AndThen([state = state_]() {});
}

absl::Status GetErrors(absl::Span<const PjRtDeviceEventRef> events) {
  absl::Status status;
  for (const auto& ev : events) {
    if (ev) {
      if (auto err = ev.GetErrorIfPresent()) {
        status.Update(*err);
      }
    }
  }
  return status;
}

void BlockUntilReady(PjRtDeviceEventPtr event) {
  DCHECK(event);
  auto state = event.state();
  if (ABSL_PREDICT_TRUE(state == PJRT_DeviceEvent_State_Ready)) {
    return;
  } else if (state == PJRT_DeviceEvent_State_Error) {
    return;
  }

  absl::Notification notification;
  event.AndThen([&notification] { notification.Notify(); });
  notification.WaitForNotification();
}

}  // namespace xla
