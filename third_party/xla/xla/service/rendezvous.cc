/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/rendezvous.h"

#include <cstdlib>
#include <string_view>

#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tsl/platform/logging.h"

namespace xla::internal {

void AwaitAndLogIfStuck(absl::Notification& ready, std::string_view name,
                        absl::Duration warn_stuck_timeout,
                        absl::Duration terminate_timeout) {
  if (ready.WaitForNotificationWithTimeout(warn_stuck_timeout)) {
    return;
  }

  LOG(ERROR) << "This thread has been waiting for `" << name << "` for "
             << absl::ToInt64Seconds(warn_stuck_timeout)
             << " seconds and may be stuck.";

  if (ready.WaitForNotificationWithTimeout(terminate_timeout)) {
    LOG(ERROR) << "Thread is unstuck! Warning above was a false-positive. "
                  "Perhaps the timeout is too short.";
    return;
  }

  LOG(ERROR)
      << "Termination timeout for `" << name << "` of "
      << absl::ToInt64Seconds(terminate_timeout)
      << " seconds exceeded. Exiting to ensure a consistent program state.";
  std::exit(42);
}

}  // namespace xla::internal
