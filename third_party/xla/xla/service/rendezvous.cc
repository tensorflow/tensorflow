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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "xla/tsl/platform/logging.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace internal {

// Waits for the rendezvous to be ready with a timeout. Returns true if the
// rendezvous is ready, false if the timeout is exceeded.
static bool WaitForReadyWithTimeout(RendezvousStateSynchronization& state,
                                    absl::Duration timeout) {
  absl::MutexLock lock(&state.mutex);

  // Keep checking if the rendezvous is ready inside a loop and update TraceMe
  // annotation to track the rendezvous progress.
  while (!state.ready) {
    size_t num_pending = state.num_threads - state.ack.load();

    tsl::profiler::TraceMe trace([&] {
      if (num_pending == 0) {
        return absl::StrFormat("Wait for rendezvous callback");
      } else {
        return absl::StrFormat("Wait %d of %d", num_pending, state.num_threads);
      }
    });

    bool timed_out = state.cv.WaitWithTimeout(&state.mutex, timeout);

    // We are done and ready.
    if (state.ready) return true;

    // We are done with waiting because the timeout is exceeded.
    if (timed_out && !state.ready) {
      return false;
    }

    // Otherwise we keep waiting.
  }

  return state.ready;
}

void AwaitAndLogIfStuck(RendezvousStateSynchronization& state, int32_t id,
                        absl::string_view name,
                        absl::Duration warn_stuck_timeout,
                        absl::Duration terminate_timeout) {
  // Wait for `warn_stuck_timeout` for the rendezvous to be ready.
  if (WaitForReadyWithTimeout(state, warn_stuck_timeout)) {
    return;
  }

  // If we are stuck, log a warning and add a trace annotation.
  tsl::profiler::TraceMe trace([&] {
    return absl::StrFormat("Stuck Waiting for %d of %d",
                           state.num_threads - state.ack, state.num_threads);
  });

  // Check if all rendezvous participants arrived to the rendezvous point and
  // incremented `ack` counter. We still can be stuck because the leader is
  // waiting for completion of rendezvous callback, but it must not be confused
  // with participants not arriving to the rendezvous point.
  bool is_all_participants_arrived = state.ack.load() == state.num_threads;

  if (is_all_participants_arrived) {
    LOG(ERROR) << absl::StreamFormat(
        "[id=%d] This thread has been waiting for `%s` for %d "
        "seconds and may be stuck. All %d threads joined the rendezvous, "
        "however the leader has not marked the rendezvous as completed. Leader "
        "can be deadlocked inside the rendezvous callback.",
        id, name, absl::ToInt64Seconds(warn_stuck_timeout), state.num_threads);

  } else {
    LOG(ERROR) << absl::StreamFormat(
        "[id=%d] This thread has been waiting for `%s` for %d seconds and may "
        "be stuck. Expected %d threads to join the rendezvous, but not all of "
        "them arrived on time.",
        id, name, absl::ToInt64Seconds(warn_stuck_timeout), state.num_threads);
  }

  // Wait for `terminate_timeout` for the rendezvous to be ready before killing
  // the process.
  if (WaitForReadyWithTimeout(state, terminate_timeout)) {
    LOG(ERROR) << "Thread is unstuck! Warning above was a false-positive. "
                  "Perhaps the timeout is too short.";
    return;
  }

  // Check again if all participants arrived to the rendezvous point.
  is_all_participants_arrived = state.ack.load() == state.num_threads;

  if (is_all_participants_arrived) {
    LOG(FATAL) << absl::StreamFormat(
        "[id=%d] Termination timeout for `%s` of %d seconds exceeded. Exiting "
        "to ensure a consistent program state. All %d threads joined the "
        "rendezvous, however the leader has not marked the rendezvous as "
        "completed. Leader can be deadlocked inside the rendezvous callback.",
        id, name, absl::ToInt64Seconds(terminate_timeout), state.num_threads);

  } else {
    LOG(FATAL) << absl::StreamFormat(
        "[id=%d] Termination timeout for `%s` of %d seconds exceeded. Exiting "
        "to ensure a consistent program state. Expected %d threads to join the "
        "rendezvous, but not all of them arrived on time.",
        id, name, absl::ToInt64Seconds(terminate_timeout), state.num_threads);
  }
}

}  // namespace internal

namespace {
inline constexpr int32_t kPending = 0;
inline constexpr int32_t kCompleted = std::numeric_limits<int32_t>::max();
}  // namespace

RendezvousFlag::RendezvousFlag() : state_(kPending) {}

RendezvousFlag::InFlightRendezvous::InFlightRendezvous(RendezvousFlag* flag)
    : flag_(flag) {}

RendezvousFlag::InFlightRendezvous::~InFlightRendezvous() {
  if (flag_ == nullptr) return;

  // Reload state and use CAS to decide if we are the one who
  // should mark rendezvous flag completed.
  int32_t state = flag_->state_.load();

  CHECK(state != kPending && state != kCompleted)  // NOLINT
      << "rendezvous can't be in pending or completed state";

  // Exit the critical section and maybe mark rendezvous as completed.
  while (!flag_->state_.compare_exchange_weak(
      state, state == 1 ? kCompleted : state - 1)) {
    // Check state after CAS failure: while we are in this function no one
    // should complete rendezvous without us or switch it back to pending.
    CHECK(state != kPending && state != kCompleted);  // NOLINT
  }
}

RendezvousFlag::InFlightRendezvous::operator bool() const {
  return flag_ != nullptr;
}

RendezvousFlag::InFlightRendezvous RendezvousFlag::TryJoin() {
  // If `state_` is `kCompleted` it means that we have at least one completed
  // rendezvous for this flag and can skip it.
  if (state_.load() == kCompleted) return InFlightRendezvous(nullptr);

  // Try to increment a state in a CAS loop to signal all other participants
  // that we joined an in-flight rendezvous.
  int32_t state = state_.load();
  while (state != kCompleted &&
         !state_.compare_exchange_weak(state, state + 1)) {
  }

  // Someone else completed the rendezvous and we don't need to join.
  if (state == kCompleted) return InFlightRendezvous(nullptr);

  return InFlightRendezvous(this);
}

bool RendezvousFlag::IsCompleted() const { return state_.load() == kCompleted; }

}  // namespace xla
