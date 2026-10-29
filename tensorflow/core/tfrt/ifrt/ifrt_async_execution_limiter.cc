/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/tfrt/ifrt/ifrt_async_execution_limiter.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/device.h"
#include "xla/tsl/concurrency/executor.h"
#include "xla/tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {

struct IfrtAsyncExecutionLimiter::PendingAcquisition {
  PendingAcquisition(IfrtAsyncExecutionLimiter* limiter,
                     tsl::thread::ThreadPool* thread_pool,
                     absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids,
                     size_t acquired_idx,
                     // Callback to invoke once permit(s) is granted!
                     absl::AnyInvocable<void(Permit) &&> callback)
      : limiter(limiter),
        thread_pool(thread_pool),
        device_ids(std::move(device_ids)),
        acquired_idx(acquired_idx),
        callback(std::move(callback)) {}

  ~PendingAcquisition() {
    if (!dispatched && acquired_idx > 0) {
      // Release the permits that were acquired but not yet started.
      limiter->Release(absl::MakeSpan(device_ids).subspan(0, acquired_idx));
    }
  }

  IfrtAsyncExecutionLimiter* limiter;
  tsl::thread::ThreadPool* thread_pool;
  absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids;
  size_t acquired_idx;
  bool dispatched = false;
  absl::AnyInvocable<void(Permit) &&> callback;
};

IfrtAsyncExecutionLimiter::IfrtAsyncExecutionLimiter(
    int max_inflight_per_device)
    : max_inflight_per_device_(std::max(1, max_inflight_per_device)) {
  if (max_inflight_per_device <= 0) {
    LOG(ERROR) << "IfrtAsyncExecutionLimiter requires max_inflight_per_device "
                  "> 0, got "
               << max_inflight_per_device << "; clamping to 1.";
  }
  DCHECK_GT(max_inflight_per_device, 0);
}

absl::InlinedVector<xla::ifrt::DeviceId, 4>
IfrtAsyncExecutionLimiter::CanonicalizeDeviceIds(
    absl::Span<const xla::ifrt::DeviceId> raw_device_ids) {
  // Sort and deduplicate devices.
  absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids(raw_device_ids.begin(),
                                                         raw_device_ids.end());
  if (device_ids.size() > 1) {
    absl::c_sort(device_ids);
    device_ids.erase(std::unique(device_ids.begin(), device_ids.end()),
                     device_ids.end());
  }
  return device_ids;
}

std::optional<IfrtAsyncExecutionLimiter::Permit>
IfrtAsyncExecutionLimiter::TryAcquire(
    absl::Span<const xla::ifrt::DeviceId> raw_device_ids) {
  absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids =
      CanonicalizeDeviceIds(raw_device_ids);
  if (device_ids.empty()) {
    // Defensive guard. Should not happen.
    return Permit();
  }

  absl::MutexLock lock(mu_);
  for (const xla::ifrt::DeviceId device_id : device_ids) {
    const auto it = queues_.find(device_id);
    if (it != queues_.end() &&
        (it->second.inflight >= max_inflight_per_device_ ||
         !it->second.waiters.empty())) {
      // Cannot acquire immediately.
      return std::nullopt;
    }
  }
  // Acquire the permits.
  for (const xla::ifrt::DeviceId device_id : device_ids) {
    ++queues_[device_id].inflight;
  }
  return Permit(this, std::move(device_ids));
}

void IfrtAsyncExecutionLimiter::RunWhenPermitted(
    absl::Span<const xla::ifrt::DeviceId> raw_device_ids,
    tsl::thread::ThreadPool& thread_pool,
    absl::AnyInvocable<void(Permit) &&> callback) {
  absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids =
      CanonicalizeDeviceIds(raw_device_ids);
  if (device_ids.empty()) {
    // Defensive guard. Should not happen.
    std::move(callback)(Permit());
    return;
  }

  {
    absl::MutexLock lock(mu_);
    size_t acquired_idx = 0;
    while (acquired_idx < device_ids.size()) {
      DeviceQueue& q = queues_[device_ids[acquired_idx]];
      // Grab permits for all devices with no waiters and available slots. For
      // a single chip, this loop iteration runs just once.
      if (q.inflight < max_inflight_per_device_ && q.waiters.empty()) {
        ++q.inflight;
        ++acquired_idx;
      } else {
        // Stop at the first contended (busy) device.
        break;
      }
    }
    if (acquired_idx < device_ids.size()) {
      // Permits for some devices were not acquired.
      auto state = std::make_shared<PendingAcquisition>(
          this, &thread_pool, std::move(device_ids), acquired_idx,
          std::move(callback));
      // Thus, join the wait for a permit.
      EnqueueCurrentDeviceLocked(std::move(state));
      return;
    }
  }

  // Permit(s) acquired. Invoke the callback.
  std::move(callback)(Permit(this, std::move(device_ids)));
}

void IfrtAsyncExecutionLimiter::EnqueueCurrentDeviceLocked(
    std::shared_ptr<PendingAcquisition> state) {
  const xla::ifrt::DeviceId target_device =
      state->device_ids[state->acquired_idx];
  // Join the wait.
  queues_[target_device].waiters.push_back(
      [this, state = std::move(state)]() mutable {
        // Upon exiting queue, this callback is invoked to acquire the permit.
        AdvancePendingAcquisition(std::move(state));
      });
}

void IfrtAsyncExecutionLimiter::AdvancePendingAcquisition(
    std::shared_ptr<PendingAcquisition> state) {
  // Ownership of the inflight slot for `device_ids[acquired_idx]` was directly
  // transferred from `Release()` to this waiter to preserve strict FIFO order.
  ++state->acquired_idx;

  if (state->acquired_idx < state->device_ids.size()) {
    // Multi-device case: We just got one device's permit, so now try to
    // acquire permits for the remaining devices (or wait on the next busy
    // device). Single-device requests skip this block (`1 < 1` is false).
    absl::MutexLock lock(mu_);
    while (state->acquired_idx < state->device_ids.size()) {
      DeviceQueue& q = queues_[state->device_ids[state->acquired_idx]];
      if (q.inflight < max_inflight_per_device_ && q.waiters.empty()) {
        ++q.inflight;
        ++state->acquired_idx;
      } else {
        EnqueueCurrentDeviceLocked(std::move(state));
        return;
      }
    }
  }

  // Permit(s) acquired. Dispatch the callback.
  tsl::Executor* executor = state->thread_pool->AsExecutor();
  executor->Execute([state = std::move(state)]() mutable {
    state->dispatched = true;
    Permit permit(state->limiter, std::move(state->device_ids));
    state->device_ids.clear();
    std::move(state->callback)(std::move(permit));
  });
}

void IfrtAsyncExecutionLimiter::Release(
    absl::Span<const xla::ifrt::DeviceId> device_ids) {
  absl::InlinedVector<absl::AnyInvocable<void() &&>, 4> ready_waiters;
  {
    absl::MutexLock lock(mu_);
    for (const xla::ifrt::DeviceId device_id : device_ids) {
      DeviceQueue& q = queues_[device_id];
      DCHECK_GT(q.inflight, 0);
      if (!q.waiters.empty()) {
        // Keep `q.inflight` unchanged: transfer the slot directly to the front
        // FIFO waiter so new arrivals cannot barge in via `TryAcquire`.
        ready_waiters.push_back(std::move(q.waiters.front()));
        q.waiters.pop_front();
      } else {
        --q.inflight;
      }
    }
  }

  for (auto& waiter : ready_waiters) {
    std::move(waiter)();
  }
}

int IfrtAsyncExecutionLimiter::InflightForDevice(
    xla::ifrt::DeviceId device_id) const {
  absl::MutexLock lock(mu_);
  const auto it = queues_.find(device_id);
  return it == queues_.end() ? 0 : it->second.inflight;
}

size_t IfrtAsyncExecutionLimiter::QueueDepthForDevice(
    xla::ifrt::DeviceId device_id) const {
  absl::MutexLock lock(mu_);
  const auto it = queues_.find(device_id);
  return it == queues_.end() ? 0 : it->second.waiters.size();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
