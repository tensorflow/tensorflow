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
#include <cstdint>
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
                     int64_t bytes_per_device, size_t acquired_idx,
                     absl::AnyInvocable<void(Permit) &&> callback)
      : limiter(limiter),
        thread_pool(thread_pool),
        device_ids(std::move(device_ids)),
        bytes_per_device(bytes_per_device),
        acquired_idx(acquired_idx),
        callback(std::move(callback)) {}

  ~PendingAcquisition() {
    if (!dispatched && acquired_idx > 0) {
      // Release the permits that were acquired before being abandoned.
      limiter->Release(absl::MakeSpan(device_ids).subspan(0, acquired_idx),
                       bytes_per_device);
    }
  }

  IfrtAsyncExecutionLimiter* limiter;
  tsl::thread::ThreadPool* thread_pool;
  absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids;
  int64_t bytes_per_device;
  size_t acquired_idx;
  bool dispatched = false;
  absl::AnyInvocable<void(Permit) &&> callback;
};

IfrtAsyncExecutionLimiter::IfrtAsyncExecutionLimiter(
    int64_t max_bytes_per_device)
    : max_bytes_per_device_(std::max<int64_t>(1, max_bytes_per_device)) {
  if (max_bytes_per_device <= 0) {
    LOG(ERROR) << "IfrtAsyncExecutionLimiter requires max_bytes_per_device > "
                  "0, got "
               << max_bytes_per_device << "; clamping to 1.";
  }
  DCHECK_GT(max_bytes_per_device, 0);
}

bool IfrtAsyncExecutionLimiter::CanFitOrIdle(const DeviceQueue& q,
                                             int64_t req_bytes) const {
  if (q.inflight_requests == 0) {
    return true;
  }
  if (q.inflight_bytes > max_bytes_per_device_) {
    return false;
  }
  return req_bytes <= max_bytes_per_device_ - q.inflight_bytes;
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
    absl::Span<const xla::ifrt::DeviceId> raw_device_ids,
    int64_t bytes_per_device) {
  const int64_t req_bytes = std::max<int64_t>(0, bytes_per_device);
  absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids =
      CanonicalizeDeviceIds(raw_device_ids);
  if (device_ids.empty()) {
    return Permit();
  }

  absl::MutexLock lock(mu_);
  for (const xla::ifrt::DeviceId device_id : device_ids) {
    const auto it = queues_.find(device_id);
    if (it != queues_.end() &&
        (!it->second.waiters.empty() || !CanFitOrIdle(it->second, req_bytes))) {
      return std::nullopt;
    }
  }
  for (const xla::ifrt::DeviceId device_id : device_ids) {
    DeviceQueue& q = queues_[device_id];
    ++q.inflight_requests;
    q.inflight_bytes += req_bytes;
  }
  return Permit(this, std::move(device_ids), req_bytes);
}

void IfrtAsyncExecutionLimiter::RunWhenPermitted(
    absl::Span<const xla::ifrt::DeviceId> raw_device_ids,
    int64_t bytes_per_device, tsl::thread::ThreadPool& thread_pool,
    absl::AnyInvocable<void(Permit) &&> callback) {
  const int64_t req_bytes = std::max<int64_t>(0, bytes_per_device);
  absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids =
      CanonicalizeDeviceIds(raw_device_ids);
  if (device_ids.empty()) {
    std::move(callback)(Permit());
    return;
  }

  {
    absl::MutexLock lock(mu_);
    size_t acquired_idx = 0;
    while (acquired_idx < device_ids.size()) {
      DeviceQueue& q = queues_[device_ids[acquired_idx]];
      // Reserve capacity inline on each uncontended device that has no FIFO
      // waiters and can fit `req_bytes` (or is completely idle).
      if (q.waiters.empty() && CanFitOrIdle(q, req_bytes)) {
        ++q.inflight_requests;
        q.inflight_bytes += req_bytes;
        ++acquired_idx;
      } else {
        break;
      }
    }
    if (acquired_idx < device_ids.size()) {
      auto state = std::make_shared<PendingAcquisition>(
          this, &thread_pool, std::move(device_ids), req_bytes, acquired_idx,
          std::move(callback));
      EnqueueCurrentDeviceLocked(std::move(state));
      return;
    }
  }

  std::move(callback)(Permit(this, std::move(device_ids), req_bytes));
}

void IfrtAsyncExecutionLimiter::EnqueueCurrentDeviceLocked(
    std::shared_ptr<PendingAcquisition> state) {
  const xla::ifrt::DeviceId target_device =
      state->device_ids[state->acquired_idx];
  const int64_t req_bytes = state->bytes_per_device;
  queues_[target_device].waiters.push_back(
      Waiter{req_bytes, [this, state = std::move(state)]() mutable {
               AdvancePendingAcquisition(std::move(state));
             }});
}

void IfrtAsyncExecutionLimiter::AdvancePendingAcquisition(
    std::shared_ptr<PendingAcquisition> state) {
  // Capacity for `device_ids[acquired_idx]` was already reserved in `Release()`
  // under `mu_` to preserve strict FIFO ordering.
  ++state->acquired_idx;

  if (state->acquired_idx < state->device_ids.size()) {
    absl::MutexLock lock(mu_);
    while (state->acquired_idx < state->device_ids.size()) {
      DeviceQueue& q = queues_[state->device_ids[state->acquired_idx]];
      if (q.waiters.empty() && CanFitOrIdle(q, state->bytes_per_device)) {
        ++q.inflight_requests;
        q.inflight_bytes += state->bytes_per_device;
        ++state->acquired_idx;
      } else {
        EnqueueCurrentDeviceLocked(std::move(state));
        return;
      }
    }
  }

  tsl::Executor* executor = state->thread_pool->AsExecutor();
  executor->Execute([state = std::move(state)]() mutable {
    state->dispatched = true;
    Permit permit(state->limiter, std::move(state->device_ids),
                  state->bytes_per_device);
    state->device_ids.clear();
    std::move(state->callback)(std::move(permit));
  });
}

void IfrtAsyncExecutionLimiter::Release(
    absl::Span<const xla::ifrt::DeviceId> device_ids, int64_t released_bytes) {
  absl::InlinedVector<absl::AnyInvocable<void() &&>, 4> ready_waiters;
  {
    absl::MutexLock lock(mu_);
    for (const xla::ifrt::DeviceId device_id : device_ids) {
      DeviceQueue& q = queues_[device_id];
      DCHECK_GT(q.inflight_requests, 0);
      DCHECK_GE(q.inflight_bytes, released_bytes);
      q.inflight_requests = std::max(0, q.inflight_requests - 1);
      q.inflight_bytes =
          std::max<int64_t>(0, q.inflight_bytes - released_bytes);
      // Wake all consecutive front FIFO waiters that now fit (including a
      // single oversized waiter at the head of the queue once
      // `inflight_requests` reaches 0).
      while (!q.waiters.empty() && CanFitOrIdle(q, q.waiters.front().bytes)) {
        Waiter w = std::move(q.waiters.front());
        q.waiters.pop_front();
        ++q.inflight_requests;
        q.inflight_bytes += w.bytes;
        ready_waiters.push_back(std::move(w.on_acquired));
      }
    }
  }

  for (auto& waiter : ready_waiters) {
    std::move(waiter)();
  }
}

int IfrtAsyncExecutionLimiter::InflightRequestsForDevice(
    xla::ifrt::DeviceId device_id) const {
  absl::MutexLock lock(mu_);
  const auto it = queues_.find(device_id);
  return it == queues_.end() ? 0 : it->second.inflight_requests;
}

int64_t IfrtAsyncExecutionLimiter::InflightBytesForDevice(
    xla::ifrt::DeviceId device_id) const {
  absl::MutexLock lock(mu_);
  const auto it = queues_.find(device_id);
  return it == queues_.end() ? 0 : it->second.inflight_bytes;
}

size_t IfrtAsyncExecutionLimiter::QueueDepthForDevice(
    xla::ifrt::DeviceId device_id) const {
  absl::MutexLock lock(mu_);
  const auto it = queues_.find(device_id);
  return it == queues_.end() ? 0 : it->second.waiters.size();
}

}  // namespace ifrt_serving
}  // namespace tensorflow
