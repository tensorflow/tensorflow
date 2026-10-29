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

#ifndef TENSORFLOW_CORE_TFRT_IFRT_IFRT_ASYNC_EXECUTION_LIMITER_H_
#define TENSORFLOW_CORE_TFRT_IFRT_IFRT_ASYNC_EXECUTION_LIMITER_H_

#include <cstddef>
#include <deque>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/device.h"
#include "xla/tsl/platform/threadpool.h"

namespace tensorflow {
namespace ifrt_serving {

// Per-device non-blocking asynchronous execution limiter for IFRT serving.
//
// Bounds count of concurrent in-flight executions (covering H2D input
// allocation, device run, and D2H output transfer) per `xla::ifrt::DeviceId`
// across all `IfrtServingExecutable` instances sharing this limiter.
// Importantly, does NOT block caller threads (ie TFRT `NonBlockingWorkQueue`).
//
// Multi-device requests canonicalize and acquire `DeviceId`s in strict
// ascending order so no deadlock. Per-device waiters, in strict FIFO order.
//
// This class is thread-safe.
class IfrtAsyncExecutionLimiter {
 public:
  // RAII move-only token representing acquired execution permits on one or more
  // devices. Releasing or destroying a `Permit` returns the permits and
  // schedules any newly unblocked waiters onto their target `ThreadPool`. A
  // permit must be acquired for a request to be in-flight.
  class Permit {
   public:
    Permit() = default;
    Permit(Permit&& other) noexcept
        : limiter_(std::exchange(other.limiter_, nullptr)),
          device_ids_(std::move(other.device_ids_)) {
      other.device_ids_.clear();
    }
    Permit& operator=(Permit&& other) noexcept {
      if (this != &other) {
        Release();
        // Hand off permit.
        limiter_ = std::exchange(other.limiter_, nullptr);
        device_ids_ = std::move(other.device_ids_);
        other.device_ids_.clear();
      }
      return *this;
    }
    Permit(const Permit&) = delete;
    Permit& operator=(const Permit&) = delete;

    ~Permit() { Release(); }

    void Release() {
      if (limiter_ != nullptr) {
        IfrtAsyncExecutionLimiter* limiter = std::exchange(limiter_, nullptr);
        absl::InlinedVector<xla::ifrt::DeviceId, 4> ids =
            std::move(device_ids_);
        device_ids_.clear();
        limiter->Release(ids);
      }
    }

    explicit operator bool() const { return limiter_ != nullptr; }

   private:
    friend class IfrtAsyncExecutionLimiter;
    Permit(IfrtAsyncExecutionLimiter* limiter,
           absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids)
        : limiter_(limiter), device_ids_(std::move(device_ids)) {}

    IfrtAsyncExecutionLimiter* limiter_ = nullptr;
    absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids_;
  };

  // `max_inflight_per_device` must be > 0 (clamped to 1 in opt builds if <= 0).
  explicit IfrtAsyncExecutionLimiter(int max_inflight_per_device);

  IfrtAsyncExecutionLimiter(const IfrtAsyncExecutionLimiter&) = delete;
  IfrtAsyncExecutionLimiter& operator=(const IfrtAsyncExecutionLimiter&) =
      delete;

  int max_inflight_per_device() const { return max_inflight_per_device_; }

  // Attempts to acquire permits for all `raw_device_ids` immediately without
  // blocking or queuing. Succeeds iff every unique device currently has fewer
  // than `max_inflight_per_device()` active permits and no queued waiters.
  // On failure, mutates no state and returns `std::nullopt`.
  std::optional<Permit> TryAcquire(
      absl::Span<const xla::ifrt::DeviceId> raw_device_ids);

  // Acquires permits for all `raw_device_ids` in canonical ascending `DeviceId`
  // order without blocking the calling thread:
  // - If all devices are immediately available, invokes `callback` inline on
  //   the calling thread with a valid `Permit`.
  // - Otherwise, enqueues in per-device FIFO order and schedules `callback` on
  //   `thread_pool` once permits for all requested devices have been granted.
  void RunWhenPermitted(absl::Span<const xla::ifrt::DeviceId> raw_device_ids,
                        tsl::thread::ThreadPool& thread_pool,
                        absl::AnyInvocable<void(Permit) &&> callback);

  // Helpers for testing to inspect internal state.
  int InflightForDevice(xla::ifrt::DeviceId device_id) const;
  size_t QueueDepthForDevice(xla::ifrt::DeviceId device_id) const;

 private:
  struct DeviceQueue {
    int inflight = 0;
    std::deque<absl::AnyInvocable<void() &&>> waiters;
  };

  // Contains state for a request in FIFO queue (to obtain a permit).
  struct PendingAcquisition;

  // Returns `raw_device_ids` sorted in strict ascending order with duplicates
  // removed, preventing AB-BA deadlocks and double-counting on multi-device
  // requests.
  static absl::InlinedVector<xla::ifrt::DeviceId, 4> CanonicalizeDeviceIds(
      absl::Span<const xla::ifrt::DeviceId> raw_device_ids);

  // Appends `state` to the FIFO waiter queue.
  void EnqueueCurrentDeviceLocked(std::shared_ptr<PendingAcquisition> state)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Acquires remaining devices in sorted order (re-queuing if contended) and,
  // once all devices are held, schedules `state->callback`.
  void AdvancePendingAcquisition(std::shared_ptr<PendingAcquisition> state);

  // Releases permits for `device_ids` (held by `Permit` or rolled back by
  // `~PendingAcquisition`). For each device, transfers the slot directly to
  // the front FIFO waiter (keeping `inflight` unchanged to prevent barging) or
  // decrements `inflight`, invoking unblocked waiters after unlocking `mu_`.
  void Release(absl::Span<const xla::ifrt::DeviceId> device_ids);

  const int max_inflight_per_device_;
  mutable absl::Mutex mu_;  // Protects the `queues_` map.
  // The FIFO queues for obtaining permits.
  absl::flat_hash_map<xla::ifrt::DeviceId, DeviceQueue> queues_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_ASYNC_EXECUTION_LIMITER_H_
