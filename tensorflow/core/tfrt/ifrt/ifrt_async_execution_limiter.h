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
#include <cstdint>
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

// Per-device non-blocking asynchronous staged-byte limiter for IFRT serving.
//
// Bounds total estimated in-flight staged HBM bytes (covering per-request H2D
// user-input buffers, device execution, and D2H output buffers) per
// `xla::ifrt::DeviceId` across all `IfrtServingExecutable` instances sharing
// this limiter, without blocking caller threads (e.g. TFRT
// `NonBlockingWorkQueue`).
//
// Forward-progress guarantee: an idle device (`inflight_requests == 0`) always
// admits at least 1 request even when that single request's `bytes_per_device`
// exceeds `max_bytes_per_device()`, so models with very large signatures never
// stall. While queued waiters exist, new arrivals never barge ahead of the
// per-device FIFO queue, preventing starvation of large requests behind a
// continuous stream of small requests.
//
// Multi-device requests canonicalize and acquire `DeviceId`s in strict
// ascending order so concurrent multi-device requests cannot deadlock.
//
// This class is thread-safe.
class IfrtAsyncExecutionLimiter {
 public:
  // RAII move-only token representing reserved staged-byte permits on one or
  // more devices. Releasing or destroying a `Permit` returns the reserved bytes
  // and schedules any newly fitting front FIFO waiters onto their `ThreadPool`.
  class Permit {
   public:
    Permit() = default;
    Permit(Permit&& other) noexcept
        : limiter_(std::exchange(other.limiter_, nullptr)),
          device_ids_(std::move(other.device_ids_)),
          bytes_per_device_(std::exchange(other.bytes_per_device_, 0)) {
      other.device_ids_.clear();
    }
    Permit& operator=(Permit&& other) noexcept {
      if (this != &other) {
        Release();
        limiter_ = std::exchange(other.limiter_, nullptr);
        device_ids_ = std::move(other.device_ids_);
        bytes_per_device_ = std::exchange(other.bytes_per_device_, 0);
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
        const int64_t bytes = std::exchange(bytes_per_device_, 0);
        limiter->Release(ids, bytes);
      }
    }

    explicit operator bool() const { return limiter_ != nullptr; }

   private:
    friend class IfrtAsyncExecutionLimiter;
    Permit(IfrtAsyncExecutionLimiter* limiter,
           absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids,
           int64_t bytes_per_device)
        : limiter_(limiter),
          device_ids_(std::move(device_ids)),
          bytes_per_device_(bytes_per_device) {}

    IfrtAsyncExecutionLimiter* limiter_ = nullptr;
    absl::InlinedVector<xla::ifrt::DeviceId, 4> device_ids_;
    int64_t bytes_per_device_ = 0;
  };

  // `max_bytes_per_device` must be > 0 (clamped to 1 in opt builds if <= 0).
  explicit IfrtAsyncExecutionLimiter(int64_t max_bytes_per_device);

  IfrtAsyncExecutionLimiter(const IfrtAsyncExecutionLimiter&) = delete;
  IfrtAsyncExecutionLimiter& operator=(const IfrtAsyncExecutionLimiter&) =
      delete;

  int64_t max_bytes_per_device() const { return max_bytes_per_device_; }

  // Attempts to acquire permits for `bytes_per_device` (clamped to >= 0) across
  // all `raw_device_ids` immediately without blocking or queuing. Succeeds iff
  // every unique device has no queued waiters and is either completely idle
  // (`inflight_requests == 0`) or can fit `bytes_per_device` within
  // `max_bytes_per_device()`. On failure, mutates no state and returns
  // `std::nullopt`.
  std::optional<Permit> TryAcquire(
      absl::Span<const xla::ifrt::DeviceId> raw_device_ids,
      int64_t bytes_per_device);

  // Acquires permits for `bytes_per_device` (clamped to >= 0) across all
  // `raw_device_ids` in canonical ascending `DeviceId` order without blocking
  // the calling thread:
  // - If all devices can admit the request immediately, invokes `callback`
  //   inline on the calling thread with a valid `Permit`.
  // - Otherwise, enqueues in per-device FIFO order and schedules `callback` on
  //   `thread_pool` once permits for all requested devices have been granted.
  void RunWhenPermitted(absl::Span<const xla::ifrt::DeviceId> raw_device_ids,
                        int64_t bytes_per_device,
                        tsl::thread::ThreadPool& thread_pool,
                        absl::AnyInvocable<void(Permit) &&> callback);

  // Helpers for testing to inspect internal state.
  int InflightRequestsForDevice(xla::ifrt::DeviceId device_id) const;
  int64_t InflightBytesForDevice(xla::ifrt::DeviceId device_id) const;
  size_t QueueDepthForDevice(xla::ifrt::DeviceId device_id) const;

 private:
  struct Waiter {
    int64_t bytes = 0;
    absl::AnyInvocable<void() &&> on_acquired;
  };

  struct DeviceQueue {
    int inflight_requests = 0;
    int64_t inflight_bytes = 0;
    std::deque<Waiter> waiters;
  };

  // Contains state for a request in FIFO queue (to obtain a permit).
  struct PendingAcquisition;

  // Returns true if `q` is completely idle (`inflight_requests == 0`, allowing
  // oversized single requests to make forward progress) or has enough remaining
  // byte budget to fit `req_bytes` within `max_bytes_per_device_`.
  bool CanFitOrIdle(const DeviceQueue& q, int64_t req_bytes) const;

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
  // `~PendingAcquisition`). Decrements active counts/bytes and wakes all
  // consecutive front FIFO waiters that now fit on each device, invoking their
  // callbacks after unlocking `mu_`.
  void Release(absl::Span<const xla::ifrt::DeviceId> device_ids,
               int64_t released_bytes);

  const int64_t max_bytes_per_device_;
  mutable absl::Mutex mu_;  // Protects the `queues_` map.
  // The FIFO queues for obtaining permits.
  absl::flat_hash_map<xla::ifrt::DeviceId, DeviceQueue> queues_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace ifrt_serving
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_IFRT_IFRT_ASYNC_EXECUTION_LIMITER_H_
