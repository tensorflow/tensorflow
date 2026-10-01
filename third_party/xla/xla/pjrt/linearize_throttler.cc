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

#include "xla/pjrt/linearize_throttler.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"
#include "absl/container/chunked_queue.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/raw_buffer.h"
#include "xla/pjrt/staging_buffer.h"
#include "xla/runtime/device_id.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "tsl/platform/context.h"

namespace xla {

class LinearizeThrottler::Throttler {
 public:
  class Credit {
   public:
    Credit() = default;
    Credit(Throttler* throttler, size_t bytes)
        : throttler_(throttler), bytes_(bytes) {}
    ~Credit() {
      if (bytes_ > 0 && throttler_ != nullptr) {
        throttler_->ReleaseAndScheduleNext(bytes_);
      }
    }
    Credit(Credit&& other)
        : throttler_(other.throttler_), bytes_(other.bytes_) {
      other.throttler_ = nullptr;
      other.bytes_ = 0;
    }
    Credit& operator=(Credit&& other) {
      using std::swap;
      swap(throttler_, other.throttler_);
      swap(bytes_, other.bytes_);
      return *this;
    }
    Credit(const Credit&) = delete;
    Credit& operator=(const Credit&) = delete;

    size_t bytes() const { return bytes_; }

   private:
    Throttler* throttler_ = nullptr;
    size_t bytes_ = 0;
  };

  Throttler(size_t max_bytes, AsyncWorkRunner* async_runner)
      : max_bytes_(max_bytes), async_runner_(async_runner) {}

  ~Throttler() {
    mu_.LockWhen(absl::Condition(
        +[](size_t* inflight_bytes) { return *inflight_bytes == 0; },
        &inflight_bytes_));
    mu_.unlock();
  }

  Credit Acquire(size_t bytes, bool allow_overdraft = false) {
    absl::MutexLock lock(mu_);
    return AcquireLocked(bytes, allow_overdraft);
  }

  template <bool kForceThreadHop>
  void Schedule(absl::AnyInvocable<void(Credit)> task, size_t bytes) {
    if (bytes == 0) {
      if constexpr (kForceThreadHop) {
        async_runner_->Execute(tsl::WithCurrentContext(
            absl::bind_front(std::move(task), Credit())));
      } else {
        std::move(task)(Credit());
      }
      return;
    }
    absl::ReleasableMutexLock lock(mu_);
    // Until chunked linearization is implemented, cap the requested bytes at
    // max_bytes_ so buffers larger than max_bytes_ can still be scheduled.
    Credit credit =
        AcquireLocked(std::min(bytes, max_bytes_), /*allow_overdraft=*/false);
    if (credit.bytes() > 0) {
      lock.Release();
      if constexpr (kForceThreadHop) {
        async_runner_->Execute(tsl::WithCurrentContext(
            absl::bind_front(std::move(task), std::move(credit))));
      } else {
        std::move(task)(std::move(credit));
      }
    } else {
      throttled_.emplace_back(bytes, std::move(task));
    }
  }

 private:
  Credit AcquireLocked(size_t bytes, bool allow_overdraft)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    if (!allow_overdraft && inflight_bytes_ + bytes > max_bytes_) {
      return Credit();
    }
    inflight_bytes_ += bytes;
    return Credit(this, bytes);
  }

  void ReleaseAndScheduleNext(size_t release_bytes) {
    absl::InlinedVector<absl::AnyInvocable<void() &&>, 4> ready_tasks;
    {
      absl::MutexLock lock(&mu_);
      DCHECK_GE(inflight_bytes_, release_bytes);
      inflight_bytes_ -= release_bytes;
      while (!throttled_.empty()) {
        auto& next = throttled_.front();
        // Until chunked linearization is implemented, cap the requested bytes
        // at max_bytes_ so buffers larger than max_bytes_ can still be
        // scheduled.
        Credit credit = AcquireLocked(std::min(next.bytes, max_bytes_),
                                      /*allow_overdraft=*/false);
        if (credit.bytes() == 0) {
          break;
        }
        ready_tasks.emplace_back(
            absl::bind_front(std::move(next.task), std::move(credit)));
        throttled_.pop_front();
      }
    }
    for (auto& task : ready_tasks) {
      async_runner_->Execute(tsl::WithCurrentContext(std::move(task)));
    }
  }

  const size_t max_bytes_;
  AsyncWorkRunner* const async_runner_ = nullptr;
  absl::Mutex mu_;
  size_t inflight_bytes_ ABSL_GUARDED_BY(mu_) = 0;
  struct ThrottledTask {
    ThrottledTask(size_t bytes, absl::AnyInvocable<void(Credit)> task)
        : bytes(bytes), task(std::move(task)) {}
    size_t bytes;
    absl::AnyInvocable<void(Credit)> task;
  };
  absl::chunked_queue<ThrottledTask> throttled_ ABSL_GUARDED_BY(mu_);
};

LinearizeThrottler::LinearizeThrottler(HostMemoryAllocator* host_allocator,
                                       AsyncWorkRunner* async_runner,
                                       Options options)
    : host_allocator_(host_allocator),
      linearize_throttler_(std::make_unique<Throttler>(
          options.max_inflight_linearization_bytes, async_runner)),
      delinearize_throttler_(std::make_unique<Throttler>(
          options.max_inflight_delinearization_bytes, async_runner)) {}

LinearizeThrottler::~LinearizeThrottler() = default;

// A staging buffer over host memory which holds a throttler credit until it is
// destroyed. `on_done` runs on destruction, before the credit is released.
class LinearizeThrottler::LinearizedBuffer : public PjRtStagingBuffer {
 public:
  LinearizedBuffer(absl::Span<uint8_t> data, Throttler::Credit credit,
                   absl::AnyInvocable<void() &&> on_done)
      : credit_(std::move(credit)), data_(data), on_done_(std::move(on_done)) {}

  ~LinearizedBuffer() override {
    if (on_done_) {
      std::move(on_done_)();
    }
  }

  absl::Span<uint8_t> data() override { return data_; }
  absl::Span<const uint8_t> const_data() const override { return data_; }

 private:
  // Declared first so that it is released last.
  Throttler::Credit credit_;
  absl::Span<uint8_t> data_;
  absl::AnyInvocable<void() &&> on_done_;
};

class LinearizeThrottler::LazyLinearizedBuffer : public PjRtStagingBuffer {
 public:
  LazyLinearizedBuffer(HostMemoryAllocator* allocator, size_t size,
                       PjRtRawBufferRef reused_buffer, Throttler::Credit credit,
                       int numa_node,
                       std::optional<LocalDeviceId> local_device_id)
      : allocator_(allocator),
        size_(size),
        reused_buffer_(std::move(reused_buffer)),
        credit_(std::move(credit)),
        numa_node_(numa_node),
        local_device_id_(local_device_id) {}

  absl::Span<uint8_t> data() override {
    if (reused_buffer_) {
      return {reinterpret_cast<uint8_t*>(reused_buffer_->GetHostPointer()),
              size_};
    }
    if (!data_) {
      data_ =
          allocator_->Allocate(size_, {
                                          .numa_node = numa_node_,
                                          .local_device_id = local_device_id_,
                                      });
    }
    return {data_.get(), size_};
  }

  absl::Span<const uint8_t> const_data() const override {
    if (reused_buffer_) {
      return {reinterpret_cast<uint8_t*>(reused_buffer_->GetHostPointer()),
              size_};
    }
    CHECK(data_) << "LazyLinearizedBuffer is uninitialized. The caller must "
                    "write into data() first.";
    return {data_.get(), size_};
  }

 private:
  HostMemoryAllocator* allocator_;
  size_t size_;
  PjRtRawBufferRef reused_buffer_;
  Throttler::Credit credit_;
  int numa_node_;
  std::optional<LocalDeviceId> local_device_id_;

  HostMemoryAllocator::OwnedPtr data_;
};

tsl::AsyncValueRef<PjRtStagingBuffer> LinearizeThrottler::AllocateStagingDest(
    bool sync, size_t size, PjRtRawBufferRef reused_buffer, int numa_node,
    std::optional<LocalDeviceId> local_device_id) {
  if (reused_buffer && reused_buffer->GetHostPointer() == nullptr) {
    reused_buffer.reset();
  }
  auto async_result = tsl::MakeUnconstructedAsyncValueRef<
      LazyLinearizedBuffer>();  // LinearizedBuffer>();
  auto allocator_into_fn =
      [async_result, size, reused_buffer, numa_node, local_device_id,
       host_allocator = host_allocator_](Throttler::Credit credit) {
        async_result.emplace(host_allocator, size, std::move(reused_buffer),
                             std::move(credit), numa_node, local_device_id);
      };
  if (sync) {
    allocator_into_fn(
        linearize_throttler_->Acquire(size, /*allow_overdraft=*/true));
  } else {
    linearize_throttler_->Schedule</*kForceThreadHop=*/false>(
        std::move(allocator_into_fn), size);
  }
  return async_result;
}

tsl::AsyncValueRef<PjRtStagingBuffer> LinearizeThrottler::CreateFromData(
    const void* data, size_t size_in_bytes,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer) {
  auto async_result = tsl::MakeUnconstructedAsyncValueRef<LinearizedBuffer>();
  linearize_throttler_->Schedule</*kForceThreadHop=*/false>(
      [async_result, data, size_in_bytes,
       on_done_with_host_buffer = std::move(on_done_with_host_buffer)](
          Throttler::Credit credit) mutable {
        // Zero-copy staging buffers are only read from.
        absl::Span<uint8_t> span(
            const_cast<uint8_t*>(static_cast<const uint8_t*>(data)),
            size_in_bytes);
        async_result.emplace(span, std::move(credit),
                             std::move(on_done_with_host_buffer));
      },
      size_in_bytes);
  return async_result;
}

tsl::AsyncValueRef<PjRtStagingBuffer>
LinearizeThrottler::AllocateForDelinearizationAsync(
    size_t size, int numa_node, std::optional<LocalDeviceId> local_device_id) {
  auto async_result = tsl::MakeUnconstructedAsyncValueRef<LinearizedBuffer>();
  delinearize_throttler_->Schedule</*kForceThreadHop=*/false>(
      [host_allocator = host_allocator_, size, numa_node, local_device_id,
       async_result](Throttler::Credit credit) mutable {
        HostMemoryAllocator::OwnedPtr memory = host_allocator->Allocate(
            size, {
                      .numa_node = numa_node,
                      .local_device_id = local_device_id,
                  });
        absl::Span<uint8_t> span(memory.get(), size);
        async_result.emplace(span, std::move(credit),
                             [memory = std::move(memory)] {});
      },
      size);
  return async_result;
}

absl::Status LinearizeThrottler::ThrottleSyncDelinearize(
    size_t size, absl::FunctionRef<absl::Status()> delinearize_fn) {
  Throttler::Credit credit =
      delinearize_throttler_->Acquire(size, /*allow_overdraft=*/true);
  return delinearize_fn();
}

}  // namespace xla
