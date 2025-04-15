/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_GPU_TFRT_STREAM_POOL_H_
#define XLA_PJRT_GPU_TFRT_STREAM_POOL_H_

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

// A bounded capacity pool of GPU streams.
class BoundedStreamPool {
 public:
  class Handle {
   public:
    Handle() = default;
    Handle(Handle const&) = delete;
    Handle(Handle&& other) {
      pool_ = other.pool_;
      stream_ = std::move(other.stream_);
      other.pool_ = nullptr;
      other.stream_ = nullptr;
    }

    ~Handle() {
      if (pool_) {
        CHECK_NE(stream_.get(), nullptr)
            << "Returning a null stream to the pool, which is not allowed.";
        pool_->Return(std::move(stream_));
      }
    }

    Handle& operator=(Handle const&) = delete;
    Handle& operator=(Handle&& other) {
      pool_ = other.pool_;
      stream_ = std::move(other.stream_);
      other.pool_ = nullptr;
      other.stream_ = nullptr;
      return *this;
    }
    stream_executor::Stream* get() const { return stream_.get(); }
    stream_executor::Stream& operator*() { return *stream_; }
    stream_executor::Stream* operator->() { return stream_.get(); }

   private:
    friend class BoundedStreamPool;
    Handle(BoundedStreamPool* pool,
           std::unique_ptr<stream_executor::Stream> stream)
        : pool_(pool), stream_(std::move(stream)) {}

    BoundedStreamPool* pool_ = nullptr;
    std::unique_ptr<stream_executor::Stream> stream_ = nullptr;
  };

  BoundedStreamPool(stream_executor::StreamExecutor* executor, int capacity);
  BoundedStreamPool(BoundedStreamPool const&) = delete;
  BoundedStreamPool(BoundedStreamPool&&) = delete;
  BoundedStreamPool& operator=(BoundedStreamPool const&) = delete;
  BoundedStreamPool& operator=(BoundedStreamPool&&) = delete;

  // Borrows a stream from the pool. Blocks if no stream is currently available.
  absl::StatusOr<Handle> Borrow();

  size_t GetAvailableStreamNum();

 private:
  friend class Handle;

  // Returns a stream to the pool.
  void Return(std::unique_ptr<::stream_executor::Stream> stream);

  absl::Mutex mu_;
  std::vector<std::unique_ptr<::stream_executor::Stream>> streams_
      ABSL_GUARDED_BY(mu_);
};

}  // namespace xla
#endif  // XLA_PJRT_GPU_TFRT_STREAM_POOL_H_
