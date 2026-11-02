/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_STREAM_POOL_H_
#define XLA_SERVICE_STREAM_POOL_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace se = ::stream_executor;

// Pool of stream_executor::Streams, which are created as needed and
// destroyed when the pool is destroyed.
class StreamPool {
 public:
  // Creates a new stream for the requested priority. Factories should create
  // the stream with that priority. The pool files a returned stream under the
  // priority it was borrowed with, not under `Stream::priority()`, so a factory
  // that cannot honor priorities still gets correct stream reuse.
  using StreamFactory =
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<se::Stream>>(
          se::StreamPriority)>;

  struct PtrDeleter {
    void operator()(se::Stream* stream) {
      pool->ReturnStream(stream, priority);
    }
    StreamPool* pool;
    se::StreamPriority priority;
  };

  // Stream pointer type returned by BorrowStream, which returns the
  // stream to the pool on destruction.
  using Ptr = std::unique_ptr<se::Stream, PtrDeleter>;

  // Convenience constructor for the default stream-creation case. Internally
  // delegates to the StreamFactory constructor. A follow-up can remove this
  // overload and update all call sites to use StreamFactory directly.
  explicit StreamPool(se::StreamExecutor* executor);
  explicit StreamPool(StreamFactory stream_factory);

  // Returns a pointer to a stream in the pool, creating a new stream
  // if none are available in the pool. The returned smart pointer
  // returns the stream to the pool on destruction.
  //
  // This method is thread-safe.
  absl::StatusOr<Ptr> BorrowStream(
      se::StreamPriority priority = se::StreamPriority::Default);

 private:
  // Puts a pointer to a stream back into the pool under `priority`, the
  // priority it was borrowed with, leaving it free for future use. Streams
  // that have previously encountered errors are deleted, and not returned to
  // the pool.
  //
  // This method is thread-safe.
  void ReturnStream(se::Stream* stream, se::StreamPriority priority);

  absl::Mutex mu_;
  // This stores streams with user-specified priority.
  absl::flat_hash_map<se::StreamPriority,
                      std::vector<std::unique_ptr<se::Stream>>>
      streams_with_pri_ ABSL_GUARDED_BY(mu_);
  StreamFactory stream_factory_;
};

}  // namespace xla

#endif  // XLA_SERVICE_STREAM_POOL_H_
