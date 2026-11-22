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

#include "xla/service/stream_pool.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"

namespace xla {

StreamPool::StreamPool(se::StreamExecutor* executor)
    : StreamPool([executor](se::StreamPriority priority) {
        return executor->CreateStream(priority);
      }) {}

StreamPool::StreamPool(StreamFactory stream_factory)
    : stream_factory_(std::move(stream_factory)) {
  CHECK(stream_factory_ != nullptr);
}

absl::StatusOr<StreamPool::Ptr> StreamPool::BorrowStream(
    se::StreamPriority priority) {
  std::unique_ptr<se::Stream> stream = nullptr;

  {
    absl::MutexLock lock(mu_);
    if (auto it = streams_with_pri_.find(priority);
        it != streams_with_pri_.end()) {
      std::vector<std::unique_ptr<se::Stream>>& streams = it->second;
      while (!streams.empty() && !stream) {
        // Try to reuse an existing stream from the pool.
        stream = std::move(streams.back());
        streams.pop_back();
        if (stream->ok()) {
          VLOG(1) << absl::StrFormat(
              "StreamPool reusing existing stream (%p) with priority: %s",
              stream.get(), se::StreamPriorityToString(priority));
        } else {
          VLOG(1) << absl::StrFormat(
              "Stream (%p) was not ok, deleting with : %s", stream.get(),
              se::StreamPriorityToString(priority));
          stream = nullptr;
        }
      }
    }
  }

  if (!stream) {
    // Create a new stream.
    ABSL_ASSIGN_OR_RETURN(stream, stream_factory_(priority));
    // Keep any name the factory assigned and append the pool's label.
    std::string name =
        absl::StrFormat("%s pool stream", se::StreamPriorityToString(priority));
    if (!stream->GetName().empty()) {
      name = absl::StrFormat("%s %s", stream->GetName(), name);
    }
    stream->SetName(std::move(name));
    VLOG(1) << absl::StrFormat("Created new stream (%p) with priority = %s",
                               stream.get(),
                               se::StreamPriorityToString(priority));
  }

  // Return the stream wrapped in Ptr, which has our special deleter semantics.
  PtrDeleter deleter = {this, priority};
  return Ptr(stream.release(), deleter);
}

void StreamPool::ReturnStream(se::Stream* stream, se::StreamPriority priority) {
  if (stream->ok()) {
    VLOG(1) << absl::StrFormat("StreamPool returning ok stream (%p)", stream);
    absl::MutexLock lock(mu_);
    streams_with_pri_[priority].emplace_back(stream);
  } else {
    // If the stream has encountered any errors, all subsequent operations on it
    // will fail. So just delete the stream, and rely on new streams to be
    // created in the future.
    VLOG(1) << absl::StrFormat("StreamPool deleting !ok stream (%p)", stream);
    delete stream;
  }
}

}  // namespace xla
