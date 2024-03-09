/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_SERVICE_EXECUTABLE_RUN_OPTIONS_H_
#define XLA_SERVICE_SERVICE_EXECUTABLE_RUN_OPTIONS_H_

#include <functional>
#include <utility>
#include <vector>

#include "xla/executable_run_options.h"
#include "xla/service/stream_pool.h"
#include "xla/statusor.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {

// Class containing options for running a LocalExecutable and other auxiliary
// data.
class ServiceExecutableRunOptions {
 public:
  // Defines the interface of the stream borrower function pointer
  // with the first argument being the device ordinal, the second
  // argument being the number of streams to borrow, and the third
  // argument being the priority of the streams.
  using StreamBorrower =
      std::function<absl::StatusOr<std::vector<StreamPool::Ptr>>(
          int, int, se::StreamPriority)>;

  ServiceExecutableRunOptions()
      : ServiceExecutableRunOptions(ExecutableRunOptions()) {}

  explicit ServiceExecutableRunOptions(ExecutableRunOptions run_options,
                                       StreamBorrower stream_borrower = nullptr)
      : run_options_(std::move(run_options)),
        stream_borrower_(std::move(stream_borrower)) {}

  // Returns reference or pointer to `ExecutableRunOptions` member.
  const ExecutableRunOptions& run_options() const { return run_options_; }
  ExecutableRunOptions* mutable_run_options() { return &run_options_; }

  // Delegate to `ExecutableRunOptions` member.
  se::Stream* stream() const { return run_options_.stream(); }
  se::DeviceMemoryAllocator* allocator() const {
    return run_options_.allocator();
  }
  int device_ordinal() const { return run_options_.device_ordinal(); }

  // Borrows a stream and returns a smart pointer which returns the stream on
  // destruction.
  absl::StatusOr<StreamPool::Ptr> BorrowStream(
      int device_ordinal,
      se::StreamPriority priority = se::StreamPriority::Default) const {
    if (!stream_borrower_) {
      return Status(absl::StatusCode::kUnimplemented, "No stream borrower");
    }

    TF_ASSIGN_OR_RETURN(
        std::vector<StreamPool::Ptr> streams,
        stream_borrower_(device_ordinal, /*num_streams=*/1, priority));
    StreamPool::Ptr stream = std::move(streams.back());
    return stream;
  }

  absl::StatusOr<std::vector<StreamPool::Ptr>> BorrowStreams(
      int device_ordinal, int num_streams,
      se::StreamPriority priority = se::StreamPriority::Default) const {
    return stream_borrower_
               ? stream_borrower_(device_ordinal, num_streams, priority)
               : Status(absl::StatusCode::kUnimplemented, "No stream borrower");
  }

 private:
  ExecutableRunOptions run_options_;
  StreamBorrower stream_borrower_;
};

}  // namespace xla

#endif  // XLA_SERVICE_SERVICE_EXECUTABLE_RUN_OPTIONS_H_
