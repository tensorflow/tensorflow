/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
#define XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/gpu/context.h"
#include "xla/stream_executor/host_memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_executor_common.h"

namespace stream_executor {

namespace gpu {

class GpuStream;

// Intermediate implementation class for StreamExecutors that are used with
// GPUs.
class GpuExecutor : public StreamExecutorCommon {
 public:
  GpuExecutor(Platform* platform, int device_ordinal)
      : StreamExecutorCommon(platform),
        context_(nullptr),
        device_ordinal_(device_ordinal) {}

  int device_ordinal() const override { return device_ordinal_; };

  // Frees unused memory cached on the device for use with graphs back to the
  // OS.
  virtual absl::Status TrimGraphMemory() = 0;

  Context* gpu_context() const { return context_; }

  absl::StatusOr<std::vector<ApiTrace>> ExtractApiTrace() override {
    absl::MutexLock lock(&logger_mu_);
    return std::move(argument_logs_);
  }

  absl::Status RecordApiTrace(ApiTrace call) override {
    absl::MutexLock lock(&logger_mu_);
    if (std::holds_alternative<GemmCallTrace>(call) &&
        (argument_logging_mode_ & kLogGemm)) {
      argument_logs_.push_back(call);
    }
    return absl::OkStatus();
  }

  bool SetArgumentLoggingMode(uint64_t mode) override {
    absl::MutexLock lock(&logger_mu_);
    argument_logging_mode_ = mode;
    return true;
  }

  uint64_t GetArgumentLoggingMode() const { return argument_logging_mode_; }

 protected:
  // Sets the context.
  void set_context(Context* context) { context_ = context; }

 private:
  // Handle for session with the library/driver. Immutable post-initialization.
  Context* context_;

  // The device ordinal value that this executor was initialized with; recorded
  // for use in getting device metadata. Immutable post-initialization.
  int device_ordinal_;

  absl::Mutex logger_mu_;

  mutable std::vector<ApiTrace> argument_logs_ ABSL_GUARDED_BY(logger_mu_);

  uint64_t argument_logging_mode_ = 0;

  GpuExecutor(const GpuExecutor&) = delete;
  void operator=(const GpuExecutor&) = delete;
};

}  // namespace gpu
}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_GPU_GPU_EXECUTOR_H_
