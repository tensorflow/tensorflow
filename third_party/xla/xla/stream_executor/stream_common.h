/* Copyright 2024 The OpenXLA Authors.

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

// The Stream is used in conjunction with the StreamExecutor "parent" to
// perform actions with a linear stream of dependencies. Dependencies can also
// be created between Streams to do task management (i.e. limit which tasks
// can be performed concurrently and specify what task dependencies exist).

#ifndef XLA_STREAM_EXECUTOR_STREAM_COMMON_H_
#define XLA_STREAM_EXECUTOR_STREAM_COMMON_H_

#include <cstdint>
#include <memory>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/fft.h"
#include "xla/stream_executor/kernel.h"
#include "xla/stream_executor/launch_dim.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/thread_annotations.h"

namespace stream_executor {

// Represents a stream of dependent computations on a GPU device.
//
// The operations within a stream execute linearly and asynchronously until
// BlockHostUntilDone() is invoked, which synchronously joins host code with
// the execution of the stream.
//
// If any given operation fails when entraining work for the stream, ok() will
// indicate that an error has occurred. After initialization, once a stream is
// !ok(), it will never be ok().
//
// Thread-safe post-initialization.
class StreamCommon : public Stream {
 public:
  // Instantiate a stream tied to parent as a platform executor. Work
  // entrained onto this stream will be launched/managed on that
  // StreamExecutor's platform.
  explicit StreamCommon(StreamExecutor *parent);

  PlatformSpecificHandle platform_specific_handle() const override;
  bool ok() const override { return !InErrorState(); }
  absl::StatusOr<Stream *> GetOrCreateSubStream() override
      TF_LOCKS_EXCLUDED(mu_);
  void ReturnSubStream(Stream *sub_stream) override TF_LOCKS_EXCLUDED(mu_);
  absl::Status Memcpy(void *host_dst, const DeviceMemoryBase &gpu_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase *gpu_dst, const void *host_src,
                      uint64_t size) override;
  absl::Status Memcpy(DeviceMemoryBase *gpu_dst,
                      const DeviceMemoryBase &gpu_src, uint64_t size) override;
  absl::Status Memset32(DeviceMemoryBase *location, uint32_t pattern,
                        uint64_t size) override;
  absl::Status BlockHostUntilDone() override TF_LOCKS_EXCLUDED(mu_);
  absl::Status DoHostCallback(absl::AnyInvocable<void() &&> callback) override;
  absl::Status DoHostCallbackWithStatus(
      absl::AnyInvocable<absl::Status() &&> callback) override;
  StreamExecutor *parent() const override {
    CHECK(parent_ != nullptr);
    return parent_;
  }

  CudaComputeCapability GetCudaComputeCapability() const override {
    return parent()->GetDeviceDescription().cuda_compute_capability();
  }

  RocmComputeCapability GetRocmComputeCapability() const override {
    return parent()->GetDeviceDescription().rocm_compute_capability();
  }
  std::variant<StreamPriority, int> priority() const override {
    return StreamPriority::Default;
  }
  absl::Status Launch(const ThreadDim &thread_dims, const BlockDim &block_dims,
                      const Kernel &k, const KernelArgs &args) override;

 protected:
  bool InErrorState() const TF_LOCKS_EXCLUDED(mu_) {
    absl::ReaderMutexLock lock(&mu_);
    return !status_.ok();
  }

  // Sets the error state if operation_retcode is false.
  // This is a useful shorthand for many stream routines.
  void CheckError(bool operation_retcode) TF_LOCKS_EXCLUDED(mu_);

  // Checks the status and logs the error message, if any.
  void CheckStatus(absl::Status status) TF_LOCKS_EXCLUDED(mu_);

  void SetError() { CheckError(false /* = operation_retcode */); }

 private:
  // The StreamExecutor that supports the operation of this stream.
  StreamExecutor *parent_;

  // mutex that guards the allocation / error state flags.
  // Mutable so that it can be obtained via const reader lock.
  mutable absl::Mutex mu_;

  // The last error (if any) of all method calls.
  absl::Status status_ ABSL_GUARDED_BY(mu_);

  // Sub-streams that are generated from this stream. Each element has a
  // pointer to sub-stream and a boolean value indicating if this substream is
  // ready to be reused.
  std::vector<std::pair<std::unique_ptr<Stream>, bool>> sub_streams_
      ABSL_GUARDED_BY(mu_);

  StreamCommon(const StreamCommon &) = delete;
  void operator=(const StreamCommon &) = delete;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_STREAM_COMMON_H_
