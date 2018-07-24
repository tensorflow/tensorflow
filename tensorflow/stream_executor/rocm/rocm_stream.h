/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

// Defines the ROCMStream type - the ROCM-specific implementation of the generic
// StreamExecutor Stream interface.

#ifndef TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_STREAM_H_
#define TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_STREAM_H_

#include "tensorflow/stream_executor/rocm/rocm_driver.h"
#include "tensorflow/stream_executor/platform/thread_annotations.h"
#include "tensorflow/stream_executor/stream_executor_internal.h"

namespace stream_executor {
namespace rocm {

class ROCMExecutor;

// Wraps a hipStream_t in order to satisfy the platform-independent
// StreamInterface.
//
// Thread-safe post-initialization.
class ROCMStream : public internal::StreamInterface {
 public:
  explicit ROCMStream(ROCMExecutor *parent)
      : parent_(parent), rocm_stream_(nullptr), completed_event_(nullptr) {}

  // Note: teardown is handled by a parent's call to DeallocateStream.
  ~ROCMStream() override {}

  void *GpuStreamHack() override { return rocm_stream_; }
  void **GpuStreamMemberHack() override {
    return reinterpret_cast<void **>(&rocm_stream_);
  }

  // Explicitly initialize the ROCM resources associated with this stream, used
  // by StreamExecutor::AllocateStream().
  bool Init();

  // Explicitly destroy the ROCM resources associated with this stream, used by
  // StreamExecutor::DeallocateStream().
  void Destroy();

  // Returns true if no work is pending or executing on the stream.
  bool IsIdle() const;

  // Retrieves an event which indicates that all work enqueued into the stream
  // has completed. Ownership of the event is not transferred to the caller, the
  // event is owned by this stream.
  hipEvent_t* completed_event() { return &completed_event_; }

  // Returns the hipStream_t value for passing to the ROCM API.
  //
  // Precond: this ROCMStream has been allocated (otherwise passing a nullptr
  // into ROCM library causes difficult-to-understand faults).
  hipStream_t rocm_stream() const {
    DCHECK(rocm_stream_ != nullptr);
    return const_cast<hipStream_t>(rocm_stream_);
  }

  ROCMExecutor *parent() const { return parent_; }

 private:
  ROCMExecutor *parent_;  // Executor that spawned this stream.
  hipStream_t rocm_stream_;  // Wrapped ROCM stream handle.

  // Event that indicates this stream has completed.
  hipEvent_t completed_event_ = nullptr;
};

// Helper functions to simplify extremely common flows.
// Converts a Stream to the underlying ROCMStream implementation.
ROCMStream *AsROCMStream(Stream *stream);

// Extracts a hipStream_t from a ROCMStream-backed Stream object.
hipStream_t AsROCMStreamValue(Stream *stream);

}  // namespace rocm
}  // namespace stream_executor

#endif  // TENSORFLOW_STREAM_EXECUTOR_ROCM_ROCM_STREAM_H_
