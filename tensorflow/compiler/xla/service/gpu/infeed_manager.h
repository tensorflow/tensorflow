/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// This header declares classes for the infeed manager and the infeed
// buffer that are used by the GPU runtime to transfer buffers into an
// executing GPU computation, e.g., to feed data into a while loop.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INFEED_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INFEED_MANAGER_H_

#include <deque>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {

// TODO(b/30467474) Once GPU infeed implementation settles, consider
// folding back the cpu and gpu infeed implementations into a generic
// one if possible.
//
// Current limitations:
// * Does not handle multiple devices/replicas.
//
// * Buffer space on GPU is allocated on every infeed enqueue request,
// and it does not handle the case when it runs out of
// memory. Potential solution is to pre-allocate a fixed amount of
// memory and block when that memory is full.

// Defines an infeed buffer that is passed to the runtime by
// the client. The client manages the memory of the buffer.
class InfeedBuffer {
 public:
  InfeedBuffer(perftools::gputools::StreamExecutor* executor, int64 length)
      : executor_(executor), length_(length) {
    device_memory_ = executor_->AllocateArray<uint8>(length);
    CHECK(!device_memory_.is_null());
  }

  ~InfeedBuffer() { executor_->Deallocate(&device_memory_); }

  int64 length() const { return length_; }

  // Callback to signal that this buffer is consumed. This helps the
  // client to manage memory for the infeed buffers.
  void Done() { delete this; }

  perftools::gputools::DeviceMemoryBase* device_memory() {
    return &device_memory_;
  }

 private:
  perftools::gputools::StreamExecutor* executor_;  // Not owned.
  const int64 length_;
  perftools::gputools::DeviceMemoryBase device_memory_;
};

// Client-side class used to enqueue infeed buffers.
class InfeedManager {
 public:
  InfeedManager();

  // Calls the completion callback for any enqueued buffers that have
  // not been dequeued by the runtime, and empties the infeed
  // queue. Reset may not be called while a runtime computation is
  // processing a dequeued buffer. The only safe way to ensure this
  // condition is to call Reset when no computation is taking place.
  void Reset();

  // Adds buffer to the infeed queue. buffer->Done will be called when
  // the buffer will no longer be accessed by the InfeedManager,
  // either as a result of a call to Reset or because the runtime has
  // dequeued and used the buffer.
  void EnqueueBuffer(InfeedBuffer* buffer);

  // Blocks until the infeed queue is non-empty, then returns the
  // buffer at the head of the queue. Sets the current buffer to be
  // the returned buffer. It is an error to call BlockingDequeueBuffer
  // if there is an unreleased current buffer, i.e.,
  // ReleaseCurrentBuffer must be called between calls to
  // BlockingDequeueBuffer.
  InfeedBuffer* BlockingDequeueBuffer();

  // Releases the current buffer, which is the last buffer returned by
  // BlockingDequeueBuffer and not yet released. device_memory must
  // match that of the current buffer.
  void ReleaseCurrentBuffer(
      perftools::gputools::DeviceMemoryBase* device_memory);

  // Returns a cached stream associated with an executor. Allocates a
  // new stream on the first invocation. On subsequent invocations, if
  // the cached executor is not the same as the requested executor,
  // returns null.
  perftools::gputools::Stream* GetStream(
      perftools::gputools::StreamExecutor* executor);

 private:
  tensorflow::mutex mu_;
  // Condition variable that is signaled every time a buffer is
  // enqueued to an empty queue.
  tensorflow::condition_variable cv_;
  // InfeedBuffer* queue contents are not owned, but buffer->Done must
  // be called when the buffer is no longer needed by the runtime.
  std::deque<InfeedBuffer*> enqueued_buffer_;
  // If non-NULL, the buffer that is currently being processed by the
  // runtime. Not owned.
  InfeedBuffer* current_buffer_;
  // Cached host to device stream for queuing infeed data.
  std::unique_ptr<perftools::gputools::Stream> host_to_device_stream_;
  // Executor that the host_to_device_stream belongs to. Not owned.
  perftools::gputools::StreamExecutor* host_to_device_executor_;
};

// Singleton creator-or-accessor: Returns the GPU infeed manager.
InfeedManager* GetOrCreateInfeedManager();

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INFEED_MANAGER_H_
