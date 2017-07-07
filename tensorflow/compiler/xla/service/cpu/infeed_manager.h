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

// This header declares the abstract class for the infeed manager that
// is used by the CPU runtime to transfer buffers into an executing
// CPU computation, e.g., to feed data into a while loop.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_INFEED_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_INFEED_MANAGER_H_

#include <deque>
#include <vector>

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/core/platform/mutex.h"

namespace xla {
namespace cpu {
namespace runtime {

// Abstract class defining an infeed buffer that is passed to the
// runtime by a client. The client manages the storage of the buffer.
class InfeedBuffer {
 public:
  virtual ~InfeedBuffer();

  virtual int32 length() = 0;
  virtual void* data() = 0;
  virtual void Done() = 0;
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

  // Adds a set of buffers to the infeed queue
  // atomically. buffer->Done will be called when the buffer will no
  // longer be accessed by the InfeedManager, either as a result of a
  // call to Reset or because the runtime has dequeued and used the
  // buffer.
  void EnqueueBuffers(const std::vector<InfeedBuffer*>& buffers);

  // Blocks until the infeed queue is non-empty, then returns the
  // buffer at the head of the queue. Sets the current buffer to be
  // the returned buffer. It is an error to call BlockingDequeueBuffer
  // if there is an unreleased current buffer, i.e.,
  // ReleaseCurrentBuffer must be called between calls to
  // BlockingDequeueBuffer.
  InfeedBuffer* BlockingDequeueBuffer();

  // Releases the current buffer, which is the last buffer returned by
  // BlockingDequeuBuffer and not yet released. length and data must
  // match the buffer->length() and buffer->data() for the current
  // buffer.
  void ReleaseCurrentBuffer(int32 length, void* data);

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
};

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_INFEED_MANAGER_H_
