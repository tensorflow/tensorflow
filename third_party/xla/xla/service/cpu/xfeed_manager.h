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

// This header declares the abstract class for the infeed manager that
// is used by the CPU runtime to transfer buffers into an executing
// CPU computation, e.g., to feed data into a while loop.

#ifndef XLA_SERVICE_CPU_XFEED_MANAGER_H_
#define XLA_SERVICE_CPU_XFEED_MANAGER_H_

#include <deque>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/shape.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace cpu {
namespace runtime {

// Abstract class defining an infeed buffer that is passed to the
// runtime by a client. The client manages the storage of the buffer.
class XfeedBuffer {
 public:
  virtual ~XfeedBuffer() = default;

  virtual int32_t length() = 0;
  virtual void* data() = 0;

  // The 'shape' parameter reflects what shape the embedded program was
  // expecting / producing with respect to this XfeedBuffer. E.g. this will
  // contain information about the layout of an outfed buffer.
  virtual void Done(absl::StatusOr<Shape> shape) = 0;
};

// Reusable component for managing the infeed and outfeed queue state.
class XfeedQueueManager {
 public:
  XfeedQueueManager(std::string queue_name) : queue_name_(queue_name) {}

  // Calls the completion callback for any enqueued buffers that have
  // not been dequeued by the runtime, and empties the
  // queue. Reset may not be called while a runtime computation is
  // processing a dequeued buffer. The only safe way to ensure this
  // condition is to call Reset when no computation is taking place.
  void Reset();

  // Adds a sequence of buffers to the queue atomically. buffer->Done will be
  // called when the buffer will no longer be accessed by the XfeedManager,
  // either as a result of a call to Reset or because the runtime has dequeued
  // and used the buffer.
  void EnqueueBuffersAtomically(absl::Span<XfeedBuffer* const> buffers);

  // Blocks until the queue is non-empty, then returns the buffer at the head of
  // the queue. Sets the current buffer to be the returned buffer. It is an
  // error to call BlockingDequeueBuffer if there is an unreleased current
  // buffer, i.e., ReleaseCurrentBuffer must be called between calls to
  // BlockingDequeueBuffer.
  XfeedBuffer* BlockingDequeueBuffer();

  // Releases the current buffer, which is the last buffer returned by
  // BlockingDequeuBuffer and not yet released. length and data must
  // match the buffer->length() and buffer->data() for the current
  // buffer.
  //
  // 'shape' communicates the shape of the buffer being released. If the program
  // passed a value that could not be decoded as a shape, 'shape' will be an
  // error status. In the case of outfeed, this indicates the layout of the
  // shape that has been outfed. In the case of infeed, this can be used for
  // sanity checking purposes.
  void ReleaseCurrentBuffer(int32_t length, void* data,
                            absl::StatusOr<Shape> shape);

 private:
  const std::string queue_name_;

  absl::Mutex mu_;

  // Condition variable that is signaled every time a buffer is
  // enqueued to an empty queue.
  absl::CondVar cv_;

  // XfeedBuffer* queue contents are not owned, but buffer->Done must
  // be called when the buffer is no longer needed by the runtime.
  std::deque<XfeedBuffer*> enqueued_buffers_;

  // If non-NULL, the buffer that is currently being processed by the
  // runtime. Not owned.
  XfeedBuffer* current_buffer_ = nullptr;
};

// Client-side class used to enqueue infeed buffers.
class XfeedManager {
 public:
  XfeedManager() = default;

  void Reset();

  XfeedQueueManager* infeed() { return &infeed_; }
  XfeedQueueManager* outfeed() { return &outfeed_; }

 private:
  XfeedQueueManager infeed_ = {"infeed"};
  XfeedQueueManager outfeed_ = {"outfeed"};
};

int64_t GetByteSizeRequirement(const Shape& shape, int64_t pointer_size);

}  // namespace runtime
}  // namespace cpu
}  // namespace xla

#endif  // XLA_SERVICE_CPU_XFEED_MANAGER_H_
