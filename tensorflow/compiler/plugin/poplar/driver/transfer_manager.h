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

#ifndef TENSORFLOW_COMPILER_XLA_PLUGIN_POPLAR_DRIVER_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_PLUGIN_POPLAR_DRIVER_TRANSFER_MANAGER_H_

#include "tensorflow/compiler/xla/service/cpu/xfeed_manager.h"
#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/core/platform/macros.h"

namespace xla {
namespace poplarplugin {

// This class is a modified version of the xla::cpu::runtime::XfeedQueueManager.
// The reason for creating a separate implementation is to add functionality
// for limiting the number of enqueued buffers and removing elements to make
// room for new ones.
class PoplarXfeedQueueManager {
 public:
  static const size_t DEFAULT_QUEUE_SIZE = std::numeric_limits<size_t>::max();

  PoplarXfeedQueueManager(string queue_name)
      : queue_name_(queue_name),
        max_size_(PoplarXfeedQueueManager::DEFAULT_QUEUE_SIZE) {}

  // Calls the completion callback for any enqueued buffers that have
  // not been dequeued by the runtime, and empties the
  // queue. Reset may not be called while a runtime computation is
  // processing a dequeued buffer. The only safe way to ensure this
  // condition is to call Reset when no computation is taking place.
  void Reset();

  // Adds a buffer to the queue atomically. buffer->Done will be
  // called when the buffer will no longer be accessed by the
  // PoplarXfeedManager, either as a result of a call to Reset or because the
  // runtime has dequeued and used the buffer. If clear_if_full is true then the
  // enqueued buffers are popped and deleted
  Status EnqueueBufferAtomically(cpu::runtime::XfeedBuffer* const buffer,
                                 bool clear_if_full = false);

  // Blocks until the queue is non-empty, then returns the buffer at the head of
  // the queue. Sets the current buffer to be the returned buffer. It is an
  // error to call BlockingDequeueBuffer if there is an unreleased current
  // buffer, i.e., ReleaseCurrentBuffer must be called between calls to
  // BlockingDequeueBuffer.
  cpu::runtime::XfeedBuffer* BlockingDequeueBuffer();

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
  void ReleaseCurrentBuffer(int32 length, void* data, StatusOr<Shape> shape);

  // Sets a maximum size on the fifo the manager owns.
  void set_size(size_t size);

  // Returns the number enqueued buffers.
  size_t size() const;

  // Checks if buffer FIFO size is at maximum size.
  bool full() const;

  // Wait till one or more buffers are available. Returns the number of
  // buffers available.
  size_t WaitForBuffers(size_t num_expected = 1);

 private:
  const string queue_name_;

  mutable tensorflow::mutex mu_;

  // Condition variable that is signaled every time a buffer is
  // enqueued.
  tensorflow::condition_variable item_enqueued_cv_;

  // Condition variable that is signaled every time a buffer is
  // dequeued.
  tensorflow::condition_variable item_dequeued_cv_;

  // XfeedBuffer* queue contents are not owned, but buffer->Done must
  // be called when the buffer is no longer needed by the runtime.
  std::deque<cpu::runtime::XfeedBuffer*> enqueued_buffers_;

  // If non-NULL, the buffer that is currently being processed by the
  // runtime. Not owned.
  cpu::runtime::XfeedBuffer* current_buffer_ = nullptr;

  // Maximum size of the buffer FIFO.
  size_t max_size_;
};

// Client-side class used to enqueue infeed buffers.
class PoplarXfeedManager {
 public:
  PoplarXfeedManager() = default;

  void Reset();

  PoplarXfeedQueueManager* infeed() { return &infeed_; }
  PoplarXfeedQueueManager* outfeed() { return &outfeed_; }

 private:
  PoplarXfeedQueueManager infeed_ = {"infeed"};
  PoplarXfeedQueueManager outfeed_ = {"outfeed"};
};

class PoplarInfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  explicit PoplarInfeedBuffer(int32 length)
      : length_(length),
        buffer_(new char[length]),
        device_memory_(buffer_, length_) {}
  ~PoplarInfeedBuffer() override { delete[] buffer_; }

  int32 length() override { return length_; }
  void* data() override { return buffer_; }
  void Done(StatusOr<Shape> /*shape*/) override { delete this; }

  se::DeviceMemoryBase* device_memory() { return &device_memory_; }

 private:
  int32 length_;
  char* buffer_;
  se::DeviceMemoryBase device_memory_;
};

class PoplarOutfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  PoplarOutfeedBuffer(int32 length, xla::Shape shape)
      : length_(length),
        status_(std::move(shape)),
        destination_(new char[length]) {}

  PoplarOutfeedBuffer(void* destination, int32 length, xla::Shape shape)
      : length_(length), status_(std::move(shape)), destination_(destination) {}

  StatusOr<Shape> WaitForNotification() {
    done_.WaitForNotification();
    return status_;
  }

  int32 length() override { return length_; }
  void* data() override { return destination_; }
  void Done(StatusOr<Shape> shape) override {
    delete[] reinterpret_cast<char*>(destination_);
  }

  StatusOr<Shape> shape() const { return status_; }

 private:
  int32 length_;
  StatusOr<Shape> status_;
  void* destination_;
  tensorflow::Notification done_;
};

class PoplarTransferManager : public GenericTransferManager {
 public:
  PoplarTransferManager();

  ~PoplarTransferManager() override = default;

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;

  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    const Shape& literal_shape,
                                    MutableBorrowingLiteral literal) override;

 private:
  Status TransferBufferToInfeed(se::StreamExecutor* executor, int64 size,
                                const void* source);

  StatusOr<cpu::runtime::XfeedBuffer*> TransferBufferToInfeedInternal(
      se::StreamExecutor* executor, int64 size, const void* source);

  // Helper that transfers a tuple of element buffers from the device's outfeed.
  StatusOr<Shape> TransferTupleBuffersFromOutfeed(
      se::StreamExecutor* executor,
      absl::Span<const std::pair<void*, int64>> buffer_data);

  // Helper that transfers an array buffer from the device's outfeed.
  StatusOr<Shape> TransferArrayBufferFromOutfeed(se::StreamExecutor* executor,
                                                 void* destination,
                                                 int64 size_bytes);

  // On success, returns the shape that was transferred from the outfeed -- if
  // is_tuple is true, the returned shape will be a tuple of the returned shapes
  // for the given buffers.
  StatusOr<Shape> TransferBuffersFromOutfeedInternal(
      se::StreamExecutor* executor,
      absl::Span<const std::pair<void*, int64>> buffer_data, bool is_tuple);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(PoplarTransferManager);
};

}  // namespace poplarplugin
}  // namespace xla

#endif
