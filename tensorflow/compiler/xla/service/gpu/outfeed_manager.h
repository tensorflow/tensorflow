/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OUTFEED_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OUTFEED_MANAGER_H_

#include <deque>
#include <vector>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/notification.h"

namespace xla {
namespace gpu {

// TODO(b/30467474) Once GPU outfeed implementation settles, consider
// folding back the cpu and gpu outfeed implementations into a generic
// one if possible.

// Defines a buffer holding the destination for an outfeed in host memory and a
// notification when that triggers when the transfer is done.
class OutfeedBuffer {
 public:
  OutfeedBuffer(int64 length) : length_(length) {}

  // Waits for the device transfer to be finished.
  std::unique_ptr<Literal> WaitUntilAvailable() {
    done_.WaitForNotification();
    return std::move(destination_);
  }

  int64 length() const { return length_; }
  void set_destination(std::unique_ptr<Literal> destination) {
    destination_ = std::move(destination);
  }
  Literal* destination() { return destination_.get(); }

  // Callback to signal that this buffer is consumed.
  void Done() { done_.Notify(); }

 private:
  std::unique_ptr<Literal> destination_;
  const int64 length_;
  tensorflow::Notification done_;
};

// Manages a thread-safe queue of buffers. The buffers are supposed to be
// produced by the transfer manager and consumed by the device.
class OutfeedManager {
 public:
  // Adds a tree of buffers to the queue. The individual buffers correspond to
  // the elements of a tuple and may be nullptr if the buffer is a tuple index
  // buffer.
  void EnqueueOutfeedDestination(
      ShapeTree<std::unique_ptr<OutfeedBuffer>>* buffers);

  // Blocks until the queue is non-empty, then returns the buffer at the head of
  // the queue.
  ShapeTree<std::unique_ptr<OutfeedBuffer>>*
  BlockingGetNextOutfeedDestination();

 private:
  tensorflow::mutex mu_;

  // Condition variable that is signaled every time a buffer is enqueued.
  tensorflow::condition_variable cv_;

  // The queue of trees of buffers. OutfeedBuffer* queue contents are not owned.
  std::deque<ShapeTree<std::unique_ptr<OutfeedBuffer>>*> enqueued_buffers_;
};

// Singleton creator-or-accessor: Returns the GPU outfeed manager.
OutfeedManager* GetOrCreateOutfeedManager();

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OUTFEED_MANAGER_H_
