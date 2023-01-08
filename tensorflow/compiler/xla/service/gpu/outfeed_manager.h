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

#include <memory>
#include <utility>

#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/xfeed_queue.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/tsl/platform/notification.h"

namespace xla {
namespace gpu {

// TODO(b/30467474) Once GPU outfeed implementation settles, consider
// folding back the cpu and gpu outfeed implementations into a generic
// one if possible.

// Defines a buffer holding the destination for an outfeed in host memory and a
// notification when that triggers when the transfer is done.
class OutfeedBuffer {
 public:
  explicit OutfeedBuffer(int64_t length) : length_(length) {}

  // Waits for the device transfer to be finished.
  void WaitUntilAvailable() { done_.WaitForNotification(); }

  int64_t length() const { return length_; }
  void set_destination(std::unique_ptr<MutableBorrowingLiteral> destination) {
    destination_ = std::move(destination);
  }
  MutableBorrowingLiteral* destination() { return destination_.get(); }

  // Callback to signal that this buffer is consumed.
  void Done() { done_.Notify(); }

 private:
  std::unique_ptr<MutableBorrowingLiteral> destination_;
  const int64_t length_;
  tsl::Notification done_;
};

// Manages a thread-safe queue of buffers. The buffers are supposed to be
// produced by the transfer manager and consumed by the device.
class OutfeedManager
    : public XfeedQueue<ShapeTree<std::unique_ptr<OutfeedBuffer>>*> {
 public:
  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    MutableBorrowingLiteral literal);
};

// Returns the GPU outfeed manager for the given stream executor.
OutfeedManager* GetOrCreateOutfeedManager(se::StreamExecutor* executor);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_OUTFEED_MANAGER_H_
