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

#include "absl/base/thread_annotations.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/gpu/xfeed_queue.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/types.h"

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

// Client-side class used to enqueue infeed buffers.
class InfeedManager
    : public BlockingXfeedQueue<ShapeTree<se::ScopedDeviceMemory<uint8_t>>> {
 public:
  explicit InfeedManager(se::StreamExecutor* executor);

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal);

 private:
  se::Stream* stream() const { return stream_.get(); }

  // Stream used to enqueue infeed device copies.
  std::unique_ptr<se::Stream> stream_;
};

// Returns the GPU infeed manager for the given stream executor,
InfeedManager* GetOrCreateInfeedManager(se::StreamExecutor* executor);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INFEED_MANAGER_H_
