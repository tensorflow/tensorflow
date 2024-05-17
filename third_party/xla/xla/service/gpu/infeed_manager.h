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

// This header declares classes for the infeed manager and the infeed
// buffer that are used by the GPU runtime to transfer buffers into an
// executing GPU computation, e.g., to feed data into a while loop.

#ifndef XLA_SERVICE_GPU_INFEED_MANAGER_H_
#define XLA_SERVICE_GPU_INFEED_MANAGER_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"
#include "xla/literal.h"
#include "xla/service/gpu/xfeed_queue.h"
#include "xla/shape_tree.h"
#include "xla/stream_executor/device_memory_handle.h"
#include "xla/stream_executor/stream_executor.h"

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
    : public BlockingXfeedQueue<ShapeTree<se::DeviceMemoryHandle>> {
 public:
  explicit InfeedManager(se::StreamExecutor* executor);

  absl::Status TransferLiteralToInfeed(se::StreamExecutor* executor,
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

#endif  // XLA_SERVICE_GPU_INFEED_MANAGER_H_
