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
#include "tensorflow/compiler/xla/service/gpu/xfeed_queue.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/types.h"
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
  InfeedBuffer() = default;
  InfeedBuffer(se::StreamExecutor* executor, int64 length)
      : device_memory_(executor, executor->AllocateArray<uint8>(length)),
        length_(length) {
    CHECK(!device_memory_->is_null());
  }

  int64 length() const { return length_; }

  se::DeviceMemoryBase* device_memory() { return device_memory_.ptr(); }

 private:
  se::ScopedDeviceMemory<uint8> device_memory_;
  int64 length_;
};

// Client-side class used to enqueue infeed buffers.
class InfeedManager : public XfeedQueue<ShapeTree<InfeedBuffer>> {
 public:
  // Returns a cached stream associated with an executor. Allocates a
  // new stream on the first invocation. On subsequent invocations, if
  // the cached executor is not the same as the requested executor,
  // returns null.
  se::Stream* GetStream(se::StreamExecutor* executor);

 private:
  // Mutex for serializing the creation of host_to_device_stream_.
  tensorflow::mutex host_to_device_stream_mu_;

  // Cached host to device stream for queuing infeed data.
  std::unique_ptr<se::Stream> host_to_device_stream_
      ABSL_GUARDED_BY(host_to_device_stream_mu_);

  // Executor that the host_to_device_stream belongs to. Not owned.
  se::StreamExecutor* host_to_device_executor_ = nullptr;
};

// Singleton creator-or-accessor: Returns the GPU infeed manager.
InfeedManager* GetOrCreateInfeedManager();

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INFEED_MANAGER_H_
