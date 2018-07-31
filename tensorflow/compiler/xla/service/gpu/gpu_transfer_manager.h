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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRANSFER_MANAGER_H_

#include <vector>

#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/compiler/xla/service/gpu/infeed_manager.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {
namespace gpu {

// An implementation of the XLA GenericTransferManager that
// handles GPU-specific infeed.
class GpuTransferManager : public GenericTransferManager {
 public:
  GpuTransferManager(se::Platform::Id id, unsigned pointer_size);
  ~GpuTransferManager() override {}

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;
  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    const Shape& literal_shape,
                                    Literal* literal) override;

 private:
  // Initiates the infeed data transfers. InfeedBuffer->Done() must be
  // called to clean up the memory allocated for InfeedBuffer.
  StatusOr<InfeedBuffer> TransferBufferToInfeedInternal(
      se::StreamExecutor* executor, int64 size, const void* source);

  // Enqueues infeed data buffers with the infeed manager after their
  // transfer completes.
  Status EnqueueBuffersToInfeed(se::StreamExecutor* executor,
                                ShapeTree<InfeedBuffer> buffers);

  TF_DISALLOW_COPY_AND_ASSIGN(GpuTransferManager);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_TRANSFER_MANAGER_H_
