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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_TRANSFER_MANAGER_H_

#include <vector>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/cpu/xfeed_manager.h"
#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/stream_executor/device_memory.h"

namespace xla {

// An implementation of the XLA GenericTransferManager that
// handles CPU-specific infeed.
class CpuTransferManager : public GenericTransferManager {
 public:
  CpuTransferManager();
  ~CpuTransferManager() override {}

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;
  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    MutableBorrowingLiteral literal) override;

  bool CanShapedBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const ShapedBuffer& device_buffer) const override {
    return true;
  }

  bool CanBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const se::DeviceMemoryBase& device_buffer) const override {
    return true;
  }

  Status ReadDynamicShapes(se::Stream* stream, ShapedBuffer* device_buffer,
                           Shape* device_shape) override;

 private:
  Status TransferBufferToInfeed(se::StreamExecutor* executor, int64 size,
                                const void* source);

  // Transfers infeed data to device. InfeedBuffer->Done() must be
  // called to clean up the memory allocated for InfeedBuffer.
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

  TF_DISALLOW_COPY_AND_ASSIGN(CpuTransferManager);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CPU_CPU_TRANSFER_MANAGER_H_
