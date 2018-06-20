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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GENERIC_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GENERIC_TRANSFER_MANAGER_H_

#include <vector>

#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// A generic implementation of the XLA TransferManager interface
// that is the base class for both CPU and GPU. For GPU, it transfers
// data between host and device (GPU). For CPU, since the "device"
// here is the host itself, there's not much for this transfer manager
// to do except memcpy the result. There is a CpuTransferManager that
// inherits from GenericTransferManager and handles CPU-specific
// infeed.
class GenericTransferManager : public TransferManager {
 public:
  GenericTransferManager(se::Platform::Id platform_id, size_t pointer_size);
  ~GenericTransferManager() override {}

  se::Platform::Id PlatformId() const override;

  StatusOr<std::unique_ptr<Literal>> TransferLiteralFromDevice(
      se::StreamExecutor* executor, const ShapedBuffer& device_buffer) override;

  Status TransferLiteralToDevice(se::StreamExecutor* executor,
                                 const LiteralSlice& literal,
                                 const ShapedBuffer& device_buffer) override;

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;
  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    const Shape& literal_shape,
                                    Literal* literal) override;

  Status ResetDevices(
      tensorflow::gtl::ArraySlice<se::StreamExecutor*> executors) override;

  int64 GetByteSizeRequirement(const Shape& shape) const override;

 protected:
  Status TransferBufferToInfeed(se::StreamExecutor* executor, int64 size,
                                const void* source) override;

  Status WriteSingleTupleIndexTable(
      se::StreamExecutor* executor,
      tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> elements,
      const Shape& shape, se::DeviceMemoryBase* region) override;

 private:
  // The platform this transfer manager targets.
  const se::Platform::Id platform_id_;

  // The size in bytes of pointers on this platform.
  const size_t pointer_size_;

  TF_DISALLOW_COPY_AND_ASSIGN(GenericTransferManager);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GENERIC_TRANSFER_MANAGER_H_
