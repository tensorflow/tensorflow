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

#ifndef XLA_SERVICE_GENERIC_TRANSFER_MANAGER_H_
#define XLA_SERVICE_GENERIC_TRANSFER_MANAGER_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>

#include "absl/base/thread_annotations.h"
#include "absl/container/node_hash_map.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/memory_allocation.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/xla_data.pb.h"

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
  struct LiteralFromDeviceMetadata : public TransferManager::TransferMetadata {
    bool callback_is_host_callback_safe = false;
  };

  GenericTransferManager(se::Platform::Id platform_id, size_t pointer_size);

  se::Platform::Id PlatformId() const override;

  void TransferLiteralFromDevice(
      se::Stream* stream, const ShapedBuffer& device_buffer,
      MutableBorrowingLiteral literal, std::function<void(Status)> done,
      const TransferMetadata* transfer_metadata) override;

  Status TransferLiteralToDeviceAsync(
      se::Stream* stream, const LiteralSlice& literal,
      const ShapedBuffer& device_buffer,
      const TransferMetadata* transfer_metadata) override;

  Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                 const LiteralSlice& literal) override;

  Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                    MutableBorrowingLiteral literal) override;

  Status ResetDevices(absl::Span<se::StreamExecutor* const> executors) override;

  int64_t GetByteSizeRequirement(const Shape& shape) const override;

  Status WriteSingleTupleIndexTable(
      se::Stream* stream, absl::Span<const se::DeviceMemoryBase> elements,
      const Shape& shape, se::DeviceMemoryBase* region) override;

  Shape HostShapeToDeviceShape(const Shape& host_shape) const override;

 private:
  // Returns whether subbyte types (types less than 1 byte, e.g. U4) should
  // have multiple values packed into a single byte on the device. Subbyte
  // bytes are never packed on the host. By default, returns false, so a byte
  // can only hold one value, but subclasses can override this.
  virtual bool PackSubbyteTypes() const { return false; }

  // Transfer a memory block of the given size from the device source into the
  // 'destination' buffer.
  //
  // size is the size to transfer to destination in bytes.
  virtual Status TransferBufferFromDevice(se::Stream* stream,
                                          const se::DeviceMemoryBase& source,
                                          int64_t size, void* destination);

  // Transfer a memory block of the given size from 'source' buffer to the given
  // destination of the device.
  //
  // size is the size to transfer from source in bytes.
  virtual Status TransferBufferToDevice(se::Stream* stream, int64_t size,
                                        const void* source,
                                        se::DeviceMemoryBase* destination);

  // Transfers a buffer of packed int4 values from the device to the host, then
  // unpacks them on the host. 'source' is a buffer with (num_elements+1)/2
  // bytes where each byte stores two int4 values. 'destination' is a buffer
  // with num_elements bytes, where a single int4 value will be written to each
  // byte in the lower 4 bits.
  virtual Status TransferInt4ArrayFromDevice(se::Stream* stream,
                                             const se::DeviceMemoryBase& source,
                                             int64_t num_elements,
                                             void* destination);

  // Packs an array of int4 values then transfers the packed buffer from the
  // host to the device. 'source' is a buffer with num_elements bytes, where the
  // lower 4 bits of each byte stores an int4 value. 'destination' is a buffer
  // with (num_elements+1)/2 bytes, where two int4 values will be written into
  // each byte.
  virtual Status TransferInt4ArrayToDevice(se::Stream* stream,
                                           int64_t num_elements,
                                           const void* source,
                                           se::DeviceMemoryBase* destination);

  // The platform this transfer manager targets.
  const se::Platform::Id platform_id_;

  // The size in bytes of pointers on this platform.
  const size_t pointer_size_;

  GenericTransferManager(const GenericTransferManager&) = delete;
  GenericTransferManager& operator=(const GenericTransferManager&) = delete;
};

}  // namespace xla

#endif  // XLA_SERVICE_GENERIC_TRANSFER_MANAGER_H_
