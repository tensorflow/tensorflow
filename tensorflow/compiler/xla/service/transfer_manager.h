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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TRANSFER_MANAGER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TRANSFER_MANAGER_H_

#include <map>
#include <set>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// The TransferManager interface lets backends provide platform-specific
// mechanisms for constructing literals from given device memory handles.
// This lets each platform customize how literals are transferred to/from the
// device in terms of padding, leading dimension, etc.
class TransferManager {
 public:
  virtual ~TransferManager() {}

  // Returns the ID of the platform that this transfer manager acts on.
  virtual se::Platform::Id PlatformId() const = 0;

  // Returns the shape of the on-device representation for the given shape on
  // the host. This is intended for use with ShapedBuffer where buffers are
  // pre-allocated by the host, e.g. TransferLiteralToDevice, without the user
  // needing to consider device-specific behaviors.
  virtual Shape HostShapeToDeviceShape(const Shape& host_shape) const {
    return host_shape;
  }

  // Returns a literal containing the data held in the given ShapedBuffer.
  // using the provided executor. The optional literal_shape will be the shape
  // for the literal. The shape of the ShapedBuffer and
  // DeviceShape(literal_shape) must be compatible, but need not have the same
  // layout.
  virtual StatusOr<std::unique_ptr<Literal>> TransferLiteralFromDevice(
      se::StreamExecutor* executor, const ShapedBuffer& device_buffer) = 0;

  // Transfers the given literal into the previously allocated device memory
  // represented by the given ShapedBuffer using the given executor. The shape
  // of the ShapedBuffer and DeviceShape(literal.shape()) must be compatible,
  // but need not have the same layout
  virtual Status TransferLiteralToDevice(se::StreamExecutor* executor,
                                         const Literal& literal,
                                         const ShapedBuffer& device_buffer) = 0;

  // Convenience methods for transferring an array to or from the device at a
  // known address. This avoids having to construct a ShapedBuffer just to
  // transfer an array at a known address.
  Status TransferArrayToDevice(se::StreamExecutor* executor,
                               const Literal& literal,
                               const se::DeviceMemoryBase& dest);
  StatusOr<std::unique_ptr<Literal>> TransferArrayFromDevice(
      se::StreamExecutor* executor, const Shape& shape,
      const se::DeviceMemoryBase& source);

  // Transfers the given literal into the Infeed interface of the device,
  // using the given executor.
  virtual Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                         const Literal& literal) = 0;

  // Transfers the given literal from the Outfeed interface of the device,
  // using the given executor.
  virtual Status TransferLiteralFromOutfeed(se::StreamExecutor* executor,
                                            const Shape& literal_shape,
                                            Literal* literal) = 0;

  // Resets the devices associated with this transfer manager.
  virtual Status ResetDevices(
      tensorflow::gtl::ArraySlice<se::StreamExecutor*> executor) = 0;

  // Given an allocated ShapedBuffer, constructs the tuple index table(s) in
  // each buffer of the given ShapedBuffer corresponding to tuple shapes. If the
  // ShapedBuffer is array-shaped this method does nothing.
  Status WriteTupleIndexTables(se::StreamExecutor* executor,
                               const ShapedBuffer& device_buffer);

  // Determines the byte size requirement for the given shape on the underlying
  // architecture. This will be used to allocate an appropriately sized memory
  // region for a host-to-device transfer.
  virtual int64 GetByteSizeRequirement(const Shape& shape) const = 0;

  // Allocate a ShapedBuffer which can hold data with the given on-host
  // shape. The on-device shape may be different as indicated by
  // HostShapeToDeviceShape.
  StatusOr<ShapedBuffer> AllocateShapedBuffer(const Shape& on_host_shape,
                                              DeviceMemoryAllocator* allocator,
                                              int device_ordinal);
  StatusOr<ScopedShapedBuffer> AllocateScopedShapedBuffer(
      const Shape& on_host_shape, DeviceMemoryAllocator* allocator,
      int device_ordinal);

  /////
  // The TransferManager class also serves as a point to register objects for
  // the various platforms.

  // Registers the TransferManager singleton for the platform kind. This is
  // assumed to be a singleton, so no ownership is transferred.
  //
  // Precondition: a platform kind must not be registered more than once.
  typedef std::unique_ptr<TransferManager> (*TransferManagerCreationFunction)();
  static void RegisterTransferManager(
      se::Platform::Id platform_id,
      TransferManagerCreationFunction transfer_manager);

  // Returns the transfer manager singleton pointer if it is available for the
  // given platform, or an error status if it is not.
  static StatusOr<TransferManager*> GetForPlatform(
      const se::Platform* platform);

 protected:
  // Transfer a memory block of the given size from 'source' buffer to the
  // Infeed interface of the device using the given executor.
  //
  // size is the size to transfer from source in bytes.
  //
  // source is the source data that must be in the target-dependent layout that
  // the Infeed HLO used in the computation expects.
  virtual Status TransferBufferToInfeed(se::StreamExecutor* executor,
                                        int64 size, const void* source) = 0;

  // Transfer a memory block of the given size from the device source into the
  // 'destination' buffer.
  //
  // size is the size to transfer to destination in bytes.
  virtual Status TransferBufferFromDevice(se::StreamExecutor* executor,
                                          const se::DeviceMemoryBase& source,
                                          int64 size, void* destination);

  // Transfer a memory block of the given size from 'source' buffer to the given
  // destination of the device.
  //
  // size is the size to transfer from source in bytes.
  virtual Status TransferBufferToDevice(se::StreamExecutor* executor,
                                        int64 size, const void* source,
                                        se::DeviceMemoryBase* destination);

  // Writes the given device-memory pointers in 'elements' to the given region
  // to construct a tuple index table in the platform-specific tuple
  // representation.
  virtual Status WriteSingleTupleIndexTable(
      se::StreamExecutor* executor,
      tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> elements,
      const Shape& shape, se::DeviceMemoryBase* region) = 0;

 private:
  // The mutex that guards the platform-to-transfer manager map.
  static tensorflow::mutex platform_transfer_manager_mutex_;

  // State kept for each kind of TransferManager.  Registration functions
  // set up creation_function, and then we use that to lazily create
  // "manager" the first time GetForPlatform is invoked for a particular id.
  struct State {
    std::unique_ptr<TransferManager> manager;
    TransferManagerCreationFunction creation_function = nullptr;
  };

  // Map from platform kind to transfer manager singleton.
  static std::map<se::Platform::Id, State>* GetPlatformTransferManagers();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TRANSFER_MANAGER_H_
