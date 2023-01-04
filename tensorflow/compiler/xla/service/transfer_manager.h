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

#include "absl/container/flat_hash_map.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/stream_executor/device_memory.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"

namespace xla {

// The TransferManager interface lets backends provide platform-specific
// mechanisms for constructing literals from given device memory handles.
// This lets each platform customize how literals are transferred to/from the
// device in terms of padding, leading dimension, etc.
class TransferManager {
 public:
  virtual ~TransferManager() = default;

  // Returns the ID of the platform that this transfer manager acts on.
  virtual se::Platform::Id PlatformId() const = 0;

  // Returns the shape of the on-device representation for the given shape on
  // the host. This is intended for use with ShapedBuffer where buffers are
  // pre-allocated by the host, e.g. TransferLiteralToDevice, without the user
  // needing to consider device-specific behaviors.
  virtual Shape HostShapeToDeviceShape(const Shape& host_shape) const {
    // Strips off any preexisting tiling or memory space information.
    // TODO(phawkins): fix clients not to including tiling or memory space
    // information in shapes passed to this function and turn this into an
    // assertion.
    return ShapeUtil::DeviceShapeToHostShape(host_shape);
  }

  // Base class for specifying platform specific transfer metadata that can be
  // used to tell the underlying implementation to perform specific optimization
  // to a transfer. Actual metadata passed to supported transfer methods should
  // subclass this class.
  class TransferMetadata {
   public:
    virtual ~TransferMetadata() = default;
  };
  // Returns a literal containing the data held in the given ShapedBuffer
  // using the provided executor. This operation is performed synchronously
  // without waiting for any other operation on a stream to complete.
  //
  // This function should be avoided in favor of the asynchronous version below.
  //
  // Optionally caller can specify platform-specific transfer metadata that
  // tells the actual implementation to do something special.
  StatusOr<Literal> TransferLiteralFromDevice(
      se::Stream* stream, const ShapedBuffer& device_buffer,
      const TransferMetadata* transfer_metadata = nullptr);

  Status TransferLiteralFromDevice(
      se::Stream* stream, const ShapedBuffer& device_buffer,
      const MutableBorrowingLiteral& literal,
      const TransferMetadata* transfer_metadata = nullptr);

  // Begins transferring a literal containing the data held in the given
  // ShapedBuffer using the provided executor.
  //
  // This operation is performed asynchronously on the given stream. It returns
  // once the transfer is enqueued. 'done' is invoked with the result when
  // complete.
  //
  // device_buffer is copied by reference and must live at least until done() is
  // invoked.
  //
  // Optionally caller can specify platform-specific transfer metadata that
  // tells the actual implementation to do something special.
  virtual void TransferLiteralFromDevice(
      se::Stream* stream, const ShapedBuffer& device_buffer,
      MutableBorrowingLiteral literal, std::function<void(Status)> done,
      const TransferMetadata* transfer_metadata) = 0;

  void TransferLiteralFromDevice(se::Stream* stream,
                                 const ShapedBuffer& device_buffer,
                                 MutableBorrowingLiteral literal,
                                 std::function<void(Status)> done) {
    return TransferLiteralFromDevice(stream, device_buffer, literal, done,
                                     nullptr);
  }

  // Transfers the given literal into the previously allocated device memory
  // represented by the given ShapedBuffer using the given executor. The shape
  // of the ShapedBuffer and DeviceShape(literal.shape()) must be compatible,
  // but need not have the same layout.
  //
  // This operation is performed synchronously without waiting for any other
  // operation on a stream to complete. This function should be avoided in favor
  // of the asynchronous version below.
  //
  // Optionally caller can specify platform-specific transfer metadata that
  // tells the actual implementation to do something special.
  Status TransferLiteralToDevice(
      se::Stream* stream, const LiteralSlice& literal,
      const ShapedBuffer& device_buffer,
      const TransferMetadata* transfer_metadata = nullptr);

  // Transfers the given literal into the previously allocated device memory
  // represented by the given ShapedBuffer using the given executor. The shape
  // of the ShapedBuffer and DeviceShape(literal.shape()) must be compatible,
  // but need not have the same layout.
  //
  // This operation is performed asynchronously on the given stream. It returns
  // once the transfer is enqueued, and may return before the transfer has
  // completed.
  //
  // The caller may free the data structures 'literal' and 'device_buffer'
  // immediately after this function returns, however their constituent buffers
  // on both host and device must remain valid until the enqueued transfer has
  // completed on 'stream'.
  //
  // Optionally caller can specify platform-specific transfer metadata that
  // tells the actual implementation to do something special.
  virtual Status TransferLiteralToDeviceAsync(
      se::Stream* stream, const LiteralSlice& literal,
      const ShapedBuffer& device_buffer,
      const TransferMetadata* transfer_metadata) = 0;

  Status TransferLiteralToDeviceAsync(se::Stream* stream,
                                      const LiteralSlice& literal,
                                      const ShapedBuffer& device_buffer) {
    return TransferLiteralToDeviceAsync(stream, literal, device_buffer,
                                        nullptr);
  }

  // Convenience methods for transferring an array to or from the device at a
  // known address. This avoids having to construct a ShapedBuffer just to
  // transfer an array at a known address.
  //
  // Optionally caller can specify platform-specific transfer metadata that
  // tells the actual implementation to do something special.
  Status TransferArrayToDevice(
      se::Stream* stream, const LiteralSlice& literal,
      const se::DeviceMemoryBase& dest,
      const TransferMetadata* transfer_metadata = nullptr);

  Status TransferArrayToDeviceAsync(
      se::Stream* stream, const LiteralSlice& literal,
      const se::DeviceMemoryBase& dest,
      const TransferMetadata* transfer_metadata = nullptr);

  StatusOr<Literal> TransferArrayFromDevice(
      se::Stream* stream, const Shape& shape,
      const se::DeviceMemoryBase& source,
      const TransferMetadata* transfer_metadata = nullptr);

  // Read from a device buffer and update the dynamic dimension sizes of
  // `host_shape` and `device_shape`. The function takes in bounded dynamic
  // shapes, and returns static shapes with dynamic shapes updated.
  // The shape of the buffer also have to be compatible with the host shape and
  // device shape.
  virtual Status ReadDynamicShapes(se::Stream* stream,
                                   ShapedBuffer* device_buffer,
                                   Shape* device_shape);

  // Transfers the given literal into the Infeed interface of the device,
  // using the given executor.
  virtual Status TransferLiteralToInfeed(se::StreamExecutor* executor,
                                         const LiteralSlice& literal) = 0;

  // Transfers the given literal from the Outfeed interface of the device,
  // using the given executor. The shape and layout are determined by the
  // shape and layout of `literal`.
  virtual Status TransferLiteralFromOutfeed(
      se::StreamExecutor* executor, MutableBorrowingLiteral literal) = 0;

  // Resets the devices associated with this transfer manager.
  virtual Status ResetDevices(
      absl::Span<se::StreamExecutor* const> executor) = 0;

  // Given an allocated ShapedBuffer, constructs the tuple index table(s) in
  // each buffer of the given ShapedBuffer corresponding to tuple shapes. If the
  // ShapedBuffer is array-shaped this method does nothing.
  Status WriteTupleIndexTables(se::Stream* stream,
                               const ShapedBuffer& device_buffer);
  Status WriteTupleIndexTablesAsync(se::Stream* stream,
                                    const ShapedBuffer& device_buffer);

  // Writes a tuple index buffer for the root of 'device_buffer', which must
  // be a tuple. Unlike WriteTupleIndexTables, only writes the root buffer,
  // rather than writing all subbuffers. This method is always asynchronous.
  Status WriteRootTupleIndexTable(se::Stream* stream,
                                  const ShapedBuffer& device_buffer);
  Status WriteRootTupleIndexTable(
      se::Stream* stream,
      const ShapeTree<MaybeOwningDeviceMemory>& buffer_tree);

  // Determines the byte size requirement for the given shape on the underlying
  // architecture. This will be used to allocate an appropriately sized memory
  // region for a host-to-device transfer.
  virtual int64_t GetByteSizeRequirement(const Shape& shape) const = 0;

  // Chooses a compact layout for 'shape', ignoring any existing layout on
  // 'shape'. What "reasonable" means is left up to the backend. The
  // intended use case is to choose a layout that avoids excessive padding on
  // devices that have tiled memory architectures.
  // The default implementation always picks a default (major-to-minor) layout.
  // Fails if 'shape' cannot be represented by the device.
  virtual StatusOr<Shape> ChooseCompactLayoutForShape(
      const Shape& host_shape) const;

  // For the given shape, chooses a layout for infeed. The returned shape
  // has the same dimensions as the original shape, and only the layout is
  // changed.
  virtual Shape ChooseGoodInfeedLayout(const Shape& shape) const;

  typedef std::function<Shape(const Shape&)> DeviceShapeRepresentationFn;

  // Allocates a ScopedShapedBuffer which can hold data with the given on-host
  // shape. The on-device shape may be different as indicated by
  // HostShapeToDeviceShape.
  StatusOr<ScopedShapedBuffer> AllocateScopedShapedBuffer(
      const Shape& on_host_shape, se::DeviceMemoryAllocator* allocator,
      int device_ordinal,
      DeviceShapeRepresentationFn shape_representation_fn = nullptr);

  // The given ShapedBuffer holds a handle to allocated memory, but it is not
  // in the general case legal to immediately copy or access that allocated
  // memory because queued operations on the device may alias that memory.
  // Memory ordering is enforced by the Stream's happens-before relationship
  // which allows eager deallocation and reallocation of buffers host-side even
  // if the device hasn't finished with them.
  //
  // In certain cases, it can be known that a ShapedBuffer does not have any
  // conflicting accesses on the device and thus is eligible to be accessed at
  // any time from the host.
  //
  // This function returns true if device_buffer can be accessed immediately
  // without waiting for the Stream's previously enqueued items. This only
  // returns true if all subbuffers in device_buffer can be accessed
  // immediately.
  virtual bool CanShapedBufferBeAccessedNow(
      se::StreamExecutor* executor, const ShapedBuffer& device_buffer) const {
    return false;
  }

  // Equivalent to CanShapedBufferBeAccessedNow but for a single device buffer.
  virtual bool CanBufferBeAccessedNow(
      se::StreamExecutor* executor,
      const se::DeviceMemoryBase& device_buffer) const {
    return false;
  }

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

  // Writes the given device-memory pointers in 'elements' to the given region
  // to construct a tuple index table in the platform-specific tuple
  // representation.
  virtual Status WriteSingleTupleIndexTable(
      se::Stream* stream, absl::Span<const se::DeviceMemoryBase> elements,
      const Shape& shape, se::DeviceMemoryBase* region) = 0;

 protected:
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

 private:
  // The mutex that guards the platform-to-transfer manager map.
  static absl::Mutex platform_transfer_manager_mutex_;

  // State kept for each kind of TransferManager.  Registration functions
  // set up creation_function, and then we use that to lazily create
  // "manager" the first time GetForPlatform is invoked for a particular id.
  struct State {
    std::unique_ptr<TransferManager> manager;
    TransferManagerCreationFunction creation_function = nullptr;
  };

  // Map from platform kind to transfer manager singleton.
  static absl::flat_hash_map<se::Platform::Id, State>*
  GetPlatformTransferManagers();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TRANSFER_MANAGER_H_
