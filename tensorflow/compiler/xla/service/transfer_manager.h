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
  virtual perftools::gputools::Platform::Id PlatformId() const = 0;

  // Transfers the region into the provided literal using the provided
  // executor. device_shape is the shape, including layout, of the data on the
  // device, while literal_shape will be the shape for the literal. device_shape
  // and literal_shape must be compatible, but need not have the same layout.
  virtual Status TransferLiteralFromDevice(
      perftools::gputools::StreamExecutor* executor,
      const perftools::gputools::DeviceMemoryBase& region,
      const Shape& device_shape, const Shape& literal_shape,
      Literal* literal) = 0;

  // Transfers the given literal into the provided region output parameter,
  // using the given executor.
  virtual Status TransferLiteralToDevice(
      perftools::gputools::StreamExecutor* executor, const Literal& literal,
      perftools::gputools::DeviceMemoryBase* region) = 0;

  // Transfers the given literal into the Infeed interface of the device,
  // using the given executor.
  virtual Status TransferLiteralToInfeed(
      perftools::gputools::StreamExecutor* executor,
      const Literal& literal) = 0;

  // Transfer a memory block of the given size from 'source' buffer to the
  // Infeed interface of the device using the given executor.
  //
  // size is the size to transfer from source in bytes.
  //
  // source is the source data that must be in the target-dependent layout that
  // the Infeed HLO used in the computation expects.
  virtual Status TransferBufferToInfeed(
      perftools::gputools::StreamExecutor* executor, int64 size,
      const void* source) = 0;

  // Transfers the given literal from the Outfeed interface of the device,
  // using the given executor.
  virtual Status TransferLiteralFromOutfeed(
      perftools::gputools::StreamExecutor* executor, const Shape& literal_shape,
      Literal* literal) = 0;

  // Resets the devices associated with this transfer manager.
  virtual Status ResetDevices(
      tensorflow::gtl::ArraySlice<perftools::gputools::StreamExecutor*>
          executor) = 0;

  // Shallow copy a tuple from the device and create a DeviceMemoryBase object
  // for each element in the tuple. A DeviceMemoryBase object refers to the
  // buffer containing the data of that element. The DeviceMemoryBase objects
  // are returned as a vector.
  virtual StatusOr<std::vector<perftools::gputools::DeviceMemoryBase>>
  ShallowCopyTupleFromDevice(
      perftools::gputools::StreamExecutor* executor,
      const perftools::gputools::DeviceMemoryBase& source,
      const Shape& shape) = 0;

  // Returns all buffer pointers that the tuple `source` refers to. Unlike
  // ShallowCopyTupleFromDevice, this function gather buffer pointers in nested
  // tuples as well. Also, the returned DeviceMemoryBase objects are
  // deduplicated.
  StatusOr<std::set<perftools::gputools::DeviceMemoryBase>>
  GatherBufferPointersFromTuple(
      perftools::gputools::StreamExecutor* executor,
      const perftools::gputools::DeviceMemoryBase& source, const Shape& shape);

  // Determines the byte size requirement for the given shape on the underlying
  // architecture. This will be used to allocate an appropriately sized memory
  // region for a host-to-device transfer.
  virtual int64 GetByteSizeRequirement(const Shape& shape) = 0;

  // Transfer a memory block of the given size from the device source into the
  // 'destination' buffer.
  //
  // size is the size to transfer to destination in bytes.
  virtual Status TransferBufferFromDevice(
      perftools::gputools::StreamExecutor* executor,
      const perftools::gputools::DeviceMemoryBase& source, int64 size,
      void* destination);

  // Transfer a memory block of the given size from 'source' buffer to the given
  // destination of the device.
  //
  // size is the size to transfer from source in bytes.
  virtual Status TransferBufferToDevice(
      perftools::gputools::StreamExecutor* executor, int64 size,
      const void* source, perftools::gputools::DeviceMemoryBase* destination);

  typedef std::unique_ptr<TransferManager> (*TransferManagerCreationFunction)();

  /////
  // The TransferManager class also serves as a point to register objects for
  // the various platforms.

  // Registers the TransferManager singleton for the platform kind. This is
  // assumed to be a singleton, so no ownership is transferred.
  //
  // Precondition: a platform kind must not be registered more than once.
  static void RegisterTransferManager(
      perftools::gputools::Platform::Id platform_id,
      TransferManagerCreationFunction transfer_manager);

  // Returns the transfer manager singleton pointer if it is available for the
  // given platform, or an error status if it is not.
  static StatusOr<TransferManager*> GetForPlatform(
      const perftools::gputools::Platform* platform);

 private:
  // Routine that returns the mutex that guards the
  // platform-to-transfer manager map.  Done as a routine to
  // ensure correct initialization ordering, since RegisterTransferManager
  // can be called during program initialization time.
  static tensorflow::mutex* platform_transfer_manager_mutex();

  // State kept for each kind of TransferManager.  Registration functions
  // set up creation_function, and then we use that to lazily create
  // "manager" the first time GetForPlatform is invoked for a particular id.
  struct State {
    std::unique_ptr<TransferManager> manager;
    TransferManagerCreationFunction creation_function = nullptr;
  };

  // Map from platform kind to transfer manager singleton.
  static std::map<perftools::gputools::Platform::Id, State>*
  GetPlatformTransferManagers();
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TRANSFER_MANAGER_H_
