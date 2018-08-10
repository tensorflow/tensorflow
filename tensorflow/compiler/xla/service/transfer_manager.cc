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

#include "tensorflow/compiler/xla/service/transfer_manager.h"

#include <string>
#include <utility>

#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/notification.h"

using ::tensorflow::strings::StrCat;

namespace xla {
/* static */ tensorflow::mutex
    TransferManager::platform_transfer_manager_mutex_(
        tensorflow::LINKER_INITIALIZED);

/* static */ std::map<se::Platform::Id, TransferManager::State>*
TransferManager::GetPlatformTransferManagers() {
  static auto* r = new std::map<se::Platform::Id, TransferManager::State>;
  return r;
}

StatusOr<std::unique_ptr<Literal>> TransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
  StatusOr<std::unique_ptr<Literal>> ret;

  se::Stream* substream = stream->GetOrCreateSubStream();
  substream->ThenWaitFor(stream);
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });

  tensorflow::Notification n;
  Status s;
  Literal literal(device_buffer.on_host_shape());
  TransferLiteralFromDevice(substream, device_buffer, literal,
                            [&](Status status) {
                              s = status;
                              n.Notify();
                            });
  n.WaitForNotification();
  if (!s.ok()) {
    return s;
  }
  return MakeUnique<Literal>(std::move(literal));
}

Status TransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    const MutableBorrowingLiteral& literal) {
  se::Stream* substream = stream->GetOrCreateSubStream();
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });

  Status ret;
  tensorflow::Notification n;
  TransferLiteralFromDevice(substream, device_buffer, literal,
                            [&](Status status) {
                              ret = status;
                              n.Notify();
                            });
  n.WaitForNotification();
  return ret;
}

Status TransferManager::TransferLiteralToDevice(
    se::Stream* stream, const LiteralSlice& literal,
    const ShapedBuffer& device_buffer) {
  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  se::Stream* substream = stream->GetOrCreateSubStream();
  substream->ThenWaitFor(stream);
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });
  TF_RETURN_IF_ERROR(
      TransferLiteralToDeviceAsync(substream, literal, device_buffer));
  return substream->BlockHostUntilDone();
}

StatusOr<std::unique_ptr<Literal>> TransferManager::TransferArrayFromDevice(
    se::Stream* stream, const Shape& shape,
    const se::DeviceMemoryBase& source) {
  StatusOr<std::unique_ptr<Literal>> ret;
  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  se::Stream* substream = stream->GetOrCreateSubStream();
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });

  tensorflow::Notification n;
  Literal literal(shape);
  Status s;
  TransferArrayFromDevice(substream, shape, source, literal,
                          [&](Status status) {
                            s = status;
                            n.Notify();
                          });
  n.WaitForNotification();
  if (!s.ok()) {
    return s;
  }
  return MakeUnique<Literal>(std::move(literal));
}

Status TransferManager::TransferArrayToDevice(
    se::Stream* stream, const LiteralSlice& literal,
    const se::DeviceMemoryBase& dest) {
  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  se::Stream* substream = stream->GetOrCreateSubStream();
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });
  TF_RETURN_IF_ERROR(TransferArrayToDeviceAsync(substream, literal, dest));
  return substream->BlockHostUntilDone();
}

Status TransferManager::TransferArrayToDeviceAsync(
    se::Stream* stream, const LiteralSlice& literal,
    const se::DeviceMemoryBase& dest) {
  const Shape on_device_shape = HostShapeToDeviceShape(literal.shape());
  TF_RET_CHECK(ShapeUtil::IsArray(on_device_shape))
      << "On-device representation of "
      << ShapeUtil::HumanString(literal.shape())
      << " is not an array: " << ShapeUtil::HumanString(on_device_shape);
  if (dest.size() < GetByteSizeRequirement(on_device_shape)) {
    return FailedPrecondition(
        "Allocation on device not large enough for array: "
        "%lld < %lld",
        dest.size(), GetByteSizeRequirement(on_device_shape));
  }
  ShapedBuffer shaped_buffer(/*on_host_shape=*/literal.shape(), on_device_shape,
                             stream->parent()->platform(),
                             stream->parent()->device_ordinal());
  shaped_buffer.set_buffer(dest, /*index=*/{});
  return TransferLiteralToDevice(stream, literal, shaped_buffer);
}

void TransferManager::TransferArrayFromDevice(
    se::Stream* stream, const Shape& shape, const se::DeviceMemoryBase& source,
    const MutableBorrowingLiteral& literal, std::function<void(Status)> done) {
  if (!ShapeUtil::Equal(HostShapeToDeviceShape(shape), shape)) {
    auto error = StrCat("Shape ", ShapeUtil::HumanString(shape),
                        " has a differently shaped representation on-device: ",
                        ShapeUtil::HumanString(HostShapeToDeviceShape(shape)));
    return done(FailedPrecondition("%s", error.c_str()));
  }
  if (source.size() < GetByteSizeRequirement(shape)) {
    return done(
        FailedPrecondition("Allocation on device not large enough for array: "
                           "%lld < %lld",
                           source.size(), GetByteSizeRequirement(shape)));
  }
  ShapedBuffer shaped_buffer(/*on_host_shape=*/shape, shape,
                             stream->parent()->platform(),
                             stream->parent()->device_ordinal());
  shaped_buffer.set_buffer(source, /*index=*/{});
  return TransferLiteralFromDevice(stream, shaped_buffer, literal,
                                   std::move(done));
}

/* static */ void TransferManager::RegisterTransferManager(
    se::Platform::Id platform_id,
    TransferManagerCreationFunction creation_function) {
  tensorflow::mutex_lock lock(
      TransferManager::platform_transfer_manager_mutex_);
  auto* managers = GetPlatformTransferManagers();
  CHECK(managers->find(platform_id) == managers->end());
  (*managers)[platform_id].creation_function = creation_function;
}

/* static */ StatusOr<TransferManager*> TransferManager::GetForPlatform(
    const se::Platform* platform) {
  tensorflow::mutex_lock lock(
      TransferManager::platform_transfer_manager_mutex_);
  auto* managers = GetPlatformTransferManagers();

  auto it = managers->find(platform->id());
  if (it == managers->end()) {
    return NotFound(
        "could not find registered transfer manager for platform %s -- check "
        "target linkage",
        platform->Name().c_str());
  }

  if (it->second.manager == nullptr) {
    // Lazily create the transfer manager the first time it is needed
    it->second.manager = (*it->second.creation_function)();
  }

  return it->second.manager.get();
}

Status TransferManager::WriteTupleIndexTables(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
  TF_RETURN_IF_ERROR(WriteTupleIndexTablesAsync(stream, device_buffer));
  return stream->BlockHostUntilDone();
}

Status TransferManager::WriteTupleIndexTablesAsync(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
  VLOG(2) << "Writing tuple index tables for " << device_buffer;

  return ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_device_shape(),
      [&](const Shape& device_subshape, const ShapeIndex& index) -> Status {
        if (ShapeUtil::IsTuple(device_subshape)) {
          se::DeviceMemoryBase device_memory = device_buffer.buffer(index);
          TF_RET_CHECK(GetByteSizeRequirement(device_subshape) ==
                       device_memory.size());

          std::vector<se::DeviceMemoryBase> elements;
          ShapeIndex element_index = index;
          for (int64 i = 0; i < ShapeUtil::TupleElementCount(device_subshape);
               ++i) {
            element_index.push_back(i);
            elements.push_back(device_buffer.buffer(element_index));
            element_index.pop_back();
          }
          return WriteSingleTupleIndexTable(stream, elements, device_subshape,
                                            &device_memory);
        }

        return Status::OK();
      });
}

Status TransferManager::TransferBufferFromDevice(
    se::Stream* stream, const se::DeviceMemoryBase& source, int64 size,
    void* destination) {
  if (source.size() < size) {
    return FailedPrecondition(
        "Source allocation on device not large enough for data tranfer: "
        "%lld < %lld",
        source.size(), size);
  }
  stream->ThenMemcpy(destination, source, size);
  return Status::OK();
}

Status TransferManager::TransferBufferToDevice(
    se::Stream* stream, int64 size, const void* source,
    se::DeviceMemoryBase* destination) {
  if (destination->size() < size) {
    return FailedPrecondition(
        "Destination allocation on device not large enough for data tranfer: "
        "%lld < %lld",
        destination->size(), size);
  }
  stream->ThenMemcpy(destination, source, size);
  return Status::OK();
}

StatusOr<ScopedShapedBuffer> TransferManager::AllocateScopedShapedBuffer(
    const Shape& on_host_shape, DeviceMemoryAllocator* allocator,
    int device_ordinal) {
  if (!LayoutUtil::HasLayout(on_host_shape)) {
    return InvalidArgument(
        "Shape must have a layout: %s",
        ShapeUtil::HumanStringWithLayout(on_host_shape).c_str());
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(on_host_shape));
  const Shape on_device_shape = HostShapeToDeviceShape(on_host_shape);
  TF_RET_CHECK(LayoutUtil::HasLayout(on_device_shape));

  ScopedShapedBuffer shaped_buffer(on_host_shape, on_device_shape, allocator,
                                   device_ordinal);

  // Allocate an appropriate sized buffer for each element in the shape
  // including the tuple pointer arrays.
  for (auto& pair : shaped_buffer.buffers()) {
    const ShapeIndex& index = pair.first;
    se::DeviceMemoryBase& memory_base = pair.second;
    const Shape& subshape = ShapeUtil::GetSubshape(on_device_shape, index);
    TF_ASSIGN_OR_RETURN(auto memory,
                        allocator->Allocate(shaped_buffer.device_ordinal(),
                                            GetByteSizeRequirement(subshape)));
    // Move the allocated buffer into the ScopedShapedBuffer, which owns it.
    memory_base = memory.Forget();
  }

  return std::move(shaped_buffer);
}

}  // namespace xla
