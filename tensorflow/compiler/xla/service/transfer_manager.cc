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

#include <functional>
#include <string>
#include <utility>

#include "absl/memory/memory.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/compiler/xla/service/compiler.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"

using absl::StrCat;

namespace xla {

/* static */ absl::Mutex TransferManager::platform_transfer_manager_mutex_(
    absl::kConstInit);

/* static */ absl::flat_hash_map<se::Platform::Id, TransferManager::State>*
TransferManager::GetPlatformTransferManagers() {
  static auto* r =
      new absl::flat_hash_map<se::Platform::Id, TransferManager::State>;
  return r;
}

TransferManager::TransferMetadata::~TransferMetadata() {}

StatusOr<Literal> TransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    const TransferMetadata* transfer_metadata) {
  StatusOr<Literal> ret;

  se::Stream* substream = stream->GetOrCreateSubStream();
  substream->ThenWaitFor(stream);
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });

  tensorflow::Notification n;
  Status s;
  Literal literal(device_buffer.on_host_shape());
  TransferLiteralFromDevice(
      substream, device_buffer, &literal,
      [&](Status status) {
        s = status;
        n.Notify();
      },
      transfer_metadata);
  n.WaitForNotification();
  if (!s.ok()) {
    return s;
  }
  return std::move(literal);
}

Status TransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    const MutableBorrowingLiteral& literal,
    const TransferMetadata* transfer_metadata) {
  se::Stream* substream = stream->GetOrCreateSubStream();
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });

  Status ret;
  tensorflow::Notification n;
  TransferLiteralFromDevice(
      substream, device_buffer, literal,
      [&](Status status) {
        ret = status;
        n.Notify();
      },
      transfer_metadata);
  n.WaitForNotification();
  return ret;
}

Status TransferManager::TransferLiteralToDevice(
    se::Stream* stream, const LiteralSlice& literal,
    const ShapedBuffer& device_buffer,
    const TransferMetadata* transfer_metadata) {
  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  se::Stream* substream = stream->GetOrCreateSubStream();
  substream->ThenWaitFor(stream);
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });
  TF_RETURN_IF_ERROR(TransferLiteralToDeviceAsync(
      substream, literal, device_buffer, transfer_metadata));
  return substream->BlockHostUntilDone();
}

StatusOr<Literal> TransferManager::TransferArrayFromDevice(
    se::Stream* stream, const Shape& shape, const se::DeviceMemoryBase& source,
    const TransferMetadata* transfer_metadata) {
  StatusOr<Literal> ret;
  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  se::Stream* substream = stream->GetOrCreateSubStream();
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });

  tensorflow::Notification n;
  Literal literal(shape);
  Status s;
  TransferArrayFromDevice(
      substream, shape, source, &literal,
      [&](Status status) {
        s = status;
        n.Notify();
      },
      transfer_metadata);
  n.WaitForNotification();
  if (!s.ok()) {
    return s;
  }
  return std::move(literal);
}

Status TransferManager::TransferArrayToDevice(
    se::Stream* stream, const LiteralSlice& literal,
    const se::DeviceMemoryBase& dest,
    const TransferMetadata* transfer_metadata) {
  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  se::Stream* substream = stream->GetOrCreateSubStream();
  auto cleanup = tensorflow::gtl::MakeCleanup(
      [&]() { stream->ReturnSubStream(substream); });
  TF_RETURN_IF_ERROR(
      TransferArrayToDeviceAsync(substream, literal, dest, transfer_metadata));
  return substream->BlockHostUntilDone();
}

Status TransferManager::TransferArrayToDeviceAsync(
    se::Stream* stream, const LiteralSlice& literal,
    const se::DeviceMemoryBase& dest,
    const TransferMetadata* transfer_metadata) {
  const Shape on_device_shape = HostShapeToDeviceShape(literal.shape());
  TF_RET_CHECK(on_device_shape.IsArray())
      << "On-device representation of "
      << ShapeUtil::HumanString(literal.shape())
      << " is not an array: " << ShapeUtil::HumanString(on_device_shape);
  if (dest.size() < GetByteSizeRequirement(on_device_shape)) {
    return FailedPrecondition(
        "Allocation on device not large enough for array: "
        "%d < %d",
        dest.size(), GetByteSizeRequirement(on_device_shape));
  }
  ShapedBuffer shaped_buffer(on_device_shape,
                             stream->parent()->device_ordinal());
  shaped_buffer.set_buffer(dest, /*index=*/{});
  return TransferLiteralToDevice(stream, literal, shaped_buffer,
                                 transfer_metadata);
}

void TransferManager::TransferArrayFromDevice(
    se::Stream* stream, const Shape& shape, const se::DeviceMemoryBase& source,
    const MutableBorrowingLiteral& literal, std::function<void(Status)> done,
    const TransferMetadata* transfer_metadata) {
  if (!Shape::Equal().MinorToMajorOnlyInLayout()(HostShapeToDeviceShape(shape),
                                                 shape)) {
    auto error = StrCat("Shape ", ShapeUtil::HumanString(shape),
                        " has a differently shaped representation on-device: ",
                        ShapeUtil::HumanString(HostShapeToDeviceShape(shape)));
    return done(FailedPrecondition("%s", error));
  }
  if (source.size() < GetByteSizeRequirement(shape)) {
    return done(
        FailedPrecondition("Allocation on device not large enough for array: "
                           "%d < %d",
                           source.size(), GetByteSizeRequirement(shape)));
  }
  ShapedBuffer shaped_buffer(shape, stream->parent()->device_ordinal());
  shaped_buffer.set_buffer(source, /*index=*/{});
  return TransferLiteralFromDevice(stream, shaped_buffer, literal,
                                   std::move(done), transfer_metadata);
}

Status TransferManager::ReadDynamicShapes(se::Stream* stream,
                                          ShapedBuffer* device_buffer,
                                          Shape* device_shape) {
  DCHECK(device_shape->is_dynamic());
  Shape original_device_shape = *device_shape;
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  TF_ASSIGN_OR_RETURN(auto compiler,
                      Compiler::GetForPlatform(stream->parent()->platform()));
  TF_RETURN_IF_ERROR(device_buffer->buffers().ForEachMutableElementWithStatus(
      [&](const ShapeIndex& index, se::DeviceMemoryBase* buffer) {
        const Shape& buffer_shape =
            ShapeUtil::GetSubshape(*device_shape, index);
        if (buffer_shape.IsTuple()) {
          return OkStatus();
        }
        Shape& device_sub_shape =
            *ShapeUtil::GetMutableSubshape(device_shape, index);
        if (device_sub_shape.is_static()) {
          return OkStatus();
        }

        // Read the dynamic shape metadata from the device stream.  The dynamic
        // shape itself is stored at the end of the buffer.
        auto shape_size_fn = compiler->ShapeSizeBytesFunction();
        Shape buffer_shape_static = ShapeUtil::MakeStaticShape(buffer_shape);
        const int64_t offset = shape_size_fn(buffer_shape_static);
        int64_t metadata_size = shape_size_fn(buffer_shape) - offset;
        if (metadata_size == 0) {
          return InvalidArgument("Dynamic shape metadata size should not be 0");
        }
        auto buffer_8 = se::DeviceMemory<uint8_t>(*buffer);
        auto metadata_buffer =
            stream->parent()->GetSubBuffer(&buffer_8, offset, metadata_size);
        TF_ASSIGN_OR_RETURN(
            auto metadata,
            TransferArrayFromDevice(
                stream,
                ShapeUtil::MakeShape(S32, {buffer_shape.dimensions_size()}),
                metadata_buffer));

        // Update shape size from metadata.
        for (int64_t i = 0; i < metadata.element_count(); ++i) {
          device_sub_shape.mutable_dimensions()[i] = metadata.Get<int32_t>({i});
        }
        return OkStatus();
      }));
  device_shape->clear_dynamic_dimensions();

  TF_RET_CHECK(ShapeUtil::DynamicShapeIsCompatible(*device_shape,
                                                   original_device_shape));
  return OkStatus();
}

/* static */ void TransferManager::RegisterTransferManager(
    se::Platform::Id platform_id,
    TransferManagerCreationFunction creation_function) {
  absl::MutexLock lock(&TransferManager::platform_transfer_manager_mutex_);
  auto* managers = GetPlatformTransferManagers();
  CHECK(managers->find(platform_id) == managers->end());
  (*managers)[platform_id].creation_function = creation_function;
}

/* static */ StatusOr<TransferManager*> TransferManager::GetForPlatform(
    const se::Platform* platform) {
  absl::MutexLock lock(&TransferManager::platform_transfer_manager_mutex_);
  auto* managers = GetPlatformTransferManagers();

  auto it = managers->find(platform->id());
  if (it == managers->end()) {
    return NotFound(
        "could not find registered transfer manager for platform %s -- check "
        "target linkage",
        platform->Name());
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
        if (device_subshape.IsTuple() &&
            ShapeUtil::TupleElementCount(device_subshape) > 0) {
          se::DeviceMemoryBase device_memory = device_buffer.buffer(index);
          TF_RET_CHECK(GetByteSizeRequirement(device_subshape) ==
                       device_memory.size());

          std::vector<se::DeviceMemoryBase> elements;
          ShapeIndex element_index = index;
          for (int64_t i = 0; i < ShapeUtil::TupleElementCount(device_subshape);
               ++i) {
            element_index.push_back(i);
            elements.push_back(device_buffer.buffer(element_index));
            element_index.pop_back();
          }
          return WriteSingleTupleIndexTable(stream, elements, device_subshape,
                                            &device_memory);
        }

        return OkStatus();
      });
}

Status TransferManager::WriteRootTupleIndexTable(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
  TF_RET_CHECK(device_buffer.on_device_shape().IsTuple());
  if (ShapeUtil::TupleElementCount(device_buffer.on_device_shape()) == 0) {
    return OkStatus();
  }
  se::DeviceMemoryBase device_memory = device_buffer.buffer({});
  TF_RET_CHECK(GetByteSizeRequirement(device_buffer.on_device_shape()) ==
               device_memory.size());

  std::vector<se::DeviceMemoryBase> elements;
  for (int64_t i = 0;
       i < ShapeUtil::TupleElementCount(device_buffer.on_device_shape()); ++i) {
    elements.push_back(device_buffer.buffer({i}));
  }
  return WriteSingleTupleIndexTable(
      stream, elements, device_buffer.on_device_shape(), &device_memory);
}

Status TransferManager::WriteRootTupleIndexTable(
    se::Stream* stream, const ShapeTree<MaybeOwningDeviceMemory>& buffer_tree) {
  TF_RET_CHECK(buffer_tree.shape().IsTuple());
  if (ShapeUtil::TupleElementCount(buffer_tree.shape()) == 0) {
    return OkStatus();
  }
  se::DeviceMemoryBase device_memory =
      buffer_tree.element({}).AsDeviceMemoryBase();
  TF_RET_CHECK(GetByteSizeRequirement(buffer_tree.shape()) ==
               device_memory.size());

  std::vector<se::DeviceMemoryBase> elements;
  for (int64_t i = 0; i < ShapeUtil::TupleElementCount(buffer_tree.shape());
       ++i) {
    elements.push_back(buffer_tree.element({i}).AsDeviceMemoryBase());
  }
  return WriteSingleTupleIndexTable(stream, elements, buffer_tree.shape(),
                                    &device_memory);
}

Status TransferManager::TransferBufferFromDevice(
    se::Stream* stream, const se::DeviceMemoryBase& source, int64_t size,
    void* destination) {
  if (source.size() < size) {
    return FailedPrecondition(
        "Source allocation on device not large enough for data transfer: "
        "%d < %d",
        source.size(), size);
  }
  stream->ThenMemcpy(destination, source, size);
  return OkStatus();
}

Status TransferManager::TransferBufferToDevice(
    se::Stream* stream, int64_t size, const void* source,
    se::DeviceMemoryBase* destination) {
  if (destination->size() < size) {
    return FailedPrecondition(
        "Destination allocation on device not large enough for data transfer: "
        "%d < %d",
        destination->size(), size);
  }
  stream->ThenMemcpy(destination, source, size);
  return OkStatus();
}

StatusOr<ScopedShapedBuffer> TransferManager::AllocateScopedShapedBuffer(
    const Shape& on_host_shape, se::DeviceMemoryAllocator* allocator,
    int device_ordinal, DeviceShapeRepresentationFn shape_representation_fn) {
  if (!LayoutUtil::HasLayout(on_host_shape)) {
    return InvalidArgument("Shape must have a layout: %s",
                           ShapeUtil::HumanStringWithLayout(on_host_shape));
  }
  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(on_host_shape));
  Shape on_device_shape = (shape_representation_fn == nullptr)
                              ? HostShapeToDeviceShape(on_host_shape)
                              : shape_representation_fn(on_host_shape);
  TF_RET_CHECK(LayoutUtil::HasLayout(on_device_shape));

  ScopedShapedBuffer shaped_buffer(std::move(on_device_shape), allocator,
                                   device_ordinal);

  // Allocate an appropriate sized buffer for each element in the shape
  // including the tuple pointer arrays.
  for (auto& pair : shaped_buffer.buffers()) {
    const ShapeIndex& index = pair.first;
    se::DeviceMemoryBase& memory_base = pair.second;
    const Shape& subshape =
        ShapeUtil::GetSubshape(shaped_buffer.on_device_shape(), index);
    TF_ASSIGN_OR_RETURN(auto memory,
                        allocator->Allocate(shaped_buffer.device_ordinal(),
                                            GetByteSizeRequirement(subshape),
                                            /*retry_on_failure=*/true,
                                            subshape.layout().memory_space()));
    // Move the allocated buffer into the ScopedShapedBuffer, which owns it.
    memory_base = memory.Release();
  }

  return std::move(shaped_buffer);
}

StatusOr<Shape> TransferManager::ChooseCompactLayoutForShape(
    const Shape& host_shape) const {
  return LayoutUtil::GetWithDefaultLayout(host_shape);
}

xla::Shape TransferManager::ChooseGoodInfeedLayout(const Shape& shape) const {
  return LayoutUtil::GetWithDefaultLayout(shape);
}

}  // namespace xla
