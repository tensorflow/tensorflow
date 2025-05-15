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

#include "xla/service/transfer_manager.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/const_init.h"
#include "absl/cleanup/cleanup.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/literal.h"
#include "xla/service/compiler.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/notification.h"
#include "tsl/platform/statusor.h"

namespace xla {

/* static */ absl::Mutex TransferManager::platform_transfer_manager_mutex_(
    absl::kConstInit);

/* static */ absl::flat_hash_map<se::Platform::Id, TransferManager::State>*
TransferManager::GetPlatformTransferManagers() {
  static auto* r =
      new absl::flat_hash_map<se::Platform::Id, TransferManager::State>;
  return r;
}

absl::StatusOr<Literal> TransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    const TransferMetadata* transfer_metadata) {
  Literal literal(device_buffer.on_host_shape());
  TF_RETURN_IF_ERROR(TransferLiteralFromDevice(stream, device_buffer, &literal,
                                               transfer_metadata));
  return std::move(literal);
}

absl::Status TransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    const MutableBorrowingLiteral& literal,
    const TransferMetadata* transfer_metadata) {
  TF_ASSIGN_OR_RETURN(se::Stream * substream, stream->GetOrCreateSubStream());
  TF_RETURN_IF_ERROR(substream->WaitFor(stream));
  absl::Cleanup cleanup = [&]() { stream->ReturnSubStream(substream); };

  absl::Status ret;
  tsl::Notification n;
  TransferLiteralFromDevice(
      substream, device_buffer, literal,
      [&](absl::Status status) {
        ret = status;
        n.Notify();
      },
      transfer_metadata);
  n.WaitForNotification();
  return ret;
}

absl::Status TransferManager::TransferLiteralToDevice(
    se::Stream* stream, const LiteralSlice& literal,
    const ShapedBuffer& device_buffer,
    const TransferMetadata* transfer_metadata) {
  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  TF_ASSIGN_OR_RETURN(se::Stream * substream, stream->GetOrCreateSubStream());
  TF_RETURN_IF_ERROR(substream->WaitFor(stream));
  absl::Cleanup cleanup = [&]() { stream->ReturnSubStream(substream); };
  TF_RETURN_IF_ERROR(TransferLiteralToDeviceAsync(
      substream, literal, device_buffer, transfer_metadata));
  return substream->BlockHostUntilDone();
}

absl::StatusOr<Literal> TransferManager::TransferArrayFromDevice(
    se::Stream* stream, const Shape& shape, const se::DeviceMemoryBase& source,
    const TransferMetadata* transfer_metadata) {
  TF_RET_CHECK(shape.IsArray());
  TF_RET_CHECK(Shape::Equal().MinorToMajorOnlyInLayout()(
      HostShapeToDeviceShape(shape), shape));
  Literal literal(shape);
  ShapedBuffer shaped_buffer(shape, stream->parent()->device_ordinal());
  shaped_buffer.set_buffer(source, /*index=*/{});
  TF_RETURN_IF_ERROR(TransferLiteralFromDevice(stream, shaped_buffer, &literal,
                                               transfer_metadata));
  return std::move(literal);
}

absl::Status TransferManager::TransferArrayToDevice(
    se::Stream* stream, const LiteralSlice& literal,
    const se::DeviceMemoryBase& dest,
    const TransferMetadata* transfer_metadata) {
  // Implement the synchronous version by waiting on the asynchronous version.
  // Use a substream so that if we are called from a HostCallback we don't
  // deadlock.
  TF_ASSIGN_OR_RETURN(se::Stream * substream, stream->GetOrCreateSubStream());
  TF_RETURN_IF_ERROR(substream->WaitFor(stream));
  absl::Cleanup cleanup = [&]() { stream->ReturnSubStream(substream); };
  TF_RETURN_IF_ERROR(
      TransferArrayToDeviceAsync(substream, literal, dest, transfer_metadata));
  return substream->BlockHostUntilDone();
}

absl::Status TransferManager::TransferArrayToDeviceAsync(
    se::Stream* stream, const LiteralSlice& literal,
    const se::DeviceMemoryBase& dest,
    const TransferMetadata* transfer_metadata) {
  TF_RET_CHECK(literal.shape().IsArray());
  ShapedBuffer shaped_buffer(HostShapeToDeviceShape(literal.shape()),
                             stream->parent()->device_ordinal());
  shaped_buffer.set_buffer(dest, /*index=*/{});
  return TransferLiteralToDeviceAsync(stream, literal, shaped_buffer,
                                      transfer_metadata);
}

absl::Status TransferManager::ReadDynamicShapes(
    se::Stream* stream, const ShapedBuffer* device_buffer,
    Shape* device_shape) {
  DCHECK(device_shape->is_dynamic());
  Shape original_device_shape = *device_shape;
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());

  TF_ASSIGN_OR_RETURN(
      auto compiler, Compiler::GetForPlatform(stream->parent()->GetPlatform()));
  TF_RETURN_IF_ERROR(device_buffer->buffers().ForEachElementWithStatus(
      [&](const ShapeIndex& index,
          const se::DeviceMemoryBase& buffer) -> absl::Status {
        const Shape& buffer_shape =
            ShapeUtil::GetSubshape(*device_shape, index);
        if (buffer_shape.IsTuple()) {
          return absl::OkStatus();
        }
        Shape& device_sub_shape =
            *ShapeUtil::GetMutableSubshape(device_shape, index);
        if (device_sub_shape.is_static()) {
          return absl::OkStatus();
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
        auto buffer_8 = se::DeviceMemory<uint8_t>(buffer);
        auto metadata_buffer = buffer_8.GetSlice(offset, metadata_size);
        TF_ASSIGN_OR_RETURN(
            auto metadata,
            TransferArrayFromDevice(
                stream,
                ShapeUtil::MakeShape(S32, {buffer_shape.dimensions_size()}),
                metadata_buffer));

        // Update shape size from metadata.
        for (int64_t i = 0; i < metadata.element_count(); ++i) {
          device_sub_shape.set_dimensions(i, metadata.Get<int32_t>({i}));
        }
        return absl::OkStatus();
      }));
  device_shape->clear_dynamic_dimensions();

  TF_RET_CHECK(ShapeUtil::DynamicShapeIsCompatible(*device_shape,
                                                   original_device_shape));
  return absl::OkStatus();
}

/* static */ void TransferManager::RegisterTransferManager(
    se::Platform::Id platform_id,
    TransferManagerCreationFunction creation_function) {
  absl::MutexLock lock(&TransferManager::platform_transfer_manager_mutex_);
  auto* managers = GetPlatformTransferManagers();
  CHECK(managers->find(platform_id) == managers->end());
  (*managers)[platform_id].creation_function = creation_function;
}

/* static */ absl::StatusOr<TransferManager*> TransferManager::GetForPlatform(
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

absl::Status TransferManager::WriteTupleIndexTables(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
  TF_RETURN_IF_ERROR(WriteTupleIndexTablesAsync(stream, device_buffer));
  return stream->BlockHostUntilDone();
}

absl::Status TransferManager::WriteTupleIndexTablesAsync(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
  VLOG(2) << "Writing tuple index tables for " << device_buffer;

  return ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_device_shape(),
      [&](const Shape& device_subshape,
          const ShapeIndex& index) -> absl::Status {
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

        return absl::OkStatus();
      });
}

absl::Status TransferManager::WriteRootTupleIndexTable(
    se::Stream* stream, const ShapedBuffer& device_buffer) {
  TF_RET_CHECK(device_buffer.on_device_shape().IsTuple());
  if (ShapeUtil::TupleElementCount(device_buffer.on_device_shape()) == 0) {
    return absl::OkStatus();
  }
  se::DeviceMemoryBase device_memory = device_buffer.buffer({});
  TF_RET_CHECK(GetByteSizeRequirement(device_buffer.on_device_shape()) ==
               device_memory.size());

  std::vector<se::DeviceMemoryBase> elements;
  elements.reserve(
      ShapeUtil::TupleElementCount(device_buffer.on_device_shape()));
  for (int64_t i = 0;
       i < ShapeUtil::TupleElementCount(device_buffer.on_device_shape()); ++i) {
    elements.push_back(device_buffer.buffer({i}));
  }
  return WriteSingleTupleIndexTable(
      stream, elements, device_buffer.on_device_shape(), &device_memory);
}

absl::Status TransferManager::WriteRootTupleIndexTable(
    se::Stream* stream, const ShapeTree<MaybeOwningDeviceMemory>& buffer_tree) {
  TF_RET_CHECK(buffer_tree.shape().IsTuple());
  if (ShapeUtil::TupleElementCount(buffer_tree.shape()) == 0) {
    return absl::OkStatus();
  }
  se::DeviceMemoryBase device_memory =
      buffer_tree.element({}).AsDeviceMemoryBase();
  TF_RET_CHECK(GetByteSizeRequirement(buffer_tree.shape()) ==
               device_memory.size());

  std::vector<se::DeviceMemoryBase> elements;
  elements.reserve(ShapeUtil::TupleElementCount(buffer_tree.shape()));
  for (int64_t i = 0; i < ShapeUtil::TupleElementCount(buffer_tree.shape());
       ++i) {
    elements.push_back(buffer_tree.element({i}).AsDeviceMemoryBase());
  }
  return WriteSingleTupleIndexTable(stream, elements, buffer_tree.shape(),
                                    &device_memory);
}

absl::StatusOr<ScopedShapedBuffer> TransferManager::AllocateScopedShapedBuffer(
    const Shape& on_host_shape, se::DeviceMemoryAllocator* allocator,
    int device_ordinal, int physical_device_ordinal,
    DeviceShapeRepresentationFn shape_representation_fn) {
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
                                   device_ordinal, physical_device_ordinal);

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
                                            LayoutUtil::MemorySpace(subshape)));
    // Move the allocated buffer into the ScopedShapedBuffer, which owns it.
    memory_base = memory.Release();
  }

  return std::move(shaped_buffer);
}

absl::StatusOr<Shape> TransferManager::ChooseCompactLayoutForShape(
    const Shape& host_shape) const {
  return LayoutUtil::GetWithDefaultLayout(host_shape);
}

xla::Shape TransferManager::ChooseGoodInfeedLayout(const Shape& shape) const {
  return LayoutUtil::GetWithDefaultLayout(shape);
}

}  // namespace xla
