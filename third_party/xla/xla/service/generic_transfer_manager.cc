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

#include "xla/service/generic_transfer_manager.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/types/span.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/primitive_util.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"

namespace xla {

GenericTransferManager::GenericTransferManager(se::Platform::Id platform_id,
                                               size_t pointer_size)
    : platform_id_(platform_id), pointer_size_(pointer_size) {}

se::Platform::Id GenericTransferManager::PlatformId() const {
  return platform_id_;
}

absl::Status GenericTransferManager::WriteSingleTupleIndexTable(
    se::Stream* stream, absl::Span<const se::DeviceMemoryBase> elements,
    const Shape& shape, se::DeviceMemoryBase* region) {
  TF_RET_CHECK(elements.size() == ShapeUtil::TupleElementCount(shape));

  auto element_pointers = std::make_shared<std::vector<const void*>>();
  element_pointers->reserve(elements.size());
  for (const se::DeviceMemoryBase& element : elements) {
    element_pointers->push_back(element.opaque());
  }
  TF_RETURN_IF_ERROR(TransferBufferToDevice(
      stream, GetByteSizeRequirement(shape), element_pointers->data(), region));
  // Ensure the buffer is transferred before we destroy element_pointers.
  TF_RETURN_IF_ERROR(
      stream->DoHostCallback([element_pointers{std::move(element_pointers)}]() {
        /* holds reference to element_pointers in closure */
      }));
  return absl::OkStatus();
}

void GenericTransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    MutableBorrowingLiteral literal, std::function<void(absl::Status)> done,
    const TransferMetadata* transfer_metadata) {
  VLOG(2) << "transferring literal from device ordinal "
          << stream->parent()->device_ordinal()
          << "; device buffer: " << device_buffer;

  absl::Status status = [&]() -> absl::Status {
    TF_RET_CHECK(stream->parent()->device_ordinal() ==
                 device_buffer.physical_device_ordinal());

    TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
        device_buffer.on_device_shape(),
        [&](const Shape& subshape, const ShapeIndex& index) -> absl::Status {
          if (subshape.IsArray()) {
            if (PackSubbyteTypes() &&
                primitive_util::IsSubByteNonPredType(subshape.element_type())) {
              if (!subshape.is_static()) {
                return absl::UnimplementedError(
                    "Int4 outputs with dynamic shapes are unsupported");
              }
              return TransferIntNArrayFromDevice(
                  stream,
                  /*source=*/device_buffer.buffer(index),
                  subshape.element_type(),
                  /*num_elements=*/ShapeUtil::ElementsIn(subshape),
                  /*destination=*/literal.untyped_data(index));
            } else {
              TF_RETURN_IF_ERROR(TransferBufferFromDevice(
                  stream,
                  /*source=*/device_buffer.buffer(index),
                  // With bounded dynamic shapes, the shape of the device buffer
                  // (bounded allocation) can be bigger than the literal.
                  /*size=*/
                  GetByteSizeRequirement(
                      ShapeUtil::GetSubshape(literal.shape(), index)),
                  /*destination=*/literal.untyped_data(index)));
            }
          }
          return absl::OkStatus();
        }));
    return absl::OkStatus();
  }();

  if (!status.ok()) {
    done(status);
    return;
  }

  // CUDA callbacks are tricky as we cannot call any CUDA driver functions from
  // within a host callback. As a result, `TransferLiteralFromDevice` must be
  // very conservative, and is synchronous by default. However, if the user
  // declares, via the metadata, that their callback is safe to call from a host
  // callback, we enqueue it and return immediately.
  if ((transfer_metadata != nullptr) &&
      tensorflow::down_cast<const LiteralFromDeviceMetadata*>(transfer_metadata)
          ->callback_is_host_callback_safe) {
    auto status = stream->DoHostCallback([done = std::move(done), stream] {
      done(stream->ok() ? absl::OkStatus()
                        : Internal("`TransferLiteralFromDevice` failed"));
    });
    if (!status.ok()) {
      LOG(ERROR) << "`DoHostCallback` failed: " << status;
    }
  } else {
    done(stream->BlockHostUntilDone());
  }
}

absl::Status GenericTransferManager::TransferLiteralToDeviceAsync(
    se::Stream* stream, const LiteralSlice& literal,
    const ShapedBuffer& device_buffer,
    const TransferMetadata* /*transfer_metadata*/) {
  const Shape& shape = literal.shape();
  VLOG(2) << "transferring literal shape to device: "
          << ShapeUtil::HumanString(shape)
          << "; device buffer: " << device_buffer;

  TF_RET_CHECK(
      ShapeUtil::Compatible(literal.shape(), device_buffer.on_device_shape()));
  TF_RET_CHECK(stream->parent()->device_ordinal() ==
               device_buffer.physical_device_ordinal());

  TF_RETURN_IF_ERROR(WriteTupleIndexTablesAsync(stream, device_buffer));

  return ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_device_shape(),
      [&](const Shape& device_subshape,
          const ShapeIndex& index) -> absl::Status {
        if (device_subshape.IsArray()) {
          int64_t size = GetByteSizeRequirement(device_subshape);
          se::DeviceMemoryBase device_memory = device_buffer.buffer(index);
          TF_RET_CHECK(size == device_memory.size());

          auto TransferBuffer = [&](const void* source) {
            if (PackSubbyteTypes() && primitive_util::IsSubByteNonPredType(
                                          device_subshape.element_type())) {
              if (!device_subshape.is_static()) {
                return absl::UnimplementedError(absl::StrCat(
                    primitive_util::LowercasePrimitiveTypeName(
                        device_subshape.element_type()),
                    " inputs with dynamic shapes are unsupported"));
              }
              return TransferIntNArrayToDevice(
                  stream, device_subshape.element_type(),
                  /*num_elements=*/ShapeUtil::ElementsIn(device_subshape),
                  /*source=*/source,
                  /*destination=*/&device_memory);
            } else {
              return TransferBufferToDevice(stream, /*size=*/size,
                                            /*source=*/source,
                                            /*destination=*/&device_memory);
            }
          };

          LiteralSlice subliteral(literal, index);
          if (device_subshape.layout() == subliteral.shape().layout()) {
            return TransferBuffer(subliteral.untyped_data());
          } else {
            // Relayout data before transferring.
            auto relaid_out = std::make_shared<Literal>(
                subliteral.Relayout(device_subshape.layout()));
            TF_RETURN_IF_ERROR(TransferBuffer(relaid_out->untyped_data()));
            // Ensure the buffer is transferred before we destroy it.
            TF_RETURN_IF_ERROR(stream->DoHostCallback(
                [keep_alive = std::move(relaid_out)] {}));
          }
        }
        return absl::OkStatus();
      });
}

absl::Status GenericTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  return Unimplemented("Generic transfer to Infeed");
}

absl::Status GenericTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, MutableBorrowingLiteral literal) {
  return Unimplemented("Generic transfer from Outfeed");
}

absl::Status GenericTransferManager::ResetDevices(
    absl::Span<se::StreamExecutor* const>
    /*executors*/) {
  return Unimplemented(
      "Device reset is not yet supported on this platform (b/30481585)");
}

absl::Status GenericTransferManager::TransferBufferFromDevice(
    se::Stream* stream, const se::DeviceMemoryBase& source, int64_t size,
    void* destination) {
  if (source.size() < size) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Source allocation on device not large enough for data transfer: "
        "%d < %d",
        source.size(), size));
  }
  return stream->Memcpy(destination, source, size);
}

absl::Status GenericTransferManager::TransferBufferToDevice(
    se::Stream* stream, int64_t size, const void* source,
    se::DeviceMemoryBase* destination) {
  if (destination->size() < size) {
    return absl::FailedPreconditionError(absl::StrFormat(
        "Destination allocation on device not large enough for data transfer: "
        "%d < %d",
        destination->size(), size));
  }
  return stream->Memcpy(destination, source, size);
}

absl::Status GenericTransferManager::TransferIntNArrayFromDevice(
    se::Stream* stream, const se::DeviceMemoryBase& source,
    PrimitiveType element_type, int64_t num_elements, void* destination) {
  int bit_width = primitive_util::BitWidth(element_type);
  int64_t elements_per_byte = 8 / bit_width;
  int64_t packed_size = CeilOfRatio(num_elements, elements_per_byte);
  auto packed_dst_data = std::make_unique<std::vector<char>>(packed_size);
  TF_RETURN_IF_ERROR(TransferBufferFromDevice(stream, source, packed_size,
                                              packed_dst_data->data()));
  TF_RETURN_IF_ERROR(
      stream->DoHostCallback([destination, bit_width, num_elements,
                              packed_dst_data = std::move(packed_dst_data)]() {
        UnpackIntN(
            bit_width, *packed_dst_data,
            absl::MakeSpan(static_cast<char*>(destination), num_elements));
      }));
  return absl::OkStatus();
}

absl::Status GenericTransferManager::TransferIntNArrayToDevice(
    se::Stream* stream, PrimitiveType element_type, int64_t num_elements,
    const void* source, se::DeviceMemoryBase* destination) {
  int bit_width = primitive_util::BitWidth(element_type);
  int64_t elements_per_byte = 8 / bit_width;
  auto packed_src_data = std::make_unique<std::vector<char>>(
      CeilOfRatio(num_elements, elements_per_byte));
  PackIntN(bit_width,
           absl::MakeSpan(static_cast<const char*>(source), num_elements),
           absl::MakeSpan(*packed_src_data));
  TF_RETURN_IF_ERROR(TransferBufferToDevice(
      stream, packed_src_data->size(), packed_src_data->data(), destination));
  return stream->DoHostCallback([keep_alive = std::move(packed_src_data)] {});
}

int64_t GenericTransferManager::GetByteSizeRequirement(
    const Shape& shape) const {
  if (shape.IsTuple() || shape.is_static()) {
    return ShapeUtil::ByteSizeOf(shape, pointer_size_);
  }
  int64_t metadata_size = sizeof(int32_t) * shape.dimensions().size();
  return ShapeUtil::ByteSizeOf(shape, pointer_size_) + metadata_size;
}

Shape GenericTransferManager::HostShapeToDeviceShape(
    const Shape& host_shape) const {
  Shape device_shape = TransferManager::HostShapeToDeviceShape(host_shape);
  if (PackSubbyteTypes() &&
      primitive_util::IsSubByteNonPredType(device_shape.element_type())) {
    device_shape.mutable_layout()->set_element_size_in_bits(
        primitive_util::BitWidth(device_shape.element_type()));
  }
  return device_shape;
}

absl::StatusOr<Shape> GenericTransferManager::ChooseCompactLayoutForShape(
    const Shape& host_shape) const {
  Shape compact_shape = LayoutUtil::GetWithDefaultLayout(host_shape);
  if (PackSubbyteTypes() &&
      primitive_util::IsSubByteNonPredType(compact_shape.element_type())) {
    compact_shape.mutable_layout()->set_element_size_in_bits(
        primitive_util::BitWidth(compact_shape.element_type()));
  }
  return compact_shape;
}

}  // namespace xla
