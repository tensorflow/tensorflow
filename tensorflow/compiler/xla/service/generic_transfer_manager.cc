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

#include "tensorflow/compiler/xla/service/generic_transfer_manager.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/service/interpreter/platform_id.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

GenericTransferManager::GenericTransferManager(se::Platform::Id platform_id,
                                               size_t pointer_size)
    : platform_id_(platform_id), pointer_size_(pointer_size) {}

se::Platform::Id GenericTransferManager::PlatformId() const {
  return platform_id_;
}

Status GenericTransferManager::WriteSingleTupleIndexTable(
    se::Stream* stream,
    tensorflow::gtl::ArraySlice<se::DeviceMemoryBase> elements,
    const Shape& shape, se::DeviceMemoryBase* region) {
  TF_RET_CHECK(elements.size() == ShapeUtil::TupleElementCount(shape));

  std::vector<const void*> element_pointers;
  for (const se::DeviceMemoryBase& element : elements) {
    element_pointers.push_back(element.opaque());
  }
  TF_RETURN_IF_ERROR(TransferBufferToDevice(
      stream, GetByteSizeRequirement(shape), element_pointers.data(), region));
  // Ensure the buffer is transferred before we destroy element_pointers.
  return stream->BlockHostUntilDone();
}

void GenericTransferManager::TransferLiteralFromDevice(
    se::Stream* stream, const ShapedBuffer& device_buffer,
    MutableBorrowingLiteral literal, std::function<void(Status)> done) {
  Status status = stream->BlockHostUntilDone();
  if (!status.ok()) {
    return done(status);
  }

  done(TransferLiteralFromDeviceInternal(stream->parent(), device_buffer,
                                         literal));
}

Status GenericTransferManager::TransferLiteralFromDeviceInternal(
    se::StreamExecutor* executor, const ShapedBuffer& device_buffer,
    MutableBorrowingLiteral literal) {
  VLOG(2) << "transferring literal from device ordinal "
          << executor->device_ordinal() << "; device buffer: " << device_buffer;
  TF_RET_CHECK(executor->device_ordinal() == device_buffer.device_ordinal());

  // The on-host and on-device shape should always be the same for the generic
  // transfer manager.
  TF_RET_CHECK(ShapeUtil::Equal(device_buffer.on_device_shape(),
                                device_buffer.on_host_shape()));

  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_host_shape(),
      [&](const Shape& subshape, const ShapeIndex& index) -> Status {
        if (ShapeUtil::IsArray(subshape)) {
          TF_RETURN_IF_ERROR(executor->SynchronousMemcpyD2H(
              /*source=*/device_buffer.buffer(index),
              /*size=*/GetByteSizeRequirement(subshape),
              /*destination=*/
              literal.untyped_data(index)));
        }

        return Status::OK();
      }));
  return Status::OK();
}

Status GenericTransferManager::TransferLiteralToDeviceAsync(
    se::Stream* stream, const LiteralSlice& literal,
    const ShapedBuffer& device_buffer) {
  const Shape& shape = literal.shape();
  VLOG(2) << "transferring literal shape to device: "
          << ShapeUtil::HumanString(shape)
          << "; device buffer: " << device_buffer;

  // The on-host and on-device shape should always be the same for the generic
  // transfer manager.
  TF_RET_CHECK(ShapeUtil::Equal(device_buffer.on_device_shape(),
                                device_buffer.on_host_shape()));

  TF_RET_CHECK(
      ShapeUtil::Compatible(literal.shape(), device_buffer.on_host_shape()));
  TF_RET_CHECK(stream->parent()->device_ordinal() ==
               device_buffer.device_ordinal());

  TF_RETURN_IF_ERROR(WriteTupleIndexTables(stream, device_buffer));

  return ShapeUtil::ForEachSubshapeWithStatus(
      device_buffer.on_host_shape(),
      [&](const Shape& device_subshape, const ShapeIndex& index) -> Status {
        se::DeviceMemoryBase device_memory = device_buffer.buffer(index);
        if (ShapeUtil::IsArray(device_subshape)) {
          TF_RET_CHECK(GetByteSizeRequirement(device_subshape) ==
                       device_memory.size());
          // Element is array-shaped: transfer array data to device buffer.
          const auto subliteral = LiteralSlice(literal, index);
          std::unique_ptr<Literal> relayed_out_literal;
          const void* source;
          if (LayoutUtil::Equal(device_subshape.layout(),
                                subliteral.shape().layout())) {
            source = subliteral.untyped_data();
            return TransferBufferToDevice(
                stream,
                /*size=*/GetByteSizeRequirement(device_subshape), source,
                &device_memory);
          } else {
            // Relayout data before transferring.
            relayed_out_literal = subliteral.Relayout(device_subshape.layout(),
                                                      /*shape_index=*/{});
            source = relayed_out_literal->untyped_data();
            TF_RETURN_IF_ERROR(TransferBufferToDevice(
                stream,
                /*size=*/GetByteSizeRequirement(device_subshape), source,
                &device_memory));
            return stream->BlockHostUntilDone();
          }
        }
        return Status::OK();
      });
}

Status GenericTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  return Unimplemented("Generic transfer to Infeed");
}

Status GenericTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, const Shape& literal_shape,
    MutableBorrowingLiteral literal) {
  return Unimplemented("Generic transfer from Outfeed");
}

Status GenericTransferManager::ResetDevices(
    tensorflow::gtl::ArraySlice<se::StreamExecutor*>
    /*executors*/) {
  return Unimplemented(
      "Device reset is not yet supported on this platform (b/30481585)");
}

int64 GenericTransferManager::GetByteSizeRequirement(const Shape& shape) const {
  return ShapeUtil::ByteSizeOf(shape, pointer_size_);
}

}  // namespace xla
