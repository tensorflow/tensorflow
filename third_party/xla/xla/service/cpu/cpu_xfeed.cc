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

#include "xla/service/cpu/cpu_xfeed.h"

#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/cleanup/cleanup.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/service/cpu/cpu_runtime.h"
#include "xla/service/cpu/xfeed_manager.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/types.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/notification.h"

namespace xla {
namespace {

class CpuInfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  explicit CpuInfeedBuffer(int32_t length)
      : length_(length), buffer_(new char[length]) {}
  ~CpuInfeedBuffer() override { delete[] buffer_; }

  int32_t length() override { return length_; }
  void* data() override { return buffer_; }
  void Done(absl::StatusOr<Shape> /*shape*/) override { delete this; }

 private:
  int32_t length_;
  char* buffer_;
};

class CpuOutfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  CpuOutfeedBuffer(void* destination, int32_t length)
      : destination_(destination), length_(length) {}

  absl::StatusOr<Shape> WaitForNotification() {
    done_.WaitForNotification();
    return status_;
  }

  int32_t length() override { return length_; }
  void* data() override { return destination_; }
  void Done(absl::StatusOr<Shape> shape) override {
    status_ = std::move(shape);
    done_.Notify();
  }

 private:
  void* destination_;
  int32_t length_;
  absl::StatusOr<Shape> status_;
  tsl::Notification done_;
};

// Transfers infeed data to device. InfeedBuffer->Done() must be called to
// clean up the memory allocated for InfeedBuffer.
absl::StatusOr<cpu::runtime::XfeedBuffer*> TransferBufferToInfeedInternal(
    int64_t size, const void* source) {
  if (size > std::numeric_limits<int32_t>::max()) {
    return InvalidArgument("CPU infeed of %d bytes exceeds maximum of %d bytes",
                           size, std::numeric_limits<int32_t>::max());
  }

  if (size <= 0) {
    return InvalidArgument("Infeed shape must have positive size; got %d",
                           size);
  }

  auto size_32 = static_cast<int32_t>(size);
  auto queued_buffer = new CpuInfeedBuffer(size_32);
  std::memcpy(queued_buffer->data(), source, size);

  return queued_buffer;
}

absl::Status TransferBufferToInfeed(int device_ordinal, int64_t size,
                                    const void* source) {
  TF_ASSIGN_OR_RETURN(cpu::runtime::XfeedBuffer * buffer,
                      TransferBufferToInfeedInternal(size, source));

  cpu::runtime::XfeedManager* xfeed_manager =
      cpu::runtime::GetXfeedManager(device_ordinal);
  xfeed_manager->infeed()->EnqueueBuffersAtomically({buffer});

  return absl::OkStatus();
}

absl::StatusOr<Shape> TransferBuffersFromOutfeedInternal(
    int device_ordinal, absl::Span<const std::pair<void*, int64_t>> buffer_data,
    bool is_tuple) {
  std::vector<std::unique_ptr<CpuOutfeedBuffer>> buffers;
  for (auto b : buffer_data) {
    int64_t size = b.second;
    if (size > std::numeric_limits<int32_t>::max()) {
      return InvalidArgument("Outfeed shape is too large: needs %d bytes",
                             size);
    }

    if (size < 0) {
      return InvalidArgument(
          "Outfeed shape must have non-negative size; got %d", size);
    }

    auto size_32 = static_cast<int32_t>(size);
    VLOG(2)
        << "Enqueueing outfeed buffer (for the device to populate) of length "
        << size_32 << "B";
    buffers.push_back(std::make_unique<CpuOutfeedBuffer>(b.first, size_32));
  }

  std::vector<cpu::runtime::XfeedBuffer*> buffer_pointers;
  buffer_pointers.reserve(buffers.size());
  for (auto& b : buffers) {
    buffer_pointers.push_back(b.get());
  }

  cpu::runtime::XfeedManager* xfeed_manager =
      cpu::runtime::GetXfeedManager(device_ordinal);
  xfeed_manager->outfeed()->EnqueueBuffersAtomically(buffer_pointers);
  VLOG(2) << "Waiting for buffer to be notified as populated.";
  std::vector<Shape> outfed_shapes;
  outfed_shapes.reserve(buffers.size());
  for (auto& buffer : buffers) {
    TF_ASSIGN_OR_RETURN(Shape outfed_shape, buffer->WaitForNotification());
    outfed_shapes.push_back(std::move(outfed_shape));
  }
  if (is_tuple) {
    return ShapeUtil::MakeTupleShape(outfed_shapes);
  }
  TF_RET_CHECK(outfed_shapes.size() == 1);
  return std::move(outfed_shapes[0]);
}

absl::StatusOr<Shape> TransferArrayBufferFromOutfeed(int device_ordinal,
                                                     void* destination,
                                                     int64_t size_bytes) {
  return TransferBuffersFromOutfeedInternal(
      device_ordinal, {{destination, size_bytes}}, /*is_tuple=*/false);
}

absl::StatusOr<Shape> TransferTupleBuffersFromOutfeed(
    int device_ordinal,
    absl::Span<const std::pair<void*, int64_t>> buffer_data) {
  return TransferBuffersFromOutfeedInternal(device_ordinal, buffer_data,
                                            /*is_tuple=*/true);
}
}  // namespace

absl::Status TransferLiteralToInfeedOnCpu(int device_ordinal,
                                          const LiteralSlice& literal) {
  const Shape& shape = literal.shape();
  VLOG(2) << "Transferring literal to infeed with shape: "
          << ShapeUtil::HumanString(shape);

  if (!shape.IsTuple()) {
    int64_t size = cpu::runtime::GetByteSizeRequirement(shape, sizeof(void*));
    return TransferBufferToInfeed(device_ordinal, size, literal.untyped_data());
  }

  if (ShapeUtil::IsNestedTuple(shape)) {
    return Unimplemented(
        "Infeed with a nested tuple shape is not supported: %s",
        ShapeUtil::HumanString(literal.shape()));
  }

  // For a tuple, we transfer each of its elements to the device and
  // enqueue the resulting destination device addresses with the
  // infeed manager.
  std::vector<cpu::runtime::XfeedBuffer*> buffers;
  buffers.reserve(ShapeUtil::TupleElementCount(shape));
  absl::Cleanup cleanup = [&buffers]() {
    for (cpu::runtime::XfeedBuffer* b : buffers) {
      b->Done(Cancelled("Failed to infeed buffer to device."));
    }
  };

  for (int64_t i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& tuple_element_shape = ShapeUtil::GetSubshape(shape, {i});
    int64_t tuple_element_size = cpu::runtime::GetByteSizeRequirement(
        tuple_element_shape, sizeof(void*));
    TF_ASSIGN_OR_RETURN(cpu::runtime::XfeedBuffer * buffer,
                        TransferBufferToInfeedInternal(
                            tuple_element_size, literal.untyped_data({i})));
    buffers.push_back(buffer);
  }

  cpu::runtime::XfeedManager* xfeed_manager =
      cpu::runtime::GetXfeedManager(device_ordinal);
  xfeed_manager->infeed()->EnqueueBuffersAtomically(buffers);

  std::move(cleanup).Cancel();
  return absl::OkStatus();
}

absl::Status TransferLiteralFromOutfeedOnCpu(int device_ordinal,
                                             MutableBorrowingLiteral literal) {
  if (!literal.shape().IsTuple()) {
    int64_t size =
        cpu::runtime::GetByteSizeRequirement(literal.shape(), sizeof(void*));
    // Note: OSS build didn't like implicit conversion from
    // literal.shape().dimensions() to the array slice on 2017-07-10.
    absl::Span<const int64_t> dimensions(
        absl::bit_cast<const int64_t*>(literal.shape().dimensions().data()),
        literal.shape().dimensions().size());
    TF_ASSIGN_OR_RETURN(Shape received_shape,
                        TransferArrayBufferFromOutfeed(
                            device_ordinal, literal.untyped_data(), size));
    TF_RET_CHECK(ShapeUtil::Compatible(received_shape, literal.shape()))
        << "Shape received from outfeed "
        << ShapeUtil::HumanString(received_shape)
        << " did not match the shape that was requested for outfeed: "
        << ShapeUtil::HumanString(literal.shape());
    TF_RET_CHECK(size == cpu::runtime::GetByteSizeRequirement(received_shape,
                                                              sizeof(void*)));
    *literal.mutable_shape_do_not_use() = received_shape;
    return absl::OkStatus();
  }

  if (ShapeUtil::IsNestedTuple(literal.shape())) {
    return Unimplemented(
        "Nested tuple outfeeds are not yet implemented on CPU.");
  }

  std::vector<std::pair<void*, int64_t>> buffer_data;
  for (int i = 0; i < literal.shape().tuple_shapes_size(); ++i) {
    const Shape& tuple_element_shape =
        ShapeUtil::GetTupleElementShape(literal.shape(), i);
    int64_t size = cpu::runtime::GetByteSizeRequirement(tuple_element_shape,
                                                        sizeof(void*));
    buffer_data.push_back({literal.untyped_data({i}), size});
  }

  TF_ASSIGN_OR_RETURN(Shape received_shape, TransferTupleBuffersFromOutfeed(
                                                device_ordinal, buffer_data));

  TF_RET_CHECK(ShapeUtil::Compatible(received_shape, literal.shape()))
      << "Shape received from outfeed "
      << ShapeUtil::HumanString(received_shape)
      << " did not match the shape that was requested for outfeed: "
      << ShapeUtil::HumanString(literal.shape());
  TF_RET_CHECK(
      cpu::runtime::GetByteSizeRequirement(literal.shape(), sizeof(void*)) ==
      cpu::runtime::GetByteSizeRequirement(received_shape, sizeof(void*)));

  TF_RET_CHECK(ShapeUtil::Equal(literal.shape(), literal.shape()));
  return absl::OkStatus();
}

absl::Status ReadDynamicShapesOnCpu(
    const ShapedBuffer* device_buffer, Shape* device_shape,
    HloCostAnalysis::ShapeSizeFunction shape_size_fn) {
  TF_RET_CHECK(device_shape->is_dynamic());
  Shape original_device_shape = *device_shape;
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
        const void* memory = buffer.opaque();

        // Read the dynamic shape metadata from the device stream.
        Shape buffer_shape_static = ShapeUtil::MakeStaticShape(buffer_shape);
        const int64_t offset = shape_size_fn(buffer_shape_static);
        int64_t metadata_size = shape_size_fn(buffer_shape) - offset;
        if (metadata_size == 0) {
          return InvalidArgument("Dynamic shape metadata size should not be 0");
        }
        auto buffer_8 = static_cast<const int8_t*>(memory);
        auto metadata_buffer =
            reinterpret_cast<const int32_t*>(buffer_8 + offset);

        // Update shape size from metadata.
        for (int64_t i = 0; i < device_sub_shape.dimensions_size(); ++i) {
          device_sub_shape.mutable_dimensions()[i] = metadata_buffer[i];
        }
        return absl::OkStatus();
      }));
  device_shape->clear_dynamic_dimensions();

  TF_RET_CHECK(ShapeUtil::DynamicShapeIsCompatible(*device_shape,
                                                   original_device_shape));
  return absl::OkStatus();
}
}  // namespace xla
