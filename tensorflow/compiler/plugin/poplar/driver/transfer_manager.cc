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

#include "tensorflow/compiler/plugin/poplar/driver/transfer_manager.h"
#include "tensorflow/compiler/plugin/poplar/driver/executor.h"
#include "tensorflow/compiler/plugin/poplar/driver/platform_id.h"

#include "tensorflow/core/lib/gtl/cleanup.h"

#include <memory>

#include "tensorflow/compiler/xla/service/transfer_manager.h"

namespace xla {
namespace poplarplugin {

class PoplarInfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  explicit PoplarInfeedBuffer(int32 length)
      : length_(length),
        buffer_(new char[length]),
        device_memory_(buffer_, length_) {}
  ~PoplarInfeedBuffer() override { delete[] buffer_; }

  int32 length() override { return length_; }
  void* data() override { return buffer_; }
  void Done(StatusOr<Shape> /*shape*/) override { delete this; }

  se::DeviceMemoryBase* device_memory() { return &device_memory_; }

 private:
  int32 length_;
  char* buffer_;
  se::DeviceMemoryBase device_memory_;
};

class PoplarOutfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  PoplarOutfeedBuffer(void* destination, int32 length)
      : destination_(destination), length_(length) {}

  StatusOr<Shape> WaitForNotification() {
    done_.WaitForNotification();
    return status_;
  }

  int32 length() override { return length_; }
  void* data() override { return destination_; }
  void Done(StatusOr<Shape> shape) override {
    status_ = std::move(shape);
    done_.Notify();
  }

 private:
  void* destination_;
  int32 length_;
  StatusOr<Shape> status_;
  tensorflow::Notification done_;
};

PoplarTransferManager::PoplarTransferManager()
    : GenericTransferManager(kPoplarPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

Status PoplarTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  const Shape& shape = literal.shape();

  if (!shape.IsTuple()) {
    int64 size = GetByteSizeRequirement(shape);
    return TransferBufferToInfeed(executor, size, literal.untyped_data());
  }

  if (ShapeUtil::IsNestedTuple(shape)) {
    return Unimplemented(
        "Infeed with a nested tuple shape is not supported: %s",
        ShapeUtil::HumanString(literal.shape()));
  }

  std::vector<cpu::runtime::XfeedBuffer*> buffers;
  buffers.reserve(ShapeUtil::TupleElementCount(shape));
  auto cleanup = tensorflow::gtl::MakeCleanup([&buffers]() {
    for (cpu::runtime::XfeedBuffer* b : buffers) {
      b->Done(Cancelled("Failed to infeed buffer to device."));
    }
  });

  for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& tuple_element_shape = ShapeUtil::GetSubshape(shape, {i});
    int64 tuple_element_size = GetByteSizeRequirement(tuple_element_shape);
    TF_ASSIGN_OR_RETURN(
        cpu::runtime::XfeedBuffer * buffer,
        TransferBufferToInfeedInternal(executor, tuple_element_size,
                                       literal.untyped_data({i})));
    buffers.push_back(buffer);
  }

  cpu::runtime::XfeedManager* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());
  xfeed_manager->infeed()->EnqueueBuffersAtomically(buffers);

  cleanup.release();
  return Status::OK();
}

Status PoplarTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, const Shape& literal_shape,
    MutableBorrowingLiteral literal) {
  if (!literal_shape.IsTuple()) {
    int64 size = GetByteSizeRequirement(literal_shape);

    absl::Span<const int64> dimensions(literal_shape.dimensions().data(),
                                       literal_shape.dimensions().size());

    TF_ASSIGN_OR_RETURN(
        Shape received_shape,
        TransferArrayBufferFromOutfeed(executor, literal.untyped_data(), size));
    TF_RET_CHECK(ShapeUtil::Compatible(received_shape, literal.shape()))
        << "Shape received from outfeed "
        << ShapeUtil::HumanString(received_shape)
        << " did not match the shape that was requested for outfeed: "
        << ShapeUtil::HumanString(literal_shape);
    TF_RET_CHECK(size == GetByteSizeRequirement(received_shape));
    *literal.mutable_shape_do_not_use() = received_shape;
    return Status::OK();
  }

  if (ShapeUtil::IsNestedTuple(literal_shape)) {
    return Unimplemented(
        "Nested tuple outfeeds are not yet implemented on IPU.");
  }

  std::vector<std::pair<void*, int64>> buffer_data;
  for (int64 i = 0; i < literal_shape.tuple_shapes_size(); ++i) {
    const Shape& tuple_element_shape =
        ShapeUtil::GetTupleElementShape(literal_shape, i);
    int64 size = GetByteSizeRequirement(tuple_element_shape);
    buffer_data.push_back({literal.untyped_data({i}), size});
  }

  TF_ASSIGN_OR_RETURN(Shape received_shape,
                      TransferTupleBuffersFromOutfeed(executor, buffer_data));

  TF_RET_CHECK(ShapeUtil::Compatible(received_shape, literal_shape))
      << "Shape received from outfeed "
      << ShapeUtil::HumanString(received_shape)
      << " did not match the shape that was requested for outfeed: "
      << ShapeUtil::HumanString(literal_shape);
  TF_RET_CHECK(GetByteSizeRequirement(literal_shape) ==
               GetByteSizeRequirement(received_shape));

  TF_RET_CHECK(ShapeUtil::Equal(literal.shape(), literal_shape));
  return Status::OK();
}

Status PoplarTransferManager::TransferBufferToInfeed(
    se::StreamExecutor* executor, int64 size, const void* source) {
  TF_ASSIGN_OR_RETURN(cpu::runtime::XfeedBuffer * buffer,
                      TransferBufferToInfeedInternal(executor, size, source));

  cpu::runtime::XfeedManager* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());
  xfeed_manager->infeed()->EnqueueBuffersAtomically({buffer});

  return Status::OK();
}

StatusOr<cpu::runtime::XfeedBuffer*>
PoplarTransferManager::TransferBufferToInfeedInternal(
    se::StreamExecutor* executor, int64 size, const void* source) {
  if (size > std::numeric_limits<int32>::max()) {
    return InvalidArgument("Infeed shape is too large: needs %d bytes", size);
  }

  if (size <= 0) {
    return InvalidArgument("Infeed shape must have positive size; got %d",
                           size);
  }

  int32 size_32 = static_cast<int32>(size);

  PoplarInfeedBuffer* queued_buffer = new PoplarInfeedBuffer(size_32);
  std::memcpy(queued_buffer->data(), source, size);

  return queued_buffer;
}

StatusOr<Shape> PoplarTransferManager::TransferTupleBuffersFromOutfeed(
    se::StreamExecutor* executor,
    absl::Span<const std::pair<void*, int64>> buffer_data) {
  return TransferBuffersFromOutfeedInternal(executor, buffer_data,
                                            /*is_tuple=*/true);
}

StatusOr<Shape> PoplarTransferManager::TransferArrayBufferFromOutfeed(
    se::StreamExecutor* executor, void* destination, int64 size_bytes) {
  return TransferBuffersFromOutfeedInternal(
      executor, {{destination, size_bytes}}, /*is_tuple=*/false);
}

StatusOr<Shape> PoplarTransferManager::TransferBuffersFromOutfeedInternal(
    se::StreamExecutor* executor,
    absl::Span<const std::pair<void*, int64>> buffer_data, bool is_tuple) {
  std::vector<std::unique_ptr<PoplarOutfeedBuffer>> buffers;
  for (auto b : buffer_data) {
    int64 size = b.second;
    if (size > std::numeric_limits<int32>::max()) {
      return InvalidArgument("Outfeed shape is too large: needs %d bytes",
                             size);
    }

    if (size <= 0) {
      return InvalidArgument("Outfeed shape must have positive size; got %d",
                             size);
    }

    int32 size_32 = static_cast<int32>(size);
    VLOG(2)
        << "Enqueueing outfeed buffer (for the device to populate) of length "
        << size_32 << "B";
    buffers.emplace_back(
        absl::make_unique<PoplarOutfeedBuffer>(b.first, size_32));
  }

  std::vector<cpu::runtime::XfeedBuffer*> buffer_pointers;
  buffer_pointers.reserve(buffers.size());
  for (auto& b : buffers) {
    buffer_pointers.push_back(b.get());
  }

  cpu::runtime::XfeedManager* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());
  xfeed_manager->outfeed()->EnqueueBuffersAtomically(buffer_pointers);
  VLOG(2) << "Waiting for buffer to be notified as populated.";
  std::vector<Shape> outfed_shapes;
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

}  // namespace poplarplugin
}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreatePoplarTransferManager() {
  return absl::make_unique<xla::poplarplugin::PoplarTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      xla::poplarplugin::kPoplarPlatformId, &CreatePoplarTransferManager);
  return true;
}

static bool module_initialized = InitModule();
