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

void PoplarXfeedManager::Reset() {
  infeed()->Reset();
  outfeed()->Reset();
}

void PoplarXfeedQueueManager::Reset() {
  tensorflow::mutex_lock l(mu_);
  CHECK(current_buffer_ == nullptr);
  for (auto buffer : enqueued_buffers_) {
    buffer->Done(ShapeUtil::MakeNil());
  }
  enqueued_buffers_.clear();
}

Status PoplarXfeedQueueManager::EnqueueBufferAtomically(
    cpu::runtime::XfeedBuffer* const buffer, bool clear_if_full) {
  tensorflow::mutex_lock l(mu_);
  if (clear_if_full && enqueued_buffers_.size() == max_size_) {
    while (false == enqueued_buffers_.empty()) {
      auto front = enqueued_buffers_.front();
      front->Done(Shape{});
      enqueued_buffers_.pop_front();
    }
  } else {
    while (enqueued_buffers_.size() == max_size_) {
      VLOG(3) << queue_name_ << ", enqueued buffers full, waiting for dequeue";
      item_dequeued_cv_.wait(l);
    }
  }

  enqueued_buffers_.push_back(buffer);
  item_enqueued_cv_.notify_one();

  return Status::OK();
}

cpu::runtime::XfeedBuffer* PoplarXfeedQueueManager::BlockingDequeueBuffer() {
  tensorflow::mutex_lock l(mu_);
  VLOG(3) << "Waiting for an available buffer.";
  while (enqueued_buffers_.empty()) {
    item_enqueued_cv_.wait(l);
  }
  VLOG(3) << "A buffer is available!";
  CHECK(current_buffer_ == nullptr);
  current_buffer_ = enqueued_buffers_.front();
  enqueued_buffers_.pop_front();

  item_dequeued_cv_.notify_one();

  return current_buffer_;
}

void PoplarXfeedQueueManager::ReleaseCurrentBuffer(int32 length, void* data,
                                                   StatusOr<Shape> shape) {
  VLOG(3) << "Releasing buffer with shape: "
          << (shape.ok() ? ShapeUtil::HumanString(shape.ValueOrDie())
                         : "<error status>");
  tensorflow::mutex_lock l(mu_);
  CHECK(current_buffer_ != nullptr);
  CHECK_EQ(length, current_buffer_->length());
  CHECK_EQ(data, current_buffer_->data());
  current_buffer_->Done(std::move(shape));
  current_buffer_ = nullptr;
}

bool PoplarXfeedQueueManager::full() const {
  tensorflow::mutex_lock l(mu_);
  return enqueued_buffers_.size() == max_size_;
}

size_t PoplarXfeedQueueManager::size() const {
  tensorflow::mutex_lock l(mu_);
  return enqueued_buffers_.size();
}

void PoplarXfeedQueueManager::set_size(size_t size) {
  tensorflow::mutex_lock l(mu_);
  max_size_ = size;
}

size_t PoplarXfeedQueueManager::WaitForBuffers(size_t num_expected) {
  tensorflow::mutex_lock l(mu_);
  while (enqueued_buffers_.size() < num_expected) {
    item_enqueued_cv_.wait(l);
  }
  return enqueued_buffers_.size();
}

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

  auto* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());
  for (int64 i = 0; i < ShapeUtil::TupleElementCount(shape); ++i) {
    const Shape& tuple_element_shape = ShapeUtil::GetSubshape(shape, {i});
    int64 tuple_element_size = GetByteSizeRequirement(tuple_element_shape);
    TF_ASSIGN_OR_RETURN(
        cpu::runtime::XfeedBuffer * buffer,
        TransferBufferToInfeedInternal(executor, tuple_element_size,
                                       literal.untyped_data({i})));

    xfeed_manager->infeed()->EnqueueBufferAtomically(buffer);
  }

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

  auto* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());
  xfeed_manager->infeed()->EnqueueBufferAtomically(buffer);

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
  auto* xfeed_manager =
      xla::poplarplugin::GetXfeedManager(executor->device_ordinal());

  std::vector<Shape> outfed_shapes;
  for (auto b : buffer_data) {
    auto* user_buffer = b.first;
    auto byte_size = b.second;
    auto outfed_buffer = reinterpret_cast<PoplarOutfeedBuffer*>(
        xfeed_manager->outfeed()->BlockingDequeueBuffer());
    TF_RET_CHECK(outfed_buffer->length() == byte_size);

    TF_ASSIGN_OR_RETURN(Shape outfed_shape, outfed_buffer->shape());
    std::memcpy(user_buffer, outfed_buffer->data(), byte_size);
    xfeed_manager->outfeed()->ReleaseCurrentBuffer(
        byte_size, outfed_buffer->data(), outfed_shape);

    outfed_shapes.emplace_back(outfed_shape);
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
