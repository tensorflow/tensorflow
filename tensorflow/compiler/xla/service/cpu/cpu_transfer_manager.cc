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

#include "tensorflow/compiler/xla/service/cpu/cpu_transfer_manager.h"

#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_runtime.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/casts.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/notification.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {

namespace {

class CpuInfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  explicit CpuInfeedBuffer(int32 length)
      : length_(length),
        buffer_(new char[length]),
        device_memory_(buffer_, length_) {}
  ~CpuInfeedBuffer() override { delete[] buffer_; }

  int32 length() override { return length_; }
  void* data() override { return buffer_; }
  void Done(StatusOr<Shape> /*shape*/) override { delete this; }

  se::DeviceMemoryBase* device_memory() { return &device_memory_; }

 private:
  int32 length_;
  char* buffer_;
  se::DeviceMemoryBase device_memory_;
};

class CpuOutfeedBuffer : public cpu::runtime::XfeedBuffer {
 public:
  CpuOutfeedBuffer(void* destination, int32 length)
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

}  // namespace

CpuTransferManager::CpuTransferManager()
    : GenericTransferManager(se::host::kHostPlatformId,
                             /*pointer_size=*/sizeof(void*)) {}

Status CpuTransferManager::TransferLiteralToInfeed(
    se::StreamExecutor* executor, const LiteralSlice& literal) {
  const Shape& shape = literal.shape();
  VLOG(2) << "Transferring literal to infeed with shape: "
          << ShapeUtil::HumanString(shape);

  if (!ShapeUtil::IsTuple(shape)) {
    int64 size = GetByteSizeRequirement(shape);
    return TransferBufferToInfeed(executor, size, literal.untyped_data());
  }

  if (ShapeUtil::IsNestedTuple(shape)) {
    return Unimplemented(
        "Infeed with a nested tuple shape is not supported: %s",
        ShapeUtil::HumanString(literal.shape()).c_str());
  }

  // For a tuple, we transfer each of its elements to the device and
  // enqueue the resulting destination device addresses with the
  // infeed manager.
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

  cpu::runtime::XfeedManager* xfeed_manager = cpu::runtime::GetXfeedManager();
  xfeed_manager->infeed()->EnqueueBuffersAtomically(buffers);

  cleanup.release();
  return Status::OK();
}

Status CpuTransferManager::TransferBufferToInfeed(se::StreamExecutor* executor,
                                                  int64 size,
                                                  const void* source) {
  TF_ASSIGN_OR_RETURN(cpu::runtime::XfeedBuffer * buffer,
                      TransferBufferToInfeedInternal(executor, size, source));

  cpu::runtime::XfeedManager* xfeed_manager = cpu::runtime::GetXfeedManager();
  xfeed_manager->infeed()->EnqueueBuffersAtomically({buffer});

  return Status::OK();
}

StatusOr<cpu::runtime::XfeedBuffer*>
CpuTransferManager::TransferBufferToInfeedInternal(se::StreamExecutor* executor,
                                                   int64 size,
                                                   const void* source) {
  if (size > std::numeric_limits<int32>::max()) {
    return InvalidArgument("Infeed shape is too large: needs %lld bytes", size);
  }

  if (size <= 0) {
    return InvalidArgument("Infeed shape must have positive size; got %lld",
                           size);
  }

  int32 size_32 = static_cast<int32>(size);
  CpuInfeedBuffer* queued_buffer = new CpuInfeedBuffer(size_32);
  Status s = executor->SynchronousMemcpyH2D(
      /*host_src=*/source, /*size=*/size, queued_buffer->device_memory());

  if (!s.ok()) {
    queued_buffer->Done(s);
    return s;
  }
  return queued_buffer;
}

Status CpuTransferManager::TransferLiteralFromOutfeed(
    se::StreamExecutor* executor, const Shape& literal_shape,
    Literal* literal) {
  if (!ShapeUtil::IsTuple(literal_shape)) {
    int64 size = GetByteSizeRequirement(literal_shape);
    // Note: OSS build didn't like implicit conversion from
    // literal_shape.dimensions() to the array slice on 2017-07-10.
    tensorflow::gtl::ArraySlice<int64> dimensions(
        tensorflow::bit_cast<const int64*>(literal_shape.dimensions().data()),
        literal_shape.dimensions().size());
    *literal = std::move(*Literal::CreateFromDimensions(
        literal_shape.element_type(), dimensions));
    TF_ASSIGN_OR_RETURN(Shape received_shape,
                        TransferArrayBufferFromOutfeed(
                            executor, literal->untyped_data(), size));
    TF_RET_CHECK(ShapeUtil::Compatible(received_shape, literal->shape()))
        << "Shape received from outfeed "
        << ShapeUtil::HumanString(received_shape)
        << " did not match the shape that was requested for outfeed: "
        << ShapeUtil::HumanString(literal_shape);
    TF_RET_CHECK(size == GetByteSizeRequirement(received_shape));
    *literal->mutable_shape_do_not_use() = received_shape;
    return Status::OK();
  }

  if (ShapeUtil::IsNestedTuple(literal_shape)) {
    return Unimplemented(
        "Nested tuple outfeeds are not yet implemented on CPU.");
  }

  std::vector<std::unique_ptr<Literal>> elements;
  std::vector<std::pair<void*, int64>> buffer_data;
  for (int64 i = 0; i < literal_shape.tuple_shapes_size(); ++i) {
    const Shape& tuple_element_shape =
        ShapeUtil::GetTupleElementShape(literal_shape, i);
    // Note: OSS build didn't like implicit conversion from
    // literal_shape.dimensions() to the array slice on 2017-07-10.
    tensorflow::gtl::ArraySlice<int64> dimensions(
        tensorflow::bit_cast<const int64*>(
            tuple_element_shape.dimensions().data()),
        tuple_element_shape.dimensions().size());
    auto empty = Literal::CreateFromDimensions(
        tuple_element_shape.element_type(), dimensions);
    int64 size = GetByteSizeRequirement(tuple_element_shape);
    buffer_data.push_back({empty->untyped_data(), size});
    elements.push_back(std::move(empty));
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

  for (int64 i = 0; i < literal_shape.tuple_shapes_size(); ++i) {
    *elements[i]->mutable_shape_do_not_use() = received_shape.tuple_shapes(i);
  }
  *literal = std::move(*Literal::MakeTupleOwned(std::move(elements)));
  TF_RET_CHECK(ShapeUtil::Equal(literal->shape(), literal_shape));
  return Status::OK();
}

StatusOr<Shape> CpuTransferManager::TransferTupleBuffersFromOutfeed(
    se::StreamExecutor* executor,
    tensorflow::gtl::ArraySlice<std::pair<void*, int64>> buffer_data) {
  return TransferBuffersFromOutfeedInternal(executor, buffer_data,
                                            /*is_tuple=*/true);
}

StatusOr<Shape> CpuTransferManager::TransferArrayBufferFromOutfeed(
    se::StreamExecutor* executor, void* destination, int64 size_bytes) {
  return TransferBuffersFromOutfeedInternal(
      executor, {{destination, size_bytes}}, /*is_tuple=*/false);
}

StatusOr<Shape> CpuTransferManager::TransferBuffersFromOutfeedInternal(
    se::StreamExecutor* executor,
    tensorflow::gtl::ArraySlice<std::pair<void*, int64>> buffer_data,
    bool is_tuple) {
  std::vector<std::unique_ptr<CpuOutfeedBuffer>> buffers;
  for (auto b : buffer_data) {
    int64 size = b.second;
    if (size > std::numeric_limits<int32>::max()) {
      return InvalidArgument("Outfeed shape is too large: needs %lld bytes",
                             size);
    }

    if (size <= 0) {
      return InvalidArgument("Outfeed shape must have positive size; got %lld",
                             size);
    }

    int32 size_32 = static_cast<int32>(size);
    VLOG(2)
        << "Enqueueing outfeed buffer (for the device to populate) of length "
        << size_32 << "B";
    buffers.emplace_back(MakeUnique<CpuOutfeedBuffer>(b.first, size_32));
  }

  std::vector<cpu::runtime::XfeedBuffer*> buffer_pointers;
  buffer_pointers.reserve(buffers.size());
  for (auto& b : buffers) {
    buffer_pointers.push_back(b.get());
  }

  cpu::runtime::XfeedManager* xfeed_manager = cpu::runtime::GetXfeedManager();
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

}  // namespace xla

static std::unique_ptr<xla::TransferManager> CreateCpuTransferManager() {
  return xla::MakeUnique<xla::CpuTransferManager>();
}

static bool InitModule() {
  xla::TransferManager::RegisterTransferManager(
      stream_executor::host::kHostPlatformId, &CreateCpuTransferManager);
  return true;
}
static bool module_initialized = InitModule();
