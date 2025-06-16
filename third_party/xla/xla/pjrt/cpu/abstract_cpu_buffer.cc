/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/pjrt/cpu/abstract_cpu_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "xla/cpu_function_runtime.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/async_work_runner.h"
#include "xla/pjrt/cpu/cpu_event.h"
#include "xla/pjrt/cpu/raw_buffer.h"
#include "xla/pjrt/cpu/tracked_cpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/cpu/cpu_executable.h"
#include "xla/service/cpu/cpu_xfeed.h"
#include "xla/service/shaped_buffer.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {
namespace {

constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB

// Unpacks and copies the packed data at `input` into the literal at the given
// ShapeIndex.
void UnpackIntNToLiteral(PrimitiveType input_element_type,
                         const CpuDeviceMemory& input,
                         MutableLiteralBase* literal,
                         const ShapeIndex& shape_index) {
  absl::Span<const char> input_span{
      static_cast<const char*>(input.untyped_data()), input.size_bytes()};
  size_t output_size = static_cast<size_t>(ShapeUtil::ByteSizeOf(
      ShapeUtil::GetSubshape(literal->shape(), shape_index)));
  absl::Span<char> output_span{
      static_cast<char*>(literal->untyped_data(shape_index)), output_size};
  primitive_util::UnpackIntN(input_element_type, input_span, output_span);
}

// `device_buffer`'s definition event must be ready before calling this
// function.
void CopyCpuBufferToLiteral(const Shape& device_shape,
                            TrackedCpuDeviceBuffer* device_buffer,
                            MutableLiteralBase* literal) {
  CHECK(!device_shape.IsTuple());
  const tsl::AsyncValueRef<CpuDeviceMemory>& b = device_buffer->buffer();
  CHECK(b.IsConcrete());
  if (primitive_util::IsSubByteNonPredType(device_shape.element_type())) {
    UnpackIntNToLiteral(device_shape.element_type(), *b, literal,
                        /*shape_index=*/{});
  } else {
    std::memcpy(literal->untyped_data(), b->untyped_data(),
                ShapeUtil::ByteSizeOf(device_shape));
  }
}

// `buffers` must be available.
ShapedBuffer AsShapedBuffer(int device_ordinal, const Shape& on_device_shape,
                            tsl::AsyncValueRef<CpuDeviceMemory> buf) {
  ShapedBuffer shaped_buffer(on_device_shape, device_ordinal);
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  CHECK(buf.IsConcrete());
  CHECK(iterator != shaped_buffer.buffers().end());
  iterator->second =
      se::DeviceMemoryBase(buf->untyped_data(), buf->size_bytes());
  ++iterator;
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

}  //  namespace

AbstractCpuBuffer::AbstractCpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedCpuDeviceBuffer> tracked_device_buffer)
    : CommonPjRtBuffer(std::move(tracked_device_buffer)),
      on_device_shape_(std::move(on_device_shape)) {}

AbstractCpuBuffer::~AbstractCpuBuffer() { AbstractCpuBuffer::Delete(); }

absl::StatusOr<Shape> AbstractCpuBuffer::logical_on_device_shape() {
  if (on_device_shape_.is_static()) {
    return on_device_shape_;
  }

  auto usage_event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return InvalidArgument(
        "logical_on_device_shape() called on deleted or donated buffer");
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  // Wait for the definition event.
  const auto& av = device_buffer->definition_event();
  BlockUntilReady(av.GetAsyncValue());
  if (auto* error = av.GetErrorIfPresent()) {
    return Internal("Error Execute: %s", error->message());
  }

  auto output_shape =
      tsl::MakeConstructedAsyncValueRef<Shape>(on_device_shape_);
  tsl::MakeRef<CpuRawBuffer>(memory_space(), device_buffer->buffer())
      ->ReadDynamicShape(output_shape, on_device_shape_);
  tsl::BlockUntilReady(output_shape);
  if (auto* error = output_shape.GetErrorIfPresent()) {
    return Internal("logical_on_device_shape failed: %s", error->message());
  }
  return output_shape.get();
}

absl::StatusOr<size_t> AbstractCpuBuffer::GetOnDeviceSizeInBytes() const {
  return ShapeUtil::ByteSizeOf(on_device_shape_);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
AbstractCpuBuffer::AcquireExternalReference() {
  class ScopedExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit ScopedExternalReference(AbstractCpuBuffer::ScopedHold hold)
        : external_reference_(std::move(hold)),
          data_(external_reference_->buffer()) {
      DCHECK(external_reference_.type() == ScopedHold::kExternalReference);
      DCHECK(data_);
      // We need to wait for the memory to be allocated before sharing it with
      // external frameworks like NumPy.
      tsl::BlockUntilReady(data_);
      CHECK(data_.IsConcrete());
      data_ptr_ = data_->untyped_data();
    }

    ~ScopedExternalReference() override = default;

   private:
    AbstractCpuBuffer::ScopedHold external_reference_;
    // Keep a reference to the underlying data used. Note that it is still
    // users' responsibility to synchronize reads and writes to the data.
    tsl::AsyncValueRef<CpuDeviceMemory> data_;
  };

  ScopedHold hold = GetBufferWithHold(ScopedHold::kExternalReference);
  TF_RETURN_IF_ERROR(hold.status());
  return {std::make_unique<ScopedExternalReference>(std::move(hold))};
}

class TrackedCpuDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedCpuDeviceBufferExternalReference(
      std::unique_ptr<TrackedCpuDeviceBuffer> tracked_device_buffer)
      : device_buffer_(std::move(tracked_device_buffer)) {
    // We need to wait for the memory to be allocated before sharing it with
    // external frameworks like NumPy.
    const auto& buffer = device_buffer_->buffer();
    tsl::BlockUntilReady(buffer);
    CHECK(buffer.IsConcrete());
    data_ptr_ = buffer->untyped_data();
  }

  ~TrackedCpuDeviceBufferExternalReference() override = default;

 private:
  std::unique_ptr<TrackedCpuDeviceBuffer> device_buffer_;
};

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
AbstractCpuBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedCpuDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedCpuDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

void AbstractCpuBuffer::Delete() {
  std::unique_ptr<TrackedCpuDeviceBuffer> device_buffer(
      static_cast<TrackedCpuDeviceBuffer*>(ReleaseBuffer().release()));
  if (device_buffer == nullptr) return;

  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> usage_events =
      device_buffer->LockUseAndTransferUsageEvents();

  std::vector<tsl::AsyncValue*> event_avs;
  event_avs.reserve(usage_events.size() + 1);
  for (auto& event : usage_events) {
    event_avs.push_back(event.GetAsyncValue());
  }

  // We should also wait for the definition event.
  event_avs.push_back(device_buffer->definition_event().GetAsyncValue());

  RunWhenReady(event_avs, [device_buffer = std::move(device_buffer)]() mutable {
    device_buffer.reset();
  });
}

absl::StatusOr<std::unique_ptr<TrackedCpuDeviceBuffer>>
AbstractCpuBuffer::Release(bool wait_for_operations_to_complete) {
  std::unique_ptr<TrackedCpuDeviceBuffer> device_buffer(
      static_cast<TrackedCpuDeviceBuffer*>(ReleaseBuffer().release()));
  if (device_buffer == nullptr) return {nullptr};

  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> events;
  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  events = device_buffer->LockUseAndTransferUsageEvents();

  if (wait_for_operations_to_complete) {
    // Block the host until all usage events have completed. Usage events
    // dominate definition events, so this also waits for the buffer to be
    // defined. Return the first error encountered.
    absl::Status first_error;
    for (const auto& av : events) {
      BlockUntilReady(av.GetAsyncValue());
      if (auto* error = av.GetErrorIfPresent()) {
        first_error.Update(Internal("Error Execute: %s", error->message()));
      }
    }
    if (!first_error.ok()) return std::move(first_error);
  }

  return device_buffer;
}

TrackedCpuDeviceBuffer* AbstractCpuBuffer::AcquireUsage(
    tsl::AsyncValueRef<CpuEvent> usage_event) {
  absl::MutexLock lock(&mu_);
  if (!device_buffer()) {
    return nullptr;
  }

  device_buffer()->AddUsageEvents(absl::MakeSpan(&usage_event, 1));
  return device_buffer();
}

AbstractCpuBuffer::ScopedHold AbstractCpuBuffer::GetBufferWithHold(
    ScopedHold::Type type) {
  absl::MutexLock lock(&mu_);
  // Ensure that at most one donation hold can be in progress at a time.
  WaitForOutstandingDonationHold();
  ScopedHold hold(this, type);
  AcquireHoldLocked(&hold);
  return hold;
}

AbstractCpuBuffer::ScopedHold AbstractCpuBuffer::AcquireDonation() {
  return GetBufferWithHold(ScopedHold::kDonation);
}

PjRtFuture<> AbstractCpuBuffer::DoAsyncWorkOnBuffer(
    absl::string_view method_name,
    absl::AnyInvocable<absl::Status(const Shape& device_shape,
                                    TrackedCpuDeviceBuffer* device_buffer) &&>
        work_on_buffer,
    bool should_do_work_sync, AsyncWorkRunner* async_work_runner) {
  auto name_generator = [buffer_name = buffer_name(), method_name]() {
    return absl::StrCat(buffer_name, "::", method_name);
  };
  tsl::profiler::TraceMe traceme(name_generator);
  if (IsEmptyTuple()) {
    return PjRtFuture<>(
        InvalidArgument("%s called on empty tuple", method_name));
  }
  auto usage_event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return PjRtFuture<>(InvalidArgument(
        "CopyToHostAsync() called on deleted or donated buffer"));
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  tsl::RCReference<tsl::AsyncValue> device_buffer_wait_av =
      device_buffer->definition_event().CopyRCRef();

  absl::StatusOr<Shape> device_shape = logical_on_device_shape();
  if (!device_shape.ok()) {
    return PjRtFuture<>(device_shape.status());
  }
  if (device_buffer_wait_av->IsConcrete() && should_do_work_sync) {
    // Unblock ToLiteral caller.
    return PjRtFuture<>(
        std::move(work_on_buffer)(*device_shape, device_buffer));
  } else {
    std::vector<tsl::RCReference<tsl::AsyncValue>> device_buffer_wait_avs{
        device_buffer_wait_av};
    PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
    // Wait for buffer definition events to finish before d2h dispatch. D2H
    // dispatch should be in parallel, e.g. one Execute event finish may trigger
    // multiple outputs' D2H, they should happen in different threads in
    // parallel.
    async_work_runner->ScheduleWhenReady(
        device_buffer_wait_avs,
        [device_buffer_wait_av = std::move(device_buffer_wait_av),
         work_on_buffer = std::move(work_on_buffer), promise, device_buffer,
         device_shape, ready_on_exit = std::move(ready_on_exit)]() mutable {
          tsl::profiler::TraceMe traceme("D2H Dispatch");
          // Errors in src buffer are surfaced to user.
          if (auto* error = device_buffer_wait_av->GetErrorIfPresent()) {
            promise.Set(*error);
            return;
          }
          auto status = std::move(work_on_buffer)(*device_shape, device_buffer);
          // Unblock ToLiteral event.
          if (!status.ok()) {
            promise.Set(status);
          } else {
            promise.Set();
          }
        });
    return PjRtFuture<>(
        std::move(promise),
        /*on_block_start=*/
        [name_generator]() {
          tsl::profiler::TraceMeProducer traceme(name_generator);
          return PjRtFutureHelpers::ProfilingKeys(
              {/*traceme_context_id =*/traceme.GetContextId()});
        },
        /*on_block_end=*/
        [name_generator](PjRtFutureHelpers::ProfilingKeys keys) {
          tsl::profiler::TraceMeConsumer traceme(name_generator,
                                                 keys.traceme_context_id);
        });
  }
}

PjRtFuture<> AbstractCpuBuffer::ToLiteralHelper(
    MutableLiteralBase* literal, AsyncWorkRunner* async_work_runner) {
  bool should_sync_copy = !literal->shape().IsTuple() &&
                          literal->size_bytes() < kSmallDataTransferByteSize;
  auto work_on_buffer =
      [literal](const Shape& device_shape,
                TrackedCpuDeviceBuffer* device_buffer) -> absl::Status {
    CopyCpuBufferToLiteral(device_shape, device_buffer, literal);
    return absl::OkStatus();
  };
  return DoAsyncWorkOnBuffer("ToLiteral", std::move(work_on_buffer),
                             should_sync_copy, async_work_runner);
}

PjRtFuture<> AbstractCpuBuffer::CopyRawToHostHelper(
    void* dst, int64_t offset, int64_t transfer_size,
    AsyncWorkRunner* async_work_runner) {
  bool should_sync_copy = transfer_size < kSmallDataTransferByteSize;
  auto work_on_buffer =
      [dst, offset, transfer_size](
          const Shape& device_shape,
          TrackedCpuDeviceBuffer* device_buffer) -> absl::Status {
    if (device_shape.IsTuple()) {
      return InvalidArgument("CopyRawToHost not implemented for tuples.");
    } else if (offset < 0 ||
               offset + transfer_size > ShapeUtil::ByteSizeOf(device_shape)) {
      return InvalidArgument("CopyRawToHost out of bounds.");
    }
    const tsl::AsyncValueRef<CpuDeviceMemory>& b = device_buffer->buffer();
    CHECK(b.IsConcrete());
    std::memcpy(dst, reinterpret_cast<char*>(b->untyped_data()) + offset,
                transfer_size);
    return absl::OkStatus();
  };
  return DoAsyncWorkOnBuffer("CopyRawToHost", std::move(work_on_buffer),
                             should_sync_copy, async_work_runner);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AbstractCpuBuffer::CopyToDeviceAcrossClients(PjRtDevice* dst_device) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
  // Avoid use-after-free on `literal` due to unsequenced move and use.
  Literal* literal_pointer = literal.get();
  absl::InlinedVector<int64_t, 4> byte_strides(
      literal->shape().dimensions().size());
  TF_RETURN_IF_ERROR(
      ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
  TF_ASSIGN_OR_RETURN(PjRtMemorySpace * dst_memory_space,
                      dst_device->default_memory_space());
  return dst_device->client()->BufferFromHostBuffer(
      literal_pointer->untyped_data(), literal_pointer->shape().element_type(),
      literal_pointer->shape().dimensions(), byte_strides,
      PjRtClient::HostBufferSemantics::kImmutableZeroCopy,
      [literal{std::move(literal)}]() { /* frees literal */ }, dst_memory_space,
      /*device_layout=*/nullptr);
}

absl::StatusOr<std::unique_ptr<TrackedCpuDeviceBuffer>>
AbstractCpuBuffer::CopyToDeviceHelper(AsyncWorkRunner* async_work_runner) {
  // Copy each leaf buffer to a destination buffer.
  auto usage_event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* src_device_buffer = AcquireUsage(usage_event);
  if (src_device_buffer == nullptr) {
    return InvalidArgument("CopyToDevice called on deleted or donated buffer");
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  auto dst_buffer = CpuDeviceMemory::CreateDelayedMemory();
  auto dst_definition_event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();

  // Wait for src buffer definition events to finish before d2d dispatch.
  // Errors are propagated asynchronously in dst buffer's definition events.
  const auto& src_definition_event = src_device_buffer->definition_event();

  auto copy_task = [src_buffer = src_device_buffer->buffer(), dst_buffer,
                    dst_definition_event, src_definition_event,
                    ready_on_exit = std::move(ready_on_exit)]() mutable {
    tsl::profiler::TraceMe traceme("D2D Dispatch");
    if (auto* error = src_definition_event.GetErrorIfPresent()) {
      // Any error discovered in src buffer are propagated to dst buffer
      // definition events, which will surface to users in
      // dst_buffer->ToLiteral().
      dst_definition_event.SetError(*error);
      return;
    }

    CHECK(src_buffer.IsConcrete());
    auto status = CpuDeviceMemory::AllocateInto(src_buffer->size_bytes(),
                                                dst_buffer.AsPtr());
    if (!status.ok()) {
      dst_definition_event.SetError(status);
      return;
    }
    std::memcpy(dst_buffer->untyped_data(), src_buffer->untyped_data(),
                src_buffer->size_bytes());
    dst_definition_event.SetStateConcrete();
  };

  src_definition_event.AndThen(
      [async_work_runner, copy_task = std::move(copy_task)]() mutable {
        async_work_runner->Schedule(std::move(copy_task));
      });

  return std::make_unique<TrackedCpuDeviceBuffer>(
      /*owns_buffers=*/true, dst_buffer, src_device_buffer->BufferSize(),
      absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4>{
          std::move(dst_definition_event)});
}

PjRtFuture<> AbstractCpuBuffer::GetReadyFuture() {
  tsl::AsyncValueRef<CpuEvent> definition_event;
  {
    absl::MutexLock lock(&mu_);
    if (!device_buffer()) {
      return PjRtFuture<>(InvalidArgument(
          "GetReadyFuture() called on deleted or donated buffer"));
    }
    definition_event = device_buffer()->definition_event();
  }
  DCHECK(definition_event);

  if (definition_event.IsAvailable()) {
    if (definition_event.IsError()) {
      const absl::Status& s = definition_event.GetError();
      return PjRtFuture<>(tsl::errors::CreateWithUpdatedMessage(
          s, absl::StrCat("Buffer Definition Event: ", s.message())));
    }
    return PjRtFuture<>(absl::OkStatus());
  } else {
    PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
    definition_event.AndThen(
        [definition_event = definition_event.AsPtr(), promise]() mutable {
          if (definition_event.IsError()) {
            const absl::Status& s = definition_event.GetError();
            promise.Set(tsl::errors::CreateWithUpdatedMessage(
                s, absl::StrCat("Buffer Definition Event: ", s.message())));
          } else {
            promise.Set();
          }
        });

    std::string message = absl::StrCat(buffer_name(), "::Await");
    return PjRtFuture<>(
        std::move(promise),
        /*on_block_start=*/
        [message]() {
          absl::string_view message_view(message);
          tsl::profiler::TraceMeProducer traceme(message_view);
          VLOG(1) << message_view;
          return PjRtFutureHelpers::ProfilingKeys(
              {/*traceme_context_id=*/traceme.GetContextId()});
        },
        /*on_block_end=*/
        [message](PjRtFutureHelpers::ProfilingKeys keys) {
          absl::string_view message_view(message);
          tsl::profiler::TraceMeConsumer traceme(message_view,
                                                 keys.traceme_context_id);
        });
  }
}

void PackOrCopy(PrimitiveType element_type, const LiteralSlice& literal,
                void* data, int64_t size) {
  if (primitive_util::IsSubByteNonPredType(element_type)) {
    const int bit_width = primitive_util::BitWidth(element_type);
    absl::Span<const char> src_data_span(
        static_cast<const char*>(literal.untyped_data()), literal.size_bytes());
    absl::Span<char> dst_data_span(static_cast<char*>(data), size);
    PackIntN(bit_width, src_data_span, dst_data_span);
  } else {
    CHECK_EQ(literal.size_bytes(), size);
    std::memcpy(data, literal.untyped_data(), size);
  }
}

/*static*/ absl::StatusOr<std::unique_ptr<TrackedCpuDeviceBuffer>>
AbstractCpuBuffer::AllocateTrackedDeviceBuffer(
    const Shape& on_device_shape,
    absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events) {
  if (on_device_shape.IsTuple()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Tuples are not supported for cpu-buffers: ",
                     on_device_shape.ToString()));
  }
  size_t byte_size = ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSIGN_OR_RETURN(tsl::AsyncValueRef<CpuDeviceMemory> device_buffer,
                      CpuDeviceMemory::Allocate(byte_size));
  return std::make_unique<TrackedCpuDeviceBuffer>(
      /*owns_buffers=*/true, std::move(device_buffer),
      std::move(definition_events));
}

/*static*/ void AbstractCpuBuffer::AllocateAvsAndEvents(
    const Shape& shape,
    absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4>* avs,
    absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4>* definition_events) {
  // Nested tuple shapes are not supported here.
  int num_leaf_buffers = shape.IsTuple() ? shape.tuple_shapes().size() : 1;
  for (int i = 0; i < num_leaf_buffers; ++i) {
    tsl::AsyncValueRef<CpuEvent> definition_event =
        tsl::MakeConstructedAsyncValueRef<CpuEvent>();
    definition_events->push_back(definition_event.CopyRef());
    avs->push_back(std::move(definition_event));
  }
}

/*static*/ bool AbstractCpuBuffer::BufferFromHostBufferSupportsZeroCopy(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides, const Shape& shape) {
  if (byte_strides && !HasMajorToMinorLayout(type, dims, *byte_strides)) {
    return false;
  }
  // Packed arrays are unpacked on host and packed on device.
  if (primitive_util::IsSubByteNonPredType(type)) {
    return false;
  }

  // If the input buffer has a default layout and is sufficiently aligned, we
  // can simply point to the input array's data without any further copies. At
  // the time of writing we require a 16-byte alignment because XLA may generate
  // code which requires it.
  if ((absl::bit_cast<std::uintptr_t>(data) &
       (cpu_function_runtime::MinAlign() - 1)) != 0) {
    return false;
  }
  return true;
}

}  // namespace xla
