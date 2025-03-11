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

#include "xla/pjrt/cpu/abstract_tfrt_cpu_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
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
#include "xla/pjrt/cpu/tracked_tfrt_cpu_device_buffer.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/cpu/cpu_event.h"
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
                         const MaybeOwningCpuMemory& input,
                         MutableLiteralBase* literal,
                         const ShapeIndex& shape_index) {
  absl::Span<const char> input_span{static_cast<const char*>(input.data()),
                                    input.size()};
  size_t output_size = static_cast<size_t>(ShapeUtil::ByteSizeOf(
      ShapeUtil::GetSubshape(literal->shape(), shape_index)));
  absl::Span<char> output_span{
      static_cast<char*>(literal->untyped_data(shape_index)), output_size};
  primitive_util::UnpackIntN(input_element_type, input_span, output_span);
}

// `device_buffer`'s definition event must be ready before calling this
// function.
void CopyCpuBufferToLiteral(const Shape& device_shape,
                            TrackedTfrtCpuDeviceBuffer* device_buffer,
                            MutableLiteralBase* literal) {
  if (!device_shape.IsTuple()) {
    const tsl::AsyncValueRef<MaybeOwningCpuMemory>& b =
        device_buffer->Buffers()[0];
    CHECK(b.IsConcrete());
    if (primitive_util::IsSubByteNonPredType(device_shape.element_type())) {
      UnpackIntNToLiteral(device_shape.element_type(), *b, literal,
                          /*shape_index=*/{});
    } else {
      std::memcpy(literal->untyped_data(), b->data(),
                  ShapeUtil::ByteSizeOf(device_shape));
    }
  } else {
    // Tuple case.
    int num_leaves = literal->shape().tuple_shapes().size();
    for (int i = 0; i < num_leaves; ++i) {
      const tsl::AsyncValueRef<MaybeOwningCpuMemory>& b =
          device_buffer->Buffers()[i];
      CHECK(b.IsConcrete());
      if (primitive_util::IsSubByteNonPredType(device_shape.element_type())) {
        UnpackIntNToLiteral(device_shape.element_type(), *b, literal, {i});
      } else {
        std::memcpy(
            literal->untyped_data({i}), b->data(),
            ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(device_shape, {i})));
      }
    }
  }
}

// `buffers` must be available.
ShapedBuffer AsShapedBuffer(
    int device_ordinal, const Shape& on_device_shape,
    absl::Span<const tsl::AsyncValueRef<MaybeOwningCpuMemory>> buffers) {
  ShapedBuffer shaped_buffer(on_device_shape, device_ordinal);
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  for (const auto& buf : buffers) {
    CHECK(buf.IsConcrete());
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = se::DeviceMemoryBase(buf->data(), buf->size());
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

}  //  namespace

AbstractTfrtCpuBuffer::AbstractTfrtCpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer)
    : on_device_shape_(std::move(on_device_shape)),
      tracked_device_buffer_(std::move(tracked_device_buffer)) {}

AbstractTfrtCpuBuffer::~AbstractTfrtCpuBuffer() {
  AbstractTfrtCpuBuffer::Delete();
}

absl::StatusOr<Shape> AbstractTfrtCpuBuffer::logical_on_device_shape() {
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

  // Safe to call `AsShapedBuffer` because the definition event is ready.
  ShapedBuffer shaped_buffer =
      AsShapedBuffer(device()->local_hardware_id().value(), on_device_shape_,
                     device_buffer->Buffers());
  Shape ret_shape = on_device_shape_;
  TF_RETURN_IF_ERROR(ReadDynamicShapesOnCpu(
      &shaped_buffer, &ret_shape, cpu::CpuExecutable::ShapeSizeBytes));
  return ret_shape;
}

absl::StatusOr<size_t> AbstractTfrtCpuBuffer::GetOnDeviceSizeInBytes() const {
  return ShapeUtil::ByteSizeOf(on_device_shape_);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
AbstractTfrtCpuBuffer::AcquireExternalReference() {
  class ScopedExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit ScopedExternalReference(
        AbstractTfrtCpuBuffer* buffer,
        tsl::AsyncValueRef<MaybeOwningCpuMemory> data)
        : buffer_(buffer), data_(std::move(data)) {
      DCHECK(data_);
      // We need to wait for the memory to be allocated before sharing it with
      // external frameworks like NumPy.
      tsl::BlockUntilReady(data_);
      CHECK(data_.IsConcrete());
      data_ptr_ = data_->data();
    }

    ~ScopedExternalReference() override { buffer_->DropExternalReference(); }

   private:
    AbstractTfrtCpuBuffer* buffer_ = nullptr;
    // Keep a reference to the underlying data used. Note that it is still
    // users' responsibility to synchronize reads and writes to the data.
    tsl::AsyncValueRef<MaybeOwningCpuMemory> data_;
  };

  absl::MutexLock lock(&mu_);
  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Buffer has been deleted or donated.");
  }

  ++external_reference_counter_;

  return {std::make_unique<ScopedExternalReference>(
      this, tracked_device_buffer_->Buffers()[0])};
}

void AbstractTfrtCpuBuffer::DropExternalReference() {
  absl::MutexLock lock(&mu_);
  CHECK_GT(external_reference_counter_, 0);
  --external_reference_counter_;
  if (external_reference_counter_ == 0 && external_references_dropped_event_) {
    external_references_dropped_event_->SetStateConcrete();
  }
}

class TrackedCpuDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedCpuDeviceBufferExternalReference(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    // We need to wait for the memory to be allocated before sharing it with
    // external frameworks like NumPy.
    const auto& buffer = tracked_device_buffer_->Buffers()[0];
    tsl::BlockUntilReady(buffer);
    CHECK(buffer.IsConcrete());
    data_ptr_ = buffer->data();
  }

  ~TrackedCpuDeviceBufferExternalReference() override = default;

 private:
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer_;
};

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
AbstractTfrtCpuBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedCpuDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

void AbstractTfrtCpuBuffer::CommitDonation() {
  absl::MutexLock lock(&mu_);
  CHECK(pending_donation_);
  CHECK(!tracked_device_buffer_);
  pending_donation_ = false;
}

void AbstractTfrtCpuBuffer::AbortDonation(
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer) {
  absl::MutexLock lock(&mu_);
  CHECK(pending_donation_);
  CHECK(!tracked_device_buffer_);
  pending_donation_ = false;
  tracked_device_buffer_ = std::move(device_buffer);
}

void AbstractTfrtCpuBuffer::Delete() {
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer;
  std::optional<tsl::AsyncValueRef<CpuEvent>> external_references_dropped_event;
  {
    absl::MutexLock lock(&mu_);
    device_buffer = ReleaseBufferLocked();
    if (device_buffer == nullptr) return;

    if (external_reference_counter_ > 0) {
      external_references_dropped_event = external_references_dropped_event_ =
          tsl::MakeConstructedAsyncValueRef<CpuEvent>();
    }
  }

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
  if (external_references_dropped_event) {
    event_avs.push_back(external_references_dropped_event->GetAsyncValue());
  }

  RunWhenReady(event_avs, [device_buffer = std::move(device_buffer)]() mutable {
    device_buffer.reset();
  });
}

bool AbstractTfrtCpuBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_ == nullptr;
}

std::unique_ptr<TrackedTfrtCpuDeviceBuffer>
AbstractTfrtCpuBuffer::ReleaseBufferLocked() {
  auto condition = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return !pending_donation_;
  };
  mu_.Await(absl::Condition(&condition));
  return std::move(tracked_device_buffer_);
}

absl::StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
AbstractTfrtCpuBuffer::Release(bool wait_for_operations_to_complete) {
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer;
  {
    absl::MutexLock lock(&mu_);
    device_buffer = ReleaseBufferLocked();
  }
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

TrackedTfrtCpuDeviceBuffer* AbstractTfrtCpuBuffer::AcquireUsage(
    tsl::AsyncValueRef<CpuEvent> usage_event) {
  absl::MutexLock lock(&mu_);
  if (!tracked_device_buffer_) {
    return nullptr;
  }

  tracked_device_buffer_->AddUsageEvents(absl::MakeSpan(&usage_event, 1));
  return tracked_device_buffer_.get();
}

absl::StatusOr<AbstractTfrtCpuBuffer::DonationTransaction>
AbstractTfrtCpuBuffer::AcquireDonation() {
  absl::MutexLock lock(&mu_);

  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Donation requested for invalid buffer");
  }

  if (external_reference_counter_ > 0) {
    return InvalidArgument(
        "Donation requested for buffer with external reference");
  }

  CHECK(!pending_donation_);
  pending_donation_ = true;

  // Swap out `tracked_device_buffer_` so that no one can acquire a usage event
  // after this point.
  return DonationTransaction(this, std::move(tracked_device_buffer_));
}

PjRtFuture<> AbstractTfrtCpuBuffer::DoAsyncWorkOnBuffer(
    absl::string_view method_name,
    absl::AnyInvocable<
        absl::Status(const Shape& device_shape,
                     TrackedTfrtCpuDeviceBuffer* device_buffer) &&>
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

PjRtFuture<> AbstractTfrtCpuBuffer::ToLiteralHelper(
    MutableLiteralBase* literal, AsyncWorkRunner* async_work_runner) {
  bool should_sync_copy = !literal->shape().IsTuple() &&
                          literal->size_bytes() < kSmallDataTransferByteSize;
  auto work_on_buffer =
      [literal](const Shape& device_shape,
                TrackedTfrtCpuDeviceBuffer* device_buffer) -> absl::Status {
    CopyCpuBufferToLiteral(device_shape, device_buffer, literal);
    return absl::OkStatus();
  };
  return DoAsyncWorkOnBuffer("ToLiteral", std::move(work_on_buffer),
                             should_sync_copy, async_work_runner);
}

PjRtFuture<> AbstractTfrtCpuBuffer::CopyRawToHostHelper(
    void* dst, int64_t offset, int64_t transfer_size,
    AsyncWorkRunner* async_work_runner) {
  bool should_sync_copy = transfer_size < kSmallDataTransferByteSize;
  auto work_on_buffer =
      [dst, offset, transfer_size](
          const Shape& device_shape,
          TrackedTfrtCpuDeviceBuffer* device_buffer) -> absl::Status {
    if (device_shape.IsTuple()) {
      return InvalidArgument("CopyRawToHost not implemented for tuples.");
    } else if (offset < 0 ||
               offset + transfer_size > ShapeUtil::ByteSizeOf(device_shape)) {
      return InvalidArgument("CopyRawToHost out of bounds.");
    }
    const tsl::AsyncValueRef<MaybeOwningCpuMemory>& b =
        device_buffer->Buffers()[0];
    CHECK(b.IsConcrete());
    std::memcpy(dst, reinterpret_cast<char*>(b->data()) + offset,
                transfer_size);
    return absl::OkStatus();
  };
  return DoAsyncWorkOnBuffer("CopyRawToHost", std::move(work_on_buffer),
                             should_sync_copy, async_work_runner);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
AbstractTfrtCpuBuffer::CopyToDeviceAcrossClients(PjRtDevice* dst_device) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
  // Avoid use-after-free on `literal` due to unsequenced move and use.
  Literal* literal_pointer = literal.get();
  absl::InlinedVector<int64_t, 4> byte_strides(literal->shape().rank());
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

absl::StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
AbstractTfrtCpuBuffer::CopyToDeviceHelper(AsyncWorkRunner* async_work_runner) {
  // Copy each leaf buffer to a destination buffer.
  auto usage_event = tsl::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* src_device_buffer = AcquireUsage(usage_event);
  if (src_device_buffer == nullptr) {
    return InvalidArgument("CopyToDevice called on deleted or donated buffer");
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  int num_leaf_buffers = src_device_buffer->Buffers().size();
  absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> src_buffers;
  absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> dst_buffers;
  absl::InlinedVector<size_t, 4> dst_buffers_sizes;
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> dst_definition_events;
  src_buffers.reserve(num_leaf_buffers);
  dst_buffers.reserve(num_leaf_buffers);
  dst_buffers_sizes.reserve(num_leaf_buffers);
  dst_definition_events.reserve(num_leaf_buffers);

  for (int i = 0; i < num_leaf_buffers; ++i) {
    src_buffers.push_back(std::move(src_device_buffer->Buffers()[i]));
    dst_buffers.push_back(
        tsl::MakeUnconstructedAsyncValueRef<MaybeOwningCpuMemory>());
    dst_buffers_sizes.push_back(src_device_buffer->BufferSizes()[i]);
    dst_definition_events.push_back(
        tsl::MakeConstructedAsyncValueRef<CpuEvent>());
  }

  // Wait for src buffer definition events to finish before d2d dispatch.
  // Errors are propagated asynchronously in dst buffer's definition events.
  const auto& src_definition_event = src_device_buffer->definition_event();

  auto copy_task = [num_leaf_buffers, src_buffers = std::move(src_buffers),
                    dst_buffers_copies = dst_buffers, dst_definition_events,
                    src_definition_event,
                    ready_on_exit = std::move(ready_on_exit)]() mutable {
    tsl::profiler::TraceMe traceme("D2D Dispatch");
    if (auto* error = src_definition_event.GetErrorIfPresent()) {
      for (int i = 0; i < num_leaf_buffers; ++i) {
        // Any error discovered in src buffer are propagated to dst buffer
        // definition events, which will surface to users in
        // dst_buffer->ToLiteral().
        dst_definition_events[i].SetError(*error);
      }
      return;
    }

    for (int i = 0; i < num_leaf_buffers; ++i) {
      // `src_buffers` are available because `src_definition_event` should have
      // been ready.
      CHECK(src_buffers[i].IsConcrete());
      auto dst_memory = MaybeOwningCpuMemory::Allocate(src_buffers[i]->size());
      if (!dst_memory.ok()) {
        dst_definition_events[i].SetError(dst_memory.status());
        continue;
      }
      dst_buffers_copies[i].emplace(std::move(*dst_memory));
      std::memcpy(dst_buffers_copies[i]->data(), src_buffers[i]->data(),
                  src_buffers[i]->size());
      dst_definition_events[i].SetStateConcrete();
    }
  };

  src_definition_event.AndThen(
      [async_work_runner, copy_task = std::move(copy_task)]() mutable {
        async_work_runner->Schedule(std::move(copy_task));
      });

  return std::make_unique<TrackedTfrtCpuDeviceBuffer>(
      on_device_shape_.IsTuple(), /*owns_buffers=*/true, std::move(dst_buffers),
      std::move(dst_buffers_sizes), std::move(dst_definition_events));
}

PjRtFuture<> AbstractTfrtCpuBuffer::GetReadyFuture() {
  tsl::AsyncValueRef<CpuEvent> definition_event;
  {
    absl::MutexLock lock(&mu_);
    if (!tracked_device_buffer_) {
      return PjRtFuture<>(InvalidArgument(
          "GetReadyFuture() called on deleted or donated buffer"));
    }
    definition_event = tracked_device_buffer_->definition_event();
  }
  DCHECK(definition_event);

  if (definition_event.IsAvailable()) {
    if (definition_event.IsError()) {
      return PjRtFuture<>(
          FailedPrecondition("Buffer Definition Event: %s",
                             definition_event.GetError().message()));
    }
    return PjRtFuture<>(absl::OkStatus());
  } else {
    PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
    definition_event.AndThen([definition_event = definition_event.AsPtr(),
                              promise]() mutable {
      if (definition_event.IsError()) {
        promise.Set(FailedPrecondition("Buffer Definition Event: %s",
                                       definition_event.GetError().message()));
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

namespace {

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

}  // namespace

// The buffer's memory should have been allocated before calling this function.
void AbstractTfrtCpuBuffer::CopyFromLiteral(
    const LiteralSlice& literal, const Shape& shape,
    absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4>* avs,
    AsyncWorkRunner* async_work_runner) {
  auto usage_event = tsl::MakeAvailableAsyncValueRef<CpuEvent>();
  auto* device_buffer = AcquireUsage(std::move(usage_event));
  CHECK(device_buffer);
  if (!shape.IsTuple()) {
    // It is OK to capture `buffer` pointer because the `output_buffer` can't be
    // deleted until all the usage holds have gone away.
    async_work_runner->Schedule(
        [literal, av = (*avs)[0].CopyRef(), device_buffer, shape]() mutable {
          tsl::profiler::TraceMe traceme("H2D Dispatch");
          const tsl::AsyncValueRef<MaybeOwningCpuMemory>& b =
              device_buffer->Buffers()[0];
          CHECK(b.IsConcrete());
          PackOrCopy(shape.element_type(), literal, b->data(), b->size());
          // Signal copy is complete.
          av->SetStateConcrete();
        });
  } else {
    // For tuple, transfer leaf literal individually in parallel.
    for (int i = 0; i < shape.tuple_shapes_size(); ++i) {
      // It is OK to capture `buffer` pointer because the `output_buffer` can't
      // be deleted until all the usage holds have gone away.
      async_work_runner->Schedule([i, literal, av = (*avs)[i].CopyRef(), shape,
                                   device_buffer]() mutable {
        tsl::profiler::TraceMe traceme("H2D Dispatch");
        auto slice = LiteralSlice(literal, {i});
        const tsl::AsyncValueRef<MaybeOwningCpuMemory>& b =
            device_buffer->Buffers()[i];
        CHECK(b.IsConcrete());
        PackOrCopy(slice.shape().element_type(), slice, b->data(), b->size());
        // Signal copy is complete.
        av->SetStateConcrete();
      });
    }
  }
}

/*static*/ absl::StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
AbstractTfrtCpuBuffer::AllocateTrackedDeviceBuffer(
    const Shape& on_device_shape,
    absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events) {
  absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers;
  if (!on_device_shape.IsTuple()) {
    size_t byte_size = ShapeUtil::ByteSizeOf(on_device_shape);
    TF_ASSIGN_OR_RETURN(tsl::AsyncValueRef<MaybeOwningCpuMemory> device_buffer,
                        MaybeOwningCpuMemory::AllocateAvailableAvr(byte_size));
    buffers.push_back(std::move(device_buffer));
    return std::make_unique<TrackedTfrtCpuDeviceBuffer>(
        /*is_tuple=*/false, /*owns_buffers=*/true, std::move(buffers),
        std::move(definition_events));
  }
  // Tuple case.
  buffers.reserve(on_device_shape.tuple_shapes().size());
  for (const auto& leaf_shape : on_device_shape.tuple_shapes()) {
    size_t byte_size = ShapeUtil::ByteSizeOf(leaf_shape);
    TF_ASSIGN_OR_RETURN(tsl::AsyncValueRef<MaybeOwningCpuMemory> device_buffer,
                        MaybeOwningCpuMemory::AllocateAvailableAvr(byte_size));
    buffers.push_back(std::move(device_buffer));
  }
  return std::make_unique<TrackedTfrtCpuDeviceBuffer>(
      /*is_tuple=*/true, /*owns_buffers=*/true, std::move(buffers),
      std::move(definition_events));
}

/*static*/ void AbstractTfrtCpuBuffer::AllocateAvsAndEvents(
    const Shape& shape,
    absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4>* avs,
    absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4>* definition_events) {
  // Nested tuple shapes are not supported here.
  int num_leaf_buffers = shape.IsTuple() ? shape.tuple_shapes_size() : 1;
  for (int i = 0; i < num_leaf_buffers; ++i) {
    tsl::AsyncValueRef<CpuEvent> definition_event =
        tsl::MakeConstructedAsyncValueRef<CpuEvent>();
    definition_events->push_back(definition_event.CopyRef());
    avs->push_back(std::move(definition_event));
  }
}

/*static*/ absl::StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
AbstractTfrtCpuBuffer::BufferFromHostBufferHelper(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    PjRtClient::HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer, const Shape& shape,
    AsyncWorkRunner* async_work_runner, absl::Mutex* transpose_mu,
    TransposePlanCache* transpose_cache) {
  bool has_default_layout =
      !byte_strides || HasMajorToMinorLayout(type, dims, *byte_strides);
  const int bit_width = primitive_util::BitWidth(type);
  // Packed arrays are unpacked on host and packed on device.
  bool is_packed = primitive_util::IsSubByteNonPredType(type);

  // If the input buffer has a default layout and is sufficiently aligned, we
  // can simply point to the input array's data without any further copies. At
  // the time of writing we require a 16-byte alignment because XLA may generate
  // code which requires it.
  bool is_aligned_data = ((absl::bit_cast<std::uintptr_t>(data) &
                           (cpu_function_runtime::MinAlign() - 1)) == 0);

  using HostBufferSemantics = PjRtClient::HostBufferSemantics;
  bool immutable_zero_copy_semantics =
      host_buffer_semantics == HostBufferSemantics::kImmutableZeroCopy;
  bool mutable_zero_copy_semantics =
      host_buffer_semantics == HostBufferSemantics::kMutableZeroCopy;

  bool can_use_zero_copy =
      has_default_layout && !is_packed && is_aligned_data &&
      (immutable_zero_copy_semantics || mutable_zero_copy_semantics);

  absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningCpuMemory>, 4> buffers;
  absl::InlinedVector<tsl::AsyncValueRef<CpuEvent>, 4> definition_events;
  absl::AnyInvocable<void() &&> on_delete_callback;
  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  bool owns_buffers = true;

  if (can_use_zero_copy && mutable_zero_copy_semantics) {
    // For a mutable zero copy semantics we pass a no-op deleter because
    // underlying buffer is owned by the caller and it will free it when
    // PjRt will call `on_done_with_host_buffer` callback.
    MaybeOwningCpuMemory::OwnedDataPtr::deleter_type no_op = +[](void*) {};
    buffers.push_back(tsl::MakeAvailableAsyncValueRef<MaybeOwningCpuMemory>(
        MaybeOwningCpuMemory::OwnedDataPtr(
            reinterpret_cast<uint8_t*>(const_cast<void*>(data)), no_op),
        byte_size));
    on_delete_callback = std::move(on_done_with_host_buffer);

  } else if (can_use_zero_copy && immutable_zero_copy_semantics) {
    // For immutable zero-copy semantics we pass non-owning cpu memory.
    owns_buffers = false;
    buffers.push_back(tsl::MakeAvailableAsyncValueRef<MaybeOwningCpuMemory>(
        const_cast<void*>(data), byte_size));
    on_delete_callback = std::move(on_done_with_host_buffer);

  } else {
    size_t dst_byte_size =
        is_packed ? CeilOfRatio<size_t>(byte_size, 8 / bit_width) : byte_size;
    TF_ASSIGN_OR_RETURN(
        tsl::AsyncValueRef<MaybeOwningCpuMemory> device_buffer,
        MaybeOwningCpuMemory::AllocateAvailableAvr(dst_byte_size));
    auto dst_data_ptr = device_buffer->data();
    buffers.push_back(device_buffer);
    if (!has_default_layout || is_packed) {
      // If the input array does not have a major-to-minor layout, transpose it
      // into major-to-minor layout. Currently we choose to always do this
      // synchronously.
      // TODO(phawkins): consider performing the transpose asynchronously.
      // TODO(phawkins): parallelize the transpose.
      std::shared_ptr<TransposePlan> transpose;
      {
        absl::InlinedVector<int64_t, 4> permutation(dims.size());
        absl::c_iota(permutation, 0);
        TransposePlan::Options options;
        options.elem_size_in_bytes = primitive_util::ByteWidth(type);
        options.dims = dims;
        options.permutation = permutation;
        if (byte_strides) {
          options.input_layout = TransposePlan::Striding{*byte_strides};
        }
        absl::MutexLock lock(transpose_mu);
        TF_ASSIGN_OR_RETURN(transpose, transpose_cache->GetOrCreate(options));
      }
      if (!is_packed) {
        transpose->Execute(data, dst_data_ptr);
      } else {
        // First transpose the unpacked data into a new temporary buffer, then
        // pack the data.
        // TODO(reedwm): Fuse the transpose and packing by having TransposePlan
        // support packing.
        auto data_transposed = std::make_unique<char[]>(byte_size);
        transpose->Execute(data, data_transposed.get());
        absl::Span<const char> src_data_span(data_transposed.get(), byte_size);
        absl::Span<char> dst_data_span(static_cast<char*>(dst_data_ptr),
                                       dst_byte_size);
        PackIntN(bit_width, src_data_span, dst_data_span);
      }
      if (on_done_with_host_buffer) {
        std::move(on_done_with_host_buffer)();
        on_done_with_host_buffer = nullptr;
      }
    } else {
      bool should_sync_copy =
          host_buffer_semantics ==
              PjRtClient::HostBufferSemantics::kImmutableOnlyDuringCall ||
          (byte_size < kSmallDataTransferByteSize);
      if (should_sync_copy) {
        std::memcpy(dst_data_ptr, data, byte_size);
        if (on_done_with_host_buffer) {
          std::move(on_done_with_host_buffer)();
          on_done_with_host_buffer = nullptr;
        }
      } else {
        tsl::AsyncValueRef<CpuEvent> copy_event =
            tsl::MakeConstructedAsyncValueRef<CpuEvent>();
        definition_events.push_back(copy_event.CopyRef());
        async_work_runner->Schedule(
            [device_buffer = std::move(device_buffer), dst_data_ptr, data,
             byte_size, copy_event = std::move(copy_event),
             on_done_with_host_buffer =
                 std::move(on_done_with_host_buffer)]() mutable {
              tsl::profiler::TraceMe traceme("H2D Dispatch");
              std::memcpy(dst_data_ptr, data, byte_size);
              if (on_done_with_host_buffer) {
                std::move(on_done_with_host_buffer)();
                on_done_with_host_buffer = nullptr;
              }
              // Signal copy is complete.
              copy_event.SetStateConcrete();
            });
      }
    }
  }
  return std::make_unique<TrackedTfrtCpuDeviceBuffer>(
      /*is_tuple=*/false, owns_buffers, std::move(buffers),
      std::move(definition_events), std::move(on_delete_callback));
}

AbstractAsyncHostToHostMemoryTransferManager::
    AbstractAsyncHostToHostMemoryTransferManager(
        absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4> avs,
        absl::InlinedVector<std::unique_ptr<AbstractTfrtCpuBuffer>, 4> buffers,
        absl::InlinedVector<TrackedTfrtCpuDeviceBuffer*, 4> device_buffers,
        absl::InlinedVector<size_t, 4> buffer_sizes,
        absl::InlinedVector<int64_t, 4> buffer_transfers_in_flight,
        absl::InlinedVector<bool, 4> last_transfer_finished,
        AsyncWorkRunner* async_work_runner)
    : transfers_in_flight_(0),
      avs_(std::move(avs)),
      buffer_transfers_in_flight_(std::move(buffer_transfers_in_flight)),
      last_transfer_finished_(std::move(last_transfer_finished)),
      buffers_(std::move(buffers)),
      device_buffers_(std::move(device_buffers)),
      buffer_sizes_(std::move(buffer_sizes)),
      async_work_runner_(async_work_runner) {}

AbstractAsyncHostToHostMemoryTransferManager::
    ~AbstractAsyncHostToHostMemoryTransferManager() {
  // Wait for in-flight transfers to finish.
  absl::Condition transfers_finished(
      +[](int* t) { return *t == 0; }, &transfers_in_flight_);
  VLOG(2) << "Waiting for in-flight transfers to finish.";
  absl::MutexLock l(&mu_);
  mu_.Await(transfers_finished);
  for (auto& avref : avs_) {
    auto av = avref;
    if (av && av->IsUnavailable()) {
      av->SetError(absl::InternalError(
          "Async transfer object was deleted before transfers completed."));
    }
  }
  VLOG(2) << "In-flight transfers finished.";
}

size_t AbstractAsyncHostToHostMemoryTransferManager::buffer_size(
    int buffer_index) const {
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffer_sizes_.size());
  return buffer_sizes_[buffer_index];
}

std::unique_ptr<PjRtBuffer>
AbstractAsyncHostToHostMemoryTransferManager::RetrieveBuffer(int buffer_index) {
  absl::MutexLock l(&mu_);
  CHECK_GE(buffer_index, 0);
  CHECK_LT(buffer_index, buffers_.size());
  return std::move(buffers_[buffer_index]);
}

absl::Status
AbstractAsyncHostToHostMemoryTransferManager::TransferLiteralToBuffer(
    int buffer_index, const LiteralSlice& literal,
    absl::AnyInvocable<void() &&> on_done) {
  const Shape& shape = literal.shape();
  if (shape.has_layout() &&
      !LayoutUtil::IsMonotonicWithDim0Major(shape.layout())) {
    return absl::UnimplementedError(
        "PjRt CPU's TransferLiteralToBuffer does not support "
        "non-major-to-minor layout");
  }
  return FillRawDataToSubBuffer(
      buffer_index,
      [literal](void* b, int64_t size) {
        PackOrCopy(literal.shape().element_type(), literal, b, size);
      },
      /*is_last_transfer=*/true, std::move(on_done));
}

absl::Status
AbstractAsyncHostToHostMemoryTransferManager::TransferRawDataToBuffer(
    int buffer_index, absl::string_view data,
    absl::AnyInvocable<void() &&> on_done) {
  return TransferRawDataToSubBuffer(
      buffer_index, data.data(), /*offset=*/0, data.size(),
      /*is_last_transfer=*/true, std::move(on_done));
}

// The definition events of `device_buffers_` must be ready before calling this
// function.
absl::Status
AbstractAsyncHostToHostMemoryTransferManager::TransferRawDataToSubBuffer(
    int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
    bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) {
  return FillRawDataToSubBuffer(
      buffer_index,
      [offset, data, transfer_size](void* b, int64_t size) {
        std::memcpy(reinterpret_cast<char*>(b) + offset, data, transfer_size);
      },
      is_last_transfer, std::move(on_done));
}

absl::Status
AbstractAsyncHostToHostMemoryTransferManager::FillRawDataToSubBuffer(
    int buffer_index,
    absl::AnyInvocable<void(void* data, int64_t size)> fill_fn,
    bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) {
  {
    // We release the lock when out of scope because
    // `async_work_runner_->Schedule` might sometimes run the closure in this
    // thread!
    absl::MutexLock l(&mu_);

    CHECK_GE(buffer_index, 0);
    CHECK_LT(buffer_index, buffers_.size());
    CHECK(!last_transfer_finished_[buffer_index]);
    ++buffer_transfers_in_flight_[buffer_index];
    ++transfers_in_flight_;
  }

  CHECK(async_work_runner_ != nullptr);
  async_work_runner_->Schedule([this, fill_fn = std::move(fill_fn),
                                is_last_transfer, on_done = std::move(on_done),
                                buffer_index]() mutable -> void {
    tsl::RCReference<tsl::AsyncValue> event;
    {
      absl::MutexLock l(&mu_);
      const auto& b = device_buffers_[buffer_index]->Buffers()[0];
      CHECK(b.IsConcrete());
      fill_fn(reinterpret_cast<char*>(b->data()), b->size());
      if (is_last_transfer) {
        last_transfer_finished_[buffer_index] = true;
      }
      --buffer_transfers_in_flight_[buffer_index];
      --transfers_in_flight_;
      if (buffer_transfers_in_flight_[buffer_index] == 0 &&
          last_transfer_finished_[buffer_index]) {
        std::swap(event, avs_[buffer_index]);
      }
    }
    // Call on_done outside the lock because it may call
    // ~AbstractAsyncHostToHostMemoryTransferManager.
    std::move(on_done)();
    if (event) {
      event->SetStateConcrete();
    }
  });
  return absl::OkStatus();
}

void AbstractAsyncHostToHostMemoryTransferManager::SetBufferError(
    int buffer_index, absl::Status error) {
  absl::MutexLock l(&mu_);
  avs_[buffer_index]->SetError(error);
}

/*static*/ absl::Status
AbstractAsyncHostToHostMemoryTransferManager::PopulateAsyncTransferManagerData(
    absl::Span<const std::unique_ptr<AbstractTfrtCpuBuffer>> buffers,
    absl::InlinedVector<TrackedTfrtCpuDeviceBuffer*, 4>& device_buffers,
    absl::InlinedVector<size_t, 4>& buffer_sizes,
    absl::InlinedVector<int64_t, 4>& buffer_transfers_in_flight,
    absl::InlinedVector<bool, 4>& last_transfer_finished) {
  buffer_transfers_in_flight.resize(buffers.size(), 0);
  last_transfer_finished.resize(buffers.size(), false);

  device_buffers.reserve(buffers.size());
  for (const auto& buffer : buffers) {
    // We can make the usage event available right away because the buffer's
    // definition event will be made available after the usage has completed.
    auto usage_event = tsl::MakeAvailableAsyncValueRef<CpuEvent>();
    auto* device_buffer = buffer->AcquireUsage(std::move(usage_event));
    CHECK(device_buffer);
    device_buffers.push_back(device_buffer);
  }

  buffer_sizes.reserve(buffers.size());
  for (const auto& buffer : buffers) {
    TF_ASSIGN_OR_RETURN(auto buffer_size, buffer->GetOnDeviceSizeInBytes());
    buffer_sizes.push_back(buffer_size);
  }

  return absl::OkStatus();
}

}  // namespace xla
