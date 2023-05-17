/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/pjrt/abstract_tfrt_cpu_buffer.h"

#include <cstddef>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/tracked_tfrt_cpu_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/transpose.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/runtime/cpu_event.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_executable.h"
#include "tensorflow/compiler/xla/service/cpu/cpu_xfeed.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/statusor.h"
#include "tensorflow/tsl/profiler/lib/connected_traceme.h"
#include "tfrt/concurrency/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value.h"  // from @tf_runtime
#include "tfrt/host_context/async_value_ref.h"  // from @tf_runtime

namespace xla {
namespace {

using ::xla::runtime::CpuEvent;

constexpr size_t kSmallDataTransferByteSize = 102400;  // 100 KiB

void CopyCpuBufferToLiteral(const Shape& device_shape,
                            TrackedTfrtCpuDeviceBuffer* device_buffer,
                            MutableLiteralBase* literal) {
  if (!device_shape.IsTuple()) {
    const std::shared_ptr<MaybeOwningCpuMemory>& b =
        device_buffer->Buffers()[0];
    std::memcpy(literal->untyped_data(), b->data(),
                ShapeUtil::ByteSizeOf(device_shape));
  } else {
    // Tuple case.
    int num_leaves = literal->shape().tuple_shapes().size();
    for (int i = 0; i < num_leaves; ++i) {
      const std::shared_ptr<MaybeOwningCpuMemory>& b =
          device_buffer->Buffers()[i];
      std::memcpy(
          literal->untyped_data({i}), b->data(),
          ShapeUtil::ByteSizeOf(ShapeUtil::GetSubshape(device_shape, {i})));
    }
  }
}

ShapedBuffer AsShapedBuffer(
    int device_ordinal, const Shape& on_device_shape,
    absl::Span<const std::shared_ptr<MaybeOwningCpuMemory>> buffers) {
  ShapedBuffer shaped_buffer(on_device_shape, device_ordinal);
  ShapeTree<se::DeviceMemoryBase>::iterator iterator =
      shaped_buffer.buffers().begin();
  for (const auto& buf : buffers) {
    CHECK(iterator != shaped_buffer.buffers().end());
    iterator->second = se::DeviceMemoryBase(buf->data(), buf->size());
    ++iterator;
  }
  CHECK(iterator == shaped_buffer.buffers().end());
  return shaped_buffer;
}

}  //  namespace

UnpinnedHostMemorySpace::UnpinnedHostMemorySpace(int id, PjRtClient* client)
    : id_(id), client_(client) {
  debug_string_ = absl::StrFormat("UnpinnedHostMemorySpace(id=%i), client: %s",
                                  id_, client_->platform_name());
}

AbstractTfrtCpuBuffer::AbstractTfrtCpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer)
    : on_device_shape_(std::move(on_device_shape)),
      tracked_device_buffer_(std::move(tracked_device_buffer)) {}

AbstractTfrtCpuBuffer::~AbstractTfrtCpuBuffer() {
  AbstractTfrtCpuBuffer::Delete();
  CHECK_EQ(external_reference_counter_, 0);
}

StatusOr<Shape> AbstractTfrtCpuBuffer::logical_on_device_shape() {
  if (on_device_shape_.is_static()) {
    return on_device_shape_;
  }

  auto usage_event = tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
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
    return InternalError("Error Execute: %s", error->message());
  }

  ShapedBuffer shaped_buffer =
      AsShapedBuffer(device()->local_hardware_id(), on_device_shape_,
                     device_buffer->Buffers());
  Shape ret_shape = on_device_shape_;
  TF_RETURN_IF_ERROR(ReadDynamicShapesOnCpu(
      &shaped_buffer, &ret_shape, cpu::CpuExecutable::ShapeSizeBytes));
  return ret_shape;
}

StatusOr<size_t> AbstractTfrtCpuBuffer::GetOnDeviceSizeInBytes() const {
  return ShapeUtil::ByteSizeOf(on_device_shape_);
}

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
AbstractTfrtCpuBuffer::AcquireExternalReference() {
  class ScopedExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit ScopedExternalReference(AbstractTfrtCpuBuffer* buffer,
                                     std::shared_ptr<MaybeOwningCpuMemory> data)
        : buffer_(buffer), data_(std::move(data)) {
      DCHECK(data_);
      data_ptr_ = data_->data();
    }

    ~ScopedExternalReference() override { buffer_->DropExternalReference(); }

   private:
    AbstractTfrtCpuBuffer* buffer_ = nullptr;
    // Keep a reference to the underlying data used. Note that it is still
    // users' responsibility to synchronize reads and writes to the data.
    std::shared_ptr<MaybeOwningCpuMemory> data_;
  };

  absl::MutexLock lock(&mu_);
  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Buffer has been deleted or donated.");
  }

  ++external_reference_counter_;

  return {std::make_unique<ScopedExternalReference>(
      this, tracked_device_buffer_->Buffers()[0])};
}

class TrackedCpuDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedCpuDeviceBufferExternalReference(
      std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    data_ptr_ = tracked_device_buffer_->Buffers()[0]->data();
  }

  ~TrackedCpuDeviceBufferExternalReference() override = default;

 private:
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer_;
};

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
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
  auto device_buffer = ReleaseBufferLocked();
  if (device_buffer == nullptr) return;

  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> usage_events =
      device_buffer->LockUseAndTransferUsageEvents();

  std::vector<tfrt::AsyncValue*> event_avs;
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

bool AbstractTfrtCpuBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_ == nullptr;
}

std::unique_ptr<TrackedTfrtCpuDeviceBuffer>
AbstractTfrtCpuBuffer::ReleaseBufferLocked() {
  absl::MutexLock lock(&mu_);
  auto condition = [this]() ABSL_SHARED_LOCKS_REQUIRED(mu_) {
    return !pending_donation_;
  };
  mu_.Await(absl::Condition(&condition));
  return std::move(tracked_device_buffer_);
}

StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
AbstractTfrtCpuBuffer::Release(bool wait_for_operations_to_complete) {
  std::unique_ptr<TrackedTfrtCpuDeviceBuffer> device_buffer =
      ReleaseBufferLocked();
  if (device_buffer == nullptr) return {nullptr};

  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> events;
  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  events = device_buffer->LockUseAndTransferUsageEvents();

  if (wait_for_operations_to_complete) {
    // Block the host until all usage events have completed. Usage events
    // dominate definition events, so this also waits for the buffer to be
    // defined. Return the first error encountered.
    Status first_error;
    for (const auto& av : events) {
      BlockUntilReady(av.GetAsyncValue());
      if (auto* error = av.GetErrorIfPresent()) {
        first_error.Update(
            InternalError("Error Execute: %s", error->message()));
      }
    }
    if (!first_error.ok()) return std::move(first_error);
  }

  return device_buffer;
}

TrackedTfrtCpuDeviceBuffer* AbstractTfrtCpuBuffer::AcquireUsage(
    tfrt::AsyncValueRef<CpuEvent> usage_event) {
  absl::MutexLock lock(&mu_);
  if (!tracked_device_buffer_) {
    return nullptr;
  }

  tracked_device_buffer_->AddUsageEvents(absl::MakeSpan(&usage_event, 1));
  return tracked_device_buffer_.get();
}

StatusOr<AbstractTfrtCpuBuffer::DonationTransaction>
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

PjRtFuture<Status> AbstractTfrtCpuBuffer::ToLiteralHelper(
    MutableLiteralBase* literal, AsyncWorkRunner* async_work_runner) {
  std::string message = absl::StrCat(buffer_name(), "::ToLiteral");
  absl::string_view message_view(message);
  tsl::profiler::TraceMe traceme(message_view);
  if (IsEmptyTuple()) {
    return PjRtFuture<Status>(
        InvalidArgument("ToLiteral called on empty tuple"));
  }
  auto usage_event = tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return PjRtFuture<Status>(InvalidArgument(
        "CopyToHostAsync() called on deleted or donated buffer"));
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  std::vector<tfrt::RCReference<tfrt::AsyncValue>> device_buffer_wait_avs = {
      device_buffer->definition_event().CopyRCRef()};
  std::vector<tfrt::RCReference<tfrt::AsyncValue>> device_buffer_wait_avs_copy =
      {device_buffer->definition_event().CopyRCRef()};

  bool should_sync_copy = device_buffer_wait_avs.empty() &&
                          literal->size_bytes() < kSmallDataTransferByteSize;
  StatusOr<Shape> device_shape = logical_on_device_shape();
  if (!device_shape.ok()) {
    return PjRtFuture<Status>(device_shape.status());
  }
  if (should_sync_copy) {
    CopyCpuBufferToLiteral(*device_shape, device_buffer, literal);
    // Unblock ToLiteral caller.
    return PjRtFuture<Status>(OkStatus());
  } else {
    auto ready_event = tfrt::MakeUnconstructedAsyncValueRef<Status>();
    // Wait for buffer definition events to finish before d2h dispatch. D2H
    // dispatch should be in parallel, e.g. one Execute event finish may trigger
    // multiple outputs' D2H, they should happen in different threads in
    // parallel.
    async_work_runner->ScheduleWhenReady(
        device_buffer_wait_avs,
        [device_buffer_wait_avs = std::move(device_buffer_wait_avs_copy),
         literal, ready_event = ready_event.CopyRef(), device_buffer,
         device_shape, ready_on_exit = std::move(ready_on_exit)]() mutable {
          tsl::profiler::TraceMe traceme("D2H Dispatch");
          // Errors in src buffer are surfaced to user.
          for (const auto& av : device_buffer_wait_avs) {
            if (auto* error = av->GetErrorIfPresent()) {
              ready_event.emplace(Internal("Error converting to literal: %s",
                                           error->message()));
              return;
            }
          }
          CopyCpuBufferToLiteral(*device_shape, device_buffer, literal);
          // Unblock ToLiteral event.
          ready_event.emplace(OkStatus());
        });
    return PjRtFuture<Status>(
        std::move(ready_event),
        /*on_block_start=*/
        [message]() {
          absl::string_view message_view(message);
          tsl::profiler::TraceMeProducer traceme(message_view);
          VLOG(1) << message_view;
          return PjRtFutureHelpers::ProfilingKeys(
              {/*traceme_context_id =*/traceme.GetContextId()});
        },
        /*on_block_end=*/
        [message](PjRtFutureHelpers::ProfilingKeys keys) {
          absl::string_view message_view(message);
          tsl::profiler::TraceMeConsumer traceme(message_view,
                                                 keys.traceme_context_id);
        });
  }
}

StatusOr<std::unique_ptr<PjRtBuffer>>
AbstractTfrtCpuBuffer::CopyToDeviceAcrossClients(PjRtDevice* dst_device) {
  TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
  // Avoid use-after-free on `literal` due to unsequenced move and use.
  Literal* literal_pointer = literal.get();
  absl::InlinedVector<int64_t, 4> byte_strides(
      literal->shape().dimensions_size());
  TF_RETURN_IF_ERROR(
      ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
  return dst_device->client()->BufferFromHostBuffer(
      literal_pointer->untyped_data(), literal_pointer->shape().element_type(),
      literal_pointer->shape().dimensions(), byte_strides,
      PjRtClient::HostBufferSemantics::kZeroCopy,
      [literal{std::move(literal)}]() { /* frees literal */ }, dst_device);
}

StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
AbstractTfrtCpuBuffer::CopyToDeviceHelper(AsyncWorkRunner* async_work_runner) {
  // Copy each leaf buffer to a destination buffer.
  auto usage_event = tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
  auto* src_device_buffer = AcquireUsage(usage_event);
  if (src_device_buffer == nullptr) {
    return InvalidArgument("CopyToDevice called on deleted or donated buffer");
  }
  MarkEventReadyOnExit ready_on_exit(std::move(usage_event));

  int num_leaf_buffers = src_device_buffer->Buffers().size();
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> src_buffers;
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> dst_buffers;
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> dst_definition_events;
  src_buffers.reserve(num_leaf_buffers);
  dst_buffers.reserve(num_leaf_buffers);
  dst_definition_events.reserve(num_leaf_buffers);

  for (int i = 0; i < num_leaf_buffers; ++i) {
    auto src_buffer = src_device_buffer->Buffers()[i];
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<MaybeOwningCpuMemory> dst_buffer,
        MaybeOwningCpuMemory::AllocateShared(src_buffer->size()));
    src_buffers.push_back(std::move(src_buffer));
    dst_buffers.push_back(std::move(dst_buffer));
    dst_definition_events.push_back(
        tfrt::MakeConstructedAsyncValueRef<CpuEvent>());
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
      on_device_shape_.IsTuple(), std::move(dst_buffers),
      std::move(dst_definition_events));
}

PjRtFuture<Status> AbstractTfrtCpuBuffer::GetReadyFuture() {
  tfrt::AsyncValueRef<CpuEvent> definition_event;
  {
    absl::MutexLock lock(&mu_);
    if (!tracked_device_buffer_) {
      return PjRtFuture<Status>(InvalidArgument(
          "GetReadyFuture() called on deleted or donated buffer"));
    }
    definition_event = tracked_device_buffer_->definition_event();
  }
  DCHECK(definition_event);

  if (definition_event.IsAvailable()) {
    if (definition_event.IsError()) {
      return PjRtFuture<Status>(
          FailedPrecondition("Buffer Definition Event: %s",
                             definition_event.GetError().message()));
    }
    return PjRtFuture<Status>(OkStatus());
  } else {
    tfrt::AsyncValueRef<Status> status_event =
        tfrt::MakeUnconstructedAsyncValueRef<Status>();

    definition_event.AndThen(
        [definition_event = definition_event.AsPtr(), status_event]() {
          if (definition_event.IsError()) {
            status_event.emplace(
                FailedPrecondition("Buffer Definition Event: %s",
                                   definition_event.GetError().message()));
          } else {
            status_event.emplace(OkStatus());
          }
        });

    std::string message = absl::StrCat(buffer_name(), "::Await");
    return PjRtFuture<Status>(
        std::move(status_event),
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

void AbstractTfrtCpuBuffer::CopyFromLiteral(
    const LiteralSlice& literal, const Shape& shape,
    absl::InlinedVector<tfrt::RCReference<tfrt::AsyncValue>, 4>* avs,
    AsyncWorkRunner* async_work_runner) {
  auto usage_event = tfrt::MakeAvailableAsyncValueRef<CpuEvent>();
  auto* device_buffer = AcquireUsage(std::move(usage_event));
  CHECK(device_buffer);
  if (!shape.IsTuple()) {
    // It is OK to capture `buffer` pointer because the `output_buffer` can't be
    // deleted until all the usage holds have gone away.
    async_work_runner->Schedule(
        [literal, av = (*avs)[0].CopyRef(), device_buffer, shape]() mutable {
          tsl::profiler::TraceMe traceme("H2D Dispatch");
          const std::shared_ptr<MaybeOwningCpuMemory>& b =
              device_buffer->Buffers()[0];
          CHECK_EQ(literal.size_bytes(), b->size());
          std::memcpy(b->data(), literal.untyped_data(), b->size());
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
        const std::shared_ptr<MaybeOwningCpuMemory>& b =
            device_buffer->Buffers()[i];
        CHECK_EQ(slice.size_bytes(), b->size());
        std::memcpy(b->data(), slice.untyped_data(), slice.size_bytes());
        // Signal copy is complete.
        av->SetStateConcrete();
      });
    }
  }
}

/*static*/ StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
AbstractTfrtCpuBuffer::AllocateTrackedDeviceBuffer(
    const Shape& on_device_shape,
    absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events) {
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> buffers;
  if (!on_device_shape.IsTuple()) {
    size_t byte_size = ShapeUtil::ByteSizeOf(on_device_shape);
    TF_ASSIGN_OR_RETURN(std::shared_ptr<MaybeOwningCpuMemory> device_buffer,
                        MaybeOwningCpuMemory::AllocateShared(byte_size));
    buffers.push_back(std::move(device_buffer));
    return std::make_unique<TrackedTfrtCpuDeviceBuffer>(
        /*is_tuple=*/false, std::move(buffers), std::move(definition_events));
  }
  // Tuple case.
  buffers.reserve(on_device_shape.tuple_shapes().size());
  for (const auto& leaf_shape : on_device_shape.tuple_shapes()) {
    size_t byte_size = ShapeUtil::ByteSizeOf(leaf_shape);
    TF_ASSIGN_OR_RETURN(std::shared_ptr<MaybeOwningCpuMemory> device_buffer,
                        MaybeOwningCpuMemory::AllocateShared(byte_size));
    buffers.push_back(std::move(device_buffer));
  }
  return std::make_unique<TrackedTfrtCpuDeviceBuffer>(
      /*is_tuple=*/true, std::move(buffers), std::move(definition_events));
}

/*static*/ void AbstractTfrtCpuBuffer::AllocateAvsAndEvents(
    const Shape& shape,
    absl::InlinedVector<tfrt::RCReference<tfrt::AsyncValue>, 4>* avs,
    absl::InlinedVector<tfrt::AsyncValueRef<runtime::CpuEvent>, 4>*
        definition_events) {
  // Nested tuple shapes are not supported here.
  int num_leaf_buffers = shape.IsTuple() ? shape.tuple_shapes_size() : 1;
  for (int i = 0; i < num_leaf_buffers; ++i) {
    tfrt::AsyncValueRef<CpuEvent> definition_event =
        tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
    definition_events->push_back(definition_event.CopyRef());
    avs->push_back(std::move(definition_event));
  }
}

/*static*/ StatusOr<std::unique_ptr<TrackedTfrtCpuDeviceBuffer>>
AbstractTfrtCpuBuffer::BufferFromHostBufferHelper(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    PjRtClient::HostBufferSemantics host_buffer_semantics,
    std::function<void()> on_done_with_host_buffer, const Shape& shape,
    AsyncWorkRunner* async_work_runner, absl::Mutex* transpose_mu,
    TransposePlanCache* transpose_cache) {
  bool has_default_layout =
      !byte_strides || HasMajorToMinorLayout(type, dims, *byte_strides);
  // If the input buffer has a default layout and is sufficiently aligned, we
  // can simply point to the input array's data without any further copies. At
  // the time of writing we require a 16-byte alignment because XLA may generate
  // code which requires it.
  bool can_use_zero_copy =
      has_default_layout &&
      host_buffer_semantics == PjRtClient::HostBufferSemantics::kZeroCopy &&
      ((absl::bit_cast<std::uintptr_t>(data) &
        (cpu_function_runtime::MinAlign() - 1)) == 0);
  absl::InlinedVector<std::shared_ptr<MaybeOwningCpuMemory>, 4> buffers;
  absl::InlinedVector<tfrt::AsyncValueRef<CpuEvent>, 4> definition_events;
  std::function<void()> on_delete_callback;
  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  if (can_use_zero_copy) {
    auto device_buffer = std::make_shared<MaybeOwningCpuMemory>(
        const_cast<void*>(data), byte_size);
    buffers.push_back(std::move(device_buffer));
    on_delete_callback = std::move(on_done_with_host_buffer);
  } else {
    TF_ASSIGN_OR_RETURN(std::shared_ptr<MaybeOwningCpuMemory> device_buffer,
                        MaybeOwningCpuMemory::AllocateShared(byte_size));
    auto dst_data_ptr = device_buffer->data();
    buffers.push_back(device_buffer);
    if (!has_default_layout) {
      // If the input array does not have a major-to-minor layout, transpose it
      // into major-to-minor layout. Currently we choose to always do this
      // synchronously.
      // TODO(phawkins): consider performing the transpose asynchronously.
      // TODO(phawkins): parallelize the transpose.
      std::shared_ptr<TransposePlan> transpose;
      {
        absl::InlinedVector<int64_t, 4> permutation(dims.size());
        absl::c_iota(permutation, 0);
        absl::MutexLock lock(transpose_mu);
        TF_ASSIGN_OR_RETURN(
            transpose, transpose_cache->GetOrCreate(
                           primitive_util::ByteWidth(type), dims, permutation,
                           TransposePlan::Striding{*byte_strides}));
      }
      transpose->Execute(data, dst_data_ptr);
      if (on_done_with_host_buffer) {
        on_done_with_host_buffer();
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
          on_done_with_host_buffer();
          on_done_with_host_buffer = nullptr;
        }
      } else {
        tfrt::AsyncValueRef<CpuEvent> copy_event =
            tfrt::MakeConstructedAsyncValueRef<CpuEvent>();
        definition_events.push_back(copy_event.CopyRef());
        async_work_runner->Schedule(
            [device_buffer = std::move(device_buffer), dst_data_ptr, data,
             byte_size, copy_event = std::move(copy_event),
             on_done_with_host_buffer =
                 std::move(on_done_with_host_buffer)]() mutable {
              tsl::profiler::TraceMe traceme("H2D Dispatch");
              std::memcpy(dst_data_ptr, data, byte_size);
              if (on_done_with_host_buffer) {
                on_done_with_host_buffer();
                on_done_with_host_buffer = nullptr;
              }
              // Signal copy is complete.
              copy_event.SetStateConcrete();
            });
      }
    }
  }
  return std::make_unique<TrackedTfrtCpuDeviceBuffer>(
      /*is_tuple=*/false, std::move(buffers), std::move(definition_events),
      std::move(on_delete_callback));
}

}  // namespace xla
