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
#include <memory>
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
#include "tensorflow/compiler/xla/runtime/cpu_event.h"
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

using ::xla::runtime::CpuEvent;

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

AbstractTfrtCpuBuffer::AbstractTfrtCpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedTfrtCpuDeviceBuffer> tracked_device_buffer)
    : on_device_shape_(std::move(on_device_shape)),
      tracked_device_buffer_(std::move(tracked_device_buffer)) {}

AbstractTfrtCpuBuffer::~AbstractTfrtCpuBuffer() {
  AbstractTfrtCpuBuffer::Delete();
  CHECK_EQ(external_reference_counter_, 0);
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

}  // namespace xla
