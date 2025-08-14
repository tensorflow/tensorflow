/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_buffer.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>

#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/layout.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/gpu/tfrt/utils.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

TfrtGpuBuffer::TfrtGpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer,
    TfrtGpuClient* client, TfrtGpuDevice* device, PjRtMemorySpace* memory_space)
    : client_(client),
      on_device_shape_(std::move(on_device_shape)),
      device_(device),
      memory_space_(CHECK_NOTNULL(memory_space)),
      tracked_device_buffer_(std::move(tracked_device_buffer)),
      donation_event_(tsl::MakeAvailableAsyncValueRef<bool>(false)),
      external_references_dropped_event_(
          tsl::MakeConstructedAsyncValueRef<GpuEvent>()) {}

TfrtGpuBuffer::~TfrtGpuBuffer() { Delete(); }

absl::StatusOr<size_t> TfrtGpuBuffer::GetOnDeviceSizeInBytes() const {
  return ShapeUtil::ByteSizeOf(on_device_shape_);
}

TrackedGpuDeviceBuffer* TfrtGpuBuffer::AcquireUsage(
    tsl::AsyncValueRef<GpuEvent> usage_event) {
  absl::MutexLock lock(&mu_);
  if (!tracked_device_buffer_) {
    return nullptr;
  }

  tracked_device_buffer_->AddUsageEvents(absl::MakeSpan(&usage_event, 1));
  return tracked_device_buffer_.get();
}

absl::StatusOr<Shape> TfrtGpuBuffer::logical_on_device_shape() {
  if (on_device_shape_.is_static()) {
    return on_device_shape_;
  }

  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return InvalidArgument(
        "logical_on_device_shape() called on deleted or donated buffer");
  }
  MarkGpuEventReadyOnExit ready_on_exit(usage_event);

  auto get_shape = [this, device_buffer]() -> absl::StatusOr<Shape> {
    if (auto* error = device_buffer->definition_event().GetErrorIfPresent()) {
      return *error;
    }

    const auto& buffer = device_buffer->buffer();

    ShapedBuffer shaped_buffer =
        buffer->AsShapedBuffer(on_device_shape_, device_);
    Shape ret_shape = on_device_shape_;
    TransferManager* transfer_manager =
        client_->xla_client()->backend().transfer_manager();

    auto stream = device_->stream();
    TF_RETURN_IF_ERROR(transfer_manager->ReadDynamicShapes(
        stream, &shaped_buffer, &ret_shape));
    {
      tsl::profiler::TraceMe traceme("BlockHostUntilDone");
      TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    }
    return ret_shape;
  };

  absl::StatusOr<Shape> shape_or;
  EnqueueWorkWhenReady(client_->blocking_thread_pool(),
                       {device_buffer->definition_event().CopyRCRef()},
                       [get_shape = std::move(get_shape), &shape_or,
                        usage_event_holder = std::move(ready_on_exit)]() {
                         shape_or = get_shape();
                       });

  tsl::BlockUntilReady(usage_event);
  return shape_or;
}

PjRtFuture<> TfrtGpuBuffer::GetReadyFuture() {
  VLOG(4) << "TfrtGpuBuffer::GetReadyFuture";
  absl::MutexLock lock(&mu_);
  if (!tracked_device_buffer_) {
    return PjRtFuture<>(InvalidArgument(
        "GetReadyFuture() called on deleted or donated buffer"));
  }
  if (!ready_promise_) {
    ready_promise_ =
        CreatePromiseForEvent(tracked_device_buffer_->ready_event());
  }
  return PjRtFuture<>(
      ready_promise_,
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme("TfrtGpuBuffer::Await");
        VLOG(4) << "TfrtGpuBuffer::Await";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id=*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme("TfrtGpuBuffer::Await",
                                               keys.traceme_context_id);
      });
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuBuffer::DonateWithControlDependency(PjRtFuture<> dependency) {
  VLOG(4) << "TfrtGpuBuffer::DonateWithControlDependency";

  TF_ASSIGN_OR_RETURN(DonationTransaction donation_transaction,
                      AcquireDonation());

  TrackedGpuDeviceBuffer* tracked_buffer = donation_transaction.device_buffer();

  if (tracked_buffer == nullptr) {
    return InvalidArgument(
        "DonateWithControlDependency was called on a deleted or donated "
        "buffer.");
  }

  // Combine the original definition event and usage event.
  tsl::AsyncValueRef<GpuEvent> usage_definition_events =
      AfterAll({tracked_buffer->LockUseAndTransferUsageEvents(),
                tracked_buffer->definition_event()});

  // Create an event for `dependency`.
  tsl::AsyncValueRef<GpuEvent> dependency_event =
      tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  dependency.OnReady([dependency_event](absl::Status status) {
    if (status.ok()) {
      dependency_event.SetStateConcrete();
    } else {
      dependency_event.SetError(status);
    }
  });

  // Create new buffer with the combined event and underlying data from the
  // original buffer.
  tsl::AsyncValueRef<GpuEvent> new_definition_event =
      AfterAll({usage_definition_events, dependency_event});
  auto new_tracked_buffer = std::make_unique<TrackedGpuDeviceBuffer>(
      tracked_buffer->buffer(), std::move(new_definition_event),
      tracked_buffer->ready_event(),
      std::move(tracked_buffer->on_delete_callback_));

  auto new_pjrt_buffer = std::make_unique<TfrtGpuBuffer>(
      on_device_shape_, std::move(new_tracked_buffer), client_, device_,
      memory_space_);

  // Commit will set the underlying device buffer unowned. This may break other
  // ongoing users. Only commit after all the pending definition and usage
  // events are ready.
  usage_definition_events.AndThen(
      [donation_transaction = std::move(donation_transaction)]() mutable {
        std::move(donation_transaction).Commit();
      });

  return new_pjrt_buffer;
}

PjRtDevice* TfrtGpuBuffer::device() const { return device_; }

PjRtClient* TfrtGpuBuffer::client() const { return client_; }

bool TfrtGpuBuffer::IsOnCpu() const {
  return memory_space() != nullptr &&
         memory_space()->kind() == PinnedHostMemorySpace::kKind;
}

const tsl::AsyncValueRef<GpuDeviceMemory>& TfrtGpuBuffer::GetBufferPtr() const {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_->buffer();
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtGpuBuffer::AcquireExternalReference() {
  class ScopedExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit ScopedExternalReference(TfrtGpuBuffer* buffer,
                                     tsl::AsyncValueRef<GpuDeviceMemory> data)
        : buffer_(buffer), data_(std::move(data)) {
      DCHECK(data_);
      data_ptr_ = data_->buffer().opaque();
    }

    ~ScopedExternalReference() override { buffer_->DropExternalReference(); }

   private:
    TfrtGpuBuffer* buffer_ = nullptr;
    // Keep a reference to the underlying data used. Note that it is still
    // users' responsibility to synchronize reads and writes to the data.
    tsl::AsyncValueRef<GpuDeviceMemory> data_;
  };

  absl::MutexLock lock(&mu_);
  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Buffer has been deleted or donated.");
  }

  // If the external reference event is concrete, it means we previously dropped
  // the last external reference but want to create one again without having
  // deleted the buffer. So we need a new external_references_dropped_event_.
  if (external_references_dropped_event_.IsConcrete()) {
    external_references_dropped_event_ =
        tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  }

  ++external_reference_counter_;

  tsl::BlockUntilReady(tracked_device_buffer_->definition_event());
  if (tracked_device_buffer_->definition_event().IsError()) {
    return tracked_device_buffer_->definition_event().GetError();
  }
  return {std::make_unique<ScopedExternalReference>(
      this, tracked_device_buffer_->buffer())};
}

}  // namespace xla
