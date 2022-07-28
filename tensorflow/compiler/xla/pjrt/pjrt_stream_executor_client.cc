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

// Implementation notes:
//
// Asynchronous execution:
// -----------------------
//
// Computations and host-to-device transfers do not need to block the host
// waiting for the operation to complete but instead return control to the host
// immediately. This allows client logic to overlap with device-side
// computation.
//
// For a good user experience, we must be careful only to enqueue operations
// that are unlikely to fail; as a rule error checking must be done eagerly
// before returning control to the client.
//
// The degree to which the client can enqueue operations ahead of the client
// is limited by a semaphore. There are at two modes: asynchronous, where we
// allow the client to enqueue up to 32 executions ahead of the device, and
// synchronous, where we limit the client to having one enqueued operation at
// a time. The value of 32 is arbitrary.
//
// Even in asynchronous mode, it is important that we do not permit
// unbounded queue-ahead. Firstly it is problematic when the user does something
// like the following in Python:
// %timeit run_computation()
// To the timeit logic, op() appears to be extremely cheap since it is deferring
// all of its real work and not blocking, and so the %timeit will run op() many
// (e.g., 10000) times to get better timing resolution, even though in reality
// it may be expensive. Secondly, on CPU the allocator is synchronized with the
// head of the compute stream, and we allocate buffers for all of the enqueued
// programs without any reuse (unlike GPU). This means that the memory usage
// is proportional to the queue size.
//
// Multi-stream execution:
// -----------------------
//
// We use a multistream execution design, where different Streams are used for
// host-to-device transfers, device-to-host transfers, and compute. This allows
// us to overlap transfers on and off the device with computation.
//
// Synchronization between streams occurs via BufferSequencingEvents that
// describe when the contents of a logical buffer are known to be valid on
// a particular stream, and when a buffer's uses have all completed.
//
// Synchronous vs asynchronous deallocation:
// -----------------------------------------
//
// See the comment on LocalDeviceState::AllocationModel for a discussion of the
// different allocation semantics on CPU, GPU, and TPU.

#include "tensorflow/compiler/xla/pjrt/pjrt_stream_executor_client.h"

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/cpu_function_runtime.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/pjrt/distributed/protocol.pb.h"
#include "tensorflow/compiler/xla/pjrt/event_pool.h"
#include "tensorflow/compiler/xla/pjrt/local_device_state.h"
#include "tensorflow/compiler/xla/pjrt/metrics.h"
#include "tensorflow/compiler/xla/pjrt/mlir_to_hlo.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_client.h"
#include "tensorflow/compiler/xla/pjrt/pjrt_future.h"
#include "tensorflow/compiler/xla/pjrt/tracked_device_buffer.h"
#include "tensorflow/compiler/xla/pjrt/utils.h"
#include "tensorflow/compiler/xla/service/computation_layout.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/hlo_cost_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_input_output_alias_config.h"
#include "tensorflow/compiler/xla/service/maybe_owning_device_memory.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/service/transfer_manager.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/fingerprint.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/profiler/lib/connected_traceme.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/profiler/lib/traceme_encode.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/host/host_platform_id.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream.h"

namespace xla {

PjRtPlatformId PjRtStreamExecutorDevice::platform_id() const {
  return client_->platform_id();
}
absl::string_view PjRtStreamExecutorDevice::platform_name() const {
  return client_->platform_name();
}

StatusOr<LocalDeviceState*> PjRtStreamExecutorDevice::GetLocalDeviceState()
    const {
  if (local_device_state_) {
    return local_device_state_.get();
  }
  return InvalidArgument("Device %s is not a local device.", DebugString());
}

absl::string_view PjRtStreamExecutorDevice::DebugString() const {
  return debug_string_;
}

std::string PjRtStreamExecutorDevice::ToString() const {
  return absl::StrCat(platform_name(), "(id=", id(), ")");
}

StatusOr<DeviceAssignment> DevicesToDeviceAssignment(
    absl::Span<const std::vector<PjRtDevice*>> devices) {
  if (devices.empty()) {
    return InvalidArgument(
        "Device assignment passed to Compile() must be non-empty.");
  }
  if (devices[0].empty()) {
    return InvalidArgument(
        "Device assignment passed to Compile() must have a nonzero number of "
        "partitions per replica; replica 0 had 0 partitions.");
  }
  DeviceAssignment xla_assignment(devices.size(), devices[0].size());
  for (int replica = 0; replica < devices.size(); ++replica) {
    if (devices[replica].size() != devices[0].size()) {
      return InvalidArgument(
          "Device assignment passed to Compile() has different numbers of "
          "partitions between replicas; %d partitions for replica %d versus %d "
          "partitions for replica 0.",
          devices[replica].size(), replica, devices[0].size());
    }
    for (int partition = 0; partition < devices[replica].size(); ++partition) {
      if (devices[0][0]->client()->platform_id() !=
          devices[replica][partition]->client()->platform_id()) {
        return InvalidArgument(
            "Device assignment passed to Compile() must have devices of a "
            "single kind, got %s for replica 0 partition 0 and %s for replica "
            "%d partition %d.",
            devices[0][0]->client()->platform_name(),
            devices[replica][partition]->client()->platform_name(), replica,
            partition);
      }
      xla_assignment(replica, partition) = devices[replica][partition]->id();
    }
  }
  return xla_assignment;
}

class CpuAllocator : public tensorflow::Allocator {
 public:
  CpuAllocator() = default;

  std::string Name() override { return "cpu"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return tensorflow::port::AlignedMalloc(num_bytes, alignment);
  }
  void DeallocateRaw(void* ptr) override {
    return tensorflow::port::AlignedFree(ptr);
  }
};

PjRtStreamExecutorClient::PjRtStreamExecutorClient(
    std::string platform_name, LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index, std::unique_ptr<se::DeviceMemoryAllocator> allocator,
    std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
    bool should_stage_host_to_device_transfers,
    std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options)
    : platform_id_(tensorflow::Fingerprint64(platform_name)),
      platform_name_(std::move(platform_name)),
      client_(client),
      host_memory_allocator_(std::move(host_memory_allocator)),
      owned_allocator_(std::move(allocator)),
      owned_devices_(std::move(devices)),
      process_index_(process_index),
      should_stage_host_to_device_transfers_(
          should_stage_host_to_device_transfers),
      gpu_run_options_(std::move(gpu_run_options)),
      thread_pool_(
          tensorflow::Env::Default(), "pjrt_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), client->device_count())),
      transpose_cache_(1024) {
  if (owned_allocator_ != nullptr) {
    allocator_ = owned_allocator_.get();
  } else {
    allocator_ = client_->backend().memory_allocator();
  }

  if (!host_memory_allocator_) {
    host_memory_allocator_ = std::make_unique<CpuAllocator>();
  }

  for (const std::unique_ptr<PjRtStreamExecutorDevice>& device :
       owned_devices_) {
    devices_.push_back(device.get());
    CHECK(id_to_device_.insert({device->id(), device.get()}).second)
        << "Duplicate device id: " << device->id();

    if (device->IsAddressable()) {
      addressable_devices_.push_back(device.get());
    }
    device->SetClient(this);
  }
  // TODO(phawkins): we don't really promise anything about the order of
  // these devices, but users may be depending on the current order. Sort into
  // device ordinal order, which is the historical order these values have
  // appeared.
  absl::c_sort(addressable_devices_,
               [](const PjRtDevice* a, const PjRtDevice* b) {
                 return a->local_hardware_id() < b->local_hardware_id();
               });
}

StatusOr<DeviceAssignment> PjRtStreamExecutorClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return client_->backend().computation_placer()->AssignDevices(num_replicas,
                                                                num_partitions);
}

StatusOr<std::unique_ptr<HloCostAnalysis>>
PjRtStreamExecutorClient::GetHloCostAnalysis() {
  return std::make_unique<HloCostAnalysis>(
      client_->backend().compiler()->ShapeSizeBytesFunction());
}

namespace {

// Ensures that it is safe to deallocate any buffers that have been enqueued in
// an operation on stream. Called only in rare error cases that are triggered
// during enqueue. These cases generally correspond to resource exhaustion.
void StallStreamOnError(LocalDeviceState* local_device, se::Stream* stream) {
  switch (local_device->allocation_model()) {
    case LocalDeviceState::kAsynchronous:
      // We can safely deallocate any dangling buffers immediately. NOTE: this
      // assumes that any buffers enqueued on stream are local to stream's
      // executor, and manual action may be needed if that condition is not met.
      break;

    case LocalDeviceState::kComputeSynchronized:
      // This will stall computation but that's ok in this very rare error
      // case.
      if (stream != local_device->compute_stream()) {
        local_device->compute_stream()->ThenWaitFor(stream);
      }
      break;

    case LocalDeviceState::kSynchronous:
      // This will stall the calling thread but that's ok in this very rare
      // error case. If the stall fails just crash, since we have no other
      // way to synchronize.
      TF_CHECK_OK(stream->BlockHostUntilDone());
      break;
  }
}

// Does all necessary bookkeeping, after a buffer is successfully enqueued onto
// a stream, to ensure that the buffer will be kept alive until its use on that
// stream is complete.
//
//   device_buffer:              the buffer that was enqueued.
//   buffer_local_device:        the device the buffer was allocated on.
//   stream_local_device:        the device that manages usage_stream.
//   event:                      an event that was recorded on usage_stream
//                               after the usage of device_buffer was enqueued.
//   usage_stream:               the stream the operation using device_buffer
//                               was enqueued on.
//   prefer_to_retain_reference: relevant only for the compute synchronous
//                               allocation model. If true, retain a reference
//                               to device_buffer until after the operation
//                               completes. If false then the compute stream
//                               will have to be synchronized past event before
//                               device_buffer can be freed.
//
// prefer_to_retain_reference encodes a heuristic set by the caller for the
// compute synchronous model:
//
// Generally when a buffer is the destination of a copy to a device, it will
// subsequently be used on the device's compute stream before being freed. In
// that case, there is no need to retain a reference to the buffer. If the
// buffer is freed before being used on the compute stream, the free will be
// delayed until the host knows that event has completed, but this is expected
// to be uncommon.
//
// When a buffer is the source of a copy from a device, we need to either retain
// a reference to the buffer until the copy completes or serialize the compute
// stream behind the copy. It is often better to retain a reference since while
// that keeps memory alive longer, it avoids stalling the compute stream.
void RecordUsage(PjRtStreamExecutorBuffer::ScopedHold device_buffer,
                 LocalDeviceState* buffer_local_device,
                 LocalDeviceState* stream_local_device,
                 std::shared_ptr<BufferSequencingEvent> event,
                 se::Stream* usage_stream, bool prefer_to_retain_reference,
                 std::vector<std::shared_ptr<TrackedDeviceBuffer>>*
                     buffers_to_release = nullptr) {
  tensorflow::profiler::TraceMe traceme("RecordUsage");
  bool retain_buffer_until_completion =
      // If the buffer wasn't allocated on the same device as the stream, always
      // retain a reference.
      (stream_local_device != buffer_local_device) ||
      // In the synchronous allocation model, always retain a reference.
      (stream_local_device->allocation_model() ==
       LocalDeviceState::kSynchronous) ||
      // In the compute synchronous model, use the caller's heuristic.
      (stream_local_device->allocation_model() ==
           LocalDeviceState::kComputeSynchronized &&
       prefer_to_retain_reference);
  if (retain_buffer_until_completion) {
    if (buffers_to_release) {
      buffers_to_release->push_back(device_buffer.buffer());
    } else {
      buffer_local_device->ThenRelease(usage_stream, device_buffer.buffer());
    }
  }
  device_buffer.ConvertUsageHold(usage_stream, event,
                                 retain_buffer_until_completion);
}

// Allocates the device buffers for a buffer that will be used as the
// destination of a copy, either from the host or another device. copy_stream
// may be nullptr, e.g., when allocating a buffer for a cross-host copy. If the
// buffer is a tuple then the tuple tables are allocated, and all necessary
// synchronization for them is dealt with, before the buffer is returned.
//
// It is safe to delete the returned PjRtBuffer without further
// synchronization if an error occurs before the buffer is used.
//
// The caller may optionally provide a definition event to be recorded in
// the buffer.
// TODO(phawkins): replace on_host_shape here with on_device_shape.
StatusOr<std::unique_ptr<PjRtStreamExecutorBuffer>> AllocateDestinationBuffer(
    const Shape& on_host_shape, PjRtDevice* device,
    LocalDeviceState* local_device, se::Stream* copy_stream,
    bool is_uninitialized_create, PjRtClient* client,
    std::shared_ptr<BufferSequencingEvent> definition_event = nullptr) {
  if (on_host_shape.IsTuple() && on_host_shape.tuple_shapes_size() == 0) {
    return InvalidArgument("Can't make a buffer from an empty tuple");
  }

  auto* se_client = tensorflow::down_cast<PjRtStreamExecutorClient*>(client);
  TransferManager* transfer_manager =
      se_client->client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer dst_buffer,
                      transfer_manager->AllocateScopedShapedBuffer(
                          on_host_shape, se_client->allocator(),
                          local_device->device_ordinal()));
  if (local_device->allocation_model() ==
      LocalDeviceState::kComputeSynchronized) {
    if (copy_stream == nullptr) {
      CHECK(is_uninitialized_create);
    } else {
      copy_stream->ThenWaitFor(local_device->compute_stream());
    }
  } else {
    DCHECK(transfer_manager->CanShapedBufferBeAccessedNow(
        local_device->compute_stream()->parent(), dst_buffer));
  }
  Shape on_device_shape = dst_buffer.on_device_shape();

  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 2>
      definition_events;
  if (is_uninitialized_create) {
    // There is not going to be any copy into the buffer so in general we don't
    // need a definition event.
    if (local_device->allocation_model() ==
        LocalDeviceState::kComputeSynchronized) {
      // The allocation is not valid until the compute stream passes this point,
      // so add a definition event in the compute stream.
      definition_events.emplace_back(std::make_shared<BufferSequencingEvent>());
      TF_ASSIGN_OR_RETURN(EventPool::Handle event,
                          local_device->event_pool().ThenAllocateAndRecordEvent(
                              local_device->compute_stream()));
      definition_events.back()->SetSequencingEvent(
          std::move(event), local_device->compute_stream());
    }
    // if the caller provided a definition event then we record that.
    if (definition_event) {
      definition_events.emplace_back(definition_event);
    }
  } else {
    // We have at least one definition event, for the copy completing to
    // the device buffers.
    if (definition_event) {
      definition_events.emplace_back(definition_event);
    } else {
      definition_events.emplace_back(std::make_shared<BufferSequencingEvent>());
    }
  }
  se::Stream* tuple_table_stream = local_device->host_to_device_stream();
  if (on_device_shape.IsTuple()) {
    // We also need to copy the tuple tables, so we'll have an additional
    // definition event for that copy to complete.
    if (tuple_table_stream != copy_stream) {
      if (local_device->allocation_model() ==
          LocalDeviceState::kComputeSynchronized) {
        tuple_table_stream->ThenWaitFor(local_device->compute_stream());
      } else {
        DCHECK(transfer_manager->CanShapedBufferBeAccessedNow(
            local_device->compute_stream()->parent(), dst_buffer));
      }
    }

    TF_RETURN_IF_ERROR(transfer_manager->WriteTupleIndexTablesAsync(
        tuple_table_stream, dst_buffer));
    // CAUTION: From this point onwards we need to be careful about returning
    // from error cases because we have started a transfer and must not allow
    // dst_buffer to be freed too soon in the non-async allocation models.

    definition_events.emplace_back(std::make_shared<BufferSequencingEvent>());
    StatusOr<EventPool::Handle> event_or =
        local_device->event_pool().ThenAllocateAndRecordEvent(
            tuple_table_stream);
    if (!event_or.ok()) {
      StallStreamOnError(local_device, tuple_table_stream);
      return event_or.status();
    }
    definition_events.back()->SetSequencingEvent(std::move(event_or).value(),
                                                 tuple_table_stream);
  }
  std::shared_ptr<TrackedDeviceBuffer> dst_device_buffer =
      TrackedDeviceBuffer::FromScopedShapedBuffer(&dst_buffer,
                                                  definition_events);

  auto py_buffer = std::make_unique<PjRtStreamExecutorBuffer>(
      on_device_shape, std::move(dst_device_buffer), client, device);

  if (on_device_shape.IsTuple()) {
    // Add a usage hold for the tuple table write and immediately convert it to
    // the appropriate form of synchronization. prefer_to_retain_reference=false
    // means don't retain a memory reference until the transfer is complete when
    // using the ComputeSynchronized allocation model. This is a heuristic
    // because in the common case destination buffers will be used on the
    // compute stream and therefore don't require any synchronization before
    // being freed. If the buffer is allocated and never used, the free will
    // take longer and this is assumed to be ok.
    RecordUsage(py_buffer->GetBufferWithUsageHold(), local_device, local_device,
                definition_events.back(), tuple_table_stream,
                /*prefer_to_retain_reference=*/false);
  }

  return py_buffer;
}

// Adds necessary synchronization after a copy has been enqueued to a buffer.
// definition_event was added when the buffer was allocated, but has not yet
// had an event recorded.
Status AddDestinationBufferSynchronization(
    LocalDeviceState* local_device,
    PjRtStreamExecutorBuffer::ScopedHold device_buffer,
    std::shared_ptr<BufferSequencingEvent> definition_event,
    se::Stream* copy_stream) {
  StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().ThenAllocateAndRecordEvent(copy_stream);
  if (!event_or.ok()) {
    StallStreamOnError(local_device, copy_stream);
    return event_or.status();
  }
  definition_event->SetSequencingEvent(std::move(event_or).value(),
                                       copy_stream);
  // prefer_to_retain_reference=false means don't retain a memory reference
  // until the transfer is complete when using the ComputeSynchronized
  // allocation model. This is a heuristic because in the common case
  // destination buffers will be used on the compute stream and therefore don't
  // require any synchronization before being freed. If the buffer is allocated
  // and never used, the free will take longer and this is assumed to be ok.
  RecordUsage(std::move(device_buffer), local_device, local_device,
              definition_event, copy_stream,
              /*prefer_to_retain_reference=*/false);
  return OkStatus();
}

}  // namespace

PjRtStreamExecutorBuffer::ScopedHold::~ScopedHold() {
  if (ok()) {
    parent_->DropHold(type_, buffer().get());
  }
}

PjRtStreamExecutorBuffer::ScopedHold::ScopedHold(ScopedHold&& other)
    : parent_(other.parent_),
      type_(other.type_),
      state_(other.state_),
      status_(std::move(other.status_)),
      buffer_(std::move(other.buffer_)) {
  // Preserve the invariant that status is invalid if buffer == nullptr.
  other.SetState(kMoved);
}

void PjRtStreamExecutorBuffer::ScopedHold::Acquire(
    StatusOr<std::shared_ptr<TrackedDeviceBuffer>>&& buffer_or) {
  CHECK(!ok());
  if (buffer_or.ok()) {
    buffer_ = buffer_or.ValueOrDie();
    SetState(kValid);
  } else {
    status_ = buffer_or.status();
    buffer_ = nullptr;
    SetState(kError);
  }
  // Check the invariant holds.
  CHECK(!ok() || buffer_ != nullptr);
}

PjRtStreamExecutorBuffer::ScopedHold::ForClosure
PjRtStreamExecutorBuffer::ScopedHold::ToClosure() {
  CHECK(ok());
  ForClosure for_closure(parent_, type_, state_, std::move(status_),
                         std::move(buffer_));
  SetState(kReleased);
  return for_closure;
}

void PjRtStreamExecutorBuffer::ScopedHold::ConvertUsageHold(
    se::Stream* usage_stream, std::shared_ptr<BufferSequencingEvent> event,
    bool reference_held) {
  CHECK(ok());
  CHECK_EQ(type_, kUsage);
  parent_->ConvertUsageHold(buffer().get(), usage_stream, std::move(event),
                            reference_held);
  SetState(kConverted);
}

void PjRtStreamExecutorBuffer::ScopedHold::ConfirmDonation() {
  CHECK(ok());
  CHECK_EQ(type_, kDonation);
  parent_->ConfirmDonation(buffer().get());
  SetState(kDonated);
}

void PjRtStreamExecutorBuffer::ScopedHold::AddToInput(
    ShapeTree<MaybeOwningDeviceMemory>::iterator* iterator,
    const ShapeTree<MaybeOwningDeviceMemory>::iterator& end,
    ExecutionInput* execution_input,
    se::DeviceMemoryAllocator* allocator) const {
  CHECK(ok());
  if (type_ == kDonation) {
    buffer()->AddToInputAsDonated(iterator, end, execution_input, allocator);
  } else {
    CHECK_EQ(type_, kUsage);
    buffer()->AddToInputAsImmutable(iterator, end);
  }
}

bool PjRtStreamExecutorBuffer::IsOnCpu() const {
  return client()->platform_id() == CpuId();
}

StatusOr<Shape> PjRtStreamExecutorBuffer::logical_on_device_shape() {
  if (on_device_shape_.is_static()) {
    return on_device_shape_;
  }
  auto* local_device = device_->local_device_state();
  auto* stream = local_device->GetDeviceToHostStream();
  ScopedHold device_buffer(this, ScopedHold::kUsage);
  {
    absl::MutexLock lock(&mu_);
    // We can't perform any other action while a donation hold is in progress.
    WaitForOutstandingDonationHold();
    if (device_buffer_ == nullptr) {
      return InvalidArgument(
          "logical_on_device_shape() called on deleted or donated buffer");
    }
    AcquireHoldLocked(&device_buffer);
  }

  WaitForBufferDefinitionEventsOnStream(*device_buffer, stream);
  ShapedBuffer shaped_buffer = device_buffer->AsShapedBuffer(on_device_shape_);
  StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().AllocateEvent(stream->parent());
  if (!event_or.ok()) {
    return event_or.status();
  }
  Shape ret_shape = on_device_shape_;
  TransferManager* transfer_manager =
      client_->client()->backend().transfer_manager();
  TF_RETURN_IF_ERROR(
      transfer_manager->ReadDynamicShapes(stream, &shaped_buffer, &ret_shape));
  return ret_shape;
}

namespace {

// Implements PjRtBuffer::ExternalReference as a wrapped
// ScopedHold::kExternalReference.
class ScopedHoldAsExternalReference : public PjRtBuffer::ExternalReference {
 public:
  explicit ScopedHoldAsExternalReference(
      PjRtStreamExecutorBuffer::ScopedHold hold)
      : external_reference_(std::move(hold)) {
    CHECK(external_reference_.type() ==
          PjRtStreamExecutorBuffer::ScopedHold::kExternalReference);
    data_ptr_ = external_reference_->device_memory().front().opaque();
  }

  ~ScopedHoldAsExternalReference() override = default;

 private:
  PjRtStreamExecutorBuffer::ScopedHold external_reference_;
};

}  // namespace

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
PjRtStreamExecutorBuffer::AcquireExternalReference() {
  ScopedHold hold = GetBufferWithExternalReference();
  Status hold_status = hold.status();
  if (!hold_status.ok()) return hold_status;
  return std::unique_ptr<ExternalReference>(
      std::make_unique<ScopedHoldAsExternalReference>(std::move(hold)));
}

class TrackedDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedDeviceBufferExternalReference(
      std::shared_ptr<TrackedDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    data_ptr_ = tracked_device_buffer_->device_memory()[0].opaque();
  }

  ~TrackedDeviceBufferExternalReference() override = default;

 private:
  std::shared_ptr<TrackedDeviceBuffer> tracked_device_buffer_;
};

StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
PjRtStreamExecutorBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<TrackedDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    std::function<void()> on_done_with_host_buffer, PjRtDevice* device) {
  tensorflow::profiler::TraceMe traceme(
      "PjRtStreamExecutorClient::BufferFromHostBuffer");
  Shape shape = ShapeUtil::MakeShape(type, dims);
  VLOG(1) << "PjRtStreamExecutorClient::BufferFromHostBuffer: shape: "
          << shape.ToString() << " device: " << device->DebugString();
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());

  absl::InlinedVector<int64_t, 4> tmp_strides;
  if (!byte_strides) {
    tmp_strides.resize(dims.size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(shape, absl::MakeSpan(tmp_strides)));
    byte_strides = tmp_strides;
  }
  int64_t size = ShapeUtil::ByteSizeOf(shape);

  TransferManager* transfer_manager = client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(Shape compact_shape,
                      transfer_manager->ChooseCompactLayoutForShape(shape));
  absl::InlinedVector<int64_t, 4> compact_shape_strides(
      compact_shape.dimensions_size());
  TF_RETURN_IF_ERROR(ShapeUtil::ByteStrides(
      compact_shape, absl::MakeSpan(compact_shape_strides)));
  bool host_and_device_strides_equal =
      (size == 0 || *byte_strides == compact_shape_strides);
  // The CPU platform is special because the "host" and the "device" are in the
  // same memory space. If the input shape is in the correct layout and we don't
  // want to defer the copy onto a thread, we can use the following fast
  // path.
  bool is_cpu_platform =
      local_device->executor()->platform()->id() == se::host::kHostPlatformId;
  if (is_cpu_platform) {
    // If we are on the host platform and the input buffer is sufficiently
    // aligned, we can simply point to the input array's data without any
    // further copies. At the time of writing we require a 16-byte alignment
    // because XLA may generate code which requires it.
    bool can_use_zero_copy =
        host_buffer_semantics == HostBufferSemantics::kZeroCopy &&
        ((absl::bit_cast<std::uintptr_t>(data) &
          (cpu_function_runtime::MinAlign() - 1)) == 0);
    if (host_and_device_strides_equal &&
        (host_buffer_semantics ==
             HostBufferSemantics::kImmutableOnlyDuringCall ||
         can_use_zero_copy)) {
      std::function<void()> on_delete_callback;
      se::DeviceMemoryBase buffer;
      // If we are on the host platform and the input buffer is sufficiently
      // aligned, we can simply point to the input array's data without any
      // further copies. At the time of writing we require a 16-byte alignment
      // because XLA may generate code which requires it.
      if (can_use_zero_copy) {
        on_delete_callback = std::move(on_done_with_host_buffer);
        buffer = se::DeviceMemoryBase(
            const_cast<void*>(static_cast<const void*>(data)), size);
      } else {
        void* staging_buffer = host_memory_allocator()->AllocateRaw(
            cpu_function_runtime::MinAlign(), size);
        buffer = se::DeviceMemoryBase(staging_buffer, size);
        std::memcpy(staging_buffer, data, size);
        if (on_done_with_host_buffer) {
          on_done_with_host_buffer();
        }
        on_delete_callback = [staging_buffer, host_memory_allocator =
                                                  host_memory_allocator()]() {
          host_memory_allocator->DeallocateRaw(staging_buffer);
        };
      }
      absl::Span<const std::shared_ptr<BufferSequencingEvent>>
          definition_events;
      auto device_buffer = std::make_shared<TrackedDeviceBuffer>(
          /*allocator=*/nullptr, local_device->device_ordinal(),
          std::initializer_list<se::DeviceMemoryBase>{buffer},
          definition_events, std::move(on_delete_callback));
      return std::unique_ptr<PjRtBuffer>(
          std::make_unique<PjRtStreamExecutorBuffer>(
              shape, std::move(device_buffer), this, device));
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtStreamExecutorBuffer> py_buffer,
      AllocateDestinationBuffer(compact_shape, device, local_device,
                                local_device->host_to_device_stream(),
                                /*is_uninitialized_create=*/false, this));

  PjRtStreamExecutorBuffer::ScopedHold device_buffer(
      py_buffer->GetBufferWithUsageHold());
  CHECK(device_buffer.ok());

  // If necessary, allocate a host-side buffer for staging host-to-device
  // transfers. On GPU this is a buffer in pinned memory.
  std::shared_ptr<void> staging_buffer;
  if (host_buffer_semantics == HostBufferSemantics::kImmutableOnlyDuringCall ||
      should_stage_host_to_device_transfers() ||
      !host_and_device_strides_equal) {
    void* ptr = host_memory_allocator()->AllocateRaw(
        tensorflow::Allocator::kAllocatorAlignment, size);
    staging_buffer = std::shared_ptr<void>(
        ptr, [host_memory_allocator = host_memory_allocator()](void* ptr) {
          host_memory_allocator->DeallocateRaw(ptr);
        });
  }

  std::shared_ptr<TransposePlan> transpose;
  if (!host_and_device_strides_equal) {
    absl::InlinedVector<int64_t, 4> permutation(dims.size());
    absl::c_reverse_copy(compact_shape.layout().minor_to_major(),
                         permutation.begin());
    absl::MutexLock lock(&transpose_mu_);
    TF_ASSIGN_OR_RETURN(transpose,
                        transpose_cache_.GetOrCreate(
                            primitive_util::ByteWidth(type), dims, permutation,
                            TransposePlan::Striding{*byte_strides}));
  }

  // Copy the buffer into a staging buffer before returning control to the
  // caller if the caller only guaranteed that the buffer is valid for the
  // duration of the call. Otherwise, we stage (if necessary) on a separate
  // thread.
  if (host_buffer_semantics == HostBufferSemantics::kImmutableOnlyDuringCall) {
    if (transpose) {
      transpose->Execute(data, staging_buffer.get());
    } else {
      std::memcpy(staging_buffer.get(), data, size);
    }
    if (on_done_with_host_buffer) {
      on_done_with_host_buffer();
      on_done_with_host_buffer = nullptr;
    }
  }

  // The host to device transfer is performed on a thread pool, mostly because
  // it includes linearization that may be slow. It is OK to capture the
  // py_buffer pointer because the py_buffer can't be deleted until all the
  // usage holds have gone away.
  // TODO(misard) assess if it would be preferable to introduce a heuristic to
  // put the transfer into the calling thread for small literals.
  auto transfer_h2d =
      [local_client = client(), transfer_manager, local_device, data, size,
       movable_device_buffer{device_buffer.ToClosure()}, shape,
       py_buffer{py_buffer.get()},
       on_device_shape{py_buffer->on_device_shape()},
       staging_buffer{std::move(staging_buffer)},
       on_done_with_host_buffer{std::move(on_done_with_host_buffer)},
       host_buffer_semantics, transpose{std::move(transpose)}]() {
        PjRtStreamExecutorBuffer::ScopedHold device_buffer(
            movable_device_buffer);
        // This function uses TF_CHECK_OK and ValueOrDie() since we have no way
        // to report failures from a callback. However, the operations here are
        // unlikely to fail and not recoverable even if we were to fail: DMAs to
        // memory that has already been allocated, and a possible Event
        // allocation.

        ShapedBuffer buffer = device_buffer->AsShapedBuffer(on_device_shape);
        // If applicable on the backend, stage the transfer via host memory
        // allocated via the host_memory_allocator. On GPU, this is pinned
        // memory.
        if (staging_buffer) {
          // If we didn't already copy the input buffer into the staging buffer,
          // do so now.
          if (host_buffer_semantics !=
              HostBufferSemantics::kImmutableOnlyDuringCall) {
            if (transpose) {
              transpose->Execute(data, staging_buffer.get());
            } else {
              std::memcpy(staging_buffer.get(), data, size);
            }
          }
          // The buffer has the same dimension order as the on-device shape, but
          // is not tiled, etc.
          BorrowingLiteral literal(
              static_cast<const char*>(staging_buffer.get()),
              ShapeUtil::DeviceShapeToHostShape(on_device_shape));
          TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
              local_device->host_to_device_stream(), literal, buffer));
        } else {
          BorrowingLiteral literal(
              reinterpret_cast<const char*>(data),
              ShapeUtil::DeviceShapeToHostShape(on_device_shape));
          // Otherwise, just transfer the literal.
          TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
              local_device->host_to_device_stream(), literal, buffer));
        }

        std::shared_ptr<BufferSequencingEvent> event =
            device_buffer->definition_events()[0];
        TF_CHECK_OK(AddDestinationBufferSynchronization(
            local_device, std::move(device_buffer), event,
            local_device->host_to_device_stream()));

        local_device->ThenExecuteCallback(
            local_device->host_to_device_stream(),
            [staging_buffer{std::move(staging_buffer)},
             on_done_with_host_buffer{std::move(on_done_with_host_buffer)}]() {
              if (on_done_with_host_buffer) {
                on_done_with_host_buffer();
              }
            });
      };
  if (is_cpu_platform) {
    // Using the thread_pool would be a double thread hop; the code
    // already defers its work onto a stream (= thread on CPU).
    transfer_h2d();
  } else {
    thread_pool()->Schedule(transfer_h2d);
  }
  return std::unique_ptr<PjRtBuffer>(std::move(py_buffer));
}

StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateUninitializedBuffer(const Shape& shape,
                                                    PjRtDevice* device) {
  return CreateUninitializedBuffer(shape, device, nullptr);
}

StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateUninitializedBuffer(
    const Shape& shape, PjRtDevice* device,
    std::shared_ptr<BufferSequencingEvent> definition_event) {
  tensorflow::profiler::TraceMe traceme(
      "PjRtStreamExecutorClient::CreateUninitializedBuffer");
  VLOG(1) << "PjRtStreamExecutorClient::CreateUninitializedBuffer: shape: "
          << shape.ToString() << " device: " << device->DebugString();
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());

  TransferManager* transfer_manager = client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(Shape compact_shape,
                      transfer_manager->ChooseCompactLayoutForShape(shape));

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtStreamExecutorBuffer> py_buffer,
      AllocateDestinationBuffer(compact_shape, device, local_device,
                                /*copy_stream=*/nullptr,
                                /*is_uninitialized_create=*/true, this,
                                definition_event));
  return std::unique_ptr<PjRtBuffer>(std::move(py_buffer));
}

StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                                PjRtDevice* device) {
  tensorflow::profiler::TraceMe traceme(
      "PjRtStreamExecutorClient::BufferFromHostLiteral");
  VLOG(1) << "PjRtStreamExecutorClient::BufferFromHostLiteral: shape: "
          << literal.shape().ToString() << " device: " << device->DebugString();
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());

  TransferManager* transfer_manager = client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(
      Shape compact_shape,
      transfer_manager->ChooseCompactLayoutForShape(literal.shape()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtStreamExecutorBuffer> py_buffer,
      AllocateDestinationBuffer(compact_shape, device, local_device,
                                local_device->host_to_device_stream(),
                                /*is_uninitialized_create=*/false, this));

  PjRtStreamExecutorBuffer::ScopedHold device_buffer(
      py_buffer->GetBufferWithUsageHold());
  CHECK(device_buffer.ok());

  // The host to device transfer is performed on a thread pool, mostly because
  // it includes linearization that may be slow. It is OK to capture the
  // py_buffer pointer because the py_buffer can't be deleted until all the
  // usage holds have gone away.
  // TODO(misard) assess if it would be preferable to introduce a heuristic to
  // put the transfer into the calling thread for small literals.
  auto transfer_h2d = [local_client = client(), transfer_manager, local_device,
                       movable_device_buffer{device_buffer.ToClosure()},
                       literal, py_buffer{py_buffer.get()},
                       on_device_shape{py_buffer->on_device_shape()}]() {
    PjRtStreamExecutorBuffer::ScopedHold device_buffer(movable_device_buffer);
    // This function uses TF_CHECK_OK and ValueOrDie() since we have no way
    // to report failures from a callback. However, the operations here are
    // unlikely to fail and not recoverable even if we were to fail: DMAs to
    // memory that has already been allocated, and a possible Event
    // allocation.

    se::Stream* h2d_stream = local_device->host_to_device_stream();
    ShapedBuffer buffer = device_buffer->AsShapedBuffer(on_device_shape);
    TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
        h2d_stream, literal, buffer));

    std::shared_ptr<BufferSequencingEvent> event =
        device_buffer->definition_events()[0];
    TF_CHECK_OK(AddDestinationBufferSynchronization(
        local_device, std::move(device_buffer), event, h2d_stream));

    // This can sometimes catch the case where the literal memory has been
    // freed before the H2D transfer was issued.
    h2d_stream->RefreshStatus()
        .IgnoreError();  // Can return error::Unimplemented
    QCHECK(h2d_stream->ok());
  };
  thread_pool()->Schedule(transfer_h2d);
  return std::unique_ptr<PjRtBuffer>(std::move(py_buffer));
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtStreamExecutorClient::MakeCrossHostReceiveBuffers(
    absl::Span<const Shape> shapes, PjRtDevice* device,
    PjRtCrossHostRecvNotifier notifier) {
  if (shapes.empty()) {
    return InvalidArgument(
        "shapes parameter empty in MakeCrossHostReceiveBuffers");
  }

  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());
  std::shared_ptr<BufferSequencingEvent> definition_event =
      std::make_shared<BufferSequencingEvent>();
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.reserve(shapes.size());
  for (const auto& shape : shapes) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer> buffer,
        AllocateDestinationBuffer(shape, device, local_device,
                                  /*copy_stream=*/nullptr,
                                  /*is_uninitialized_create=*/false, this,
                                  definition_event));
    buffers.push_back(std::move(buffer));
  }

  TF_RETURN_IF_ERROR(EnqueueCrossHostReceive(
      buffers, std::move(definition_event), std::move(notifier), std::nullopt));
  return buffers;
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtStreamExecutorClient::MakeCrossHostReceiveBuffersForGather(
    absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
    PjRtDevice* device, PjRtCrossHostRecvNotifier notifier) {
  VLOG(2) << "Making " << gather_details.size()
          << " cross host receive buffers for gather";
  if (gather_details.empty()) {
    return InvalidArgument(
        "gather_details parameter empty in "
        "MakeCrossHostReceiveBuffersForGather");
  }

  if (shapes.size() != gather_details.size()) {
    return InvalidArgument(
        "gather_details parameter has length %lld but shapes "
        "parameter has length %lld in "
        "MakeCrossHostReceiveBuffersForGather",
        gather_details.size(), shapes.size());
  }

  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());
  std::shared_ptr<BufferSequencingEvent> definition_event =
      std::make_shared<BufferSequencingEvent>();
  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.reserve(shapes.size());
  for (int i = 0; i < shapes.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer> buffer,
        AllocateDestinationBuffer(shapes[i], device, local_device,
                                  /*copy_stream=*/nullptr,
                                  /*is_uninitialized_create=*/false, this,
                                  definition_event));
    buffers.push_back(std::move(buffer));
  }

  TF_RETURN_IF_ERROR(
      EnqueueCrossHostReceive(buffers, std::move(definition_event),
                              std::move(notifier), gather_details));
  return buffers;
}

StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtDevice* device,
    std::function<void()> on_delete_callback) {
  se::DeviceMemoryBase buffer(device_ptr, ShapeUtil::ByteSizeOf(shape));
  absl::Span<const std::shared_ptr<BufferSequencingEvent>> definition_events;
  auto device_buffer = std::make_shared<TrackedDeviceBuffer>(
      /*allocator=*/nullptr, device->local_hardware_id(),
      std::initializer_list<se::DeviceMemoryBase>{buffer}, definition_events,
      std::move(on_delete_callback));
  return std::unique_ptr<PjRtBuffer>(std::make_unique<PjRtStreamExecutorBuffer>(
      shape, std::move(device_buffer), this, device));
}

// Transfer the given literal to the infeed queue of the given local device.
Status PjRtStreamExecutorDevice::TransferToInfeed(const LiteralSlice& literal) {
  // Only support infeed to local device.
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device, GetLocalDeviceState());
  return local_device->client()->TransferToInfeedLocal(
      literal, local_device->device_ordinal());
}

Status PjRtStreamExecutorDevice::TransferFromOutfeed(
    MutableBorrowingLiteral literal) {
  VLOG(1) << "PjRtStreamExecutorDevice::TransferFromOutfeed";
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device, GetLocalDeviceState());
  return local_device->client()->TransferFromOutfeedLocal(
      local_device->device_ordinal(), literal);
}

StatusOr<PjRtDevice*> PjRtStreamExecutorClient::LookupAddressableDevice(
    int local_hardware_id) const {
  for (auto* device : addressable_devices_) {
    if (local_hardware_id == device->local_hardware_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_hardware_id %d",
                         local_hardware_id);
}

PjRtStreamExecutorBuffer::PjRtStreamExecutorBuffer(
    Shape on_device_shape, std::shared_ptr<TrackedDeviceBuffer> device_buffer,
    PjRtClient* client, PjRtDevice* device)
    : client_(tensorflow::down_cast<PjRtStreamExecutorClient*>(client)),
      on_device_shape_(std::move(on_device_shape)),
      device_(tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)),
      device_buffer_(std::move(device_buffer)) {
  for (int i = 0; i < ScopedHold::Type::kMaxValue; ++i) {
    holds_[i] = 0;
  }
}

PjRtStreamExecutorBuffer::~PjRtStreamExecutorBuffer() {
  Delete();
  for (int i = 0; i < ScopedHold::Type::kMaxValue; ++i) {
    CHECK_EQ(holds_[i], 0);
  }
}

void PjRtStreamExecutorBuffer::WaitForOutstandingUsageHolds() {
  auto not_in_usage_hold = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return holds_[ScopedHold::kUsage] == 0;
  };
  mu_.Await(absl::Condition(&not_in_usage_hold));
}

void PjRtStreamExecutorBuffer::WaitForOutstandingDonationHold() {
  auto not_in_donation_hold = [&]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return holds_[ScopedHold::kDonation] == 0;
  };
  mu_.Await(absl::Condition(&not_in_donation_hold));
}

StatusOr<std::shared_ptr<TrackedDeviceBuffer>>
PjRtStreamExecutorBuffer::Release(bool wait_for_operations_to_complete) {
  tensorflow::profiler::TraceMe trace_me("PjRtStreamExecutorBuffer::Release");
  std::shared_ptr<TrackedDeviceBuffer> device_buffer;
  TrackedDeviceBuffer::StreamAndEventContainer events;
  {
    absl::MutexLock lock(&mu_);
    // We first wait for a donation hold to complete if there is one in
    // progress. If the donation succeeds via ConfirmDonation() then it will
    // set device_buffer_ to nullptr before returning to this thread.
    WaitForOutstandingDonationHold();
    if (device_buffer_ == nullptr) {
      return std::shared_ptr<TrackedDeviceBuffer>();
    }
    // Set device_buffer_ to null now so that no other
    // thread can add a hold while we are in WaitForOutstandingUsageHolds()
    // below.
    std::swap(device_buffer_, device_buffer);
    WaitForOutstandingUsageHolds();
    // Now that all holds have completed and no more can be added, we can get
    // the final set of usage events.
    events = device_buffer->LockUseAndTransferUsageEvents();
  }
  LocalDeviceState* local_device_state = device_->local_device_state();
  if (wait_for_operations_to_complete) {
    // Block the host until all usage events have completed. Usage events
    // dominate definition events, so this also waits for the buffer to be
    // defined.
    std::unique_ptr<se::Stream> stream;
    for (const auto& stream_and_event : events) {
      if (!stream_and_event.event->IsComplete()) {
        if (stream == nullptr) {
          stream = local_device_state->BorrowStreamFromPool();
        }
        stream_and_event.event->WaitForEventOnStream(stream.get());
      }
    }
    if (stream != nullptr) {
      TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
      local_device_state->ReturnStreamToPool(std::move(stream));
    }
  } else {
    if (local_device_state->allocation_model() ==
        LocalDeviceState::kComputeSynchronized) {
      std::unique_ptr<se::Stream> block_stream;
      for (const auto& stream_and_event : events) {
        // We only need to do something for events that didn't already acquire a
        // reference to the buffer, and also which the compute stream didn't
        // already wait for. Based on our heuristics this rare case should only
        // occur when a buffer was copied to a device and then never used there.
        // In that case we get a new stream and use it to hold onto a reference
        // to the buffer until the events are complete.
        if (!stream_and_event.reference_held &&
            !stream_and_event.event->DefinedOn(
                local_device_state->compute_stream()) &&
            !stream_and_event.event->IsComplete()) {
          if (block_stream == nullptr) {
            block_stream = local_device_state->BorrowStreamFromPool();
          }
          stream_and_event.event->WaitForEventOnStream(block_stream.get());
        }
      }
      if (block_stream != nullptr) {
        se::Stream* block_stream_ptr = block_stream.release();
        local_device_state->ThenExecuteCallback(
            block_stream_ptr,
            [device_buffer, block_stream_ptr, local_device_state]() {
              local_device_state->ReturnStreamToPool(
                  std::unique_ptr<se::Stream>(block_stream_ptr));
            });
      }
    }
  }
  return device_buffer;
}

void PjRtStreamExecutorBuffer::Delete() {
  VLOG(1) << "PjRtStreamExecutorBuffer::Delete";
  // When wait_for_reads_to_complete is false, Release should never fail.
  TF_CHECK_OK(Release(/*wait_for_operations_to_complete=*/false).status());
}

bool PjRtStreamExecutorBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return device_buffer_ == nullptr;
}

StatusOr<std::shared_ptr<TrackedDeviceBuffer>>
PjRtStreamExecutorBuffer::GetBufferForHoldLocked(ScopedHold::Type type) {
  // All callers should have called WaitForOutstandingDonationHold().
  CHECK_EQ(holds_[ScopedHold::kDonation], 0);
  if (type == ScopedHold::kDonation) {
    if (device_buffer_ == nullptr) {
      return InvalidArgument("Donation requested for invalid buffer");
    }
    if (holds_[ScopedHold::kExternalReference] > 0) {
      return InvalidArgument(
          "Donation requested for buffer with external reference");
    }
    // First add the donation hold.
    ++holds_[type];
    // Then wait for any usage holds to be dropped or converted. No new usage
    // holds can be added until we drop the donation hold so this wait will
    // complete eventually.
    WaitForOutstandingUsageHolds();
    // Because we added a donation hold, nobody could release the buffer while
    // we were waiting.
    CHECK(device_buffer_ != nullptr);
  } else {
    if (device_buffer_ == nullptr) {
      return InvalidArgument("Buffer has been deleted or donated.");
    } else {
      ++holds_[type];
    }
  }
  return device_buffer_;
}

void PjRtStreamExecutorBuffer::AcquireHoldLocked(ScopedHold* hold) {
  hold->Acquire(GetBufferForHoldLocked(hold->type()));
}

void PjRtStreamExecutorBuffer::ConvertUsageHold(
    TrackedDeviceBuffer* buffer, se::Stream* usage_stream,
    std::shared_ptr<BufferSequencingEvent> event, bool reference_held) {
  absl::MutexLock lock(&mu_);
  CHECK(device_buffer_.get() == buffer || device_buffer_ == nullptr);
  buffer->AddUsageEvent(usage_stream, std::move(event), reference_held);
  CHECK_GT(holds_[ScopedHold::kUsage], 0);
  --holds_[ScopedHold::kUsage];
}

void PjRtStreamExecutorBuffer::ConfirmDonation(
    TrackedDeviceBuffer* device_buffer) {
  {
    absl::MutexLock lock(&mu_);
    CHECK_EQ(holds_[ScopedHold::kUsage], 0);
    CHECK_EQ(holds_[ScopedHold::kExternalReference], 0);
    CHECK_EQ(holds_[ScopedHold::kDonation], 1);
    holds_[ScopedHold::kDonation] = 0;
    CHECK(device_buffer_.get() == device_buffer);
    // As a sanity check ensure no more usage events can be added to the buffer.
    device_buffer->LockUseAndTransferUsageEvents();
    // Give up ownership of the device memory so we don't free it when the last
    // reference to device_buffer_ goes away.
    device_buffer->ReleaseDeviceMemory();
    // Make *this invalid so it can't be used again. Any threads blocking in
    // Release or GetBufferWithHold will see an invalid buffer and return.
    device_buffer_.reset();
  }
}

void PjRtStreamExecutorBuffer::DropHold(ScopedHold::Type type,
                                        TrackedDeviceBuffer* buffer) {
  absl::MutexLock lock(&mu_);
  CHECK(device_buffer_.get() == buffer || device_buffer_ == nullptr);
  CHECK_GT(holds_[type], 0);
  --holds_[type];
  if (type == ScopedHold::kDonation) {
    CHECK_EQ(holds_[ScopedHold::kDonation], 0);
    CHECK_EQ(holds_[ScopedHold::kUsage], 0);
    CHECK_EQ(holds_[ScopedHold::kExternalReference], 0);
  }
}

PjRtFuture<Status> PjRtStreamExecutorBuffer::ToLiteral(
    MutableLiteralBase* literal) {
  VLOG(1) << "PjRtStreamExecutorBuffer::ToLiteral";
  if (IsEmptyTuple()) {
    return PjRtFuture<Status>(
        InvalidArgument("ToLiteral called on empty tuple"));
  }
  LocalDeviceState* local_device = device_->local_device_state();
  se::Stream* stream = local_device->GetDeviceToHostStream();
  ScopedHold device_buffer(this, ScopedHold::kUsage);
  {
    absl::MutexLock lock(&mu_);
    // We can't perform any other action while a donation hold is in progress.
    WaitForOutstandingDonationHold();
    if (device_buffer_ == nullptr) {
      return PjRtFuture<Status>(InvalidArgument(
          "CopyToHostAsync() called on deleted or donated buffer"));
    }
    AcquireHoldLocked(&device_buffer);
  }

  WaitForBufferDefinitionEventsOnStream(*device_buffer, stream);
  ShapedBuffer shaped_buffer = device_buffer->AsShapedBuffer(on_device_shape_);
  StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().AllocateEvent(stream->parent());
  if (!event_or.ok()) {
    return PjRtFuture<Status>(event_or.status());
  }
  auto promise = PjRtFuture<Status>::CreatePromise();
  client_->client()->backend().transfer_manager()->TransferLiteralFromDevice(
      stream, shaped_buffer, literal,
      [promise](Status status) mutable { promise.Set(status); });

  auto usage_event = std::make_shared<BufferSequencingEvent>();
  local_device->event_pool().ThenRecordEvent(stream, event_or.ValueOrDie());
  usage_event->SetSequencingEvent(std::move(event_or).value(), stream);
  // When using the ComputeSynchronized allocation model, retain a reference to
  // the device_buffer until the copy completes, to ensure that the buffer isn't
  // deleted or donated while it is still in use. The choice of retaining a
  // reference at the host is a heuristic; the alternative is to ensure, before
  // freeing the buffer, that the compute stream is synchronized past the
  // transfer, but it seems better to hold onto the buffer too long than to
  // stall the compute stream, particularly since the overwhelmingly common
  // use case of CopyToHostAsync will hold onto the reference long enough to
  // read the buffer in a subsequent call to ToLiteral.
  RecordUsage(std::move(device_buffer), local_device, local_device, usage_event,
              stream,
              /*prefer_to_retain_reference=*/true);

  return PjRtFuture<Status>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tensorflow::profiler::TraceMeProducer traceme(
            "PjRtStreamExecutorBuffer::ToLiteral");
        VLOG(1) << "PjRtStreamExecutorBuffer::ToLiteral";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tensorflow::profiler::TraceMeConsumer traceme(
            "PjRtStreamExecutorBuffer::ToLiteral", keys.traceme_context_id);
      });
}

StatusOr<size_t> PjRtStreamExecutorBuffer::GetOnDeviceSizeInBytes() const {
  absl::MutexLock lock(&mu_);
  if (device_buffer_ == nullptr) {
    return InvalidArgument(
        "GetOnDeviceSizeInBytes called on deleted or donated buffer");
  }
  if (device_buffer_->device_memory().size() != 1) {
    return InvalidArgument(
        "GetOnDeviceSizeInBytes called on tuple-shaped buffer");
  }
  return device_buffer_->device_memory()[0].size();
}

PjRtFuture<Status> PjRtStreamExecutorBuffer::CopyRawToHost(
    void* dst, int64_t offset, int64_t transfer_size) {
  return client_->CopyRawSubBufferToHost(this, dst, offset, transfer_size);
}

StatusOr<ShapedBuffer> PjRtStreamExecutorBuffer::AsShapedBuffer() const {
  absl::MutexLock lock(&mu_);
  if (device_buffer_ == nullptr) {
    return InvalidArgument(
        "Attempted to fetch value of invalid/deleted buffer.");
  }
  return device_buffer_->AsShapedBuffer(on_device_shape_);
}

PjRtStreamExecutorBuffer::ScopedHold
PjRtStreamExecutorBuffer::GetBufferWithHold(ScopedHold::Type type) {
  absl::MutexLock lock(&mu_);
  // Ensure that at most one donation hold can be in progress at a time.
  WaitForOutstandingDonationHold();
  ScopedHold hold(this, type);
  AcquireHoldLocked(&hold);
  return hold;
}

StatusOr<std::pair<std::unique_ptr<PjRtBuffer>,
                   std::shared_ptr<BufferSequencingEvent>>>
PjRtStreamExecutorBuffer::CopyToDeviceHelper(
    PjRtDevice* dst_device, LocalDeviceState* dst_local_device,
    LocalDeviceState* transfer_local_device, se::Stream* transfer_stream,
    std::shared_ptr<TrackedDeviceBuffer> src_device_buffer) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtStreamExecutorBuffer> py_buffer,
                      AllocateDestinationBuffer(
                          ShapeUtil::DeviceShapeToHostShape(on_device_shape_),
                          dst_device, dst_local_device, transfer_stream,
                          /*is_uninitialized_create=*/false, client_));

  TF_ASSIGN_OR_RETURN(ShapedBuffer src_buffer, AsShapedBuffer());

  WaitForBufferDefinitionEventsOnStream(*src_device_buffer, transfer_stream);

  ScopedHold dst_device_buffer(py_buffer->GetBufferWithUsageHold());
  CHECK(dst_device_buffer.ok());
  ShapedBuffer dst_buffer = dst_device_buffer->AsShapedBuffer(on_device_shape_);

  // Copy the leaf buffers.
  StatusOr<std::shared_ptr<BufferSequencingEvent>> copy_event_or =
      [&]() -> StatusOr<std::shared_ptr<BufferSequencingEvent>> {
    for (const auto& leaf : src_buffer.buffers().leaves()) {
      const ShapeIndex& index = leaf.first;
      const se::DeviceMemoryBase& input_buffer = leaf.second;
      const se::DeviceMemoryBase& output_buffer = dst_buffer.buffer(index);
      TF_RET_CHECK(input_buffer.size() == output_buffer.size())
          << "input: " << input_buffer.size()
          << " output: " << output_buffer.size();
      if (input_buffer.size() != 0) {
        TF_RETURN_IF_ERROR(transfer_local_device->ThenMemcpyDeviceToDevice(
            transfer_stream, dst_local_device->compute_stream(), input_buffer,
            output_buffer));
      }
    }
    std::shared_ptr<BufferSequencingEvent> event =
        dst_device_buffer->definition_events()[0];
    TF_RETURN_IF_ERROR(AddDestinationBufferSynchronization(
        transfer_local_device, std::move(dst_device_buffer), event,
        transfer_stream));
    return event;
  }();
  if (!copy_event_or.ok()) {
    StallStreamOnError(transfer_local_device, transfer_stream);
    if (transfer_local_device == dst_local_device) {
      // Some copies may have been enqueued before the error was returned, and
      // StallStreamOnError only makes sure the destination device is ok, so
      // make sure that the src buffer remains valid until after any transfers
      // have completed.
      device_->local_device_state()->ThenRelease(transfer_stream,
                                                 std::move(src_device_buffer));
    }
    return copy_event_or.status();
  }

  return std::pair<std::unique_ptr<PjRtBuffer>,
                   std::shared_ptr<BufferSequencingEvent>>(
      std::unique_ptr<PjRtStreamExecutorBuffer>(std::move(py_buffer)),
      std::move(copy_event_or).value());
}

StatusOr<std::unique_ptr<PjRtBuffer>> PjRtStreamExecutorBuffer::CopyToDevice(
    PjRtDevice* dst_device) {
  tensorflow::profiler::TraceMe traceme(
      "PjRtStreamExecutorBuffer::CopyToDevice");
  VLOG(1) << "PjRtStreamExecutorBuffer::CopyToDevice";
  if (dst_device == device_) {
    return InvalidArgument(
        "CopyToDevice cannot accept the same source and destination devices");
  }

  // Copying across PjRtClients involves a copy through the host.
  if (dst_device->client() != client_) {
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
    // Avoid use-after-free on `literal` due to unsequenced move and use.
    Literal* literal_pointer = literal.get();
    absl::InlinedVector<int64_t, 4> byte_strides(
        literal->shape().dimensions_size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
    return dst_device->client()->BufferFromHostBuffer(
        literal_pointer->untyped_data(),
        literal_pointer->shape().element_type(),
        literal_pointer->shape().dimensions(), byte_strides,
        PjRtStreamExecutorClient::HostBufferSemantics::kZeroCopy,
        [literal{std::move(literal)}]() { /* frees literal */ }, dst_device);
  }

  TF_ASSIGN_OR_RETURN(
      LocalDeviceState * dst_local_device,
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(dst_device)
          ->GetLocalDeviceState());
  LocalDeviceState* transfer_local_device =
      client_->EnqueueD2DTransfersOnSrcStream() ? device_->local_device_state()
                                                : dst_local_device;
  CHECK_EQ(dst_local_device->allocation_model(),
           transfer_local_device->allocation_model());

  se::Stream* transfer_stream =
      transfer_local_device->GetDeviceToDeviceStream();

  ScopedHold src_device_buffer(this, ScopedHold::kUsage);
  {
    absl::MutexLock lock(&mu_);
    // We can't perform any other action while a donation hold is in progress.
    WaitForOutstandingDonationHold();
    if (device_buffer_ == nullptr) {
      return InvalidArgument(
          "CopyToDevice called on deleted or donated buffer");
    }
    AcquireHoldLocked(&src_device_buffer);
  }

  StatusOr<std::pair<std::unique_ptr<PjRtBuffer>,
                     std::shared_ptr<BufferSequencingEvent>>>
      buffer_and_event_or = CopyToDeviceHelper(
          dst_device, dst_local_device, transfer_local_device, transfer_stream,
          src_device_buffer.buffer());
  if (!buffer_and_event_or.ok()) {
    return buffer_and_event_or.status();
  }

  auto& buffer_and_event = buffer_and_event_or.ValueOrDie();
  std::unique_ptr<PjRtBuffer>& buffer = buffer_and_event.first;
  std::shared_ptr<BufferSequencingEvent>& event = buffer_and_event.second;

  // prefer_to_retain_reference=*/true means that, when using the
  // ComputeSynchronized allocation model, retain a reference to the
  // src_device_buffer until the copy completes. This is a heuristic; the
  // alternative is to ensure, before freeing the buffer, that the compute
  // stream is synchronized past the transfer, but it seems better to hold onto
  // the buffer too long than to stall the compute stream.
  RecordUsage(std::move(src_device_buffer), device_->local_device_state(),
              transfer_local_device, event, transfer_stream,
              /*prefer_to_retain_reference=*/true);

  return std::move(buffer);
}

void PjRtStreamExecutorBuffer::CopyToRemoteDevice(
    absl::string_view serialized_descriptor, RemoteSendCallback on_done) {
  VLOG(1) << "PjRtStreamExecutorBuffer::CopyToRemoteDevice";
  client_->CopyToRemoteDevice(this, serialized_descriptor, std::move(on_done));
}

void PjRtStreamExecutorBuffer::CopyToRemoteDeviceScattered(
    absl::Span<const std::pair<std::string, RemoteSendCallback>>
        serialized_descriptors_and_callbacks,
    const ScatterDetails& scatter_details) {
  VLOG(1) << "PjRtStreamExecutorBuffer::CopyToRemoteDeviceScattered";
  client_->CopyToRemoteDeviceScattered(
      this, serialized_descriptors_and_callbacks, scatter_details);
}

PjRtFuture<Status> PjRtStreamExecutorBuffer::GetReadyFuture() {
  std::shared_ptr<TrackedDeviceBuffer> device_buffer;
  PjRtFuture<Status>::Promise definition_promise;
  {
    absl::MutexLock lock(&mu_);
    if (device_buffer_ == nullptr) {
      return PjRtFuture<Status>(InvalidArgument(
          "GetReadyFuture() called on deleted or donated buffer"));
    }
    if (!definition_promise_) {
      device_buffer = device_buffer_;
      definition_promise_ = PjRtFuture<Status>::CreatePromise();
    }
    definition_promise = definition_promise_;
  }

  if (device_buffer) {
    LocalDeviceState* local_device_state = device_->local_device_state();
    std::unique_ptr<se::Stream> stream;
    for (auto& event : device_buffer->definition_events()) {
      if (!event->IsComplete()) {
        if (stream == nullptr) {
          stream = local_device_state->BorrowStreamFromPool();
        }
        event->WaitForEventOnStream(stream.get());
      }
    }
    if (stream != nullptr) {
      auto* stream_ptr = stream.release();
      // We already borrowed a stream from the pool so we can safely do the
      // callback directly on that stream instead of bouncing through
      // local_device_state->ThenExecuteCallback. The direct callback saves
      // significant time.
      stream_ptr->ThenDoHostCallback(
          [definition_promise, stream_ptr, local_device_state]() mutable {
            local_device_state->ReturnStreamToPool(
                std::unique_ptr<se::Stream>(stream_ptr));
            definition_promise.Set(OkStatus());
          });
    } else {
      // All events are already complete.
      definition_promise.Set(OkStatus());
    }
  }

  return PjRtFuture<Status>(
      std::move(definition_promise),
      /*on_block_start=*/
      []() {
        tensorflow::profiler::TraceMeProducer traceme(
            "PjRtStreamExecutorBuffer::Await");
        VLOG(1) << "PjRtStreamExecutorBuffer::Await";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id=*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tensorflow::profiler::TraceMeConsumer traceme(
            "PjRtStreamExecutorBuffer::Await", keys.traceme_context_id);
      });
}

namespace {

// Helper struct for the tuple that is transiently constructed to hold the
// arguments of an execution.
struct TupleHandle {
  // The ExecutionInput describing the tuple.
  ExecutionInput execution_input;
  // A definition event that has been recorded on the host_to_device stream
  // after the tuple table transfer.
  std::shared_ptr<BufferSequencingEvent> event;
};

Status CheckCompatibleShapes(bool strict_shape_checking,
                             const Shape& buffer_shape,
                             const Shape& execution_shape,
                             const TransferManager& transfer_manager,
                             int parameter_index) {
  // TODO(misard) Support casting of tuple parameters.
  if (strict_shape_checking || buffer_shape.IsTuple()) {
    if (!ShapeUtil::Equal(buffer_shape, execution_shape)) {
      return InvalidArgument(
          "Executable expected shape %s for argument %d but got "
          "incompatible "
          "shape %s",
          ShapeUtil::HumanStringWithLayout(execution_shape), parameter_index,
          ShapeUtil::HumanStringWithLayout(buffer_shape));
    }
  } else {
    if (transfer_manager.GetByteSizeRequirement(buffer_shape) !=
        transfer_manager.GetByteSizeRequirement(execution_shape)) {
      return InvalidArgument(
          "Executable expected shape %s for argument %d but got "
          "incompatible "
          "shape %s",
          ShapeUtil::HumanStringWithLayout(execution_shape), parameter_index,
          ShapeUtil::HumanStringWithLayout(buffer_shape));
    }
  }
  return OkStatus();
}

// Makes a tuple from the arguments to an execution.
StatusOr<TupleHandle> MakeTupleHelper(
    PjRtStreamExecutorClient* client, LocalDeviceState* local_device,
    bool strict_shape_checking, const Shape& tupled_parameter_shape,
    absl::Span<PjRtBuffer* const> py_buffers,
    absl::Span<const PjRtStreamExecutorBuffer::ScopedHold> device_buffers,
    int device_ordinal) {
  se::DeviceMemoryAllocator* allocator = client->allocator();
  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();

  if (tupled_parameter_shape.tuple_shapes_size() != py_buffers.size()) {
    return InvalidArgument("Executable expected %lld parameters but got %lld",
                           tupled_parameter_shape.tuple_shapes_size(),
                           py_buffers.size());
  }
  for (int i = 0; i < py_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(CheckCompatibleShapes(
        strict_shape_checking, py_buffers[i]->on_device_shape(),
        tupled_parameter_shape.tuple_shapes(i), *transfer_manager, i));
  }

  se::Stream* stream = local_device->host_to_device_stream();
  TF_ASSIGN_OR_RETURN(
      se::OwningDeviceMemory root_table_memory,
      allocator->Allocate(
          device_ordinal,
          transfer_manager->GetByteSizeRequirement(tupled_parameter_shape)));

  if (local_device->allocation_model() ==
      LocalDeviceState::kComputeSynchronized) {
    stream->ThenWaitFor(local_device->compute_stream());
  } else {
    DCHECK(transfer_manager->CanBufferBeAccessedNow(
        local_device->compute_stream()->parent(), root_table_memory.cref()));
  }

  ExecutionInput execution_input(tupled_parameter_shape);
  ShapeTree<MaybeOwningDeviceMemory>::iterator input_iterator =
      execution_input.MutableBuffers()->begin();
  ShapeTree<MaybeOwningDeviceMemory>::iterator iterator_end =
      execution_input.MutableBuffers()->end();
  // First set the root tuple table which is the first buffer in the ShapeTree.
  execution_input.SetBuffer(
      input_iterator->first,
      MaybeOwningDeviceMemory(std::move(root_table_memory)));
  ++input_iterator;
  // Then set each sub-tuple in turn from the parameters.
  for (const PjRtStreamExecutorBuffer::ScopedHold& device_buffer :
       device_buffers) {
    device_buffer.AddToInput(&input_iterator, iterator_end, &execution_input,
                             allocator);
  }
  CHECK(input_iterator == iterator_end);

  TF_RETURN_IF_ERROR(transfer_manager->WriteRootTupleIndexTable(
      stream, execution_input.Buffers()));
  StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().ThenAllocateAndRecordEvent(stream);
  if (!event_or.ok()) {
    StallStreamOnError(local_device, stream);
    return event_or.status();
  }

  auto transfer_event = std::make_shared<BufferSequencingEvent>();
  transfer_event->SetSequencingEvent(std::move(event_or).value(), stream);
  return TupleHandle({std::move(execution_input), std::move(transfer_event)});
}

// Converts a ScopedShapedBuffer returned from an execution into a
// PjRtBuffer.
std::unique_ptr<PjRtBuffer> OutputBufferHelper(
    ScopedShapedBuffer* result_buffer,
    std::shared_ptr<BufferSequencingEvent> definition_event, PjRtClient* client,
    PjRtDevice* device, LocalDeviceState* local_device,
    std::vector<std::shared_ptr<TrackedDeviceBuffer>>& buffers_to_release) {
  std::shared_ptr<TrackedDeviceBuffer> out_buffer =
      TrackedDeviceBuffer::FromScopedShapedBuffer(result_buffer,
                                                  {definition_event});
  auto pjrt_buffer = std::make_unique<PjRtStreamExecutorBuffer>(
      result_buffer->on_device_shape(), std::move(out_buffer), client, device);
  RecordUsage(pjrt_buffer->GetBufferWithUsageHold(), local_device, local_device,
              definition_event, local_device->compute_stream(),
              /*prefer_to_retain_reference=*/false, &buffers_to_release);
  return std::unique_ptr<PjRtBuffer>(std::move(pjrt_buffer));
}
}  // namespace

PjRtStreamExecutorExecutable::PjRtStreamExecutorExecutable(
    std::vector<std::unique_ptr<LocalExecutable>> executables,
    bool parameter_is_tupled_arguments,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices,
    PjRtStreamExecutorClient* client)
    : client_(client),
      device_assignment_(std::move(device_assignment)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {
  TransferManager* transfer_manager =
      client_->client()->backend().transfer_manager();
  executables_.reserve(executables.size());
  for (auto& executable : executables) {
    const auto& computation_layout =
        executable->executable()->module().entry_computation_layout();
    std::vector<Shape> parameter_shapes;
    parameter_shapes.reserve(computation_layout.parameter_count());
    for (int i = 0; i < computation_layout.parameter_count(); ++i) {
      parameter_shapes.push_back(transfer_manager->HostShapeToDeviceShape(
          computation_layout.parameter_shape(i)));
    }
    executables_.emplace_back(std::move(executable));
    on_device_executable_parameter_shapes_.push_back(
        std::move(parameter_shapes));
  }

  int num_partitions;
  if (device_assignment_ == nullptr) {
    // This must go after `executables_` is initialized.
    VLOG(3) << "PjRtStreamExecutorExecutable portable single-core";
    num_partitions = 1;
    CHECK(addressable_devices_.empty());
  } else {
    // This must go after `executables_` is initialized.
    VLOG(3) << "PjRtStreamExecutorExecutable device_assignment:\n"
            << device_assignment_->ToString();
    CHECK_GE(addressable_devices_.size(), 1) << device_assignment_->ToString();
    CHECK_LE(addressable_devices_.size(), client_->addressable_device_count())
        << "Inconsistent local device count.";
    num_partitions = device_assignment_->computation_count();
  }

  // SPMD sharding produces a single executable for multiple partitions.
  if (executables_.size() > 1) {
    CHECK_EQ(num_partitions, executables_.size())
        << "Number of executables " << executables_.size()
        << " did not match number of partitions " << num_partitions;
  }
}

Status PjRtStreamExecutorExecutable::SetUpDonation(bool tuple_inputs) {
  parameters_that_must_be_donated_.reserve(executables_.size());
  for (auto& executable : executables_) {
    TF_ASSIGN_OR_RETURN(std::vector<int> parameters_to_donate,
                        ComputeParametersThatMustBeDonated(
                            executable->executable()->module(), tuple_inputs));
    parameters_that_must_be_donated_.emplace_back(
        std::move(parameters_to_donate));
  }
  return OkStatus();
}

absl::string_view PjRtStreamExecutorExecutable::name() const {
  Executable* executable = executables_[0]->executable();
  if (executable->has_module()) {
    return executable->module().name();
  } else {
    return "<unknown executable>";
  }
}

absl::Span<int const> PjRtStreamExecutorExecutable::ParametersThatMustBeDonated(
    int executable_idx) const {
  return parameters_that_must_be_donated_[executable_idx];
}

StatusOr<std::vector<ExecutionInput>>
PjRtStreamExecutorExecutable::MakeExecutionInputsAndWaitForEvents(
    int device_ordinal, const ExecuteOptions& options,
    absl::Span<const Shape> executable_parameter_shapes,
    absl::Span<PjRtBuffer* const> argument_handles,
    absl::Span<const PjRtStreamExecutorBuffer::ScopedHold> device_buffers,
    absl::flat_hash_set<BufferSequencingEvent*>& events) const {
  std::vector<ExecutionInput> execution_inputs;
  LocalDeviceState* device_state = &(client_->device_state(device_ordinal));
  TransferManager* transfer_manager =
      client_->client()->backend().transfer_manager();
  // Lift tuple_handle outside the conditional so that the event it returns is
  // not destroyed until after the loop below that waits on events.
  std::optional<TupleHandle> tuple_handle;
  if (parameter_is_tupled_arguments_ && !options.arguments_are_tupled) {
    TF_ASSIGN_OR_RETURN(
        tuple_handle,
        MakeTupleHelper(client_, device_state, options.strict_shape_checking,
                        executable_parameter_shapes[0], argument_handles,
                        device_buffers, device_ordinal));
    events.insert(tuple_handle->event.get());
    execution_inputs.emplace_back(std::move(tuple_handle->execution_input));
  } else {
    if (argument_handles.size() != executable_parameter_shapes.size()) {
      return InvalidArgument("Executable expected %lld arguments but got %lld",
                             executable_parameter_shapes.size(),
                             argument_handles.size());
    }
    execution_inputs.reserve(argument_handles.size());
    for (int i = 0; i < argument_handles.size(); ++i) {
      PjRtBuffer* handle = argument_handles[i];

      // Make an ExecutionInput from the device buffer.
      TF_RETURN_IF_ERROR(CheckCompatibleShapes(
          options.strict_shape_checking, handle->on_device_shape(),
          executable_parameter_shapes[i], *transfer_manager, i));
      execution_inputs.emplace_back(executable_parameter_shapes[i]);
      ExecutionInput& execution_input = execution_inputs.back();
      ShapeTree<MaybeOwningDeviceMemory>::iterator input_iterator =
          execution_input.MutableBuffers()->begin();
      ShapeTree<MaybeOwningDeviceMemory>::iterator iterator_end =
          execution_input.MutableBuffers()->end();
      device_buffers[i].AddToInput(&input_iterator, iterator_end,
                                   &execution_input, client_->allocator());
      CHECK(input_iterator == iterator_end);
    }
  }

  for (BufferSequencingEvent* event : events) {
    event->WaitForEventOnStream(device_state->compute_stream());
  }

  return execution_inputs;
}

// Enqueues a computation onto the compute stream. Each buffer returned in
// device_buffers has a usage hold added that must be dropped on error or
// converted on success.
StatusOr<ScopedShapedBuffer> PjRtStreamExecutorExecutable::EnqueueExecution(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    int executable_idx, const RunId& run_id, const ExecuteOptions& options,
    PjRtDevice* device,
    std::vector<PjRtStreamExecutorBuffer::ScopedHold>* device_buffers,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<std::function<void()>>& compute_callbacks) const {
  int device_ordinal = tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                           ->local_device_state()
                           ->device_ordinal();
  LocalDeviceState* device_state = &(client_->device_state(device_ordinal));
  tensorflow::profiler::TraceMeConsumer activity(
      "PjRtStreamExecutorExecutable::EnqueueExecution",
      tensorflow::profiler::ContextType::kPjRt, run_id.ToInt());
  VLOG(3) << "Replica " << replica << ", partition " << partition
          << " mapped to device ordinal for execution: " << device_ordinal;

  absl::flat_hash_set<BufferSequencingEvent*> events;
  device_buffers->reserve(argument_handles.size());
  absl::Span<int const> donated_params =
      ParametersThatMustBeDonated(executable_idx);
  auto donate_it = donated_params.begin();
  for (int i = 0; i < argument_handles.size(); ++i) {
    auto* handle =
        tensorflow::down_cast<PjRtStreamExecutorBuffer*>(argument_handles[i]);
    if (handle->device() != device) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %s, but replica is assigned to device %s.",
          i, replica, handle->device()->DebugString(), device->DebugString());
    }
    bool must_donate = donate_it != donated_params.end() && *donate_it == i;
    if (must_donate) {
      ++donate_it;
    }
    device_buffers->emplace_back(handle->GetBufferWithHold(
        must_donate ? PjRtStreamExecutorBuffer::ScopedHold::kDonation
                    : PjRtStreamExecutorBuffer::ScopedHold::kUsage));
    PjRtStreamExecutorBuffer::ScopedHold& device_buffer =
        device_buffers->back();
    if (!device_buffer.ok()) {
      return InvalidArgument(
          "Invalid buffer passed to Execute() as argument %d to replica %d: "
          "%s",
          i, replica, device_buffer.status().ToString());
    }
    // If we are trying to donate the buffer wait on the usage events as well
    // as the definition events to ensure that all reads have been completed
    // before the buffer is mutated. Usage holds are excluded during a donation
    // hold so we know that the set of usage events won't be modified while we
    // are enqueueing.
    GetDeviceBufferEvents(*device_buffer, /*get_usage_events=*/must_donate,
                          &events);
  }

  if (options.arguments_are_tupled) {
    if (!parameter_is_tupled_arguments_) {
      return InvalidArgument(
          "Arguments may only be supplied as a tuple when the executable was "
          "compiled with a single tupled parameter");
    }
    if (argument_handles.size() != 1) {
      return InvalidArgument(
          "Option arguments_are_tupled was true but %d buffers were passed to "
          "execution",
          argument_handles.size());
    }
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<ExecutionInput> execution_inputs,
      MakeExecutionInputsAndWaitForEvents(
          device_ordinal, options,
          on_device_executable_parameter_shapes_[executable_idx],
          argument_handles, *device_buffers, events));

  ExecutableRunOptions run_options;
  run_options.set_stream(device_state->compute_stream());
  run_options.set_host_to_device_stream(device_state->host_to_device_stream());
  run_options.set_allocator(client_->allocator());
  run_options.set_intra_op_thread_pool(
      client_->client()->backend().eigen_intra_op_thread_pool_device());
  run_options.set_device_assignment(device_assignment.get());
  run_options.set_run_id(run_id);
  run_options.set_rng_seed(device_state->GetNewPrngSeed());
  run_options.set_gpu_executable_run_options(client_->gpu_run_options());
  run_options.set_launch_id(options.launch_id);
  if (run_options.launch_id() != 0) {
    VLOG(3) << "launch id for " << name() << ": " << run_options.launch_id();
  }

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  std::shared_ptr<Semaphore::ScopedReservation> compute_reservation;
  {
    tensorflow::profiler::TraceMe traceme("ComputeSemaphoreAcquire");
    compute_reservation = std::make_shared<Semaphore::ScopedReservation>(
        device_state->compute_semaphore().ScopedAcquire(1));
  }

  StatusOr<ExecutionOutput> result_buffer_or_status =
      executables_[executable_idx]->RunAsync(std::move(execution_inputs),
                                             run_options);

  VLOG(1) << "Replica " << replica << " partition " << partition
          << " completed; ok=" << result_buffer_or_status.ok();

  if (!result_buffer_or_status.ok()) {
    return result_buffer_or_status.status();
  }

  if (device_state->allocation_model() == LocalDeviceState::kSynchronous) {
    ExecutionOutput& execution_output = result_buffer_or_status.ValueOrDie();
    // If we used a transient tuple for the arguments we donated its root table
    // buffer. In that case, and/or if we donated any input buffers that were
    // not aliased, the donated buffers are going to be passed back to us via
    // the execution output. We need to ensure they aren't freed until after
    // execution completes. (Currently XLA does not support aliasing tuple
    // tables, so if any donated parameter is a tuple there will be donated but
    // unaliased buffers.)
    std::vector<se::OwningDeviceMemory> donated_memory =
        execution_output.ConsumeToBeReleased();
    absl::InlinedVector<se::DeviceMemoryBase, 3> donated_ptrs;
    donated_ptrs.reserve(donated_memory.size());
    for (se::OwningDeviceMemory& owning : donated_memory) {
      // Release the owning memory so we can pass it to the closure.
      donated_ptrs.push_back(owning.Release());
    }
    compute_callbacks.push_back(
        [references{std::make_tuple(executables_[executable_idx],
                                    compute_reservation, device_assignment)},
         donated_ptrs{std::move(donated_ptrs)}, allocator{client_->allocator()},
         device_ordinal]() {
          for (const auto& ptr : donated_ptrs) {
            TF_CHECK_OK(allocator->Deallocate(device_ordinal, ptr));
          }
        });
  } else {
    // Any donated memory returned by the ExecutionOutput can be immediately
    // freed.
    compute_callbacks.push_back(
        [to_release{std::make_tuple(executables_[executable_idx],
                                    compute_reservation,
                                    device_assignment)}]() {});
  }

  return std::move(result_buffer_or_status).value().ConsumeResult();
}

std::vector<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorExecutable::MakeOutputBuffers(
    int device_ordinal, const ExecuteOptions& options,
    ScopedShapedBuffer result_buffer,
    std::shared_ptr<BufferSequencingEvent> definition_event, PjRtDevice* device,
    std::vector<std::function<void()>>& compute_callbacks,
    std::vector<std::shared_ptr<TrackedDeviceBuffer>>& buffers_to_release)
    const {
  tensorflow::profiler::TraceMe traceme("MakeOutputBuffers");
  std::vector<std::unique_ptr<PjRtBuffer>> outputs;
  LocalDeviceState* device_state = &(client_->device_state(device_ordinal));
  if (options.untuple_result && result_buffer.on_device_shape().IsTuple()) {
    int tuple_count = result_buffer.on_device_shape().tuple_shapes_size();
    outputs.reserve(tuple_count);
    // Take ownership of each of the output values, leaving only the root table
    // in result_buffer.
    for (int i = 0; i < tuple_count; ++i) {
      ScopedShapedBuffer tuple_buffer = result_buffer.TakeSubTree({i});
      outputs.push_back(OutputBufferHelper(&tuple_buffer, definition_event,
                                           client_, device, device_state,
                                           buffers_to_release));
    }
    if (device_state->allocation_model() == LocalDeviceState::kSynchronous) {
      // Don't release the root buffer until after execution completes.
      ShapedBuffer root_buffer_holder = result_buffer.release();
      se::DeviceMemoryBase root_buffer = root_buffer_holder.root_buffer();
      compute_callbacks.push_back(
          [root_buffer, allocator{client_->allocator()}, device_ordinal]() {
            TF_CHECK_OK(allocator->Deallocate(device_ordinal, root_buffer));
          });
    }
  } else {
    outputs.push_back(OutputBufferHelper(&result_buffer, definition_event,
                                         client_, device, device_state,
                                         buffers_to_release));
  }
  return outputs;
}

StatusOr<PjRtLoadedExecutable::Result>
PjRtStreamExecutorExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options, bool fill_future,
    PjRtDevice* device) const {
  const uint64_t start_time_usecs = tensorflow::Env::Default()->NowMicros();
  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int device_id = (*device_assignment_)(replica, partition);
    TF_ASSIGN_OR_RETURN(device, client_->LookupDevice(device_id));
    device_assignment = device_assignment_;
  } else {
    CHECK(device_assignment_ == nullptr);
    CHECK_EQ(replica, 0);
    CHECK_EQ(partition, 0);
    CHECK(addressable_devices_.empty());
    device_assignment = std::make_shared<DeviceAssignment>(1, 1);
    (*device_assignment)(0, 0) = device->id();
  }

  CHECK_EQ(device->process_index(), client_->process_index());
  int device_ordinal = tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                           ->local_device_state()
                           ->device_ordinal();
  tensorflow::profiler::TraceMe traceme(
      "PjRtStreamExecutorExecutable::ExecuteHelper");
  VLOG(1) << "Replica " << replica << ", partition " << partition
          << " mapped to device ordinal for execution: " << device_ordinal;

  // SPMD sharding produces a single executable for multiple partitions.
  int executable_idx = executables_.size() > 1 ? partition : 0;

  std::vector<std::function<void()>> compute_callbacks;
  std::vector<PjRtStreamExecutorBuffer::ScopedHold> device_buffers;
  device_buffers.reserve(argument_handles.size());
  StatusOr<ScopedShapedBuffer> result_buffer_or_status = EnqueueExecution(
      argument_handles, replica, partition, executable_idx, run_id, options,
      device, &device_buffers, std::move(device_assignment), compute_callbacks);

  if (!result_buffer_or_status.ok()) {
    LOG(ERROR) << "Execution of replica " << replica
               << " failed: " << result_buffer_or_status.status();
    return result_buffer_or_status.status();
  }
  ScopedShapedBuffer result_buffer = std::move(result_buffer_or_status).value();

  LocalDeviceState* device_state = &(client_->device_state(device_ordinal));
  se::Stream* stream = device_state->compute_stream();
  StatusOr<EventPool::Handle> event_or =
      device_state->event_pool().ThenAllocateAndRecordEvent(stream);
  if (!event_or.ok()) {
    StallStreamOnError(device_state, stream);
    for (PjRtStreamExecutorBuffer::ScopedHold& b : device_buffers) {
      if (b.type() == PjRtStreamExecutorBuffer::ScopedHold::kDonation) {
        // Even though there was an error we need to call ConfirmDonation, which
        // renders b invalid, since the computation has been enqueued and b has
        // been donated.
        b.ConfirmDonation();
      }
    }
    return event_or.status();
  }
  auto definition_event = std::make_shared<BufferSequencingEvent>();
  definition_event->SetSequencingEvent(std::move(event_or).value(), stream);
  std::vector<std::shared_ptr<TrackedDeviceBuffer>> buffers_to_release;
  std::vector<std::unique_ptr<PjRtBuffer>> outputs = MakeOutputBuffers(
      device_ordinal, options, std::move(result_buffer), definition_event,
      device, compute_callbacks, buffers_to_release);

  for (PjRtStreamExecutorBuffer::ScopedHold& b : device_buffers) {
    // prefer_to_retain_reference=false because when using the
    // ComputeSynchronized allocation model we don't need to retain a reference
    // to the device_buffer during execution because by definition the compute
    // stream is synchronized past the execution.
    if (b.type() == PjRtStreamExecutorBuffer::ScopedHold::kUsage) {
      RecordUsage(std::move(b), device_state, device_state, definition_event,
                  stream,
                  /*prefer_to_retain_reference=*/false, &buffers_to_release);
    } else {
      CHECK(b.type() == PjRtStreamExecutorBuffer::ScopedHold::kDonation);
      b.ConfirmDonation();
    }
  }

  std::optional<PjRtFuture<Status>> future;
  if (fill_future) {
    auto promise = PjRtFuture<Status>::CreatePromise();
    future = PjRtFuture<Status>(promise);
    compute_callbacks.push_back(
        [promise = std::move(promise)]() mutable { promise.Set(OkStatus()); });
  }
  device_state->ThenExecuteCallback(
      stream, [callbacks{std::move(compute_callbacks)},
               buffers_to_release{std::move(buffers_to_release)}]() {
        for (auto& fn : callbacks) {
          fn();
        }
      });
  ReportExecutableEnqueueTime(tensorflow::Env::Default()->NowMicros() -
                              start_time_usecs);
  return Result({/*future=*/std::move(future), /*buffers=*/std::move(outputs)});
}

StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
PjRtStreamExecutorExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<Status>>>& returned_futures) {
  if (device_assignment_ == nullptr) {
    return InvalidArgument("Execute expects a non-null device_assignment");
  }

  RunId run_id;
  tensorflow::profiler::TraceMeProducer activity(
      "PjRtStreamExecutorExecutable::Execute",
      tensorflow::profiler::ContextType::kPjRt, run_id.ToInt());

  const int num_addressable_devices = addressable_devices_.size();

  if (argument_handles.size() != num_addressable_devices) {
    return InvalidArgument(
        "Attempted to execute with %d argument lists when local device "
        "count is %d (total replica count: %d, partition count: %d)",
        argument_handles.size(), num_addressable_devices, num_replicas(),
        num_partitions());
  }

  VLOG(1) << "Executing computation " << name()
          << "; num_replicas=" << num_replicas()
          << " num_partitions=" << num_partitions()
          << " num_addressable_devices=" << num_addressable_devices;
  std::vector<StatusOr<Result>> results(num_addressable_devices);
  if (num_addressable_devices == 1) {
    // Fast-path if there is only one device — run the computation on the
    // current thread.
    const int replica = addressable_device_logical_ids_[0].replica;
    const int partition = addressable_device_logical_ids_[0].partition;
    results[0] = ExecuteHelper(argument_handles[0], replica, partition, run_id,
                               options, returned_futures.has_value());
  } else {
    absl::Mutex mu;
    int running = num_addressable_devices;
    int failed = 0;
    Status first_failure_status;

    for (int i = 0; i < num_addressable_devices; ++i) {
      const int replica = addressable_device_logical_ids_[i].replica;
      const int partition = addressable_device_logical_ids_[i].partition;
      PjRtDevice* device = addressable_devices_[i];
      const LocalDeviceState& device_state =
          *tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
               ->local_device_state();
      device_state.execute_thread()->Schedule([&, replica, partition, i] {
        results[i] =
            ExecuteHelper(argument_handles[i], replica, partition, run_id,
                          options, returned_futures.has_value());

        absl::MutexLock lock(&mu);
        --running;
        if (!results[i].ok()) {
          if (failed == 0) {
            first_failure_status = results[i].status();
          }
          ++failed;
        }
      });
    }

    auto done_running_or_failed = [&]() {
      mu.AssertHeld();
      return running == 0 || failed > 0;
    };
    absl::MutexLock lock(&mu);
    mu.Await(absl::Condition(&done_running_or_failed));
    if (failed > 0) {
      auto done_running = [&]() {
        mu.AssertHeld();
        return running == 0;
      };
      // If execution does not terminate within a reasonable amount of time,
      // we may be stuck at a cross-replica barrier on-device. Terminate the
      // process since that's the only way we can escape this situation at the
      // moment (b/130629719).
      if (!mu.AwaitWithTimeout(absl::Condition(&done_running),
                               absl::Seconds(10))) {
        LOG(FATAL)
            << "Replicated computation launch failed, but not all replicas "
               "terminated. Aborting process to work around deadlock. "
               "Failure message (there may have been multiple failures, see "
               "the error log for all failures): \n\n"
            << first_failure_status.error_message();
      }
    }
  }
  VLOG(1) << "Replicated execution complete.";

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> wrapped_results(
      num_addressable_devices);
  if (returned_futures.has_value()) {
    returned_futures->reserve(num_addressable_devices);
  }
  for (int i = 0; i < num_addressable_devices; ++i) {
    const int replica = addressable_device_logical_ids_[i].replica;
    const int partition = addressable_device_logical_ids_[i].partition;
    auto& statusor = results[i];
    if (!statusor.ok()) {
      if (returned_futures.has_value()) {
        returned_futures->clear();
      }
      if (num_addressable_devices == 1) {
        return statusor.status();
      } else {
        return AppendStatus(
            statusor.status(),
            absl::StrFormat("while running replica %d and partition %d of a "
                            "replicated computation (other "
                            "replicas may have failed as well).",
                            replica, partition));
      }
    }
    wrapped_results[i] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      returned_futures->push_back(*std::move(statusor->future));
    }
  }
  return wrapped_results;
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtStreamExecutorExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  if (device_assignment_ == nullptr) {
    return InvalidArgument("ExecuteShard expects a non-null device_assignment");
  }
  for (int i = 0; i < addressable_devices_.size(); ++i) {
    if (addressable_devices_[i] == device) {
      VLOG(1) << "ExecuteShard executes computation " << name()
              << " on assigned replica/partition on device "
              << device->DebugString();
      TF_ASSIGN_OR_RETURN(
          auto result,
          ExecuteHelper(argument_handles,
                        addressable_device_logical_ids_[i].replica,
                        addressable_device_logical_ids_[i].partition, RunId(),
                        options, fill_future));
      returned_future = std::move(result.future);
      return std::move(result.buffers);
    }
  }
  return InvalidArgument(
      "ExecuteShard attempted to execute on device id %d which is not "
      "addressable by this client",
      device->id());
}

StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtStreamExecutorExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options,
    std::optional<PjRtFuture<Status>>& returned_future, bool fill_future) {
  if (device_assignment_ != nullptr) {
    return InvalidArgument("ExecutePortable gets a non-portable executable");
  }
  if (num_replicas() != 1 || num_partitions() != 1) {
    return InvalidArgument(
        "ExecutePortable expects a single-core executable but gets "
        "one with %d replica %d partition",
        num_replicas(), num_partitions());
  }
  if (device == nullptr) {
    return InvalidArgument("ExecutePortable expects a device to be specified");
  }
  VLOG(1) << "ExecutePortable executes single-core portable executable "
          << name();
  TF_ASSIGN_OR_RETURN(auto result, ExecuteHelper(argument_handles,
                                                 /*replica=*/0,
                                                 /*partition=*/0, RunId(),
                                                 options, fill_future, device));
  returned_future = std::move(result.future);
  return std::move(result.buffers);
}

StatusOr<std::vector<std::shared_ptr<HloModule>>>
PjRtStreamExecutorExecutable::GetHloModules() const {
  std::vector<std::shared_ptr<HloModule>> modules;
  modules.reserve(executables().size());
  for (const auto& local_exec : executables()) {
    if (!local_exec->executable()->has_module()) {
      return InvalidArgument("Executable does not have HLO modules.");
    }
    modules.push_back(local_exec->executable()->shared_module());
  }
  return std::move(modules);
}

StatusOr<PjRtStreamExecutorClient::ExecutableExtras>
PjRtStreamExecutorClient::GetExecutableExtras(CompileOptions* options) {
  ExecutableExtras extras;
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<PjRtStreamExecutorExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  ExecutableBuildOptions& build_options = options->executable_build_options;
  if (!build_options.compile_thread_pool()) {
    build_options.set_compile_thread_pool(thread_pool());
  }
  if (!build_options.device_allocator()) {
    build_options.set_device_allocator(allocator());
  }

  int num_replicas;
  int num_partitions;
  TF_RETURN_IF_ERROR(ParseDeviceAssignmentCompileOptions(
      options->compile_portable_executable, &options->executable_build_options,
      [this](int num_replicas, int num_partitions) {
        return this->GetDefaultDeviceAssignment(num_replicas, num_partitions);
      },
      &num_replicas, &num_partitions, &device_assignment));

  // Find devices that are addressable by this client/task.
  if (device_assignment != nullptr) {
    addressable_device_logical_ids.reserve(num_replicas * num_partitions);
    addressable_devices.reserve(num_replicas * num_partitions);
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        int device_id = (*device_assignment)(replica, partition);
        TF_ASSIGN_OR_RETURN(PjRtDevice * device, LookupDevice(device_id));
        if (device->process_index() != process_index()) {
          VLOG(3) << "Non-local device: " << device_id;
          continue;
        }
        PjRtLoadedExecutable::LogicalDeviceIds logica_device_ids;
        logica_device_ids.replica = replica;
        logica_device_ids.partition = partition;
        addressable_device_logical_ids.push_back(std::move(logica_device_ids));
        addressable_devices.push_back(device);
      }
    }
    if (addressable_devices.empty()) {
      return InvalidArgument(
          "Device assignment (%s) does not have any local devices.",
          device_assignment->ToString());
    }

    if (build_options.device_ordinal() < 0) {
      build_options.set_device_ordinal(
          addressable_devices.front()->local_hardware_id());
    }
  }
  return extras;
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::Compile(const XlaComputation& computation,
                                  CompileOptions options) {
  tensorflow::profiler::TraceMe traceme("PjRtStreamExecutorClient::Compile");
  VLOG(1) << "PjRtStreamExecutorClient::Compile";

  TF_ASSIGN_OR_RETURN(ExecutableExtras extras, GetExecutableExtras(&options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<PjRtStreamExecutorExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  std::vector<const Shape*> argument_layout_pointers;
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [local_client = client()](Shape shape) {
        return local_client->backend()
            .transfer_manager()
            ->ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      client()->Compile(computation, argument_layout_pointers,
                        options.executable_build_options));

  auto executable = std::make_unique<PjRtStreamExecutorExecutable>(
      std::move(local_executables), options.parameter_is_tupled_arguments,
      std::move(device_assignment), std::move(addressable_device_logical_ids),
      std::move(addressable_devices), this);
  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(options.parameter_is_tupled_arguments));
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::Compile(mlir::ModuleOp module,
                                  CompileOptions options) {
  XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false));
  return Compile(xla_computation, options);
}

}  // namespace xla
