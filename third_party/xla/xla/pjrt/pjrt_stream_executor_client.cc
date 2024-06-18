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

#include "xla/pjrt/pjrt_stream_executor_client.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/casts.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_computation.h"
#include "xla/executable_run_options.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/metrics.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_layout.h"
#include "xla/service/executable.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/env.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

PjRtStreamExecutorMemorySpace::PjRtStreamExecutorMemorySpace(
    int id, PjRtDevice* device, absl::string_view kind, int kind_id)
    : id_(id), device_(device), kind_(kind), kind_id_(kind_id) {
  DCHECK(device_ != nullptr && device_->client() != nullptr);
  auto* client = device_->client();
  to_string_ = absl::StrFormat("MEMORY_SPACE_%i", id_);
  debug_string_ = absl::StrFormat(
      "PjRtStreamExecutorMemory(id=%i, process_index=%i, client=%s)", id_,
      client->process_index(), client->platform_name());
}

PjRtPlatformId PjRtStreamExecutorDevice::platform_id() const {
  return client_->platform_id();
}
absl::string_view PjRtStreamExecutorDevice::platform_name() const {
  return client_->platform_name();
}

absl::StatusOr<LocalDeviceState*>
PjRtStreamExecutorDevice::GetLocalDeviceState() const {
  if (local_device_state_) {
    return local_device_state_.get();
  }
  return InvalidArgument("Device %s is not a local device.", DebugString());
}

absl::StatusOr<DeviceAssignment> DevicesToDeviceAssignment(
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

class CpuAllocator : public tsl::Allocator {
 public:
  CpuAllocator() = default;

  std::string Name() override { return "cpu"; }

  void* AllocateRaw(size_t alignment, size_t num_bytes) override {
    return tsl::port::AlignedMalloc(num_bytes, alignment);
  }
  void DeallocateRaw(void* ptr) override { return tsl::port::AlignedFree(ptr); }
};

PjRtStreamExecutorClient::PjRtStreamExecutorClient(
    std::string platform_name, LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index, std::unique_ptr<se::DeviceMemoryAllocator> allocator,
    std::unique_ptr<tsl::Allocator> host_memory_allocator,
    bool should_stage_host_to_device_transfers,
    std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options)
    : platform_id_(tsl::Fingerprint64(platform_name)),
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
          tsl::Env::Default(), "pjrt_thread_pool",
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
                 return a->local_device_id() < b->local_device_id();
               });
}

absl::StatusOr<DeviceAssignment>
PjRtStreamExecutorClient::GetDefaultDeviceAssignment(int num_replicas,
                                                     int num_partitions) const {
  return client_->backend().computation_placer()->AssignDevices(num_replicas,
                                                                num_partitions);
}

absl::StatusOr<Layout> PjRtStreamExecutorClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  Shape shape = ShapeUtil::MakeShape(element_type, dims);
  TF_ASSIGN_OR_RETURN(
      shape,
      client()->backend().transfer_manager()->ChooseCompactLayoutForShape(
          shape));
  return shape.layout();
}

absl::StatusOr<std::unique_ptr<HloCostAnalysis>>
PjRtStreamExecutorClient::GetHloCostAnalysis() const {
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
        auto status = local_device->compute_stream()->WaitFor(stream);
        if (!status.ok()) {
          LOG(ERROR) << "Stalling compute stream failed: " << status;
        }
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
  tsl::profiler::TraceMe traceme("RecordUsage");
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
      buffer_local_device->ThenRelease(usage_stream, device_buffer.buffer())
          .IgnoreError();
    }
  }
  device_buffer.ConvertUsageHold(usage_stream, event,
                                 retain_buffer_until_completion);
}

// Adds necessary synchronization after a copy has been enqueued to a buffer.
// definition_event was added when the buffer was allocated, but has not yet
// had an event recorded.
absl::Status AddDestinationBufferSynchronization(
    LocalDeviceState* local_device,
    PjRtStreamExecutorBuffer::ScopedHold device_buffer,
    std::shared_ptr<BufferSequencingEvent> definition_event,
    se::Stream* copy_stream) {
  absl::StatusOr<EventPool::Handle> event_or =
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
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtStreamExecutorBuffer>>
AllocateDestinationBuffer(
    const Shape& on_host_shape, PjRtDevice* device,
    LocalDeviceState* local_device, se::Stream* copy_stream,
    bool is_uninitialized_create, PjRtStreamExecutorClient* client,
    std::shared_ptr<BufferSequencingEvent> definition_event) {
  if (on_host_shape.IsTuple() && on_host_shape.tuple_shapes_size() == 0) {
    return InvalidArgument("Can't make a buffer from an empty tuple");
  }

  auto* se_client = tensorflow::down_cast<PjRtStreamExecutorClient*>(client);
  TransferManager* transfer_manager =
      se_client->client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer dst_buffer,
                      transfer_manager->AllocateScopedShapedBuffer(
                          on_host_shape, se_client->allocator(),
                          local_device->local_device_id().value()));
  if (local_device->allocation_model() ==
      LocalDeviceState::kComputeSynchronized) {
    if (copy_stream == nullptr) {
      CHECK(is_uninitialized_create);
    } else {
      CHECK(copy_stream->WaitFor(local_device->compute_stream()).ok());
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
    // But if the caller provided a definition event then we record that. Also
    // put it as the first definition event so that we can guarantee only the
    // first one might not have event recorded.
    if (definition_event) {
      definition_events.emplace_back(definition_event);
    }
    if (local_device->allocation_model() ==
        LocalDeviceState::kComputeSynchronized) {
      // The allocation is not valid until the compute stream passes this point,
      // so add a definition event in the compute stream.
      definition_events.emplace_back(
          std::make_shared<BufferSequencingEvent>(client->thread_pool()));
      TF_ASSIGN_OR_RETURN(EventPool::Handle event,
                          local_device->event_pool().ThenAllocateAndRecordEvent(
                              local_device->compute_stream()));
      definition_events.back()->SetSequencingEvent(
          std::move(event), local_device->compute_stream());
    }
  } else {
    // We have at least one definition event, for the copy completing to
    // the device buffers.
    if (definition_event) {
      definition_events.emplace_back(definition_event);
    } else {
      definition_events.emplace_back(
          std::make_shared<BufferSequencingEvent>(client->thread_pool()));
    }
  }
  se::Stream* tuple_table_stream = local_device->host_to_device_stream();
  if (on_device_shape.IsTuple()) {
    // We also need to copy the tuple tables, so we'll have an additional
    // definition event for that copy to complete.
    if (tuple_table_stream != copy_stream) {
      if (local_device->allocation_model() ==
          LocalDeviceState::kComputeSynchronized) {
        DCHECK(
            tuple_table_stream->WaitFor(local_device->compute_stream()).ok());
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

    definition_events.emplace_back(
        std::make_shared<BufferSequencingEvent>(client->thread_pool()));
    absl::StatusOr<EventPool::Handle> event_or =
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
      on_device_shape, std::move(dst_device_buffer), client, device,
      device->default_memory_space().value_or(nullptr));

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
    absl::StatusOr<std::shared_ptr<TrackedDeviceBuffer>>&& buffer_or) {
  CHECK(!ok());
  if (buffer_or.ok()) {
    buffer_ = buffer_or.value();
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

bool PjRtStreamExecutorBuffer::IsOnCpu() const { return false; }

absl::StatusOr<Shape> PjRtStreamExecutorBuffer::logical_on_device_shape() {
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
  absl::StatusOr<EventPool::Handle> event_or =
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

  absl::Status WaitUntilBufferReadyOnStream(std::intptr_t stream) override {
    for (const std::shared_ptr<BufferSequencingEvent>& event :
         external_reference_->definition_events()) {
      TF_RETURN_IF_ERROR(event->WaitForEventOnExternalStream(stream));
    }
    return absl::OkStatus();
  }

 private:
  PjRtStreamExecutorBuffer::ScopedHold external_reference_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
PjRtStreamExecutorBuffer::AcquireExternalReference() {
  ScopedHold hold = GetBufferWithExternalReference();
  absl::Status hold_status = hold.status();
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

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
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

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorBuffer::DonateWithControlDependency(PjRtFuture<> dependency) {
  VLOG(1) << "PjRtStreamExecutorBuffer::DonateWithControlDependency";
  std::unique_ptr<PjRtBuffer> new_buffer;

  auto tracked_buffer =
      GetBufferWithHold(PjRtStreamExecutorBuffer::ScopedHold::kDonation);

  if (!tracked_buffer.ok()) {
    return InvalidArgument(
        "Invalid buffer passed to DonateWithControlDependency: %s",
        tracked_buffer.status().ToString());
  }

  // Copy all the data in the existing tracked_buffer.
  absl::InlinedVector<se::DeviceMemoryBase, 4> buffers(
      tracked_buffer->device_memory().begin(),
      tracked_buffer->device_memory().end());
  auto original_definition_events = tracked_buffer->definition_events();
  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
      definition_events;

  auto definition_event_for_status =
      std::make_shared<BufferSequencingEvent>(client()->thread_pool());
  // definition_event_for_status must be the first one so that it blocks other
  // actions like D2H transfer from execution before the buffer is ready.
  definition_events.push_back(definition_event_for_status);
  definition_events.insert(definition_events.end(),
                           original_definition_events.begin(),
                           original_definition_events.end());

  auto new_device_buffer = std::make_shared<TrackedDeviceBuffer>(
      tracked_buffer->allocator(), device()->local_device_id().value(),
      std::move(buffers), std::move(definition_events),
      /*on_delete_callback=*/nullptr);

  // Make the new buffer which is identical to the old, except for the new
  // definition event.
  new_buffer =
      std::unique_ptr<PjRtBuffer>(std::make_unique<PjRtStreamExecutorBuffer>(
          on_device_shape(), std::move(new_device_buffer), client(), device(),
          device()->default_memory_space().value_or(nullptr)));

  PjRtStreamExecutorDevice* device = this->device();
  LocalDeviceState* local_device = device->local_device_state();
  dependency.OnReady(
      [definition_event_for_status = std::move(definition_event_for_status),
       local_device](absl::Status status) mutable {
        // Forward the absl::Status from the supplied dependency to the
        // definition event.
        auto stream = local_device->BorrowStreamFromPool();
        auto event =
            local_device->event_pool().ThenAllocateAndRecordEvent(stream.get());
        TF_CHECK_OK(event.status());
        definition_event_for_status->SetSequencingEvent(
            std::move(event).value(), stream.get());
        local_device->ReturnStreamToPool(std::move(stream));
      });

  tracked_buffer.ConfirmDonation();
  return new_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer, PjRtDevice* device,
    const Layout* device_layout) {
  tsl::profiler::TraceMe traceme(
      "PjRtStreamExecutorClient::BufferFromHostBuffer");
  Shape device_shape = ShapeUtil::MakeShape(type, dims);
  VLOG(1) << "PjRtStreamExecutorClient::BufferFromHostBuffer: shape: "
          << device_shape.ToString() << " device: " << device->DebugString();
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());

  absl::InlinedVector<int64_t, 4> tmp_strides;
  if (!byte_strides) {
    tmp_strides.resize(dims.size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(device_shape, absl::MakeSpan(tmp_strides)));
    byte_strides = tmp_strides;
  }
  int64_t size = ShapeUtil::ByteSizeOf(device_shape);

  TransferManager* transfer_manager = client()->backend().transfer_manager();
  if (device_layout != nullptr) {
    *(device_shape.mutable_layout()) = *device_layout;
  } else {
    TF_ASSIGN_OR_RETURN(
        device_shape,
        transfer_manager->ChooseCompactLayoutForShape(device_shape));
  }
  absl::InlinedVector<int64_t, 4> shape_strides(device_shape.dimensions_size());
  TF_RETURN_IF_ERROR(
      ShapeUtil::ByteStrides(device_shape, absl::MakeSpan(shape_strides)));
  bool host_and_device_strides_equal =
      (size == 0 || *byte_strides == shape_strides);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtStreamExecutorBuffer> py_buffer,
      AllocateDestinationBuffer(device_shape, device, local_device,
                                local_device->host_to_device_stream(),
                                /*is_uninitialized_create=*/false, this));

  PjRtStreamExecutorBuffer::ScopedHold device_buffer(
      py_buffer->GetBufferWithUsageHold());
  CHECK(device_buffer.ok());

  std::shared_ptr<TransposePlan> transpose;
  if (!host_and_device_strides_equal) {
    absl::InlinedVector<int64_t, 4> permutation(dims.size());
    absl::c_reverse_copy(device_shape.layout().minor_to_major(),
                         permutation.begin());
    TransposePlan::Options options;
    options.elem_size_in_bytes = primitive_util::ByteWidth(type);
    options.dims = dims;
    options.permutation = permutation;
    options.input_layout = TransposePlan::Striding{*byte_strides};
    absl::MutexLock lock(&transpose_mu_);
    TF_ASSIGN_OR_RETURN(transpose, transpose_cache_.GetOrCreate(options));
  }

  bool should_pack = primitive_util::IsSubByteNonPredType(type) &&
                     transfer_manager->PackSubbyteTypes();
  int64_t packed_size;
  if (should_pack) {
    packed_size =
        CeilOfRatio<int64_t>(size, 8 / primitive_util::BitWidth(type));
  } else {
    packed_size = size;
  }

  // If necessary, allocate a host-side buffer for staging host-to-device
  // transfers. On GPU this is a buffer in pinned memory.
  std::shared_ptr<void> staging_buffer;
  bool must_use_staging_buffer =
      host_buffer_semantics == HostBufferSemantics::kImmutableOnlyDuringCall ||
      !host_and_device_strides_equal || packed_size != size;
  // Allocating multigigabyte pinned buffers can be very slow. In that case,
  // using a staging buffer is probably worse than not using one.
  // TODO(phawkins): add chunking for transfers.
  if (must_use_staging_buffer || (should_stage_host_to_device_transfers() &&
                                  packed_size < (int64_t{1} << 30))) {
    void* ptr = host_memory_allocator()->AllocateRaw(
        tsl::Allocator::kAllocatorAlignment, transpose ? size : packed_size);
    staging_buffer = std::shared_ptr<void>(
        ptr, [host_memory_allocator = host_memory_allocator()](void* ptr) {
          host_memory_allocator->DeallocateRaw(ptr);
        });
  }

  // Copy the buffer into a staging buffer before returning control to the
  // caller if the caller only guaranteed that the buffer is valid for the
  // duration of the call. Otherwise, we stage (if necessary) on a separate
  // thread.
  if (host_buffer_semantics == HostBufferSemantics::kImmutableOnlyDuringCall) {
    if (transpose) {
      transpose->Execute(data, staging_buffer.get());
      if (should_pack) {
        primitive_util::PackIntN(
            type,
            absl::MakeConstSpan(static_cast<const char*>(staging_buffer.get()),
                                size),
            absl::MakeSpan(static_cast<char*>(staging_buffer.get()),
                           packed_size));
      }
    } else {
      if (should_pack) {
        primitive_util::PackIntN(
            type, absl::MakeConstSpan(static_cast<const char*>(data), size),
            absl::MakeSpan(static_cast<char*>(staging_buffer.get()),
                           packed_size));
      } else {
        std::memcpy(staging_buffer.get(), data, size);
      }
    }
    if (on_done_with_host_buffer) {
      std::move(on_done_with_host_buffer)();
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
       type, packed_size, movable_device_buffer{device_buffer.ToClosure()},
       device_shape, should_pack, py_buffer{py_buffer.get()},
       on_device_shape{py_buffer->on_device_shape()},
       staging_buffer{std::move(staging_buffer)},
       on_done_with_host_buffer =
           on_done_with_host_buffer
               ? std::make_shared<absl::AnyInvocable<void() &&>>(
                     std::move(on_done_with_host_buffer))
               : nullptr,
       host_buffer_semantics, transpose{std::move(transpose)}]() {
        PjRtStreamExecutorBuffer::ScopedHold device_buffer(
            movable_device_buffer);
        // This function uses TF_CHECK_OK and value() since we have no way
        // to report failures from a callback. However, the operations here are
        // unlikely to fail and not recoverable even if we were to fail: DMAs to
        // memory that has already been allocated, and a possible Event
        // allocation.

        se::DeviceMemoryBase device_memory = device_buffer->device_memory()[0];

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
              if (should_pack) {
                primitive_util::PackIntN(
                    type,
                    absl::MakeConstSpan(
                        static_cast<const char*>(staging_buffer.get()), size),
                    absl::MakeSpan(static_cast<char*>(staging_buffer.get()),
                                   packed_size));
              }
            } else {
              if (should_pack) {
                primitive_util::PackIntN(
                    type,
                    absl::MakeConstSpan(static_cast<const char*>(data), size),
                    absl::MakeSpan(static_cast<char*>(staging_buffer.get()),
                                   packed_size));
              } else {
                std::memcpy(staging_buffer.get(), data, size);
              }
            }
          }
          TF_CHECK_OK(local_device->host_to_device_stream()->Memcpy(
              &device_memory, staging_buffer.get(), packed_size));
        } else {
          TF_CHECK_OK(local_device->host_to_device_stream()->Memcpy(
              &device_memory, data, packed_size));
        }

        std::shared_ptr<BufferSequencingEvent> event =
            device_buffer->definition_events()[0];
        TF_CHECK_OK(AddDestinationBufferSynchronization(
            local_device, std::move(device_buffer), event,
            local_device->host_to_device_stream()));

        TF_CHECK_OK(local_device->ThenExecuteCallback(
            local_device->host_to_device_stream(),
            [staging_buffer{std::move(staging_buffer)},
             on_done_with_host_buffer{
                 std::move(on_done_with_host_buffer)}]() mutable {
              if (on_done_with_host_buffer) {
                std::move (*on_done_with_host_buffer)();
              }
            }));
      };
  thread_pool()->Schedule(transfer_h2d);
  return std::unique_ptr<PjRtBuffer>(std::move(py_buffer));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtDevice* device) {
  return BufferFromHostBuffer(
      data, type, dims, byte_strides, host_buffer_semantics,
      std::move(on_done_with_host_buffer), device, /*device_layout=*/nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  if (memory_space->devices().size() == 1) {
    return BufferFromHostBuffer(data, type, dims, byte_strides,
                                host_buffer_semantics,
                                std::move(on_done_with_host_buffer),
                                memory_space->devices()[0], device_layout);
  }
  return absl::UnimplementedError(absl::StrCat(
      "BufferFromHostBuffer with PjRtMemorySpace is not implemented on "
      "platform: ",
      platform_name()));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateUninitializedBuffer(const Shape& shape,
                                                    PjRtDevice* device) {
  return CreateUninitializedBuffer(shape, device, nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateUninitializedBuffer(
    const Shape& shape, PjRtDevice* device,
    std::shared_ptr<BufferSequencingEvent> definition_event) {
  tsl::profiler::TraceMe traceme(
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

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateErrorBuffer(absl::Status error,
                                            const Shape& shape,
                                            PjRtMemorySpace* memory) {
  if (memory->client() != this) {
    return absl::InvalidArgumentError(
        "Memory space is not attached to this client");
  }
  auto* device = memory->devices()[0];
  VLOG(1) << "PjRtStreamExecutorClient::CreateErrorBuffer: shape: "
          << shape.ToString() << " device: " << device->DebugString()
          << " error: " << error;

  auto definition_event =
      std::make_shared<BufferSequencingEvent>(this->thread_pool());
  definition_event->SetDefinedStatus(error);

  // Create an empty buffer.
  auto* se_client = tensorflow::down_cast<PjRtStreamExecutorClient*>(this);
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());
  absl::Span<se::DeviceMemoryBase> buffers;
  auto dummy_device_buffer = std::make_shared<TrackedDeviceBuffer>(
      se_client->allocator(), local_device->local_device_id().value(), buffers,
      absl::MakeSpan(&definition_event, 1),
      /*on_delete_callback=*/nullptr);

  auto py_buffer = std::make_unique<PjRtStreamExecutorBuffer>(
      shape, std::move(dummy_device_buffer), this, device,
      device->default_memory_space().value_or(nullptr));
  return py_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                                PjRtDevice* device) {
  tsl::profiler::TraceMe traceme(
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
    // This function uses TF_CHECK_OK and value() since we have no way
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

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                                PjRtMemorySpace* memory_space) {
  if (memory_space->devices().size() == 1) {
    return BufferFromHostLiteral(literal, memory_space->devices()[0]);
  }
  return absl::UnimplementedError(absl::StrCat(
      "BufferFromHostLiteral with PjRtMemorySpace is not implemented on "
      "platform: ",
      platform_name()));
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
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
      std::make_shared<BufferSequencingEvent>(this->thread_pool());
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

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
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
      std::make_shared<BufferSequencingEvent>(this->thread_pool());
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

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtDevice* device,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream) {
  se::DeviceMemoryBase buffer(device_ptr, ShapeUtil::ByteSizeOf(shape));

  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());

  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 2>
      definition_events;
  definition_events.emplace_back(
      std::make_shared<BufferSequencingEvent>(this->thread_pool()));

  se::Stream* definition_stream;
  if (!stream) {
    definition_stream = local_device->compute_stream();
  } else {
    TF_ASSIGN_OR_RETURN(definition_stream,
                        local_device->GetStreamFromExternalStream(*stream));
  }
  TF_ASSIGN_OR_RETURN(
      EventPool::Handle event,
      local_device->event_pool().ThenAllocateAndRecordEvent(definition_stream));
  definition_events.back()->SetSequencingEvent(std::move(event),
                                               definition_stream);

  auto device_buffer = std::make_shared<TrackedDeviceBuffer>(
      /*allocator=*/nullptr, device->local_device_id().value(),
      std::initializer_list<se::DeviceMemoryBase>{buffer}, definition_events,
      std::move(on_delete_callback));
  return std::unique_ptr<PjRtBuffer>(std::make_unique<PjRtStreamExecutorBuffer>(
      shape, std::move(device_buffer), this, device,
      device->default_memory_space().value_or(nullptr)));
}

// Transfer the given literal to the infeed queue of the given local device.
absl::Status PjRtStreamExecutorDevice::TransferToInfeed(
    const LiteralSlice& literal) {
  // Only support infeed to local device.
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device, GetLocalDeviceState());
  return local_device->client()->TransferToInfeedLocal(
      literal, local_device->local_hardware_id().value());
}

absl::Status PjRtStreamExecutorDevice::TransferFromOutfeed(
    MutableBorrowingLiteral literal) {
  VLOG(1) << "PjRtStreamExecutorDevice::TransferFromOutfeed";
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device, GetLocalDeviceState());
  return local_device->client()->TransferFromOutfeedLocal(
      local_device->local_hardware_id().value(), literal);
}

void PjRtStreamExecutorDevice::AttachMemorySpace(
    PjRtMemorySpace* memory_space) {
  CHECK(memory_space != nullptr);
  CHECK(client_ == memory_space->client()) << absl::StrFormat(
      "Could not attach a PjRtStreamExecutorDevice to a PjRtMemorySpace owned "
      "by a different client, the device's client: %s, the memory space's "
      "client: %s.",
      client_->platform_name(), memory_space->client()->platform_name());

  memory_spaces_.push_back(memory_space);
  memory_spaces_by_id_.emplace(memory_space->kind_id(), memory_space);
}

absl::Span<PjRtMemorySpace* const> PjRtStreamExecutorDevice::memory_spaces()
    const {
  return memory_spaces_;
}

absl::StatusOr<PjRtMemorySpace*>
PjRtStreamExecutorDevice::default_memory_space() const {
  return Unimplemented("default_memory_space is not supported.");
}

absl::StatusOr<PjRtMemorySpace*> PjRtStreamExecutorDevice::memory_space_by_kind(
    absl::string_view memory_space_kind) const {
  auto it =
      absl::c_find_if(memory_spaces_, [memory_space_kind](PjRtMemorySpace* ms) {
        return ms->kind() == memory_space_kind;
      });
  if (it != memory_spaces_.end()) {
    return *it;
  }
  return absl::InternalError(
      absl::StrCat("No memory space found (kind: ", memory_space_kind, ")"));
}

absl::StatusOr<PjRtMemorySpace*>
PjRtStreamExecutorDevice::memory_space_by_kind_id(int id) const {
  auto it = memory_spaces_by_id_.find(id);
  if (it == memory_spaces_by_id_.end()) {
    return absl::InternalError(
        absl::StrCat("No memory space found (kind_id: ", id, ")"));
  }
  return it->second;
}

absl::StatusOr<std::intptr_t>
PjRtStreamExecutorDevice::GetStreamForExternalReadyEvents() const {
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device, GetLocalDeviceState());
  se::Stream* stream = local_device->GetExternalReadyEventStream();
  void* raw_stream = stream->platform_specific_handle().stream;
  if (raw_stream == nullptr) {
    return Unimplemented(
        "GetStreamForExternalReadyEvents not implemented for platform '%s'.",
        platform_name());
  }
  return absl::bit_cast<std::intptr_t>(raw_stream);
}

absl::StatusOr<PjRtDevice*> PjRtStreamExecutorClient::LookupAddressableDevice(
    xla::PjRtLocalDeviceId local_device_id) const {
  for (auto* device : addressable_devices_) {
    if (local_device_id == device->local_device_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_device_id %d",
                         local_device_id.value());
}

absl::Span<PjRtMemorySpace* const> PjRtStreamExecutorClient::memory_spaces()
    const {
  return memory_spaces_;
}

PjRtStreamExecutorBuffer::PjRtStreamExecutorBuffer(
    Shape on_device_shape, std::shared_ptr<TrackedDeviceBuffer> device_buffer,
    PjRtClient* client, PjRtDevice* device, PjRtMemorySpace* memory_space)
    : client_(tensorflow::down_cast<PjRtStreamExecutorClient*>(client)),
      on_device_shape_(std::move(on_device_shape)),
      device_(tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)),
      memory_space_(memory_space),
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

absl::StatusOr<std::shared_ptr<TrackedDeviceBuffer>>
PjRtStreamExecutorBuffer::Release(bool wait_for_operations_to_complete) {
  tsl::profiler::TraceMe trace_me("PjRtStreamExecutorBuffer::Release");
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
      se::Stream* block_stream = nullptr;
      for (const auto& stream_and_event : events) {
        VLOG(2)
            << "Checking whether need to wait for stream_and_event: stream: "
            << stream_and_event.stream
            << "; event: " << stream_and_event.event.get()
            << "; reference_held: " << stream_and_event.reference_held
            << "; is_predetermined_error: "
            << stream_and_event.event->IsPredeterminedError();
        // We only need to do something for events that didn't already acquire a
        // reference to the buffer, and also which the compute stream didn't
        // already wait for. Based on our heuristics this rare case should only
        // occur when a buffer was copied to a device and then never used there.
        // In that case we get a new stream and use it to hold onto a reference
        // to the buffer until the events are complete.
        //
        // It is also important that we check IsPredeterminedError before
        // checking DefinedOn(compute_stream) because otherwise DefinedOn would
        // indefinitely wait since the event is never recorded when the buffer
        // is predetermined error.
        if (!stream_and_event.event->IsPredeterminedError() &&
            !stream_and_event.reference_held &&
            !stream_and_event.event->DefinedOn(
                local_device_state->compute_stream()) &&
            !stream_and_event.event->IsComplete()) {
          if (block_stream == nullptr) {
            block_stream = local_device_state->GetFixedSizePoolUsageStream();
          }
          VLOG(2) << "Waiting for stream_and_event: stream: "
                  << stream_and_event.stream
                  << "; event: " << stream_and_event.event.get()
                  << "; reference_held: " << stream_and_event.reference_held
                  << "; is_predetermined_error: "
                  << stream_and_event.event->IsPredeterminedError();
          stream_and_event.event->WaitForEventOnStream(block_stream);
        }
      }
      for (const auto& definition_event : device_buffer->definition_events()) {
        VLOG(2) << "Checking whether need to wait for definition_event: "
                << definition_event.get() << "; is_predetermined_error: "
                << definition_event->IsPredeterminedError();
        // Here we wait for the definition events to complete on block_stream as
        // well, if they are not on the compute stream and not also recorded as
        // usage events.
        //
        // It is also important that we check IsPredeterminedError before
        // checking DefinedOn(compute_stream) because otherwise DefinedOn would
        // indefinitely wait since the event is never recorded when the buffer
        // is predetermined error.
        //
        // Since it's possible that definition_event.SetSequencingEvent()
        // is called on a different host thread than this host thread, when in
        // future more conditions are added to this check, we should be careful
        // about whether we put them before the DefinedOn check or after it.
        // For example, we shouldn't add an IsDefined() check before the
        // DefinedOn() check here because that could potentially cause a
        // shortcut where we don't wait for
        // definition_event.SetSequencingEvent() on the other thread and
        // eventually cause memory corruption.
        if (!definition_event->IsPredeterminedError() &&
            !definition_event->DefinedOn(
                local_device_state->compute_stream()) &&
            !definition_event->IsComplete()) {
          if (block_stream == nullptr) {
            block_stream = local_device_state->GetFixedSizePoolUsageStream();
          }
          VLOG(2) << "Waiting for definition_event: " << definition_event.get()
                  << "; is_predetermined_error: "
                  << definition_event->IsPredeterminedError();
          definition_event->WaitForEventOnStream(block_stream);
        }
      }
      if (block_stream != nullptr) {
        TF_RETURN_IF_ERROR(local_device_state->ThenExecuteCallback(
            block_stream, [device_buffer]() {
              // Drops device_buffer shared pointer.
            }));
      }
    }
  }
  return device_buffer;
}

void PjRtStreamExecutorBuffer::Delete() {
  VLOG(1) << "PjRtStreamExecutorBuffer::Delete";

  // When wait_for_reads_to_complete is false, Release should never fail.
  //
  // The only usage events that
  // Release(/*wait_for_operations_to_complete=*/false) doesn't wait for are
  // events defined on the compute stream. All streams other than the compute
  // stream are expected to WaitFor compute stream before any write operations.
  TF_CHECK_OK(Release(/*wait_for_operations_to_complete=*/false).status());
}

bool PjRtStreamExecutorBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return device_buffer_ == nullptr;
}

absl::StatusOr<std::shared_ptr<TrackedDeviceBuffer>>
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

PjRtFuture<> PjRtStreamExecutorBuffer::LazyToLiteral(
    absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator) {
  auto buffer = std::move(generator)();
  if (!buffer.ok()) {
    return PjRtFuture<>(buffer.status());
  }
  return ToLiteral(buffer.value());
}

PjRtFuture<> PjRtStreamExecutorBuffer::ToLiteral(MutableLiteralBase* literal) {
  VLOG(1) << "PjRtStreamExecutorBuffer::ToLiteral";
  if (IsEmptyTuple()) {
    return PjRtFuture<>(InvalidArgument("ToLiteral called on empty tuple"));
  }
  LocalDeviceState* local_device = device_->local_device_state();
  se::Stream* stream = local_device->GetDeviceToHostStream();
  ScopedHold device_buffer(this, ScopedHold::kUsage);
  {
    absl::MutexLock lock(&mu_);
    // We can't perform any other action while a donation hold is in progress.
    WaitForOutstandingDonationHold();
    if (device_buffer_ == nullptr) {
      return PjRtFuture<>(InvalidArgument(
          "CopyToHostAsync() called on deleted or donated buffer"));
    }
    AcquireHoldLocked(&device_buffer);
  }

  auto promise = PjRtFuture<>::CreatePromise();
  auto usage_event =
      std::make_shared<BufferSequencingEvent>(client_->thread_pool());

  TransferManager* transfer_manager =
      client_->client()->backend().transfer_manager();

  auto tracked_device_buffer = device_buffer.buffer();

  // When using the ComputeSynchronized allocation model, retain a
  // reference to the device_buffer until the copy completes, to
  // ensure that the buffer isn't deleted or donated while it is still
  // in use. The choice of retaining a reference at the host is a
  // heuristic; the alternative is to ensure, before freeing the
  // buffer, that the compute stream is synchronized past the
  // transfer, but it seems better to hold onto the buffer too long
  // than to stall the compute stream, particularly since the
  // overwhelmingly common use case of CopyToHostAsync will hold onto
  // the reference long enough to read the buffer in a subsequent call
  // to ToLiteral.
  device_buffer.ConvertUsageHold(stream, usage_event, /*reference_held=*/true);

  auto async_to_literal = [usage_event, tracked_device_buffer, stream,
                           transfer_manager = std::move(transfer_manager),
                           on_device_shape{on_device_shape_}, literal, promise,
                           local_device]() mutable {
    absl::StatusOr<EventPool::Handle> event_or =
        local_device->event_pool().AllocateEvent(stream->parent());
    if (!event_or.ok()) {
      promise.Set(event_or.status());
      return;
    }

    absl::Status defined_status =
        tracked_device_buffer->definition_events()[0]->GetDefinedStatus();
    if (!defined_status.ok()) {
      promise.Set(defined_status);
      return;
    }

    WaitForBufferDefinitionEventsOnStream(*tracked_device_buffer, stream);
    ShapedBuffer shaped_buffer =
        tracked_device_buffer->AsShapedBuffer(on_device_shape);

    GenericTransferManager::LiteralFromDeviceMetadata transfer_metadata;
    // We never call device functions from the `done` callback.
    transfer_metadata.callback_is_host_callback_safe = true;

    TransferManager::TransferMetadata* transfer_metadata_ptr =
        (dynamic_cast<GenericTransferManager*>(transfer_manager) != nullptr)
            ? &transfer_metadata
            : nullptr;

    transfer_manager->TransferLiteralFromDevice(
        stream, shaped_buffer, literal,
        [promise](absl::Status status) mutable {
          promise.Set(std::move(status));
        },
        transfer_metadata_ptr);

    local_device->event_pool().ThenRecordEvent(stream, event_or.value());
    usage_event->SetSequencingEvent(std::move(event_or).value(), stream);

    defined_status = local_device->ThenRelease(stream, tracked_device_buffer);
    if (!defined_status.ok()) {
      promise.Set(defined_status);
    }
  };

  tracked_device_buffer->definition_events()[0]->ExecuteOrAddToFutureTasks(
      absl::StrFormat("async_to_literal_%p", literal),
      std::move(async_to_literal));

  return PjRtFuture<>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "PjRtStreamExecutorBuffer::ToLiteral");
        VLOG(1) << "PjRtStreamExecutorBuffer::ToLiteral";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            "PjRtStreamExecutorBuffer::ToLiteral", keys.traceme_context_id);
      });
}

absl::StatusOr<size_t> PjRtStreamExecutorBuffer::GetOnDeviceSizeInBytes()
    const {
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

PjRtFuture<> PjRtStreamExecutorBuffer::CopyRawToHost(void* dst, int64_t offset,
                                                     int64_t transfer_size) {
  return client_->CopyRawSubBufferToHost(this, PjRtFuture<void*>(dst), offset,
                                         transfer_size);
}

PjRtFuture<> PjRtStreamExecutorBuffer::CopyRawToHostFuture(
    PjRtFuture<void*> dst, int64_t offset, int64_t transfer_size) {
  return client_->CopyRawSubBufferToHost(this, dst, offset, transfer_size);
}

absl::StatusOr<ShapedBuffer> PjRtStreamExecutorBuffer::AsShapedBuffer() const {
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

absl::StatusOr<std::pair<std::unique_ptr<PjRtBuffer>,
                         std::shared_ptr<BufferSequencingEvent>>>
PjRtStreamExecutorBuffer::CopyToDeviceHelper(
    PjRtDevice* dst_device, LocalDeviceState* dst_local_device,
    LocalDeviceState* transfer_local_device, LocalDeviceState* src_local_device,
    se::Stream* transfer_stream,
    std::shared_ptr<TrackedDeviceBuffer> src_device_buffer) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtStreamExecutorBuffer> py_buffer,
                      AllocateDestinationBuffer(
                          ShapeUtil::DeviceShapeToHostShape(on_device_shape_),
                          dst_device, dst_local_device, transfer_stream,
                          /*is_uninitialized_create=*/false, client_));

  ScopedHold dst_device_buffer(py_buffer->GetBufferWithUsageHold());
  CHECK(dst_device_buffer.ok());

  std::shared_ptr<BufferSequencingEvent> copy_event =
      dst_device_buffer->definition_events()[0];

  // Copy the leaf buffers.
  auto async_copy_to_device = [src_device_buffer,
                               dst_device_buffer =
                                   std::move(dst_device_buffer.buffer()),
                               transfer_stream = std::move(transfer_stream),
                               copy_event,
                               on_device_shape{py_buffer->on_device_shape()},
                               src_local_device = std::move(src_local_device),
                               transfer_local_device =
                                   std::move(transfer_local_device),
                               dst_local_device =
                                   std::move(dst_local_device)]() mutable {
    tsl::profiler::TraceMe traceme(
        "PjRtStreamExecutorBuffer::CopyToDeviceHelper::async_copy_to_"
        "device");
    VLOG(1)
        << "PjRtStreamExecutorBuffer::CopyToDeviceHelper::async_copy_to_device";

    absl::Status defined_status =
        src_device_buffer->definition_events()[0]->GetDefinedStatus();
    // Only proceeds to transfer when the buffer doesn't hold an error.
    if (defined_status.ok()) {
      WaitForBufferDefinitionEventsOnStream(*src_device_buffer,
                                            transfer_stream);

      ShapedBuffer src_buffer =
          src_device_buffer->AsShapedBuffer(on_device_shape);

      ShapedBuffer dst_buffer =
          dst_device_buffer->AsShapedBuffer(on_device_shape);
      for (const auto& leaf : src_buffer.buffers().leaves()) {
        const ShapeIndex& index = leaf.first;
        const se::DeviceMemoryBase& input_buffer = leaf.second;
        const se::DeviceMemoryBase& output_buffer = dst_buffer.buffer(index);
        CHECK_EQ(input_buffer.size(), output_buffer.size());
        if (input_buffer.size() != 0) {
          auto status = transfer_local_device->ThenMemcpyDeviceToDevice(
              transfer_stream, dst_local_device->compute_stream(), input_buffer,
              output_buffer);
          if (!status.ok()) {
            LOG(ERROR) << "D2D memory copy failed due to: " << status;
            StallStreamOnError(transfer_local_device, transfer_stream);
            if (transfer_local_device == dst_local_device) {
              // Some copies may have been enqueued before the error was
              // returned, and StallStreamOnError only makes sure the
              // destination device is ok, so make sure that the src buffer
              // remains valid until after any transfers have completed.
              auto status = src_local_device->ThenRelease(
                  transfer_stream, std::move(src_device_buffer));
              if (!status.ok()) {
                LOG(ERROR) << "ThenRelease failed due to: " << status;
              }
            }
            return;
          }
        }
      }

      absl::StatusOr<EventPool::Handle> event_or =
          transfer_local_device->event_pool().ThenAllocateAndRecordEvent(
              transfer_stream);
      if (!event_or.ok()) {
        StallStreamOnError(transfer_local_device, transfer_stream);
        LOG(ERROR) << event_or.status();
        return;
      }
      copy_event->SetSequencingEvent(std::move(event_or).value(),
                                     transfer_stream);
    } else {
      copy_event->SetDefinedStatus(defined_status);
    }

    auto status = src_local_device->ThenRelease(transfer_stream,
                                                std::move(src_device_buffer));
    if (!status.ok()) {
      LOG(ERROR) << "ThenRelease failed due to: " << status;
    }
  };

  src_device_buffer->definition_events()[0]->ExecuteOrAddToFutureTasks(
      absl::StrFormat("async_copy_to_device_%p",
                      dst_device_buffer.buffer().get()),
      std::move(async_copy_to_device));

  RecordUsage(std::move(dst_device_buffer), transfer_local_device,
              transfer_local_device, copy_event, transfer_stream,
              /*prefer_to_retain_reference=*/false);

  return std::pair<std::unique_ptr<PjRtBuffer>,
                   std::shared_ptr<BufferSequencingEvent>>(
      std::unique_ptr<PjRtStreamExecutorBuffer>(std::move(py_buffer)),
      std::move(copy_event));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorBuffer::CopyToDevice(PjRtDevice* dst_device) {
  tsl::profiler::TraceMe traceme("PjRtStreamExecutorBuffer::CopyToDevice");
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
        PjRtStreamExecutorClient::HostBufferSemantics::kImmutableZeroCopy,
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

  absl::StatusOr<std::pair<std::unique_ptr<PjRtBuffer>,
                           std::shared_ptr<BufferSequencingEvent>>>
      buffer_and_event_or = CopyToDeviceHelper(
          dst_device, dst_local_device, transfer_local_device,
          device_->local_device_state(), transfer_stream,
          src_device_buffer.buffer());
  if (!buffer_and_event_or.ok()) {
    return buffer_and_event_or.status();
  }

  auto& buffer_and_event = buffer_and_event_or.value();
  std::unique_ptr<PjRtBuffer>& buffer = buffer_and_event.first;
  std::shared_ptr<BufferSequencingEvent>& event = buffer_and_event.second;

  // prefer_to_retain_reference=*/true means that, when using the
  // ComputeSynchronized allocation model, retain a reference to the
  // src_device_buffer until the copy completes. This is a heuristic; the
  // alternative is to ensure, before freeing the buffer, that the compute
  // stream is synchronized past the transfer, but it seems better to hold onto
  // the buffer too long than to stall the compute stream.
  src_device_buffer.ConvertUsageHold(transfer_stream, event,
                                     /*reference_held=*/true);

  return std::move(buffer);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorBuffer::CopyToMemorySpace(PjRtMemorySpace* dst_memory_space) {
  if (dst_memory_space->devices().size() == 1) {
    return CopyToDevice(dst_memory_space->devices()[0]);
  }
  return Unimplemented("CopyToMemorySpace is not supported");
}

void PjRtStreamExecutorBuffer::CopyToRemoteDevice(
    PjRtFuture<std::string> serialized_descriptor, RemoteSendCallback on_done) {
  VLOG(1) << "PjRtStreamExecutorBuffer::CopyToRemoteDevice";
  auto desc = serialized_descriptor.Await();
  if (desc.ok()) {
    client_->CopyToRemoteDevice(this, *desc, std::move(on_done));
  } else {
    on_done(desc.status(), /*sends_enqueued=*/false);
  }
}

void PjRtStreamExecutorBuffer::CopyToRemoteDeviceScattered(
    PjRtFuture<std::vector<std::string>> serialized_descriptors,
    std::vector<RemoteSendCallback> callbacks,
    const ScatterDetails& scatter_details) {
  VLOG(1) << "PjRtStreamExecutorBuffer::CopyToRemoteDeviceScattered";
  auto res = serialized_descriptors.Await();
  if (res.ok()) {
    client_->CopyToRemoteDeviceScattered(this, *std::move(res),
                                         std::move(callbacks), scatter_details);
  } else {
    for (const auto& cb : callbacks) {
      cb(res.status(), /*sends_enqueued=*/false);
    }
  }
}

PjRtFuture<> PjRtStreamExecutorBuffer::GetReadyFuture() {
  std::shared_ptr<TrackedDeviceBuffer> device_buffer;
  PjRtFuture<>::Promise definition_promise;
  {
    absl::MutexLock lock(&mu_);
    if (device_buffer_ == nullptr) {
      return PjRtFuture<>(InvalidArgument(
          "GetReadyFuture() called on deleted or donated buffer"));
    }
    if (!definition_promise_) {
      device_buffer = device_buffer_;
      definition_promise_ = PjRtFuture<>::CreatePromise();
    }
    definition_promise = definition_promise_;
  }

  if (device_buffer) {
    LocalDeviceState* local_device_state = device_->local_device_state();
    auto async_wait_for_events =
        [device_buffer, local_device_state = std::move(local_device_state),
         definition_promise]() mutable {
          std::unique_ptr<se::Stream> stream;
          absl::Status defined_status =
              device_buffer->definition_events()[0]->GetDefinedStatus();
          if (!defined_status.ok()) {
            definition_promise.Set(defined_status);
            return;
          }
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
            // We already borrowed a stream from the pool so we can safely do
            // the callback directly on that stream instead of bouncing through
            // local_device_state->ThenExecuteCallback. The direct callback
            // saves significant time.
            auto status = stream_ptr->DoHostCallback(
                [definition_promise, stream_ptr, local_device_state,
                 event_with_status =
                     device_buffer->definition_events()[0]]() mutable {
                  local_device_state->ReturnStreamToPool(
                      std::unique_ptr<se::Stream>(stream_ptr));
                  definition_promise.Set(event_with_status->GetDefinedStatus());
                });
            if (!status.ok()) {
              definition_promise.Set(status);
              return;
            }
          } else {
            // All events are already complete; set the `definition_promise`
            // with the status of the buffer's first definition event which may
            // have error status to propagate.
            definition_promise.Set(
                device_buffer->definition_events()[0]->GetDefinedStatus());
          }
        };
    device_buffer->definition_events()[0]->ExecuteOrAddToFutureTasks(
        absl::StrFormat("async_wait_for_events_%p", &async_wait_for_events),
        std::move(async_wait_for_events));
  }

  return PjRtFuture<>(
      std::move(definition_promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "PjRtStreamExecutorBuffer::Await");
        VLOG(1) << "PjRtStreamExecutorBuffer::Await";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id=*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
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

absl::Status CheckCompatibleShapes(bool strict_shape_checking,
                                   const Shape& buffer_on_device_shape,
                                   const Shape& execution_shape,
                                   const TransferManager& transfer_manager,
                                   int parameter_index) {
  // Handle the special case: the underlying pjrt buffer of a JAX token may have
  // shape `pred[0]`.
  if (execution_shape.IsToken() &&
      buffer_on_device_shape.element_type() == PrimitiveType::PRED &&
      buffer_on_device_shape.dimensions_size() == 1 &&
      buffer_on_device_shape.dimensions(0) == 0) {
    return absl::OkStatus();
  }
  // TODO(misard) Support casting of tuple parameters.
  if (strict_shape_checking || buffer_on_device_shape.IsTuple()) {
    if (!ShapeUtil::Compatible(buffer_on_device_shape, execution_shape)) {
      return InvalidArgument(
          "Executable expected shape %s for argument %d but got "
          "incompatible "
          "shape %s",
          ShapeUtil::HumanStringWithLayout(execution_shape), parameter_index,
          ShapeUtil::HumanStringWithLayout(buffer_on_device_shape));
    }
  } else {
    const int64_t buffer_size =
        transfer_manager.GetByteSizeRequirement(buffer_on_device_shape);
    const int64_t execute_size =
        transfer_manager.GetByteSizeRequirement(execution_shape);
    if (buffer_on_device_shape.is_static() && buffer_size != execute_size) {
      return InvalidArgument(
          "Executable expected shape %s for argument %d but got "
          "incompatible "
          "shape %s",
          ShapeUtil::HumanStringWithLayout(execution_shape), parameter_index,
          ShapeUtil::HumanStringWithLayout(buffer_on_device_shape));
    }
    if (!buffer_on_device_shape.is_static() && buffer_size < execute_size) {
      return InvalidArgument(
          "Executable expected shape %s for argument %d but got "
          "incompatible "
          "shape %s",
          ShapeUtil::HumanStringWithLayout(execution_shape), parameter_index,
          ShapeUtil::HumanStringWithLayout(buffer_on_device_shape));
    }
  }
  return absl::OkStatus();
}

// Makes a tuple from the arguments to an execution.
absl::StatusOr<TupleHandle> MakeTupleHelper(
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
    TF_RETURN_IF_ERROR(stream->WaitFor(local_device->compute_stream()));
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
  absl::StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().ThenAllocateAndRecordEvent(stream);
  if (!event_or.ok()) {
    StallStreamOnError(local_device, stream);
    return event_or.status();
  }

  auto transfer_event =
      std::make_shared<BufferSequencingEvent>(client->thread_pool());
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
      result_buffer->on_device_shape(), std::move(out_buffer), client, device,
      device->default_memory_space().value_or(nullptr));
  RecordUsage(pjrt_buffer->GetBufferWithUsageHold(), local_device, local_device,
              definition_event, local_device->compute_stream(),
              /*prefer_to_retain_reference=*/false, &buffers_to_release);
  return std::unique_ptr<PjRtBuffer>(std::move(pjrt_buffer));
}

bool IsAllZeros(const DeviceAssignment& assignment) {
  return std::all_of(
      assignment.begin(), assignment.end(),
      [](const DeviceAssignment::value_type& v) { return v == 0; });
}

}  // namespace

PjRtStreamExecutorLoadedExecutable::PjRtStreamExecutorLoadedExecutable(
    std::vector<std::unique_ptr<LocalExecutable>> executables,
    bool parameter_is_tupled_arguments,
    std::shared_ptr<DeviceAssignment> device_assignment,
    CompileOptions compile_options,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices,
    PjRtStreamExecutorClient* client)
    : client_(client),
      device_assignment_(std::move(device_assignment)),
      compile_options_(std::move(compile_options)),
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
    VLOG(3) << "PjRtStreamExecutorLoadedExecutable portable single-core";
    num_partitions = 1;
    CHECK(addressable_devices_.empty());
  } else {
    // This must go after `executables_` is initialized.
    VLOG(3) << "PjRtStreamExecutorLoadedExecutable device_assignment:\n"
            << device_assignment_->ToString();
    CHECK_GE(addressable_devices_.size(), 1) << device_assignment_->ToString();

    if ((device_assignment_->replica_count() > 1 ||
         device_assignment_->computation_count() > 1) &&
        IsAllZeros(*device_assignment_)) {
      // This code path should only be triggered when we intentionally compile
      // an HLO without having enough devices to actually run it. See the
      // "--run=false" option in
      // tensorflow/compiler/xla/tools/multihost_hlo_runner/hlo_runner_main.cc.
      // That will help us debug the XLA compiler locally.
      LOG(INFO)
          << "A workaround is in effect to allow compiling multi-device "
             "HLOs on machines with fewer devices. Don't run this executable.";
    } else {
      CHECK_LE(addressable_devices_.size(), client_->addressable_device_count())
          << "Inconsistent local device count.";
    }

    num_partitions = device_assignment_->computation_count();
  }

  // SPMD sharding produces a single executable for multiple partitions.
  if (executables_.size() > 1) {
    CHECK_EQ(num_partitions, executables_.size())
        << "Number of executables " << executables_.size()
        << " did not match number of partitions " << num_partitions;
  }
}

absl::Status PjRtStreamExecutorLoadedExecutable::SetUpDonation(
    bool tuple_inputs) {
  parameters_that_must_be_donated_.reserve(executables_.size());
  for (auto& executable : executables_) {
    TF_ASSIGN_OR_RETURN(std::vector<int> parameters_to_donate,
                        ComputeParametersThatMustBeDonated(
                            executable->executable()->module(), tuple_inputs));
    parameters_that_must_be_donated_.emplace_back(
        std::move(parameters_to_donate));
  }
  return absl::OkStatus();
}

absl::string_view PjRtStreamExecutorLoadedExecutable::name() const {
  Executable* executable = executables_[0]->executable();
  if (executable->has_module()) {
    return executable->module().name();
  } else {
    return "<unknown executable>";
  }
}

absl::Span<int const>
PjRtStreamExecutorLoadedExecutable::ParametersThatMustBeDonated(
    int executable_idx) const {
  return parameters_that_must_be_donated_[executable_idx];
}

absl::StatusOr<std::vector<ExecutionInput>>
PjRtStreamExecutorLoadedExecutable::MakeExecutionInputsAndWaitForEvents(
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

template <typename T>
static const T* FindCallback(int channel_id, absl::Span<const T> callbacks) {
  // TODO(ezhulenev): Can we use binary search here assuming that callbacks
  // are sorted by channel id? Are they always sorted?
  auto it = absl::c_find_if(callbacks, [&](const T& callback) {
    return callback.channel_id == channel_id;
  });
  return it == callbacks.end() ? nullptr : &*it;
}

using tsl::AsyncValueRef;
using tsl::MakeConstructedAsyncValueRef;

// Converts PjRt SendCallbacks to an XLA StreamExecutor send function.
static SendDeviceMemoryFunction ConvertSendCallbacksToSendFunction(
    int replica, const ExecuteOptions& options,
    tsl::thread::ThreadPool* thread_pool) {
  // Check if we have callbacks registered for the given replica.
  if (replica >= options.send_callbacks.size()) {
    return [replica](int64_t channel_id, se::Stream*, const Shape&,
                     const se::DeviceMemoryBase&,
                     const absl::flat_hash_map<std::string, std::string>&) {
      return Internal(
          "Don't send a buffer to the channel_id=%d, there was no send "
          "callbacks registered for the replica=%d",
          channel_id, replica);
    };
  }

  // SendCallbacks registered for a device ordinal. Can be empty.
  absl::Span<const SendCallback> callbacks = options.send_callbacks[replica];

  return [callbacks, thread_pool](
             int64_t channel_id, se::Stream* stream, const Shape& shape,
             const se::DeviceMemoryBase& src,
             const absl::flat_hash_map<std::string, std::string>&)
             -> absl::StatusOr<AsyncValueRef<std::unique_ptr<se::Event>>> {
    VLOG(3) << "Send " << src.size() << " bytes to channel #" << channel_id
            << " (shape=" << shape.ToString() << ")";

    const SendCallback* send = FindCallback(channel_id, callbacks);
    if (!send) {
      return InvalidArgument(
          "Failed to send a buffer to the channel_id=%d, callback not found",
          channel_id);
    }

    // Allocate event that will signal completion of send operation. We do not
    // actually track the completion of the send callback, we only have to keep
    // the device memory long enough to complete the memcpy command.
    TF_ASSIGN_OR_RETURN(auto se_event, stream->parent()->CreateEvent());
    auto done_event = MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
        std::move(se_event));

    thread_pool->Schedule([done_event, stream, src, channel_id, shape, send] {
      tsl::profiler::TraceMe trace([&] {
        return tsl::profiler::TraceMeEncode(
            "PjRtStreamExecutorLoadedExecutable::Send",
            {{"channel_id", channel_id}});
      });

      // Allocate chunk on the host for copying data from device.
      PjRtChunk chunk = PjRtChunk::AllocateDefault(src.size());

      auto status = stream->Memcpy(chunk.data(), src, src.size());
      if (!status.ok()) {
        done_event.SetError(status);
        return;
      }
      status = stream->RecordEvent(done_event.get().get());
      if (!status.ok()) {
        done_event.SetError(status);
        return;
      }

      // Wait for the data to be available on the host.
      if (auto st = stream->BlockHostUntilDone(); !st.ok()) {
        done_event.SetError(absl::InternalError(absl::StrFormat(
            "failed to synchronize send operation with a stream: %s",
            st.message())));
        return;
      }

      // Pass chunk to the registered callback.
      auto sent = send->callback({shape}, std::move(chunk),
                                 /*total_size_in_bytes=*/src.size(),
                                 /*done=*/true);

      if (!sent.ok()) {
        done_event.SetError(sent);
      } else {
        done_event.SetStateConcrete();
      }
    });

    return std::move(done_event);
  };
}

namespace {
class StreamExecutorCopyToDeviceStream : public CopyToDeviceStream {
 public:
  StreamExecutorCopyToDeviceStream(
      int64_t channel_id, se::Stream* stream, se::DeviceMemoryBase dst,
      AsyncValueRef<std::unique_ptr<se::Event>> done)
      : CopyToDeviceStream(dst.size(), /*granule_bytes=*/1),
        channel_id_(channel_id),
        stream_(stream),
        dst_(dst),
        done_(std::move(done)) {}

  PjRtFuture<> AddChunk(PjRtChunk chunk) final {
    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode(
          "StreamExecutorCopyToDeviceStream::AddChunk",
          {{"channel_id", channel_id_}});
    });

    absl::ReleasableMutexLock lock(&mu_);

    VLOG(3) << "Add chunk to a H2D channel #" << channel_id_ << ": "
            << "size=" << chunk.size() << ", "
            << "current_bytes=" << current_bytes_ << ", "
            << "total_bytes=" << total_bytes_;

    if (chunk.size() % granule_size_in_bytes() != 0) {
      done_.SetError(absl::InvalidArgumentError(absl::StrFormat(
          "Chunk size (%d) was not a multiple of the granule size (%d)",
          chunk.size(), granule_size_in_bytes())));
      return PjRtFuture<>(done_.GetError());
    }

    if (current_bytes_ + chunk.size() > total_bytes_) {
      done_.SetError(absl::InvalidArgumentError(
          absl::StrFormat("Adding chunk of size %d would overflow buffer of "
                          "size %d (%d already transferred)",
                          chunk.size(), total_bytes_, current_bytes_)));
      return PjRtFuture<>(done_.GetError());
    }

    se::DeviceMemoryBase dst(
        reinterpret_cast<std::byte*>(dst_.opaque()) + current_bytes_,
        dst_.size() - current_bytes_);

    current_bytes_ += chunk.size();
    bool complete = IsCompleteLocked();
    lock.Release();

    auto copied = stream_->Memcpy(&dst, chunk.data(), chunk.size());
    if (!copied.ok()) {
      done_.SetError(copied);
      return PjRtFuture<>(done_.GetError());
    }

    // Delete chunk once the memcpy operation completes.
    auto* chunk_ptr = std::make_unique<PjRtChunk>(std::move(chunk)).release();
    auto deleted = stream_->DoHostCallback([chunk_ptr]() { delete chunk_ptr; });
    if (!deleted.ok()) {
      done_.SetError(deleted);
      return PjRtFuture<>(done_.GetError());
    }

    // Record done event once processed the last chunk. It is the caller
    // responsibility to synchronize with this event before submitting any new
    // computations to the stream.
    if (complete) {
      auto recorded = stream_->RecordEvent(done_.get().get());
      if (!recorded.ok()) {
        done_.SetError(recorded);
        return PjRtFuture<>(done_.GetError());
      }
      done_.SetStateConcrete();
    }

    return PjRtFuture<>(absl::OkStatus());
  }

 private:
  int64_t channel_id_;
  se::Stream* stream_;
  se::DeviceMemoryBase dst_;

  // Async value will become available after we'll submit the last memcpy
  // operation, and the event will be recorded on the stream.
  AsyncValueRef<std::unique_ptr<se::Event>> done_;
};
}  // namespace

static RecvDeviceMemoryFunction ConvertRecvCallbacksToRecvFunction(
    int replica, const ExecuteOptions& options) {
  // Check if we have callbacks registered for the given replica.
  if (replica >= options.send_callbacks.size()) {
    return [replica](int64_t channel_id, se::Stream*, const Shape&,
                     se::DeviceMemoryBase*,
                     const absl::flat_hash_map<std::string, std::string>&) {
      return InvalidArgument(
          "Failed to receive a buffer from the channel_id=%d, there was no "
          "recv callbacks registered for the replica=%d",
          channel_id, replica);
    };
  }

  // RecvCallbacks registered for a device ordinal. Can be empty.
  absl::Span<const RecvCallback> callbacks = options.recv_callbacks[replica];

  return [callbacks](int64_t channel_id, se::Stream* stream, const Shape& shape,
                     se::DeviceMemoryBase* dst,
                     const absl::flat_hash_map<std::string, std::string>&)
             -> absl::StatusOr<AsyncValueRef<std::unique_ptr<se::Event>>> {
    VLOG(3) << "Recv from channel #" << channel_id
            << " (shape=" << shape.ToString() << ")";

    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode(
          "PjRtStreamExecutorLoadedExecutable::Recv",
          {{"channel_id", channel_id}});
    });

    const RecvCallback* recv = FindCallback(channel_id, callbacks);
    if (!recv) {
      return InvalidArgument(
          "Failed to recv a buffer from the channel_id=%d, callback not found",
          channel_id);
    }

    // Allocate event that will signal completion of recv operation. We record
    // it on a stream after submitting the memcpy for the last chunk (see
    // `StreamExecutorCopyToDeviceStream` implementation above).
    TF_ASSIGN_OR_RETURN(auto event, stream->parent()->CreateEvent());
    auto done_event = MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
        std::move(event));

    recv->callback({shape}, std::make_unique<StreamExecutorCopyToDeviceStream>(
                                channel_id, stream, *dst, done_event));

    return std::move(done_event);
  };
}

// Enqueues a computation onto the compute stream. Each buffer returned in
// device_buffers has a usage hold added that must be dropped on error or
// converted on success.
// When `options` has non-zero `launch_id`, use `launch_id` instead of `run_id`
// to initialize `run_options`.
absl::StatusOr<ScopedShapedBuffer>
PjRtStreamExecutorLoadedExecutable::EnqueueExecution(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    int executable_idx, const RunId& run_id, const ExecuteOptions& options,
    PjRtDevice* device,
    std::vector<PjRtStreamExecutorBuffer::ScopedHold>* device_buffers,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<std::function<void()>>& compute_callbacks) const {
  int device_ordinal = tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                           ->local_device_state()
                           ->local_device_id()
                           .value();
  LocalDeviceState* device_state = &(client_->device_state(device_ordinal));
  tsl::profiler::TraceMeConsumer activity(
      "PjRtStreamExecutorLoadedExecutable::EnqueueExecution",
      tsl::profiler::ContextType::kPjRt, run_id.ToInt());
  VLOG(3) << "Replica " << replica << ", partition " << partition
          << " mapped to device ordinal for execution: " << device_ordinal;

  absl::flat_hash_set<BufferSequencingEvent*> events;
  device_buffers->reserve(argument_handles.size());
  absl::Span<int const> donated_params =
      ParametersThatMustBeDonated(executable_idx);
  auto donate_it = donated_params.begin();
  absl::flat_hash_set<PjRtStreamExecutorBuffer*> used_buffers;
  absl::flat_hash_set<PjRtStreamExecutorBuffer*> donated_buffers;
  for (int i = 0; i < argument_handles.size(); ++i) {
    auto* handle =
        tensorflow::down_cast<PjRtStreamExecutorBuffer*>(argument_handles[i]);
    if (handle->device() != device) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %s, but replica is assigned to device %s.",
          i, replica, handle->device()->DebugString(), device->DebugString());
    }
    bool donation_denied_at_runtime =
        options.non_donatable_input_indices.contains(i);
    bool must_donate = donate_it != donated_params.end() && *donate_it == i &&
                       !donation_denied_at_runtime;
    if (must_donate) {
      ++donate_it;
    }
    bool already_used = !used_buffers.emplace(handle).second;
    bool already_donated =
        must_donate ? !donated_buffers.emplace(handle).second
                    : donated_buffers.find(handle) != donated_buffers.end();
    if (must_donate && already_donated) {
      return InvalidArgument(
          "Attempt to donate the same buffer twice in Execute() (second use: "
          "flattened argument %d, replica %d). "
          "Toy example for this bug: `f(donate(a), donate(a))`.",
          i, replica);
    } else if (must_donate && already_used) {
      return InvalidArgument(
          "Attempt to donate a buffer which is also used by the same call to "
          "Execute() (second use: flattened argument %d, replica %d). "
          "Toy example for this bug: `f(a, donate(a))`.",
          i, replica);
    } else if (already_donated) {
      return InvalidArgument(
          "Attempt to use a buffer that was previously donated in the same "
          "call to Execute() (second use: flattened argument %d, replica %d). "
          "Toy example for this bug: `f(donate(a), a)`.",
          i, replica);
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
    if (device_state->allocation_model() ==
        LocalDeviceState::kComputeSynchronized) {
      GetDeviceBufferEvents(*device_buffer, /*get_usage_events=*/false,
                            &events);
    }

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

  // Schedule async send operations in the client thread pool.
  auto* thread_pool = client_->thread_pool();

  // Create a PjRt<->StreamExecutor adaptors to send/recv device memory as
  // PjRt chunks via the user-provided callbacks.
  SendDeviceMemoryFunction send_device_memory =
      ConvertSendCallbacksToSendFunction(replica, options, thread_pool);
  RecvDeviceMemoryFunction recv_device_memory =
      ConvertRecvCallbacksToRecvFunction(replica, options);

  ExecutableRunOptions run_options;
  run_options.set_stream(device_state->compute_stream());
  run_options.set_host_to_device_stream(device_state->host_to_device_stream());
  run_options.set_device_to_host_stream(device_state->GetDeviceToHostStream());
  run_options.set_allocator(client_->allocator());
  run_options.set_intra_op_thread_pool(
      client_->client()->backend().eigen_intra_op_thread_pool_device());
  run_options.set_device_assignment(device_assignment.get());
  if (options.launch_id != 0) {
    run_options.set_run_id(RunId(options.launch_id));
  } else {
    run_options.set_run_id(run_id);
  }
  run_options.set_rng_seed(device_state->GetNewPrngSeed());
  run_options.set_gpu_executable_run_options(client_->gpu_run_options());
  run_options.set_launch_id(options.launch_id);
  run_options.set_send_device_memory_function(&send_device_memory);
  run_options.set_recv_device_memory_function(&recv_device_memory);
  if (run_options.launch_id() != 0) {
    VLOG(3) << "launch id for " << name() << ": " << run_options.launch_id();
  }
  if (options.context != nullptr) {
    run_options.set_ffi_execution_context(&options.context->ffi_context());
  }
  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  std::shared_ptr<Semaphore::ScopedReservation> compute_reservation;
  {
    tsl::profiler::TraceMe traceme("ComputeSemaphoreAcquire");
    compute_reservation = std::make_shared<Semaphore::ScopedReservation>(
        device_state->compute_semaphore().ScopedAcquire(1));
  }

  absl::StatusOr<ExecutionOutput> result_buffer_or_status =
      executables_[executable_idx]->RunAsync(std::move(execution_inputs),
                                             run_options);

  VLOG(1) << "Replica " << replica << " partition " << partition
          << " completed; ok=" << result_buffer_or_status.ok();

  if (!result_buffer_or_status.ok()) {
    return result_buffer_or_status.status();
  }

  if (device_state->allocation_model() == LocalDeviceState::kSynchronous) {
    ExecutionOutput& execution_output = result_buffer_or_status.value();
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
PjRtStreamExecutorLoadedExecutable::MakeOutputBuffers(
    int device_ordinal, const ExecuteOptions& options,
    ScopedShapedBuffer result_buffer,
    std::shared_ptr<BufferSequencingEvent> definition_event, PjRtDevice* device,
    std::vector<std::function<void()>>& compute_callbacks,
    std::vector<std::shared_ptr<TrackedDeviceBuffer>>& buffers_to_release)
    const {
  tsl::profiler::TraceMe traceme("MakeOutputBuffers");
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

static absl::Status GetFirstInputError(
    absl::Span<PjRtBuffer* const> argument_handles) {
  for (auto* handle : argument_handles) {
    auto* buffer = tensorflow::down_cast<PjRtStreamExecutorBuffer*>(handle);
    PjRtStreamExecutorBuffer::ScopedHold hold =
        buffer->GetBufferWithUsageHold();
    for (const auto& event : hold->definition_events()) {
      if (event->IsPredeterminedError()) {
        return event->GetDefinedStatus();
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<PjRtLoadedExecutable::Result>
PjRtStreamExecutorLoadedExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const RunId& run_id, const ExecuteOptions& options, bool fill_future,
    PjRtDevice* device) const {
  const uint64_t start_time_usecs = tsl::Env::Default()->NowMicros();
  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int64_t device_id = (*device_assignment_)(replica, partition);
    PjRtGlobalDeviceId global_device_id(device_id);
    TF_ASSIGN_OR_RETURN(device, client_->LookupDevice(global_device_id));
    device_assignment = device_assignment_;
  } else {
    CHECK(device_assignment_ == nullptr);
    CHECK_EQ(replica, 0);
    CHECK_EQ(partition, 0);
    CHECK(addressable_devices_.empty());
    device_assignment = std::make_shared<DeviceAssignment>(1, 1);
    (*device_assignment)(0, 0) = device->id();
  }

  absl::Status input_error = GetFirstInputError(argument_handles);
  if (!input_error.ok()) {
    TF_ASSIGN_OR_RETURN(PjRtMemorySpace * memory_space,
                        device->default_memory_space());
    std::vector<std::unique_ptr<PjRtBuffer>> outputs;
    TF_ASSIGN_OR_RETURN(auto hlo_modules, GetHloModules());
    for (const auto& hlo_module : hlo_modules) {
      TF_ASSIGN_OR_RETURN(
          auto error_buffer,
          client_->CreateErrorBuffer(input_error, hlo_module->result_shape(),
                                     memory_space));
      outputs.push_back(std::move(error_buffer));
    }
    auto future = std::make_optional(PjRtFuture<>(input_error));
    return Result({std::move(future), /*buffers=*/std::move(outputs)});
  }

  CHECK_EQ(device->process_index(), client_->process_index());
  int device_ordinal = tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                           ->local_device_state()
                           ->local_device_id()
                           .value();
  tsl::profiler::TraceMe traceme(
      "PjRtStreamExecutorLoadedExecutable::ExecuteHelper");
  VLOG(1) << "Replica " << replica << ", partition " << partition
          << " mapped to device ordinal for execution: " << device_ordinal;

  // SPMD sharding produces a single executable for multiple partitions.
  int executable_idx = executables_.size() > 1 ? partition : 0;

  std::vector<std::function<void()>> compute_callbacks;
  std::vector<PjRtStreamExecutorBuffer::ScopedHold> device_buffers;
  device_buffers.reserve(argument_handles.size());
  absl::StatusOr<ScopedShapedBuffer> result_buffer_or_status = EnqueueExecution(
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
  absl::StatusOr<EventPool::Handle> event_or =
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
  auto definition_event =
      std::make_shared<BufferSequencingEvent>(client_->thread_pool());
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

  std::optional<PjRtFuture<>> future;
  if (fill_future) {
    auto promise = PjRtFuture<>::CreatePromise();
    future = PjRtFuture<>(promise);
    compute_callbacks.push_back(
        [promise = std::move(promise)]() mutable { promise.Set(); });
  }
  TF_RETURN_IF_ERROR(device_state->ThenExecuteCallback(
      stream, [callbacks{std::move(compute_callbacks)},
               buffers_to_release{std::move(buffers_to_release)}]() {
        for (auto& fn : callbacks) {
          fn();
        }
      }));
  metrics::ReportExecutableEnqueueTime(tsl::Env::Default()->NowMicros() -
                                       start_time_usecs);
  return Result({/*future=*/std::move(future), /*buffers=*/std::move(outputs)});
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
PjRtStreamExecutorLoadedExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  if (device_assignment_ == nullptr) {
    return InvalidArgument("Execute expects a non-null device_assignment");
  }

  RunId run_id;
  tsl::profiler::TraceMeProducer activity(
      "PjRtStreamExecutorLoadedExecutable::Execute",
      tsl::profiler::ContextType::kPjRt, run_id.ToInt());

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
  std::vector<absl::StatusOr<Result>> results(num_addressable_devices);
  if (num_addressable_devices == 1 && !ThisThreadIsInsideHostCallback()) {
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
    absl::Status first_failure_status;

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
            << first_failure_status.message();
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

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtStreamExecutorLoadedExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
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

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtStreamExecutorLoadedExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
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

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
PjRtStreamExecutorLoadedExecutable::GetHloModules() const {
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

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
PjRtStreamExecutorLoadedExecutable::GetOutputMemoryKinds() const {
  return Unimplemented("GetOutputMemoryKinds is not supported.");
}

absl::StatusOr<std::string>
PjRtStreamExecutorLoadedExecutable::FingerprintExecutable() const {
  if (executables_.size() != 1) {
    return absl::InternalError(
        "Fingerprinting multiple executables within one "
        "PjRtStreamExecutorLoadedExecutable is not supported.");
  }

  Executable* executable = executables_[0]->executable();
  if (executable->has_module()) {
    return executable->module().GetFingerprint128();
  } else {
    return absl::InternalError("Executable does not have HLO modules.");
  }
}

absl::StatusOr<PjRtStreamExecutorClient::ExecutableExtras>
PjRtStreamExecutorClient::GetExecutableExtras(CompileOptions* options) {
  ExecutableExtras extras;
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<PjRtStreamExecutorLoadedExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  ExecutableBuildOptions& build_options = options->executable_build_options;
  if (!build_options.compile_thread_pool()) {
    build_options.set_compile_thread_pool(thread_pool());
  }
  if (!build_options.device_allocator()) {
    build_options.set_device_allocator(allocator());
  }

  auto layout_callback = [local_client = client()](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    ExecutableBuildOptions build_options;
    std::vector<const Shape*> argument_layout_pointers;
    std::optional<std::vector<Shape>> argument_layouts;
    Shape result_layout;
    TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
        XlaComputation(module.ToProto()),
        [local_client = local_client](Shape shape) {
          return local_client->backend()
              .transfer_manager()
              ->ChooseCompactLayoutForShape(shape);
        },
        argument_layouts, &build_options, &argument_layout_pointers));
    result_layout = *build_options.result_layout();
    return std::make_pair(*argument_layouts, result_layout);
  };

  build_options.set_layout_canonicalization_callback(layout_callback);

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
        int64_t device_id = (*device_assignment)(replica, partition);
        PjRtGlobalDeviceId global_device_id(device_id);

        TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                            LookupDevice(global_device_id));
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

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::Compile(const XlaComputation& computation,
                                  CompileOptions options) {
  tsl::profiler::TraceMe traceme("PjRtStreamExecutorClient::Compile");
  VLOG(1) << "PjRtStreamExecutorClient::Compile";
  options.executable_build_options.set_process_index(process_index());
  TF_RET_CHECK(device_count() % addressable_device_count() == 0)
      << "Each process is expected to have the same number of devices";
  options.executable_build_options.set_process_count(
      device_count() / addressable_device_count());
  auto input_options = options;

  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());

  TF_ASSIGN_OR_RETURN(ExecutableExtras extras, GetExecutableExtras(&options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<PjRtStreamExecutorLoadedExecutable::LogicalDeviceIds>&
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

  auto executable = std::make_unique<PjRtStreamExecutorLoadedExecutable>(
      std::move(local_executables), options.parameter_is_tupled_arguments,
      std::move(device_assignment), std::move(input_options),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);

  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(options.parameter_is_tupled_arguments));
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::Compile(mlir::ModuleOp module,
                                  CompileOptions options) {
  XlaComputation xla_computation;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false));
  return Compile(xla_computation, options);
}

absl::StatusOr<std::string> PjRtStreamExecutorClient::SerializeExecutable(
    const PjRtLoadedExecutable& executable) const {
  const PjRtStreamExecutorLoadedExecutable* se_executable =
      tensorflow::down_cast<const PjRtStreamExecutorLoadedExecutable*>(
          &executable);

  absl::Span<const std::shared_ptr<LocalExecutable>> local_executables =
      se_executable->executables();
  if (local_executables.empty()) {
    return Internal("No local executable");
  }
  if (local_executables.size() != 1) {
    return Unimplemented(
        "PjRtStreamExecutorClient::SerializeExecutable unimplemented for MPMD "
        "executables");
  }

  Executable* built_executable = local_executables[0]->executable();
  Compiler* compiler = client_->backend().compiler();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      compiler->Export(built_executable));
  TF_ASSIGN_OR_RETURN(std::string serialized, aot_result->SerializeAsString());
  if (serialized.empty()) {
    return Internal(
        "PjRtStreamExecutorClient::SerializeExecutable proto serialization "
        "failed");
  }
  ExecutableAndOptionsProto proto;
  *proto.mutable_serialized_executable() = std::move(serialized);
  TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                      se_executable->compile_options_.ToProto());
  return proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::DeserializeExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options) {
  ExecutableAndOptionsProto proto;
  if (serialized.size() > std::numeric_limits<int>::max()) {
    return Internal(
        "PjRtStreamExecutorClient::DeserializeExecutable proto too large "
        "(>2GB)");
  }
  if (!proto.ParseFromArray(serialized.data(), serialized.size())) {
    return Internal(
        "PjRtStreamExecutorClient::DeserializeExecutable proto "
        "deserialization "
        "failed");
  }

  CompileOptions compile_options;
  if (options.has_value()) {
    compile_options = *std::move(options);
  } else {
    TF_ASSIGN_OR_RETURN(compile_options,
                        CompileOptions::FromProto(proto.compile_options()));
  }
  auto input_options = compile_options;

  tsl::profiler::TraceMe traceme(
      "PjRtStreamExecutorClient::DeserializeExecutable");
  VLOG(1) << "PjRtStreamExecutorClient::DeserializeExecutable";

  TF_ASSIGN_OR_RETURN(ExecutableExtras extras,
                      GetExecutableExtras(&compile_options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<PjRtStreamExecutorLoadedExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  std::string str = std::move(*proto.mutable_serialized_executable());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LocalExecutable> loaded,
      client()->Load(str, compile_options.executable_build_options));

  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.push_back(std::move(loaded));

  auto executable = std::make_unique<PjRtStreamExecutorLoadedExecutable>(
      std::move(local_executables),
      compile_options.parameter_is_tupled_arguments,
      std::move(device_assignment), std::move(input_options),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);

  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(compile_options.parameter_is_tupled_arguments));
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::LoadSerializedExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options,
    const LoadOptions& load_options) {
  return DeserializeExecutable(serialized, options);
}

}  // namespace xla
