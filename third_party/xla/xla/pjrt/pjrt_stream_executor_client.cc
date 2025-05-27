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
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/abstract_tracked_device_buffer.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/metrics.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/profiling/device_time_measurement.h"
#include "xla/pjrt/profiling/profiling_context.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/stream_executor_executable.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_layout.h"
#include "xla/service/dump.h"
#include "xla/service/executable.h"
#include "xla/service/generic_transfer_manager.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/stream.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/mem.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
#include "tsl/profiler/lib/traceme.h"

namespace xla {

template <typename T>
static std::function<void()> WrapClosureAsCopyable(T cb) {
  return [state = std::make_shared<T>(std::move(cb))]() { return (*state)(); };
}

PjRtStreamExecutorMemorySpace::PjRtStreamExecutorMemorySpace(
    int id, PjRtDevice* device, absl::string_view kind, int kind_id)
    : id_(id), device_(device), kind_(kind), kind_id_(kind_id) {
  to_string_ = absl::StrFormat("MEMORY_SPACE_%i", id_);
  debug_string_ = absl::StrFormat("PjRtStreamExecutorMemory(id=%i, device=%s)",
                                  id_, device_->DebugString());
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
    int process_index,
    std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces,
    std::unique_ptr<se::DeviceMemoryAllocator> allocator,
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
      owned_memory_spaces_(std::move(memory_spaces)),
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

  for (const std::unique_ptr<PjRtMemorySpace>& memory_space :
       owned_memory_spaces_) {
    memory_spaces_.push_back(memory_space.get());
  }
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
void RecordUsage(PjRtStreamExecutorBuffer::ScopedHold device_buffer,
                 LocalDeviceState* buffer_local_device,
                 LocalDeviceState* stream_local_device,
                 std::shared_ptr<BufferSequencingEvent> event,
                 se::Stream* usage_stream,
                 std::vector<tsl::RCReference<RawSEDeviceMemory>>*
                     buffers_to_release = nullptr) {
  tsl::profiler::TraceMe traceme("RecordUsage");
  bool retain_buffer_until_completion =
      // If the buffer wasn't allocated on the same device as the stream, always
      // retain a reference.
      (stream_local_device != buffer_local_device) ||
      // In the synchronous allocation model, always retain a reference.
      (stream_local_device->allocation_model() ==
       LocalDeviceState::kSynchronous);
  if (retain_buffer_until_completion) {
    if (buffers_to_release) {
      buffers_to_release->push_back(device_buffer->device_memory());
    } else {
      buffer_local_device
          ->ThenRelease(usage_stream, device_buffer->device_memory())
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
  return absl::OkStatus();
}

// We wait for events that the compute stream didn't already wait for. Based on
// our heuristics, for usage events, this rare case should only occur when a
// buffer was copied to a device and then never used there. In that case we get
// a new stream and use it to hold onto a reference to the buffer until the
// events are complete.
void MaybeWaitForEventOnStream(BufferSequencingEvent* event,
                               LocalDeviceState* local_device_state,
                               se::Stream*& stream) {
  if (!event->IsPredeterminedErrorOrDefinedOn(
          local_device_state->compute_stream()) &&
      !event->IsComplete()) {
    if (stream == nullptr) {
      stream = local_device_state->GetFixedSizePoolUsageStream();
    }
    VLOG(2) << "Waiting for event: " << event
            << "; is_predetermined_error: " << event->IsPredeterminedError()
            << "; on stream: " << stream;
    event->WaitForEventOnStream(stream);
  }
}

}  // namespace

absl::StatusOr<std::unique_ptr<PjRtStreamExecutorBuffer>>
AllocateDestinationBuffer(
    const Shape& on_host_shape, PjRtDevice* device,
    LocalDeviceState* local_device, se::Stream* copy_stream,
    bool is_uninitialized_create, PjRtStreamExecutorClient* client,
    std::shared_ptr<BufferSequencingEvent> definition_event,
    PjRtMemorySpace* memory_space) {
  if (on_host_shape.IsTuple()) {
    return InvalidArgument(
        "Cannot allocate a PjRtStreamExecutorBuffer for a tuple.");
  }

  PjRtMemorySpace* default_memory_space =
      device->default_memory_space().value_or(nullptr);
  if (!memory_space) {
    memory_space = default_memory_space;
  }
  bool is_pinned_host_memory =
      memory_space && (memory_space->kind() == PinnedHostMemorySpace::kKind);
  // Only allow pinned host memory or device memory.
  if (memory_space != default_memory_space && !is_pinned_host_memory) {
    return InvalidArgument("Buffer allocation: invalid memory space");
  }

  auto* se_client = tensorflow::down_cast<PjRtStreamExecutorClient*>(client);
  TransferManager* transfer_manager =
      se_client->client()->backend().transfer_manager();

  // Communicate the desired memory space to the allocator via the shape
  // callback.
  auto memory_space_shape_fn = [is_pinned_host_memory,
                                transfer_manager](const Shape& shape) {
    Shape result = transfer_manager->HostShapeToDeviceShape(shape);
    if (is_pinned_host_memory) {
      result.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
    }
    return result;
  };

  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer dst_buffer,
      transfer_manager->AllocateScopedShapedBuffer(
          on_host_shape, se_client->allocator(),
          local_device->local_device_id().value(),
          local_device->local_hardware_id().value(), memory_space_shape_fn));
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
      definition_events.push_back(definition_event);
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
      definition_events.push_back(definition_event);
    } else {
      definition_events.emplace_back(
          std::make_shared<BufferSequencingEvent>(client->thread_pool()));
    }
  }

  auto mem = RawSEDeviceMemory::Create(dst_buffer.buffer({}),
                                       device->local_device_id(),
                                       dst_buffer.memory_allocator());
  dst_buffer.clear();

  auto dst_device_buffer = std::make_unique<TrackedDeviceBuffer>(
      device, std::move(mem), definition_events);

  auto py_buffer = std::make_unique<PjRtStreamExecutorBuffer>(
      on_device_shape, std::move(dst_device_buffer), client, device,
      memory_space);
  return py_buffer;
}

void PjRtStreamExecutorBuffer::ScopedHold::ConvertUsageHold(
    se::Stream* usage_stream, std::shared_ptr<BufferSequencingEvent> event,
    bool reference_held) {
  CHECK(ok());
  CHECK_EQ(type(), kUsage);
  parent()->ConvertUsageHold(buffer(), usage_stream, std::move(event),
                             reference_held);
  SetState(kConverted);
}

bool PjRtStreamExecutorBuffer::IsOnCpu() const {
  return memory_space() != nullptr &&
         memory_space()->kind() == PinnedHostMemorySpace::kKind;
}

absl::StatusOr<Shape> PjRtStreamExecutorBuffer::logical_on_device_shape() {
  if (on_device_shape_.is_static()) {
    return on_device_shape_;
  }
  auto* local_device = device_->local_device_state();
  auto* stream = local_device->GetDeviceToHostStream();
  auto device_buffer = GetBufferWithUsageHold();
  if (!device_buffer.ok()) {
    return InvalidArgument(
        "logical_on_device_shape() called on deleted or donated buffer: %s",
        device_buffer.status().ToString());
  }

  WaitForBufferDefinitionEventsOnStream(device_buffer->definition_events(),
                                        stream);
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
    data_ptr_ = external_reference_->device_memory()->opaque();
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
      tsl::RCReference<RawSEDeviceMemory> memory)
      : memory_(std::move(memory)) {
    data_ptr_ = memory_->opaque();
  }

  ~TrackedDeviceBufferExternalReference() override = default;

 private:
  tsl::RCReference<RawSEDeviceMemory> memory_;
};

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
PjRtStreamExecutorBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(tsl::RCReference<RawSEDeviceMemory> tracked_device_buffer,
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
  const auto& original_definition_events = tracked_buffer->definition_events();
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

  auto new_device_buffer = std::make_unique<TrackedDeviceBuffer>(
      device(), tracked_buffer->device_memory(), std::move(definition_events));

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

// BufferFromHostBuffer() is used to create a buffer either for a device, or
// for a host memory, depending on `memory_space`. The memory copy is needed
// for both cases, either from the unpinned host memory to device, or from
// the unpinned host memory to the pinned host memory.
absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostBufferInternal(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer, PjRtDevice* device,
    const Layout* device_layout, PjRtMemorySpace* memory_space) {
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
  absl::InlinedVector<int64_t, 4> shape_strides(
      device_shape.dimensions().size());
  TF_RETURN_IF_ERROR(
      ShapeUtil::ByteStrides(device_shape, absl::MakeSpan(shape_strides)));
  bool host_and_device_strides_equal =
      (size == 0 || *byte_strides == shape_strides);

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtStreamExecutorBuffer> py_buffer,
      AllocateDestinationBuffer(device_shape, device, local_device,
                                local_device->host_to_device_stream(),
                                /*is_uninitialized_create=*/false, this,
                                /*definition_event=*/nullptr, memory_space));

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
  if (must_use_staging_buffer || (!IsDmaMapped(data, packed_size) &&
                                  (should_stage_host_to_device_transfers() &&
                                   packed_size < (int64_t{1} << 30)))) {
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

  std::shared_ptr<BufferSequencingEvent> event =
      device_buffer->definition_events()[0];

  // The host to device transfer is performed on a thread pool, mostly because
  // it includes linearization that may be slow. It is OK to capture the
  // py_buffer pointer because the py_buffer can't be deleted until all the
  // usage holds have gone away.
  // TODO(misard) assess if it would be preferable to introduce a heuristic to
  // put the transfer into the calling thread for small literals.
  auto transfer_h2d =
      [local_client = client(), transfer_manager, local_device, data, size,
       type, packed_size, event,
       device_memory_owned = device_buffer->device_memory(), device_shape,
       should_pack, py_buffer{py_buffer.get()},
       on_device_shape{py_buffer->on_device_shape()},
       staging_buffer{std::move(staging_buffer)},
       on_done_with_host_buffer =
           on_done_with_host_buffer
               ? std::make_shared<absl::AnyInvocable<void() &&>>(
                     std::move(on_done_with_host_buffer))
               : nullptr,
       host_buffer_semantics, transpose{std::move(transpose)}]() mutable {
        // This function uses TF_CHECK_OK and value() since we have no way
        // to report failures from a callback. However, the operations here are
        // unlikely to fail and not recoverable even if we were to fail: DMAs to
        // memory that has already been allocated, and a possible Event
        // allocation.

        se::DeviceMemoryBase device_memory = device_memory_owned->mem();

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

        TF_CHECK_OK(AddDestinationBufferSynchronization(
            local_device, event, local_device->host_to_device_stream()));

        local_device
            ->ThenRelease(local_device->host_to_device_stream(),
                          device_memory_owned)
            .IgnoreError();

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
  thread_pool()->Schedule(WrapClosureAsCopyable(std::move(transfer_h2d)));
  RecordUsage(std::move(device_buffer), local_device, local_device, event,
              local_device->host_to_device_stream());
  return std::unique_ptr<PjRtBuffer>(std::move(py_buffer));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  return BufferFromHostBufferInternal(
      data, type, dims, byte_strides, host_buffer_semantics,
      std::move(on_done_with_host_buffer), memory_space->devices()[0],
      device_layout, memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateUninitializedBuffer(
    const Shape& shape, PjRtMemorySpace* memory_space) {
  return CreateUninitializedBuffer(shape, memory_space, nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateUninitializedBuffer(
    const Shape& shape, PjRtMemorySpace* memory_space,
    std::shared_ptr<BufferSequencingEvent> definition_event) {
  tsl::profiler::TraceMe traceme(
      "PjRtStreamExecutorClient::CreateUninitializedBuffer");
  VLOG(1) << "PjRtStreamExecutorClient::CreateUninitializedBuffer: shape: "
          << shape.ToString()
          << " memory_space: " << memory_space->DebugString();
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices().front();
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
  auto dummy_device_buffer = std::make_unique<TrackedDeviceBuffer>(
      device, tsl::RCReference<RawSEDeviceMemory>(),
      absl::MakeSpan(&definition_event, 1));

  auto py_buffer = std::make_unique<PjRtStreamExecutorBuffer>(
      shape, std::move(dummy_device_buffer), this, device,
      device->default_memory_space().value_or(nullptr));
  return py_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                                PjRtMemorySpace* memory_space,
                                                const Layout* device_layout) {
  if (device_layout) {
    return absl::UnimplementedError(absl::StrCat(
        "BufferFromHostLiteral with device_layout is not implemented on "
        "platform: ",
        platform_name()));
  }
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices().front();

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

  std::shared_ptr<BufferSequencingEvent> event =
      device_buffer->definition_events()[0];

  // The host to device transfer is performed on a thread pool, mostly because
  // it includes linearization that may be slow. It is OK to capture the
  // py_buffer pointer because the py_buffer can't be deleted until all the
  // usage holds have gone away.
  // TODO(misard) assess if it would be preferable to introduce a heuristic to
  // put the transfer into the calling thread for small literals.
  auto transfer_h2d = [local_client = client(), transfer_manager, local_device,
                       device_memory = device_buffer->device_memory(), device,
                       event, literal, py_buffer{py_buffer.get()},
                       on_device_shape{
                           py_buffer->on_device_shape()}]() mutable {
    // This function uses TF_CHECK_OK and value() since we have no way
    // to report failures from a callback. However, the operations here are
    // unlikely to fail and not recoverable even if we were to fail: DMAs to
    // memory that has already been allocated, and a possible Event
    // allocation.

    se::Stream* h2d_stream = local_device->host_to_device_stream();

    ShapedBuffer buffer =
        device_memory->AsShapedBuffer(device, on_device_shape);
    TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
        h2d_stream, literal, buffer));

    TF_CHECK_OK(
        AddDestinationBufferSynchronization(local_device, event, h2d_stream));

    local_device->ThenRelease(h2d_stream, device_memory).IgnoreError();

    // This can sometimes catch the case where the literal memory has been
    // freed before the H2D transfer was issued.
    h2d_stream->RefreshStatus()
        .IgnoreError();  // Can return error::Unimplemented
    QCHECK(h2d_stream->ok());
  };
  thread_pool()->Schedule(WrapClosureAsCopyable(std::move(transfer_h2d)));
  RecordUsage(std::move(device_buffer), local_device, local_device, event,
              local_device->host_to_device_stream());
  return std::unique_ptr<PjRtBuffer>(std::move(py_buffer));
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
      buffers, std::move(definition_event), std::move(notifier)));
  return buffers;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream) {
  CHECK_EQ(memory_space->devices().size(), 1);
  auto* device = memory_space->devices().front();

  auto buffer = RawSEDeviceMemory::CreateForeign(
      se::DeviceMemoryBase(device_ptr, ShapeUtil::ByteSizeOf(shape)),
      std::move(on_delete_callback));

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

  auto device_buffer = std::make_unique<TrackedDeviceBuffer>(
      device, std::move(buffer), definition_events);
  return std::unique_ptr<PjRtBuffer>(std::make_unique<PjRtStreamExecutorBuffer>(
      shape, std::move(device_buffer), this, device,
      device->default_memory_space().value_or(nullptr)));
}

absl::Status PjRtStreamExecutorClient::DmaMap(void* data, size_t buffer_size) {
  tsl::profiler::TraceMe trace_me("PjRtStreamExecutorClient::DmaMap");
  TF_ASSIGN_OR_RETURN(
      LocalDeviceState * local_device,
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(addressable_devices_[0])
          ->GetLocalDeviceState());
  bool success = local_device->compute_stream()->parent()->HostMemoryRegister(
      data, buffer_size);
  if (!success) {
    return absl::InternalError(absl::StrFormat(
        "Failed to register host memory at address: %ps", data));
  }
  absl::MutexLock lock(&dma_maps_mutex_);
  dma_maps_.insert({data, buffer_size});
  return absl::OkStatus();
}

absl::Status PjRtStreamExecutorClient::DmaUnmap(void* data) {
  tsl::profiler::TraceMe trace_me("PjRtStreamExecutorClient::DmaUnmap");
  TF_ASSIGN_OR_RETURN(
      LocalDeviceState * local_device,
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(addressable_devices_[0])
          ->GetLocalDeviceState());
  bool success =
      local_device->compute_stream()->parent()->HostMemoryUnregister(data);
  if (!success) {
    return absl::InternalError(absl::StrFormat(
        "Failed to unregister host memory at address: %ps", data));
  }
  absl::MutexLock lock(&dma_maps_mutex_);
  dma_maps_.erase(data);
  return absl::OkStatus();
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

void PjRtStreamExecutorDevice::AttachMemorySpace(PjRtMemorySpace* memory_space,
                                                 bool is_default) {
  CHECK(memory_space != nullptr);
  CHECK(client_ == memory_space->client()) << absl::StrFormat(
      "Could not attach a PjRtStreamExecutorDevice to a PjRtMemorySpace owned "
      "by a different client, the device's client: %s, the memory space's "
      "client: %s.",
      client_->platform_name(), memory_space->client()->platform_name());

  memory_spaces_.push_back(memory_space);
  memory_spaces_by_id_.emplace(memory_space->kind_id(), memory_space);
  if (is_default) {
    CHECK(default_memory_space_ == nullptr)
        << "Default memory space already set to "
        << default_memory_space_->DebugString() << ".";
    default_memory_space_ = memory_space;
  }
}

absl::Span<PjRtMemorySpace* const> PjRtStreamExecutorDevice::memory_spaces()
    const {
  return memory_spaces_;
}

absl::StatusOr<PjRtMemorySpace*>
PjRtStreamExecutorDevice::default_memory_space() const {
  if (default_memory_space_ == nullptr) {
    return absl::InternalError(
        "No default memory space is set for this device.");
  }
  return default_memory_space_;
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
    Shape on_device_shape, std::unique_ptr<TrackedDeviceBuffer> device_buffer,
    PjRtClient* client, PjRtDevice* device, PjRtMemorySpace* memory_space)
    : CommonPjRtBuffer(std::move(device_buffer)),
      client_(tensorflow::down_cast<PjRtStreamExecutorClient*>(client)),
      on_device_shape_(std::move(on_device_shape)),
      device_(tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)),
      memory_space_(memory_space) {}

PjRtStreamExecutorBuffer::~PjRtStreamExecutorBuffer() {
  Delete();
}

absl::StatusOr<tsl::RCReference<RawSEDeviceMemory>>
PjRtStreamExecutorBuffer::Release(bool wait_for_operations_to_complete) {
  tsl::profiler::TraceMe trace_me("PjRtStreamExecutorBuffer::Release");
  std::unique_ptr<TrackedDeviceBuffer> device_buffer(
      static_cast<TrackedDeviceBuffer*>(ReleaseBuffer().release()));
  if (device_buffer == nullptr) {
    return tsl::RCReference<RawSEDeviceMemory>();
  }
  TrackedDeviceBuffer::StreamAndEventContainer events =
      device_buffer->LockUseAndTransferUsageEvents();
  auto device_memory = device_buffer->device_memory();
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
      // If an event is not defined yet, we wait for it to be defined in a new
      // thread in the thread pool.
      // This allows the host to schedule:
      //   create buffer -> use -> delete -> fulfill
      absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 5>
          events_to_wait_for_in_a_different_thread;
      auto maybe_wait_for_event_on_block_stream_or_add_to_events_to_wait =
          [&events_to_wait_for_in_a_different_thread, local_device_state,
           &block_stream](const std::shared_ptr<BufferSequencingEvent>& event) {
            if (local_device_state->allow_delete_before_fulfill() &&
                !event->IsDefined()) {
              // Wait for the event to be defined in a different thread.
              events_to_wait_for_in_a_different_thread.push_back(event);
            } else {
              MaybeWaitForEventOnStream(event.get(), local_device_state,
                                        block_stream);
            }
          };
      for (const auto& stream_and_event : events) {
        VLOG(4)
            << "Checking whether need to wait for stream_and_event: stream: "
            << stream_and_event.stream
            << "; event: " << stream_and_event.event.get()
            << "; reference_held: " << stream_and_event.reference_held
            << "; is_predetermined_error: "
            << stream_and_event.event->IsPredeterminedError();
        // We only need to do something for events that didn't already acquire a
        // reference to the buffer and for other situations described in the
        // comment of MaybeWaitForEventOnStream()
        if (!stream_and_event.reference_held) {
          maybe_wait_for_event_on_block_stream_or_add_to_events_to_wait(
              stream_and_event.event);
        }
      }
      for (const auto& definition_event : device_buffer->definition_events()) {
        VLOG(4) << "Checking whether need to wait for definition_event: "
                << definition_event.get() << "; is_predetermined_error: "
                << definition_event->IsPredeterminedError();
        // Here we wait for the definition events to complete on block_stream as
        // well, in case they are not also usage events.
        maybe_wait_for_event_on_block_stream_or_add_to_events_to_wait(
            definition_event);
      }
      if (!events_to_wait_for_in_a_different_thread.empty()) {
        VLOG(3) << "Going to wait for "
                << events_to_wait_for_in_a_different_thread.size()
                << " events in a different thread.";
        // We always use the cleanup_thread instead of using the
        // client->thread_pool() here to avoid exhausting the client thread
        // pool.
        local_device_state->cleanup_thread()->Schedule(
            [events_to_wait_for_in_a_different_thread =
                 std::move(events_to_wait_for_in_a_different_thread),
             local_device_state, device_memory, block_stream]() mutable {
              for (const auto& event :
                   events_to_wait_for_in_a_different_thread) {
                MaybeWaitForEventOnStream(event.get(), local_device_state,
                                          block_stream);
              }
              if (block_stream != nullptr) {
                TF_CHECK_OK(local_device_state->ThenExecuteCallback(
                    block_stream, [device_memory]() {
                      // Drops device_memory shared pointer.
                    }));
              }
            });
      } else if (block_stream != nullptr) {
        TF_RETURN_IF_ERROR(local_device_state->ThenExecuteCallback(
            block_stream, [device_memory]() {
              // Drops device_memory shared pointer.
            }));
      }
    }
  }
  return device_memory;
}

void PjRtStreamExecutorBuffer::Delete() {
  VLOG(3) << "PjRtStreamExecutorBuffer::Delete";

  // When wait_for_reads_to_complete is false, Release should never fail.
  //
  // The only usage events that
  // Release(/*wait_for_operations_to_complete=*/false) doesn't wait for are
  // events defined on the compute stream. All streams other than the compute
  // stream are expected to WaitFor compute stream before any write operations.
  TF_CHECK_OK(Release(/*wait_for_operations_to_complete=*/false).status());
}

void PjRtStreamExecutorBuffer::ConvertUsageHold(
    TrackedDeviceBuffer* buffer, se::Stream* usage_stream,
    std::shared_ptr<BufferSequencingEvent> event, bool reference_held) {
  absl::MutexLock lock(&mu_);
  CHECK(device_buffer() == buffer || device_buffer() == nullptr);
  buffer->AddUsageEvent(usage_stream, std::move(event), reference_held);
  DecrementUsage();
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
  VLOG(3) << "PjRtStreamExecutorBuffer::ToLiteral";
  if (IsEmptyTuple()) {
    return PjRtFuture<>(InvalidArgument("ToLiteral called on empty tuple"));
  }
  LocalDeviceState* local_device = device_->local_device_state();
  se::Stream* stream = local_device->GetDeviceToHostStream();
  auto device_buffer = GetBufferWithUsageHold();
  if (!device_buffer.ok()) {
    return PjRtFuture<>(
        InvalidArgument("ToLiteral() called on deleted or donated buffer: %s",
                        device_buffer.status().ToString()));
  }

  auto promise = PjRtFuture<>::CreatePromise();
  auto usage_event =
      std::make_shared<BufferSequencingEvent>(client_->thread_pool());

  TransferManager* transfer_manager =
      client_->client()->backend().transfer_manager();

  auto device_memory = device_buffer->device_memory();
  auto definition_events = device_buffer->definition_events();
  auto first_definition_event = definition_events[0];

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

  std::shared_ptr<TransposePlan> transpose;
  if (on_device_shape().IsArray()) {
    xla::Layout literal_layout;
    if (literal->shape().has_layout()) {
      literal_layout = literal->shape().layout();
    } else {
      literal_layout = LayoutUtil::MakeDescendingLayout(
          on_device_shape().dimensions().size());
    }

    if (on_device_shape().layout() != literal_layout) {
      absl::InlinedVector<int64_t, 4> byte_strides(
          on_device_shape().dimensions().size());
      absl::Status s = ShapeUtil::ByteStrides(on_device_shape(),
                                              absl::MakeSpan(byte_strides));
      if (!s.ok()) {
        return PjRtFuture<>(s);
      }
      absl::Span<const int64_t> dims = on_device_shape().dimensions();
      absl::InlinedVector<int64_t, 4> permutation(dims.size());
      absl::c_reverse_copy(literal_layout.minor_to_major(),
                           permutation.begin());
      TransposePlan::Options options;
      options.elem_size_in_bytes =
          primitive_util::ByteWidth(on_device_shape().element_type());
      options.dims = on_device_shape().dimensions();
      options.permutation = permutation;
      options.input_layout = TransposePlan::Striding{byte_strides};
      {
        absl::MutexLock lock(&client_->transpose_mu_);
        absl::StatusOr<std::shared_ptr<TransposePlan>> t =
            client_->transpose_cache_.GetOrCreate(options);
        if (!t.ok()) {
          return PjRtFuture<>(t.status());
        }
        transpose = *std::move(t);
      }
    }
  }

  auto async_to_literal = [usage_event,
                           device_memory = std::move(device_memory),
                           definition_events = std::move(definition_events),
                           stream, device = device_,
                           transfer_manager = std::move(transfer_manager),
                           on_device_shape{on_device_shape_}, literal,
                           transpose, promise, local_device]() mutable {
    absl::StatusOr<EventPool::Handle> event_or =
        local_device->event_pool().AllocateEvent(stream->parent());
    if (!event_or.ok()) {
      promise.Set(event_or.status());
      return;
    }

    absl::Status defined_status = definition_events[0]->GetDefinedStatus();
    if (!defined_status.ok()) {
      promise.Set(defined_status);
      return;
    }

    WaitForBufferDefinitionEventsOnStream(absl::MakeSpan(definition_events),
                                          stream);
    ShapedBuffer shaped_buffer =
        device_memory->AsShapedBuffer(device, on_device_shape);

    GenericTransferManager::LiteralFromDeviceMetadata transfer_metadata;
    // We never call device functions from the `done` callback.
    transfer_metadata.callback_is_host_callback_safe = true;

    TransferManager::TransferMetadata* transfer_metadata_ptr =
        (dynamic_cast<GenericTransferManager*>(transfer_manager) != nullptr)
            ? &transfer_metadata
            : nullptr;

    if (transpose) {
      // Copy the device buffer to a temporary literal with descending
      // layout and transpose to the requested layout.

      Shape stage_shape = literal->shape();
      *stage_shape.mutable_layout() =
          LayoutUtil::MakeDescendingLayout(stage_shape.dimensions().size());
      auto staged = std::make_shared<Literal>(stage_shape);

      transfer_manager->TransferLiteralFromDevice(
          stream, shaped_buffer, staged.get(),
          [transpose, promise, staged, literal](absl::Status status) mutable {
            if (status.ok()) {
              transpose->Execute(staged->untyped_data(),
                                 literal->untyped_data());
            }
            promise.Set(std::move(status));
          },
          transfer_metadata_ptr);
    } else {
      transfer_manager->TransferLiteralFromDevice(
          stream, shaped_buffer, literal,
          [promise](absl::Status status) mutable {
            promise.Set(std::move(status));
          },
          transfer_metadata_ptr);
    }

    local_device->event_pool().ThenRecordEvent(stream, event_or.value());
    usage_event->SetSequencingEvent(std::move(event_or).value(), stream);

    defined_status = local_device->ThenRelease(stream, device_memory);
    if (!defined_status.ok()) {
      promise.Set(defined_status);
    }
  };

  first_definition_event->ExecuteOrAddToFutureTasks(
      absl::StrFormat("async_to_literal_%p", literal),
      std::move(async_to_literal));

  return PjRtFuture<>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "PjRtStreamExecutorBuffer::ToLiteral");
        VLOG(3) << "PjRtStreamExecutorBuffer::ToLiteral";
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
  if (device_buffer() == nullptr || !device_buffer()->device_memory()) {
    return InvalidArgument(
        "GetOnDeviceSizeInBytes called on deleted or donated buffer");
  }
  return device_buffer()->device_memory()->mem().size();
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
    PjRtMemorySpace* dst_memory_space, LocalDeviceState* transfer_local_device,
    LocalDeviceState* src_local_device, se::Stream* transfer_stream,
    const TrackedDeviceBuffer& src_device_buffer) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtStreamExecutorBuffer> py_buffer,
                      AllocateDestinationBuffer(
                          ShapeUtil::DeviceShapeToHostShape(on_device_shape_),
                          dst_device, dst_local_device, transfer_stream,
                          /*is_uninitialized_create=*/false, client_,
                          /*definition_event=*/nullptr, dst_memory_space));

  ScopedHold dst_device_buffer(py_buffer->GetBufferWithUsageHold());
  CHECK(dst_device_buffer.ok());

  std::shared_ptr<BufferSequencingEvent> copy_event =
      dst_device_buffer->definition_events()[0];

  // Copy the leaf buffers.
  auto async_copy_to_device = [src_memory = src_device_buffer.device_memory(),
                               src_definition_events =
                                   src_device_buffer.definition_events(),
                               dst_memory = dst_device_buffer->device_memory(),
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
    VLOG(3)
        << "PjRtStreamExecutorBuffer::CopyToDeviceHelper::async_copy_to_device";

    absl::Status defined_status = src_definition_events[0]->GetDefinedStatus();
    // Only proceeds to transfer when the buffer doesn't hold an error.
    if (defined_status.ok()) {
      WaitForBufferDefinitionEventsOnStream(src_definition_events,
                                            transfer_stream);

      const se::DeviceMemoryBase& input_buffer = src_memory->mem();
      const se::DeviceMemoryBase& output_buffer = dst_memory->mem();
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
            auto status =
                src_local_device->ThenRelease(transfer_stream, src_memory);
            if (!status.ok()) {
              LOG(ERROR) << "ThenRelease failed due to: " << status;
            }
          }
          return;
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

    auto status =
        src_local_device->ThenRelease(transfer_stream, std::move(src_memory));
    if (!status.ok()) {
      LOG(ERROR) << "ThenRelease failed due to: " << status;
    }
  };

  src_device_buffer.definition_events()[0]->ExecuteOrAddToFutureTasks(
      absl::StrFormat("async_copy_to_device_%p", dst_device_buffer.buffer()),
      std::move(async_copy_to_device));

  RecordUsage(std::move(dst_device_buffer), transfer_local_device,
              transfer_local_device, copy_event, transfer_stream);

  return std::pair<std::unique_ptr<PjRtBuffer>,
                   std::shared_ptr<BufferSequencingEvent>>(
      std::unique_ptr<PjRtStreamExecutorBuffer>(std::move(py_buffer)),
      std::move(copy_event));
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorBuffer::CopyToDeviceMemorySpace(
    PjRtDevice* dst_device, PjRtMemorySpace* dst_memory_space) {
  // Copying across PjRtClients involves a copy through the host.
  if (dst_device->client() != client_) {
    TF_ASSIGN_OR_RETURN(std::shared_ptr<Literal> literal, ToLiteralSync());
    // Avoid use-after-free on `literal` due to unsequenced move and use.
    Literal* literal_pointer = literal.get();
    absl::InlinedVector<int64_t, 4> byte_strides(
        literal->shape().dimensions().size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(literal->shape(), absl::MakeSpan(byte_strides)));
    return dst_device->client()->BufferFromHostBuffer(
        literal_pointer->untyped_data(),
        literal_pointer->shape().element_type(),
        literal_pointer->shape().dimensions(), byte_strides,
        PjRtStreamExecutorClient::HostBufferSemantics::kImmutableZeroCopy,
        [literal{std::move(literal)}]() { /* frees literal */ },
        dst_memory_space, /*device_layout=*/nullptr);
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

  auto src_device_buffer = GetBufferWithUsageHold();
  if (!src_device_buffer.ok()) {
    return InvalidArgument(
        "CopyToDevice() called on deleted or donated buffer: %s",
        src_device_buffer.status().ToString());
  }

  absl::StatusOr<std::pair<std::unique_ptr<PjRtBuffer>,
                           std::shared_ptr<BufferSequencingEvent>>>
      buffer_and_event_or = CopyToDeviceHelper(
          dst_device, dst_local_device, dst_memory_space, transfer_local_device,
          device_->local_device_state(), transfer_stream, *src_device_buffer);
  if (!buffer_and_event_or.ok()) {
    return buffer_and_event_or.status();
  }

  auto& buffer_and_event = buffer_and_event_or.value();
  std::unique_ptr<PjRtBuffer>& buffer = buffer_and_event.first;
  std::shared_ptr<BufferSequencingEvent>& event = buffer_and_event.second;

  src_device_buffer.ConvertUsageHold(transfer_stream, event,
                                     /*reference_held=*/true);

  return std::move(buffer);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
PjRtStreamExecutorBuffer::CopyToMemorySpace(PjRtMemorySpace* dst_memory_space) {
  if (dst_memory_space->devices().size() == 1) {
    return CopyToDeviceMemorySpace(dst_memory_space->devices()[0],
                                   dst_memory_space);
  }
  return Unimplemented("CopyToMemorySpace is not supported");
}

void PjRtStreamExecutorBuffer::CopyToRemoteDevice(
    PjRtFuture<std::string> serialized_descriptor, RemoteSendCallback on_done) {
  VLOG(3) << "PjRtStreamExecutorBuffer::CopyToRemoteDevice";
  auto desc = serialized_descriptor.Await();
  if (desc.ok()) {
    client_->CopyToRemoteDevice(this, *desc, std::move(on_done));
  } else {
    on_done(desc.status(), /*sends_enqueued=*/false);
  }
}

PjRtFuture<> PjRtStreamExecutorBuffer::GetReadyFuture() {
  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 2>
      definition_events;
  PjRtFuture<>::Promise definition_promise;
  {
    absl::MutexLock lock(&mu_);
    if (device_buffer() == nullptr) {
      return PjRtFuture<>(InvalidArgument(
          "GetReadyFuture() called on deleted or donated buffer"));
    }
    if (!definition_promise_) {
      definition_events = device_buffer()->definition_events();
      definition_promise_ = PjRtFuture<>::CreatePromise();
    }
    definition_promise = definition_promise_;
  }

  if (!definition_events.empty()) {
    LocalDeviceState* local_device_state = device_->local_device_state();
    auto first_definition_event = definition_events[0];
    auto async_wait_for_events =
        [definition_events = std::move(definition_events),
         local_device_state = std::move(local_device_state),
         definition_promise]() mutable {
          std::unique_ptr<se::Stream> stream;
          absl::Status defined_status =
              definition_events[0]->GetDefinedStatus();
          if (!defined_status.ok()) {
            definition_promise.Set(defined_status);
            return;
          }
          for (auto& event : definition_events) {
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
                 event_with_status = definition_events[0]]() mutable {
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
            definition_promise.Set(definition_events[0]->GetDefinedStatus());
          }
        };
    first_definition_event->ExecuteOrAddToFutureTasks(
        absl::StrFormat("async_wait_for_events_%p", &async_wait_for_events),
        std::move(async_wait_for_events));
  }

  return PjRtFuture<>(
      std::move(definition_promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "PjRtStreamExecutorBuffer::Await");
        VLOG(3) << "PjRtStreamExecutorBuffer::Await";
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
      buffer_on_device_shape.dimensions().size() == 1 &&
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
static absl::StatusOr<std::pair<ShapeTree<PjRtStreamExecutorExecutionInput>,
                                std::shared_ptr<BufferSequencingEvent>>>
MakeTupleHelper(
    PjRtStreamExecutorClient* client, LocalDeviceState* local_device,
    bool strict_shape_checking, const Shape& tupled_parameter_shape,
    absl::Span<PjRtBuffer* const> py_buffers,
    absl::Span<const PjRtStreamExecutorBuffer::ScopedHold> device_buffers,
    int device_ordinal) {
  se::DeviceMemoryAllocator* allocator = client->allocator();
  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();

  if (tupled_parameter_shape.tuple_shapes().size() != py_buffers.size()) {
    return InvalidArgument("Executable expected %lld parameters but got %lld",
                           tupled_parameter_shape.tuple_shapes().size(),
                           py_buffers.size());
  }
  for (int i = 0; i < py_buffers.size(); ++i) {
    TF_RETURN_IF_ERROR(CheckCompatibleShapes(
        strict_shape_checking, py_buffers[i]->on_device_shape(),
        tupled_parameter_shape.tuple_shapes(i), *transfer_manager, i));
  }

  se::Stream* stream = local_device->host_to_device_stream();
  TF_ASSIGN_OR_RETURN(
      se::OwningDeviceMemory owned_root_table_memory,
      allocator->Allocate(
          device_ordinal,
          transfer_manager->GetByteSizeRequirement(tupled_parameter_shape)));
  auto root_table_memory = owned_root_table_memory.cref();

  if (local_device->allocation_model() ==
      LocalDeviceState::kComputeSynchronized) {
    TF_RETURN_IF_ERROR(stream->WaitFor(local_device->compute_stream()));
  } else {
    DCHECK(transfer_manager->CanBufferBeAccessedNow(
        local_device->compute_stream()->parent(), root_table_memory));
  }

  ShapeTree<PjRtStreamExecutorExecutionInput> execution_input(
      tupled_parameter_shape);
  auto input_iterator = execution_input.begin();
  auto iterator_end = execution_input.end();
  // First set the root tuple table which is the first buffer in the ShapeTree.
  input_iterator->second = {
      true,
      RawSEDeviceMemory::Create(owned_root_table_memory.Release(),
                                local_device->local_device_id(), allocator)};
  ++input_iterator;
  // Then set each sub-tuple in turn from the parameters.
  for (const PjRtStreamExecutorBuffer::ScopedHold& device_buffer :
       device_buffers) {
    input_iterator->second = {
        device_buffer.type() == PjRtStreamExecutorBuffer::ScopedHold::kDonation,
        device_buffer->device_memory()};
    ++input_iterator;
  }
  CHECK(input_iterator == iterator_end);

  std::vector<se::DeviceMemoryBase> elements;
  size_t num_elements = ShapeUtil::TupleElementCount(tupled_parameter_shape);
  elements.reserve(num_elements);
  for (int64_t i = 0; i < num_elements; ++i) {
    elements.push_back(execution_input.element({i}).buf->mem());
  }

  TF_RETURN_IF_ERROR(transfer_manager->WriteSingleTupleIndexTable(
      stream, elements, tupled_parameter_shape, &root_table_memory));
  absl::StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().ThenAllocateAndRecordEvent(stream);
  if (!event_or.ok()) {
    StallStreamOnError(local_device, stream);
    return event_or.status();
  }

  auto transfer_event =
      std::make_shared<BufferSequencingEvent>(client->thread_pool());
  transfer_event->SetSequencingEvent(std::move(event_or).value(), stream);
  return std::make_pair(std::move(execution_input), std::move(transfer_event));
}

// Converts a ScopedShapedBuffer returned from an execution into a
// PjRtBuffer.
absl::StatusOr<std::unique_ptr<PjRtBuffer>> OutputBufferHelper(
    ShapeTree<tsl::RCReference<RawSEDeviceMemory>> result_buffer,
    std::shared_ptr<BufferSequencingEvent> definition_event, PjRtClient* client,
    PjRtDevice* device, LocalDeviceState* local_device,
    std::vector<tsl::RCReference<RawSEDeviceMemory>>& buffers_to_release) {
  if (result_buffer.shape().IsTuple()) {
    return absl::InternalError("OutputBufferHelper called on tuple.");
  }
  absl::InlinedVector<tsl::RCReference<RawSEDeviceMemory>, 1> buffers;
  for (auto& item : result_buffer) {
    buffers.push_back(std::move(item.second));
  }
  auto out_buffer = std::make_unique<TrackedDeviceBuffer>(
      device, std::move(buffers[0]),
      absl::Span<const std::shared_ptr<BufferSequencingEvent>>{
          definition_event});
  const Shape& shape = result_buffer.shape();
  PjRtMemorySpace* memory_space =
      device->default_memory_space().value_or(nullptr);
  if (shape.has_layout()) {
    switch (shape.layout().memory_space()) {
      case Layout::kGenericFastMemorySpace:
      case Layout::kDefaultMemorySpace:
        // Nothing to do, we have already set the default memory space.
        break;
      case Layout::kHostMemorySpace: {
        TF_ASSIGN_OR_RETURN(
            memory_space,
            tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                ->memory_space_by_kind_id(PinnedHostMemorySpace::kKindId));
        break;
      }
      default:
        return absl::InternalError(
            absl::StrCat("Unsupported memory space in output layout: ",
                         shape.layout().memory_space()));
    }
  }
  auto pjrt_buffer = std::make_unique<PjRtStreamExecutorBuffer>(
      result_buffer.shape(), std::move(out_buffer), client, device,
      memory_space);
  RecordUsage(pjrt_buffer->GetBufferWithUsageHold(), local_device, local_device,
              definition_event, local_device->compute_stream(),
              &buffers_to_release);
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
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(fingerprint_);
  for (auto& executable : executables) {
    const auto& computation_layout =
        executable->executable()->module().entry_computation_layout();
    std::vector<Shape> parameter_shapes;
    parameter_shapes.reserve(computation_layout.parameter_count());
    for (int i = 0; i < computation_layout.parameter_count(); ++i) {
      parameter_shapes.push_back(transfer_manager->HostShapeToDeviceShape(
          computation_layout.parameter_shape(i)));
    }
    fingerprint = tsl::FingerprintCat128(
        fingerprint,
        tsl::Fingerprint128(executable->executable()->module().ToString(
            HloPrintOptions::ModuleFingerprint())));
    executables_.emplace_back(std::move(executable));
    on_device_executable_parameter_shapes_.push_back(
        std::move(parameter_shapes));
  }
  fingerprint_ = absl::StrCat(fingerprint.low64, fingerprint.high64);

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

absl::StatusOr<std::vector<ShapeTree<PjRtStreamExecutorExecutionInput>>>
PjRtStreamExecutorLoadedExecutable::MakeExecutionInputsAndWaitForEvents(
    int device_ordinal, const ExecuteOptions& options,
    absl::Span<const Shape> executable_parameter_shapes,
    absl::Span<PjRtBuffer* const> argument_handles,
    absl::Span<const PjRtStreamExecutorBuffer::ScopedHold> device_buffers,
    absl::flat_hash_set<BufferSequencingEvent*>& events) const {
  std::vector<ShapeTree<PjRtStreamExecutorExecutionInput>> execution_inputs;
  LocalDeviceState* device_state = &(client_->device_state(device_ordinal));
  TransferManager* transfer_manager =
      client_->client()->backend().transfer_manager();
  // Lift tuple_write_event outside the conditional so that the event it
  // returns is not destroyed until after the loop below that waits on events.
  std::shared_ptr<BufferSequencingEvent> tuple_write_event;
  if (parameter_is_tupled_arguments_ && !options.arguments_are_tupled) {
    TF_ASSIGN_OR_RETURN(
        auto tuple_handle,
        MakeTupleHelper(client_, device_state, options.strict_shape_checking,
                        executable_parameter_shapes[0], argument_handles,
                        device_buffers, device_ordinal));
    tuple_write_event = std::move(tuple_handle.second);
    execution_inputs.emplace_back(std::move(tuple_handle.first));
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
      ShapeTree<PjRtStreamExecutorExecutionInput>& execution_input =
          execution_inputs.back();
      auto input_iterator = execution_input.begin();
      auto iterator_end = execution_input.end();
      const auto& buf = device_buffers[i]->device_memory();
      CHECK(input_iterator != iterator_end);
      input_iterator->second = {
          device_buffers[i].type() ==
              PjRtStreamExecutorBuffer::ScopedHold::kDonation,
          buf};
      ++input_iterator;
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

absl::StatusOr<PjRtStreamExecutorExecutionOutput>
PjRtStreamExecutorClient::RunAsync(
    LocalExecutable& exec, PjRtDevice* device,
    std::vector<ShapeTree<PjRtStreamExecutorExecutionInput>> arguments,
    ExecutableRunOptions run_options) {
  std::vector<ExecutionInput> xla_arguments;
  for (ShapeTree<PjRtStreamExecutorExecutionInput>& input : arguments) {
    xla_arguments.emplace_back(input.shape());
    auto& tmp = xla_arguments.back();
    auto it = tmp.MutableBuffers()->begin();
    for (auto& v : input) {
      if (v.second.is_donated) {
        it->second = MaybeOwningDeviceMemory(se::OwningDeviceMemory(
            v.second.buf->mem(), device->local_device_id().value(),
            run_options.allocator()));
        tmp.SetUnownedIndex(it->first);
      } else {
        it->second = MaybeOwningDeviceMemory(v.second.buf->mem());
      }
      ++it;
    }
  }

  TF_ASSIGN_OR_RETURN(
      ExecutionOutput output,
      exec.RunAsync(std::move(xla_arguments), std::move(run_options)));
  ScopedShapedBuffer ssb = output.ConsumeResult();
  xla::ShapeTree<tsl::RCReference<RawSEDeviceMemory>> results(
      ssb.on_device_shape());
  auto it = results.begin();
  se::DeviceMemoryAllocator* allocator = ssb.memory_allocator();
  ShapedBuffer released_ssb = ssb.release();
  for (auto& buf : released_ssb.buffers()) {
    CHECK(it != results.end());
    it->second = RawSEDeviceMemory::Create(
        buf.second, device->local_device_id(), allocator);
    ++it;
  }
  CHECK(it == results.end());
  for (ShapeTree<PjRtStreamExecutorExecutionInput>& input : arguments) {
    for (auto& v : input) {
      if (v.second.is_donated) {
        v.second.buf->UnsafeReleaseMemory();
      }
    }
  }
  return PjRtStreamExecutorExecutionOutput(
      {std::move(results), {}, output.ConsumeToBeReleased()});
}

// Enqueues a computation onto the compute stream. Each buffer returned in
// device_buffers has a usage hold added that must be dropped on error or
// converted on success.
// When `options` has non-zero `launch_id`, use `launch_id` instead of `run_id`
// to initialize `run_options`.
absl::StatusOr<ShapeTree<tsl::RCReference<RawSEDeviceMemory>>>
PjRtStreamExecutorLoadedExecutable::EnqueueExecution(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    int executable_idx, const RunId& run_id, const ExecuteOptions& options,
    PjRtDevice* device,
    std::vector<PjRtStreamExecutorBuffer::ScopedHold>* device_buffers,
    std::shared_ptr<DeviceAssignment> device_assignment,
    std::vector<absl::AnyInvocable<void() &&>>& compute_callbacks) const {
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
  absl::flat_hash_map<const void*, std::pair<bool, int>> donation_clashes;
  donation_clashes.reserve(argument_handles.size());
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
    TF_RETURN_IF_ERROR(TestBufferDonationClashes(
        handle, donation_clashes, must_donate, i, replica, partition));
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
      std::vector<ShapeTree<PjRtStreamExecutorExecutionInput>> execution_inputs,
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
  run_options.set_device_ordinal(device_state->local_device_id().value());
  run_options.set_local_device_count(client_->client()->device_count());

  run_options.set_physical_device_ordinal(
      device_state->local_hardware_id().value());
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
  run_options.set_execution_profile(options.execution_profile);
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

  auto start_time_ns = std::make_shared<uint64_t>();
  std::optional<uint64_t> key = xla::GetDeviceTimeMeasurementKey();
  // Record the start time of the execution by placing a callback on the stream
  // directly before the execution. If this callback is added, another callback
  // will be added directly after the execution to record the elapsed device
  // time.
  if (key.has_value()) {
    TF_RETURN_IF_ERROR(device_state->ThenExecuteCallback(
        device_state->compute_stream(), [start_time_ns]() {
          *start_time_ns = tsl::Env::Default()->NowNanos();
        }));
  }

  VLOG(1) << "Start calling RunAsync for "
          << executables_[executable_idx]->executable()->module().name()
          << ", device=" << device->DebugString()
          << ", run_id=" << run_options.run_id().ToInt();

  if (VLOG_IS_ON(2)) {
    auto executable_name =
        executables_[executable_idx]->executable()->module().name();
    absl::Status host_callback_status = run_options.stream()->DoHostCallback(
        [executable_name, launch_id(run_options.run_id().ToInt()), device]() {
          VLOG(2) << "Start device execution for " << executable_name
                  << ", launch_id: " << launch_id
                  << ", device: " << device->DebugString();
        });
    if (!host_callback_status.ok()) {
      LOG(WARNING)
          << "Failed to do host callback for start device execution for "
          << executable_name << ", status = " << host_callback_status;
    }
  }

  absl::StatusOr<PjRtStreamExecutorExecutionOutput> result_buffer_or_status =
      client_->RunAsync(*executables_[executable_idx], device,
                        std::move(execution_inputs), run_options);

  if (VLOG_IS_ON(2)) {
    auto executable_name =
        executables_[executable_idx]->executable()->module().name();
    absl::Status host_callback_status = run_options.stream()->DoHostCallback(
        [executable_name, launch_id(run_options.run_id().ToInt()), device]() {
          VLOG(2) << "Finish device execution for " << executable_name
                  << ", launch_id: " << launch_id
                  << ", device: " << device->DebugString();
        });
    if (!host_callback_status.ok()) {
      LOG(WARNING)
          << "Failed to do host callback for start device execution for "
          << executable_name << ", status = " << host_callback_status;
    }
  }

  VLOG(1) << "Finish calling RunAsync for "
          << executables_[executable_idx]->executable()->module().name()
          << ", device=" << device->DebugString()
          << ", run_id=" << run_options.run_id().ToInt()
          << ", replica=" << replica << ", partition=" << partition
          << ", completed, ok=" << result_buffer_or_status.ok();

  if (!result_buffer_or_status.ok()) {
    return result_buffer_or_status.status();
  }

  // Add a callback on the stream to record the elapsed device time of the
  // executable execution.
  //
  // Do not place other callbacks between the callback recording the start time
  // and this callback because their execution time will incorrectly count
  // toward device execution time.
  //
  // This callback is only added if there is a valid key to guarantee that
  // either both or none of the device time measurement callbacks are added to
  // the stream, and to avoid needing a mutex.
  if (key.has_value()) {
    TF_RETURN_IF_ERROR(device_state->ThenExecuteCallback(
        device_state->compute_stream(),
        [key, start_time_ns,
         device_type = GetDeviceType(client_->platform_id())]() {
          auto elapsed = absl::FromUnixNanos(tsl::Env::Default()->NowNanos()) -
                         absl::FromUnixNanos(*start_time_ns);
          xla::RecordDeviceTimeMeasurement(*key, elapsed, device_type);
        }));
  }

  if (device_state->allocation_model() == LocalDeviceState::kSynchronous) {
    // If we used a transient tuple for the arguments we donated its root table
    // buffer. In that case, and/or if we donated any input buffers that were
    // not aliased, the donated buffers are going to be passed back to us via
    // the execution output. We need to ensure they aren't freed until after
    // execution completes. (Currently XLA does not support aliasing tuple
    // tables, so if any donated parameter is a tuple there will be donated but
    // unaliased buffers.)
    compute_callbacks.push_back(
        [donated_memory = std::move(result_buffer_or_status->to_be_released),
         se_donated_memory =
             std::move(result_buffer_or_status->se_to_be_released),
         exe = executables_[executable_idx]]() mutable {});
  } else {
    // Any donated memory returned by the ExecutionOutput can be immediately
    // freed.
    compute_callbacks.push_back(
        [to_release{std::make_tuple(executables_[executable_idx],
                                    compute_reservation,
                                    device_assignment)}]() {});
  }

  return std::move(std::move(result_buffer_or_status).value().result);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
PjRtStreamExecutorLoadedExecutable::MakeOutputBuffers(
    int device_ordinal, const ExecuteOptions& options,
    ShapeTree<tsl::RCReference<RawSEDeviceMemory>> result_buffer,
    std::shared_ptr<BufferSequencingEvent> definition_event, PjRtDevice* device,
    std::vector<absl::AnyInvocable<void() &&>>& compute_callbacks,
    std::vector<tsl::RCReference<RawSEDeviceMemory>>& buffers_to_release)
    const {
  tsl::profiler::TraceMe traceme("MakeOutputBuffers");
  std::vector<std::unique_ptr<PjRtBuffer>> outputs;
  LocalDeviceState* device_state = &(client_->device_state(device_ordinal));
  if (result_buffer.shape().IsTuple()) {
    int tuple_count = result_buffer.shape().tuple_shapes().size();
    outputs.reserve(tuple_count);
    // Take ownership of each of the output values, leaving only the root table
    // in result_buffer.
    for (int i = 0; i < tuple_count; ++i) {
      TF_ASSIGN_OR_RETURN(
          ShapeTree<tsl::RCReference<RawSEDeviceMemory>> tuple_buffer,
          result_buffer.SubShapeTree({i}));
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<PjRtBuffer> buffer,
          OutputBufferHelper(std::move(tuple_buffer), definition_event, client_,
                             device, device_state, buffers_to_release));
      outputs.push_back(std::move(buffer));
    }
    if (device_state->allocation_model() == LocalDeviceState::kSynchronous) {
      // Don't release the root buffer until after execution completes.
      auto root_buffer = result_buffer.find({})->second;
      compute_callbacks.push_back([root_buffer = std::move(root_buffer)]() {});
    }
  } else {
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<PjRtBuffer> buffer,
        OutputBufferHelper(std::move(result_buffer), definition_event, client_,
                           device, device_state, buffers_to_release));
    outputs.push_back(std::move(buffer));
  }
  return outputs;
}

static absl::Status GetFirstInputError(
    absl::Span<PjRtBuffer* const> argument_handles) {
  for (auto* handle : argument_handles) {
    auto* buffer = tensorflow::down_cast<PjRtStreamExecutorBuffer*>(handle);
    PjRtStreamExecutorBuffer::ScopedHold hold =
        buffer->GetBufferWithUsageHold();
    if (!hold.ok()) {
      return hold.status();
    }
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

  std::vector<absl::AnyInvocable<void() &&>> compute_callbacks;
  std::vector<PjRtStreamExecutorBuffer::ScopedHold> device_buffers;
  device_buffers.reserve(argument_handles.size());
  absl::StatusOr<ShapeTree<tsl::RCReference<RawSEDeviceMemory>>>
      result_buffer_or_status =
          EnqueueExecution(argument_handles, replica, partition, executable_idx,
                           run_id, options, device, &device_buffers,
                           std::move(device_assignment), compute_callbacks);

  if (!result_buffer_or_status.ok()) {
    LOG(ERROR) << "Execution of replica " << replica
               << " failed: " << result_buffer_or_status.status();
    return result_buffer_or_status.status();
  }
  ShapeTree<tsl::RCReference<RawSEDeviceMemory>> result_buffer =
      std::move(result_buffer_or_status).value();

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
  std::vector<tsl::RCReference<RawSEDeviceMemory>> buffers_to_release;
  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<PjRtBuffer>> outputs,
      MakeOutputBuffers(device_ordinal, options, std::move(result_buffer),
                        definition_event, device, compute_callbacks,
                        buffers_to_release));

  for (PjRtStreamExecutorBuffer::ScopedHold& b : device_buffers) {
    if (b.type() == PjRtStreamExecutorBuffer::ScopedHold::kUsage) {
      RecordUsage(std::move(b), device_state, device_state, definition_event,
                  stream, &buffers_to_release);
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
               buffers_to_release{std::move(buffers_to_release)}]() mutable {
        for (auto& fn : callbacks) {
          std::move(fn)();
        }
        callbacks.clear();
      }));
  metrics::ReportExecutableEnqueueTime(tsl::Env::Default()->NowMicros() -
                                       start_time_usecs);
  return Result({/*future=*/std::move(future), /*buffers=*/std::move(outputs)});
}

absl::Status PjRtStreamExecutorLoadedExecutable::VerifyCompatibleDevices()
    const {
  const int num_addressable_devices = addressable_devices_.size();
  for (int i = 0; i < num_addressable_devices; ++i) {
    PjRtDevice* device = addressable_devices_[i];
    const int device_ordinal =
        tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
            ->local_device_state()
            ->local_device_id()
            .value();
    const int partition = addressable_device_logical_ids_[i].partition;
    const int executable_idx = executables_.size() > 1 ? partition : 0;
    TF_RETURN_IF_ERROR(executables_[executable_idx]->VerifyRunDeviceCompatible(
        device_ordinal));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
PjRtStreamExecutorLoadedExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  if (device_assignment_ == nullptr) {
    return InvalidArgument("Execute expects a non-null device_assignment");
  }
  if (input_hlo_snapshot_bits_.has_value()) {
    HloUnoptimizedSnapshot hlo_snapshot;
    *hlo_snapshot.mutable_hlo_module() = input_hlo_snapshot_bits_->hlo_module;
    for (const auto& argument_handle : argument_handles) {
      HloInputs hlo_inputs;
      for (const auto& buffer : argument_handle) {
        TF_ASSIGN_OR_RETURN(auto literal, buffer->ToLiteralSync());
        *hlo_inputs.add_arguments() = literal->ToProto();
      }
      *hlo_snapshot.add_partitions() = std::move(hlo_inputs);
    }
    DumpHloUnoptimizedSnapshotIfEnabled(
        hlo_snapshot, input_hlo_snapshot_bits_->debug_options);
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

  TF_RETURN_IF_ERROR(VerifyCompatibleDevices());
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
    std::unique_ptr<ProfilingContext> pc = CreateProfilingContext();
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
        std::unique_ptr<WithProfilingContext> wpc =
            CreateWithProfilingContext(pc.get());
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

namespace {

absl::StatusOr<absl::string_view> MemoryKindFromSimpleShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.has_layout()) {
    return default_memory_kind;
  }
  switch (shape.layout().memory_space()) {
    case Layout::kHostMemorySpace:
      return PinnedHostMemorySpace::kKind;
    case Layout::kGenericFastMemorySpace:
    case Layout::kDefaultMemorySpace:
      return default_memory_kind;
    default:
      return InvalidArgument("Unexpected memory space %d in output layout",
                             shape.layout().memory_space());
  }
}

absl::StatusOr<std::vector<absl::string_view>> MemoryKindsFromShape(
    const Shape& shape, absl::string_view default_memory_kind) {
  if (!shape.IsTuple()) {
    TF_ASSIGN_OR_RETURN(absl::string_view memory_kind,
                        MemoryKindFromSimpleShape(shape, default_memory_kind));
    return {{memory_kind}};
  }
  std::vector<absl::string_view> result;
  result.reserve(shape.tuple_shapes().size());
  for (const auto& element_shape : shape.tuple_shapes()) {
    TF_ASSIGN_OR_RETURN(
        absl::string_view element_memory_kind,
        MemoryKindFromSimpleShape(element_shape, default_memory_kind));
    result.push_back(element_memory_kind);
  }
  return result;
}

}  // namespace

absl::StatusOr<std::vector<std::vector<absl::string_view>>>
PjRtStreamExecutorLoadedExecutable::GetOutputMemoryKinds() const {
  TF_ASSIGN_OR_RETURN(auto shapes, GetOutputShapes());
  if (addressable_devices().empty()) {
    return Unimplemented(
        "GetOutputMemoryKinds is not supported when there are no addressable "
        "devices in PjRtStreamExecutorLoadedExecutable.");
  }
  TF_ASSIGN_OR_RETURN(PjRtMemorySpace * default_memory_space,
                      addressable_devices()[0]->default_memory_space());
  std::vector<std::vector<absl::string_view>> out;
  out.reserve(shapes.size());
  for (const auto& shape : shapes) {
    TF_ASSIGN_OR_RETURN(
        std::vector<absl::string_view> memory_kind,
        MemoryKindsFromShape(shape, default_memory_space->kind()));
    out.push_back(memory_kind);
  }
  return out;
}

absl::Status PjRtStreamExecutorClient::UpdateCompileOptions(
    CompileOptions* options, bool lookup_addressable_devices) {
  return UpdateCompileOptionsInternal(options, /*returned_extras=*/nullptr,
                                      lookup_addressable_devices);
}

absl::StatusOr<PjRtStreamExecutorClient::ExecutableExtras>
PjRtStreamExecutorClient::UpdateCompileOptionsAndGetExecutableExtras(
    CompileOptions* options) {
  ExecutableExtras extras;
  TF_RETURN_IF_ERROR(UpdateCompileOptionsInternal(
      options, &extras, /*lookup_addressable_devices=*/true));
  return extras;
}

absl::Status PjRtStreamExecutorClient::UpdateCompileOptionsInternal(
    CompileOptions* options, ExecutableExtras* returned_extras,
    bool lookup_addressable_devices) {
  ExecutableBuildOptions& build_options = options->executable_build_options;
  if (!build_options.compile_thread_pool()) {
    build_options.set_compile_thread_pool(thread_pool());
  }
  if (!build_options.device_allocator()) {
    build_options.set_device_allocator(allocator());
  }

  auto layout_callback = [local_client = client(),
                          options](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    ExecutableBuildOptions build_options = options->executable_build_options;
    std::vector<const Shape*> argument_layout_pointers;
    std::optional<std::vector<Shape>> argument_layouts =
        options->argument_layouts;
    Shape result_layout;
    const bool allow_auto_layout =
        build_options.has_debug_options() &&
        build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
    TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
        XlaComputation(module.ToProto()),
        [local_client,
         allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
          if (allow_auto_layout && !shape.has_layout()) {
            return shape;
          }
          return local_client->backend()
              .transfer_manager()
              ->ChooseCompactLayoutForShape(shape);
        },
        argument_layouts, &build_options, &argument_layout_pointers));
    result_layout = *build_options.result_layout();
    return std::make_pair(*argument_layouts, result_layout);
  };

  build_options.set_layout_canonicalization_callback(layout_callback);

  // We don't look up devices when it is not required. It could fail if
  // we look up a device ID on a client with a different topology.
  // Note that we always look up devices for XLA GPU shard autotuning, as it
  // needs to know the number of processes and the current process index.
  const bool use_xla_gpu_shard_autotuning =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_gpu_shard_autotuning();
  if (!lookup_addressable_devices && !use_xla_gpu_shard_autotuning) {
    if (build_options.device_ordinal() < 0) {
      build_options.set_device_ordinal(0);
    }
    return absl::OkStatus();
  }

  ExecutableExtras extras;
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<PjRtStreamExecutorLoadedExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

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
    absl::flat_hash_set<int> all_process_indices;
    std::optional<int> this_process_index;
    for (int replica = 0; replica < num_replicas; ++replica) {
      for (int partition = 0; partition < num_partitions; ++partition) {
        int64_t device_id = (*device_assignment)(replica, partition);
        PjRtGlobalDeviceId global_device_id(device_id);

        TF_ASSIGN_OR_RETURN(PjRtDevice * device,
                            LookupDevice(global_device_id));
        all_process_indices.insert(device->process_index());
        if (device->process_index() != process_index()) {
          VLOG(3) << "Non-local device: " << device_id;
          continue;
        }
        if (!this_process_index.has_value()) {
          this_process_index = all_process_indices.size() - 1;
        }
        PjRtLoadedExecutable::LogicalDeviceIds logica_device_ids;
        logica_device_ids.replica = replica;
        logica_device_ids.partition = partition;
        addressable_device_logical_ids.push_back(std::move(logica_device_ids));
        addressable_devices.push_back(device);
      }
    }
    if (addressable_devices.empty()) {
      if (build_options.device_ordinal() < 0) {
        build_options.set_device_ordinal(0);
      }
    } else {
      if (build_options.device_ordinal() < 0) {
        build_options.set_device_ordinal(
            addressable_devices.front()->local_hardware_id().value());
      }
      build_options.set_process_index(*this_process_index);
      build_options.set_process_count(all_process_indices.size());
    }
  }
  if (returned_extras != nullptr) {
    *returned_extras = std::move(extras);
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
PjRtStreamExecutorClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options, bool lookup_addressable_devices) {
  tsl::profiler::TraceMe traceme("PjRtStreamExecutorClient::CompileInternal");
  VLOG(1) << "PjRtStreamExecutorClient::CompileInternal";
  if (key_value_store().has_value() &&
      !options.executable_build_options.key_value_store()) {
    options.executable_build_options.set_key_value_store(*key_value_store());
  }
  auto input_options = options;

  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());
  TF_RETURN_IF_ERROR(
      UpdateCompileOptions(&options, lookup_addressable_devices));

  // It is important to set the canonicalization callback after creating
  // a copy of the options so that the executable's options remain without
  // the callback - the callback would break the executable's serializability.
  if (layout_canonicalization_callback) {
    options.executable_build_options.set_layout_canonicalization_callback(
        layout_canonicalization_callback);
  }

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      client()->Compile(computation, argument_layout_pointers,
                        options.executable_build_options));

  return BuildPjRtExecutable(std::move(local_executables), input_options);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
PjRtStreamExecutorClient::Compile(const XlaComputation& computation,
                                  CompileOptions options) {
  return Compile(computation, options, /*lookup_addressable_devices=*/false);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
PjRtStreamExecutorClient::Compile(const XlaComputation& computation,
                                  CompileOptions options,
                                  bool lookup_addressable_devices) {
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [local_client = client(),
       allow_auto_layout](Shape shape) -> absl::StatusOr<Shape> {
        if (allow_auto_layout && !shape.has_layout()) {
          return shape;
        }
        return local_client->backend()
            .transfer_manager()
            ->ChooseCompactLayoutForShape(shape);
      },
      options.argument_layouts, &options.executable_build_options,
      &argument_layout_pointers));
  return CompileInternal(computation, argument_layout_pointers,
                         /* layout_canonicalization_callback = */ nullptr,
                         options, lookup_addressable_devices);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
PjRtStreamExecutorClient::Compile(mlir::ModuleOp module,
                                  CompileOptions options) {
  return Compile(module, options, /*lookup_addressable_devices=*/false);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
PjRtStreamExecutorClient::Compile(mlir::ModuleOp module, CompileOptions options,
                                  bool lookup_addressable_devices) {
  XlaComputation xla_computation;
  const ExecutableBuildOptions& exec_build_options =
      options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, exec_build_options.use_shardy_partitioner()));

  // If the compile options specify argument layout, then let's
  // fall back to using the options to determine layouts.
  if (options.argument_layouts) {
    return Compile(xla_computation, options, lookup_addressable_devices);
  }

  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> arg_layout_modes,
                      GetArgLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<LayoutMode> out_layout_modes,
                      GetOutputLayoutModes(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> arg_memory_spaces,
                      GetArgMemoryKinds(module));
  TF_ASSIGN_OR_RETURN(std::vector<MemorySpaceColor> out_memory_spaces,
                      GetOutputMemoryKinds(module));

  // If auto-sharding modifies shapes of arguments and/or result,
  // we get a callback to restore the layouts. Let us restore the layouts
  // according to the attributes we parsed from MLIR.
  auto layout_callback = [local_client = client(), &arg_layout_modes,
                          &out_layout_modes, &arg_memory_spaces,
                          &out_memory_spaces](const HloModule& module)
      -> absl::StatusOr<std::pair<std::vector<Shape>, Shape>> {
    XlaComputation xla_computation(XlaComputation(module.ToProto()));
    return LayoutModesToXlaShapes(
        xla_computation, arg_layout_modes, out_layout_modes, arg_memory_spaces,
        out_memory_spaces,
        [local_client](Shape shape) -> absl::StatusOr<Shape> {
          return local_client->backend()
              .transfer_manager()
              ->ChooseCompactLayoutForShape(shape);
        });
  };

  // This call will update result_layout in options.executable_build_options.
  TF_ASSIGN_OR_RETURN(auto arg_layouts_and_pointers,
                      LayoutModesToXla(
                          xla_computation, arg_layout_modes, out_layout_modes,
                          arg_memory_spaces, out_memory_spaces,
                          [this](Shape shape) -> absl::StatusOr<Shape> {
                            return this->client()
                                ->backend()
                                .transfer_manager()
                                ->ChooseCompactLayoutForShape(shape);
                          },
                          options.executable_build_options));

  return CompileInternal(xla_computation, arg_layouts_and_pointers.second,
                         layout_callback, options, lookup_addressable_devices);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::CompileAndLoad(const XlaComputation& computation,
                                         CompileOptions options) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtExecutable> executable,
      Compile(computation, options, /*lookup_addressable_devices=*/true));
  return Load(std::move(executable), LoadOptions());
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::CompileAndLoad(mlir::ModuleOp module,
                                         CompileOptions options) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtExecutable> executable,
      Compile(module, options, /*lookup_addressable_devices=*/true));
  return Load(std::move(executable), LoadOptions());
}

namespace {

constexpr absl::string_view kPjRtClientName = "PjRtStreamExecutorClient";

}  // namespace

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
  *proto.mutable_pjrt_client_name() = kPjRtClientName;
  return proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
PjRtStreamExecutorClient::BuildPjRtExecutable(
    std::vector<std::unique_ptr<LocalExecutable>> local_executables,
    CompileOptions compile_options) {
  if (local_executables.empty()) {
    return Internal("No local executable");
  }
  if (local_executables.size() != 1) {
    return Unimplemented("Multiple executables are not supported");
  }
  Executable* built_executable = local_executables[0]->executable();
  if (!built_executable->has_module()) {
    return absl::InternalError("Executable does not have HLO modules.");
  }
  const auto& hlo_module = built_executable->module();

  const int num_replicas = hlo_module.config().replica_count();
  const int num_partitions = hlo_module.config().num_partitions();
  const std::string name = hlo_module.name();
  const std::string fingerprint = hlo_module.GetFingerprint128();

  return std::make_unique<StreamExecutorExecutable>(
      std::move(compile_options), std::move(local_executables), client_,
      num_replicas, num_partitions, name, fingerprint,
      memory_spaces()[0]->kind());
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
PjRtStreamExecutorClient::DeserializeExecutable(
    absl::string_view serialized,
    std::optional<CompileOptions> compile_options) {
  TF_ASSIGN_OR_RETURN(
      auto local_executables_and_options,
      DeserializeToLocalExecutable(serialized, compile_options));

  return BuildPjRtExecutable(std::move(local_executables_and_options.first),
                             local_executables_and_options.second);
}

absl::StatusOr<
    std::pair<std::vector<std::unique_ptr<LocalExecutable>>, CompileOptions>>
PjRtStreamExecutorClient::DeserializeToLocalExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options) {
  ExecutableAndOptionsProto proto;
  if (serialized.size() > std::numeric_limits<int>::max()) {
    return Internal("Proto is too large (>2GB)");
  }
  if (!proto.ParseFromArray(serialized.data(), serialized.size())) {
    return Internal("Proto deserialization failed");
  }
  if (!proto.pjrt_client_name().empty() &&
      proto.pjrt_client_name() != kPjRtClientName) {
    return Internal(
        "Serialized executable is from an incompatible PjRt client type. "
        "PjRt client type expected by the serialized executable: %s",
        proto.pjrt_client_name());
  }

  CompileOptions compile_options;
  if (options.has_value()) {
    compile_options = *std::move(options);
  } else {
    TF_ASSIGN_OR_RETURN(compile_options,
                        CompileOptions::FromProto(proto.compile_options()));
  }

  tsl::profiler::TraceMe traceme(
      "PjRtStreamExecutorClient::DeserializeToLocalExecutable");
  VLOG(1) << "PjRtStreamExecutorClient::DeserializeToLocalExecutable";

  std::string str = std::move(*proto.mutable_serialized_executable());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LocalExecutable> loaded,
      client()->Load(str, compile_options.executable_build_options));

  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.push_back(std::move(loaded));

  return std::make_pair(std::move(local_executables), compile_options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::LoadSerializedExecutable(
    absl::string_view serialized, std::optional<CompileOptions> options,
    const LoadOptions& load_options) {
  TF_ASSIGN_OR_RETURN(auto local_executables_and_options,
                      DeserializeToLocalExecutable(serialized, options));
  return LoadInternal(std::move(local_executables_and_options.first),
                      local_executables_and_options.second);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::LoadInternal(
    std::vector<std::unique_ptr<LocalExecutable>> local_executables,
    CompileOptions compile_options) {
  auto input_options = compile_options;

  TF_RETURN_IF_ERROR(compile_options.ApplyAllOptionOverrides());

  TF_ASSIGN_OR_RETURN(
      ExecutableExtras extras,
      UpdateCompileOptionsAndGetExecutableExtras(&compile_options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<PjRtStreamExecutorLoadedExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  const auto& ex_options = compile_options.executable_build_options;
  const bool xla_gpu_dump_hlo_unoptimized_snapshots =
      ex_options.has_debug_options() &&
      ex_options.debug_options().xla_gpu_dump_hlo_unoptimized_snapshots();
  HloModuleProto hlo_module_proto;
  if (xla_gpu_dump_hlo_unoptimized_snapshots) {
    hlo_module_proto = local_executables[0]->executable()->module().ToProto();
  }

  auto executable = std::make_unique<PjRtStreamExecutorLoadedExecutable>(
      std::move(local_executables),
      compile_options.parameter_is_tupled_arguments,
      std::move(device_assignment), std::move(input_options),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);

  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(compile_options.parameter_is_tupled_arguments));
  if (xla_gpu_dump_hlo_unoptimized_snapshots) {
    executable->SetInputHloSnapshotBits(
        std::move(hlo_module_proto),
        compile_options.executable_build_options.debug_options());
  }
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
PjRtStreamExecutorClient::Load(std::unique_ptr<PjRtExecutable> executable,
                               const LoadOptions& load_options) {
  auto se_executable = absl::WrapUnique(
      tensorflow::down_cast<StreamExecutorExecutable*>(executable.release()));
  CompileOptions compile_options = se_executable->compile_options();

  tsl::profiler::TraceMe traceme("PjRtStreamExecutorClient::Load");
  VLOG(1) << "PjRtStreamExecutorClient::Load";

  TF_ASSIGN_OR_RETURN(auto local_executables, se_executable->ConsumeExecutable(
                                                  client(), compile_options));
  return LoadInternal(std::move(local_executables), compile_options);
}

bool PjRtStreamExecutorClient::IsDmaMapped(const void* data_start,
                                           int64_t transfer_size) {
  absl::MutexLock lock(&dma_maps_mutex_);
  if (!dma_maps_.empty()) {
    void* data_end = (char*)data_start + transfer_size;
    for (const auto& [map_start, map_size] : dma_maps_) {
      void* map_end = (char*)map_start + map_size;
      if (data_start >= map_start && data_end <= map_end) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace xla
