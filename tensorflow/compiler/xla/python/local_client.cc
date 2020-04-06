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
// Synchronization between streams occurs via BufferDefinitionEvents that
// describe when the contents of a logical buffer are known to be valid on
// a particular stream, and when a buffer's uses have all completed.
//
// Synchronous vs asynchronous deallocation:
// -----------------------------------------
//
// See the comment on LocalDeviceState::AllocationModel for a discussion of the
// different allocation semantics on CPU, GPU, and TPU.

#include "tensorflow/compiler/xla/python/local_client.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/container/inlined_vector.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/python/distributed/protocol.pb.h"
#include "tensorflow/compiler/xla/python/event_pool.h"
#include "tensorflow/compiler/xla/python/local_device_state.h"
#include "tensorflow/compiler/xla/python/shared_device_buffer.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/stream_executor/device_memory.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"
#include "tensorflow/stream_executor/event.h"
#include "tensorflow/stream_executor/lib/statusor.h"
#include "tensorflow/stream_executor/stream.h"

namespace xla {

StatusOr<LocalDeviceState*> Device::GetLocalDeviceState() const {
  if (local_device_state_) {
    return local_device_state_.get();
  }
  return InvalidArgument("Device %s is not a local device.", DebugString());
}

std::string Device::DebugString() const {
  return absl::StrCat(platform_name(), ":", id());
}

StatusOr<DeviceAssignment> DevicesToDeviceAssignment(
    absl::Span<const std::vector<Device*>> devices) {
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
      if (devices[0][0]->platform_name() !=
          devices[replica][partition]->platform_name()) {
        return InvalidArgument(
            "Device assignment passed to Compile() must have devices of a "
            "single kind, got %s for replica 0 partition 0 and %s for replica "
            "%d partition %d.",
            devices[0][0]->platform_name(),
            devices[replica][partition]->platform_name(), replica, partition);
      }
      xla_assignment(replica, partition) = devices[replica][partition]->id();
    }
  }
  return xla_assignment;
}

PyLocalClient::PyLocalClient(
    std::string platform_name, LocalClient* client,
    std::vector<std::unique_ptr<Device>> devices, int host_id,
    std::unique_ptr<se::DeviceMemoryAllocator> allocator,
    std::unique_ptr<tensorflow::Allocator> host_memory_allocator,
    std::unique_ptr<GpuExecutableRunOptions> gpu_run_options)
    : platform_name_(std::move(platform_name)),
      client_(client),
      devices_(std::move(devices)),
      host_id_(host_id),
      owned_allocator_(std::move(allocator)),
      host_memory_allocator_(std::move(host_memory_allocator)),
      gpu_run_options_(std::move(gpu_run_options)),
      h2d_transfer_pool_(tensorflow::Env::Default(), "py_xla_h2d_transfer",
                         client->device_count()) {
  if (owned_allocator_ != nullptr) {
    allocator_ = owned_allocator_.get();
  } else {
    allocator_ = client_->backend().memory_allocator();
  }

  for (const std::unique_ptr<Device>& device : devices_) {
    CHECK(id_to_device_.insert({device->id(), device.get()}).second)
        << "Duplicate device id: " << device->id();

    if (device->local_device_state()) {
      int idx = device->local_device_state()->device_ordinal();
      if (idx >= local_devices_.size()) {
        local_devices_.resize(idx + 1);
      }
      CHECK(local_devices_[idx] == nullptr) << idx;
      local_devices_[idx] = device.get();
    }
  }
  for (int idx = 0; idx < local_devices_.size(); ++idx) {
    CHECK(local_devices_[idx] != nullptr) << idx;
  }
}

StatusOr<DeviceAssignment> PyLocalClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return client_->backend().computation_placer()->AssignDevices(num_replicas,
                                                                num_partitions);
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
void RecordUsage(SharedDeviceBuffer::ScopedUsage device_buffer,
                 LocalDeviceState* buffer_local_device,
                 LocalDeviceState* stream_local_device,
                 std::shared_ptr<BufferDefinitionEvent> event,
                 se::Stream* usage_stream, bool prefer_to_retain_reference) {
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
    buffer_local_device->ThenRelease(usage_stream,
                                     device_buffer.buffer_reference());
  }
  device_buffer.Convert(usage_stream, event, retain_buffer_until_completion);
}

// Allocates the device buffers for a buffer that will be used as the
// destination of a copy, either from the host or another device. copy_stream
// may be nullptr, e.g., when allocating a buffer for a cross-host copy. If the
// buffer is a tuple then the tuple tables are allocated, and all necessary
// synchronization for them is dealt with, before the buffer is returned.
//
// It is safe to delete the returned PyLocalBuffer without further
// synchronization if an error occurs before the buffer is used.
StatusOr<std::unique_ptr<PyLocalBuffer>> AllocateDestinationBuffer(
    const Shape& on_host_shape, Device* device, LocalDeviceState* local_device,
    se::Stream* copy_stream, PyLocalClient* client) {
  if (on_host_shape.IsTuple() && on_host_shape.tuple_shapes_size() == 0) {
    return InvalidArgument("Can't make a buffer from an empty tuple");
  }

  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer dst_buffer,
      transfer_manager->AllocateScopedShapedBuffer(
          on_host_shape, client->allocator(), local_device->device_ordinal()));
  if (local_device->allocation_model() ==
      LocalDeviceState::kComputeSynchronized) {
    CHECK(copy_stream != nullptr);
    copy_stream->ThenWaitFor(local_device->compute_stream());
  } else {
    DCHECK(transfer_manager->CanShapedBufferBeAccessedNow(
        local_device->compute_stream()->parent(), dst_buffer));
  }
  Shape on_device_shape = dst_buffer.on_device_shape();

  absl::InlinedVector<std::shared_ptr<BufferDefinitionEvent>, 2>
      definition_events;
  // We always have at least one definition event, for the copy completing to
  // the device buffers.
  definition_events.emplace_back(std::make_shared<BufferDefinitionEvent>());
  se::Stream* tuple_table_stream = local_device->host_to_device_stream();
  if (on_device_shape.IsTuple()) {
    // We also need to copy the tuple tables, so we'll have a second defintion
    // event for that copy to complete.
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

    definition_events.emplace_back(std::make_shared<BufferDefinitionEvent>());
    StatusOr<EventPool::Handle> event_or =
        local_device->event_pool().ThenAllocateAndRecordEvent(
            tuple_table_stream);
    if (!event_or.ok()) {
      StallStreamOnError(local_device, tuple_table_stream);
      return event_or.status();
    }
    definition_events[1]->SetDefinitionEvent(event_or.ConsumeValueOrDie(),
                                             tuple_table_stream);
  }
  std::shared_ptr<SharedDeviceBuffer> dst_device_buffer =
      SharedDeviceBuffer::FromScopedShapedBuffer(&dst_buffer,
                                                 definition_events);
  if (on_device_shape.IsTuple()) {
    // Add a usage hold for the tuple table write and immediately convert it to
    // the appropriate form of synchronization. prefer_to_retain_reference=false
    // means don't retain a memory reference until the transfer is complete when
    // using the ComputeSynchronized allocation model. This is a heuristic
    // because in the common case destination buffers will be used on the
    // compute stream and therefore don't require any synchronization before
    // being freed. If the buffer is allocated and never used, the free will
    // take longer and this is assumed to be ok.
    RecordUsage(
        std::move(SharedDeviceBuffer::ScopedUsage().Acquire(dst_device_buffer)),
        local_device, local_device, definition_events[1], tuple_table_stream,
        /*prefer_to_retain_reference=*/false);
  }

  return absl::make_unique<PyLocalBuffer>(on_host_shape, on_device_shape,
                                          std::move(dst_device_buffer), client,
                                          device);
}

// Adds necessary synchronization after a copy has been enqueued to a buffer.
// definition_event was added when the buffer was allocated, but has not yet
// had an event recorded.
Status AddDestinationBufferSynchronization(
    LocalDeviceState* local_device,
    SharedDeviceBuffer::ScopedUsage device_buffer,
    std::shared_ptr<BufferDefinitionEvent> definition_event,
    se::Stream* copy_stream) {
  StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().ThenAllocateAndRecordEvent(copy_stream);
  if (!event_or.ok()) {
    StallStreamOnError(local_device, copy_stream);
    return event_or.status();
  }
  definition_event->SetDefinitionEvent(event_or.ConsumeValueOrDie(),
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
  return Status::OK();
}

}  // namespace

/* static */
StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalBuffer::FromHostBuffer(
    const void* data, const Shape& shape, bool force_copy,
    std::shared_ptr<void> buffer_reference, PyLocalClient* client,
    Device* device) {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::FromHostBuffer");
  VLOG(2) << "PyLocalBuffer::FromHostBuffer: shape: " << shape.ToString()
          << " device: " << device->DebugString();
  if (shape.IsTuple()) {
    return InvalidArgument("Use FromHostLiteral to transfer a tuple");
  }
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      device->GetLocalDeviceState());

  // If we are on the host platform and the input buffer is sufficiently
  // aligned, we can simply point to the input array's data without any further
  // copies. We require a 64-byte alignment because XLA may generate AVX512
  // code which requires it. If the client allocator doesn't align quite as
  // aggressively, (e.g., NumPy doesn't) there's a high chance this test will
  // fail.
  static constexpr int kMinimumAlignment = 64;
  if (!force_copy &&
      ((absl::bit_cast<std::uintptr_t>(data) & (kMinimumAlignment - 1)) == 0) &&
      local_device->executor()->platform_kind() == se::PlatformKind::kHost) {
    std::function<void()> on_delete_callback =
        [buffer_reference{std::move(buffer_reference)}]() {
          // Frees buffer_reference.
        };
    se::DeviceMemoryBase buffer(const_cast<void*>(data),
                                ShapeUtil::ByteSizeOf(shape));
    absl::Span<const std::shared_ptr<BufferDefinitionEvent>> definition_events;
    auto device_buffer = std::make_shared<SharedDeviceBuffer>(
        /*allocator=*/nullptr, local_device->device_ordinal(),
        std::initializer_list<se::DeviceMemoryBase>{buffer}, definition_events,
        std::move(on_delete_callback));
    return absl::make_unique<PyLocalBuffer>(
        shape, shape, std::move(device_buffer), client, device);
  }

  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(Shape compact_shape,
                      transfer_manager->ChooseCompactLayoutForShape(shape));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PyLocalBuffer> py_buffer,
      AllocateDestinationBuffer(compact_shape, device, local_device,
                                local_device->host_to_device_stream(), client));

  SharedDeviceBuffer::ScopedUsage device_buffer(
      py_buffer->GetBufferWithUsageHold());
  CHECK(device_buffer.IsValid());

  // The host to device transfer is performed on a thread pool, mostly because
  // it includes linearization that may be slow.
  // TODO(misard) assess if it would be preferable to introduce a heuristic to
  // put the transfer into the calling thread for small literals.
  auto transfer_h2d = [client, transfer_manager, local_device,
                       device_buffer_ref{device_buffer.Release()}, data, shape,
                       compact_shape,
                       on_device_shape{py_buffer->on_device_shape()},
                       buffer_reference{std::move(buffer_reference)}]() {
    SharedDeviceBuffer::ScopedUsage device_buffer;
    device_buffer.Transfer(device_buffer_ref);
    // This function uses TF_CHECK_OK and ValueOrDie() since we have no way
    // to report failures from a callback. However, the operations here are
    // unlikely to fail and not recoverable even if we were to fail: DMAs to
    // memory that has already been allocated, and a possible Event
    // allocation.

    ShapedBuffer buffer = device_buffer->AsShapedBuffer(
        compact_shape, on_device_shape, client->client()->platform());

    std::shared_ptr<void> staging_buffer;

    // If applicable on the backend, stage the transfer via host memory
    // allocated via the host_memory_allocator. On GPU, this is pinned
    // memory.
    if (client->host_memory_allocator()) {
      int64 size = ShapeUtil::ByteSizeOf(shape);
      void* ptr = client->host_memory_allocator()->AllocateRaw(
          tensorflow::Allocator::kAllocatorAlignment, size);
      staging_buffer = std::shared_ptr<void>(ptr, [client](void* ptr) {
        client->host_memory_allocator()->DeallocateRaw(ptr);
      });
      std::memcpy(ptr, data, size);
      BorrowingLiteral literal(static_cast<const char*>(staging_buffer.get()),
                               shape);
      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          local_device->host_to_device_stream(), literal, buffer));
    } else {
      BorrowingLiteral literal(static_cast<const char*>(data), shape);
      // Otherwise, just transfer the literal.
      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          local_device->host_to_device_stream(), literal, buffer));
    }

    std::shared_ptr<BufferDefinitionEvent> event =
        device_buffer->definition_events()[0];
    TF_CHECK_OK(AddDestinationBufferSynchronization(
        local_device, std::move(device_buffer), event,
        local_device->host_to_device_stream()));

    local_device->ThenRelease(
        local_device->host_to_device_stream(),
        std::make_pair(buffer_reference, std::move(staging_buffer)));
  };
  client->h2d_transfer_pool()->Schedule(transfer_h2d);
  return py_buffer;
}

/* static */
StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalBuffer::FromHostLiteral(
    const LiteralSlice& literal, PyLocalClient* client, Device* device) {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::FromHostLiteral");
  VLOG(2) << "PyLocalBuffer::FromHostLiteral: shape: "
          << literal.shape().ToString() << " device: " << device->DebugString();
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      device->GetLocalDeviceState());

  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(
      Shape compact_shape,
      transfer_manager->ChooseCompactLayoutForShape(literal.shape()));
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PyLocalBuffer> py_buffer,
      AllocateDestinationBuffer(compact_shape, device, local_device,
                                local_device->host_to_device_stream(), client));

  SharedDeviceBuffer::ScopedUsage device_buffer(
      py_buffer->GetBufferWithUsageHold());
  CHECK(device_buffer.IsValid());

  // The host to device transfer is performed on a thread pool, mostly because
  // it includes linearization that may be slow.
  // TODO(misard) assess if it would be preferable to introduce a heuristic to
  // put the transfer into the calling thread for small literals.
  auto transfer_h2d = [client, transfer_manager, local_device,
                       device_buffer_ref{device_buffer.Release()}, literal,
                       compact_shape,
                       on_device_shape{py_buffer->on_device_shape()}]() {
    SharedDeviceBuffer::ScopedUsage device_buffer;
    device_buffer.Transfer(device_buffer_ref);
    // This function uses TF_CHECK_OK and ValueOrDie() since we have no way
    // to report failures from a callback. However, the operations here are
    // unlikely to fail and not recoverable even if we were to fail: DMAs to
    // memory that has already been allocated, and a possible Event
    // allocation.

    ShapedBuffer buffer = device_buffer->AsShapedBuffer(
        compact_shape, on_device_shape, client->client()->platform());
    TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
        local_device->host_to_device_stream(), literal, buffer));

    std::shared_ptr<BufferDefinitionEvent> event =
        device_buffer->definition_events()[0];
    TF_CHECK_OK(AddDestinationBufferSynchronization(
        local_device, std::move(device_buffer), event,
        local_device->host_to_device_stream()));
  };
  client->h2d_transfer_pool()->Schedule(transfer_h2d);
  return py_buffer;
}

/*static*/ void PyLocalBuffer::MakeCrossHostReceiveBuffers(
    absl::Span<const Shape> shapes, PyLocalClient* client, Device* device,
    PyLocalCrossHostRecvNotifier&& notifier) {
  if (shapes.empty()) {
    notifier(InvalidArgument(
        "shapes parameter empty in MakeCrossHostReceiveBuffers"));
    return;
  }

  auto local_device_or = device->GetLocalDeviceState();
  if (!local_device_or.ok()) {
    notifier(local_device_or.status());
    return;
  }
  LocalDeviceState* local_device = local_device_or.ConsumeValueOrDie();

  std::vector<std::unique_ptr<PyLocalBuffer>> buffers;
  buffers.reserve(shapes.size());
  for (const auto& shape : shapes) {
    StatusOr<std::unique_ptr<PyLocalBuffer>> buffer_or =
        AllocateDestinationBuffer(shape, device, local_device,
                                  /*copy_stream=*/nullptr, client);
    if (!buffer_or.ok()) {
      notifier(buffer_or.status());
      return;
    }
    buffers.push_back(buffer_or.ConsumeValueOrDie());
  }

  client->EnqueueCrossHostReceive(std::move(buffers), std::move(notifier));
}

PyLocalBuffer::PyLocalBuffer(Shape on_host_shape, Shape on_device_shape,
                             std::shared_ptr<SharedDeviceBuffer> device_buffer,
                             PyLocalClient* client, Device* device)
    : client_(client),
      on_host_shape_(std::move(on_host_shape)),
      on_device_shape_(std::move(on_device_shape)),
      device_(device),
      device_buffer_(std::move(device_buffer)) {}

PyLocalBuffer::~PyLocalBuffer() { Delete(); }

StatusOr<std::shared_ptr<SharedDeviceBuffer>> PyLocalBuffer::Release(
    bool wait_for_operations_to_complete) {
  std::shared_ptr<SharedDeviceBuffer> device_buffer;
  {
    absl::MutexLock lock(&mu_);
    if (device_buffer_ == nullptr) {
      return std::shared_ptr<SharedDeviceBuffer>();
    }
    host_value_ = nullptr;
    std::swap(device_buffer_, device_buffer);
  }
  SharedDeviceBuffer::StreamAndEventContainer events =
      device_buffer->LockUseAndTransferUsageEvents();
  LocalDeviceState* local_device_state = device_->local_device_state();
  if (wait_for_operations_to_complete) {
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
        local_device_state->ThenExecuteOnCallbackThread(
            block_stream.get(),
            [device_buffer, block_stream_ptr{block_stream.release()},
             local_device_state]() {
              local_device_state->ReturnStreamToPool(
                  std::unique_ptr<se::Stream>(block_stream_ptr));
            });
      }
    }
  }
  return device_buffer;
}

void PyLocalBuffer::Delete() {
  // When wait_for_reads_to_complete is false, Release should never fail.
  TF_CHECK_OK(Release(/*wait_for_operations_to_complete=*/false).status());
}

bool PyLocalBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return device_buffer_ == nullptr;
}

Status PyLocalBuffer::CopyToHostAsync() {
  if (IsEmptyTuple()) {
    return InvalidArgument("CopyToHostAsync called on empty tuple");
  }
  SharedDeviceBuffer::ScopedUsage device_buffer;
  std::shared_ptr<HostValue> host_value;
  LocalDeviceState* local_device = device_->local_device_state();
  se::Stream* stream = local_device->GetDeviceToHostStream();
  {
    absl::MutexLock lock(&mu_);
    if (device_buffer_ == nullptr) {
      return InvalidArgument("CopyToHostAsync() called on invalid buffer.");
    }
    if (host_value_) {
      // The host value has already been requested or is available.
      return Status::OK();
    }
    host_value = host_value_ = std::make_shared<HostValue>();
    device_buffer.Acquire(device_buffer_);
  }
  WaitForBufferDefinitionEventsOnStream(*device_buffer, stream);
  host_value->value = std::make_shared<Literal>(on_host_shape_);
  ShapedBuffer shaped_buffer = device_buffer->AsShapedBuffer(
      on_host_shape_, on_device_shape_, client_->client()->platform());
  client_->client()->backend().transfer_manager()->TransferLiteralFromDevice(
      stream, shaped_buffer, host_value->value.get(),
      [host_value](Status done_status) {
        host_value->status = done_status;
        host_value->ready.Notify();
      });

  auto usage_event = std::make_shared<BufferDefinitionEvent>();
  StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().ThenAllocateAndRecordEvent(stream);
  if (!event_or.ok()) {
    // Allocating the event failed, so synchronize
    // the host on the copy and then drop the device buffer hold.
    StallStreamOnError(local_device, stream);
    return event_or.status();
  }
  usage_event->SetDefinitionEvent(event_or.ConsumeValueOrDie(), stream);
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
  return Status::OK();
}

StatusOr<std::shared_ptr<Literal>> PyLocalBuffer::ToLiteral() {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::ToLiteral");
  TF_RETURN_IF_ERROR(CopyToHostAsync());
  std::shared_ptr<HostValue> host_value;
  {
    absl::MutexLock lock(&mu_);
    host_value = host_value_;
  }
  if (host_value == nullptr) {
    return InvalidArgument("ToLiteral called on invalid buffer");
  }
  host_value->ready.WaitForNotification();
  TF_RETURN_IF_ERROR(host_value->status);
  return host_value->value;
}

StatusOr<ShapedBuffer> PyLocalBuffer::AsShapedBuffer() const {
  absl::MutexLock lock(&mu_);
  if (device_buffer_ == nullptr) {
    return InvalidArgument(
        "Attempted to fetch value of invalid/deleted buffer.");
  }
  return device_buffer_->AsShapedBuffer(on_host_shape_, on_device_shape_,
                                        client_->client()->platform());
}

SharedDeviceBuffer::ScopedUsage PyLocalBuffer::GetBufferWithUsageHold() {
  absl::MutexLock lock(&mu_);
  SharedDeviceBuffer::ScopedUsage usage;
  return std::move(usage.Acquire(device_buffer_));
}

std::shared_ptr<SharedDeviceBuffer>
PyLocalBuffer::GetBufferWithExternalReference() {
  absl::MutexLock lock(&mu_);
  if (device_buffer_ == nullptr) {
    return nullptr;
  }
  device_buffer_->AddExternalReference();
  return device_buffer_;
}

StatusOr<std::pair<std::unique_ptr<PyLocalBuffer>,
                   std::shared_ptr<BufferDefinitionEvent>>>
PyLocalBuffer::CopyToDeviceHelper(
    Device* dst_device, LocalDeviceState* dst_local_device,
    LocalDeviceState* transfer_local_device, se::Stream* transfer_stream,
    std::shared_ptr<SharedDeviceBuffer> src_device_buffer) {
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PyLocalBuffer> py_buffer,
      AllocateDestinationBuffer(on_host_shape_, dst_device, dst_local_device,
                                transfer_stream, client_));

  TF_ASSIGN_OR_RETURN(ShapedBuffer src_buffer, AsShapedBuffer());

  WaitForBufferDefinitionEventsOnStream(*src_device_buffer, transfer_stream);

  SharedDeviceBuffer::ScopedUsage dst_device_buffer =
      py_buffer->GetBufferWithUsageHold();
  CHECK(dst_device_buffer.IsValid());
  ShapedBuffer dst_buffer = dst_device_buffer->AsShapedBuffer(
      on_host_shape_, on_device_shape_, client_->client()->platform());

  // Copy the leaf buffers.
  StatusOr<std::shared_ptr<BufferDefinitionEvent>> copy_event_or =
      [&]() -> StatusOr<std::shared_ptr<BufferDefinitionEvent>> {
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
    std::shared_ptr<BufferDefinitionEvent> event =
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
                                                 src_device_buffer);
    }
    return copy_event_or.status();
  }

  return std::pair<std::unique_ptr<PyLocalBuffer>,
                   std::shared_ptr<BufferDefinitionEvent>>(
      std::move(py_buffer), copy_event_or.ConsumeValueOrDie());
}

StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalBuffer::CopyToDevice(
    Device* dst_device) {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::CopyToDevice");
  if (dst_device == device_) {
    return InvalidArgument(
        "CopyToDevice cannot accept the same source and destination devices");
  }

  TF_ASSIGN_OR_RETURN(LocalDeviceState * dst_local_device,
                      dst_device->GetLocalDeviceState());
  LocalDeviceState* transfer_local_device =
      client_->EnqueueD2DTransfersOnSrcStream() ? device_->local_device_state()
                                                : dst_local_device;
  CHECK_EQ(dst_local_device->allocation_model(),
           transfer_local_device->allocation_model());

  se::Stream* transfer_stream =
      transfer_local_device->GetDeviceToDeviceStream();

  SharedDeviceBuffer::ScopedUsage src_device_buffer;
  {
    absl::MutexLock lock(&mu_);
    if (device_buffer_ == nullptr) {
      return InvalidArgument("CopyToDevice called on invalid buffer");
    }
    src_device_buffer.Acquire(device_buffer_);
  }

  StatusOr<std::pair<std::unique_ptr<PyLocalBuffer>,
                     std::shared_ptr<BufferDefinitionEvent>>>
      buffer_and_event_or = CopyToDeviceHelper(
          dst_device, dst_local_device, transfer_local_device, transfer_stream,
          src_device_buffer.buffer_reference());
  if (!buffer_and_event_or.ok()) {
    return buffer_and_event_or.status();
  }

  auto& [buffer, event] = buffer_and_event_or.ValueOrDie();
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

Status PyLocalBuffer::CopyToRemoteDevice(
    absl::string_view serialized_descriptor) {
  return client_->CopyToRemoteDevice(this, serialized_descriptor);
}

Status PyLocalBuffer::BlockHostUntilReady() {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::BlockHostUntilReady");
  std::shared_ptr<SharedDeviceBuffer> device_buffer;
  {
    absl::MutexLock lock(&mu_);
    if (device_buffer_ == nullptr) {
      return InvalidArgument("BlockHostUntilReady() called on invalid buffer.");
    }
    device_buffer = device_buffer_;
  }
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
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
    local_device_state->ReturnStreamToPool(std::move(stream));
  }
  return Status::OK();
}

namespace {

// Helper struct for the tuple that is transiently constructed to hold the
// arguments of an execution.
struct TupleHandle {
  // The device buffer holding the root of the tuple table.
  se::OwningDeviceMemory root_table;
  // The ShapedBuffer describing the tuple. Does not own any of its buffers.
  std::unique_ptr<ShapedBuffer> shaped_buffer;
  // A definition event that has been recorded on the host_to_device stream
  // after the tuple table transfer.
  std::shared_ptr<BufferDefinitionEvent> event;
};

// Makes a tuple from the arguments to an execution.
StatusOr<TupleHandle> MakeTupleHelper(
    PyLocalClient* client, LocalDeviceState* local_device,
    absl::Span<const ShapedBuffer> shaped_buffers, int device_ordinal) {
  std::vector<Shape> host_shapes;
  std::vector<Shape> device_shapes;
  host_shapes.reserve(shaped_buffers.size());
  device_shapes.reserve(shaped_buffers.size());
  for (const ShapedBuffer& buffer : shaped_buffers) {
    host_shapes.push_back(buffer.on_host_shape());
    device_shapes.push_back(buffer.on_device_shape());
  }
  Shape on_host_shape = ShapeUtil::MakeTupleShape(host_shapes);
  Shape on_device_shape = ShapeUtil::MakeTupleShape(device_shapes);

  se::DeviceMemoryAllocator* allocator = client->allocator();
  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();
  se::Stream* stream = local_device->host_to_device_stream();
  TF_ASSIGN_OR_RETURN(
      se::OwningDeviceMemory root_table_memory,
      allocator->Allocate(
          device_ordinal,
          transfer_manager->GetByteSizeRequirement(on_host_shape)));

  // tuple_buffer holds the device buffers for all the arguments and the root
  // table, and does not own any of them.
  auto tuple_buffer = absl::make_unique<ShapedBuffer>(
      on_host_shape, on_device_shape, client->client()->platform(),
      device_ordinal);
  tuple_buffer->set_buffer(root_table_memory.cref(), {});
  for (int i = 0; i < shaped_buffers.size(); ++i) {
    for (const auto& sub_buffer : shaped_buffers[i].buffers()) {
      ShapeIndex index = sub_buffer.first;
      index.push_front(i);
      tuple_buffer->set_buffer(sub_buffer.second, index);
    }
  }

  if (local_device->allocation_model() ==
      LocalDeviceState::kComputeSynchronized) {
    stream->ThenWaitFor(local_device->compute_stream());
  } else {
    DCHECK(transfer_manager->CanShapedBufferBeAccessedNow(
        local_device->compute_stream()->parent(), *tuple_buffer));
  }

  TF_RETURN_IF_ERROR(
      transfer_manager->WriteRootTupleIndexTable(stream, *tuple_buffer));
  StatusOr<EventPool::Handle> event_or =
      local_device->event_pool().ThenAllocateAndRecordEvent(stream);
  if (!event_or.ok()) {
    StallStreamOnError(local_device, stream);
    return event_or.status();
  }

  auto transfer_event = std::make_shared<BufferDefinitionEvent>();
  transfer_event->SetDefinitionEvent(event_or.ConsumeValueOrDie(), stream);
  return TupleHandle({std::move(root_table_memory), std::move(tuple_buffer),
                      std::move(transfer_event)});
}

// Converts a ScopedShapedBuffer returned from an execution into a
// PyLocalBuffer.
std::unique_ptr<PyLocalBuffer> OutputBufferHelper(
    ScopedShapedBuffer* result_buffer,
    std::shared_ptr<BufferDefinitionEvent> definition_event,
    PyLocalClient* client, Device* device, LocalDeviceState* local_device) {
  std::shared_ptr<SharedDeviceBuffer> out_buffer =
      SharedDeviceBuffer::FromScopedShapedBuffer(result_buffer,
                                                 {definition_event});
  RecordUsage(std::move(SharedDeviceBuffer::ScopedUsage().Acquire(out_buffer)),
              local_device, local_device, definition_event,
              local_device->compute_stream(),
              /*prefer_to_retain_reference=*/false);
  return absl::make_unique<PyLocalBuffer>(
      result_buffer->on_host_shape(), result_buffer->on_device_shape(),
      std::move(out_buffer), client, device);
}

static Device* LookupDevice(const PyLocalClient& client, int device_id) {
  auto it = client.id_to_device().find(device_id);
  CHECK(it != client.id_to_device().end())
      << "Unknown device id: " << device_id;
  return it->second;
}

}  // namespace

PyLocalExecutable::PyLocalExecutable(
    std::vector<std::unique_ptr<LocalExecutable>> executables,
    bool tuple_arguments, DeviceAssignment device_assignment,
    PyLocalClient* client)
    : client_(client),
      device_assignment_(std::make_shared<DeviceAssignment>(device_assignment)),
      tuple_arguments_(tuple_arguments) {
  executables_.reserve(executables.size());
  for (auto& executable : executables) {
    executables_.emplace_back(std::move(executable));
  }

  // This must go after `executables_` is initialized.
  VLOG(1) << "PyLocalExecutable " << name() << " device_assignment:\n"
          << device_assignment_->ToString();

  const int num_replicas = device_assignment_->replica_count();
  const int num_partitions = device_assignment_->computation_count();

  // SPMD sharding produces a single executable for multiple partitions.
  if (executables_.size() > 1) {
    CHECK_EQ(num_partitions, executables_.size())
        << "Number of executables " << executables_.size()
        << " did not match number of partitions " << num_partitions;
  }

  for (int replica = 0; replica < num_replicas; ++replica) {
    for (int partition = 0; partition < num_partitions; ++partition) {
      int device_id = (*device_assignment_)(replica, partition);
      Device* device = LookupDevice(*client_, device_id);
      if (device->host_id() != client_->host_id()) {
        VLOG(3) << "Non-local device: " << device_id;
        continue;
      }
      local_logical_device_ids_.emplace_back(replica, partition);
      local_devices_.push_back(device);
    }
  }
  CHECK_GE(local_devices_.size(), 1) << device_assignment_->ToString();
  CHECK_LE(local_devices_.size(), client_->local_device_count())
      << "Inconsistent local device count.";
}

const std::string& PyLocalExecutable::name() const {
  Executable* executable = executables_[0]->executable();
  if (executable->has_module()) {
    return executable->module().name();
  } else {
    static const std::string* unknown_name =
        new std::string("<unknown executable>");
    return *unknown_name;
  }
}

// Enqueues a computation onto the compute stream. Each buffer returned in
// device_buffers has a usage hold added that must be dropped on error or
// converted on success.
StatusOr<ScopedShapedBuffer> PyLocalExecutable::EnqueueExecution(
    absl::Span<PyLocalBuffer* const> argument_handles, int replica,
    int partition, int executable_idx, const RunId& run_id,
    const ExecuteOptions& options, Device* device,
    std::vector<SharedDeviceBuffer::ScopedUsage>* device_buffers) const {
  int device_ordinal = device->local_device_state()->device_ordinal();
  tensorflow::profiler::TraceMe traceme([&] {
    return absl::StrCat("LocalExecutable::Execute#run_id=", run_id.ToInt(),
                        "#");
  });
  VLOG(3) << "Replica " << replica << ", partition " << partition
          << " mapped to device ordinal for execution: " << device_ordinal;

  absl::flat_hash_set<BufferDefinitionEvent*> events;
  std::vector<ShapedBuffer> argument_buffers;
  std::vector<const ShapedBuffer*> argument_buffer_ptrs;
  device_buffers->reserve(argument_handles.size());
  argument_buffers.reserve(argument_handles.size());
  argument_buffer_ptrs.reserve(argument_handles.size());
  for (int i = 0; i < argument_handles.size(); ++i) {
    PyLocalBuffer* handle = argument_handles[i];
    SharedDeviceBuffer::ScopedUsage device_buffer =
        handle->GetBufferWithUsageHold();
    if (!device_buffer.IsValid()) {
      return InvalidArgument(
          "Deleted buffer passed to Execute() as argument %d to replica %d", i,
          replica);
    }
    if (handle->device() != device) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %s, but replica is assigned to device %s.",
          i, replica, handle->device()->DebugString(), device->DebugString());
    }
    ShapedBuffer shaped_buffer = device_buffer->AsShapedBuffer(
        handle->on_host_shape(), handle->on_device_shape(),
        handle->client()->client()->platform());
    argument_buffers.push_back(std::move(shaped_buffer));
    argument_buffer_ptrs.push_back(&argument_buffers.back());
    GetDeviceBufferDefinitionEvents(*device_buffer, &events);
    device_buffers->push_back(std::move(device_buffer));
    VLOG(4) << "Argument " << i
            << " buffer: " << argument_buffers.back().ToString();
  }

  LocalDeviceState* device_state = &client_->device_state(device_ordinal);
  TupleHandle tuple_handle;
  if (tuple_arguments_) {
    TF_ASSIGN_OR_RETURN(tuple_handle,
                        MakeTupleHelper(client_, device_state, argument_buffers,
                                        device_ordinal));
    argument_buffer_ptrs = {tuple_handle.shaped_buffer.get()};
    events.insert(tuple_handle.event.get());
    // CAUTION: a copy has been enqueued into tuple_handle.root_table so it is
    // important not to free the root_table on error without ensuring that
    // necessary synchronization has been done.
  }

  for (BufferDefinitionEvent* event : events) {
    event->WaitForEventOnStream(device_state->compute_stream());
  }

  ExecutableRunOptions run_options;
  run_options.set_stream(device_state->compute_stream());
  run_options.set_host_to_device_stream(device_state->host_to_device_stream());
  run_options.set_allocator(client_->allocator());
  run_options.set_intra_op_thread_pool(
      client_->client()->backend().eigen_intra_op_thread_pool_device());
  run_options.set_device_assignment(device_assignment_.get());
  run_options.set_run_id(run_id);
  run_options.set_rng_seed(device_state->GetNewPrngSeed());
  run_options.set_gpu_executable_run_options(client_->gpu_run_options());

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  auto compute_reservation = std::make_shared<Semaphore::ScopedReservation>(
      device_state->compute_semaphore().ScopedAcquire(1));

  StatusOr<ScopedShapedBuffer> result_buffer_or_status =
      executables_[executable_idx]->RunAsync(argument_buffer_ptrs, run_options);

  VLOG(1) << "Replica " << replica << " partition " << partition
          << " completed; ok=" << result_buffer_or_status.ok();

  if (device_state->allocation_model() == LocalDeviceState::kSynchronous) {
    // Free the root tuple table after execution has completed.
    device_state->ThenExecuteOnCallbackThread(
        device_state->compute_stream(),
        [references{std::make_tuple(executables_[executable_idx],
                                    compute_reservation, device_assignment_)},
         root_buffer{tuple_handle.root_table.Release()},
         allocator{client_->allocator()}, device_ordinal]() {
          TF_CHECK_OK(allocator->Deallocate(device_ordinal, root_buffer));
        });

  } else {
    // The root tuple table can be freed as soon as the computation is
    // enqueued.
    device_state->ThenRelease(
        device_state->compute_stream(),
        std::make_tuple(executables_[executable_idx], compute_reservation,
                        device_assignment_));
  }

  return result_buffer_or_status;
}

StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>>
PyLocalExecutable::ExecuteHelper(
    absl::Span<PyLocalBuffer* const> argument_handles, int replica,
    int partition, const RunId& run_id, const ExecuteOptions& options) const {
  const int device_id = (*device_assignment_)(replica, partition);
  Device* device = LookupDevice(*client_, device_id);

  CHECK_EQ(device->host_id(), client_->host_id());
  int device_ordinal = device->local_device_state()->device_ordinal();
  tensorflow::profiler::TraceMe traceme("LocalExecutable::Execute");
  VLOG(3) << "Replica " << replica << ", partition " << partition
          << " mapped to device ordinal for execution: " << device_ordinal;

  // SPMD sharding produces a single executable for multiple partitions.
  int executable_idx = executables_.size() > 1 ? partition : 0;

  std::vector<SharedDeviceBuffer::ScopedUsage> device_buffers;
  device_buffers.reserve(argument_handles.size());
  StatusOr<ScopedShapedBuffer> result_buffer_or_status =
      EnqueueExecution(argument_handles, replica, partition, executable_idx,
                       run_id, options, device, &device_buffers);

  if (!result_buffer_or_status.ok()) {
    LOG(ERROR) << "Execution of replica " << replica
               << " failed: " << result_buffer_or_status.status();
    return result_buffer_or_status.status();
  }
  ScopedShapedBuffer result_buffer =
      result_buffer_or_status.ConsumeValueOrDie();

  LocalDeviceState* device_state = &client_->device_state(device_ordinal);
  se::Stream* stream = device_state->compute_stream();
  StatusOr<EventPool::Handle> event_or =
      device_state->event_pool().ThenAllocateAndRecordEvent(stream);
  if (!event_or.ok()) {
    StallStreamOnError(device_state, stream);
    return event_or.status();
  }
  auto definition_event = std::make_shared<BufferDefinitionEvent>();
  definition_event->SetDefinitionEvent(event_or.ConsumeValueOrDie(), stream);
  std::vector<std::unique_ptr<PyLocalBuffer>> outputs;
  if (options.untuple_result && result_buffer.on_host_shape().IsTuple()) {
    int tuple_count = result_buffer.on_host_shape().tuple_shapes_size();
    outputs.reserve(tuple_count);
    // Take ownership of each of the output values, leaving only the root table
    // in result_buffer.
    for (int i = 0; i < tuple_count; ++i) {
      ScopedShapedBuffer tuple_buffer = result_buffer.TakeSubTree({i});
      outputs.push_back(OutputBufferHelper(&tuple_buffer, definition_event,
                                           client_, device, device_state));
    }
    if (device_state->allocation_model() == LocalDeviceState::kSynchronous) {
      // Don't release the root buffer until after execution completes.
      ShapedBuffer root_buffer_holder = result_buffer.release();
      se::DeviceMemoryBase root_buffer = root_buffer_holder.root_buffer();
      device_state->ThenExecuteOnCallbackThread(
          device_state->compute_stream(),
          [root_buffer, allocator{client_->allocator()}, device_ordinal]() {
            TF_CHECK_OK(allocator->Deallocate(device_ordinal, root_buffer));
          });
    }
  } else {
    outputs.push_back(OutputBufferHelper(&result_buffer, definition_event,
                                         client_, device, device_state));
  }

  for (SharedDeviceBuffer::ScopedUsage& b : device_buffers) {
    // prefer_to_retain_reference=false because when using the
    // ComputeSynchronized allocation model we don't need to retain a reference
    // to the device_buffer during execution because by definition the compute
    // stream is synchronized past the execution.
    RecordUsage(std::move(b), device_state, device_state, definition_event,
                stream, /*prefer_to_retain_reference=*/false);
  }

  return outputs;
}

StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>>
PyLocalExecutable::Execute(absl::Span<PyLocalBuffer* const> argument_handles,
                           const ExecuteOptions& options) const {
  if (num_replicas() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d replicas using Execute()",
        num_replicas());
  }
  if (num_partitions() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d partitions using Execute()",
        num_partitions());
  }
  VLOG(1) << "Executing computation " << name();
  return ExecuteHelper(argument_handles, /*replica=*/0, /*partition=*/0,
                       RunId(), options);
}

StatusOr<std::vector<std::vector<std::unique_ptr<PyLocalBuffer>>>>
PyLocalExecutable::ExecuteOnLocalDevices(
    absl::Span<const std::vector<PyLocalBuffer*>> argument_handles,
    const ExecuteOptions& options) const {
  RunId run_id;
  tensorflow::profiler::TraceMe traceme([&] {
    return absl::StrCat(
        "LocalExecutable::ExecuteOnLocalDevices#run_id=", run_id.ToInt(), "#");
  });

  const int num_local_devices = local_devices_.size();

  if (argument_handles.size() != num_local_devices) {
    return InvalidArgument(
        "Attempted to execute with %d argument lists when local device "
        "count is %d (total replica count: %d, partition count: %d)",
        argument_handles.size(), num_local_devices, num_replicas(),
        num_partitions());
  }

  VLOG(1) << "Executing computation " << name()
          << "; num_replicas=" << num_replicas()
          << " num_partitions=" << num_partitions() << " num_local_devices=8"
          << num_local_devices;
  std::vector<StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>>> results(
      num_local_devices);
  if (num_local_devices == 1) {
    // Fast-path if there is only one device — run the computation on the
    // current thread.
    const int replica = local_logical_device_ids_[0].first;
    const int partition = local_logical_device_ids_[0].second;
    results[0] =
        ExecuteHelper(argument_handles[0], replica, partition, run_id, options);
  } else {
    absl::Mutex mu;
    int running = num_local_devices;
    int failed = 0;
    Status first_failure_status;

    for (int i = 0; i < num_local_devices; ++i) {
      const int replica = local_logical_device_ids_[i].first;
      const int partition = local_logical_device_ids_[i].second;
      Device* device = local_devices_[i];
      const LocalDeviceState& device_state = *device->local_device_state();
      device_state.execute_thread()->Schedule([&, replica, partition, i] {
        results[i] = ExecuteHelper(argument_handles[i], replica, partition,
                                   run_id, options);

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

  std::vector<std::vector<std::unique_ptr<PyLocalBuffer>>> wrapped_results(
      num_local_devices);
  for (int i = 0; i < num_local_devices; ++i) {
    const int replica = local_logical_device_ids_[i].first;
    const int partition = local_logical_device_ids_[i].second;
    auto& statusor = results[i];
    if (!statusor.ok()) {
      return AppendStatus(
          statusor.status(),
          absl::StrFormat("while running replica %d and partition %d of a"
                          "replicated computation (other "
                          "replicas may have failed as well).",
                          replica, partition));
    }
    wrapped_results[i] = std::move(statusor.ValueOrDie());
  }
  return wrapped_results;
}

/*static*/ StatusOr<std::unique_ptr<PyLocalExecutable>>
PyLocalExecutable::Compile(const XlaComputation& computation,
                           PyLocalClient* client, CompileOptions options) {
  tensorflow::profiler::TraceMe traceme("LocalExecutable::Compile");

  ExecutableBuildOptions& build_options = options.executable_build_options;
  if (!build_options.device_allocator()) {
    build_options.set_device_allocator(client->allocator());
  }

  if (!build_options.has_device_assignment()) {
    VLOG(2) << "PyLocalExecutable::Compile using default device_assignment.";
    TF_ASSIGN_OR_RETURN(
        DeviceAssignment device_assignment,
        client->GetDefaultDeviceAssignment(build_options.num_replicas(),
                                           build_options.num_partitions()));
    build_options.set_device_assignment(device_assignment);
  }
  VLOG(2) << "PyLocalExecutable::Compile device_assignment:\n"
          << build_options.device_assignment().ToString();

  if (!options.argument_layouts) {
    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        computation.GetProgramShape());
    options.argument_layouts = program_shape.parameters();
    for (Shape& shape : *options.argument_layouts) {
      LayoutUtil::ClearLayout(&shape);
    }
  }
  std::vector<const Shape*> argument_layout_pointers;
  argument_layout_pointers.reserve(options.argument_layouts->size());

  // Assign a default layout to any array subshapes that are missing layouts.
  auto assign_layouts = [client](Shape* shape) {
    return ShapeUtil::ForEachMutableSubshapeWithStatus(
        shape, [&](Shape* subshape, const ShapeIndex&) {
          if (subshape->IsArray() && !subshape->has_layout()) {
            LayoutUtil::SetToDefaultLayout(subshape);
            TF_ASSIGN_OR_RETURN(*subshape,
                                client->client()
                                    ->backend()
                                    .transfer_manager()
                                    ->ChooseCompactLayoutForShape(*subshape));
          }
          return Status::OK();
        });
  };

  for (Shape& layout : *options.argument_layouts) {
    argument_layout_pointers.push_back(&layout);
    TF_RETURN_IF_ERROR(assign_layouts(&layout));
  }

  Shape result_layout;
  if (build_options.result_layout()) {
    result_layout = *build_options.result_layout();
  } else {
    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        computation.GetProgramShape());
    result_layout = program_shape.result();
    LayoutUtil::ClearLayout(&result_layout);
  }
  TF_RETURN_IF_ERROR(assign_layouts(&result_layout));
  build_options.set_result_layout(result_layout);

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      client->client()->Compile(computation, argument_layout_pointers,
                                build_options));

  return absl::make_unique<PyLocalExecutable>(
      std::move(local_executables), options.tuple_arguments,
      build_options.device_assignment(), client);
}

}  // namespace xla
