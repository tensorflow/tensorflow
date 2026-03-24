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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_async_host_to_device_transfer_manager.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/layout.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_device.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/gpu/tfrt/utils.h"
#include "xla/pjrt/host_memory_allocator.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/service/gpu_topology.pb.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/device_address_allocator.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/profiler/lib/traceme.h"

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

#if defined(PLATFORM_WINDOWS)
// Required to build successfully with Mingw
#undef CreateEvent
#endif

namespace xla {

// Checks that the input buffers passed in by the user have the correct size
// on device for the compiled program.

absl::StatusOr<std::unique_ptr<TfrtGpuAsyncHostToDeviceTransferManager>>
TfrtGpuAsyncHostToDeviceTransferManager::Create(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    TfrtGpuDevice* device, TfrtGpuClient* client,
    PjRtMemorySpace* memory_space) {
  if (device_layouts.has_value() &&
      device_layouts->size() != shape_specs.size()) {
    return InvalidArgument(
        "Number of layouts %d does not match the number of shapes %d",
        device_layouts->size(), shape_specs.size());
  }
  absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers;
  absl::InlinedVector<tsl::AsyncValueRef<GpuDeviceMemory>, 4> buffer_ptrs;
  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events;
  absl::InlinedVector<Shape, 4> device_shapes;
  buffers.reserve(shape_specs.size());
  buffer_ptrs.reserve(shape_specs.size());
  definition_events.reserve(shape_specs.size());
  device_shapes.reserve(shape_specs.size());
  for (int i = 0; i < shape_specs.size(); ++i) {
    const PjRtClient::ShapeSpec& shape_spec = shape_specs[i];
    if (shape_spec.element_type == TUPLE) {
      return Unimplemented("Async buffer transfer of tuples not implemented.");
    }
    // Initialize a definition event for each async buffer. The definition
    // event will block the buffer usage until the transfer is done.
    tsl::AsyncValueRef<GpuEvent> copy_event =
        tsl::MakeConstructedAsyncValueRef<GpuEvent>();
    // Since transfer of tuples are not supported, we can use a single event
    // for each buffer.
    definition_events.push_back(copy_event.CopyRef());
    Shape& device_shape = device_shapes.emplace_back(
        ShapeUtil::MakeShape(shape_spec.element_type, shape_spec.dims));
    if (device_layouts.has_value() && (*device_layouts)[i].has_value()) {
      *device_shape.mutable_layout() = *(*device_layouts)[i];
    } else {
      TF_ASSIGN_OR_RETURN(device_shape,
                          client->xla_client()
                              ->backend()
                              .transfer_manager()
                              ->ChooseCompactLayoutForShape(device_shape));
    }
    absl::StatusOr<std::unique_ptr<TfrtGpuBuffer>> buffer =
        AllocateTfrtGpuDestinationBuffer(device_shape, definition_events.back(),
                                         device, client, memory_space);
    if (!buffer.ok()) {
      copy_event.SetError(buffer.status());
      return absl::InternalError("Failed to allocate buffer.");
    } else {
      buffer_ptrs.push_back(buffer->get()->GetBufferPtr());
    }

    buffers.push_back(std::move(*buffer));
  }

  return std::make_unique<TfrtGpuAsyncHostToDeviceTransferManager>(
      std::move(buffers), std::move(buffer_ptrs), std::move(definition_events),
      std::move(device_shapes), device);
}

TfrtGpuAsyncHostToDeviceTransferManager::
    TfrtGpuAsyncHostToDeviceTransferManager(
        absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers,
        absl::InlinedVector<tsl::AsyncValueRef<GpuDeviceMemory>, 4> buffer_ptrs,
        absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events,
        absl::InlinedVector<Shape, 4> device_shapes, TfrtGpuDevice* device)
    : buffers_(std::move(buffers)),
      buffer_ptrs_(std::move(buffer_ptrs)),
      buffer_sizes_(GetBufferSizes(buffers_)),
      definition_events_(std::move(definition_events)),
      device_shapes_(std::move(device_shapes)),
      device_(device),
      client_(tsl::down_cast<TfrtGpuClient*>(device_->client())) {
  VLOG(3) << "TfrtGpuAsyncHostToDeviceTransferManager::"
             "TfrtGpuAsyncHostToDeviceTransferManager: this="
          << this << " buffers_.size()=" << buffers_.size();
  transfers_in_flight_.resize(buffer_ptrs_.size(), 0);
  last_transfer_started_.resize(buffer_ptrs_.size(), false);
}

TfrtGpuAsyncHostToDeviceTransferManager::
    ~TfrtGpuAsyncHostToDeviceTransferManager() {
  for (int buffer_index = 0; buffer_index < transfers_in_flight_.size();
       buffer_index++) {
    absl::MutexLock l(mu_);
    // Make sure we don't leave dangling pointers in cleanup routines even
    // if the client lets the object go out of scope.
    mu_.Await(absl::Condition(
        +[](size_t* transfers_in_flight) { return *transfers_in_flight == 0; },
        &transfers_in_flight_[buffer_index]));
  }
}

absl::Status TfrtGpuAsyncHostToDeviceTransferManager::TransferLiteralToBuffer(
    int buffer_index, const LiteralSlice& literal,
    absl::AnyInvocable<void() &&> on_done) {
  tsl::profiler::TraceMe traceme(
      "AsyncHostToDeviceTransferManager::TransferLiteralToBuffer");

  VLOG(3) << "TfrtGpuAsyncHostToDeviceTransferManager::"
             "TransferLiteralToBuffer: this="
          << this << " buffer_index=" << buffer_index
          << ", device=" << device_->DebugString();

  auto* client = tsl::down_cast<TfrtGpuClient*>(device_->client());
  DCHECK(client);

  TransferManager* transfer_manager =
      client->xla_client()->backend().transfer_manager();

  tsl::AsyncValueRef<GpuDeviceMemory> buffer;
  {
    absl::MutexLock l(mu_);

    DCHECK_LT(buffer_index, buffer_ptrs_.size());
    if (last_transfer_started_[buffer_index]) {
      return InvalidArgument(
          "TransferLiteralToBuffer requested for buffer index %d which has "
          "already been fully transferred",
          buffer_index);
    }
    last_transfer_started_[buffer_index] = true;
    buffer = buffer_ptrs_[buffer_index];
    DCHECK(buffer);

    ++transfers_in_flight_[buffer_index];
  }

  // The host to device transfer is performed on a thread pool, mostly
  // because it includes linearization that may be slow.
  // TODO(misard) assess if it would be preferable to introduce a heuristic
  // to put the transfer into the calling thread for small literals.
  auto h2d_copy = [this, buffer_index, transfer_manager,
                   literal = std::move(literal), buffer = std::move(buffer),
                   on_done = std::move(on_done)]() mutable {
    VLOG(3) << "Start transfer h2d for literal with shape "
            << literal.shape().ToString() << " on device "
            << device_->DebugString();

    tsl::profiler::TraceMe traceme(
        "TfrtGpuAsyncHostToDeviceTransferManager::TransferLiteralToBuffer::"
        "h2d_copy");

    // Initiate linearization and transfer of the buffer on the stream.
    ShapedBuffer shaped_buffer =
        buffer->AsShapedBuffer(device_shapes_[buffer_index], device_);

    auto stream = device_->stream();
    CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(stream, literal,
                                                            shaped_buffer));

    absl::Status status = BlockHostUntilDoneWithHostCallback(stream);
    VLOG(3) << "Finish transfer h2d for literal with shape "
            << literal.shape().ToString() << " on device "
            << device_->DebugString() << " with status " << status;

    CHECK_OK(status) << "Failed to block host until done";

    CleanUp(buffer_index, std::move(on_done));
  };
  // Enqueue the transfer to the h2d thread.
  client_->blocking_thread_pool()->Schedule(std::move(h2d_copy));
  return absl::OkStatus();
}

absl::Status
TfrtGpuAsyncHostToDeviceTransferManager::TransferRawDataToSubBuffer(
    int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
    bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) {
  VLOG(3) << "TfrtGpuAsyncHostToDeviceTransferManager::"
             "TransferRawDataToSubBuffer: this="
          << this << " buffer_index=" << buffer_index << " offset=" << offset
          << " transfer_size=" << transfer_size
          << " is_last_transfer=" << is_last_transfer
          << " device=" << device_->DebugString();

  auto* client = tsl::down_cast<TfrtGpuClient*>(device_->client());
  DCHECK(client);

  HostMemoryAllocator::OwnedPtr staging_buffer;
  if (client->ShouldStageHostToDeviceTransfers(data, transfer_size)) {
    HostMemoryAllocator* host_memory_allocator =
        client->GetHostMemoryAllocator();
    if (host_memory_allocator == nullptr) {
      return InvalidArgument(
          "host_memory_allocator should be initialized for staging buffer "
          "transfer.");
    }
    staging_buffer = host_memory_allocator->Allocate(transfer_size);
  }

  se::DeviceAddressBase sub_buffer;
  {
    absl::MutexLock l(mu_);
    DCHECK_LT(buffer_index, buffer_ptrs_.size());
    if (last_transfer_started_[buffer_index]) {
      return InvalidArgument(
          "TransferRawData requested for buffer index %d which has "
          "already been fully transferred",
          buffer_index);
    }
    if (is_last_transfer) {
      last_transfer_started_[buffer_index] = true;
    }
    DCHECK(buffer_ptrs_[buffer_index]);
    tsl::AsyncValueRef<GpuDeviceMemory>& buffer_memory =
        buffer_ptrs_[buffer_index];
    CHECK_LE(offset, buffer_memory->size_bytes());
    CHECK_LE(transfer_size, buffer_memory->size_bytes() - offset);
    if (transfer_size < buffer_memory->size_bytes()) {
      sub_buffer = buffer_memory->buffer().GetByteSlice(offset, transfer_size);
    } else {
      sub_buffer = buffer_memory->buffer();
    }

    ++transfers_in_flight_[buffer_index];
  }

  auto h2d_copy = [transfer_size, staging_buffer = std::move(staging_buffer),
                   data, sub_buffer = std::move(sub_buffer), buffer_index,
                   is_last_transfer, on_done = std::move(on_done),
                   this]() mutable {
    tsl::profiler::TraceMe traceme([&] {
      return tsl::profiler::TraceMeEncode(
          "TfrtGpuAsyncHostToDeviceTransferManager::"
          "TransferRawDataToSubBuffer::"
          "h2d_copy",
          {
              {"device", device_->id()},
              {"buffer_index", buffer_index},
              {"size", transfer_size},
              {"is_last_transfer", is_last_transfer},
          });
    });
    if (transfer_size != 0) {
      if (staging_buffer != nullptr) {
        std::memcpy(staging_buffer.get(), data, transfer_size);
        VLOG(3) << "H2D staging copy done: " << data << " -> "
                << staging_buffer.get() << " (" << transfer_size << " bytes)";
      }

      auto stream = device_->stream();

      const void* host_data_ptr = staging_buffer ? staging_buffer.get() : data;
      VLOG(3) << "H2D copy: " << host_data_ptr << " -> " << sub_buffer.opaque()
              << " (" << transfer_size << " bytes) on device "
              << device_->DebugString();
      CHECK_OK(stream->Memcpy(&sub_buffer, host_data_ptr, transfer_size))
          << "Failed to copy data to GPU";

      absl::Status status = BlockHostUntilDoneWithHostCallback(stream);
      VLOG(3) << "H2D copy done: " << status;
      CHECK_OK(status) << "Failed to block host until done";
    }
    CleanUp(buffer_index, std::move(on_done));
  };
  // Enqueue the transfer to the h2d thread.
  // Note: The ordering of transfers enqueued via this method is not
  // guaranteed.  If multiple transfers for the same buffer are submitted,
  // their execution order may vary.
  client_->blocking_thread_pool()->Schedule(std::move(h2d_copy));
  return absl::OkStatus();
}

void TfrtGpuAsyncHostToDeviceTransferManager::SetBufferError(
    int buffer_index, absl::Status error) {
  {
    absl::MutexLock l(mu_);
    // For a given buffer_index, SetBufferError can't be called twice, or
    // called after the last transfer has been enqueued.
    CHECK(!definition_events_[buffer_index].IsConcrete());
    definition_events_[buffer_index].SetError(error);
  }
  LOG(ERROR) << "SetBufferError sets the " << buffer_index
             << "th buffer error: " << error;
}

absl::InlinedVector<size_t, 4>
TfrtGpuAsyncHostToDeviceTransferManager::GetBufferSizes(
    absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4>& buffers) {
  absl::InlinedVector<size_t, 4> buffer_sizes;
  buffer_sizes.reserve(buffers.size());
  for (const auto& buffer : buffers) {
    buffer_sizes.push_back(buffer->GetOnDeviceSizeInBytes().value());
  }
  return buffer_sizes;
}

void TfrtGpuAsyncHostToDeviceTransferManager::CleanUp(
    int buffer_index, absl::AnyInvocable<void() &&> on_done) {
  TfrtGpuClient* client = client_;
  {
    tsl::profiler::TraceMe traceme(
        "TfrtGpuAsyncHostToDeviceTransferManager::CleanUp");

    absl::MutexLock l(mu_);

    bool last_transfer_started = last_transfer_started_[buffer_index];
    size_t transfers_in_flight = transfers_in_flight_[buffer_index];
    VLOG(3) << "CleanUp for buffer_index=" << buffer_index
            << " last_transfer_started_=" << last_transfer_started
            << " transfers_in_flight_=" << transfers_in_flight
            << "; this: " << this;

    bool is_last_transfer_for_buffer =
        last_transfer_started && transfers_in_flight == 1;
    if (is_last_transfer_for_buffer) {
      // Drop our reference to the TrackedDeviceBuffer for this buffer.
      CHECK(buffer_ptrs_[buffer_index]);
      buffer_ptrs_[buffer_index] = nullptr;
      definition_events_[buffer_index].SetStateConcrete();
    }
    CHECK_GT(transfers_in_flight_[buffer_index], 0);
    --transfers_in_flight_[buffer_index];
  }

  // Call on_done after finishing all housekeeping and releasing the lock.
  // Note: Use a copy of `client_` because accessing `this->client_` directly
  // here is unsafe, as the manager instance could be destroyed after
  // `transfers_in_flight_` is decremented and the mutex released,
  // invalidating member access.
  client->non_blocking_thread_pool()->Schedule(std::move(on_done));
}

}  // namespace xla
