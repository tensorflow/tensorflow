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

#include "xla/pjrt/gpu/tfrt/tfrt_gpu_client.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/debug_options_flags.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_print_options.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/maybe_owning.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/protocol.pb.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/host_memory_allocator.h"
#include "xla/pjrt/gpu/tfrt/tfrt_gpu_executable.h"
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
#include "xla/pjrt/gpu/tfrt/utils.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/layout_mode.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/semaphore.h"
#include "xla/pjrt/stream_executor_executable.h"
#include "xla/pjrt/transpose.h"
#include "xla/pjrt/utils.h"
#include "xla/primitive_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/global_device_id.h"
#include "xla/service/hlo.pb.h"
#include "xla/service/hlo_cost_analysis.h"
#include "xla/service/maybe_owning_device_memory.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_layout.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/cuda/cuda_compute_capability.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"
#include "xla/tsl/platform/status.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/threadpool.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/mem.h"
#include "tsl/platform/protobuf.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/context_types.h"
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
namespace {

// Checks that the input buffers passed in by the user have the correct size
// on device for the compiled program.

class TfrtGpuAsyncHostToDeviceTransferManager final
    : public PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  static absl::StatusOr<
      std::unique_ptr<TfrtGpuAsyncHostToDeviceTransferManager>>
  Create(absl::Span<const PjRtClient::ShapeSpec> shape_specs,
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
        return Unimplemented(
            "Async buffer transfer of tuples not implemented.");
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
          AllocateTfrtGpuDestinationBuffer(device_shape,
                                           definition_events.back(), device,
                                           client, memory_space);
      if (!buffer.ok()) {
        copy_event.SetError(buffer.status());
        return absl::InternalError("Failed to allocate buffer.");
      } else {
        buffer_ptrs.push_back(buffer->get()->GetBufferPtr());
      }

      buffers.push_back(std::move(*buffer));
    }

    return std::make_unique<TfrtGpuAsyncHostToDeviceTransferManager>(
        std::move(buffers), std::move(buffer_ptrs),
        std::move(definition_events), std::move(device_shapes), device);
  }

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

  ~TfrtGpuAsyncHostToDeviceTransferManager() override {
    for (int buffer_index = 0; buffer_index < transfers_in_flight_.size();
         buffer_index++) {
      absl::MutexLock l(&mu_);
      // Make sure we don't leave dangling pointers in cleanup routines even
      // if the client lets the object go out of scope.
      mu_.Await(absl::Condition(
          +[](size_t* transfers_in_flight) {
            return *transfers_in_flight == 0;
          },
          &transfers_in_flight_[buffer_index]));
    }
  }

  size_t buffer_count() const override { return buffer_sizes_.size(); };

  size_t buffer_size(int buffer_index) const override {
    DCHECK_LT(buffer_index, buffer_sizes_.size());
    return buffer_sizes_[buffer_index];
  }

  PjRtDevice* device() const override { return device_; }

  std::unique_ptr<PjRtBuffer> RetrieveBuffer(int buffer_index) override {
    absl::MutexLock l(&mu_);
    DCHECK_LT(buffer_index, buffers_.size());
    return std::move(buffers_[buffer_index]);
  };

  absl::Status TransferLiteralToBuffer(
      int buffer_index, const LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override {
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
      absl::MutexLock l(&mu_);

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
      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          stream, literal, shaped_buffer));

      absl::Status status;
      {
        tsl::profiler::TraceMe traceme("BlockHostUntilDone");
        status = stream->BlockHostUntilDone();
      }
      VLOG(3) << "Finish transfer h2d for literal with shape "
              << literal.shape().ToString() << " on device "
              << device_->DebugString() << " with status " << status;

      CHECK_OK(status) << "Failed to block host until done";

      CleanUp(buffer_index, std::move(on_done));
    };
    // Enqueue the transfer to the h2d thread.
    EnqueueWork(client_->blocking_thread_pool(), std::move(h2d_copy));
    return absl::OkStatus();
  }

  absl::Status TransferRawDataToBuffer(
      int buffer_index, absl::string_view data,
      absl::AnyInvocable<void() &&> on_done) override {
    return TransferRawDataToSubBuffer(buffer_index, data.data(),
                                      /*offset=*/0, data.size(),
                                      /*is_last_transfer=*/true,
                                      std::move(on_done));
  }

  absl::Status TransferRawDataToSubBuffer(
      int buffer_index, const void* data, int64_t offset, int64_t transfer_size,
      bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) override {
    VLOG(3) << "TfrtGpuAsyncHostToDeviceTransferManager::"
               "TransferRawDataToSubBuffer: this="
            << this << " buffer_index=" << buffer_index << " offset=" << offset
            << " transfer_size=" << transfer_size
            << " is_last_transfer=" << is_last_transfer
            << " device=" << device_->DebugString();

    auto* client = tsl::down_cast<TfrtGpuClient*>(device_->client());
    DCHECK(client);

    HostMemoryAllocator::OwnedPtr staging_buffer;
    if (client->should_stage_host_to_device_transfers() &&
        !client->IsDmaMapped(data, transfer_size)) {
      HostMemoryAllocator* host_memory_allocator =
          client->host_memory_allocator();
      if (host_memory_allocator == nullptr) {
        return InvalidArgument(
            "host_memory_allocator should be initialized for staging buffer "
            "transfer.");
      }
      staging_buffer = host_memory_allocator->Allocate(transfer_size);
    }

    se::DeviceMemoryBase sub_buffer;
    {
      absl::MutexLock l(&mu_);
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
        sub_buffer =
            buffer_memory->buffer().GetByteSlice(offset, transfer_size);
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

        const void* host_data_ptr =
            staging_buffer ? staging_buffer.get() : data;
        VLOG(3) << "H2D copy: " << host_data_ptr << " -> "
                << sub_buffer.opaque() << " (" << transfer_size
                << " bytes) on device " << device_->DebugString();
        TF_CHECK_OK(stream->Memcpy(&sub_buffer, host_data_ptr, transfer_size))
            << "Failed to copy data to GPU";

        absl::Status status;
        {
          tsl::profiler::TraceMe traceme("BlockHostUntilDone");
          status = stream->BlockHostUntilDone();
        }
        VLOG(3) << "H2D copy done: " << status;
        CHECK_OK(status) << "Failed to block host until done";
      }
      CleanUp(buffer_index, std::move(on_done));
    };
    // Enqueue the transfer to the h2d thread.
    // Note: The ordering of transfers enqueued via this method is not
    // guaranteed.  If multiple transfers for the same buffer are submitted,
    // their execution order may vary.
    EnqueueWork(client_->blocking_thread_pool(), std::move(h2d_copy));
    return absl::OkStatus();
  }

  void SetBufferError(int buffer_index, absl::Status error) override {
    {
      absl::MutexLock l(&mu_);
      // For a given buffer_index, SetBufferError can't be called twice, or
      // called after the last transfer has been enqueued.
      CHECK(!definition_events_[buffer_index].IsConcrete());
      definition_events_[buffer_index].SetError(error);
    }
    LOG(ERROR) << "SetBufferError sets the " << buffer_index
               << "th buffer error: " << error;
  }

  void AddTransferMetadata(const TransferMetadata& meta) override {}

 private:
  static absl::InlinedVector<size_t, 4> GetBufferSizes(
      absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4>& buffers) {
    absl::InlinedVector<size_t, 4> buffer_sizes;
    buffer_sizes.reserve(buffers.size());
    for (const auto& buffer : buffers) {
      buffer_sizes.push_back(buffer->GetOnDeviceSizeInBytes().value());
    }
    return buffer_sizes;
  }

  void CleanUp(int buffer_index, absl::AnyInvocable<void() &&> on_done) {
    TfrtGpuClient* client = client_;
    {
      tsl::profiler::TraceMe traceme(
          "TfrtGpuAsyncHostToDeviceTransferManager::CleanUp");

      absl::MutexLock l(&mu_);

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
    EnqueueWork(client->non_blocking_thread_pool(), std::move(on_done));
  }

  absl::Mutex mu_;
  // The newly created buffers, which will be returned to the caller via
  // Retrieve.
  absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers_
      ABSL_GUARDED_BY(mu_);

  absl::InlinedVector<tsl::AsyncValueRef<GpuDeviceMemory>, 4> buffer_ptrs_
      ABSL_GUARDED_BY(mu_);
  // Cached versions of the sizes of all the buffers, so we can return them
  // without acquiring mu_.
  const absl::InlinedVector<size_t, 4> buffer_sizes_;
  // True if the last transfer for a buffer has been initiated. Used to
  // prevent a client initiating another transfer after the last transfer has
  // already been initiated.
  absl::InlinedVector<bool, 4> last_transfer_started_ ABSL_GUARDED_BY(mu_);
  // The buffer definition events on all the buffers, unblocked once the
  // corresponding buffer transfer has completed.
  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events_
      ABSL_GUARDED_BY(mu_);
  // Device shapes for all buffers with either compact or custom layout.
  const absl::InlinedVector<Shape, 4> device_shapes_;
  // Count of transfers that have been started but have not yet called
  // cleanup. Used to block in the destructor to avoid dangling pointers in
  // cleanup.
  absl::InlinedVector<size_t, 4> transfers_in_flight_ ABSL_GUARDED_BY(mu_);

  TfrtGpuDevice* const device_;  // not owned.
  TfrtGpuClient* const client_;  // not owned.
};

// Converts PjRt SendCallbacks to an XLA StreamExecutor send function.

}  // namespace

TfrtGpuMemorySpace::TfrtGpuMemorySpace(int id, PjRtDevice* device,
                                       absl::string_view kind, int kind_id)
    : id_(id), device_(device), kind_(kind), kind_id_(kind_id) {
  DCHECK(device_ != nullptr && device_->client() != nullptr);
  to_string_ = absl::StrCat("MEMORY_SPACE_", id_);
  debug_string_ = absl::StrFormat("TfrtGpuMemory(id=%i, device=%s)", id_,
                                  device_->DebugString());
}

const int TfrtGpuDeviceMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(TfrtGpuDeviceMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

TfrtGpuDeviceMemorySpace::TfrtGpuDeviceMemorySpace(int id, PjRtDevice* device)
    : TfrtGpuMemorySpace(id, device, kKind, kKindId) {}

PjRtClient* TfrtGpuDevice::client() const { return client_; }

void TfrtGpuDevice::SetClient(TfrtGpuClient* client) {
  CHECK(client_ == nullptr);
  client_ = client;

  // We have to define debug_string_ and to_string_ here, because
  // platform_name() requires client_ to be set.
  CHECK(!client_->platform_name().empty());
  std::string device_name =
      absl::StrCat(MakeAsciiTitlecase(client_->platform_name()), "Device");
  description_.SetDebugString(
      absl::StrCat(client_->platform_name(), ":", id()));
  description_.SetToString(absl::StrCat(device_name, "(id=", id(), ")"));
}

absl::StatusOr<TransferManager*> TfrtGpuDevice::GetTransferManager() {
  // Downcast Base class to TfrtGpuClient.
  if (client_ == nullptr) {
    return absl::InternalError("Client is null");
  }
  return client_->xla_client()->backend().transfer_manager();
}

absl::Status TfrtGpuDevice::TransferToInfeed(const LiteralSlice& literal) {
  TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager, GetTransferManager());

  return transfer_manager->TransferLiteralToInfeed(executor_, literal);
}

absl::Status TfrtGpuDevice::TransferFromOutfeed(
    MutableBorrowingLiteral literal) {
  TF_ASSIGN_OR_RETURN(TransferManager * transfer_manager, GetTransferManager());

  return transfer_manager->TransferLiteralFromOutfeed(executor_, literal);
}

int TfrtGpuDevice::GetNewPrngSeed() {
  absl::MutexLock lock(&mu_);
  int x = 0;
  do {
    x = prng_seed_distribution_(prng_seed_generator_);
  } while (x == 0);
  return x;
}

void TfrtGpuDevice::AttachMemorySpace(PjRtMemorySpace* memory_space,
                                      bool is_default) {
  CHECK(memory_space != nullptr);
  CHECK(client_ == memory_space->client()) << absl::StrFormat(
      "Could not attach a TfrtGpuExecutable to a PjRtMemorySpace owned "
      "by a different client, the device's client: %s, the memory space's "
      "client: %s.",
      client_->platform_name(), memory_space->client()->platform_name());

  memory_spaces_.push_back(memory_space);
  memory_spaces_by_kind_id_.emplace(memory_space->kind_id(), memory_space);
  if (is_default) {
    CHECK(default_memory_space_ == nullptr)
        << "Default memory space already set to "
        << default_memory_space_->DebugString() << ".";
    default_memory_space_ = memory_space;
  }
}

absl::Span<PjRtMemorySpace* const> TfrtGpuDevice::memory_spaces() const {
  return memory_spaces_;
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::memory_space_by_kind_id(
    int id) const {
  auto it = memory_spaces_by_kind_id_.find(id);
  if (it == memory_spaces_by_kind_id_.end()) {
    return absl::InternalError(
        absl::StrCat("No memory space found (kind_id: ", id, ")"));
  }
  return it->second;
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::memory_space_by_kind(
    absl::string_view kind) const {
  auto it = absl::c_find_if(memory_spaces_, [kind](PjRtMemorySpace* ms) {
    return ms->kind() == kind;
  });
  if (it != memory_spaces_.end()) {
    return *it;
  }
  return absl::InternalError(
      absl::StrCat("No memory space found (kind: ", kind, ")"));
}

absl::StatusOr<PjRtMemorySpace*> TfrtGpuDevice::default_memory_space() const {
  if (default_memory_space_ == nullptr) {
    return absl::InternalError(
        "No default memory space is set for this device.");
  }
  return default_memory_space_;
}

absl::StatusOr<tsl::AllocatorStats> TfrtGpuDevice::GetAllocatorStats() const {
  if (!IsAddressable()) {
    return FailedPrecondition(
        "GetAllocatorStats() is allowed only for addressable devices");
  }

  auto* allocator_adapter =
      dynamic_cast<se::MultiDeviceAdapter*>(client_->allocator());
  if (!allocator_adapter) {
    return Unimplemented(
        "GetAllocatorStats() is only implemented with MultiDeviceAdapter "
        "allocator");
  }

  TF_ASSIGN_OR_RETURN(auto allocator, allocator_adapter->GetAllocator(
                                          local_device_id().value()));

  auto stats = allocator->GetStats();
  TF_RET_CHECK(stats.has_value());
  return stats.value();
}

tsl::AsyncValueRef<GpuEvent> TfrtGpuDevice::SetLastCollectiveLaunchEvent(
    tsl::AsyncValueRef<GpuEvent> event) {
  absl::MutexLock lock(&mu_);
  VLOG(3) << "SetLastCollectiveLaunchEvent: IsAvailable: "
          << event.IsAvailable() << "; pointer: " << event.GetAsyncValue()
          << "Old Event: IsAvailable: "
          << last_collective_launch_event_.IsAvailable()
          << "; pointer: " << last_collective_launch_event_.GetAsyncValue();
  std::swap(last_collective_launch_event_, event);
  return event;
}

TfrtGpuClient::TfrtGpuClient(
    std::string platform_name, int process_index, xla::LocalClient* xla_client,
    std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
    bool should_stage_host_to_device_transfers,
    MaybeOwning<se::DeviceMemoryAllocator> allocator,
    std::unique_ptr<tsl::Allocator> host_memory_allocator,
    std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store,
    std::shared_ptr<const GpuTopology> gpu_topology)
    : process_index_(process_index),
      platform_name_(std::move(platform_name)),
      xla_client_(CHECK_NOTNULL(xla_client)),
      should_stage_host_to_device_transfers_(
          should_stage_host_to_device_transfers),
      allocator_(std::move(allocator)),
      host_memory_allocator_(std::make_unique<HostMemoryAllocator>(
          std::move(host_memory_allocator))),
      devices_(InitializeDevices(this, devices)),
      id_to_device_(GetIdToDeviceMap(devices)),
      addressable_devices_(GetAddressableDevicePointers(devices)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      owned_memory_spaces_(
          InitializeMemorySpaces(devices.size(), addressable_devices_)),
      memory_spaces_(GetMemorySpacePointers(owned_memory_spaces_)),
      gpu_run_options_(std::move(gpu_run_options)),
      transpose_cache_(1024),
      topology_(GetTopology(platform_name_, std::move(gpu_topology),
                            addressable_devices_)),
      kv_store_(std::move(kv_store)),
      owned_devices_(std::move(devices)),
      compile_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), tsl::ThreadOptions(),
          "TfrtGpuClient_compile_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()),
          true)),
      blocking_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), tsl::ThreadOptions(),
          "TfrtGpuClient_blocking_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()),
          true)),
      non_blocking_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), tsl::ThreadOptions(),
          "TfrtGpuClient_non_blocking_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()),
          true)) {
  LOG(INFO) << "TfrtGpuClient created with " << addressable_devices_.size()
            << " / " << devices_.size() << " addressable devices.";
}

TfrtGpuClient::~TfrtGpuClient() { LOG(INFO) << "TfrtGpuClient destroyed."; }

absl::string_view TfrtGpuClient::platform_version() const {
#define STRINGIFY2(X) #X
#define STRINGIFY(X) STRINGIFY2(X)
#if TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)  // rocm
  // TF_ROCM_VERSION format may change in future. Use it
  // cautiously
  return "rocm " STRINGIFY(TF_ROCM_VERSION);
#elif GOOGLE_CUDA && defined(CUDART_VERSION)  // cuda
  return "cuda " STRINGIFY(CUDART_VERSION);
#else
  return "<unknown>";
#endif  // TENSORFLOW_USE_ROCM && defined(TF_ROCM_VERSION)
}

absl::StatusOr<PjRtDevice*> TfrtGpuClient::LookupDevice(
    PjRtGlobalDeviceId global_device_id) const {
  auto it = id_to_device_.find(global_device_id);
  if (it != id_to_device_.end()) {
    return it->second;
  }
  return InvalidArgument("No matching device found for device_id %d",
                         global_device_id.value());
}

absl::StatusOr<PjRtDevice*> TfrtGpuClient::LookupAddressableDevice(
    PjRtLocalDeviceId local_device_id) const {
  for (auto* device : addressable_devices_) {
    if (local_device_id == device->local_device_id()) {
      return device;
    }
  }
  return InvalidArgument("No matching device found for local_hardware_id %d",
                         local_device_id.value());
}

absl::StatusOr<Layout> TfrtGpuClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  return topology_.GetDefaultLayout(element_type, dims);
}

absl::StatusOr<std::unique_ptr<HloCostAnalysis>>
TfrtGpuClient::GetHloCostAnalysis() const {
  return std::make_unique<HloCostAnalysis>(
      xla_client_->backend().compiler()->ShapeSizeBytesFunction());
}

absl::Span<PjRtMemorySpace* const> TfrtGpuClient::memory_spaces() const {
  return memory_spaces_;
}

std::optional<PjRtPluginAttributes> TfrtGpuClient::plugin_attributes() const {
  PjRtPluginAttributes attributes =
      PjRtClient::plugin_attributes().value_or(PjRtPluginAttributes());
  attributes.pjrt_c_api_major_version = 0;
  attributes.pjrt_c_api_minor_version = 0;
  attributes.attributes["serialize_with_sdy"] = PjRtValueType(true);
  attributes.attributes["supports_cross_host_transfers"] = PjRtValueType(true);
  return attributes;
}

absl::StatusOr<DeviceAssignment> TfrtGpuClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return computation_placer_->AssignDevices(num_replicas, num_partitions);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::Compile(
    const XlaComputation& computation, CompileOptions options) {
  return Compile(computation, options, /*lookup_addressable_devices=*/false);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::Compile(
    const XlaComputation& computation, CompileOptions options,
    bool lookup_addressable_devices) {
  std::vector<const Shape*> argument_layout_pointers;
  const ExecutableBuildOptions& build_options =
      options.executable_build_options;
  const bool allow_auto_layout =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_pjrt_allow_auto_layout_in_hlo();
  TF_RETURN_IF_ERROR(DetermineArgumentLayoutsFromCompileOptions(
      computation,
      [local_client = xla_client_,
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

  // TODO: b/382117736 - Record free gpu memory.
  // Ref:
  // https://github.com/openxla/xla/blob/b729ae319d85d5ec1ec11c488092c2d6683a63f2/xla/pjrt/gpu/se_gpu_pjrt_client.cc#L792-L809
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options, bool lookup_addressable_devices) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::CompileInternal");
  VLOG(1) << "TfrtGpuClient::CompileInternal";
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
      xla_client_->Compile(computation, argument_layout_pointers,
                           options.executable_build_options));

  return BuildPjRtExecutable(computation.proto(), std::move(local_executables),
                             std::move(input_options));
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::Compile(
    mlir::ModuleOp module, CompileOptions options) {
  return Compile(module, options, /*lookup_addressable_devices=*/false);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::Compile(
    mlir::ModuleOp module, CompileOptions options,
    bool lookup_addressable_devices) {
  XlaComputation xla_computation;
  ExecutableBuildOptions& exec_build_options = options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, &exec_build_options));

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
  auto layout_callback = [local_client = xla_client_, &arg_layout_modes,
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
                            return this->xla_client_->backend()
                                .transfer_manager()
                                ->ChooseCompactLayoutForShape(shape);
                          },
                          options.executable_build_options));
  return CompileInternal(xla_computation, arg_layouts_and_pointers.second,
                         layout_callback, options, lookup_addressable_devices);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileAndLoad(const XlaComputation& computation,
                              CompileOptions options) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                      Compile(computation, options,
                              /*lookup_addressable_devices=*/true));
  return Load(std::move(executable), LoadOptions());
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                      Compile(module, options,
                              /*lookup_addressable_devices=*/true));
  return Load(std::move(executable), LoadOptions());
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::CreateViewOfDeviceBuffer(
    void* device_ptr, const Shape& shape, PjRtMemorySpace* memory_space,
    std::function<void()> on_delete_callback,
    std::optional<std::intptr_t> stream) {
  CHECK_EQ(memory_space->devices().size(), 1);
  auto* device = memory_space->devices().front();
  size_t byte_size = ShapeUtil::ByteSizeOf(shape);
  se::DeviceMemoryBase device_memory(device_ptr, byte_size);
  auto non_owning_buffer = GpuDeviceMemory(device_memory);
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<GpuDeviceMemory>(
          std::move(non_owning_buffer));
  auto tracked_device_buffer = std::make_unique<TrackedGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      /*definition_event=*/tsl::MakeAvailableAsyncValueRef<GpuEvent>(),
      /*ready_event=*/tsl::MakeAvailableAsyncValueRef<GpuEvent>(),
      std::move(on_delete_callback));
  return std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tsl::down_cast<TfrtGpuDevice*>(device), memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::CreateUninitializedBuffer(const Shape& shape,
                                         PjRtMemorySpace* memory_space) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::CreateUninitializedBuffer");
  VLOG(1) << "TfrtGpuClient::CreateUninitializedBuffer: shape: "
          << shape.ToString()
          << " memory_space: " << memory_space->DebugString();
  TransferManager* transfer_manager =
      xla_client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(Shape compact_shape,
                      transfer_manager->ChooseCompactLayoutForShape(shape));
  return AllocateTfrtGpuDestinationBuffer(
      compact_shape, tsl::MakeAvailableAsyncValueRef<GpuEvent>(),
      tsl::down_cast<TfrtGpuDevice*>(memory_space->devices()[0]), this,
      memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
TfrtGpuClient::BuildPjRtExecutable(
    std::optional<HloModuleProto> unoptimized_hlo_module_proto,
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
      std::move(compile_options), std::move(unoptimized_hlo_module_proto),
      std::move(local_executables), xla_client_, num_replicas, num_partitions,
      name, fingerprint, memory_spaces()[0]->kind());
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
TfrtGpuClient::DeserializeExecutable(
    absl::string_view serialized,
    std::optional<CompileOptions> compile_options) {
  TF_ASSIGN_OR_RETURN(
      auto local_executables_and_options,
      DeserializeToLocalExecutable(serialized, compile_options));

  return BuildPjRtExecutable(/*unoptimized_hlo_module_proto=*/std::nullopt,
                             std::move(local_executables_and_options.first),
                             local_executables_and_options.second);
}

absl::StatusOr<
    std::pair<std::vector<std::unique_ptr<LocalExecutable>>, CompileOptions>>
TfrtGpuClient::DeserializeToLocalExecutable(
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

  tsl::profiler::TraceMe traceme("TfrtGpuClient::DeserializeToLocalExecutable");
  VLOG(1) << "TfrtGpuClient::DeserializeToLocalExecutable";

  std::string str = std::move(*proto.mutable_serialized_executable());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<LocalExecutable> loaded,
      xla_client_->Load(str, compile_options.executable_build_options));

  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.push_back(std::move(loaded));

  return std::make_pair(std::move(local_executables), compile_options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::LoadSerializedExecutable(absl::string_view serialized,
                                        std::optional<CompileOptions> options,
                                        const LoadOptions& load_options) {
  TF_ASSIGN_OR_RETURN(auto local_executables_and_options,
                      DeserializeToLocalExecutable(serialized, options));
  return LoadInternal(std::move(local_executables_and_options.first),
                      local_executables_and_options.second);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::LoadInternal(
    std::vector<std::unique_ptr<LocalExecutable>> local_executables,
    CompileOptions compile_options) {
  auto input_options = compile_options;

  TF_RETURN_IF_ERROR(compile_options.ApplyAllOptionOverrides());

  TF_ASSIGN_OR_RETURN(
      ExecutableExtras extras,
      UpdateCompileOptionsAndGetExecutableExtras(&compile_options));
  std::shared_ptr<DeviceAssignment>& device_assignment =
      extras.device_assignment;
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
      addressable_device_logical_ids = extras.addressable_device_logical_ids;
  std::vector<PjRtDevice*>& addressable_devices = extras.addressable_devices;

  const auto& ex_options = compile_options.executable_build_options;
  const bool xla_dump_hlo_unoptimized_snapshots =
      ex_options.has_debug_options() &&
      ex_options.debug_options().xla_dump_hlo_unoptimized_snapshots();
  HloModuleProto hlo_module_proto;
  if (xla_dump_hlo_unoptimized_snapshots) {
    hlo_module_proto = local_executables[0]->executable()->module().ToProto();
  }

  auto executable = std::make_unique<TfrtGpuExecutable>(
      std::move(local_executables),
      compile_options.parameter_is_tupled_arguments,
      std::move(device_assignment), std::move(input_options),
      std::move(addressable_device_logical_ids), std::move(addressable_devices),
      this);

  TF_RETURN_IF_ERROR(
      executable->SetUpDonation(compile_options.parameter_is_tupled_arguments));
  if (xla_dump_hlo_unoptimized_snapshots) {
    executable->SetInputHloSnapshotBits(
        std::move(hlo_module_proto),
        compile_options.executable_build_options.debug_options());
  }
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(executable));
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>> TfrtGpuClient::Load(
    std::unique_ptr<PjRtExecutable> executable,
    const LoadOptions& load_options) {
  auto se_executable = absl::WrapUnique(
      tensorflow::down_cast<StreamExecutorExecutable*>(executable.release()));
  CompileOptions compile_options = se_executable->compile_options();

  tsl::profiler::TraceMe traceme("TfrtGpuClient::Load");
  VLOG(1) << "TfrtGpuClient::Load";

  TF_ASSIGN_OR_RETURN(
      auto local_executables,
      se_executable->ConsumeExecutable(xla_client_, compile_options));
  return LoadInternal(std::move(local_executables), compile_options);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuClient::CreateErrorBuffer(
    absl::Status error, const Shape& shape, PjRtMemorySpace* memory_space) {
  CHECK_EQ(memory_space->devices().size(), 1);
  if (memory_space->client() != this) {
    return absl::InvalidArgumentError(
        "Memory space is not attached to this client");
  }

  if (IsMemorySpaceKind<UnpinnedHostMemorySpace>(memory_space)) {
    return absl::InvalidArgumentError(
        "Error buffers are not supported for unpinned host memory yet");
  }

  TfrtGpuDevice* device =
      tensorflow::down_cast<TfrtGpuDevice*>(memory_space->devices().front());
  VLOG(4) << "TfrtGpuClient::CreateErrorBuffer: shape: " << shape.ToString()
          << " device: " << device->DebugString() << " error: " << error;

  auto error_async_value_ref = tsl::MakeErrorAsyncValueRef(error);
  auto tracked_device_buffer = std::make_unique<TrackedGpuDeviceBuffer>(
      /*buffer=*/error_async_value_ref,
      /*definition_event=*/error_async_value_ref,
      /*ready_event=*/error_async_value_ref);
  return std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tsl::down_cast<TfrtGpuDevice*>(device), memory_space);
}

absl::Status TfrtGpuClient::UpdateCompileOptions(
    CompileOptions* options, bool lookup_addressable_devices) {
  return UpdateCompileOptionsInternal(options, /*returned_extras=*/nullptr,
                                      lookup_addressable_devices);
}

absl::StatusOr<TfrtGpuClient::ExecutableExtras>
TfrtGpuClient::UpdateCompileOptionsAndGetExecutableExtras(
    CompileOptions* options) {
  ExecutableExtras extras;
  TF_RETURN_IF_ERROR(UpdateCompileOptionsInternal(
      options, &extras, /*lookup_addressable_devices=*/true));
  return extras;
}

absl::Status TfrtGpuClient::UpdateCompileOptionsInternal(
    CompileOptions* options, ExecutableExtras* returned_extras,
    bool lookup_addressable_devices) {
  ExecutableBuildOptions& build_options = options->executable_build_options;
  if (!build_options.compile_thread_pool()) {
    build_options.set_compile_thread_pool(compile_thread_pool_.get());
  }
  if (!build_options.device_allocator()) {
    build_options.set_device_allocator(allocator());
  }

  auto layout_callback = [local_client = xla_client_,
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
  std::vector<TfrtGpuExecutable::LogicalDeviceIds>&
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
          VLOG(4) << "Non-local device: " << device_id;
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

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuClient::BufferFromHostBuffer(
    const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
    std::optional<absl::Span<int64_t const>> byte_strides,
    HostBufferSemantics host_buffer_semantics,
    absl::AnyInvocable<void() &&> on_done_with_host_buffer,
    PjRtMemorySpace* memory_space, const Layout* device_layout) {
  TfrtGpuDevice* device =
      tsl::down_cast<TfrtGpuDevice*>(memory_space->devices()[0]);

  tsl::profiler::TraceMe traceme("TfrtGpuClient::BufferFromHostBuffer");
  Shape device_shape = ShapeUtil::MakeShape(type, dims);

  VLOG(3) << "TfrtGpuClient::BufferFromHostBuffer: shape: "
          << device_shape.ToString() << " device: " << device->DebugString();

  absl::InlinedVector<int64_t, 4> tmp_strides;
  if (!byte_strides) {
    tmp_strides.resize(dims.size());
    TF_RETURN_IF_ERROR(
        ShapeUtil::ByteStrides(device_shape, absl::MakeSpan(tmp_strides)));
    byte_strides = tmp_strides;
  }

  int64_t byte_size = ShapeUtil::ByteSizeOf(device_shape);

  TransferManager* transfer_manager = xla_client_->backend().transfer_manager();
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
      (byte_size == 0 || *byte_strides == shape_strides);

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
        CeilOfRatio<int64_t>(byte_size, 8 / primitive_util::BitWidth(type));
  } else {
    packed_size = byte_size;
  }
  auto dst_definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<TfrtGpuBuffer> output_buffer,
                      AllocateTfrtGpuDestinationBuffer(
                          device_shape, dst_definition_event.CopyRef(), device,
                          this, memory_space, packed_size));
  auto copy_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TrackedGpuDeviceBuffer* allocated_dst_buffer =
      output_buffer->AcquireUsage(copy_event);
  CHECK(allocated_dst_buffer != nullptr);
  auto gpu_buffer = allocated_dst_buffer->buffer();
  if (gpu_buffer.IsError()) {
    return gpu_buffer.GetError();
  }

  // If necessary, allocate a host-side buffer for staging host-to-device
  // transfers. On GPU this is a buffer in pinned memory.
  HostMemoryAllocator::OwnedPtr staging_buffer;
  bool must_use_staging_buffer =
      host_buffer_semantics == HostBufferSemantics::kImmutableOnlyDuringCall ||
      !host_and_device_strides_equal || packed_size != byte_size;

  // Allocating multigigabyte pinned buffers can be very slow. In that case,
  // using a staging buffer is probably worse than not using one.
  bool should_stage_transfers = !IsDmaMapped(data, packed_size) &&
                                should_stage_host_to_device_transfers() &&
                                packed_size < (int64_t{1} << 30);

  bool use_staging_buffer = must_use_staging_buffer || should_stage_transfers;

  auto copy_to_staging_buffer = [allocator = host_memory_allocator(), byte_size,
                                 type, packed_size,
                                 transpose{std::move(transpose)},
                                 should_pack](const void* src_buf) mutable {
    tsl::profiler::TraceMe traceme("BufferFromHostBuffer::H2D_staging_copy");

    HostMemoryAllocator::OwnedPtr staging_buffer =
        allocator->Allocate(transpose ? byte_size : packed_size);
    void* buffer = staging_buffer.get();
    const void* data_ptr = src_buf;
    VLOG(3) << "H2D staging copy: " << src_buf << " -> " << buffer << "("
            << byte_size << " -> " << packed_size << " bytes)";

    if (transpose) {
      transpose->Execute(data_ptr, buffer);
      data_ptr = buffer;
    }
    if (should_pack) {
      primitive_util::PackIntN(
          type,
          absl::MakeConstSpan(static_cast<const char*>(data_ptr), byte_size),
          absl::MakeSpan(static_cast<char*>(buffer), packed_size));
      data_ptr = buffer;
    }
    if (data_ptr != buffer) {
      std::memcpy(buffer, data_ptr, byte_size);
    }
    VLOG(3) << "H2D staging copy done";
    return staging_buffer;
  };

  auto h2d_do_copy = [device, packed_size, copy_event(std::move(copy_event)),
                      dst_definition_event(std::move(dst_definition_event)),
                      gpu_buffer{gpu_buffer.CopyRef()}](const void* src_buf) {
    tsl::profiler::TraceMe traceme([&] {
      return tsl::profiler::TraceMeEncode(
          "BufferFromHostBuffer::H2D_GPU_copy",
          {{"device", device->id()}, {"size", packed_size}});
    });
    auto stream = device->stream();

    se::DeviceMemoryBase dest = gpu_buffer->buffer();
    VLOG(3) << "H2D copy: " << src_buf << " -> " << dest.opaque() << " ("
            << packed_size << " bytes) on device " << device->DebugString();

    absl::Status status = stream->Memcpy(&dest, src_buf, packed_size);

    if (!status.ok()) {
      copy_event.SetError(status);
      dst_definition_event.SetError(status);
      return;
    }

    {
      tsl::profiler::TraceMe traceme("BlockHostUntilDone");
      status = stream->BlockHostUntilDone();
    }
    VLOG(3) << "H2D copy done. " << status;

    if (status.ok()) {
      copy_event.SetStateConcrete();
      dst_definition_event.SetStateConcrete();
    } else {
      copy_event.SetError(status);
      dst_definition_event.SetError(status);
    }
  };

  // Define H2D copy lambda. First, copy host data to staging buffer, then copy
  // staging buffer to GPU device.
  auto h2d_copy = [this, use_staging_buffer, data,
                   on_done_with_host_buffer =
                       std::move(on_done_with_host_buffer),
                   copy_to_staging_buffer(std::move(copy_to_staging_buffer)),
                   h2d_do_copy(std::move(h2d_do_copy))]() mutable {
    if (use_staging_buffer) {
      // Copy to the target data to staging buffer first.
      HostMemoryAllocator::OwnedPtr staging_buffer;
      staging_buffer = copy_to_staging_buffer(data);

      // Call on_done_with_host_buffer to release the data buffer.
      if (on_done_with_host_buffer) {
        std::move(on_done_with_host_buffer)();
      }

      // Copy the data from the staging buffer to GPU.
      EnqueueWork(blocking_thread_pool_.get(),
                  [h2d_do_copy(std::move(h2d_do_copy)),
                   staging_buffer(std::move(staging_buffer))]() {
                    h2d_do_copy(staging_buffer.get());
                  });
    } else {
      EnqueueWork(blocking_thread_pool_.get(),
                  [h2d_do_copy(std::move(h2d_do_copy)), data,
                   on_done_with_host_buffer =
                       std::move(on_done_with_host_buffer)]() mutable {
                    // Copy the data directly to GPU.
                    h2d_do_copy(data);

                    // Call on_done_with_host_buffer to release the data buffer.
                    if (on_done_with_host_buffer) {
                      std::move(on_done_with_host_buffer)();
                    }
                  });
    }
  };

  if (host_buffer_semantics == HostBufferSemantics::kImmutableOnlyDuringCall) {
    h2d_copy();
  } else {
    EnqueueWork(non_blocking_thread_pool_.get(), std::move(h2d_copy));
  }

  return output_buffer;
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
TfrtGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  VLOG(4) << "TfrtGpuClient::CreateBuffersForAsyncHostToDevice";
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices()[0];
  auto* tfrt_gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
  return TfrtGpuAsyncHostToDeviceTransferManager::Create(
      shape_specs, device_layouts, tfrt_gpu_device, this, memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::BufferFromHostLiteral(const LiteralSlice& literal,
                                     PjRtMemorySpace* memory_space,
                                     const Layout* device_layout) {
  if (device_layout) {
    return absl::UnimplementedError(absl::StrCat(
        "BufferFromHostLiteral with device_layout is not implemented on "
        "platform: ",
        platform_name()));
  }
  PjRtDevice* device = memory_space->devices()[0];
  tsl::profiler::TraceMe traceme("TfrtGpuClient::BufferFromHostLiteral");

  VLOG(3) << "TfrtGpuClient::BufferFromHostLiteral: shape: "
          << literal.shape().ToString() << " device: " << device->DebugString();

  const Shape& shape = literal.shape();
  if (shape.IsTuple()) {
    return Unimplemented(
        "Tuple case is not supported in TfrtGpuClient::BufferFromHostLiteral");
  }

  // Add a placeholder definition event for each leaf buffer when creating the
  // buffer. They are set only after h2d dispatch.
  tsl::AsyncValueRef<GpuEvent> definition_event =
      tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TfrtGpuBuffer> output_buffer,
      AllocateTfrtGpuDestinationBuffer(shape, definition_event,
                                       tsl::down_cast<TfrtGpuDevice*>(device),
                                       this, memory_space));

  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = output_buffer->AcquireUsage(usage_event);
  CHECK(device_buffer);

  // It is OK to capture `buffer` pointer because the `output_buffer` can't
  // be deleted until all the usage holds have gone away.
  VLOG(4) << "BufferFromHostLiteral for device_buffer: " << device_buffer;
  EnqueueWork(
      non_blocking_thread_pool_.get(),
      [literal, definition_event, device_buffer, shape, this,
       device = tsl::down_cast<TfrtGpuDevice*>(device),
       usage_event = std::move(usage_event)]() mutable {
        tsl::profiler::TraceMe traceme("BufferFromHostLiteral::H2D_Dispatch");
        TransferManager* transfer_manager =
            xla_client()->backend().transfer_manager();

        auto stream = device->stream();

        const auto& buffer = device_buffer->buffer();
        if (literal.shape().IsArray()) {
          CHECK_EQ(literal.size_bytes(), buffer->size_bytes());
        }

        ShapedBuffer shaped_buffer = buffer->AsShapedBuffer(shape, device);

        TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
            stream, literal, shaped_buffer));

        absl::Status status;
        {
          tsl::profiler::TraceMe traceme("BlockHostUntilDone");
          status = stream->BlockHostUntilDone();
        }
        CHECK_OK(status) << "Failed to block host until done";
        VLOG(3) << "BufferFromHostLiteral done for device_buffer: "
                << device_buffer;

        definition_event.SetStateConcrete();
        usage_event.SetStateConcrete();
      });
  return std::unique_ptr<PjRtBuffer>(std::move(output_buffer));
}

absl::Status TfrtGpuClient::DmaMap(void* data, size_t buffer_size) {
  tsl::profiler::TraceMe trace_me("TfrtGpuClient::DmaMap");
  se::StreamExecutor* executor =
      tensorflow::down_cast<TfrtGpuDevice*>(addressable_devices_[0])
          ->executor();
  DCHECK(executor);
  bool success = executor->HostMemoryRegister(data, buffer_size);
  if (!success) {
    return absl::InternalError(absl::StrFormat(
        "Failed to register host memory at address: %ps", data));
  }
  absl::MutexLock lock(&dma_maps_mutex_);
  dma_maps_.insert({data, buffer_size});
  return absl::OkStatus();
}

absl::Status TfrtGpuClient::DmaUnmap(void* data) {
  tsl::profiler::TraceMe trace_me("TfrtGpuClient::DmaUnmap");
  se::StreamExecutor* executor =
      tensorflow::down_cast<TfrtGpuDevice*>(addressable_devices_[0])
          ->executor();
  DCHECK(executor);
  bool success = executor->HostMemoryUnregister(data);
  if (!success) {
    return absl::InternalError(absl::StrFormat(
        "Failed to unregister host memory at address: %ps", data));
  }
  absl::MutexLock lock(&dma_maps_mutex_);
  dma_maps_.erase(data);
  return absl::OkStatus();
}

bool TfrtGpuClient::IsDmaMapped(const void* data_start, int64_t transfer_size) {
  absl::MutexLock lock(&dma_maps_mutex_);
  if (dma_maps_.empty()) {
    return false;
  }
  auto it = dma_maps_.lower_bound(data_start);
  if (it == dma_maps_.end()) {
    return false;
  }
  void* data_end = (char*)data_start + transfer_size;
  void* map_end = (char*)it->first + it->second;
  return data_end <= map_end;
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    const GpuClientOptions& options) {
#if TENSORFLOW_USE_ROCM
  const auto* pjrt_platform_name = xla::RocmName();
#elif TENSORFLOW_USE_SYCL
  const auto* pjrt_platform_name = xla::SyclName();
#else   // TENSORFLOW_USE_ROCM
  const auto* pjrt_platform_name = xla::CudaName();
#endif  // TENSORFLOW_USE_ROCM

  TF_ASSIGN_OR_RETURN(
      LocalClient * xla_client,
      GetGpuXlaClient(options.platform_name, options.allowed_devices));
  EnablePeerAccess(xla_client->backend().stream_executors());

  std::unique_ptr<tsl::Allocator> host_memory_allocator;
  if (!xla_client->backend().stream_executors().empty()) {
    TF_ASSIGN_OR_RETURN(
        host_memory_allocator,
        GetGpuHostAllocator(xla_client->backend().stream_executors().front()));
  }

  auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
  if (options.enable_mock_nccl) {
    gpu_run_options->set_enable_mock_collectives();
  }

  static const bool xla_gpu_require_exclusive_lock =
      xla::GetDebugOptionsFromFlags().xla_gpu_require_exclusive_lock();
  if (xla_gpu_require_exclusive_lock) {
    gpu_run_options->set_requires_exclusive_lock_on_gpu();
  }

  std::shared_ptr<KeyValueStoreInterface> kv_store = options.kv_store;
  if (options.enable_mock_nccl) {
    kv_store = std::make_shared<InMemoryKeyValueStore>();
  }
  TF_RET_CHECK(options.num_nodes == 1 || kv_store != nullptr);
  TF_ASSIGN_OR_RETURN(
      DeviceTopologyPair device_topology_pair,
      BuildDistributedDevices(
          pjrt_platform_name, xla_client, options.node_id, options.num_nodes,
          gpu_run_options.get(), kv_store, options.enable_mock_nccl,
          options.mock_gpu_topology, options.partition_index, absl::Minutes(2),
          absl::Minutes(5)));

  std::vector<std::unique_ptr<TfrtGpuDevice>> devices =
      std::move(device_topology_pair.first);
  auto gpu_topology = std::shared_ptr<const GpuTopology>(
      GpuTopology::FromProto(device_topology_pair.second));

  TF_ASSIGN_OR_RETURN(
      auto allocator,
      CreateDeviceAllocator(xla_client, options.allocator_config, devices));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtGpuClient>(
      std::move(pjrt_platform_name), options.node_id, xla_client,
      std::move(devices), options.should_stage_host_to_device_transfers,
      std::move(allocator), std::move(host_memory_allocator),
      std::move(gpu_run_options), std::move(kv_store),
      std::move(gpu_topology)));
}

class TrackedGpuDeviceBufferExternalReference
    : public PjRtBuffer::ExternalReference {
 public:
  explicit TrackedGpuDeviceBufferExternalReference(
      std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    data_ptr_ = tracked_device_buffer_->buffer()->buffer().opaque();
  }

  ~TrackedGpuDeviceBufferExternalReference() override = default;

 private:
  std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer_;
};

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtGpuBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedGpuDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedGpuDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

PjRtFuture<> TfrtGpuBuffer::ToLiteral(MutableLiteralBase* literal) {
  VLOG(3) << "TfrtGpuBuffer::ToLiteral for a tensor of shape "
          << literal->shape().ToString();
  return ToLiteralHelper(PjRtFuture<MutableLiteralBase*>(literal));
}

PjRtFuture<> TfrtGpuBuffer::ToLiteralHelper(
    PjRtFuture<MutableLiteralBase*> literal) {
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::ToLiteral");
  auto promise = PjRtFuture<>::CreatePromise();
  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    promise.Set(
        InvalidArgument("ToLiteral() called on deleted or donated buffer"));
    return PjRtFuture<>(promise);
  }

  bool unpack_subbyte_types =
      client_->xla_client()->backend().transfer_manager()->PackSubbyteTypes();

  auto literal_and_transpose_promise =
      PjRtFuture<std::pair<MutableLiteralBase*,
                           std::shared_ptr<TransposePlan>>>::CreatePromise();
  PjRtFuture<std::pair<MutableLiteralBase*, std::shared_ptr<TransposePlan>>>
      literal_and_transpose_future(literal_and_transpose_promise);
  literal.OnReady(
      [client = client_, on_device_shape{on_device_shape_},
       promise = std::move(literal_and_transpose_promise)](
          const absl::StatusOr<MutableLiteralBase*>& value) mutable {
        if (!value.ok()) {
          promise.Set(value.status());
          return;
        }

        MutableLiteralBase* literal = *std::move(value);

        std::shared_ptr<TransposePlan> transpose;
        if (on_device_shape.IsArray()) {
          xla::Layout literal_layout;
          if (literal->shape().has_layout()) {
            literal_layout = literal->shape().layout();
          } else {
            literal_layout = LayoutUtil::MakeDescendingLayout(
                on_device_shape.dimensions().size());
          }

          if (on_device_shape.layout() != literal_layout) {
            absl::InlinedVector<int64_t, 4> byte_strides(
                on_device_shape.dimensions().size());
            absl::Status s = ShapeUtil::ByteStrides(
                on_device_shape, absl::MakeSpan(byte_strides));
            if (!s.ok()) {
              promise.Set(s);
              return;
            }
            absl::Span<const int64_t> dims = on_device_shape.dimensions();
            absl::InlinedVector<int64_t, 4> permutation(dims.size());
            absl::c_reverse_copy(literal_layout.minor_to_major(),
                                 permutation.begin());
            TransposePlan::Options options;
            options.elem_size_in_bytes =
                primitive_util::ByteWidth(on_device_shape.element_type());
            options.dims = on_device_shape.dimensions();
            options.permutation = permutation;
            options.input_layout = TransposePlan::Striding{byte_strides};
            {
              absl::MutexLock lock(&client->transpose_mu_);
              absl::StatusOr<std::shared_ptr<TransposePlan>> t =
                  client->transpose_cache_.GetOrCreate(options);
              if (!t.ok()) {
                promise.Set(t.status());
                return;
              }
              transpose = *std::move(t);
            }
          }
        }
        promise.Set(std::make_pair(literal, std::move(transpose)));
      });

  auto d2h_copy = [device(device_), device_buffer,
                   usage_event(std::move(usage_event)), promise,
                   client = client_, on_device_shape{on_device_shape_},
                   unpack_subbyte_types,
                   literal_and_transpose =
                       std::move(literal_and_transpose_future)]() mutable {
    tsl::profiler::TraceMe traceme("ToLiteral::D2H_copy");
    if (device_buffer->definition_event().IsError()) {
      usage_event.SetStateConcrete();
      VLOG(3) << "device_buffer->definition_event().GetError(): "
              << device_buffer->definition_event().GetError();

      promise.Set(device_buffer->definition_event().GetError());
      return;
    }
    size_t byte_size = device_buffer->buffer()->buffer().size();

    PrimitiveType type = on_device_shape.element_type();
    bool should_unpack =
        unpack_subbyte_types && primitive_util::IsSubByteNonPredType(type);

    literal_and_transpose.OnReady(
        [device = std::move(device), device_buffer = std::move(device_buffer),
         usage_event = std::move(usage_event), promise = std::move(promise),
         client = std::move(client),
         on_device_shape = std::move(on_device_shape), should_unpack,
         byte_size](
            const absl::StatusOr<
                std::pair<MutableLiteralBase*, std::shared_ptr<TransposePlan>>>&
                value) mutable {
          if (!value.ok()) {
            usage_event.SetStateConcrete();
            promise.Set(value.status());
            return;
          }

          auto [literal, transpose] = *std::move(value);
          HostMemoryAllocator::OwnedPtr staging_buffer;
          void* buffer_ptr;
          if (on_device_shape.IsArray()) {
            staging_buffer =
                client->host_memory_allocator()->Allocate(byte_size);
            buffer_ptr = staging_buffer.get();
          } else {
            CHECK_EQ(byte_size, 0);
            buffer_ptr = nullptr;
          }

          {
            tsl::profiler::TraceMe traceme2([&] {
              return tsl::profiler::TraceMeEncode("ToLiteral::D2H_GPU_copy",
                                                  {
                                                      {"device", device->id()},
                                                      {"size", byte_size},
                                                  });
            });
            MarkGpuEventReadyOnExit ready_on_exit(std::move(usage_event));

            auto stream = device->stream();

            VLOG(3) << "D2H copy: "
                    << device_buffer->buffer()->buffer().opaque() << " -> "
                    << buffer_ptr << " (" << byte_size << " bytes)";
            CHECK_OK(stream->Memcpy(
                buffer_ptr, device_buffer->buffer()->buffer(), byte_size))
                << "stream->Memcpy failed copying from GPU to host";

            absl::Status status;
            {
              tsl::profiler::TraceMe traceme("BlockHostUntilDone");
              status = stream->BlockHostUntilDone();
            }
            VLOG(3) << "D2H copy done. " << status;
            if (!status.ok()) {
              VLOG(3) << "stream->BlockHostUntilDone failed: " << status;
              promise.Set(status);
              return;
            }
          }
          void* buffer;
          if (should_unpack) {
            tsl::profiler::TraceMe traceme("ToLiteral::D2H_staging_copy");
            int64_t unpacked_size = ShapeUtil::ElementsIn(on_device_shape);
            if (transpose != nullptr) {
              buffer = tsl::port::AlignedMalloc(
                  unpacked_size, tsl::Allocator::kAllocatorAlignment);
            } else {
              buffer = literal->untyped_data();
            }
            primitive_util::UnpackIntN(
                on_device_shape.element_type(),
                absl::MakeConstSpan(static_cast<const char*>(buffer_ptr),
                                    byte_size),
                absl::MakeSpan(static_cast<char*>(buffer), unpacked_size));
            VLOG(3) << "D2H staging copy done";
          } else {
            buffer = buffer_ptr;
          }
          if (transpose != nullptr) {
            tsl::profiler::TraceMe traceme("Transpose");
            transpose->Execute(buffer,
                               static_cast<char*>(literal->untyped_data()));
            if (should_unpack) {
              tsl::port::AlignedFree(buffer);
            }
          }
          if (on_device_shape.IsArray() && !should_unpack &&
              transpose == nullptr) {
            std::memcpy(literal->untyped_data(), buffer, byte_size);
          }
          promise.Set(absl::OkStatus());
        });
  };
  EnqueueWorkWhenReady(client_->blocking_thread_pool(),
                       {device_buffer->definition_event().CopyRCRef()},
                       std::move(d2h_copy));

  return PjRtFuture<>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme("TfrtGpuBuffer::ToLiteral");
        VLOG(3) << "TfrtGpuBuffer::ToLiteral::OnBlockStart";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme("TfrtGpuBuffer::ToLiteral",
                                               keys.traceme_context_id);
      });
}

PjRtFuture<> TfrtGpuBuffer::LazyToLiteral(
    absl::AnyInvocable<PjRtFuture<MutableLiteralBase*>() &&> generator) {
  VLOG(3) << "TfrtGpuBuffer::LazyToLiteral";
  auto buffer = std::move(generator)();
  return ToLiteralHelper(std::move(buffer));
}

PjRtFuture<> TfrtGpuBuffer::CopyRawToHostFuture(PjRtFuture<void*> dst_future,
                                                int64_t offset,
                                                int64_t transfer_size) {
  VLOG(3) << "TfrtGpuBuffer::CopyRawToHostFuture";
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::CopyRawToHostFuture");
  auto promise = PjRtFuture<>::CreatePromise();
  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  MarkGpuEventReadyOnExit usage_event_holder(std::move(usage_event));
  if (device_buffer == nullptr) {
    return PjRtFuture<>(
        InvalidArgument("ToLiteral() called on deleted or donated buffer"));
  }
  auto d2h_copy = [device(device_), device_buffer, promise,
                   usage_event_holder = std::move(usage_event_holder),
                   client = client_, offset, transfer_size](void* dst) mutable {
    if (device_buffer->definition_event().IsError()) {
      LOG(ERROR) << "device_buffer->definition_event().GetError(): "
                 << device_buffer->definition_event().GetError();
      promise.Set(device_buffer->definition_event().GetError());
      return;
    }
    se::DeviceMemoryBase device_memory = device_buffer->buffer()->buffer();
    if (offset < 0 || offset > device_memory.size() ||
        device_memory.size() - offset < transfer_size) {
      LOG(ERROR) << "Copy raw buffer called on buffer size "
                 << device_memory.size() << " with invalid offset " << offset
                 << ", transfer size " << transfer_size;
      promise.Set(
          InvalidArgument("Copy raw buffer called on buffer size %lld with "
                          "invalid offset %lld, transfer size %lld",
                          device_memory.size(), offset, transfer_size));
      return;
    }

    se::DeviceMemoryBase sub_buffer;
    if (transfer_size < device_memory.size()) {
      sub_buffer = device_memory.GetByteSlice(offset, transfer_size);
    } else {
      sub_buffer = device_memory;
    }

    HostMemoryAllocator::OwnedPtr staging_buffer;
    const bool use_staging = client->should_stage_host_to_device_transfers() &&
                             !client->IsDmaMapped(dst, transfer_size);

    if (use_staging) {
      staging_buffer = client->host_memory_allocator()->Allocate(transfer_size);
    }

    void* host_ptr = use_staging ? staging_buffer.get() : dst;

    auto stream = device->stream();

    VLOG(3) << "D2H copy: " << sub_buffer.opaque() << " -> " << host_ptr << " ("
            << transfer_size << " bytes)";
    absl::Status status = stream->Memcpy(host_ptr, sub_buffer, transfer_size);
    if (!status.ok()) {
      LOG(ERROR) << "stream->Memcpy failed: " << status;
      promise.Set(status);
      return;
    }

    if (use_staging) {
      status = stream->DoHostCallback(
          [dst, staging_buffer = std::move(staging_buffer), transfer_size,
           promise,
           usage_event_holder = std::move(usage_event_holder)]() mutable {
            tsl::profiler::TraceMe traceme3(
                "CopyRawToHostFuture::D2H_staging_copy");
            std::memcpy(dst, staging_buffer.get(), transfer_size);
            VLOG(3) << "D2H staging copy done: " << staging_buffer.get()
                    << " -> " << dst << " (" << transfer_size << " bytes)";
            promise.Set(absl::OkStatus());
          });
    } else {
      status = stream->DoHostCallback(
          [promise,
           usage_event_holder = std::move(usage_event_holder)]() mutable {
            promise.Set(absl::OkStatus());
          });
    }

    if (!status.ok()) {
      LOG(ERROR) << "stream->DoHostCallback failed: " << status;
      promise.Set(status);
    }
  };

  dst_future.OnReady(
      [client(client_), promise, device_buffer,
       d2h_copy = std::move(d2h_copy)](absl::StatusOr<void*> dst_or) mutable {
        if (!dst_or.ok()) {
          promise.Set(dst_or.status());
          LOG(ERROR) << "dst resolved to an error: " << dst_or.status();
          return;
        }
        EnqueueWorkWhenReady(client->blocking_thread_pool(),
                             {device_buffer->definition_event().CopyRCRef()},
                             [dst = std::move(dst_or.value()),
                              d2h_copy = std::move(d2h_copy)]() mutable {
                               std::move(d2h_copy)(dst);
                             });
      });

  return PjRtFuture<>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "TfrtGpuBuffer::CopyRawToHostFuture");
        VLOG(3) << "TfrtGpuBuffer::CopyRawToHostFuture";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            "TfrtGpuBuffer::CopyRawToHostFuture", keys.traceme_context_id);
      });
}

void TfrtGpuBuffer::Delete() {
  tsl::profiler::TraceMe traceme("Gpu buffer delete");
  VLOG(4) << " TfrtGpuBuffer::Delete";
  std::unique_ptr<TrackedGpuDeviceBuffer> device_buffer;
  tsl::AsyncValueRef<GpuEvent> external_references_dropped_event;
  {
    absl::MutexLock lock(&mu_);
    device_buffer = ReleaseBufferLocked();
    if (device_buffer == nullptr) {
      return;
    }

    if (external_reference_counter_ > 0) {
      external_references_dropped_event =
          external_references_dropped_event_.CopyRef();
    } else {
      external_references_dropped_event =
          tsl::MakeAvailableAsyncValueRef<GpuEvent>();
    }
  }
  if (device_buffer == nullptr) return;

  tsl::AsyncValueRef<bool> donation_event = GetDonationEvent();

  // Now that all holds have completed and no more can be added, we can get
  // the final set of usage events.
  tsl::AsyncValueRef<GpuEvent> usage_event =
      device_buffer->LockUseAndTransferUsageEvents();

  std::array event_avs{
      usage_event.GetAsyncValue(),
      // We should also wait for the definition event.
      device_buffer->definition_event().GetAsyncValue(),
      donation_event.GetAsyncValue(),
      external_references_dropped_event.GetAsyncValue(),
  };

  tsl::RunWhenReady(
      event_avs, [device_buffer = std::move(device_buffer),
                  usage_event(std::move(usage_event)),
                  donation_event(std::move(donation_event))]() mutable {
        VLOG(4) << "device_buffer is being deleted: " << device_buffer.get();
        device_buffer.reset();
      });
}

bool TfrtGpuBuffer::IsDeleted() const {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_ == nullptr;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuBuffer::CopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::CopyToMemorySpace");
  PjRtDevice* dst_device = dst_memory_space->devices()[0];

  VLOG(1) << "TfrtGpuBuffer::CopyToMemorySpace:  dst_device: "
          << dst_device->DebugString()
          << " dst_memory_space: " << dst_memory_space->kind();

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
        TfrtGpuClient::HostBufferSemantics::kImmutableZeroCopy,
        [literal{std::move(literal)}]() { /* frees literal */ },
        dst_memory_space,
        /*device_layout=*/nullptr);
  }

  // Copy each leaf buffer to a destination buffer.
  auto src_usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TrackedGpuDeviceBuffer* src_device_buffer = AcquireUsage(src_usage_event);
  if (src_device_buffer == nullptr) {
    return InvalidArgument(
        "CopyToMemorySpace called on deleted or donated buffer");
  }

  TfrtGpuDevice* gpu_src_device = tsl::down_cast<TfrtGpuDevice*>(device());
  TfrtGpuDevice* gpu_dst_device = tsl::down_cast<TfrtGpuDevice*>(dst_device);
  tsl::AsyncValueRef<GpuDeviceMemory> src_buffer = src_device_buffer->buffer();

  auto dst_definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TF_ASSIGN_OR_RETURN(auto output_buffer,
                      AllocateTfrtGpuDestinationBuffer(
                          on_device_shape_, dst_definition_event.CopyRef(),
                          gpu_dst_device, client_, dst_memory_space));
  auto dst_usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TrackedGpuDeviceBuffer* allocated_dst_device_buffer =
      output_buffer->AcquireUsage(dst_usage_event);
  CHECK(allocated_dst_device_buffer != nullptr);
  auto allocated_dst_buffer = allocated_dst_device_buffer->buffer();

  absl::AnyInvocable<void()> transfer_d2d =
      [src_buffer(src_buffer.CopyRef()),
       allocated_dst_buffer(allocated_dst_buffer.CopyRef()),
       dst_definition_event(dst_definition_event.CopyRef()),
       src_definition_event(src_device_buffer->definition_event().CopyRef()),
       src_device(gpu_src_device), dst_device(gpu_dst_device),
       src_usage_event(src_usage_event.CopyRef()),
       dst_usage_event(dst_usage_event.CopyRef())]() {
        VLOG(3) << "Request to transfer D2D from "
                << src_buffer->buffer().opaque() << " on device "
                << src_device->id() << " to "
                << allocated_dst_buffer->buffer().opaque() << " on device "
                << dst_device->id();
        tsl::profiler::TraceMe trace([&] {
          return tsl::profiler::TraceMeEncode(
              "CopyToMemorySpace::D2D_copy",
              {
                  {"src_device", src_device->id()},
                  {"dst_device", dst_device->id()},
                  {"size", src_buffer->buffer().size()},
              });
        });

        MarkGpuEventReadyOnExit ready_on_exit_src(std::move(src_usage_event));
        MarkGpuEventReadyOnExit ready_on_exit_dst(std::move(dst_usage_event));

        if (const absl::Status* error =
                dst_definition_event.GetErrorIfPresent()) {
          allocated_dst_buffer.SetError(*error);
          dst_definition_event.SetError(*error);
          return;
        }

        if (const absl::Status* error =
                src_definition_event.GetErrorIfPresent()) {
          allocated_dst_buffer.SetError(*error);
          dst_definition_event.SetError(*error);
          return;
        }

        auto stream = dst_device->stream();

        se::DeviceMemoryBase dst(allocated_dst_buffer->buffer());
        VLOG(3) << "D2D copy: " << src_buffer->buffer().opaque() << " -> "
                << dst.opaque() << " (" << src_buffer->buffer().size()
                << " bytes)";
        absl::Status status = stream->Memcpy(&dst, src_buffer->buffer(),
                                             src_buffer->buffer().size());
        if (!status.ok()) {
          dst_definition_event.SetError(status);
          return;
        }
        {
          tsl::profiler::TraceMe traceme("BlockHostUntilDone");
          status = stream->BlockHostUntilDone();
        }
        if (status.ok()) {
          VLOG(3) << "D2D copy done. dst: " << dst.opaque();
          dst_definition_event.SetStateConcrete();
        } else {
          LOG(ERROR) << "D2D copy failed. dst: " << dst.opaque()
                     << " status: " << status;
          dst_definition_event.SetError(status);
        }
      };

  EnqueueWorkWhenReady(client_->blocking_thread_pool(),
                       {src_device_buffer->ready_event().CopyRCRef()},
                       std::move(transfer_d2d));
  return output_buffer;
}

void TfrtGpuBuffer::DropExternalReference() {
  absl::MutexLock lock(&mu_);
  CHECK_GT(external_reference_counter_, 0);
  --external_reference_counter_;
  if (external_reference_counter_ == 0) {
    external_references_dropped_event_.SetStateConcrete();
  }
}

absl::StatusOr<std::unique_ptr<TrackedGpuDeviceBuffer>> TfrtGpuBuffer::Release(
    bool wait_for_operations_to_complete) {
  auto donation_event = GetDonationEvent();
  tsl::BlockUntilReady(donation_event);
  std::unique_ptr<TrackedGpuDeviceBuffer> device_buffer;
  {
    absl::MutexLock lock(&mu_);
    device_buffer = ReleaseBufferLocked();
  }
  if (device_buffer == nullptr) return {nullptr};

  std::array events{
      // Now that all holds have completed and no more can be added, we can get
      // the final set of usage events.
      device_buffer->LockUseAndTransferUsageEvents(),
      device_buffer->definition_event().CopyRef(),
  };

  if (wait_for_operations_to_complete) {
    // Block the host until all usage events have completed. Usage events
    // dominate definition events, so this also waits for the buffer to be
    // defined. Return the first error encountered.
    absl::Status first_error;
    for (const auto& av : events) {
      tsl::BlockUntilReady(av);
      if (auto* error = av.GetErrorIfPresent()) {
        first_error.Update(*error);
      }
    }
    if (!first_error.ok()) return std::move(first_error);
  }

  return device_buffer;
}

std::unique_ptr<TrackedGpuDeviceBuffer> TfrtGpuBuffer::ReleaseBufferLocked() {
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::ReleaseBufferLocked");
  return std::move(tracked_device_buffer_);
}


absl::StatusOr<TfrtGpuBuffer::DonationTransaction>
TfrtGpuBuffer::AcquireDonation() {
  absl::MutexLock lock(&mu_);

  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Donation requested for invalid buffer");
  }

  if (external_reference_counter_ > 0) {
    return InvalidArgument(
        "Donation requested for buffer with external reference");
  }

  CHECK(donation_event_.IsAvailable());
  CHECK(!donation_event_.get());
  donation_event_ = tsl::MakeUnconstructedAsyncValueRef<bool>();

  // Swap out `tracked_device_buffer_` so that no one can acquire a usage
  // event after this point.
  VLOG(4) << "TfrtGpuBuffer::AcquireDonation: " << tracked_device_buffer_.get();
  return DonationTransaction(donation_event_,
                             std::move(tracked_device_buffer_));
}


}  // namespace xla
