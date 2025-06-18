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
#include "xla/pjrt/gpu/tfrt/tracked_gpu_device_buffer.h"
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
#include "xla/pjrt/worker_thread.h"
#include "xla/primitive_util.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/executable.h"
#include "xla/service/generic_transfer_manager.h"
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

namespace xla {
namespace {

constexpr absl::string_view kPjRtClientName = "TfrtGpuClient";

PjRtFuture<>::Promise CreatePromiseForEvent(
    tsl::AsyncValueRef<xla::GpuEvent> event) {
  PjRtFuture<>::Promise promise = PjRtFuture<>::CreatePromise();
  auto done_fn = [promise, event]() mutable {
    if (const absl::Status* error = event.GetErrorIfPresent()) {
      VLOG(3) << "Setting future: " << *error;
      promise.Set(*error);
    } else {
      VLOG(3) << "Setting future to OK";
      promise.Set();
    }
  };
  if (event.IsAvailable()) {
    // If the event is available, we can set the promise immediately.
    done_fn();
  } else {
    event.AndThen(std::move(done_fn));
  }
  return promise;
}

absl::StatusOr<Shape> GetDestinationDeviceShape(const Shape& host_shape,
                                                TfrtGpuDevice* device,
                                                TfrtGpuClient* client,
                                                PjRtMemorySpace* memory_space) {
  if (host_shape.IsTuple()) {
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

  TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(host_shape));
  TransferManager* transfer_manager =
      client->xla_client()->backend().transfer_manager();
  Shape device_shape = transfer_manager->HostShapeToDeviceShape(host_shape);
  if (is_pinned_host_memory) {
    device_shape.mutable_layout()->set_memory_space(Layout::kHostMemorySpace);
  }
  TF_RET_CHECK(LayoutUtil::HasLayout(device_shape));
  return device_shape;
}

absl::StatusOr<std::unique_ptr<TfrtGpuBuffer>> AllocateTfrtGpuDestinationBuffer(
    const Shape& on_host_shape, tsl::AsyncValueRef<GpuEvent> definition_event,
    TfrtGpuDevice* device, TfrtGpuClient* client, PjRtMemorySpace* memory_space,
    int64_t pack_size = 0) {
  if (on_host_shape.IsTuple()) {
    return Unimplemented(
        "tuple case not implemented for AllocateTfrtGpuDestinationBuffer");
  }
  TF_ASSIGN_OR_RETURN(
      Shape on_device_shape,
      GetDestinationDeviceShape(on_host_shape, device, client, memory_space));
  size_t byte_size =
      pack_size > 0 ? pack_size : ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSIGN_OR_RETURN(
      auto device_buffer,
      GpuDeviceMemory::Allocate(client->allocator(),
                                device->local_device_id().value(), byte_size,
                                LayoutUtil::MemorySpace(on_device_shape)));
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<GpuDeviceMemory>(
          std::move(device_buffer));

  // TODO: Use the right ready event instead of the definition event.
  tsl::AsyncValueRef<GpuEvent> ready_event = definition_event.CopyRef();
  return std::make_unique<TfrtGpuBuffer>(
      on_device_shape,
      std::make_unique<TrackedGpuDeviceBuffer>(buffer_async_value_ref,
                                               std::move(definition_event),
                                               std::move(ready_event)),
      client, device, memory_space);
}

void EnqueueWork(tsl::thread::ThreadPool* pool,
                 absl::AnyInvocable<void() &&> callee) {
  // TSL TheadPool expects std::function that must be copyable, so we are
  // forced to do a little bit of manual memory management here.
  pool->Schedule(
      [ptr = new absl::AnyInvocable<void() &&>(std::move(callee))]() {
        std::move (*ptr)();
        delete ptr;
      });
}

// Enqueue to a thread pool when all `values` are ready.
void EnqueueWorkWhenReady(
    tsl::thread::ThreadPool* pool,
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> values,
    absl::AnyInvocable<void()> callee) {
  tsl::RunWhenReady(values, [pool, callee = std::move(callee)]() mutable {
    VLOG(3) << "EnqueueWork: pool: " << pool;
    EnqueueWork(pool, std::move(callee));
  });
}

std::string MakeComputeCapabilityString(
    const stream_executor::DeviceDescription* desc) {
  stream_executor::GpuComputeCapability cc = desc->gpu_compute_capability();
  if (std::holds_alternative<stream_executor::CudaComputeCapability>(cc)) {
    auto nvcc = std::get<stream_executor::CudaComputeCapability>(cc);
    return absl::StrCat(nvcc.major, ".", nvcc.minor);
  }
  if (std::holds_alternative<stream_executor::RocmComputeCapability>(cc)) {
    auto rocmcc = std::get<stream_executor::RocmComputeCapability>(cc);
    return rocmcc.gfx_version();
  }
  return "unknown";
}

bool IsAllZeros(const DeviceAssignment& assignment) {
  return std::all_of(
      assignment.begin(), assignment.end(),
      [](const DeviceAssignment::value_type& v) { return v == 0; });
}

std::vector<tsl::RCReference<tsl::AsyncValue>> CopyAsyncValues(
    absl::Span<const tsl::RCReference<tsl::AsyncValue>> events) {
  std::vector<tsl::RCReference<tsl::AsyncValue>> avs;
  avs.reserve(events.size());
  for (const auto& ev : events) {
    avs.push_back(ev);
  }
  return avs;
}

// Checks that the input buffers passed in by the user have the correct size
// on device for the compiled program.
absl::Status CheckBufferCompatibilities(
    absl::Span<int64_t const> input_buffer_sizes_in_bytes,
    absl::Span<TrackedGpuDeviceBuffer* const> input_buffers) {
  if (input_buffers.size() != input_buffer_sizes_in_bytes.size()) {
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), input_buffer_sizes_in_bytes.size());
  }
  for (int i = 0; i < input_buffers.size(); ++i) {
    const auto& buffer = input_buffers[i];
    if (input_buffer_sizes_in_bytes[i] != buffer->buffer()->size_bytes()) {
      return InvalidArgument(
          "Executable expected parameter %d of size %lld but got buffer with"
          " incompatible size %lld ",
          i, input_buffer_sizes_in_bytes[i], buffer->buffer()->size_bytes());
    }
  }
  return absl::OkStatus();
}

template <typename MemorySpaceKind>
bool IsMemorySpaceKind(const PjRtMemorySpace* memory_space) {
  return memory_space->kind_id() == MemorySpaceKind::kKindId;
}

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
            << this << " buffer_index=" << buffer_index;
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
    auto transfer_h2d = [this, buffer_index, transfer_manager,
                         literal = std::move(literal),
                         buffer = std::move(buffer),
                         on_done = std::move(on_done)]() mutable {
      VLOG(3) << "Start transfer h2d for literal with shape "
              << literal.shape().ToString() << " on device "
              << device_->DebugString();

      tsl::profiler::TraceMe traceme(
          "TfrtGpuAsyncHostToDeviceTransferManager::TransferLiteralToBuffer::"
          "transfer_h2d");

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
    EnqueueWork(client_->blocking_thread_pool(), std::move(transfer_h2d));
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
            << " is_last_transfer=" << is_last_transfer;

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

    auto copy_to_gpu = [transfer_size,
                        staging_buffer = std::move(staging_buffer), data,
                        sub_buffer = std::move(sub_buffer), buffer_index,
                        is_last_transfer, on_done = std::move(on_done),
                        this]() mutable {
      tsl::profiler::TraceMe traceme([&] {
        return tsl::profiler::TraceMeEncode(
            "TfrtGpuAsyncHostToDeviceTransferManager::"
            "TransferRawDataToSubBuffer::"
            "copy_to_gpu",
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
                << sub_buffer.opaque() << " (" << transfer_size << " bytes)";
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
    EnqueueWork(client_->blocking_thread_pool(), std::move(copy_to_gpu));
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

std::optional<stream_executor::GpuTargetConfigProto> GetTargetConfigForDevices(
    absl::Span<PjRtDevice* const> devices) {
  if (devices.empty()) {
    return std::nullopt;
  }
  // Temporary ability to disable TargetConfig via env var until
  // internal tests can be fixed.
  const char* disable_target_config_str =
      std::getenv("PJRT_GPU_SE_DISABLE_TARGET_CONFIG");
  int disable_target_config = 0;
  if (disable_target_config_str &&
      absl::SimpleAtoi(disable_target_config_str, &disable_target_config)) {
    if (disable_target_config == 1) {
      return std::nullopt;
    }
  }
  for (const PjRtDevice* device : devices) {
    se::StreamExecutor* executor =
        tensorflow::down_cast<const TfrtGpuDevice*>(device)->executor();
    if (executor != nullptr) {
      return xla::Compiler::TargetConfig(executor).ToProto();
    }
  }
  return std::nullopt;
}

absl::flat_hash_map<std::string, PjRtDeviceAttribute> GetAttrsForDevices(
    std::optional<stream_executor::GpuTargetConfigProto> target_config) {
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attrs;
  if (target_config.has_value()) {
    std::string attr;
    if (tsl::protobuf::TextFormat::PrintToString(*target_config, &attr)) {
      attrs["target_config"] = std::move(attr);
    }
  }
  return attrs;
}

class TfrtGpuCopyToDeviceStream : public CopyToDeviceStream {
 public:
  TfrtGpuCopyToDeviceStream(int64_t channel_id, se::Stream* stream,
                            se::DeviceMemoryBase dst,
                            tsl::AsyncValueRef<std::unique_ptr<se::Event>> done)
      : CopyToDeviceStream(dst.size(), /*granule_bytes=*/1),
        channel_id_(channel_id),
        stream_(stream),
        dst_(dst),
        done_(std::move(done)) {}

  PjRtFuture<> AddChunk(PjRtChunk chunk) final {
    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode("TfrtGpuCopyToDeviceStream::AddChunk",
                                          {{"channel_id", channel_id_}});
    });

    absl::ReleasableMutexLock lock(&mu_);

    VLOG(4) << "Add chunk to a H2D channel #" << channel_id_ << ": "
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

    VLOG(3) << "H2D copy: " << chunk.data() << " -> " << dst.opaque() << " ("
            << chunk.size() << " bytes)";
    auto copied = stream_->Memcpy(&dst, chunk.data(), chunk.size());
    if (!copied.ok()) {
      done_.SetError(copied);
      return PjRtFuture<>(done_.GetError());
    }

    // Delete chunk once the memcpy operation completes.
    auto deleted = stream_->DoHostCallback(
        [chunk = std::move(chunk), buffer_opaque = dst.opaque()]() {
          VLOG(3) << "H2D copy done. " << buffer_opaque;
        });
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
  tsl::AsyncValueRef<std::unique_ptr<se::Event>> done_;
};

template <typename T>
const T* FindCallback(int channel_id, absl::Span<const T> callbacks) {
  // TODO(ezhulenev): Can we use binary search here assuming that callbacks
  // are sorted by channel id? Are they always sorted?
  auto it = absl::c_find_if(callbacks, [&](const T& callback) {
    return callback.channel_id == channel_id;
  });
  return it == callbacks.end() ? nullptr : &*it;
}

// Converts PjRt SendCallbacks to an XLA StreamExecutor send function.
SendDeviceMemoryFunction ConvertSendCallbacksToSendFunction(
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
             -> absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>> {
    VLOG(4) << "Send " << src.size() << " bytes to channel #" << channel_id
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
    auto done_event =
        tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
            std::move(se_event));

    thread_pool->Schedule([done_event, stream, src, channel_id, shape, send] {
      tsl::profiler::TraceMe trace([&] {
        return tsl::profiler::TraceMeEncode("TfrtGpuExecutable::Send",
                                            {{"channel_id", channel_id}});
      });

      // Allocate chunk on the host for copying data from device.
      PjRtChunk chunk = PjRtChunk::AllocateDefault(src.size());

      VLOG(3) << "D2H copy: " << src.opaque() << " -> " << chunk.data() << " ("
              << src.size() << " bytes)";
      auto status = stream->Memcpy(chunk.data(), src, src.size());
      if (!status.ok()) {
        LOG(ERROR) << "Failed to copy data from device to host: " << status;
        done_event.SetError(status);
        return;
      }
      status = stream->RecordEvent(done_event.get().get());
      if (!status.ok()) {
        done_event.SetError(status);
        return;
      }

      // Wait for the data to be available on the host.
      {
        tsl::profiler::TraceMe traceme("BlockHostUntilDone");
        status = stream->BlockHostUntilDone();
      }
      VLOG(3) << "D2H copy done. " << status;
      if (!status.ok()) {
        done_event.SetError(absl::InternalError(absl::StrFormat(
            "failed to synchronize send operation with a stream: %s",
            status.message())));
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

RecvDeviceMemoryFunction ConvertRecvCallbacksToRecvFunction(
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
             -> absl::StatusOr<tsl::AsyncValueRef<std::unique_ptr<se::Event>>> {
    VLOG(4) << "Recv from channel #" << channel_id
            << " (shape=" << shape.ToString() << ")";

    tsl::profiler::TraceMe trace([&] {
      return tsl::profiler::TraceMeEncode("TfrtGpuExecutable::Recv",
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
    // `TfrtGpuCopyToDeviceStream` implementation above).
    TF_ASSIGN_OR_RETURN(auto event, stream->parent()->CreateEvent());
    auto done_event =
        tsl::MakeConstructedAsyncValueRef<std::unique_ptr<se::Event>>(
            std::move(event));

    recv->callback({shape}, std::make_unique<TfrtGpuCopyToDeviceStream>(
                                channel_id, stream, *dst, done_event));

    return std::move(done_event);
  };
}

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

TfrtGpuDevice::TfrtGpuDevice(Options&& options)
    : id_(options.id),
      local_device_id_(options.local_device_id),
      local_hardware_id_(options.local_hardware_id),
      executor_(options.executor),
      stream_(options.executor == nullptr
                  ? nullptr
                  : options.executor->CreateStream().value()),
      prng_seed_generator_(prng_seed_device_()),
      prng_seed_distribution_(std::numeric_limits<int>::min(),
                              std::numeric_limits<int>::max()),
      last_collective_launch_event_(
          tsl::MakeAvailableAsyncValueRef<GpuEvent>()),
      description_(options.id, options.process_index, options.platform_version),
      max_inflight_computations_semaphore_(
          /*capacity=*/options.max_inflight_computations) {
  std::array<int, 1> coords = {local_device_id_.value()};
  description_.SetCoords(coords);
  std::vector<int64_t> v_coords(description_.coords().begin(),
                                description_.coords().end());

  description_.SetAttributes({
      {"coords", xla::PjRtDeviceAttribute(v_coords)},
      {"device_vendor", options.device_vendor},
      {"slice_index", static_cast<int64_t>(options.slice_index)},
      {"compute_capability",
       xla::PjRtDeviceAttribute(options.compute_capability)},
      {"core_count", static_cast<int64_t>(options.core_count)},
  });

  description_.SetDebugString(absl::StrCat("TFRT_GPU_", id_));
  description_.SetToString(absl::StrCat("GpuDevice(id=", id_, ")"));
}

TfrtGpuDevice::~TfrtGpuDevice() {
  // Block the host until all pending work on the stream is done. This is to
  // avoid user-after-free errors in host callbacks.
  if (stream_ != nullptr) {
    absl::Status status = stream_->BlockHostUntilDone();
    if (!status.ok()) {
      LOG(ERROR) << "Failed to wait for stream to finish: " << status;
    }
  }
}

void TfrtGpuDevice::SetClient(PjRtClient* client) {
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
  TfrtGpuClient* client = tensorflow::down_cast<TfrtGpuClient*>(client_);
  if (client == nullptr) {
    return absl::InternalError("Client is null");
  }
  return client->xla_client()->backend().transfer_manager();
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

  auto* allocator_adapter = dynamic_cast<se::MultiDeviceAdapter*>(allocator());
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

se::DeviceMemoryAllocator* TfrtGpuDevice::allocator() const {
  return tensorflow::down_cast<TfrtGpuClient*>(client())->allocator();
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

namespace {

std::vector<PjRtMemorySpace*> GetMemorySpacePointers(
    const std::vector<std::unique_ptr<PjRtMemorySpace>>& memory_spaces) {
  std::vector<PjRtMemorySpace*> memory_spaces_ptrs;
  memory_spaces_ptrs.reserve(memory_spaces.size());
  for (const std::unique_ptr<PjRtMemorySpace>& memory_space : memory_spaces) {
    memory_spaces_ptrs.push_back(memory_space.get());
  }
  return memory_spaces_ptrs;
}

std::vector<PjRtDevice*> InitializeDevices(
    PjRtClient* client,
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& owned_devices) {
  std::vector<PjRtDevice*> devices;
  devices.reserve(owned_devices.size());
  for (const std::unique_ptr<TfrtGpuDevice>& device : owned_devices) {
    device->SetClient(client);
    devices.push_back(device.get());
  }
  return devices;
}

absl::flat_hash_map<PjRtGlobalDeviceId, TfrtGpuDevice*> GetIdToDeviceMap(
    absl::Span<const std::unique_ptr<TfrtGpuDevice>> devices) {
  absl::flat_hash_map<PjRtGlobalDeviceId, TfrtGpuDevice*> id_to_device;
  for (const std::unique_ptr<TfrtGpuDevice>& device : devices) {
    CHECK(id_to_device.emplace(device->global_device_id(), device.get()).second)
        << "Duplicate device id: " << device->id();
  }
  return id_to_device;
}

std::vector<PjRtDevice*> GetAddressableDevicePointers(
    absl::Span<const std::unique_ptr<TfrtGpuDevice>> devices) {
  std::vector<PjRtDevice*> addressable_devices;
  for (const std::unique_ptr<TfrtGpuDevice>& device : devices) {
    if (device->IsAddressable()) {
      addressable_devices.push_back(device.get());
    }
  }
  // TODO(phawkins): we don't really promise anything about the order of
  // these devices, but users may be depending on the current order. Sort into
  // device ordinal order, which is the historical order these values have
  // appeared.
  absl::c_sort(addressable_devices,
               [](const PjRtDevice* a, const PjRtDevice* b) {
                 return a->local_device_id() < b->local_device_id();
               });
  return addressable_devices;
}

StreamExecutorGpuTopologyDescription GetTopology(
    absl::string_view platform_name,
    std::shared_ptr<const GpuTopology> gpu_topology,
    absl::Span<PjRtDevice* const> devices) {
  auto target_config = GetTargetConfigForDevices(devices);
  return StreamExecutorGpuTopologyDescription(
      tsl::Fingerprint64(platform_name), platform_name, std::move(gpu_topology),
      GetAttrsForDevices(target_config), target_config);
}

std::vector<std::unique_ptr<PjRtMemorySpace>> InitializeMemorySpaces(
    int global_device_count,
    absl::Span<PjRtDevice* const> addressable_devices) {
  std::vector<std::unique_ptr<PjRtMemorySpace>> memory_spaces;
  for (auto* device : addressable_devices) {
    // Use the device id to construct a globally unique memory space id. We do
    // not promise that memory space ids and device ids are the same.
    TfrtGpuDevice* gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
    // Initialize the default memory space.
    const int global_device_id = gpu_device->global_device_id().value();
    auto memory_space =
        std::make_unique<TfrtGpuDeviceMemorySpace>(global_device_id, device);
    gpu_device->AttachMemorySpace(memory_space.get(), /*is_default=*/true);
    memory_spaces.push_back(std::move(memory_space));
  }
  const int basePinnedId = global_device_count;
  for (auto* device : addressable_devices) {
    TfrtGpuDevice* gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
    const int global_device_id = gpu_device->global_device_id().value();
    auto pinned = std::make_unique<PinnedHostMemorySpace>(
        basePinnedId + global_device_id, device);
    gpu_device->AttachMemorySpace(pinned.get());
    memory_spaces.push_back(std::move(pinned));
  }
  // We don't promise anything about the order of memory spaces, but this
  // sorting is done for consistency with the device list that's sorted above.
  absl::c_sort(memory_spaces, [](const std::unique_ptr<PjRtMemorySpace>& a,
                                 const std::unique_ptr<PjRtMemorySpace>& b) {
    return a->id() < b->id();
  });
  return memory_spaces;
}

}  // namespace

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

  return BuildPjRtExecutable(std::move(local_executables),
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

absl::StatusOr<std::string> TfrtGpuExecutable::SerializeExecutable() const {
  if (executables_.size() != 1) {
    // TODO(b/382117736): Change SerializeExecutable interface to support
    // multiple partitions.
    return absl::FailedPreconditionError(
        "SerializeExecutable with >1 partitions not yet supported");
  }
  Executable* built_executable = executables_[0]->executable();
  Compiler* compiler = client_->xla_client()->backend().compiler();
  TF_ASSIGN_OR_RETURN(std::unique_ptr<AotCompilationResult> aot_result,
                      compiler->Export(built_executable));
  TF_ASSIGN_OR_RETURN(std::string serialized, aot_result->SerializeAsString());
  if (serialized.empty()) {
    return Internal(
        "TfrtGpuExecutable::SerializeExecutable proto serialization "
        "failed");
  }
  ExecutableAndOptionsProto proto;
  *proto.mutable_serialized_executable() = std::move(serialized);
  TF_ASSIGN_OR_RETURN(*proto.mutable_compile_options(),
                      compile_options_.ToProto());
  *proto.mutable_pjrt_client_name() = kPjRtClientName;
  return proto.SerializeAsString();
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
TfrtGpuClient::BuildPjRtExecutable(
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
      std::move(compile_options), std::move(local_executables), xla_client_,
      num_replicas, num_partitions, name, fingerprint,
      memory_spaces()[0]->kind());
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>>
TfrtGpuClient::DeserializeExecutable(
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
  const bool xla_gpu_dump_hlo_unoptimized_snapshots =
      ex_options.has_debug_options() &&
      ex_options.debug_options().xla_gpu_dump_hlo_unoptimized_snapshots();
  HloModuleProto hlo_module_proto;
  if (xla_gpu_dump_hlo_unoptimized_snapshots) {
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
  if (xla_gpu_dump_hlo_unoptimized_snapshots) {
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
  VLOG(4) << "TfrtGpuClient::BufferFromHostBuffer: shape: "
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

  auto copy_to_staging_buffer = [allocator = host_memory_allocator(), data,
                                 byte_size, type, packed_size,
                                 transpose{std::move(transpose)}, should_pack,
                                 on_done_with_host_buffer = std::move(
                                     on_done_with_host_buffer)]() mutable {
    tsl::profiler::TraceMe traceme("BufferFromHostBuffer::H2D_staging_copy");

    HostMemoryAllocator::OwnedPtr staging_buffer =
        allocator->Allocate(transpose ? byte_size : packed_size);
    void* buffer = staging_buffer.get();
    const void* data_ptr = data;
    VLOG(3) << "H2D staging copy: " << data << " -> " << buffer << "("
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
    if (on_done_with_host_buffer) {
      std::move(on_done_with_host_buffer)();
    }
    VLOG(3) << "H2D staging copy done";
    return staging_buffer;
  };

  auto copy_to_gpu = [device, packed_size, data,
                      copy_event(std::move(copy_event)),
                      dst_definition_event(std::move(dst_definition_event)),
                      gpu_buffer{gpu_buffer.CopyRef()}](
                         HostMemoryAllocator::OwnedPtr staging_buffer) {
    tsl::profiler::TraceMe traceme([&] {
      return tsl::profiler::TraceMeEncode(
          "BufferFromHostBuffer::H2D_GPU_copy",
          {{"device", device->id()}, {"size", packed_size}});
    });
    auto stream = device->stream();

    se::DeviceMemoryBase dest = gpu_buffer->buffer();
    const void* host_data_ptr;
    if (staging_buffer) {
      host_data_ptr = staging_buffer.get();
    } else {
      host_data_ptr = data;
    }
    VLOG(3) << "H2D copy: " << host_data_ptr << " -> " << dest.opaque() << " ("
            << packed_size << " bytes)";
    absl::Status status = stream->Memcpy(&dest, host_data_ptr, packed_size);
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
  auto h2d_copy = [this, use_staging_buffer,
                   copy_to_staging_buffer(std::move(copy_to_staging_buffer)),
                   copy_to_gpu(std::move(copy_to_gpu))]() mutable {
    HostMemoryAllocator::OwnedPtr staging_buffer;
    if (use_staging_buffer) {
      staging_buffer = copy_to_staging_buffer();
    }

    EnqueueWork(blocking_thread_pool_.get(),
                [copy_to_gpu(std::move(copy_to_gpu)),
                 staging_buffer(std::move(staging_buffer))]() mutable {
                  copy_to_gpu(std::move(staging_buffer));
                });
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
  VLOG(4) << "TfrtGpuClient::BufferFromHostLiteral: shape: "
          << literal.shape().ToString() << " device: " << device->DebugString();
  const Shape& shape = literal.shape();

  // Add a placeholder definition event for each leaf buffer when creating the
  // buffer. They are set only after h2d dispatch.
  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events;
  absl::InlinedVector<tsl::RCReference<tsl::AsyncValue>, 4> avs;
  int num_leaf_buffers = shape.IsTuple() ? shape.tuple_shapes().size() : 1;
  for (int i = 0; i < num_leaf_buffers; ++i) {
    tsl::AsyncValueRef<GpuEvent> definition_event =
        tsl::MakeConstructedAsyncValueRef<GpuEvent>();
    definition_events.push_back(definition_event.CopyRef());
    avs.push_back(std::move(definition_event));
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TfrtGpuBuffer> output_buffer,
      AllocateTfrtGpuDestinationBuffer(shape, AfterAll(definition_events),
                                       tsl::down_cast<TfrtGpuDevice*>(device),
                                       this, memory_space));

  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = output_buffer->AcquireUsage(usage_event);
  CHECK(device_buffer);
  if (shape.IsTuple()) {
    return Unimplemented(
        "Tuple case is not supported in TfrtGpuClient::BufferFromHostLiteral");
  }
  // It is OK to capture `buffer` pointer because the `output_buffer` can't
  // be deleted until all the usage holds have gone away.
  VLOG(4) << "BufferFromHostLiteral for device_buffer: " << device_buffer;
  EnqueueWork(
      non_blocking_thread_pool_.get(),
      [literal, av = avs[0], device_buffer, shape, this,
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
                << device_buffer << " AsyncValue: " << av.get();

        av->SetStateConcrete();
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

namespace {

absl::StatusOr<std::unique_ptr<tsl::Allocator>> CreateAllocatorForDevice(
    se::StreamExecutor* executor, const GpuAllocatorConfig& allocator_config) {
  switch (allocator_config.kind) {
    case GpuAllocatorConfig::Kind::kCudaAsync:
      return absl::UnimplementedError(
          "CudaAsync allocator is not supported in TfrtGpuClient.");
    case GpuAllocatorConfig::Kind::kDefault:
    case GpuAllocatorConfig::Kind::kBFC:
      LOG_FIRST_N(INFO, 1) << "Using BFC allocator.";
      return CreateBFCAllocator(executor, allocator_config.memory_fraction,
                                allocator_config.preallocate,
                                allocator_config.gpu_system_memory_size);
    case GpuAllocatorConfig::Kind::kPlatform:
      LOG(FATAL) << "Platform allocator should be handled before calling this "
                    "function.";
  }
}

absl::StatusOr<MaybeOwning<se::DeviceMemoryAllocator>> CreateDeviceAllocator(
    LocalClient* xla_client, const GpuAllocatorConfig& allocator_config,
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& devices) {
  if (allocator_config.kind == GpuAllocatorConfig::Kind::kPlatform) {
    LOG(INFO) << "Using platform allocator.";
    if (allocator_config.collective_memory_size != 0) {
      LOG(WARNING)
          << "collective_memory_size is non-zero, but allocator kind is set "
             "to \"platform\". Collective memory will not be allocated.";
    }
    return MaybeOwning<se::DeviceMemoryAllocator>(
        xla_client->backend().memory_allocator());
  }

  std::vector<se::MultiDeviceAdapter::AllocatorInfo> allocators;
  for (const auto& device : devices) {
    se::StreamExecutor* executor = device->executor();
    if (executor == nullptr) {
      // Skips remote devices.
      continue;
    }

    // The stream in the allocator will be used during compilation.
    se::Stream* stream = device->stream();
    TF_ASSIGN_OR_RETURN(auto allocator,
                        CreateAllocatorForDevice(executor, allocator_config));
    allocators.emplace_back(
        std::move(allocator), stream,
        /*memory_space=*/static_cast<int>(se::MemoryType::kDevice),
        executor->device_ordinal(), executor->GetPlatform());

    TF_ASSIGN_OR_RETURN(
        auto collective_bfc_allocator,
        CreateCollectiveBFCAllocator(
            executor,
            /*memory_fraction=*/1.0 - allocator_config.memory_fraction,
            allocator_config.collective_memory_size));
    allocators.emplace_back(std::move(collective_bfc_allocator), stream,
                            /*memory_space=*/1, executor->device_ordinal(),
                            executor->GetPlatform());

    TF_ASSIGN_OR_RETURN(auto host_allocator, GetGpuHostAllocator(executor));
    allocators.emplace_back(
        std::move(host_allocator), stream,
        /*memory_space=*/static_cast<int>(se::MemoryType::kHost),
        executor->device_ordinal(), executor->GetPlatform());
  }
  return MaybeOwning<se::DeviceMemoryAllocator>(
      std::make_unique<se::MultiDeviceAdapter>(xla_client->platform(),
                                               std::move(allocators)));
}

using DeviceTopologyPair =
    std::pair<std::vector<std::unique_ptr<TfrtGpuDevice>>, GpuTopologyProto>;

absl::StatusOr<DeviceTopologyPair> BuildDistributedDevices(
    absl::string_view platform_name, LocalClient* xla_client, int node_id,
    int num_nodes, gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store, bool enable_mock_nccl,
    std::optional<absl::string_view> mock_gpu_topology,
    std::optional<int> slice_index, absl::Duration get_local_topology_timeout,
    absl::Duration get_global_topology_timeout) {
  std::vector<std::unique_ptr<TfrtGpuDevice>> devices;
  LocalTopologyProto local_topology;
  local_topology.set_node_id(node_id);
  auto boot_id_str_or_status = GetBootIdString();
  if (!boot_id_str_or_status.ok()) {
    LOG(INFO) << boot_id_str_or_status.status();
  } else {
    local_topology.set_boot_id(boot_id_str_or_status.value());
  }
  if (slice_index.has_value()) {
    local_topology.set_slice_index(*slice_index);
  }
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    const se::Platform* platform = executor->GetPlatform();

    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(executor->device_ordinal()));
    DeviceProto* device_proto = local_topology.add_devices();
    device_proto->set_local_device_ordinal(executor->device_ordinal());
    device_proto->set_name(desc->name());
    device_proto->set_vendor(desc->device_vendor());
    device_proto->set_compute_capability(
        MakeComputeCapabilityString(desc.get()));
    device_proto->set_core_count(desc->core_count());

    // TODO: hhb
    // const se::GpuComputeCapability& compute_capability =
    //     desc->gpu_compute_capability();
    // if (std::holds_alternative<se::CudaComputeCapability>(compute_capability)
    // &&
    //     std::get<se::CudaComputeCapability>(compute_capability).major >= 9) {
    //   auto fabric_info = GetDeviceFabricInfo(executor->device_ordinal());
    //   if (fabric_info.ok()) {
    //     device_proto->set_fabric_uuid(*fabric_info);
    //   }
    // }
  }

  GlobalTopologyProto global_topology;
  if (enable_mock_nccl) {
    TopologySizes sizes;
    if (mock_gpu_topology.has_value()) {
      TF_ASSIGN_OR_RETURN(sizes, TopologySizes::FromString(*mock_gpu_topology));
    } else {
      // If there is no topology spec, we assume that each node is a slice,
      // there is one process (host) on each slice and each host
      // has all the local devices.
      sizes.num_slices = num_nodes;
      sizes.num_hosts_per_slice = 1;
      sizes.num_devices_per_host = local_topology.devices().size();
    }

    if (sizes.num_devices_per_host != local_topology.devices().size()) {
      return absl::InternalError(
          "The number of devices per host in 'mock_gpu_topology' "
          "must be the same as the number of devices in the local topology");
    }

    if (sizes.num_slices * sizes.num_hosts_per_slice != num_nodes) {
      return absl::InternalError(
          "The number of hosts in 'mock_gpu_topology' "
          "must be the same as 'num_nodes'");
    }

    std::vector<LocalTopologyProto> local_topologies(num_nodes, local_topology);
    for (int i = 0; i < sizes.num_slices; ++i) {
      for (int j = 0; j < sizes.num_hosts_per_slice; j++) {
        int node_id = i * sizes.num_hosts_per_slice + j;
        local_topologies[node_id].set_node_id(node_id);
        local_topologies[node_id].set_boot_id(absl::StrCat(i));
      }
    }
    TF_ASSIGN_OR_RETURN(global_topology,
                        BuildGlobalTopology(absl::MakeSpan(local_topologies),
                                            /*assign_global_device_ids=*/true));
  } else {
    TF_RETURN_IF_ERROR(ExchangeTopologies(
        platform_name, node_id, num_nodes, get_local_topology_timeout,
        get_global_topology_timeout, kv_store.get(), local_topology,
        &global_topology, /*assign_global_device_ids=*/true));
  }

  std::map<int, GlobalDeviceId> gpu_device_ids;
  absl::flat_hash_map<GlobalDeviceId, int> device_to_node;
  for (const LocalTopologyProto& node : global_topology.nodes()) {
    for (const DeviceProto& device_proto : node.devices()) {
      GlobalDeviceId global_device_id(device_proto.global_device_id());
      device_to_node[global_device_id] = node.node_id();
      TfrtGpuDevice::Options options;
      if (node.node_id() == node_id) {
        gpu_device_ids[device_proto.local_device_ordinal()] = global_device_id;
        // Assign some descriptive names for profiling tools.
        // TODO: hhb
        // NameDeviceAndLauncherThread(node, device_proto,
        //                             local_device->execute_thread());

        TF_ASSIGN_OR_RETURN(se::StreamExecutor * executor,
                            xla_client->backend().stream_executor(
                                device_proto.local_device_ordinal()));
        options.local_device_id = executor->device_ordinal();
        options.local_hardware_id = executor->device_ordinal();
        options.executor = executor;
      } else {
        options.local_device_id = -1;
        options.local_hardware_id = -1;
        options.executor = nullptr;
      }
      options.id = device_proto.global_device_id();
      options.process_index = node.node_id();
      options.slice_index = device_proto.slice_index();
      options.max_inflight_computations = 32;
      options.platform_version = device_proto.name();
      options.device_vendor = device_proto.vendor();
      options.compute_capability = device_proto.compute_capability();
      options.core_count = device_proto.core_count();

      auto device = std::make_unique<TfrtGpuDevice>(std::move(options));
      devices.push_back(std::move(device));
    }
  }
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    TF_RET_CHECK(gpu_device_ids.find(executor->device_ordinal()) !=
                 gpu_device_ids.end());
  }
  gpu_executable_run_options->set_gpu_global_device_ids(
      std::move(gpu_device_ids));

  TF_ASSIGN_OR_RETURN(xla::Collectives * collectives,
                      xla::CollectivesRegistry::Default("gpu"));
  xla::gpu::GpuCollectives* gpu_collectives =
      tsl::down_cast<xla::gpu::GpuCollectives*>(collectives);

  if (gpu_collectives == nullptr) {
    return absl::InternalError("Failed to get GPU collectives");
  }

  TF_RETURN_IF_ERROR(gpu_collectives->InitializeTopology(
      {node_id, global_topology.nodes().size(),
       xla_client->backend().stream_executors().size(), kv_store,
       device_to_node, gpu_executable_run_options}));

  TF_ASSIGN_OR_RETURN(GpuTopologyProto gpu_topology,
                      BuildGpuTopology(global_topology));
  return std::make_pair(std::move(devices), gpu_topology);
}

}  // namespace

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
      BuildDistributedDevices(pjrt_platform_name, xla_client, options.node_id,
                              options.num_nodes, gpu_run_options.get(),
                              kv_store, options.enable_mock_nccl,
                              options.mock_gpu_topology, options.slice_index,
                              absl::Minutes(2), absl::Minutes(5)));

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
  MarkGpuEventReadyOnExit ready_on_exit(std::move(usage_event));

  // Wait for the definition event.
  const auto& av = device_buffer->definition_event();
  tsl::BlockUntilReady(av);
  if (auto* error = av.GetErrorIfPresent()) {
    return absl::InternalError(
        absl::StrFormat("Error Execute: %s", error->message()));
  }

  const auto& buffer = device_buffer->buffer();

  ShapedBuffer shaped_buffer =
      buffer->AsShapedBuffer(on_device_shape_, device_);
  Shape ret_shape = on_device_shape_;
  TransferManager* transfer_manager =
      client_->xla_client()->backend().transfer_manager();

  auto stream = device_->stream();
  TF_RETURN_IF_ERROR(
      transfer_manager->ReadDynamicShapes(stream, &shaped_buffer, &ret_shape));
  {
    tsl::profiler::TraceMe traceme("BlockHostUntilDone");
    TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  }
  return ret_shape;
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

  auto copy_to_host = [device(device_), device_buffer,
                       usage_event(std::move(usage_event)), literal, promise,
                       client = client_, on_device_shape{on_device_shape_},
                       unpack_subbyte_types, transpose]() mutable {
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

    HostMemoryAllocator::OwnedPtr staging_buffer;
    void* buffer_ptr;
    if (on_device_shape.IsArray()) {
      staging_buffer = client->host_memory_allocator()->Allocate(byte_size);
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

      VLOG(3) << "D2H copy: " << device_buffer->buffer()->buffer().opaque()
              << " -> " << buffer_ptr << " (" << byte_size << " bytes)";
      CHECK_OK(stream->Memcpy(buffer_ptr, device_buffer->buffer()->buffer(),
                              byte_size))
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
        buffer = tsl::port::AlignedMalloc(unpacked_size,
                                          tsl::Allocator::kAllocatorAlignment);
      } else {
        buffer = literal->untyped_data();
      }
      primitive_util::UnpackIntN(
          on_device_shape.element_type(),
          absl::MakeConstSpan(static_cast<const char*>(buffer_ptr), byte_size),
          absl::MakeSpan(static_cast<char*>(buffer), unpacked_size));
      VLOG(3) << "D2H staging copy done";
    } else {
      buffer = buffer_ptr;
    }
    if (transpose != nullptr) {
      tsl::profiler::TraceMe traceme("Transpose");
      transpose->Execute(buffer, static_cast<char*>(literal->untyped_data()));
      if (should_unpack) {
        tsl::port::AlignedFree(buffer);
      }
    }
    if (on_device_shape.IsArray() && !should_unpack && transpose == nullptr) {
      std::memcpy(literal->untyped_data(), buffer, byte_size);
    }
    promise.Set(absl::OkStatus());
  };
  EnqueueWorkWhenReady(client_->blocking_thread_pool(),
                       {device_buffer->definition_event().CopyRCRef()},
                       std::move(copy_to_host));

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
    absl::AnyInvocable<absl::StatusOr<MutableLiteralBase*>() &&> generator) {
  auto buffer = std::move(generator)();
  if (!buffer.ok()) {
    return PjRtFuture<>(buffer.status());
  }
  return ToLiteral(buffer.value());
}

// TODO - b/383558503: Fix cognitive complexity from ClangTidy.
PjRtFuture<> TfrtGpuBuffer::CopyRawToHostFuture(PjRtFuture<void*> dst,
                                                int64_t offset,
                                                int64_t transfer_size) {
  VLOG(3) << "TfrtGpuBuffer::CopyRawToHostFuture";
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::CopyRawToHostFuture");
  auto promise = PjRtFuture<>::CreatePromise();
  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  auto* device_buffer = AcquireUsage(usage_event);
  if (device_buffer == nullptr) {
    return PjRtFuture<>(
        InvalidArgument("ToLiteral() called on deleted or donated buffer"));
  }
  EnqueueWorkWhenReady(
      client_->blocking_thread_pool(),
      {device_buffer->definition_event().CopyRCRef()},
      [device(device_), device_buffer, usage_event(std::move(usage_event)), dst,
       promise, client = client_, offset, transfer_size]() mutable {
        tsl::profiler::TraceMe traceme("CopyRawToHostFuture::D2H_copy");
        if (device_buffer->definition_event().IsError()) {
          usage_event.SetStateConcrete();
          LOG(ERROR) << "device_buffer->definition_event().GetError(): "
                     << device_buffer->definition_event().GetError();
          promise.Set(device_buffer->definition_event().GetError());
          return;
        }
        se::DeviceMemoryBase device_memory = device_buffer->buffer()->buffer();
        if (offset < 0 || offset > device_memory.size() ||
            device_memory.size() - offset < transfer_size) {
          usage_event.SetStateConcrete();
          LOG(ERROR) << "Copy raw buffer called on buffer size "
                     << device_memory.size() << " with invalid offset "
                     << offset << ", transfer size " << transfer_size;
          promise.Set(
              InvalidArgument("Copy raw buffer called on buffer size %lld with "
                              "invalid offset %lld, transfer size %lld",
                              device_memory.size(), offset, transfer_size));
          return;
        }

        std::unique_ptr<se::DeviceMemoryBase> sub_buffer;
        if (transfer_size < device_memory.size()) {
          sub_buffer = std::make_unique<se::DeviceMemoryBase>(
              device_memory.GetByteSlice(offset, transfer_size));
        } else {
          sub_buffer = std::make_unique<se::DeviceMemoryBase>(device_memory);
        }
        dst.OnReady([client = std::move(client), promise = std::move(promise),
                     usage_event = std::move(usage_event),
                     device = std::move(device),
                     sub_buffer = std::move(sub_buffer),
                     transfer_size](absl::StatusOr<void*> dst) mutable {
          HostMemoryAllocator::OwnedPtr staging_buffer;
          if (client->should_stage_host_to_device_transfers() &&
              !client->IsDmaMapped(dst.value(), transfer_size)) {
            staging_buffer =
                client->host_memory_allocator()->Allocate(transfer_size);
          }

          {
            tsl::profiler::TraceMe traceme2([&] {
              return tsl::profiler::TraceMeEncode(
                  "CopyRawToHostFuture::D2H_GPU_copy",
                  {
                      {"device", device->id()},
                      {"size", transfer_size},
                  });
            });
            MarkGpuEventReadyOnExit ready_on_exit(std::move(usage_event));
            auto stream = device->stream();
            void* host_ptr =
                staging_buffer != nullptr ? staging_buffer.get() : dst.value();

            VLOG(3) << "D2H copy: " << sub_buffer->opaque() << " -> "
                    << host_ptr << " (" << transfer_size << " bytes)";
            CHECK_OK(stream->Memcpy(host_ptr, *sub_buffer, transfer_size))
                << "stream->Memcpy failed copying from GPU to host";
            absl::Status status;
            {
              tsl::profiler::TraceMe traceme("BlockHostUntilDone");
              status = stream->BlockHostUntilDone();
            }
            VLOG(3) << "D2H copy done. " << status;
            if (!status.ok()) {
              LOG(ERROR) << "stream->BlockHostUntilDone failed: " << status;
              promise.Set(status);
              return;
            }
          }
          if (!dst.ok()) {
            promise.Set(dst.status());
            LOG(ERROR) << "dst.status(): " << dst.status();
            return;
          }
          if (staging_buffer != nullptr) {
            tsl::profiler::TraceMe traceme3(
                "CopyRawToHostFuture::D2H staging copy");
            std::memcpy(dst.value(), staging_buffer.get(), transfer_size);
            VLOG(3) << "D2H staging copy done: " << staging_buffer.get()
                    << " -> " << dst.value() << " (" << transfer_size
                    << " bytes)";
          }
          promise.Set(absl::OkStatus());
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

bool TfrtGpuBuffer::IsDeleted() {
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

        // TODO: Use the destination device stream for D2D copies.
        auto stream = src_device->stream();
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
                       {src_device_buffer->definition_event().CopyRCRef()},
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

TfrtGpuExecutable::TfrtGpuExecutable(
    std::vector<std::unique_ptr<LocalExecutable>> executables,
    bool parameter_is_tupled_arguments,
    std::shared_ptr<DeviceAssignment> device_assignment,
    CompileOptions compile_options,
    std::vector<LogicalDeviceIds> addressable_device_logical_ids,
    std::vector<PjRtDevice*> addressable_devices, TfrtGpuClient* client)
    : client_(client),
      device_assignment_(std::move(device_assignment)),
      compile_options_(std::move(compile_options)),
      parameter_is_tupled_arguments_(parameter_is_tupled_arguments),
      addressable_device_logical_ids_(
          std::move(addressable_device_logical_ids)),
      addressable_devices_(std::move(addressable_devices)) {
  TransferManager* transfer_manager =
      client_->xla_client()->backend().transfer_manager();
  tsl::Fprint128 fingerprint = tsl::Fingerprint128(fingerprint_);
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
    on_device_executable_parameter_shapes_.push_back(
        std::make_shared<std::vector<Shape>>(std::move(parameter_shapes)));

    auto input_buffer_sizes_in_bytes = std::make_shared<std::vector<int64_t>>();

    // Assume compiled program expects either many non-tupled arguments or a
    // singled tupled argument, or no arguments. Nested tuple is not yet
    // supported.
    if (computation_layout.parameter_count() == 0) {
      // No arguments. Do nothing.
    } else if (computation_layout.parameter_count() == 1 &&
               computation_layout.parameter_shape(0).IsTuple()) {
      const std::vector<Shape>& tuple_shapes =
          computation_layout.parameter_shape(0).tuple_shapes();
      input_buffer_sizes_in_bytes->reserve(tuple_shapes.size());
      for (const Shape& shape : tuple_shapes) {
        input_buffer_sizes_in_bytes->push_back(ShapeUtil::ByteSizeOf(shape));
      }
    } else {
      const std::vector<ShapeLayout>& parameter_layouts =
          computation_layout.parameter_layouts();
      input_buffer_sizes_in_bytes->reserve(parameter_layouts.size());
      for (const ShapeLayout& layout : parameter_layouts) {
        input_buffer_sizes_in_bytes->push_back(
            ShapeUtil::ByteSizeOf(layout.shape()));
      }
    }
    input_buffer_sizes_in_bytes_.push_back(
        std::move(input_buffer_sizes_in_bytes));

    fingerprint = tsl::FingerprintCat128(
        fingerprint,
        tsl::Fingerprint128(executable->executable()->module().ToString(
            HloPrintOptions::ModuleFingerprint())));
    executables_.emplace_back(std::move(executable));
  }
  fingerprint_ = absl::StrCat(fingerprint.low64, fingerprint.high64);

  int num_partitions;
  if (device_assignment_ == nullptr) {
    // This must go after `executables_` is initialized.
    VLOG(4) << "TfrtGpuExecutable portable single-core";
    num_partitions = 1;
    CHECK(addressable_devices_.empty());
  } else {
    // This must go after `executables_` is initialized.
    VLOG(4) << "TfrtGpuExecutable device_assignment:\n"
            << device_assignment_->ToString();

    if ((device_assignment_->replica_count() > 1 ||
         device_assignment_->computation_count() > 1) &&
        IsAllZeros(*device_assignment_)) {
      // This code path should only be triggered when we intentionally compile
      // an HLO without having enough devices to actually run it. See the
      // "--run=false" option in
      // tensorflow/compiler/xla/tools/multihost_hlo_runner/hlo_runner_main.cc.
      // That will help us debug the XLA compiler locally.
      LOG(INFO) << "A workaround is in effect to allow compiling multi-device "
                   "HLOs on machines with fewer devices. Don't run this "
                   "executable.";
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

absl::StatusOr<PjRtLoadedExecutable::Result> TfrtGpuExecutable::ExecuteHelper(
    absl::Span<PjRtBuffer* const> argument_handles, int replica, int partition,
    const ExecuteOptions& options, bool fill_future, TfrtGpuDevice* device) {
  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int device_id = (*device_assignment_)(replica, partition);
    VLOG(3) << "device_id: " << device_id;
    TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                        client_->LookupDevice(PjRtGlobalDeviceId(device_id)));
    device = tsl::down_cast<TfrtGpuDevice*>(pjrt_device);
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

  tsl::profiler::TraceMeProducer activity(
      [&] {
        return tsl::profiler::TraceMeEncode("TfrtGpuExecutable::ExecuteHelper",
                                            {{"launch_id", options.launch_id},
                                             {"device_id", device->id()},
                                             {"name", name()}});
      },
      tsl::profiler::ContextType::kPjRt, options.launch_id);

  VLOG(1) << "ExecuteHelper " << name() << ": " << options.launch_id
          << "; replica: " << replica << "; partition: " << partition
          << "; mapped to device ordinal for execution: " << device->id();

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  std::unique_ptr<Semaphore::ScopedReservation> compute_reservation;
  {
    tsl::profiler::TraceMe traceme_compute_reservation(
        "TfrtGpuExecutable::ExecuteHelper::acquire_semaphore");

    VLOG(1) << "Trying to acquire semaphore for " << name() << " on device "
            << device->DebugString();
    compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
        device->max_inflight_computations_semaphore().ScopedAcquire(1));
    VLOG(1) << "Acquired semaphore for " << name() << " on device "
            << device->DebugString();
  }

  // Handle inputs.
  if (options.arguments_are_tupled) {
    if (!parameter_is_tupled_arguments_) {
      return InvalidArgument(
          "Arguments may only be supplied as a tuple when the executable was"
          "compiled with a single tupled parameter");
    }
    if (argument_handles.size() != 1) {
      return InvalidArgument(
          "Option arguments_are_tupled was true but %d buffers were passed to"
          "execution",
          argument_handles.size());
    }
  }

  // SPMD sharding produces a single executable for multiple partitions.
  int executable_idx = executables_.size() > 1 ? partition : 0;

  // `scheduled_event` indicates whether gpu computation is dispatched to the
  // stream and whether there was an error.
  auto scheduled_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();

  // `complete_event` indicates whether gpu computation is complete and whether
  // there was an error.
  tsl::AsyncValueRef<GpuEvent> complete_event =
      tsl::MakeConstructedAsyncValueRef<GpuEvent>();

  absl::InlinedVector<TfrtGpuBuffer::DonationTransaction, 4>
      donation_transactions;

  absl::InlinedVector<TrackedGpuDeviceBuffer*, 4> tracked_buffers;
  absl::InlinedVector<bool, 4> buffer_is_donated;
  tracked_buffers.reserve(argument_handles.size());
  buffer_is_donated.reserve(argument_handles.size());
  // To avoid clobbering inputs, we must ensure that
  //   `extra_deps` = inputs' definition events + donated inputs' usage events.
  // This also ensures that the returned `complete_event` dominates all inputs'
  // events, and thus output buffer only need to contain `complete_event` as
  // the single definition event.
  std::vector<tsl::RCReference<tsl::AsyncValue>> prepare_input_deps;
  std::vector<tsl::RCReference<tsl::AsyncValue>> input_deps;
  input_deps.reserve(argument_handles.size() + 1);

  absl::Span<int const> donated_params =
      parameters_that_must_be_donated_[executable_idx];
  auto donate_it = donated_params.begin();

  absl::flat_hash_map<const void*, std::pair<bool, int>> donation_clashes;
  donation_clashes.reserve(argument_handles.size());
  for (int i = 0; i < argument_handles.size(); ++i) {
    PjRtBuffer* handle = argument_handles[i];
    auto* tfrt_buffer = tsl::down_cast<TfrtGpuBuffer*>(handle);

    if (tfrt_buffer->device() != device) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %s, but replica is assigned to device %s.",
          i, replica, tfrt_buffer->device()->DebugString(),
          device->DebugString());
    }
    bool donation_denied_at_runtime =
        options.non_donatable_input_indices.contains(i);
    bool must_donate = donate_it != donated_params.end() && *donate_it == i &&
                       !donation_denied_at_runtime;
    auto tracked_buffer_or = [&]() -> absl::StatusOr<TrackedGpuDeviceBuffer*> {
      TrackedGpuDeviceBuffer* tracked_buffer = nullptr;
      if (must_donate) {
        VLOG(3) << "Buffer for argument_handles[" << i << "] is donated";

        ++donate_it;
        TF_RETURN_IF_ERROR(TestBufferDonationClashes(
            handle, donation_clashes, must_donate, i, replica, partition));
        TF_ASSIGN_OR_RETURN(auto donation_transaction,
                            tfrt_buffer->AcquireDonation());

        // After acquiring the buffer for donation, we retrieve the dependent
        // usage events. Note that we don't need any locking here as
        // AcquireDonation() is supposed to synchronize with other usages.
        input_deps.push_back(
            donation_transaction.device_buffer()->AfterAllUsageEvents());
        tracked_buffer = donation_transaction.device_buffer();
        donation_transactions.push_back(std::move(donation_transaction));
        buffer_is_donated.push_back(true);
      } else {
        tracked_buffer = tfrt_buffer->AcquireUsage(complete_event);
        if (!tracked_buffer) {
          return InvalidArgument(
              "Invalid buffer passed: buffer has been deleted or donated.");
        }
        buffer_is_donated.push_back(false);
      }
      return tracked_buffer;
    }();

    if (!tracked_buffer_or.ok()) {
      // If something failed when preparing the input, we still need to add it
      // to the input deps so that it can poison the output buffers.
      auto error_av = tsl::MakeErrorAsyncValueRef(tracked_buffer_or.status());
      prepare_input_deps.push_back(error_av);
      input_deps.push_back(error_av);

      LOG(ERROR) << "argument_handles[" << i
                 << "]: failed to get tracked buffer with status "
                 << tracked_buffer_or.status();
    } else {
      TrackedGpuDeviceBuffer* tracked_buffer = tracked_buffer_or.value();
      tracked_buffers.push_back(tracked_buffer);
      prepare_input_deps.push_back(tracked_buffer->buffer().CopyRCRef());

      VLOG(3) << "argument_handles[" << i
              << "]: addr = " << tracked_buffer->buffer()->buffer().opaque()
              << ", logical shape = "
              << tfrt_buffer->logical_on_device_shape()->ToString();

      // Definition events are never modified after buffer construction. If they
      // are available and have no error, they can be skipped in input deps.
      // In contrast, already known errors in the input are taken as deps so
      // that they can poison output buffers.
      const auto& definition_event = tracked_buffer->definition_event();
      if (!definition_event.IsAvailable() || definition_event.IsError()) {
        VLOG(3) << "definition_event is not available: AsyncValue pointer: "
                << definition_event.GetAsyncValue();
        input_deps.push_back(definition_event.CopyRCRef());
      }
    }
  }

  {
    // Schedule only one collective at a time.
    tsl::AsyncValueRef<GpuEvent> ordering_event =
        tsl::MakeConstructedAsyncValueRef<GpuEvent>();
    tsl::AsyncValueRef<GpuEvent> last_collective_launch_event =
        device->SetLastCollectiveLaunchEvent(scheduled_event);
    // We don't use last_collective_launch_event directly because we don't
    // want the previous failure to be propagated to the current execution.
    last_collective_launch_event.AndThen(
        [event = ordering_event.CopyRef()]() { event.SetStateConcrete(); });
    input_deps.push_back(std::move(ordering_event));
  }

  std::vector<tsl::AsyncValueRef<GpuDeviceMemory>> output_buffers;
  std::vector<std::unique_ptr<PjRtBuffer>> outputs;
  auto gpu_executable = executables_[executable_idx];
  TF_ASSIGN_OR_RETURN(std::vector<Shape> output_shapes, GetOutputShapes());
  const Shape& result_shape = output_shapes[executable_idx];
  bool untuple_result = options.untuple_result;
  bool result_is_tuple = result_shape.IsTuple();
  if (options.untuple_result && result_shape.IsTuple()) {
    output_buffers.reserve(result_shape.tuple_shapes().size());
    outputs.reserve(output_buffers.size());
    for (int i = 0; i < result_shape.tuple_shapes().size(); ++i) {
      output_buffers.push_back(
          tsl::MakeUnconstructedAsyncValueRef<GpuDeviceMemory>());
      // Program execution writes to output buffers so it's a definition
      // event.
      auto leaf_tracked_device_buffer =
          std::make_unique<TrackedGpuDeviceBuffer>(
              output_buffers.back().CopyRef(), scheduled_event.CopyRef(),
              complete_event.CopyRef());
      VLOG(4) << "created leaf_tracked_device_buffer: "
              << leaf_tracked_device_buffer.get();

      const Shape& shape = result_shape.tuple_shapes(i);
      PjRtMemorySpace* memory_space =
          device->default_memory_space().value_or(nullptr);
      if (shape.has_layout() &&
          shape.layout().memory_space() == Layout::kHostMemorySpace) {
        TF_ASSIGN_OR_RETURN(memory_space, device->memory_space_by_kind_id(
                                              PinnedHostMemorySpace::kKindId));
      }

      auto output = std::make_unique<TfrtGpuBuffer>(
          result_shape.tuple_shapes(i), std::move(leaf_tracked_device_buffer),
          client_, device, memory_space);
      outputs.push_back(std::move(output));
    }
  } else {
    output_buffers.push_back(
        tsl::MakeUnconstructedAsyncValueRef<GpuDeviceMemory>());
    // Program execution writes to output buffers so it's a definition event.
    auto tracked_device_buffer = std::make_unique<TrackedGpuDeviceBuffer>(
        output_buffers.back().CopyRef(),
        /*definition_event=*/scheduled_event.CopyRef(),
        complete_event.CopyRef());
    VLOG(4) << "created tracked_device_buffer: " << tracked_device_buffer.get();

    const Shape& shape = result_shape;
    PjRtMemorySpace* memory_space =
        device->default_memory_space().value_or(nullptr);
    if (shape.has_layout() &&
        shape.layout().memory_space() == Layout::kHostMemorySpace) {
      TF_ASSIGN_OR_RETURN(memory_space, device->memory_space_by_kind_id(
                                            PinnedHostMemorySpace::kKindId));
    }

    auto tfrt_output_buffer = std::make_unique<TfrtGpuBuffer>(
        result_shape, std::move(tracked_device_buffer), client_, device,
        memory_space);
    outputs.push_back(std::move(tfrt_output_buffer));
  }

  auto ffi_context =
      options.context != nullptr ? &options.context->ffi_context() : nullptr;

  // Create a PjRt<->StreamExecutor adaptors to send/recv device memory as
  // PjRt chunks via the user-provided callbacks.
  SendDeviceMemoryFunction send_device_memory =
      ConvertSendCallbacksToSendFunction(replica, options,
                                         client_->non_blocking_thread_pool());
  RecvDeviceMemoryFunction recv_device_memory =
      ConvertRecvCallbacksToRecvFunction(replica, options);

  auto execute_fn =
      [replica, partition, device, launch_id(options.launch_id),
       output_buffers(output_buffers), complete_event(complete_event.CopyRef()),
       scheduled_event(scheduled_event.CopyRef()),
       untuple_result(untuple_result), result_is_tuple(result_is_tuple),
       donation_transactions(std::move(donation_transactions)),
       parameter_shapes(on_device_executable_parameter_shapes_[executable_idx]),
       gpu_executable(std::move(gpu_executable)),
       device_assignment(device_assignment), executable_name(name()),
       ffi_context(ffi_context), inputs_avs(CopyAsyncValues(input_deps)),
       execution_profile(options.execution_profile),
       send_device_memory(std::move(send_device_memory)),
       recv_device_memory(std::move(recv_device_memory)),
       compute_reservation(std::move(compute_reservation)),
       client = client_](std::vector<ExecutionInput> execution_inputs) mutable {
        VLOG(1) << "execute_fn for " << executable_name
                << ", launch_id: " << launch_id << ", replica: " << replica
                << ", device: " << device->DebugString();

        tsl::profiler::TraceMeConsumer producer(
            [&] {
              return tsl::profiler::TraceMeEncode(
                  "execute_fn", {
                                    {"launch_id", launch_id},
                                    {"device_id", device->id()},
                                });
            },
            tsl::profiler::ContextType::kPjRt, launch_id);

        auto set_error = [&](absl::Status status) {
          for (auto& output_buffer : output_buffers) {
            output_buffer.SetError(status);
          }
          complete_event.SetError(status);
          scheduled_event.SetError(status);
        };

        for (const auto& av : inputs_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            set_error(*error);
            return;
          }
        }

        auto stream = device->stream();
        ExecutableRunOptions run_options;
        run_options.set_stream(stream);
        run_options.set_host_to_device_stream(stream);
        run_options.set_device_to_host_stream(stream);
        run_options.set_allocator(client->allocator());
        run_options.set_device_assignment(device_assignment.get());
        run_options.set_run_id(RunId(launch_id));
        run_options.set_rng_seed(device->GetNewPrngSeed());
        run_options.set_gpu_executable_run_options(
            CHECK_NOTNULL(client->gpu_run_options()));
        run_options.set_launch_id(launch_id);
        run_options.set_local_device_count(client->device_count());
        run_options.set_device_ordinal(device->local_device_id().value());
        run_options.set_physical_device_ordinal(
            device->local_hardware_id().value());
        run_options.set_ffi_execution_context(ffi_context);
        run_options.set_intra_op_thread_pool(
            client->xla_client()
                ->backend()
                .eigen_intra_op_thread_pool_device());
        run_options.set_send_device_memory_function(&send_device_memory);
        run_options.set_recv_device_memory_function(&recv_device_memory);
        run_options.set_execution_profile(execution_profile);

        // TODO(phawkins): *technically* this should probably happen after
        // calling RunAsync(). But that causes a large performance problem: it
        // prevents the main thread from freeing the buffer objects.
        for (auto& donation_transaction : donation_transactions) {
          VLOG(3) << "Committing donation transaction: "
                  << donation_transaction.device_buffer();
          std::move(donation_transaction).Commit();
        }

        VLOG(1) << "Start calling RunAsync for " << executable_name
                << ", device=" << device->DebugString()
                << ", launch_id=" << launch_id << ", replica=" << replica
                << ", partition=" << partition;

        if (VLOG_IS_ON(2)) {
          absl::Status host_callback_status =
              stream->DoHostCallback([executable_name, launch_id, device]() {
                VLOG(1) << "Start device execution for " << executable_name
                        << ", launch_id: " << launch_id
                        << ", device: " << device->DebugString();
              });
          if (!host_callback_status.ok()) {
            LOG(WARNING)
                << "Failed to do host callback for start device execution for "
                << executable_name << ", status = " << host_callback_status;
          }
        }

        absl::StatusOr<ExecutionOutput> result_buffer_or_status =
            gpu_executable->RunAsync(std::move(execution_inputs), run_options);

        if (VLOG_IS_ON(2)) {
          absl::Status host_callback_status =
              stream->DoHostCallback([executable_name, launch_id, device]() {
                VLOG(1) << "Finish device execution for " << executable_name
                        << ", launch_id: " << launch_id
                        << ", device: " << device->DebugString();
              });
          if (!host_callback_status.ok()) {
            LOG(WARNING)
                << "Failed to do host callback for finish device execution for "
                << executable_name << ", status = " << host_callback_status;
          }
        }

        VLOG(1) << "Finish calling RunAsync for " << executable_name
                << ", device=" << device->DebugString()
                << ", launch_id=" << launch_id << ", replica=" << replica
                << ", partition=" << partition
                << ", completed, ok=" << result_buffer_or_status.ok();

        if (!result_buffer_or_status.ok()) {
          LOG(ERROR) << "Calling RunAsync failed for executable "
                     << executable_name << " on device "
                     << device->DebugString()
                     << ", status = " << result_buffer_or_status.status();
          set_error(result_buffer_or_status.status());
          return;
        }

        ExecutionOutput& execution_output = result_buffer_or_status.value();
        ScopedShapedBuffer output = execution_output.ConsumeResult();
        if (untuple_result && result_is_tuple) {
          for (int i = 0; i < output_buffers.size(); ++i) {
            ScopedShapedBuffer tuple_buffer = output.TakeSubTree({i});
            stream_executor::DeviceMemoryBase* elem =
                tuple_buffer.buffers().mutable_element({});
            VLOG(3) << "untuple: output_buffers[" << i
                    << "].emplace: " << elem->opaque();
            output_buffers[i].emplace(stream_executor::OwningDeviceMemory(
                *elem, device->local_device_id().value(), device->allocator()));
            *elem = se::DeviceMemoryBase();
          }
        } else {
          CHECK_EQ(output_buffers.size(), 1);
          auto* elem = output.buffers().mutable_element({});
          VLOG(3) << "output_buffers[0].emplace: " << elem->opaque();
          output_buffers.front().emplace(stream_executor::OwningDeviceMemory(
              *elem, device->local_device_id().value(), device->allocator()));
          *elem = se::DeviceMemoryBase();
        }

        // Set the scheduled event to concrete to indicate that the scheduling
        // has completed, so that the next execute_fn can start.
        scheduled_event.SetStateConcrete();

        absl::Status status;
        {
          tsl::profiler::TraceMe traceme("BlockHostUntilDone");
          status = stream->BlockHostUntilDone();
        }
        if (!status.ok()) {
          LOG(ERROR) << "BlockHostUntilDone failed for executable "
                     << executable_name << " on device "
                     << device->DebugString() << ", status = " << status;
          complete_event.SetError(status);
        } else {
          complete_event.SetStateConcrete();
        }

        VLOG(1) << "execute_fn for " << executable_name
                << ", launch_id: " << launch_id << ", replica=" << replica
                << ", partition=" << partition
                << ", device: " << device->DebugString()
                << " is done with status " << status;
      };

  auto prepare_inputs =
      [replica, blocking_thread_pool = client_->blocking_thread_pool(),
       launch_id(options.launch_id), executable_name(name()), device,
       tracked_buffers(std::move(tracked_buffers)),
       buffer_is_donated(std::move(buffer_is_donated)),
       prepare_inputs_avs(CopyAsyncValues(prepare_input_deps)),
       complete_event(complete_event.CopyRef()),
       scheduled_event(scheduled_event.CopyRef()),
       output_buffers(std::move(output_buffers)),
       execute_fn(std::move(execute_fn)), input_deps(std::move(input_deps)),
       parameter_shapes(on_device_executable_parameter_shapes_[executable_idx]),
       parameter_is_tupled_arguments(parameter_is_tupled_arguments_),
       arguments_are_tupled(options.arguments_are_tupled),
       input_buffer_sizes_in_bytes(
           input_buffer_sizes_in_bytes_[executable_idx])]() mutable {
        tsl::profiler::TraceMeConsumer activity(
            [&] {
              return tsl::profiler::TraceMeEncode(
                  "prepare_inputs", {
                                        {"launch_id", launch_id},
                                        {"device_id", device->id()},
                                    });
            },
            tsl::profiler::ContextType::kPjRt, launch_id);

        auto set_error = [&](absl::Status status) {
          complete_event.SetError(status);
          scheduled_event.SetError(status);
          for (auto& output_buffer : output_buffers) {
            output_buffer.SetError(status);
          }
        };

        for (const auto& av : prepare_inputs_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            set_error(*error);
            return;
          }
        }

        VLOG(3) << "prepare_inputs for " << executable_name
                << ", launch_id: " << launch_id << ", replica: " << replica
                << ", device: " << device->DebugString();
        DCHECK_EQ(tracked_buffers.size(), buffer_is_donated.size());

        absl::Status status = CheckBufferCompatibilities(
            *input_buffer_sizes_in_bytes, tracked_buffers);
        if (!status.ok()) {
          set_error(status);
          return;
        }

        std::vector<ExecutionInput> inputs;
        if (parameter_is_tupled_arguments && !arguments_are_tupled) {
          inputs.emplace_back(
              ShapeTree<MaybeOwningDeviceMemory>(&parameter_shapes->front()));
          ExecutionInput& input = inputs.back();
          for (int i = 0; i < tracked_buffers.size(); ++i) {
            VLOG(4) << "tupled input[" << i
                    << "]: " << tracked_buffers[i]->buffer()->buffer().opaque();
            if (buffer_is_donated[i]) {
              input.SetUnownedBuffer(
                  {i}, MaybeOwningDeviceMemory(se::OwningDeviceMemory(
                           tracked_buffers[i]->buffer()->buffer(),
                           device->local_hardware_id().value(),
                           device->allocator())));
            } else {
              input.SetBuffer({i}, MaybeOwningDeviceMemory(
                                       tracked_buffers[i]->buffer()->buffer()));
            }
          }
        } else {
          inputs.reserve(tracked_buffers.size());
          for (int i = 0; i < tracked_buffers.size(); ++i) {
            VLOG(4) << "untupled input[" << i
                    << "]: " << tracked_buffers[i]->buffer()->buffer().opaque();
            inputs.emplace_back(
                ShapeTree<MaybeOwningDeviceMemory>(&(*parameter_shapes)[i]));
            ExecutionInput& input = inputs.back();
            if (buffer_is_donated[i]) {
              input.SetUnownedBuffer(
                  {}, MaybeOwningDeviceMemory(se::OwningDeviceMemory(
                          tracked_buffers[i]->buffer()->buffer(),
                          device->local_hardware_id().value(),
                          device->allocator())));
            } else {
              input.SetBuffer({}, MaybeOwningDeviceMemory(
                                      tracked_buffers[i]->buffer()->buffer()));
            }
          }
        }

        EnqueueWorkWhenReady(blocking_thread_pool, input_deps,
                             [execute_fn(std::move(execute_fn)),
                              inputs(std::move(inputs))]() mutable {
                               execute_fn(std::move(inputs));
                             });
      };
  EnqueueWorkWhenReady(client_->non_blocking_thread_pool(), prepare_input_deps,
                       std::move(prepare_inputs));

  // Create output TFRT buffers.
  std::optional<PjRtFuture<>> future;
  if (fill_future) {
    PjRtFuture<>::Promise complete_promise =
        CreatePromiseForEvent(complete_event);
    future = PjRtFuture<>(complete_promise);
  }
  return Result({/*future=*/std::move(future),
                 /*buffers=*/std::move(outputs)});
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
TfrtGpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::Execute",
                                          tsl::profiler::ContextType::kPjRt,
                                          options.launch_id);
  if (device_assignment_ == nullptr) {
    return InvalidArgument("Execute expects a non-null device_assignment");
  }
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

  std::vector<std::vector<std::unique_ptr<PjRtBuffer>>> wrapped_results(
      num_addressable_devices);
  if (returned_futures.has_value()) {
    returned_futures->resize(num_addressable_devices);
  }
  if (num_addressable_devices == 1 && !ThisThreadIsInsideHostCallback()) {
    // Fast-path if there is only one device — run the computation on the
    // current thread.
    const int replica = addressable_device_logical_ids_[0].replica;
    const int partition = addressable_device_logical_ids_[0].partition;

    // TODO(b/382117736): Dump HLO snapshot.
    // Dump once before running, in case there's a crash.
    // MaybeDumpHloSnapshot(gpu_executable_->module(), options.launch_id,
    //                      argument_handles[0], {});

    auto statusor = ExecuteHelper(argument_handles[0], replica, partition,
                                  options, returned_futures.has_value());

    if (!statusor.ok()) {
      return std::move(statusor).status();
    }

    wrapped_results[0] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      (*returned_futures)[0] = std::move(*statusor->future);
    }

    // TODO(b/382117736): Dump HLO snapshot.
    // MaybeDumpHloSnapshot(cpu_executable_->module(), options.launch_id,
    //                      argument_handles[0], wrapped_results[0]);
  } else {
    absl::Mutex mu;
    int running = num_addressable_devices;
    int failed = 0;
    absl::Status first_failure_status;

    for (int i = 0; i < num_addressable_devices; ++i) {
      const int replica = addressable_device_logical_ids_[i].replica;
      const int partition = addressable_device_logical_ids_[i].partition;
      const int device_id = (*device_assignment_)(replica, partition);
      TF_ASSIGN_OR_RETURN(PjRtDevice * pjrt_device,
                          client_->LookupDevice(PjRtGlobalDeviceId(device_id)));
      TfrtGpuDevice* gpu_device =
          tensorflow::down_cast<TfrtGpuDevice*>(pjrt_device);

      VLOG(1) << "Try to run ExecuteHelper for " << name() << " on device "
              << gpu_device->DebugString()
              << ", launch_id: " << options.launch_id;

      // Gang schedule collectives to ensure that collectives with the same
      // launch_id are run at the same time. We conservatively run only one
      // collective at a time, because we may not have enough threads to run
      // arbitrary number of collectives concurrently.
      EnqueueWork(
          client_->non_blocking_thread_pool(),
          [this, replica, partition, i, &argument_handles, &options,
           &returned_futures, &wrapped_results, &mu, &running, &failed,
           &first_failure_status] {
            auto statusor =
                ExecuteHelper(argument_handles[i], replica, partition, options,
                              returned_futures.has_value());
            if (statusor.ok()) {
              wrapped_results[i] = std::move(statusor->buffers);
              if (returned_futures.has_value()) {
                (*returned_futures)[i] = std::move(*statusor->future);
              }
            }

            absl::MutexLock lock(&mu);
            --running;
            if (!statusor.ok()) {
              if (failed == 0) {
                first_failure_status = AppendStatus(
                    std::move(statusor).status(),
                    absl::StrFormat(
                        "while running replica %d and partition %d of a "
                        "replicated computation (other "
                        "replicas may have failed as well).",
                        replica, partition));
              }
              ++failed;
            }
          });
    }

    {
      auto done_running = [&]() {
        mu.AssertHeld();
        return running == 0;
      };
      absl::MutexLock lock(&mu);
      mu.Await(absl::Condition(&done_running));
    }

    if (!first_failure_status.ok()) return first_failure_status;
  }
  VLOG(1) << "Replicated execution complete.";

  return wrapped_results;
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtGpuExecutable::ExecuteSharded(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::ExecuteSharded",
                                          tsl::profiler::ContextType::kPjRt,
                                          options.launch_id);
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
                        addressable_device_logical_ids_[i].partition, options,
                        fill_future));
      returned_future = std::move(result.future);
      return std::move(result.buffers);
    }
  }
  return InvalidArgument(
      "ExecuteShard attempted to execute on device id %d which is not "
      "addressable by this client",
      device->global_device_id().value());
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
TfrtGpuExecutable::ExecutePortable(
    absl::Span<PjRtBuffer* const> argument_handles, PjRtDevice* device,
    const ExecuteOptions& options, std::optional<PjRtFuture<>>& returned_future,
    bool fill_future) {
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::ExecutePortable",
                                          tsl::profiler::ContextType::kPjRt,
                                          options.launch_id);
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
  TF_ASSIGN_OR_RETURN(auto result,
                      ExecuteHelper(argument_handles,
                                    /*replica=*/0,
                                    /*partition=*/0, options, fill_future,
                                    tsl::down_cast<TfrtGpuDevice*>(device)));
  returned_future = std::move(result.future);
  return std::move(result.buffers);
}

absl::string_view TfrtGpuExecutable::name() const {
  Executable* executable = executables_[0]->executable();
  if (executable->has_module()) {
    return executable->module().name();
  } else {
    return "<unknown executable>";
  }
}

absl::StatusOr<std::vector<std::shared_ptr<HloModule>>>
TfrtGpuExecutable::GetHloModules() const {
  std::vector<std::shared_ptr<HloModule>> modules;
  modules.reserve(executables_.size());
  for (const auto& local_exec : executables_) {
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
TfrtGpuExecutable::GetOutputMemoryKinds() const {
  TF_ASSIGN_OR_RETURN(auto shapes, GetOutputShapes());
  if (addressable_devices().empty()) {
    return Unimplemented(
        "GetOutputMemoryKinds is not supported when there are no addressable "
        "devices in TfrtGpuExecutable.");
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

absl::Status TfrtGpuExecutable::SetUpDonation(bool tuple_inputs) {
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

absl::StatusOr<CompiledMemoryStats> TfrtGpuExecutable::GetCompiledMemoryStats()
    const {
  if (executables_.size() != 1) {
    return Unimplemented(
        "Retrieving CompiledMemoryStats is not supported for multiple "
        "executables.");
  }
  CompiledMemoryStats memory_stats = CompiledMemoryStats();
  memory_stats.generated_code_size_in_bytes = SizeOfGeneratedCodeInBytes();
  const BufferAssignmentProto* proto =
      executables_[0]->executable()->buffer_assignment_proto();
  if (proto != nullptr) {
    memory_stats.serialized_buffer_assignment = proto->SerializeAsString();
  }
  memory_stats.PopulateBufferStatsFromAllocations(
      executables_[0]->executable()->GetAllocations());
  return memory_stats;
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
