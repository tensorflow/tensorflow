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
#include "absl/types/span.h"
#include "unsupported/Eigen/CXX11/Tensor"
#include "mlir/IR/BuiltinOps.h"
#include "xla/client/executable_build_options.h"
#include "xla/client/local_client.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/compile_options.pb.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/gpu/tfrt/gpu_event.h"
#include "xla/pjrt/gpu/tfrt/host_memory_allocator.h"
#include "xla/pjrt/gpu/tfrt/stream_pool.h"
#include "xla/pjrt/gpu/tfrt/tracked_tfrt_gpu_device_buffer.h"
#include "xla/pjrt/host_callback.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/mlir_to_hlo.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_common.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
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
    const Shape& on_host_shape,
    absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events,
    TfrtGpuDevice* device, TfrtGpuClient* client,
    PjRtMemorySpace* memory_space) {
  if (on_host_shape.IsTuple()) {
    return Unimplemented(
        "tuple case not implemented for AllocateTfrtGpuDestinationBuffer");
  }
  TF_ASSIGN_OR_RETURN(
      Shape on_device_shape,
      GetDestinationDeviceShape(on_host_shape, device, client, memory_space));
  size_t byte_size = ShapeUtil::ByteSizeOf(on_device_shape);
  TF_ASSIGN_OR_RETURN(auto device_buffer, MaybeOwningGpuMemory::AllocateShared(
                                              device->allocator(), byte_size));
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(device_buffer));
  return std::make_unique<TfrtGpuBuffer>(
      on_device_shape,
      std::make_unique<TrackedTfrtGpuDeviceBuffer>(
          buffer_async_value_ref, std::move(definition_events)),
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
    VLOG(2) << "EnqueueWork: pool: " << pool;
    EnqueueWork(pool, std::move(callee));
  });
}

std::string MakeComputeCapabilityString(
    const stream_executor::DeviceDescription* desc) {
  stream_executor::GpuComputeCapability cc = desc->gpu_compute_capability();
  if (std::holds_alternative<stream_executor::CudaComputeCapability>(cc)) {
    auto nvcc = std::get<stream_executor::CudaComputeCapability>(cc);
    return absl::StrCat(nvcc.major, ".", nvcc.minor);
  } else if (std::holds_alternative<stream_executor::RocmComputeCapability>(
                 cc)) {
    auto rocmcc = std::get<stream_executor::RocmComputeCapability>(cc);
    return rocmcc.gfx_version();
  } else {
    return "unknown";
  }
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
    absl::Span<TrackedTfrtGpuDeviceBuffer* const> input_buffers) {
  if (input_buffers.size() != input_buffer_sizes_in_bytes.size()) {
    return InvalidArgument(
        "Execution supplied %lld buffers but compiled program expected %lld "
        "buffers",
        input_buffers.size(), input_buffer_sizes_in_bytes.size());
  }
  for (int i = 0; i < input_buffers.size(); ++i) {
    const auto& buffer = input_buffers[i];
    if (input_buffer_sizes_in_bytes[i] != buffer->buffer()->size()) {
      return InvalidArgument(
          "Executable expected parameter %d of size %lld but got buffer with"
          " incompatible size %lld ",
          i, input_buffer_sizes_in_bytes[i], buffer->buffer()->size());
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
    absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningGpuMemory>, 4>
        buffer_ptrs;
    absl::InlinedVector<absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4>, 4>
        definition_events;
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
      definition_events.push_back({copy_event.CopyRef()});
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
      absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningGpuMemory>, 4>
          buffer_ptrs,
      absl::InlinedVector<absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4>,
                          4>
          definition_events,
      absl::InlinedVector<Shape, 4> device_shapes, TfrtGpuDevice* device)
      : buffers_(std::move(buffers)),
        h2d_thread_(std::make_unique<WorkerThread>(
            tsl::Env::Default(),
            "TfrtGpuAsyncHostToDeviceTransferManager_h2d_thread")),
        buffer_ptrs_(std::move(buffer_ptrs)),
        buffer_sizes_(GetBufferSizes(buffers_)),
        definition_events_(std::move(definition_events)),
        device_shapes_(std::move(device_shapes)),
        remaining_buffer_count_(buffers_.size()),
        device_(device),
        client_(tsl::down_cast<TfrtGpuClient*>(device_->client())) {
    VLOG(2) << "TfrtGpuAsyncHostToDeviceTransferManager::"
               "TfrtGpuAsyncHostToDeviceTransferManager: this="
            << this << " buffers_.size()=" << buffers_.size();

    last_transfer_started_.resize(buffer_ptrs_.size(), false);
  }

  ~TfrtGpuAsyncHostToDeviceTransferManager() override {
    auto transfers_finished = [this]() {
      mu_.AssertHeld();
      return transfers_in_flight_ == 0;
    };
    {
      absl::MutexLock l(&mu_);
      // Make sure we don't leave dangling pointers in cleanup routines even
      // if the client lets the object go out of scope.
      mu_.Await(absl::Condition(&transfers_finished));
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
    VLOG(2) << "TfrtGpuAsyncHostToDeviceTransferManager::"
               "TransferLiteralToBuffer: this="
            << this << " buffer_index=" << buffer_index;
    auto* client = tsl::down_cast<TfrtGpuClient*>(device_->client());
    DCHECK(client);

    TransferManager* transfer_manager =
        client->xla_client()->backend().transfer_manager();

    tsl::AsyncValueRef<MaybeOwningGpuMemory> buffer;
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

      ++transfers_in_flight_;
    }

    // The host to device transfer is performed on a thread pool, mostly
    // because it includes linearization that may be slow.
    // TODO(misard) assess if it would be preferable to introduce a heuristic
    // to put the transfer into the calling thread for small literals.
    auto transfer_h2d = [this, buffer_index, transfer_manager,
                         literal = std::move(literal),
                         buffer = std::move(buffer),
                         on_done = std::move(on_done)]() mutable {
      tsl::profiler::TraceMe traceme(
          "TfrtGpuAsyncHostToDeviceTransferManager::TransferLiteralToBuffer::"
          "transfer_h2d");

      // Initiate linearization and transfer of the buffer on the stream.
      ShapedBuffer shaped_buffer =
          buffer->AsShapedBuffer(device_shapes_[buffer_index], device_);

      absl::StatusOr<BoundedStreamPool::Handle> stream =
          device_->stream_pool().Borrow();
      TF_CHECK_OK(stream.status());
      CHECK_NE(stream->get(), nullptr);

      GenericTransferManager::LiteralFromDeviceMetadata transfer_metadata;
      // We never call device functions from the `done` callback.
      transfer_metadata.callback_is_host_callback_safe = true;
      TransferManager::TransferMetadata* transfer_metadata_ptr =
          (dynamic_cast<GenericTransferManager*>(transfer_manager) != nullptr)
              ? &transfer_metadata
              : nullptr;

      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          stream->get(), literal, shaped_buffer, transfer_metadata_ptr));

      TF_CHECK_OK((*stream)->BlockHostUntilDone())
          << "Failed to block host until done";

      CleanUp(buffer_index, /*is_last_transfer=*/true, std::move(on_done));
    };
    // Enqueue the transfer to the h2d thread.
    h2d_thread_->Schedule(std::move(transfer_h2d));
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
    VLOG(2) << "TfrtGpuAsyncHostToDeviceTransferManager::"
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
      tsl::AsyncValueRef<MaybeOwningGpuMemory>& buffer_memory =
          buffer_ptrs_[buffer_index];
      CHECK_LE(offset, buffer_memory->size());
      CHECK_LE(transfer_size, buffer_memory->size() - offset);
      if (transfer_size < buffer_memory->size()) {
        sub_buffer =
            buffer_memory->buffer().GetByteSlice(offset, transfer_size);
      } else {
        sub_buffer = buffer_memory->buffer();
      }

      ++transfers_in_flight_;
    }

    auto copy_to_gpu =
        [transfer_size, staging_buffer = std::move(staging_buffer), data,
         sub_buffer = std::move(sub_buffer), buffer_index, is_last_transfer,
         on_done = std::move(on_done), this]() mutable {
          if (transfer_size != 0) {
            if (staging_buffer != nullptr) {
              std::memcpy(staging_buffer.get(), data, transfer_size);
            }

            absl::StatusOr<BoundedStreamPool::Handle> stream =
                device_->stream_pool().Borrow();
            TF_CHECK_OK(stream.status())
                << "Failed to borrow a stream from the pool";
            CHECK_NE(stream->get(), nullptr);

            TF_CHECK_OK((*stream)->Memcpy(
                &sub_buffer, staging_buffer ? staging_buffer.get() : data,
                transfer_size))
                << "Failed to copy data to GPU";
            auto status = (*stream)->BlockHostUntilDone();
            TF_CHECK_OK(status) << "Failed to block host until done";
          }
          CleanUp(buffer_index, is_last_transfer, std::move(on_done));
        };
    // Enqueue the transfer to the h2d thread.
    h2d_thread_->Schedule(std::move(copy_to_gpu));
    return absl::OkStatus();
  }

  void SetBufferError(int buffer_index, absl::Status error) override {
    {
      absl::MutexLock l(&mu_);
      // For a given buffer_index, SetBufferError can't be called twice, or
      // called after the last transfer has been enqueued.
      CHECK(!definition_events_[buffer_index].back().IsConcrete());
      definition_events_[buffer_index].back().SetError(error);
    }
    VLOG(1) << "SetBufferError sets the " << buffer_index
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

  void CleanUp(int buffer_index, bool is_last_transfer,
               absl::AnyInvocable<void() &&> on_done) {
    {
      absl::MutexLock l(&mu_);

      CHECK_GT(transfers_in_flight_, 0);
      --transfers_in_flight_;
      VLOG(2) << "CleanUp for buffer_index=" << buffer_index
              << " is_last_transfer=" << is_last_transfer
              << " remaining_buffer_count_=" << remaining_buffer_count_
              << "; this: " << this;

      if (is_last_transfer) {
        // Drop our reference to the TrackedDeviceBuffer for this buffer.
        CHECK(buffer_ptrs_[buffer_index]);
        buffer_ptrs_[buffer_index] = nullptr;
        CHECK_GT(remaining_buffer_count_, 0);
        --remaining_buffer_count_;
        definition_events_[buffer_index].back().SetStateConcrete();
        if (remaining_buffer_count_ == 0) {
          VLOG(2) << "TransferLiteralToBuffer for all buffers is done. this: "
                  << this;
        }
      }
    }

    // Call on_done after finishing all housekeeping and releasing the lock.
    EnqueueWork(client_->non_blocking_thread_pool(), std::move(on_done));
  }

  absl::Mutex mu_;
  // The newly created buffers, which will be returned to the caller via
  // Retrieve.
  absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers_
      ABSL_GUARDED_BY(mu_);

  // Just a single thread, to ensure transfers are ordered. Its lifetime is
  // managed by H2DTransferManager. We assume `h2d_thread` is destructed before
  // `client_`, so `on_done` callbacks on `h2d_thread` will be handled by
  // threads managed by `client_`.
  std::unique_ptr<WorkerThread> h2d_thread_;

  absl::InlinedVector<tsl::AsyncValueRef<MaybeOwningGpuMemory>, 4> buffer_ptrs_
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
  absl::InlinedVector<absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4>, 4>
      definition_events_ ABSL_GUARDED_BY(mu_);
  // Device shapes for all buffers with either compact or custom layout.
  const absl::InlinedVector<Shape, 4> device_shapes_;
  // Count of buffers that have not yet been fully transferred.
  size_t remaining_buffer_count_ ABSL_GUARDED_BY(mu_);
  // Count of transfers that have been started but have not yet called
  // cleanup. Used to block in the destructor to avoid dangling pointers in
  // cleanup.
  int transfers_in_flight_ ABSL_GUARDED_BY(mu_) = 0;

  TfrtGpuDevice* const device_;  // not owned.
  TfrtGpuClient* const client_;  // not owned.
};

std::optional<stream_executor::GpuTargetConfigProto> GetTargetConfigForDevices(
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& devices) {
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
  return xla::Compiler::TargetConfig(devices.front()->executor()).ToProto();
}

absl::flat_hash_map<std::string, PjRtDeviceAttribute> GetAttrsForDevices(
    const std::vector<std::unique_ptr<TfrtGpuDevice>>& devices) {
  absl::flat_hash_map<std::string, PjRtDeviceAttribute> attrs;
  auto target_config = GetTargetConfigForDevices(devices);
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
    VLOG(3) << "Recv from channel #" << channel_id
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
      stream_pool_(options.executor, options.stream_capacity),
      compute_stream_pool_(options.executor, options.max_inflight_computations),
      allocator_(std::move(options.allocator)),
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

  CHECK_OK(executor_->CreateStream().status()) << "Failed to create stream";

  se_allocator_ = std::make_unique<se::TfAllocatorAdapter>(
      allocator_.get(), executor_->GetPlatform());
}

void TfrtGpuDevice::SetClient(PjRtClient* client) {
  CHECK(client_ == nullptr);
  client_ = client;

  // We have to define debug_string_ and to_string_ here, because
  // platform_name() requires client_ to be set.
  std::string device_name =
      absl::StrCat(MakeAsciiTitlecase(client_->platform_name()), "Device");
  description_.SetDebugString(
      absl::StrCat(client_->platform_name(), ":", id()));
  description_.SetToString(absl::StrCat(device_name, "(id=", id(), ")"));
}

absl::Status TfrtGpuDevice::TransferToInfeed(const LiteralSlice& literal) {
  return Unimplemented("TransferToInfeed");
}

absl::Status TfrtGpuDevice::TransferFromOutfeed(
    MutableBorrowingLiteral literal) {
  return Unimplemented("TransferFromOutfeed");
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
      dynamic_cast<se::TfAllocatorAdapter*>(se_allocator_.get());
  if (!allocator_adapter) {
    return Unimplemented(
        "GetAllocatorStats() is only implemented with TfAllocatorAdapter"
        "allocator");
  }

  TF_ASSIGN_OR_RETURN(auto allocator, allocator_adapter->GetAllocator(
                                          local_device_id().value()));

  auto stats = allocator->GetStats();
  TF_RET_CHECK(stats.has_value());
  return stats.value();
}

void TfrtGpuDevice::SetLastCollectiveLaunchEvent(
    tsl::AsyncValueRef<GpuEvent> event) {
  absl::MutexLock lock(&mu_);
  VLOG(2) << "SetLastCollectiveLaunchEvent: IsAvailable: "
          << event.IsAvailable() << "; pointer: " << event.GetAsyncValue();
  last_collective_launch_event_ = std::move(event);
}

tsl::AsyncValueRef<GpuEvent> TfrtGpuDevice::GetLastCollectiveLaunchEvent() {
  absl::MutexLock lock(&mu_);
  VLOG(2) << "GetLastCollectiveLaunchEvent: IsAvailable: "
          << last_collective_launch_event_.IsAvailable()
          << "; pointer: " << last_collective_launch_event_.GetAsyncValue();
  return last_collective_launch_event_.CopyRef();
}

TfrtGpuClient::TfrtGpuClient(
    int process_index, xla::LocalClient* xla_client,
    std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
    bool should_stage_host_to_device_transfers,
    std::unique_ptr<tsl::Allocator> host_memory_allocator,
    std::shared_ptr<const GpuTopology> gpu_topology)
    : process_index_(process_index),
      xla_client_(CHECK_NOTNULL(xla_client)),
      should_stage_host_to_device_transfers_(
          should_stage_host_to_device_transfers),
      host_memory_allocator_(std::make_unique<HostMemoryAllocator>(
          std::move(host_memory_allocator))),
      owned_devices_(std::move(devices)),
      computation_placer_(std::make_unique<ComputationPlacer>()),
      compile_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_compile_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()))),
      blocking_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_blocking_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()))),
      non_blocking_thread_pool_(std::make_unique<tsl::thread::ThreadPool>(
          tsl::Env::Default(), "TfrtGpuClient_non_blocking_thread_pool",
          std::max<int>(DefaultThreadPoolSize(), xla_client->device_count()))),
      transpose_cache_(1024),
      platform_name_(xla::CudaName()),
      topology_(tsl::Fingerprint64(platform_name_), platform_name_,
                std::move(gpu_topology), GetAttrsForDevices(owned_devices_),
                GetTargetConfigForDevices(owned_devices_)) {
  for (const std::unique_ptr<TfrtGpuDevice>& device : owned_devices_) {
    devices_.push_back(device.get());
    CHECK(
        id_to_device_.emplace(device->global_device_id(), device.get()).second)
        << "Duplicate device id: " << device->id();

    device->SetClient(this);
    if (device->IsAddressable()) {
      int idx = device->local_hardware_id().value();
      if (idx >= addressable_devices_.size()) {
        addressable_devices_.resize(idx + 1);
      }
      CHECK(addressable_devices_[idx] == nullptr) << idx;
      addressable_devices_[idx] = device.get();
    }
  }
  for (int idx = 0; idx < addressable_devices_.size(); ++idx) {
    CHECK(addressable_devices_[idx] != nullptr) << idx;
  }

  for (auto* device : addressable_devices()) {
    // Use the device id to construct a globally unique memory space id. We do
    // not promise that memory space ids and device ids are the same.
    TfrtGpuDevice* gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
    // Initialize the default memory space.
    const int global_device_id = gpu_device->global_device_id().value();
    auto memory_space =
        std::make_unique<TfrtGpuDeviceMemorySpace>(global_device_id, device);
    gpu_device->AttachMemorySpace(memory_space.get(), /*is_default=*/true);
    memory_spaces_.push_back(memory_space.get());
    owned_memory_spaces_.push_back(std::move(memory_space));
  }
  const int basePinnedId = device_count();
  for (auto* device : addressable_devices()) {
    TfrtGpuDevice* gpu_device = tensorflow::down_cast<TfrtGpuDevice*>(device);
    const int global_device_id = gpu_device->global_device_id().value();
    auto pinned = std::make_unique<PinnedHostMemorySpace>(
        basePinnedId + global_device_id, device);
    gpu_device->AttachMemorySpace(pinned.get());
    memory_spaces_.push_back(pinned.get());
    owned_memory_spaces_.push_back(std::move(pinned));
  }
  // We don't promise anything about the order of memory spaces, but this
  // sorting is done for consistency with the device list that's sorted above.
  absl::c_sort(memory_spaces_,
               [](const PjRtMemorySpace* a, const PjRtMemorySpace* b) {
                 return a->id() < b->id();
               });

  LOG(INFO) << "TfrtGpuClient created.";
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
  Shape shape = ShapeUtil::MakeShape(element_type, dims);
  TF_ASSIGN_OR_RETURN(
      shape,
      xla_client_->backend().transfer_manager()->ChooseCompactLayoutForShape(
          shape));
  return shape.layout();
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
                         options);

  // TODO: b/382117736 - Record free gpu memory.
  // Ref:
  // https://github.com/openxla/xla/blob/b729ae319d85d5ec1ec11c488092c2d6683a63f2/xla/pjrt/gpu/se_gpu_pjrt_client.cc#L792-L809
}

absl::StatusOr<std::unique_ptr<PjRtExecutable>> TfrtGpuClient::CompileInternal(
    const XlaComputation& computation,
    const std::vector<const Shape*>& argument_layout_pointers,
    LayoutCanonicalizationCallback layout_canonicalization_callback,
    CompileOptions options) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::CompileInternal");
  VLOG(1) << "TfrtGpuClient::CompileInternal";
  if (key_value_store().has_value() &&
      !options.executable_build_options.key_value_store()) {
    options.executable_build_options.set_key_value_store(*key_value_store());
  }
  auto input_options = options;

  TF_RETURN_IF_ERROR(options.ApplyAllOptionOverrides());
  TF_RETURN_IF_ERROR(UpdateCompileOptions(&options));

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
  XlaComputation xla_computation;
  const ExecutableBuildOptions& exec_build_options =
      options.executable_build_options;
  TF_RETURN_IF_ERROR(MlirToXlaComputation(
      module, xla_computation,
      /*use_tuple_args=*/options.parameter_is_tupled_arguments,
      /*return_tuple=*/false, exec_build_options.use_shardy_partitioner()));

  // TODO: b/382117736 - Add support for LayoutModesToXlaShapes
  // Ref:
  // https://github.com/openxla/xla/blob/b729ae319d85d5ec1ec11c488092c2d6683a63f2/xla/pjrt/pjrt_stream_executor_client.cc#L3538-L3586
  return Compile(xla_computation, options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileAndLoad(const XlaComputation& computation,
                              CompileOptions options) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                      Compile(computation, options));
  return Load(std::move(executable), LoadOptions());
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
TfrtGpuClient::CompileAndLoad(mlir::ModuleOp module, CompileOptions options) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<PjRtExecutable> executable,
                      Compile(module, options));
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
  auto non_owning_buffer = MaybeOwningGpuMemory(device_memory);
  auto buffer_async_value_ref =
      tsl::MakeAvailableAsyncValueRef<MaybeOwningGpuMemory>(
          std::move(non_owning_buffer));
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      /*definition_event=*/tsl::MakeAvailableAsyncValueRef<GpuEvent>(),
      std::move(on_delete_callback));
  return std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tsl::down_cast<TfrtGpuDevice*>(device), memory_space);
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>>
TfrtGpuClient::CreateUninitializedBuffer(const Shape& shape,
                                         PjRtMemorySpace* memory_space) {
  tsl::profiler::TraceMe traceme("TfrtGpuClient::CreateUninitializedBuffer");
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "TfrtGpuClient::CreateUninitializedBuffer: shape: "
              << shape.DebugString()
              << " memory_space: " << memory_space->DebugString();
  }
  TransferManager* transfer_manager =
      xla_client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(Shape compact_shape,
                      transfer_manager->ChooseCompactLayoutForShape(shape));
  return AllocateTfrtGpuDestinationBuffer(
      compact_shape, /*definition_events=*/{},
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
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "TfrtGpuClient::CreateErrorBuffer: shape: " << shape.ToString()
              << " device: " << device->DebugString() << " error: " << error;
  }

  auto buffer_async_value_ref = tsl::MakeErrorAsyncValueRef(error);
  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(buffer_async_value_ref),
      /*definition_event=*/tsl::MakeErrorAsyncValueRef(std::move(error)));
  return std::make_unique<TfrtGpuBuffer>(
      shape, std::move(tracked_device_buffer), this,
      tsl::down_cast<TfrtGpuDevice*>(device), memory_space);
}

absl::Status TfrtGpuClient::UpdateCompileOptions(CompileOptions* options) {
  return UpdateCompileOptionsInternal(options, /*returned_extras=*/nullptr);
}

absl::StatusOr<TfrtGpuClient::ExecutableExtras>
TfrtGpuClient::UpdateCompileOptionsAndGetExecutableExtras(
    CompileOptions* options) {
  ExecutableExtras extras;
  TF_RETURN_IF_ERROR(UpdateCompileOptionsInternal(options, &extras));
  return extras;
}

absl::Status TfrtGpuClient::UpdateCompileOptionsInternal(
    CompileOptions* options, ExecutableExtras* returned_extras) {
  ExecutableBuildOptions& build_options = options->executable_build_options;
  const int original_device_ordinal = build_options.device_ordinal();
  if (!build_options.compile_thread_pool()) {
    build_options.set_compile_thread_pool(compile_thread_pool_.get());
  }
  if (!build_options.device_allocator()) {
    build_options.set_device_allocator(
        xla_client_->backend().memory_allocator());
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

  if (build_options.device_ordinal() < 0) {
    build_options.set_device_ordinal(0);
  }

  // We need the information of device assignment for
  //   1) XLA GPU shard autotuning, as the process count and the current process
  //   index are needed;
  //   2) getting executable extras, as the addressable devices are needed.
  const bool use_xla_gpu_shard_autotuning =
      build_options.has_debug_options() &&
      build_options.debug_options().xla_gpu_shard_autotuning();
  if (!use_xla_gpu_shard_autotuning && returned_extras == nullptr) {
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
      return InvalidArgument(
          "Device assignment (%s) does not have any local devices.",
          device_assignment->ToString());
    }

    if (original_device_ordinal < 0) {
      build_options.set_device_ordinal(
          addressable_devices.front()->local_hardware_id().value());
    }

    build_options.set_process_index(*this_process_index);
    build_options.set_process_count(all_process_indices.size());
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
  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "TfrtGpuClient::BufferFromHostBuffer: shape: "
              << device_shape.ToString()
              << " device: " << device->DebugString();
  }
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

  auto gpu_buffer = tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>();
  tsl::AsyncValueRef<GpuEvent> copy_event =
      tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  absl::InlinedVector<tsl::AsyncValueRef<GpuEvent>, 4> definition_events{
      copy_event.CopyRef(),
  };

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
    tsl::profiler::TraceMe traceme("H2D staging copy");

    HostMemoryAllocator::OwnedPtr staging_buffer =
        allocator->Allocate(transpose ? byte_size : packed_size);
    void* buffer = staging_buffer.get();
    const void* data_ptr = data;

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
    return staging_buffer;
  };

  auto copy_to_gpu = [device, packed_size, data,
                      copy_event(std::move(copy_event)),
                      gpu_buffer{gpu_buffer.CopyRef()}](
                         HostMemoryAllocator::OwnedPtr staging_buffer) {
    tsl::profiler::TraceMe traceme("H2D GPU copy");
    auto gpu_buffer_or =
        MaybeOwningGpuMemory::AllocateShared(device->allocator(), packed_size);
    if (gpu_buffer_or.ok()) {
      gpu_buffer.emplace(std::move(gpu_buffer_or.value()));
    } else {
      gpu_buffer.SetError(gpu_buffer_or.status());
      copy_event.SetError(gpu_buffer_or.status());
      return;
    }

    absl::StatusOr<BoundedStreamPool::Handle> handle_or =
        device->stream_pool().Borrow();
    if (!handle_or.ok()) {
      copy_event.SetError(handle_or.status());
      return;
    }
    BoundedStreamPool::Handle stream = std::move(handle_or.value());

    se::DeviceMemoryBase dest = gpu_buffer->buffer();
    const void* host_data_ptr;
    if (staging_buffer) {
      host_data_ptr = staging_buffer.get();
    } else {
      host_data_ptr = data;
    }
    absl::Status status = stream->Memcpy(&dest, host_data_ptr, packed_size);
    if (!status.ok()) {
      copy_event.SetError(status);
      return;
    }
    status = stream->BlockHostUntilDone();
    if (status.ok()) {
      copy_event.SetStateConcrete();
    } else {
      copy_event.SetError(status);
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

  auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
      std::move(gpu_buffer), std::move(definition_events));

  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtGpuBuffer>(
      device_shape, std::move(tracked_device_buffer), this, device,
      memory_space));
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
TfrtGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  VLOG(3) << "TfrtGpuClient::CreateBuffersForAsyncHostToDevice";
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
  if (VLOG_IS_ON(1)) {
    LOG(INFO) << "TfrtGpuClient::BufferFromHostLiteral: shape: "
              << literal.shape().DebugString()
              << " device: " << device->DebugString();
  }
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
      AllocateTfrtGpuDestinationBuffer(shape, std::move(definition_events),
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
  VLOG(2) << "BufferFromHostLiteral for device_buffer: " << device_buffer;
  EnqueueWork(non_blocking_thread_pool_.get(),
              [literal, av = avs[0], device_buffer, shape, this,
               device = tsl::down_cast<TfrtGpuDevice*>(device),
               usage_event = std::move(usage_event)]() mutable {
                tsl::profiler::TraceMe traceme("H2D Dispatch");
                TransferManager* transfer_manager =
                    xla_client()->backend().transfer_manager();

                absl::StatusOr<BoundedStreamPool::Handle> handle_or =
                    device->stream_pool().Borrow();
                CHECK_OK(handle_or.status())
                    << "Failed to borrow a stream from the pool";
                BoundedStreamPool::Handle stream = std::move(handle_or.value());

                const auto& buffer = device_buffer->buffer();
                CHECK_EQ(literal.size_bytes(), buffer->size());

                ShapedBuffer shaped_buffer =
                    buffer->AsShapedBuffer(shape, device);

                CHECK_NE(stream.get(), nullptr);
                TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
                    stream.get(), literal, shaped_buffer));

                auto status = stream->BlockHostUntilDone();
                CHECK_OK(status) << "Failed to block host until done";
                VLOG(2) << "BufferFromHostLiteral done for device_buffer: "
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

static absl::StatusOr<std::vector<std::unique_ptr<TfrtGpuDevice>>>
GetTfrtGpuDevices(LocalClient* xla_client,
                  const GpuAllocatorConfig& allocator_config) {
  std::vector<std::unique_ptr<TfrtGpuDevice>> devices;
  int i = 0;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    TF_ASSIGN_OR_RETURN(
        auto allocator,
        CreateBFCAllocator(executor, allocator_config.memory_fraction,
                           allocator_config.preallocate,
                           allocator_config.gpu_system_memory_size));

    TfrtGpuDevice::Options options;
    options.id = i;
    // TODO: b/382117736 - Support multi-host
    options.process_index = 0;
    options.slice_index = 0;
    options.local_device_id = PjRtLocalDeviceId(i);
    options.local_hardware_id = PjRtLocalHardwareId(i);
    options.executor = executor;
    options.allocator = std::move(allocator);
    options.stream_capacity = 4;
    options.max_inflight_computations = 32;
    const se::Platform* platform = executor->GetPlatform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(options.local_hardware_id.value()));
    options.platform_version = desc->name();
    options.device_vendor = desc->device_vendor();
    options.compute_capability = MakeComputeCapabilityString(desc.get());
    options.core_count = desc->core_count();

    auto device = std::make_unique<TfrtGpuDevice>(std::move(options));
    devices.push_back(std::move(device));
    ++i;
  }
  return std::move(devices);
}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetTfrtGpuClient(
    const GpuClientOptions& options) {
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
  TF_ASSIGN_OR_RETURN(std::vector<std::unique_ptr<TfrtGpuDevice>> devices,
                      GetTfrtGpuDevices(xla_client, options.allocator_config));

  GpuTopologyProto gpu_topology_proto;
  for (const auto& device : devices) {
    if (gpu_topology_proto.platform_version().empty()) {
      gpu_topology_proto.set_platform_version(
          std::string(device->device_kind()));
    }
    gpu_topology_proto.add_device_ids(device->id());
  }

  // TODO: b/382117736 - Support multi-host
  gpu_topology_proto.set_num_slices(1);
  gpu_topology_proto.set_num_hosts_per_slice(1);
  gpu_topology_proto.set_num_devices_per_host(devices.size());

  auto gpu_topology = std::shared_ptr<const GpuTopology>(
      GpuTopology::FromProto(gpu_topology_proto));

  return std::unique_ptr<PjRtClient>(std::make_unique<TfrtGpuClient>(
      /*process_index=*/0, xla_client, std::move(devices),
      options.should_stage_host_to_device_transfers,
      std::move(host_memory_allocator), gpu_topology));
}

TfrtGpuBuffer::TfrtGpuBuffer(
    Shape on_device_shape,
    std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer,
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

TrackedTfrtGpuDeviceBuffer* TfrtGpuBuffer::AcquireUsage(
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

  TF_ASSIGN_OR_RETURN(BoundedStreamPool::Handle stream,
                      device_->stream_pool_.Borrow());
  TF_RETURN_IF_ERROR(transfer_manager->ReadDynamicShapes(
      stream.get(), &shaped_buffer, &ret_shape));
  TF_RETURN_IF_ERROR(stream->BlockHostUntilDone());
  return ret_shape;
}

PjRtFuture<> TfrtGpuBuffer::GetReadyFuture() {
  tsl::AsyncValueRef<GpuEvent> definition_event;
  absl::MutexLock lock(&mu_);
  if (!tracked_device_buffer_) {
    return PjRtFuture<>(InvalidArgument(
        "GetReadyFuture() called on deleted or donated buffer"));
  }
  definition_event = tracked_device_buffer_->definition_event();
  DCHECK(definition_event);
  if (!definition_promise_) {
    definition_promise_ = PjRtFuture<>::CreatePromise();
    if (definition_event.IsAvailable()) {
      if (definition_event.IsError()) {
        return PjRtFuture<>(
            FailedPrecondition("Buffer Definition Event: %s",
                               definition_event.GetError().message()));
      }
      definition_promise_.Set(absl::OkStatus());
    } else {
      definition_event.AndThen(
          [definition_event,
           definition_promise = definition_promise_]() mutable {
            if (definition_event.IsError()) {
              VLOG(2) << "definition_event.GetError(): "
                      << definition_event.GetError();
              definition_promise.Set(definition_event.GetError());
            } else {
              definition_promise.Set(absl::OkStatus());
            }
          });
    }
  }

  return PjRtFuture<>(
      definition_promise_,
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme("TfrtGpuBuffer::Await");
        VLOG(1) << "TfrtGpuBuffer::Await";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id=*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme("TfrtGpuBuffer::Await",
                                               keys.traceme_context_id);
      });
}

bool TfrtGpuBuffer::IsOnCpu() const {
  return memory_space() != nullptr &&
         memory_space()->kind() == PinnedHostMemorySpace::kKind;
}

const tsl::AsyncValueRef<MaybeOwningGpuMemory>& TfrtGpuBuffer::GetBufferPtr()
    const {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_->buffer();
}

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtGpuBuffer::AcquireExternalReference() {
  class ScopedExternalReference : public PjRtBuffer::ExternalReference {
   public:
    explicit ScopedExternalReference(
        TfrtGpuBuffer* buffer, tsl::AsyncValueRef<MaybeOwningGpuMemory> data)
        : buffer_(buffer), data_(std::move(data)) {
      DCHECK(data_);
      data_ptr_ = data_->buffer().opaque();
    }

    ~ScopedExternalReference() override { buffer_->DropExternalReference(); }

   private:
    TfrtGpuBuffer* buffer_ = nullptr;
    // Keep a reference to the underlying data used. Note that it is still
    // users' responsibility to synchronize reads and writes to the data.
    tsl::AsyncValueRef<MaybeOwningGpuMemory> data_;
  };

  absl::MutexLock lock(&mu_);
  if (tracked_device_buffer_ == nullptr) {
    return InvalidArgument("Buffer has been deleted or donated.");
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
      std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer)
      : tracked_device_buffer_(std::move(tracked_device_buffer)) {
    data_ptr_ = tracked_device_buffer_->buffer()->buffer().opaque();
  }

  ~TrackedGpuDeviceBufferExternalReference() override = default;

 private:
  std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer_;
};

absl::StatusOr<std::unique_ptr<PjRtBuffer::ExternalReference>>
TfrtGpuBuffer::ReleaseDeviceMemoryOwnership(
    bool wait_for_operations_to_complete) {
  if (on_device_shape_.IsTuple()) {
    return InvalidArgument(
        "ReleaseDeviceMemoryOwnership allowed only for non-tuple");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<TrackedTfrtGpuDeviceBuffer> tracked_device_buffer,
      Release(wait_for_operations_to_complete));

  std::unique_ptr<PjRtBuffer::ExternalReference> ref;
  if (tracked_device_buffer) {
    ref = std::make_unique<TrackedGpuDeviceBufferExternalReference>(
        std::move(tracked_device_buffer));
  }
  return ref;
}

PjRtFuture<> TfrtGpuBuffer::ToLiteral(MutableLiteralBase* literal) {
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

  auto copy_to_host = [device(device_), device_buffer,
                       usage_event(std::move(usage_event)), literal, promise,
                       client = client_, on_device_shape{on_device_shape_},
                       unpack_subbyte_types]() mutable {
    tsl::profiler::TraceMe traceme("D2H copy");
    if (device_buffer->definition_event().IsError()) {
      usage_event.SetStateConcrete();
      VLOG(2) << "device_buffer->definition_event().GetError(): "
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
    if (should_unpack) {
      staging_buffer = client->host_memory_allocator()->Allocate(byte_size);
      buffer_ptr = staging_buffer.get();
    } else {
      buffer_ptr = literal->untyped_data();
    }

    {
      tsl::profiler::TraceMe traceme2("D2H GPU copy");
      MarkGpuEventReadyOnExit ready_on_exit(std::move(usage_event));

      auto stream_or = device->stream_pool().Borrow();
      if (!stream_or.ok()) {
        promise.Set(stream_or.status());
        return;
      }
      BoundedStreamPool::Handle stream = std::move(stream_or.value());

      CHECK_OK(stream->Memcpy(buffer_ptr, device_buffer->buffer()->buffer(),
                              byte_size))
          << "stream->Memcpy failed copying from GPU to host";

      absl::Status status = stream->BlockHostUntilDone();
      if (!status.ok()) {
        VLOG(2) << "stream->BlockHostUntilDone failed: " << status;
        promise.Set(status);
        return;
      }
    }
    tsl::profiler::TraceMe traceme3("D2H staging copy");
    if (should_unpack) {
      int64_t unpacked_size = ShapeUtil::ElementsIn(on_device_shape);
      primitive_util::UnpackIntN(
          on_device_shape.element_type(),
          absl::MakeConstSpan(static_cast<const char*>(buffer_ptr), byte_size),
          absl::MakeSpan(static_cast<char*>(literal->untyped_data()),
                         unpacked_size));
    }
    VLOG(2) << "D2H staging copy done";
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
        VLOG(1) << "TfrtGpuBuffer::ToLiteral";
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
        tsl::profiler::TraceMe traceme("D2H copy");
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
            tsl::profiler::TraceMe traceme2("D2H GPU copy");
            MarkGpuEventReadyOnExit ready_on_exit(std::move(usage_event));
            auto stream_or = device->stream_pool().Borrow();
            if (!stream_or.ok()) {
              promise.Set(stream_or.status());
              LOG(ERROR) << "Failed to borrow a stream from the pool";
              return;
            }
            BoundedStreamPool::Handle stream = std::move(stream_or.value());
            void* host_ptr =
                staging_buffer != nullptr ? staging_buffer.get() : dst.value();

            CHECK_OK(stream->Memcpy(host_ptr, *sub_buffer, transfer_size))
                << "stream->Memcpy failed copying from GPU to host";
            absl::Status status = stream->BlockHostUntilDone();
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
            tsl::profiler::TraceMe traceme3("D2H staging copy");
            std::memcpy(dst.value(), staging_buffer.get(), transfer_size);
            VLOG(4) << "D2H staging copy done";
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
        VLOG(1) << "TfrtGpuBuffer::CopyRawToHostFuture";
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
  std::unique_ptr<TrackedTfrtGpuDeviceBuffer> device_buffer;
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
        VLOG(3) << "device_buffer is being deleted: " << device_buffer.get();
        device_buffer.reset();
      });
}

bool TfrtGpuBuffer::IsDeleted() {
  absl::MutexLock lock(&mu_);
  return tracked_device_buffer_ == nullptr;
}

absl::StatusOr<std::unique_ptr<PjRtBuffer>> TfrtGpuBuffer::CopyToMemorySpace(
    PjRtMemorySpace* dst_memory_space) {
  // TODO: b/382117736 -  Support non-default memory spaces.
  tsl::profiler::TraceMe traceme("TfrtGpuBuffer::CopyToMemorySpace");
  PjRtDevice* dst_device = dst_memory_space->devices()[0];

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
  auto usage_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();
  TrackedTfrtGpuDeviceBuffer* src_device_buffer = AcquireUsage(usage_event);
  if (src_device_buffer == nullptr) {
    return InvalidArgument(
        "CopyToMemorySpace called on deleted or donated buffer");
  }

  TfrtGpuDevice* gpu_dst_device = tsl::down_cast<TfrtGpuDevice*>(dst_device);
  tsl::AsyncValueRef<MaybeOwningGpuMemory> src_buffer =
      src_device_buffer->buffer();
  auto dst_buffer = tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>();
  auto dst_definition_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();

  absl::AnyInvocable<void()> transfer_d2d =
      [src_buffer(src_buffer.CopyRef()), dst_buffer(dst_buffer.CopyRef()),
       dst_definition_event(dst_definition_event.CopyRef()),
       src_definition_event(src_device_buffer->definition_event().CopyRef()),
       dst_device(gpu_dst_device), usage_event(usage_event.CopyRef())]() {
        tsl::profiler::TraceMe traceme("D2D copy");
        if (const absl::Status* error =
                dst_definition_event.GetErrorIfPresent()) {
          dst_buffer.SetError(*error);
          dst_definition_event.SetError(*error);
          usage_event.SetStateConcrete();
          return;
        }

        if (const absl::Status* error =
                src_definition_event.GetErrorIfPresent()) {
          dst_buffer.SetError(*error);
          dst_definition_event.SetError(*error);
          usage_event.SetStateConcrete();
          return;
        }
        MarkGpuEventReadyOnExit ready_on_exit(std::move(usage_event));
        absl::StatusOr<MaybeOwningGpuMemory> allocated_dst_buffer =
            MaybeOwningGpuMemory::AllocateShared(dst_device->allocator(),
                                                 src_buffer->buffer().size());
        if (!allocated_dst_buffer.ok()) {
          dst_buffer.SetError(allocated_dst_buffer.status());
          dst_definition_event.SetError(allocated_dst_buffer.status());
          return;
        }
        dst_buffer.emplace(std::move(allocated_dst_buffer.value()));

        absl::StatusOr<BoundedStreamPool::Handle> stream =
            dst_device->stream_pool_.Borrow();
        if (!stream.ok()) {
          dst_definition_event.SetError(stream.status());
          return;
        }
        se::DeviceMemoryBase dst(dst_buffer->buffer());
        absl::Status status = stream->get()->Memcpy(
            &dst, src_buffer->buffer(), src_buffer->buffer().size());
        if (!status.ok()) {
          dst_definition_event.SetError(status);
          return;
        }
        status = stream->get()->BlockHostUntilDone();
        if (status.ok()) {
          dst_definition_event.SetStateConcrete();
        } else {
          dst_definition_event.SetError(status);
        }
      };

  EnqueueWorkWhenReady(client_->blocking_thread_pool(),
                       {src_device_buffer->definition_event().CopyRCRef()},
                       std::move(transfer_d2d));
  return std::unique_ptr<PjRtBuffer>(std::make_unique<TfrtGpuBuffer>(
      on_device_shape_,
      std::make_unique<TrackedTfrtGpuDeviceBuffer>(
          std::move(dst_buffer), std::move(dst_definition_event)),
      client(), tsl::down_cast<TfrtGpuDevice*>(dst_device), dst_memory_space));
}

void TfrtGpuBuffer::DropExternalReference() {
  absl::MutexLock lock(&mu_);
  CHECK_GT(external_reference_counter_, 0);
  --external_reference_counter_;
  if (external_reference_counter_ == 0) {
    external_references_dropped_event_.SetStateConcrete();
  }
}

absl::StatusOr<std::unique_ptr<TrackedTfrtGpuDeviceBuffer>>
TfrtGpuBuffer::Release(bool wait_for_operations_to_complete) {
  auto donation_event = GetDonationEvent();
  tsl::BlockUntilReady(donation_event);
  std::unique_ptr<TrackedTfrtGpuDeviceBuffer> device_buffer;
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

std::unique_ptr<TrackedTfrtGpuDeviceBuffer>
TfrtGpuBuffer::ReleaseBufferLocked() {
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
        tsl::Fingerprint128(executable->executable()->module().ToString()));
    executables_.emplace_back(std::move(executable));
  }
  fingerprint_ = absl::StrCat(fingerprint.low64, fingerprint.high64);

  int num_partitions;
  if (device_assignment_ == nullptr) {
    // This must go after `executables_` is initialized.
    VLOG(3) << "TfrtGpuExecutable portable single-core";
    num_partitions = 1;
    CHECK(addressable_devices_.empty());
  } else {
    // This must go after `executables_` is initialized.
    if (VLOG_IS_ON(3)) {
      VLOG(3) << "TfrtGpuExecutable device_assignment:\n"
              << device_assignment_->ToString();
    }
    CHECK_GE(addressable_devices_.size(), 1) << device_assignment_->ToString();

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
    const RunId& run_id, const ExecuteOptions& options,
    tsl::AsyncValueRef<GpuEvent> last_collective_launch_event, bool fill_future,
    TfrtGpuDevice* device) {
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::ExecuteHelper",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());

  std::shared_ptr<DeviceAssignment> device_assignment;
  if (device == nullptr) {
    CHECK(device_assignment_ != nullptr);
    const int device_id = (*device_assignment_)(replica, partition);
    VLOG(2) << "device_id: " << device_id;
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

  if (VLOG_IS_ON(2)) {
    LOG(INFO) << "ExecuteHelper " << name() << ": " << options.launch_id
              << "; replica: " << replica << "; partition: " << partition
              << "; mapped to device ordinal for execution: " << device->id();
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

  // `execute_event` indicates whether gpu computation is complete and whether
  // there was an error.
  auto execute_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();

  // `dispatch_event` indicates whether gpu computation is dispatched to the
  // stream and whether there was an error.
  auto dispatch_event = tsl::MakeConstructedAsyncValueRef<GpuEvent>();

  absl::InlinedVector<TfrtGpuBuffer::DonationTransaction, 4>
      donation_transactions;

  absl::InlinedVector<TrackedTfrtGpuDeviceBuffer*, 4> tracked_buffers;
  absl::InlinedVector<bool, 4> buffer_is_donated;
  tracked_buffers.reserve(argument_handles.size());
  buffer_is_donated.reserve(argument_handles.size());
  // To avoid clobbering inputs, we must ensure that
  //   `extra_deps` = inputs' definition events + donated inputs' usage events.
  // This also ensures that the returned `execute_event` dominates all inputs'
  // events, and thus output buffer only need to contain `execute_event` as
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
    auto tracked_buffer_or =
        [&]() -> absl::StatusOr<TrackedTfrtGpuDeviceBuffer*> {
      TrackedTfrtGpuDeviceBuffer* tracked_buffer = nullptr;
      if (must_donate) {
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
        tracked_buffer = tfrt_buffer->AcquireUsage(execute_event);
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
    } else {
      TrackedTfrtGpuDeviceBuffer* tracked_buffer = tracked_buffer_or.value();
      tracked_buffers.push_back(tracked_buffer);
      prepare_input_deps.push_back(tracked_buffer->buffer().CopyRCRef());

      // Definition events are never modified after buffer construction. If they
      // are available and have no error, they can be skipped in input deps.
      // In contrast, already known errors in the input are taken as deps so
      // that they can poison output buffers.
      const auto& definition_event = tracked_buffer->definition_event();
      if (!definition_event.IsAvailable() || definition_event.IsError()) {
        VLOG(2) << "definition_event is not available: AsyncValue pointer: "
                << definition_event.GetAsyncValue();
        input_deps.push_back(definition_event.CopyRCRef());
      }
    }
  }

  // Schedule only one collective at a time.
  bool is_a_collective_launch = !!last_collective_launch_event;
  if (is_a_collective_launch) {
    VLOG(2) << "last_collective_launch_event: AsyncValue pointer: "
            << last_collective_launch_event.GetAsyncValue()
            << "; IsAvailable: " << last_collective_launch_event.IsAvailable();
    input_deps.push_back(std::move(last_collective_launch_event));
    device->SetLastCollectiveLaunchEvent(dispatch_event);
  }

  std::vector<tsl::AsyncValueRef<MaybeOwningGpuMemory>> output_buffers;
  std::vector<std::unique_ptr<PjRtBuffer>> outputs;
  auto gpu_executable = executables_[executable_idx];
  Executable* executable = gpu_executable->executable();
  const Shape& result_shape = executable->result_shape();
  bool untuple_result = options.untuple_result;
  bool result_is_tuple = result_shape.IsTuple();
  if (options.untuple_result && result_shape.IsTuple()) {
    output_buffers.reserve(result_shape.tuple_shapes().size());
    outputs.reserve(output_buffers.size());
    for (int i = 0; i < result_shape.tuple_shapes().size(); ++i) {
      output_buffers.push_back(
          tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>());
      // Program execution writes to output buffers so it's a definition
      // event.
      auto leaf_tracked_device_buffer =
          std::make_unique<TrackedTfrtGpuDeviceBuffer>(
              output_buffers.back().CopyRef(), execute_event.CopyRef());
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
        tsl::MakeUnconstructedAsyncValueRef<MaybeOwningGpuMemory>());
    // Program execution writes to output buffers so it's a definition event.
    auto tracked_device_buffer = std::make_unique<TrackedTfrtGpuDeviceBuffer>(
        output_buffers.back().CopyRef(),
        /*definition_event=*/execute_event.CopyRef());
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

  // The choice of where we wait is arbitrary; the reason for the wait is
  // pacing to avoid problems such as memory fragmentation and running ahead
  // too far, not for correctness. Placing it before the executable launch
  // allows the inputs for the next executable to be fetched even if the
  // launch is delayed.
  VLOG(1) << "Going to get compute reservation for " << name() << ": "
          << options.launch_id << "; replica: " << replica;
  std::unique_ptr<Semaphore::ScopedReservation> compute_reservation;
  {
    tsl::profiler::TraceMe t("waiting for compute reservation");
    compute_reservation = std::make_unique<Semaphore::ScopedReservation>(
        device->max_inflight_computations_semaphore().ScopedAcquire(1));
  }
  VLOG(1) << "Got compute reservation for " << name() << ": "
          << options.launch_id << "; replica: " << replica;
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
      [replica, partition, device, launch_id(options.launch_id), run_id(run_id),
       output_buffers(output_buffers), execute_event(execute_event.CopyRef()),
       dispatch_event(dispatch_event.CopyRef()), untuple_result(untuple_result),
       result_is_tuple(result_is_tuple),
       compute_reservation(std::move(compute_reservation)),
       donation_transactions(std::move(donation_transactions)),
       parameter_shapes(on_device_executable_parameter_shapes_[executable_idx]),
       gpu_executable(std::move(gpu_executable)),
       device_assignment(device_assignment), executable_name(name()),
       ffi_context(ffi_context), inputs_avs(CopyAsyncValues(input_deps)),
       execution_profile(options.execution_profile),
       send_device_memory(std::move(send_device_memory)),
       recv_device_memory(std::move(recv_device_memory)),
       client = client_](std::vector<ExecutionInput> execution_inputs) mutable {
        VLOG(1) << "execute_fn for " << executable_name << ": " << launch_id
                << "; replica: " << replica;

        tsl::profiler::TraceMeProducer producer(
            [&] {
              return tsl::profiler::TraceMeEncode(
                  "execute_fn",
                  {{"launch_id", std::to_string(launch_id)},
                   {"device_ordinal", device->local_device_id().value()}});
            },
            tsl::profiler::ContextType::kTfExecutor, run_id.ToInt());

        auto set_error = [&](absl::Status status) {
          for (auto& output_buffer : output_buffers) {
            output_buffer.SetError(status);
          }
          execute_event.SetError(status);
          dispatch_event.SetError(status);
        };

        for (const auto& av : inputs_avs) {
          if (auto* error = av->GetErrorIfPresent()) {
            set_error(*error);
            return;
          }
        }

        absl::StatusOr<BoundedStreamPool::Handle> stream_or =
            device->compute_stream_pool().Borrow();
        if (!stream_or.ok()) {
          set_error(stream_or.status());
          return;
        }

        BoundedStreamPool::Handle stream = std::move(stream_or.value());
        ExecutableRunOptions run_options;
        run_options.set_stream(stream.get());
        run_options.set_host_to_device_stream(stream.get());
        run_options.set_device_to_host_stream(stream.get());
        run_options.set_allocator(device->se_allocator_.get());
        run_options.set_device_assignment(device_assignment.get());

        if (launch_id != 0) {
          run_options.set_run_id(RunId(launch_id));
        } else {
          run_options.set_run_id(run_id);
        }
        run_options.set_rng_seed(device->GetNewPrngSeed());
        run_options.set_gpu_executable_run_options(client->gpu_run_options());
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
        VLOG(2) << "launch id for " << executable_name << ": "
                << run_options.launch_id();

        // TODO(phawkins): *technically* this should probably happen after
        // calling RunAsync(). But that causes a large performance problem: it
        // prevents the main thread from freeing the buffer objects.
        for (auto& donation_transaction : donation_transactions) {
          VLOG(2) << "Committing donation transaction: "
                  << donation_transaction.device_buffer();
          std::move(donation_transaction).Commit();
        }

        if (VLOG_IS_ON(1)) {
          VLOG(1) << "Start calling RunAsync for executable " << executable_name
                  << " on device " << device->DebugString();
        }

        absl::StatusOr<ExecutionOutput> result_buffer_or_status =
            gpu_executable->RunAsync(std::move(execution_inputs), run_options);

        if (VLOG_IS_ON(1)) {
          VLOG(1) << "Finish calling RunAsync for executable "
                  << executable_name << " on device " << device->DebugString()
                  << ", replica " << replica << ", partition " << partition
                  << ", status=" << result_buffer_or_status.status();
        }

        if (!result_buffer_or_status.ok()) {
          LOG(ERROR) << "Calling RunAsync failed for executable "
                     << executable_name << " on device "
                     << device->DebugString()
                     << ", status = " << result_buffer_or_status.status();
          set_error(result_buffer_or_status.status());
          return;
        }

        // Set the dispatch event to concrete to indicate that the dispatch has
        // completed, so that the next execute_fn can start.
        dispatch_event.SetStateConcrete();

        ExecutionOutput& execution_output = result_buffer_or_status.value();
        ScopedShapedBuffer output = execution_output.ConsumeResult();
        if (untuple_result && result_is_tuple) {
          for (int i = 0; i < output_buffers.size(); ++i) {
            ScopedShapedBuffer tuple_buffer = output.TakeSubTree({i});
            auto* elem = tuple_buffer.buffers().mutable_element({});
            VLOG(4) << "untuple: output_buffers[" << i
                    << "].emplace: " << elem->opaque();
            output_buffers[i].emplace(device->allocator(), *elem);
            *elem = se::DeviceMemoryBase();
          }
        } else {
          CHECK_EQ(output_buffers.size(), 1);
          auto* elem = output.buffers().mutable_element({});
          VLOG(4) << "output_buffers[0].emplace: " << elem->opaque();
          output_buffers.front().emplace(device->allocator(), *elem);
          *elem = se::DeviceMemoryBase();
        }

        absl::Status status = stream->BlockHostUntilDone();
        if (!status.ok()) {
          execute_event.SetError(status);
          return;
        }
        execute_event.SetStateConcrete();
      };

  auto prepare_inputs =
      [blocking_thread_pool = client_->blocking_thread_pool(), run_id(run_id),
       device, tracked_buffers(std::move(tracked_buffers)),
       buffer_is_donated(std::move(buffer_is_donated)),
       prepare_inputs_avs(CopyAsyncValues(prepare_input_deps)),
       execute_event(execute_event.CopyRef()),
       dispatch_event(dispatch_event.CopyRef()),
       output_buffers(std::move(output_buffers)),
       execute_fn(std::move(execute_fn)), input_deps(std::move(input_deps)),
       parameter_shapes(on_device_executable_parameter_shapes_[executable_idx]),
       parameter_is_tupled_arguments(parameter_is_tupled_arguments_),
       arguments_are_tupled(options.arguments_are_tupled),
       input_buffer_sizes_in_bytes(
           input_buffer_sizes_in_bytes_[executable_idx])]() mutable {
        tsl::profiler::TraceMeConsumer activity(
            "prepare_inputs", tsl::profiler::ContextType::kPjRt,
            run_id.ToInt());


        auto set_error = [&](absl::Status status) {
          execute_event.SetError(status);
          dispatch_event.SetError(status);
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

        VLOG(2) << "prepare_inputs";
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
            if (buffer_is_donated[i]) {
              input.SetUnownedBuffer(
                  {i}, MaybeOwningDeviceMemory(se::OwningDeviceMemory(
                           tracked_buffers[i]->buffer()->buffer(),
                           device->local_hardware_id().value(),
                           device->se_allocator_.get())));
            } else {
              input.SetBuffer({i}, MaybeOwningDeviceMemory(
                                       tracked_buffers[i]->buffer()->buffer()));
            }
          }
        } else {
          inputs.reserve(tracked_buffers.size());
          for (int i = 0; i < tracked_buffers.size(); ++i) {
            inputs.emplace_back(
                ShapeTree<MaybeOwningDeviceMemory>(&(*parameter_shapes)[i]));
            ExecutionInput& input = inputs.back();
            if (buffer_is_donated[i]) {
              input.SetUnownedBuffer(
                  {}, MaybeOwningDeviceMemory(se::OwningDeviceMemory(
                          tracked_buffers[i]->buffer()->buffer(),
                          device->local_hardware_id().value(),
                          device->se_allocator_.get())));
            } else {
              input.SetBuffer({}, MaybeOwningDeviceMemory(
                                      tracked_buffers[i]->buffer()->buffer()));
            }
          }
        }

        VLOG(2) << "prepare_inputs done";
        VLOG(2) << "EnqueueWorkWhenReady; blocking_thread_pool: "
                << blocking_thread_pool;

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
    auto promise = PjRtFuture<>::CreatePromise();
    future = PjRtFuture<>(promise);
    execute_event.AndThen([promise = std::move(promise),
                           event = execute_event.CopyRef()]() mutable {
      absl::Status s;
      if (auto* error = event.GetErrorIfPresent()) {
        s = *error;
      }
      VLOG(1) << "Setting future: " << s;
      promise.Set(s);
    });
  }
  return Result({/*future=*/std::move(future),
                 /*buffers=*/std::move(outputs)});
}

absl::StatusOr<std::vector<std::vector<std::unique_ptr<PjRtBuffer>>>>
TfrtGpuExecutable::Execute(
    absl::Span<const std::vector<PjRtBuffer*>> argument_handles,
    const ExecuteOptions& options,
    std::optional<std::vector<PjRtFuture<>>>& returned_futures) {
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::Execute",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());
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
    // MaybeDumpHloSnapshot(gpu_executable_->module(), run_id,
    //                      argument_handles[0], {});

    auto statusor = ExecuteHelper(
        argument_handles[0], replica, partition, run_id, options,
        /*last_collective_launch_event=*/tsl::AsyncValueRef<GpuEvent>(),
        returned_futures.has_value());

    if (!statusor.ok()) {
      return std::move(statusor).status();
    }

    wrapped_results[0] = std::move(statusor->buffers);
    if (returned_futures.has_value()) {
      (*returned_futures)[0] = std::move(*statusor->future);
    }

    // TODO(b/382117736): Dump HLO snapshot.
    // MaybeDumpHloSnapshot(cpu_executable_->module(), run_id,
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

      VLOG(2) << "ExecuteHelper: " << i << " " << replica << " " << partition
              << " " << device_id << " "
              << gpu_device->local_hardware_id().value();

      // Gang schedule collectives to ensure that collectives with the same
      // RunId are run at the same time. We conservatively run only one
      // collective at a time, because we may not have enough threads to run
      // arbitrary number of collectives concurrently.
      EnqueueWork(
          client_->non_blocking_thread_pool(),
          [this, gpu_device, replica, partition, i, &argument_handles, &run_id,
           &options, &returned_futures, &wrapped_results, &mu, &running,
           &failed, &first_failure_status] {
            auto statusor = ExecuteHelper(
                argument_handles[i], replica, partition, run_id, options,
                gpu_device->GetLastCollectiveLaunchEvent(),
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
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::ExecuteSharded",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());
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
          ExecuteHelper(
              argument_handles, addressable_device_logical_ids_[i].replica,
              addressable_device_logical_ids_[i].partition, run_id, options,
              tsl::down_cast<TfrtGpuDevice*>(device)
                  ->GetLastCollectiveLaunchEvent(),
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
  RunId run_id(options.launch_id);
  tsl::profiler::TraceMeProducer activity("TfrtGpuExecutable::ExecutePortable",
                                          tsl::profiler::ContextType::kPjRt,
                                          run_id.ToInt());
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
  TF_ASSIGN_OR_RETURN(
      auto result,
      ExecuteHelper(
          argument_handles,
          /*replica=*/0,
          /*partition=*/0, run_id, options,
          /*last_collective_launch_event=*/tsl::AsyncValueRef<GpuEvent>(),
          fill_future, tsl::down_cast<TfrtGpuDevice*>(device)));
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
  const HloProto* proto = executables_[0]->executable()->hlo_proto();
  if (proto != nullptr) {
    memory_stats.serialized_hlo_proto = proto->SerializeAsString();
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
  VLOG(3) << "TfrtGpuBuffer::AcquireDonation: " << tracked_device_buffer_.get();
  return DonationTransaction(donation_event_,
                             std::move(tracked_device_buffer_));
}

}  // namespace xla
