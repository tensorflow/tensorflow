/* Copyright 2020 The OpenXLA Authors.

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

#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/base/thread_annotations.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/any_invocable.h"
#include "absl/functional/bind_front.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/memory/memory.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "mlir/IR/BuiltinOps.h"
#include "xla/backends/gpu/collectives/gpu_clique_key.h"
#include "xla/backends/gpu/collectives/gpu_cliques.h"
#include "xla/backends/gpu/collectives/gpu_collectives.h"
#include "xla/client/local_client.h"
#include "xla/core/collectives/clique_id.h"
#include "xla/core/collectives/collectives.h"
#include "xla/core/collectives/collectives_registry.h"
#include "xla/core/collectives/communicator.h"
#include "xla/core/collectives/rank_id.h"
#include "xla/executable_run_options.h"
#include "xla/hlo/builder/xla_computation.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/gpu/gpu_topology.h"
#include "xla/pjrt/gpu/gpu_topology.pb.h"
#include "xla/pjrt/gpu/se_gpu_topology_description.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h"
#include "xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "xla/pjrt/stream_executor_executable.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/pjrt/worker_thread.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/concurrency/async_value_ref.h"
#include "xla/tsl/concurrency/ref_count.h"
#include "xla/tsl/distributed_runtime/coordination/coordination_service_agent.h"
#include "xla/tsl/framework/allocator.h"
#include "xla/tsl/lib/strings/proto_serialization.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/protobuf/coordination_service.pb.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/protobuf.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/nvtx_utils.h"
#include "tsl/profiler/lib/traceme.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "xla/debug_options_flags.h"
#include "xla/pjrt/gpu/gpu_metrics.h"
#include "xla/pjrt/proto/compile_options.pb.h"
#include "xla/pjrt/stream_executor_executable.pb.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/gpu_executable.h"
#include "xla/service/gpu/gpu_memory_space_assignment.h"
#include "xla/service/gpu/stream_executor_util.h"
#include "xla/xla.pb.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/service/gpu/model/gpu_collective_performance_model.h"
#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/util.h"

namespace xla {
class GpuAsyncHostToDeviceTransferManager
    : public xla::PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  static absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  Create(absl::Span<const PjRtClient::ShapeSpec> shape_specs,
         std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
         PjRtStreamExecutorDevice* device, PjRtStreamExecutorClient* client,
         PjRtMemorySpace* memory_space) {
    if (device_layouts.has_value() &&
        device_layouts->size() != shape_specs.size()) {
      return InvalidArgument(
          "Number of layouts %d does not match the number of shapes %d",
          device_layouts->size(), shape_specs.size());
    }
    absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers;
    absl::InlinedVector<tsl::RCReference<RawSEDeviceMemory>, 4> buffer_ptrs;
    absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
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
      definition_events.push_back(
          std::make_shared<BufferSequencingEvent>(client->thread_pool()));
      Shape& device_shape = device_shapes.emplace_back(
          ShapeUtil::MakeShape(shape_spec.element_type, shape_spec.dims));
      if (device_layouts.has_value() && (*device_layouts)[i].has_value()) {
        *device_shape.mutable_layout() = *(*device_layouts)[i];
      } else {
        TF_ASSIGN_OR_RETURN(device_shape,
                            client->client()
                                ->backend()
                                .transfer_manager()
                                ->ChooseCompactLayoutForShape(device_shape));
      }
      LocalDeviceState* local_device = device->local_device_state();
      se::Stream* h2d_stream = local_device->host_to_device_stream();
      TF_ASSIGN_OR_RETURN(auto buffer,
                          AllocateDestinationBuffer(
                              device_shape, device, local_device, h2d_stream,
                              /*is_uninitialized_create=*/true, client,
                              definition_events.back(), memory_space));
      // Get a temporary hold just so we can fish out a shared_ptr to the
      // TrackedDeviceBuffer. It's ok to drop the hold before return the
      // buffers, because the invariants of this class ensure that the buffer
      // definition event will not fire until after all of this class' uses of
      // the TrackedDeviceBuffer have completed.
      auto* se_buffer =
          tensorflow::down_cast<PjRtStreamExecutorBuffer*>(buffer.get());
      DCHECK(se_buffer);
      auto hold = se_buffer->GetBufferWithUsageHold();
      buffer_ptrs.push_back(hold->device_memory());
      buffers.push_back(std::move(buffer));
    }

    return std::make_unique<GpuAsyncHostToDeviceTransferManager>(
        std::move(buffers), std::move(buffer_ptrs),
        std::move(definition_events), std::move(device_shapes), device);
  }

  GpuAsyncHostToDeviceTransferManager(
      absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers,
      absl::InlinedVector<tsl::RCReference<RawSEDeviceMemory>, 4> buffer_ptrs,
      absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
          definition_events,
      absl::InlinedVector<Shape, 4> device_shapes,
      PjRtStreamExecutorDevice* device)
      : buffers_(std::move(buffers)),
        buffer_ptrs_(std::move(buffer_ptrs)),
        definition_events_(std::move(definition_events)),
        device_shapes_(std::move(device_shapes)),
        remaining_buffer_count_(buffer_ptrs_.size()),
        transfers_in_flight_(0),
        device_(device) {
    buffer_sizes_.reserve(buffer_ptrs_.size());
    for (const auto& ptr : buffer_ptrs_) {
      DCHECK(ptr);
      buffer_sizes_.push_back(ptr->mem().size());
    }
    last_transfer_started_.resize(buffer_ptrs_.size(), false);
  }

  ~GpuAsyncHostToDeviceTransferManager() override {
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

  size_t buffer_count() const override { return buffers_.size(); };

  size_t buffer_size(int buffer_index) const override {
    DCHECK_LT(buffer_index, buffer_sizes_.size());
    return buffer_sizes_[buffer_index];
  }

  PjRtDevice* device() const override { return device_; }

  std::unique_ptr<PjRtBuffer> RetrieveBuffer(int buffer_index) override {
    DCHECK_LT(buffer_index, buffers_.size());
    return std::move(buffers_[buffer_index]);
  };

  absl::Status TransferLiteralToBuffer(
      int buffer_index, const LiteralSlice& literal,
      absl::AnyInvocable<void() &&> on_done) override {
    tsl::profiler::TraceMe traceme(
        "GpuAsyncHostToDeviceTransferManager::TransferLiteralToBuffer");
    auto* stream = device_->local_device_state()->host_to_device_stream();
    auto* se_client =
        tensorflow::down_cast<PjRtStreamExecutorClient*>(device_->client());
    DCHECK(se_client);

    TransferManager* transfer_manager =
        se_client->client()->backend().transfer_manager();

    tsl::RCReference<RawSEDeviceMemory> buffer;
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

    // The host to device transfer is performed on a thread pool, mostly because
    // it includes linearization that may be slow.
    // TODO(misard) assess if it would be preferable to introduce a heuristic to
    // put the transfer into the calling thread for small literals.
    auto transfer_h2d = [this, buffer_index, stream, transfer_manager, literal,
                         device = device_, device_buffer = buffer,
                         local_device =
                             std::move(device_->local_device_state()),
                         on_done = std::move(on_done)]() mutable {
      tsl::profiler::TraceMe traceme(
          "GpuAsyncHostToDeviceTransferManager::TransferLiteralToBuffer::"
          "transfer_"
          "h2d");

      auto event = local_device->event_pool().AllocateEvent(stream->parent());

      // Initiate linearization and transfer of the buffer on the stream.
      ShapedBuffer buffer =
          device_buffer->AsShapedBuffer(device, device_shapes_[buffer_index]);
      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          stream, literal, buffer));
      local_device->event_pool().ThenRecordEvent(stream, event.value());

      // Call cleanup once the transfer has finished on the stream.
      auto cleanup = [this, buffer_index, stream, on_done = std::move(on_done),
                      event = std::move(event).value()]() mutable {
        CleanUp(buffer_index, std::move(event), stream,
                /*is_last_transfer=*/true, std::move(on_done));
      };
      auto status = stream->DoHostCallback(std::move(cleanup));
      if (!status.ok()) {
        LOG(ERROR) << "DoHostCallback failed: " << status;
      }
    };
    se_client->thread_pool()->Schedule(
        ([ptr = new absl::AnyInvocable<void()>(std::move(transfer_h2d))]() {
          (*ptr)();
          delete ptr;
        }));
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
    auto* stream = device_->local_device_state()->host_to_device_stream();

    auto* client =
        tensorflow::down_cast<PjRtStreamExecutorClient*>(device_->client());
    bool should_stage_host_to_device_transfers =
        client->should_stage_host_to_device_transfers() &&
        (!client->IsDmaMapped(data, transfer_size));

    std::shared_ptr<void> staging_buffer;
    if (should_stage_host_to_device_transfers) {
      auto* host_memory_allocator = client->host_memory_allocator();
      if (host_memory_allocator == nullptr) {
        return InvalidArgument(
            "host_memory_allocator should be initialized for staging buffer "
            "transfer.");
      }

      void* ptr = host_memory_allocator->AllocateRaw(
          tsl::Allocator::kAllocatorAlignment, transfer_size);
      staging_buffer = std::shared_ptr<void>(
          ptr, [host_memory_allocator = host_memory_allocator](void* ptr) {
            host_memory_allocator->DeallocateRaw(ptr);
          });
    }

    absl::ReleasableMutexLock l(&mu_);
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
    auto& buffer_memory = buffer_ptrs_[buffer_index]->mem();
    se::DeviceMemoryBase sub_buffer;
    CHECK_LE(offset, buffer_memory.size());
    CHECK_LE(transfer_size, buffer_memory.size() - offset);
    if (transfer_size < buffer_memory.size()) {
      sub_buffer = buffer_memory.GetByteSlice(offset, transfer_size);
    } else {
      sub_buffer = buffer_memory;
    }

    ++transfers_in_flight_;
    // Release the lock before transfer in case transfer or cleanup could be
    // called on this thread, to avoid deadlock.
    l.Release();

    auto event = device_->local_device_state()->event_pool().AllocateEvent(
        stream->parent());

    if (transfer_size != 0) {
      if (staging_buffer != nullptr) {
        auto copy_to_staging_buffer = [data, transfer_size,
                                       staging_buffer]() mutable {
          std::memcpy(staging_buffer.get(), data, transfer_size);
        };
        if (auto status =
                stream->DoHostCallback(std::move(copy_to_staging_buffer));
            !status.ok()) {
          return status;
        }
        if (auto status = stream->Memcpy(&sub_buffer, staging_buffer.get(),
                                         transfer_size);
            !status.ok()) {
          return status;
        }
      } else if (auto status = stream->Memcpy(&sub_buffer, data, transfer_size);
                 !status.ok()) {
        return status;
      }
    }
    device_->local_device_state()->event_pool().ThenRecordEvent(stream,
                                                                event.value());

    auto cleanup = [this, buffer_index, event = std::move(event).value(),
                    stream, is_last_transfer, on_done = std::move(on_done),
                    staging_buffer = std::move(staging_buffer)]() mutable {
      CleanUp(buffer_index, std::move(event), stream, is_last_transfer,
              std::move(on_done));
    };
    return stream->DoHostCallback(std::move(cleanup));
  }

  void SetBufferError(int buffer_index, absl::Status error) override {
    {
      absl::MutexLock l(&mu_);
      // For a given buffer_index, SetBufferError can't be called twice, or
      // called after the last transfer has been enqueued.
      CHECK(!definition_events_[buffer_index]->IsDefined());
      definition_events_[buffer_index]->SetDefinedStatus(error);
    }
    VLOG(1) << "SetBufferError sets the " << buffer_index
            << "th buffer error: " << error;
  }

  void AddTransferMetadata(const TransferMetadata& meta) override {}

 private:
  absl::Mutex mu_;
  // The newly created buffers, which will be returned to the caller via
  // Retrieve.
  absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers_;
  // Cached versions of the sizes of all the buffers, so we can return them
  // without acquiring mu_.
  absl::InlinedVector<size_t, 4> buffer_sizes_;
  // References to the underlying storage for all the buffers, which ensures
  // that the buffers can't be freed before all transfers complete.
  absl::InlinedVector<tsl::RCReference<RawSEDeviceMemory>, 4> buffer_ptrs_
      ABSL_GUARDED_BY(mu_);
  // True if the last transfer for a buffer has been initiated. Used to prevent
  // a client initiating another transfer after the last transfer has already
  // been initiated.
  absl::InlinedVector<bool, 4> last_transfer_started_ ABSL_GUARDED_BY(mu_);
  // The buffer definition events on all the buffers, unblocked once the
  // corresponding buffer transfer has completed.
  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
      definition_events_ ABSL_GUARDED_BY(mu_);
  // Device shapes for all buffers with either compact or custom layout.
  const absl::InlinedVector<Shape, 4> device_shapes_;
  // Count of buffers that have not yet been fully transferred.
  size_t remaining_buffer_count_ ABSL_GUARDED_BY(mu_);
  // Count of transfers that have been started but have not yet called cleanup.
  // Used to block in the destructor to avoid dangling pointers in cleanup.
  int transfers_in_flight_ ABSL_GUARDED_BY(mu_);

  PjRtStreamExecutorDevice* device_;  // not owned.

  void CleanUp(int buffer_index, EventPool::Handle event, se::Stream* stream,
               bool is_last_transfer, absl::AnyInvocable<void() &&> on_done) {
    {
      absl::MutexLock l(&mu_);

      CHECK_GT(transfers_in_flight_, 0);
      --transfers_in_flight_;
      if (is_last_transfer) {
        // Drop our reference to the TrackedDeviceBuffer for this buffer.
        CHECK(buffer_ptrs_[buffer_index]);
        buffer_ptrs_[buffer_index] = tsl::RCReference<xla::RawSEDeviceMemory>();
        CHECK_GT(remaining_buffer_count_, 0);
        --remaining_buffer_count_;
        definition_events_[buffer_index]->SetSequencingEvent(std::move(event),
                                                             stream);
        if (remaining_buffer_count_ == 0) {
          VLOG(1) << "TransferLiteralToBuffer for all buffers is done.";
        }
      }
    }

    // Call on_done after finishing all housekeeping and releasing the lock.
    std::move(on_done)();
  }
};

static std::optional<stream_executor::GpuTargetConfigProto>
GetTargetConfigForDevices(absl::Span<PjRtDevice* const> devices) {
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
    LocalDeviceState* local_device_state =
        tensorflow::down_cast<const PjRtStreamExecutorDevice*>(device)
            ->local_device_state();
    if (local_device_state != nullptr) {
      return xla::Compiler::TargetConfig(local_device_state->executor())
          .ToProto();
    }
  }
  return std::nullopt;
}

static absl::flat_hash_map<std::string, PjRtDeviceAttribute> GetAttrsForDevices(
    absl::Span<PjRtDevice* const> devices) {
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

// Aborts all NCCL collectives when a task fails, as reported by the
// JobStateUpdate.
absl::Status AbortOnFailure(
    const tsl::CoordinationServiceAgent::JobStateUpdate& update) {
  if (update.previous_state.empty()) {
    // When a job first starts, there is no previous job state.
    return absl::OkStatus();
  }

  // We expect update.previous_state and update.current_state to have the same
  // size, and we expect for every i, previous_state[i] and current_state[i]
  // correspond to the same task.
  if (update.previous_state.size() != update.current_state.size()) {
    return FailedPrecondition(
        "Previous and current job states have different sizes: %d vs %d",
        update.previous_state.size(), update.current_state.size());
  }

  std::vector<uint64_t> failed_incarnations;
  for (int i = 0; i < update.previous_state.size(); ++i) {
    const tensorflow::CoordinatedTaskStateInfo& previous =
        update.previous_state[i];
    const tensorflow::CoordinatedTaskStateInfo& current =
        update.current_state[i];
    if (previous.task().task_id() != current.task().task_id()) {
      return FailedPrecondition(
          "Previous and current job states have mismatched task ids: %d vs %d",
          previous.task().task_id(), current.task().task_id());
    }
    if (previous.state() !=
        tensorflow::CoordinatedTaskState::TASKSTATE_CONNECTED) {
      // A task that was not previously connected cannot fail.
      continue;
    }
    if (current.state() !=
            tensorflow::CoordinatedTaskState::TASKSTATE_CONNECTED ||
        previous.incarnation() != current.incarnation()) {
      // The task is either failed, or restarted with a different incarnation.
      VLOG(1) << "Task " << previous.task().task_id() << " (incarnation "
              << previous.incarnation() << ") failed";
      failed_incarnations.push_back(previous.incarnation());
    }
  }

  if (!failed_incarnations.empty()) {
    return xla::gpu::AbortCliquesWithIncarnations(failed_incarnations);
  }
  return absl::OkStatus();
}

StreamExecutorGpuClient::StreamExecutorGpuClient(
    std::string platform_name, LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index, std::unique_ptr<se::DeviceMemoryAllocator> allocator,
    std::unique_ptr<tsl::Allocator> host_memory_allocator,
    bool should_stage_host_to_device_transfers,
    std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store,
    std::shared_ptr<DistributedRuntimeClient> distributed_client,
    bool abort_collectives_on_failure,
    std::shared_ptr<const GpuTopology> gpu_topology)
    : xla::PjRtStreamExecutorClient(
          platform_name, client, std::move(devices), process_index,
          /*memory_spaces=*/{},  // Initialized below.
          std::move(allocator), std::move(host_memory_allocator),
          should_stage_host_to_device_transfers, std::move(gpu_run_options)),
      topology_(xla::StreamExecutorGpuTopologyDescription(
          tsl::Fingerprint64(platform_name), platform_name,
          std::move(gpu_topology), GetAttrsForDevices(addressable_devices()),
          GetTargetConfigForDevices(addressable_devices()))),
      kv_store_(std::move(kv_store)),
      distributed_client_(std::move(distributed_client)) {
  const int basePinnedId = device_count();
  for (auto* device : addressable_devices()) {
    // Use the device id to construct a globally unique memory space id. We do
    // not promise that memory space ids and device ids are the same.
    const int id = device->id();
    auto memory_space =
        std::make_unique<StreamExecutorGpuHbmMemorySpace>(id, device);
    tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)->AttachMemorySpace(
        memory_space.get(), /*is_default=*/true);
    owned_memory_spaces_.push_back(std::move(memory_space));
    auto pinned =
        std::make_unique<PinnedHostMemorySpace>(basePinnedId + id, device);
    tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)->AttachMemorySpace(
        pinned.get());
    owned_memory_spaces_.push_back(std::move(pinned));
  }
  for (const std::unique_ptr<PjRtMemorySpace>& memory_space :
       owned_memory_spaces_) {
    memory_spaces_.push_back(memory_space.get());
  }

  // We don't promise anything about the order of memory spaces, but this
  // sorting is done for consistency with the device list that's sorted above.
  absl::c_sort(memory_spaces_,
               [](const PjRtMemorySpace* a, const PjRtMemorySpace* b) {
                 return a->id() < b->id();
               });

  // Add a JobStateCallback to abort collectives when tasks fail.
  if (abort_collectives_on_failure && distributed_client_) {
    absl::StatusOr<tsl::CoordinationServiceAgent*> agent =
        distributed_client_->GetCoordinationServiceAgent();
    if (agent.ok()) {
      (*agent)->AddJobStateCallback(
          [](const tsl::CoordinationServiceAgent::JobStateUpdate& update) {
            if (absl::Status s = AbortOnFailure(update); !s.ok()) {
              LOG(ERROR) << "Error aborting on failure: " << s;
            }
          });
    }
  }
}

absl::string_view StreamExecutorGpuClient::platform_version() const {
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

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
StreamExecutorGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const PjRtClient::ShapeSpec> shape_specs,
    std::optional<absl::Span<const std::optional<Layout>>> device_layouts,
    PjRtMemorySpace* memory_space) {
  CHECK_EQ(memory_space->devices().size(), 1);
  PjRtDevice* device = memory_space->devices()[0];
  auto* stream_executor_device =
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device);
  return xla::GpuAsyncHostToDeviceTransferManager::Create(
      shape_specs, std::move(device_layouts), stream_executor_device, this,
      memory_space);
}

absl::StatusOr<absl::flat_hash_map<GlobalDeviceId, uint64_t>>
StreamExecutorGpuClient::GetLatestIncarnations() {
  // Get the coordination service agent.
  if (!distributed_client_) {
    return FailedPrecondition("No distributed client");
  }
  TF_ASSIGN_OR_RETURN(tsl::CoordinationServiceAgent * agent,
                      distributed_client_->GetCoordinationServiceAgent());

  // Get the latest incarnation for every task.
  TF_ASSIGN_OR_RETURN(int num_tasks, topology_.ProcessCount());
  std::vector<int> tasks(num_tasks);
  std::iota(tasks.begin(), tasks.end(), 0);
  TF_ASSIGN_OR_RETURN(std::vector<uint64_t> task_incarnations,
                      agent->Incarnations(tasks));

  // Map every device to its incarnation.
  absl::flat_hash_map<GlobalDeviceId, uint64_t> device_incarnations;
  for (const PjRtDevice* device : devices()) {
    device_incarnations[GlobalDeviceId(device->global_device_id().value())] =
        task_incarnations[device->process_index()];
  }
  return device_incarnations;
}

gpu::GpuExecutableRunOptions* StreamExecutorGpuClient::gpu_run_options() {
  absl::StatusOr<absl::flat_hash_map<GlobalDeviceId, uint64_t>> incarnations =
      GetLatestIncarnations();
  if (!incarnations.ok()) {
    VLOG(1) << "Unable to set incarnations in GpuExecutableRunOptions: "
            << incarnations.status();
  } else {
    gpu_run_options_->set_incarnations(*std::move(incarnations));
  }
  return gpu_run_options_.get();
}

absl::StatusOr<xla::DeviceAssignment>
StreamExecutorGpuClient::GetDefaultDeviceAssignment(int num_replicas,
                                                    int num_partitions) const {
  if (num_partitions == 1 && num_replicas <= addressable_devices().size()) {
    xla::DeviceAssignment assignment(num_replicas, 1);
    for (int i = 0; i < num_replicas; ++i) {
      assignment(i, 0) = addressable_devices().at(i)->id();
    }
    return assignment;
  }
  // Fallback to default global device assignment if we can't run locally.
  return PjRtStreamExecutorClient::GetDefaultDeviceAssignment(num_replicas,
                                                              num_partitions);
}

PjRtFuture<> StreamExecutorGpuClient::CopyRawSubBufferToHost(
    PjRtBuffer* pjrt_buffer, PjRtFuture<void*> dst, int64_t offset,
    int64_t transfer_size) {
  auto* buffer = tensorflow::down_cast<PjRtStreamExecutorBuffer*>(pjrt_buffer);
  DCHECK(buffer);
  PjRtStreamExecutorDevice* device = buffer->device();
  LocalDeviceState* local_device = device->local_device_state();
  se::Stream* stream = local_device->GetDeviceToHostStream();

  // Acquire the usage hold inline so that the buffer is kept alive even if
  // `dst` is not immediately available.
  PjRtStreamExecutorBuffer::ScopedHold hold(buffer->GetBufferWithUsageHold());
  if (!hold.ok()) {
    return PjRtFuture<>(hold.status());
  }

  auto device_memory = hold->device_memory();
  if (!device_memory) {
    return PjRtFuture<>(
        InvalidArgument("Copy raw buffer called on an invalid buffer"));
  }

  auto promise = PjRtFuture<>::CreatePromise();
  auto usage_event =
      std::make_shared<BufferSequencingEvent>(this->thread_pool());

  auto definition_events = hold->definition_events();
  auto first_definition_event = definition_events[0];

  // When using the ComputeSynchronized allocation model, retain a reference to
  // the device_buffer until the copy completes, to ensure that the buffer isn't
  // deleted or donated while it is still in use. The choice of retaining a
  // reference at the host is a heuristic; the alternative is to ensure, before
  // freeing the buffer, that the compute stream is synchronized past the
  // transfer, but it seems better to hold onto the buffer too long than to
  // stall the compute stream.
  hold.ConvertUsageHold(stream, usage_event, /*reference_held=*/true);

  auto async_copy = [this, promise, offset, transfer_size, stream, local_device,
                     owning_device_memory = std::move(device_memory),
                     definition_events = std::move(definition_events),
                     usage_event = std::move(usage_event)](
                        absl::StatusOr<void*> dst) mutable {
    absl::StatusOr<EventPool::Handle> event =
        local_device->event_pool().AllocateEvent(stream->parent());
    if (!event.ok()) {
      promise.Set(event.status());
      return;
    }

    absl::Status defined_status = definition_events[0]->GetDefinedStatus();
    if (!defined_status.ok()) {
      promise.Set(defined_status);
      return;
    }

    auto& device_memory = owning_device_memory->mem();
    if (offset < 0 || offset > device_memory.size() ||
        device_memory.size() - offset < transfer_size) {
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

    WaitForBufferDefinitionEventsOnStream(absl::MakeSpan(definition_events),
                                          stream);

    if (transfer_size != 0) {
      if (should_stage_host_to_device_transfers() &&
          !IsDmaMapped(dst.value(), transfer_size)) {
        if (host_memory_allocator() == nullptr) {
          promise.Set(
              InvalidArgument("host_memory_allocator should be initialized for "
                              "staging buffer transfer."));
          return;
        }
        void* ptr = host_memory_allocator()->AllocateRaw(
            tsl::Allocator::kAllocatorAlignment, transfer_size);

        std::shared_ptr<void> staging_buffer = std::shared_ptr<void>(
            ptr, [host_memory_allocator = host_memory_allocator()](void* ptr) {
              host_memory_allocator->DeallocateRaw(ptr);
            });
        if (auto status = stream->Memcpy(staging_buffer.get(), *sub_buffer,
                                         transfer_size);
            !status.ok()) {
          promise.Set(std::move(status));
          return;
        }
        auto copy_to_staging_buffer = [dst, transfer_size,
                                       staging_buffer]() mutable {
          std::memcpy(*dst, staging_buffer.get(), transfer_size);
        };
        if (auto status = stream->DoHostCallback(copy_to_staging_buffer);
            !status.ok()) {
          promise.Set(std::move(status));
          return;
        }
      } else {
        // D2H request holds a non-owned pointer into sub_buffer base address
        // that needs to outlive the transfer until the stream callback is
        // invoked.
        auto status = stream->Memcpy(*dst, *sub_buffer, transfer_size);
        if (!status.ok()) {
          promise.Set(std::move(status));
          return;
        }
      }
    }

    local_device->event_pool().ThenRecordEvent(stream, event.value());
    usage_event->SetSequencingEvent(std::move(event).value(), stream);

    auto callback_status = local_device->ThenExecuteCallback(
        stream, [promise, owning_device_memory =
                              std::move(owning_device_memory)]() mutable {
          promise.Set();
        });
    if (!callback_status.ok()) {
      promise.Set(std::move(callback_status));
      return;
    }
  };

  first_definition_event->ExecuteOrAddToFutureTasks(
      absl::StrFormat("async_copy_raw_sub_buffer_to_host_%p", &async_copy),
      [this, dst, async_copy = std::move(async_copy)]() mutable {
        dst.OnReady([this, async_copy = std::move(async_copy)](
                        absl::StatusOr<void*> dst) {
          // Trampoline through a thread pool since GPUs do not allow calling
          // D2H inside the callback's context.
          thread_pool()->Schedule(absl::bind_front(async_copy, std::move(dst)));
        });
      });

  return PjRtFuture<>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "StreamExecutorGpuClient::CopyRawSubBufferToHost");
        VLOG(1) << "StreamExecutorGpuClient::CopyRawSubBufferToHost";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            "StreamExecutorGpuClient::CopyRawSubBufferToHost",
            keys.traceme_context_id);
      });
}

PjRtFuture<> StreamExecutorGpuClient::CopyRawHostToDevice(
    LocalDeviceState* local_device,
    tsl::RCReference<RawSEDeviceMemory> device_buffer, const void* src,
    int64_t offset, int64_t transfer_size) {
  auto promise = PjRtFuture<>::CreatePromise();
  se::Stream* stream = local_device->host_to_device_stream();
  thread_pool()->Schedule([local_device, stream,
                           buffer = std::move(device_buffer), src, offset,
                           transfer_size, promise]() mutable {
    se::DeviceMemoryBase sub_buffer = buffer->mem();
    if (transfer_size < sub_buffer.size()) {
      sub_buffer = sub_buffer.GetByteSlice(offset, transfer_size);
    }
    auto status = stream->Memcpy(&sub_buffer, src, transfer_size);
    if (!status.ok()) {
      promise.Set(std::move(status));
      return;
    }
    auto callback_status = local_device->ThenExecuteCallback(
        stream,
        [promise, buffer = std::move(buffer)]() mutable { promise.Set(); });
  });
  return PjRtFuture<>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "StreamExecutorGpuClient::CopyRawHostToDevice");
        VLOG(1) << "StreamExecutorGpuClient::CopyRawHostToDevice";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            "StreamExecutorGpuClient::CopyRawHostToDevice",
            keys.traceme_context_id);
      });
}

PjRtFuture<> StreamExecutorGpuClient::CopyRawDeviceToHost(
    LocalDeviceState* local_device,
    tsl::RCReference<RawSEDeviceMemory> device_buffer, void* dst,
    int64_t offset, int64_t transfer_size) {
  auto promise = PjRtFuture<>::CreatePromise();
  se::Stream* stream = local_device->GetDeviceToHostStream();
  thread_pool()->Schedule([local_device, stream,
                           buffer = std::move(device_buffer), dst, offset,
                           transfer_size, promise]() mutable {
    se::DeviceMemoryBase sub_buffer = buffer->mem();
    if (transfer_size < sub_buffer.size()) {
      sub_buffer = sub_buffer.GetByteSlice(offset, transfer_size);
    }
    auto status = stream->Memcpy(dst, sub_buffer, transfer_size);
    if (!status.ok()) {
      promise.Set(std::move(status));
      return;
    }
    auto callback_status = local_device->ThenExecuteCallback(
        stream,
        [promise, buffer = std::move(buffer)]() mutable { promise.Set(); });
  });
  return PjRtFuture<>(
      std::move(promise),
      /*on_block_start=*/
      []() {
        tsl::profiler::TraceMeProducer traceme(
            "StreamExecutorGpuClient::CopyRawDeviceToHost");
        VLOG(1) << "StreamExecutorGpuClient::CopyRawDeviceToHost";
        return PjRtFutureHelpers::ProfilingKeys(
            {/*traceme_context_id =*/traceme.GetContextId()});
      },
      /*on_block_end=*/
      [](PjRtFutureHelpers::ProfilingKeys keys) {
        tsl::profiler::TraceMeConsumer traceme(
            "StreamExecutorGpuClient::CopyRawDeviceToHost",
            keys.traceme_context_id);
      });
}

void StreamExecutorGpuClient::CopyToRemoteDevice(
    PjRtBuffer* buffer, absl::string_view serialized_descriptor,
    PjRtBuffer::RemoteSendCallback on_done) {
  // Get the default GpuCollectives instance.
  absl::StatusOr<Collectives*> collectives =
      CollectivesRegistry::Default("gpu");
  if (!collectives.ok()) {
    on_done(collectives.status(), /*sends_were_enqueued=*/false);
  }
  gpu::GpuCollectives* gpu_collectives =
      tsl::down_cast<gpu::GpuCollectives*>(*collectives);
  if (gpu_collectives == nullptr) {
    on_done(absl::InternalError("Failed to get GPU collectives"),
            /*sends_were_enqueued=*/false);
  }

  // Parse the CliqueId;
  CliqueId clique_id(serialized_descriptor);

  // Get the local device.
  absl::StatusOr<LocalDeviceState*> local_device =
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(buffer->device())
          ->GetLocalDeviceState();
  if (!local_device.ok()) {
    on_done(local_device.status(), /*sends_were_enqueued=*/false);
  }

  // Get the buffer's shape.
  absl::StatusOr<Shape> shape = buffer->HostShape();
  if (!shape.ok()) {
    on_done(shape.status(), /*sends_were_enqueued=*/false);
  }

  // Acquire a hold on the buffer.
  auto* handle = tensorflow::down_cast<PjRtStreamExecutorBuffer*>(buffer);
  PjRtStreamExecutorBuffer::ScopedHold hold = handle->GetBufferWithUsageHold();

  auto send = [gpu_collectives, clique_id, on_done, mem = hold->device_memory(),
               local_device = *local_device, shape = *shape,
               dtype = buffer->element_type(),
               stream = (*local_device)->GetDeviceToDeviceStream()]() mutable {
    auto f = [&]() -> absl::Status {
      // Create a communicator.
      //
      // TODO(mwhittaker): The way we are constructing GpuCliqueKeys is a big
      // hack. This code doesn't know the GlobalDeviceId of the sending process.
      // Instead, we use two arbitrary GlobalDeviceIds. This works because
      // NcclCommunicators don't actually use the GlobalDeviceIds.  Instead,
      // they just need to the know the number of devices (2 in this case).
      gpu::GpuCliqueKey clique_key(
          /*devices=*/{GlobalDeviceId(0), GlobalDeviceId(1)},
          /*num_local_participants=*/1);
      CliqueIds clique_ids(clique_id);
      gpu::GpuCollectives::Device collectives_device(local_device->executor());
      std::vector<Collectives::DeviceRank> ranks = {
          Collectives::DeviceRank(&collectives_device, RankId(1))};
      gpu::GpuCollectives::Config config;
      TF_ASSIGN_OR_RETURN(
          std::vector<std::unique_ptr<Communicator>> communicators,
          gpu_collectives->CreateCommunicators(clique_key, clique_ids, ranks,
                                               config));
      CHECK_EQ(communicators.size(), 1);
      std::unique_ptr<Communicator> communicator = std::move(communicators[0]);

      // Send data to the receiver.
      tsl::AsyncValueRef<Communicator::Event> send_event = communicator->Send(
          mem->mem(), shape.element_type(), ShapeUtil::ElementsIn(shape),
          RankId(0), gpu::GpuCollectives::On(*stream));

      // Wait for the send to finish.
      tsl::BlockUntilReady(send_event);
      if (send_event.IsError()) {
        return send_event.GetError();
      }

      // Keep mem alive until the Send has finished executing. Note that
      // send_event is fulfilled when the send is enqueued, but not necessarily
      // executed.
      TF_RETURN_IF_ERROR(local_device->ThenRelease(stream, mem));

      return absl::OkStatus();
    };

    if (absl::Status s = f(); !s.ok()) {
      on_done(s, /*sends_were_enqueued=*/false);
    } else {
      on_done(absl::OkStatus(), /*sends_were_enqueued=*/true);
    }
  };
  thread_pool()->Schedule(send);
}

absl::StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
StreamExecutorGpuClient::MakeCrossHostReceiveBuffers(
    absl::Span<const Shape> shapes, PjRtDevice* device,
    PjRtCrossHostRecvNotifier notifier) {
  // Validate arguments.
  if (shapes.empty()) {
    return InvalidArgument(
        "shapes parameter empty in MakeCrossHostReceiveBuffers");
  }
  if (shapes.size() != 1) {
    // TODO(mwhittaker): Support more than one shape.
    return Unimplemented(
        "StreamExecutorGpuClient::MakeCrossHostReceiveBuffers currently only "
        "supports one shape, but got %d",
        shapes.size());
  }
  Shape shape = shapes[0];

  // Get the default GpuCollectives instance.
  TF_ASSIGN_OR_RETURN(Collectives * collectives,
                      CollectivesRegistry::Default("gpu"));
  gpu::GpuCollectives* gpu_collectives =
      tsl::down_cast<gpu::GpuCollectives*>(collectives);
  if (gpu_collectives == nullptr) {
    return absl::InternalError("Failed to get GPU collectives");
  }

  // Allocate an uninitialized buffer. The buffer will be populated with data
  // received from the sending process.
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)
                          ->GetLocalDeviceState());
  se::Stream* stream = local_device->GetDeviceToDeviceStream();
  std::shared_ptr<BufferSequencingEvent> definition_event =
      std::make_shared<BufferSequencingEvent>(this->thread_pool());
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<PjRtStreamExecutorBuffer> buffer,
      AllocateDestinationBuffer(shape, device, local_device,
                                /*copy_stream=*/stream,
                                /*is_uninitialized_create=*/true, this,
                                definition_event));

  // Acquire a hold on the buffer to access the underlying memory.
  PjRtStreamExecutorBuffer::ScopedHold hold = buffer->GetBufferWithUsageHold();

  auto recv = [gpu_collectives, notifier, local_device, definition_event,
               stream, mem = hold->device_memory(), shape = shapes[0],
               dtype = buffer->element_type()]() mutable {
    auto f = [&]() -> absl::Status {
      // Create a CliqueId.
      TF_ASSIGN_OR_RETURN(CliqueId clique_id,
                          gpu_collectives->CreateUniqueCliqueId());

      // Notify the caller with the CliqueId. They will send the id to the
      // sender.
      //
      // TODO(mwhittaker): Implement cancellation.
      notifier(PjRtCrossHostRecvState{
          /*descriptors=*/{
              PjRtCrossHostRecvDescriptors{{clique_id.ToString()}}},
          /*cancel_notifier=*/nullptr,
      });

      // Create a communicator.
      //
      // TODO(mwhittaker): The way we are constructing GpuCliqueKeys is a big
      // hack. This code doesn't know the GlobalDeviceId of the sending process.
      // Instead, we use two arbitrary GlobalDeviceIds. This works because
      // NcclCommunicators don't actually use the GlobalDeviceIds. Instead, they
      // just need to the know the number of devices (2 in this case).
      gpu::GpuCliqueKey clique_key(
          /*devices=*/{GlobalDeviceId(0), GlobalDeviceId(1)},
          /*num_local_participants=*/1);
      CliqueIds clique_ids(clique_id);
      gpu::GpuCollectives::Device collectives_device(local_device->executor());
      std::vector<Collectives::DeviceRank> ranks = {
          Collectives::DeviceRank(&collectives_device, RankId(0))};
      gpu::GpuCollectives::Config config;
      TF_ASSIGN_OR_RETURN(
          std::vector<std::unique_ptr<Communicator>> communicators,
          gpu_collectives->CreateCommunicators(clique_key, clique_ids, ranks,
                                               config));
      CHECK_EQ(communicators.size(), 1);
      std::unique_ptr<Communicator> communicator = std::move(communicators[0]);

      // Receive data from the sender.
      tsl::AsyncValueRef<Communicator::Event> recv_event = communicator->Recv(
          mem->mem(), shape.element_type(), ShapeUtil::ElementsIn(shape),
          RankId(1), gpu::GpuCollectives::On(*stream));

      // Wait for the receive to finish.
      tsl::BlockUntilReady(recv_event);
      if (recv_event.IsError()) {
        return recv_event.GetError();
      }

      // Keep mem alive until the Recv has finished executing. Note that
      // recv_event is fulfilled when the receive is enqueued, but not
      // necessarily executed.
      TF_RETURN_IF_ERROR(local_device->ThenRelease(stream, mem));

      // Set definition event.
      TF_ASSIGN_OR_RETURN(
          EventPool::Handle event,
          local_device->event_pool().ThenAllocateAndRecordEvent(stream));
      definition_event->SetSequencingEvent(std::move(event), stream);

      return absl::OkStatus();
    };

    if (absl::Status s = f(); !s.ok()) {
      definition_event->SetDefinedStatus(s);
    }
  };
  thread_pool()->Schedule(recv);

  std::vector<std::unique_ptr<PjRtBuffer>> buffers;
  buffers.push_back(std::move(buffer));
  return buffers;
}

absl::StatusOr<Layout> StreamExecutorGpuClient::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) {
  return topology_.GetDefaultLayout(element_type, dims);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::CompileAndLoad(mlir::ModuleOp module,
                                        CompileOptions options) {
  auto executable = PjRtStreamExecutorClient::CompileAndLoad(module, options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  for (const PjRtDevice* device : addressable_devices()) {
    LocalDeviceState* local_device_state =
        tensorflow::down_cast<const PjRtStreamExecutorDevice*>(device)
            ->local_device_state();
    int64_t free_memory, total_memory;
    if (local_device_state != nullptr) {
      se::StreamExecutor* executor = local_device_state->executor();
      int device_ordinal = executor->device_ordinal();
      if (executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
        gpu_metrics::RecordFreeGpuSystemMemory(device_ordinal, free_memory);
      } else {
        LOG(ERROR) << "Failed to query available memory for GPU "
                   << device_ordinal;
      }
    }
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return executable;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::CompileAndLoad(const XlaComputation& computation,
                                        CompileOptions options) {
  auto executable =
      PjRtStreamExecutorClient::CompileAndLoad(computation, options);

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  for (const PjRtDevice* device : addressable_devices()) {
    LocalDeviceState* local_device_state =
        tensorflow::down_cast<const PjRtStreamExecutorDevice*>(device)
            ->local_device_state();
    int64_t free_memory, total_memory;
    if (local_device_state != nullptr) {
      se::StreamExecutor* executor = local_device_state->executor();
      int device_ordinal = executor->device_ordinal();
      if (executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
        gpu_metrics::RecordFreeGpuSystemMemory(device_ordinal, free_memory);
      } else {
        LOG(ERROR) << "Failed to query available memory for GPU "
                   << device_ordinal;
      }
    }
  }
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  return executable;
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::LoadSerialized(absl::string_view serialized,
                                        std::optional<CompileOptions> options,
                                        const LoadOptions& load_options) {
  return PjRtStreamExecutorClient::LoadSerializedExecutable(serialized, options,
                                                            load_options);
}

namespace {

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

absl::StatusOr<std::unique_ptr<se::GpuCudaMallocAsyncAllocator>>
CreateCudaAsyncAllocator(const LocalDeviceState& device, double memory_fraction,
                         bool reserve_memory, bool create_new_pool,
                         bool sync_mode, bool compute_stats = true) {
  se::StreamExecutor* executor = device.executor();
  int device_ordinal = executor->device_ordinal();

  int64_t free_memory;
  int64_t total_memory;
  if (!executor->DeviceMemoryUsage(&free_memory, &total_memory)) {
    return Unavailable("Failed to query available memory from device %i",
                       device_ordinal);
  }
  // To allow full GPU memory to be visible to the Cuda Async allocator
  // if using unified memory.
  // When unified memory is enabled, allow GPU memory oversubscription by
  // setting memory_fraction > 1.
  size_t allocator_memory = total_memory * memory_fraction;
  if (reserve_memory) {
    LOG(INFO) << "XLA backend allocating " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CudaAsyncAllocator.";
  } else {
    LOG(INFO) << "XLA backend will use up to " << allocator_memory
              << " bytes on device " << device_ordinal
              << " for CudaAsyncAllocator.";
  }

  auto allocator = std::make_unique<se::GpuCudaMallocAsyncAllocator>(
      /*platform_device_id*/ tsl::PlatformDeviceId(device_ordinal),
      /*create_new_pool*/ create_new_pool,
      /*new_pool_size*/ allocator_memory,
      /*reserve_memory*/ reserve_memory,
      /*reserve_memory_size*/ reserve_memory ? allocator_memory : 0,
      /*sync_mode*/ sync_mode,
      /*compute_stats*/ compute_stats);

  allocator->SetStreamAndPreallocateMemory(
      device.compute_stream()->platform_specific_handle().stream);

  return allocator;
}

#else  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020
absl::StatusOr<std::unique_ptr<tsl::Allocator>> CreateCudaAsyncAllocator(
    const LocalDeviceState& device, double memory_fraction, bool reserve_memory,
    bool create_new_pool, bool sync_mode, bool compute_stats = true) {
  return FailedPrecondition("CUDA async allocator requires CUDA >= 11.2");
}

#endif  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

// Builds a LocalDeviceState for each GPU present.
absl::StatusOr<std::map<int, std::unique_ptr<LocalDeviceState>>>
BuildLocalDeviceStates(LocalClient* xla_client) {
  std::map<int, std::unique_ptr<LocalDeviceState>> addressable_devices;
  for (se::StreamExecutor* executor :
       xla_client->backend().stream_executors()) {
    addressable_devices.emplace(
        executor->device_ordinal(),
        std::make_unique<LocalDeviceState>(
            executor, xla_client, LocalDeviceState::kComputeSynchronized,
            /*max_inflight_computations=*/32,
            /*allow_event_reuse=*/true, /*use_callback_stream=*/true));
  }
  return std::move(addressable_devices);
}

// Constructs a GPU device memory allocator to use, according to the allocator
// configuration the client requested.
absl::StatusOr<std::unique_ptr<se::DeviceMemoryAllocator>>
GetStreamExecutorGpuDeviceAllocator(
    se::Platform* platform, const GpuAllocatorConfig& allocator_config,
    const std::map<int, std::unique_ptr<LocalDeviceState>>&
        addressable_devices) {
  std::vector<se::MultiDeviceAdapter::AllocatorInfo> allocators;
  switch (allocator_config.kind) {
    case GpuAllocatorConfig::Kind::kCudaAsync: {
      for (const auto& ordinal_and_device : addressable_devices) {
        TF_ASSIGN_OR_RETURN(
            auto async_allocator,
            CreateCudaAsyncAllocator(
                *(ordinal_and_device.second), allocator_config.memory_fraction,
                allocator_config.preallocate, false, false, true));
        allocators.emplace_back(std::move(async_allocator),
                                ordinal_and_device.second->compute_stream(),
                                /*memory_space=*/0);
      }
      break;
    }

    case GpuAllocatorConfig::Kind::kDefault:
    case GpuAllocatorConfig::Kind::kBFC: {
      LOG(INFO) << "Using BFC allocator.";
      for (const auto& ordinal_and_device : addressable_devices) {
        TF_ASSIGN_OR_RETURN(
            auto bfc_allocator,
            CreateBFCAllocator(ordinal_and_device.second->executor(),
                               allocator_config.memory_fraction,
                               allocator_config.preallocate,
                               allocator_config.gpu_system_memory_size));
        allocators.emplace_back(std::move(bfc_allocator),
                                ordinal_and_device.second->compute_stream(),
                                /*memory_space=*/0);
      }
      break;
    }

    case GpuAllocatorConfig::Kind::kPlatform:
      LOG(INFO) << "Using platform allocator.";
      if (allocator_config.collective_memory_size != 0) {
        LOG(WARNING)
            << "collective_memory_size is non-zero, but allocator kind is set "
               "to \"platform\". Collective memory will not be allocated.";
      }
      // Returning null will cause the client to use the default backend
      // allocator.
      return nullptr;
  }

  // Add any additional allocators for alternate memory spaces.
  for (const auto& ordinal_and_device : addressable_devices) {
    TF_ASSIGN_OR_RETURN(
        auto collective_bfc_allocator,
        CreateCollectiveBFCAllocator(
            ordinal_and_device.second->executor(),
            /*memory_fraction=*/1.0 - allocator_config.memory_fraction,
            allocator_config.collective_memory_size));
    allocators.emplace_back(std::move(collective_bfc_allocator),
                            ordinal_and_device.second->compute_stream(),
                            /*memory_space=*/1);
  }

  for (const auto& ordinal_and_device : addressable_devices) {
    TF_ASSIGN_OR_RETURN(
        auto host_allocator,
        GetGpuHostAllocator(ordinal_and_device.second->executor()));
    allocators.emplace_back(std::move(host_allocator),
                            ordinal_and_device.second->compute_stream(),
                            /*memory_space=*/
                            static_cast<int>(se::MemoryType::kHost));
  }

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020
  const auto& debug_options = xla::GetDebugOptionsFromFlags();
  if (debug_options.xla_gpu_temp_buffer_use_separate_color()) {
    // Add memory allocator to allocate memory buffers with persistent temp
    // memory space color.
    for (const auto& ordinal_and_device : addressable_devices) {
      TF_ASSIGN_OR_RETURN(
          auto async_allocator,
          CreateCudaAsyncAllocator(*(ordinal_and_device.second), 1.0, false,
                                   true, true, true));
      allocators.emplace_back(
          std::move(async_allocator),
          ordinal_and_device.second->compute_stream(),
          /*memory_space=*/gpu::kTempBufferMemorySpaceColor);
    }
  }
#endif
  return std::make_unique<se::MultiDeviceAdapter>(platform,
                                                  std::move(allocators));
}

// Name the devices and threads that launch work on them. Note: the launcher
// thread is only used if there are multiple devices driven by a single process.
void NameDeviceAndLauncherThread(const LocalTopologyProto& node,
                                 const DeviceProto& device_proto,
                                 WorkerThread* launcher_thread) {
  auto suffix = absl::StrFormat(":#global=%d,local=%d,process=%d,slice=%d#",
                                device_proto.global_device_id(),
                                device_proto.local_device_ordinal(),
                                node.node_id(), device_proto.slice_index());
  // Name the device.
  tsl::profiler::NameDevice(device_proto.local_device_ordinal(),
                            absl::StrCat("Xla", suffix));
  // Name the thread that launches work on this device. This is deferred
  // until after ExchangeTopologies has been called so the global device
  // id and slice index are known. These are not available when the thread
  // is created.
  launcher_thread->Schedule([name = absl::StrCat("XlaLauncher", suffix)] {
    tsl::profiler::NameCurrentThread(name);
  });
}

}  // namespace

absl::StatusOr<DeviceTopologyPair> BuildDistributedDevices(
    absl::string_view platform_name,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id, int num_nodes,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store, bool enable_mock_nccl,
    std::optional<absl::string_view> mock_gpu_topology,
    std::optional<int> slice_index, absl::Duration get_local_topology_timeout,
    absl::Duration get_global_topology_timeout) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  LocalTopologyProto local_topology;
  local_topology.set_node_id(node_id);
  std::string boot_id_str;
  auto boot_id_str_or_status = GetBootIdString();
  if (!boot_id_str_or_status.ok()) {
    LOG(INFO) << boot_id_str_or_status.status();
  } else {
    boot_id_str = boot_id_str_or_status.value();
  }
  local_topology.set_boot_id(boot_id_str);
  if (slice_index.has_value()) {
    local_topology.set_slice_index(*slice_index);
  }
  for (const auto& ordinal_and_device : local_device_states) {
    const se::Platform* platform =
        ordinal_and_device.second->executor()->GetPlatform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(
            ordinal_and_device.second->local_hardware_id().value()));
    DeviceProto* device_proto = local_topology.add_devices();
    device_proto->set_local_device_ordinal(ordinal_and_device.first);
    device_proto->set_name(desc->name());
    device_proto->set_vendor(desc->device_vendor());
    auto compute_capability = MakeComputeCapabilityString(desc.get());
    device_proto->set_compute_capability(compute_capability);
    device_proto->set_core_count(desc->core_count());
#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 12040
    if (std::stoi(compute_capability) >= 9) {
      auto fabric_info = GetDeviceFabricInfo(ordinal_and_device.first);
      if (fabric_info.ok()) {
        device_proto->set_fabric_uuid(*fabric_info);
      }
    }
#endif  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 12040
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
      std::unique_ptr<LocalDeviceState> local_device;
      if (node.node_id() == node_id) {
        auto it = local_device_states.find(device_proto.local_device_ordinal());
        TF_RET_CHECK(it != local_device_states.end())
            << device_proto.local_device_ordinal();
        TF_RET_CHECK(it->second != nullptr);
        local_device = std::move(it->second);
        gpu_device_ids[device_proto.local_device_ordinal()] = global_device_id;
        // Assign some descriptive names for profiling tools.
        NameDeviceAndLauncherThread(node, device_proto,
                                    local_device->execute_thread());
      }
      auto device = std::make_unique<StreamExecutorGpuDevice>(
          device_proto.global_device_id(), std::move(local_device),
          device_proto.name(), device_proto.vendor(),
          device_proto.compute_capability(), device_proto.core_count(),
          node.node_id(), device_proto.slice_index());
      devices.push_back(std::move(device));
    }
  }
  for (const auto& device : local_device_states) {
    TF_RET_CHECK(device.second == nullptr);
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
      {node_id, global_topology.nodes().size(), local_device_states.size(),
       kv_store, device_to_node, gpu_executable_run_options}));

  TF_ASSIGN_OR_RETURN(GpuTopologyProto gpu_topology,
                      BuildGpuTopology(global_topology));
  return std::make_pair(std::move(devices), gpu_topology);
}

std::string MakeComputeCapabilityString(const se::DeviceDescription* desc) {
  se::GpuComputeCapability cc = desc->gpu_compute_capability();
  if (std::holds_alternative<se::CudaComputeCapability>(cc)) {
    auto nvcc = std::get<se::CudaComputeCapability>(cc);
    return absl::StrCat(nvcc.major, ".", nvcc.minor);
  } else if (std::holds_alternative<se::RocmComputeCapability>(cc)) {
    auto rocmcc = std::get<se::RocmComputeCapability>(cc);
    return rocmcc.gfx_version();
  } else {
    return "unknown";
  }
}

StreamExecutorGpuDevice::StreamExecutorGpuDevice(
    int id, std::unique_ptr<LocalDeviceState> local_device_state,
    std::string device_kind, std::string device_vendor,
    std::string compute_capability, int core_count, int node_id,
    int slice_index)
    : PjRtStreamExecutorDevice(id, std::move(local_device_state),
                               /*process_index=*/node_id,
                               std::move(device_kind)),
      device_vendor_(std::move(device_vendor)),
      slice_index_(slice_index) {
  std::array<int, 1> coords = {local_device_id().value()};
  description().SetCoords(coords);
  std::vector<int64_t> v_coords(description().coords().begin(),
                                description().coords().end());

  description().SetAttributes(
      {{"coords", xla::PjRtDeviceAttribute(v_coords)},
       {"device_vendor", device_vendor_},
       {"slice_index", static_cast<int64_t>(slice_index)},
       {"compute_capability", xla::PjRtDeviceAttribute(compute_capability)},
       {"core_count", static_cast<int64_t>(core_count)}});
  description().SetToString(absl::StrFormat(
      "StreamExecutorGpuDevice(device_kind=%s, id=%i, process_index=%i, "
      "slice_index=%i))",
      description().device_kind(), id, process_index(), slice_index));
  description().SetDebugString(absl::StrFormat("%s_%i(process=%i,(%i))",
                                               description().device_kind(), id,
                                               process_index(), v_coords[0]));
}

int StreamExecutorGpuDevice::slice_index() const { return slice_index_; }

absl::string_view StreamExecutorGpuDevice::device_vendor() const {
  return device_vendor_;
}

absl::StatusOr<tsl::AllocatorStats> StreamExecutorGpuDevice::GetAllocatorStats()
    const {
  if (!IsAddressable()) {
    return FailedPrecondition(
        "GetAllocatorStats() is allowed only for addressable devices");
  }

  auto* allocator_adapter = dynamic_cast<se::MultiDeviceAdapter*>(
      tensorflow::down_cast<PjRtStreamExecutorClient*>(client())->allocator());
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

absl::Span<int const> StreamExecutorGpuDevice::coords() const {
  return description().coords();
}

absl::StatusOr<PjRtMemorySpace*> StreamExecutorGpuDevice::default_memory_space()
    const {
  return memory_space_by_kind_id(StreamExecutorGpuHbmMemorySpace::kKindId);
}

const int StreamExecutorGpuHbmMemorySpace::kKindId = []() {
  uint32_t kind_id = tsl::Fingerprint32(StreamExecutorGpuHbmMemorySpace::kKind);
  return static_cast<int>(kind_id);
}();

StreamExecutorGpuHbmMemorySpace::StreamExecutorGpuHbmMemorySpace(
    int id, PjRtDevice* device)
    : PjRtStreamExecutorMemorySpace(id, device, kKind, kKindId) {}

absl::StatusOr<std::unique_ptr<PjRtClient>> GetStreamExecutorGpuClient(
    const GpuClientOptions& options) {
#if TENSORFLOW_USE_ROCM
  auto pjrt_platform_name = xla::RocmName();
#elif TENSORFLOW_USE_SYCL
  auto pjrt_platform_name = xla::SyclName();
#else   // TENSORFLOW_USE_ROCM
  auto pjrt_platform_name = xla::CudaName();
#endif  // TENSORFLOW_USE_ROCM

  TF_ASSIGN_OR_RETURN(
      LocalClient * xla_client,
      GetGpuXlaClient(options.platform_name, options.allowed_devices));
  std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states;
  TF_ASSIGN_OR_RETURN(local_device_states, BuildLocalDeviceStates(xla_client));
  EnablePeerAccess(xla_client->backend().stream_executors());
  TF_ASSIGN_OR_RETURN(auto allocator,
                      GetStreamExecutorGpuDeviceAllocator(
                          xla_client->platform(), options.allocator_config,
                          local_device_states));
  TF_ASSIGN_OR_RETURN(
      auto host_memory_allocator,
      GetGpuHostAllocator(local_device_states.begin()->second->executor()));

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
      BuildDistributedDevices(pjrt_platform_name,
                              std::move(local_device_states), options.node_id,
                              options.num_nodes, gpu_run_options.get(),
                              kv_store, options.enable_mock_nccl,
                              options.mock_gpu_topology, options.slice_index));

  auto gpu_topology = std::shared_ptr<const GpuTopology>(
      GpuTopology::FromProto(device_topology_pair.second));

  return std::unique_ptr<PjRtClient>(std::make_unique<StreamExecutorGpuClient>(
      pjrt_platform_name, xla_client, std::move(device_topology_pair.first),
      options.node_id, std::move(allocator), std::move(host_memory_allocator),
      options.should_stage_host_to_device_transfers, std::move(gpu_run_options),
      std::move(kv_store), std::move(options.distributed_runtime_client),
      options.abort_collectives_on_failure, std::move(gpu_topology)));
}

std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> BuildLocalDevices(
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id) {
  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  for (auto& ordinal_and_device : local_device_states) {
    const se::DeviceDescription& desc =
        ordinal_and_device.second->executor()->GetDeviceDescription();
    auto device = std::make_unique<StreamExecutorGpuDevice>(
        ordinal_and_device.first, std::move(ordinal_and_device.second),
        desc.name(), desc.device_vendor(), MakeComputeCapabilityString(&desc),
        desc.core_count(), node_id);
    devices.push_back(std::move(device));
  }
  return devices;
}

absl::StatusOr<std::string> GetDeviceFabricInfo(const int device_ordinal) {
#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 12040
  if (!gpu::GpuPerformanceWithCollectiveModel::InitNvml()) {
    return absl::InternalError("Failed to initialize NVML library.");
  }

  // NVML library is not a part of the CUDA toolkit, so there might be a
  // situation when user is using CUDA 12.4 an higher, but the host NVML
  // version doen't have the required functions.
  if (xla_nvmlDeviceGetHandleByPciBusId_v2 == nullptr ||
      xla_nvmlDeviceGetGpuFabricInfoV == nullptr) {
    return absl::InternalError("NVML library doesn't have required functions.");
  }

  char pciBusId[] = "00000000:00:00.0";
  cudaDeviceGetPCIBusId(pciBusId, sizeof(pciBusId), device_ordinal);
  nvmlDevice_t device;
  auto get_bus_id_status =
      xla_nvmlDeviceGetHandleByPciBusId_v2(pciBusId, &device);
  CHECK_EQ(get_bus_id_status, NVML_SUCCESS);

  nvmlGpuFabricInfoV_t fabricInfo = {
      .version = nvmlGpuFabricInfo_v2,
      .state = NVML_GPU_FABRIC_STATE_NOT_SUPPORTED};
  auto get_fabric_info_status =
      xla_nvmlDeviceGetGpuFabricInfoV(device, &fabricInfo);
  CHECK_EQ(get_fabric_info_status, NVML_SUCCESS);

  if (fabricInfo.state == NVML_GPU_FABRIC_STATE_NOT_SUPPORTED) {
    std::string error_message =
        "NVML doesn't support extracting fabric info or NVLink is not used by "
        "the device.";
    VLOG(2) << error_message;
    return absl::InternalError(error_message);
  }

  CHECK_EQ(sizeof(fabricInfo.clusterUuid), 16);
  std::string uuid_str = absl::StrFormat(
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      fabricInfo.clusterUuid[0], fabricInfo.clusterUuid[1],
      fabricInfo.clusterUuid[2], fabricInfo.clusterUuid[3],
      fabricInfo.clusterUuid[4], fabricInfo.clusterUuid[5],
      fabricInfo.clusterUuid[6], fabricInfo.clusterUuid[7],
      fabricInfo.clusterUuid[8], fabricInfo.clusterUuid[9],
      fabricInfo.clusterUuid[10], fabricInfo.clusterUuid[11],
      fabricInfo.clusterUuid[12], fabricInfo.clusterUuid[13],
      fabricInfo.clusterUuid[14], fabricInfo.clusterUuid[15]);
  return absl::StrCat(uuid_str, "/", std::to_string(fabricInfo.cliqueId));
#else   // defined(GOOGLE_CUDA) && CUDA_VERSION >= 12040
  std::string error_message = "NVML usage is not supported";
  VLOG(2) << error_message;
  return absl::InternalError(error_message);
#endif  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 12040
}

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
static absl::Status CheckAlignment(const BufferAllocation& allocation,
                                   se::DeviceMemoryBase buffer, int arg_idx) {
  const int64_t expected_alignment = [&] {
    if (allocation.is_entry_computation_parameter()) {
      return gpu::kEntryParameterAlignBytes;
    } else if (allocation.is_constant()) {
      return gpu::kConstantBufferAlignBytes;
    } else {
      return gpu::kXlaAllocatedBufferAlignBytes;
    }
  }();
  if (!buffer.is_null() &&
      reinterpret_cast<uintptr_t>(buffer.opaque()) % expected_alignment != 0) {
    return Internal(
        "Address of buffer %d must be a multiple of %x, but "
        "was %p",
        arg_idx, expected_alignment, buffer.opaque());
  }
  return absl::OkStatus();
}
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

absl::StatusOr<PjRtStreamExecutorExecutionOutput>
StreamExecutorGpuClient::RunAsync(
    LocalExecutable& exec, PjRtDevice* device,
    std::vector<ShapeTree<PjRtStreamExecutorExecutionInput>> arguments,
    ExecutableRunOptions run_options_inp) {
#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
  std::vector<const Shape*> argument_shapes;
  argument_shapes.reserve(arguments.size());
  for (const ShapeTree<PjRtStreamExecutorExecutionInput>& arg : arguments) {
    argument_shapes.push_back(&arg.shape());
  }

  TF_ASSIGN_OR_RETURN(auto options_and_stream,
                      exec.RunHelper(argument_shapes, run_options_inp));
  auto* gpu_exec =
      tensorflow::down_cast<xla::gpu::GpuExecutable*>(exec.executable());
  const ServiceExecutableRunOptions* run_options = &options_and_stream.first;
  se::DeviceMemoryAllocator* const memory_allocator = run_options->allocator();

  se::StreamExecutor* executor = run_options->stream()->parent();

  // Use the `device_ordinal` from the `run_options` if it is provided. This is
  // the ordinal of the logical devices (e.g., virtual GPUs). If it is not
  // provided, the ordinals of the logical and physical devices are the same.
  const int device_ordinal = run_options->device_ordinal() != -1
                                 ? run_options->device_ordinal()
                                 : executor->device_ordinal();

  XLA_SCOPED_LOGGING_TIMER(
      absl::StrCat("GpuExecutable::ExecuteAsyncOnStreamImpl(",
                   gpu_exec->module_name(), ")"));

  // GpuExecutable always bound to a single GpuContext during its execution, so
  // we activate it once to skip expensive context activations later.
  auto activation = executor->Activate();

  // Lock the GPU with a shared lock so that we don't interfere with autotuning
  // that may be running during JIT compilation while allowing multiple XLA
  // computations to use the same GPU simultaneously. We do not add locking for
  // "recursive" invocations, which are done when holding a lock already.
  std::variant<absl::ReaderMutexLock, absl::WriterMutexLock> gpu_lock(
      std::in_place_index_t<0>{}, &gpu::GetGpuMutex(executor));

  // Maybe update to a writer lock to get exclusive access to underlying GPU.
  if (auto* gpu_opts = run_options->run_options().gpu_executable_run_options();
      gpu_opts && gpu_opts->requires_exclusive_lock_on_gpu()) {
    gpu_lock.emplace<1>(&gpu::GetGpuMutex(executor));
  }

  const gpu::GpuExecutable::BufferAllocToDeviceMemoryMap* globals;
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Resolve constant globals"); },
        tsl::profiler::TraceMeLevel::kInfo);

    TF_ASSIGN_OR_RETURN(
        globals, gpu_exec->ResolveConstantGlobals(run_options->stream()));
  }

  absl::Span<const BufferAllocation> allocations = gpu_exec->GetAllocations();

  std::vector<se::DeviceMemoryBase> buffers(allocations.size());
  {
    tsl::profiler::TraceMe hlo_module_activity(
        [&] { return std::string("Build buffer allocations"); },
        tsl::profiler::TraceMeLevel::kInfo);
    const int64_t num_buffers = allocations.size();
    for (int64_t i = 0; i < num_buffers; ++i) {
      const BufferAllocation& allocation = allocations[i];
      se::DeviceMemoryBase& buffer = buffers[i];
      if (allocation.is_thread_local()) {
        // buffer = se::DeviceMemoryBase{};
      } else if (allocation.is_entry_computation_parameter()) {
        int64_t param_no = allocation.parameter_number();
        buffer = [&] {
          return arguments[param_no]
              .element(allocation.param_shape_index())
              .buf->mem();
        }();
        if (buffer.is_null() && buffer.size() > 0) {
          return FailedPrecondition(
              "Cannot run XLA computation because pointer to (sub-)buffer at "
              "index %s of parameter %d was null.  All pointers to "
              "(sub-)buffers must not be null, unless the (sub-)buffer has "
              "zero elements.",
              allocation.param_shape_index().ToString(), param_no);
        }
      } else if (allocation.is_constant()) {
        auto it = globals->find(i);
        if (it != globals->end()) {
          buffer = it->second;
        }
      } else {
        // Allocate each allocation that might escape, or is the temp buffer.
        CHECK(allocation.maybe_live_out() ||
              allocation.IsPreallocatedTempBuffer());
        const int64_t buffer_size = allocation.size();
        if (buffer_size > 0) {
          TF_ASSIGN_OR_RETURN(
              se::OwningDeviceMemory owning_buffer,
              memory_allocator->Allocate(device_ordinal, buffer_size,
                                         /*retry_on_failure=*/true,
                                         /*memory_space=*/allocation.color()));
          buffer = owning_buffer.Release();
        }
      }
      TF_RETURN_IF_ERROR(CheckAlignment(allocation, buffer, i));
    }
  }
  xla::gpu::BufferAllocations buffer_allocations(buffers, device_ordinal,
                                                 memory_allocator);
  VLOG(3) << buffer_allocations.ToString();

  std::set<se::DeviceMemoryBase> buffers_in_result;

  xla::ShapeTree<tsl::RCReference<RawSEDeviceMemory>> results(
      gpu_exec->output_shape());

  for (auto& p : results) {
    const ShapeIndex& index = p.first;
    if (!gpu_exec->output_info().contains(index)) {
      continue;
    }
    const gpu::GpuExecutable::OutputInfo& output_info =
        gpu_exec->output_info().at(index);
    const BufferAllocation* allocation =
        &allocations[output_info.allocation_index];
    se::DeviceMemoryBase result_buffer;

    VLOG(4) << "Looking at: allocation " << output_info.allocation_index
            << " @ index: " << index.ToString();

    if (output_info.alias_config) {
      PjRtStreamExecutorExecutionInput& input =
          *arguments[allocation->parameter_number()].mutable_element(
              allocation->param_shape_index());
      if (output_info.alias_config->must_alias() && !input.is_donated) {
        return InvalidArgument(
            "An input was configured to be must-alias at "
            "compile time but not donated at runtime: allocation %d",
            output_info.allocation_index);
      }
      if (input.is_donated) {
        // If the caller passes the ownership of the device memory, reuse it
        // as the output buffer. It is up to the caller whether or not to
        // donate a buffer; the aliasing information describes which buffers
        // may alias, not buffers that must alias.
        buffers_in_result.insert(input.buf->mem());
        p.second = input.buf;
        input.is_donated = false;
        continue;
      } else if (!output_info.passthrough &&
                 !ShapeUtil::GetSubshape(gpu_exec->output_shape(), index)
                      .IsTuple()) {
        // The guard is above is not to insert copy-protection when aliasing
        // pass-through params, as we do not need to write into the output
        // buffer.
        VLOG(3) << "Using copy-protection: aliasing is specified, but the "
                   "buffer is not donated; allocating a fresh buffer";
        int64_t allocation_size = ShapeUtil::ByteSizeOf(
            ShapeUtil::GetSubshape(gpu_exec->output_shape(), index));
        absl::StatusOr<se::OwningDeviceMemory> allocated_buffer =
            memory_allocator->Allocate(device_ordinal, allocation_size,
                                       /*retry_on_failure=*/true,
                                       /*memory_space=*/allocation->color());
        if (!allocated_buffer.ok()) {
          return gpu_exec->VerboseAllocationError(allocated_buffer.status());
        }
        result_buffer = allocated_buffer->Release();
        se::DeviceMemoryBase& aliased_buffer =
            buffer_allocations.GetMutableDeviceAddress(
                output_info.allocation_index);
        CHECK_EQ(aliased_buffer.size(), result_buffer.size());
        TF_RETURN_IF_ERROR(run_options->stream()->MemcpyD2D(
            &result_buffer, aliased_buffer, aliased_buffer.size()));
        aliased_buffer = result_buffer;
      }
    }

    if (result_buffer.is_null()) {
      // The source instruction should have a non-parameter buffer
      // assigned.
      result_buffer =
          buffer_allocations.GetDeviceAddress(output_info.allocation_index);
    }
    buffers_in_result.insert(result_buffer);

    p.second = RawSEDeviceMemory::Create(
        result_buffer, device->local_device_id(), memory_allocator);
  }

  TF_RETURN_IF_ERROR(gpu_exec->ExecuteThunks(buffer_allocations, run_options));

  TF_RETURN_IF_ERROR(buffer_allocations.TearDown(buffers_in_result,
                                                 gpu_exec->GetAllocations()));

  std::vector<tsl::RCReference<RawSEDeviceMemory>> to_be_released;

  // Free allocations for arguments.
  for (ShapeTree<PjRtStreamExecutorExecutionInput>& input : arguments) {
    for (auto& v : input) {
      if (v.second.is_donated) {
        to_be_released.push_back(std::move(v.second.buf));
      }
    }
  }

  return PjRtStreamExecutorExecutionOutput(
      {std::move(results), std::move(to_be_released), {}});
#else
  return PjRtStreamExecutorClient::RunAsync(exec, device, std::move(arguments),
                                            std::move(run_options_inp));
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}

}  // namespace xla
