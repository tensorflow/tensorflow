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
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
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
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/client/local_client.h"
#include "xla/client/xla_computation.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/literal.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/distributed/key_value_store_interface.h"
#include "xla/pjrt/distributed/topology_util.h"
#include "xla/pjrt/event_pool.h"
#include "xla/pjrt/gpu/gpu_helpers.h"
#include "xla/pjrt/host_memory_spaces.h"
#include "xla/pjrt/local_device_state.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_compiler.h"
#include "xla/pjrt/pjrt_device_description.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_future.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/stream_executor_executable.h"
#include "xla/pjrt/tracked_device_buffer.h"
#include "xla/service/compiler.h"
#include "xla/service/computation_placer.h"
#include "xla/service/global_device_id.h"
#include "xla/service/shaped_buffer.h"
#include "xla/service/transfer_manager.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_memory.h"
#include "xla/stream_executor/device_memory_allocator.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/framework/allocator.h"
#include "tsl/lib/strings/proto_serialization.h"
#include "tsl/platform/casts.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/fingerprint.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/threadpool.h"
#include "tsl/profiler/lib/connected_traceme.h"
#include "tsl/profiler/lib/traceme.h"

#if defined(GOOGLE_CUDA) || defined(TENSORFLOW_USE_ROCM)
#include "xla/pjrt/compile_options.pb.h"
#include "xla/pjrt/gpu/gpu_metrics.h"
#include "xla/pjrt/gpu/nccl_id_store.h"
#include "xla/pjrt/stream_executor_executable.pb.h"
#include "xla/service/gpu/gpu_compiler.h"
#include "xla/xla.pb.h"
#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#if GOOGLE_CUDA
#include "third_party/gpus/cuda/include/cuda.h"
#include "third_party/gpus/cuda/include/cuda_runtime_api.h"
#include "xla/stream_executor/gpu/gpu_cudamallocasync_allocator.h"
#elif TENSORFLOW_USE_ROCM
#include "rocm/rocm_config.h"
#endif

#include "xla/service/gpu/gpu_executable_run_options.h"
#include "xla/statusor.h"
#include "xla/stream_executor/integrations/device_mem_allocator.h"
#include "xla/stream_executor/integrations/tf_allocator_adapter.h"
#include "xla/util.h"

namespace xla {
class AsyncHostToDeviceTransferManager
    : public xla::PjRtClient::AsyncHostToDeviceTransferManager {
 public:
  static absl::StatusOr<std::unique_ptr<AsyncHostToDeviceTransferManager>>
  Create(absl::Span<const Shape> shapes, PjRtStreamExecutorDevice* device,
         PjRtStreamExecutorClient* client) {
    absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers;
    absl::InlinedVector<std::shared_ptr<TrackedDeviceBuffer>, 4> buffer_ptrs;
    absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
        definition_events;
    buffers.reserve(shapes.size());
    buffer_ptrs.reserve(shapes.size());
    definition_events.reserve(shapes.size());
    for (const auto& shape : shapes) {
      if (shape.IsTuple()) {
        return Unimplemented(
            "Async buffer transfer of tuples not implemented.");
      }
      // Initialize a definition event for each async buffer. The definition
      // event will block the buffer usage until the transfer is done.
      definition_events.push_back(
          std::make_shared<BufferSequencingEvent>(client->thread_pool()));
      TF_ASSIGN_OR_RETURN(Shape compact_shape,
                          client->client()
                              ->backend()
                              .transfer_manager()
                              ->ChooseCompactLayoutForShape(shape));
      LocalDeviceState* local_device = device->local_device_state();
      se::Stream* h2d_stream = local_device->host_to_device_stream();
      TF_ASSIGN_OR_RETURN(auto buffer,
                          AllocateDestinationBuffer(
                              compact_shape, device, local_device, h2d_stream,
                              /*is_uninitialized_create=*/true, client,
                              definition_events.back()));
      // Get a temporary hold just so we can fish out a shared_ptr to the
      // TrackedDeviceBuffer. It's ok to drop the hold before return the
      // buffers, because the invariants of this class ensure that the buffer
      // definition event will not fire until after all of this class' uses of
      // the TrackedDeviceBuffer have completed.
      auto* se_buffer =
          tensorflow::down_cast<PjRtStreamExecutorBuffer*>(buffer.get());
      DCHECK(se_buffer);
      auto hold = se_buffer->GetBufferWithUsageHold();
      buffer_ptrs.push_back(hold.buffer());
      buffers.push_back(std::move(buffer));
    }

    return std::make_unique<AsyncHostToDeviceTransferManager>(
        std::move(buffers), std::move(buffer_ptrs),
        std::move(definition_events), device);
  }

  AsyncHostToDeviceTransferManager(
      absl::InlinedVector<std::unique_ptr<PjRtBuffer>, 4> buffers,
      absl::InlinedVector<std::shared_ptr<TrackedDeviceBuffer>, 4> buffer_ptrs,
      absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
          definition_events,
      PjRtStreamExecutorDevice* device)
      : buffers_(std::move(buffers)),
        buffer_ptrs_(std::move(buffer_ptrs)),
        definition_events_(std::move(definition_events)),
        remaining_buffer_count_(buffer_ptrs_.size()),
        transfers_in_flight_(0),
        device_(device) {
    buffer_sizes_.reserve(buffer_ptrs_.size());
    for (const auto& ptr : buffer_ptrs_) {
      DCHECK_EQ(ptr->device_memory().size(), 1);
      buffer_sizes_.push_back(ptr->device_memory()[0].size());
    }
    last_transfer_started_.resize(buffer_ptrs_.size(), false);
  }

  ~AsyncHostToDeviceTransferManager() override {
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
        "AsyncHostToDeviceTransferManager::TransferLiteralToBuffer");
    auto* stream = device_->local_device_state()->host_to_device_stream();
    auto* se_client =
        tensorflow::down_cast<PjRtStreamExecutorClient*>(device_->client());
    DCHECK(se_client);

    TransferManager* transfer_manager =
        se_client->client()->backend().transfer_manager();
    TF_ASSIGN_OR_RETURN(
        Shape compact_shape,
        transfer_manager->ChooseCompactLayoutForShape(literal.shape()));

    std::shared_ptr<TrackedDeviceBuffer> buffer;
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
      if (buffer->device_memory().empty()) {
        return InvalidArgument(
            "TransferLiteralToBuffer requested for buffer index %d which has "
            "been donated. Async transfer of donated buffers is not supported "
            "in SE:GPU",
            buffer_index);
      }
      DCHECK_EQ(buffer->device_memory().size(), 1);

      auto& buffer_memory = buffer->device_memory()[0];
      if (transfer_manager->GetByteSizeRequirement(compact_shape) !=
          buffer_memory.size()) {
        return InvalidArgument(
            "TransferLiteralToBuffer shape %s has size %lld "
            "but buffer has size %lld",
            ShapeUtil::HumanStringWithLayout(compact_shape),
            transfer_manager->GetByteSizeRequirement(compact_shape),
            buffer_memory.size());
      }
      ++transfers_in_flight_;
    }

    // The host to device transfer is performed on a thread pool, mostly because
    // it includes linearization that may be slow.
    // TODO(misard) assess if it would be preferable to introduce a heuristic to
    // put the transfer into the calling thread for small literals.
    auto transfer_h2d = [this, buffer_index, stream, transfer_manager, literal,
                         device_buffer = buffer.get(), compact_shape,
                         local_device =
                             std::move(device_->local_device_state()),
                         on_done = std::move(on_done)]() mutable {
      tsl::profiler::TraceMe traceme(
          "AsyncHostToDeviceTransferManager::TransferLiteralToBuffer::transfer_"
          "h2d");

      auto event = local_device->event_pool().AllocateEvent(stream->parent());

      // Initiate linearization and transfer of the buffer on the stream.
      ShapedBuffer buffer = device_buffer->AsShapedBuffer(compact_shape);
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
        client->should_stage_host_to_device_transfers();
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
    if (buffer_ptrs_[buffer_index]->device_memory().empty()) {
      return InvalidArgument(
          "TransferRawDataToSubBuffer requested for buffer index %d which has "
          "been donated. Async transfer of donated buffers is not supported "
          "in SE:GPU",
          buffer_index);
    }
    DCHECK_EQ(buffer_ptrs_[buffer_index]->device_memory().size(), 1);
    auto& buffer_memory = buffer_ptrs_[buffer_index]->device_memory()[0];
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
  absl::InlinedVector<std::shared_ptr<TrackedDeviceBuffer>, 4> buffer_ptrs_
      ABSL_GUARDED_BY(mu_);
  // True if the last transfer for a buffer has been initiated. Used to prevent
  // a client initiating another transfer after the last transfer has already
  // been initiated.
  absl::InlinedVector<bool, 4> last_transfer_started_ ABSL_GUARDED_BY(mu_);
  // The buffer definition events on all the buffers, unblocked once the
  // corresponding buffer transfer has completed.
  absl::InlinedVector<std::shared_ptr<BufferSequencingEvent>, 4>
      definition_events_ ABSL_GUARDED_BY(mu_);
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
        buffer_ptrs_[buffer_index] = nullptr;
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

StreamExecutorGpuClient::StreamExecutorGpuClient(
    std::string platform_name, LocalClient* client,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices,
    int process_index, std::unique_ptr<se::DeviceMemoryAllocator> allocator,
    std::unique_ptr<tsl::Allocator> host_memory_allocator,
    bool should_stage_host_to_device_transfers,
    std::unique_ptr<gpu::GpuExecutableRunOptions> gpu_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store)
    : xla::PjRtStreamExecutorClient(
          platform_name, client, std::move(devices), process_index,
          std::move(allocator), std::move(host_memory_allocator),
          should_stage_host_to_device_transfers, std::move(gpu_run_options)),
      topology_(xla::StreamExecutorGpuTopologyDescription::Create(
          tsl::Fingerprint64(platform_name), platform_name,
          devices_.back()->device_kind(), devices_)),
      kv_store_(std::move(kv_store)) {
  for (auto* device : addressable_devices()) {
    // Use the device id to construct a globally unique memory space id. We do
    // not promise that memory space ids and device ids are the same.
    const int id = device->id();
    auto memory_space =
        std::make_unique<StreamExecutorGpuHbmMemorySpace>(id, device);
    tensorflow::down_cast<PjRtStreamExecutorDevice*>(device)->AttachMemorySpace(
        memory_space.get());
    owned_memory_spaces_.push_back(std::move(memory_space));
    const size_t basePinnedId = devices.size();
    auto pinned = std::make_unique<PinnedHostMemorySpace>(basePinnedId, device);
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
    absl::Span<const Shape> shapes, PjRtDevice* device) {
  auto* stream_executor_device =
      tensorflow::down_cast<PjRtStreamExecutorDevice*>(device);
  return xla::AsyncHostToDeviceTransferManager::Create(
      shapes, stream_executor_device, this);
}

absl::StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
StreamExecutorGpuClient::CreateBuffersForAsyncHostToDevice(
    absl::Span<const Shape> shapes, PjRtMemorySpace* memory_space) {
  CHECK_EQ(memory_space->devices().size(), 1);
  return CreateBuffersForAsyncHostToDevice(shapes, memory_space->devices()[0]);
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

  auto device_buffer = hold.buffer();
  if (device_buffer->device_memory().size() != 1) {
    return PjRtFuture<>(InvalidArgument("Copy raw buffer called on tuple"));
  }

  auto promise = PjRtFuture<>::CreatePromise();
  auto usage_event =
      std::make_shared<BufferSequencingEvent>(this->thread_pool());

  // When using the ComputeSynchronized allocation model, retain a reference to
  // the device_buffer until the copy completes, to ensure that the buffer isn't
  // deleted or donated while it is still in use. The choice of retaining a
  // reference at the host is a heuristic; the alternative is to ensure, before
  // freeing the buffer, that the compute stream is synchronized past the
  // transfer, but it seems better to hold onto the buffer too long than to
  // stall the compute stream.
  hold.ConvertUsageHold(stream, usage_event, /*reference_held=*/true);

  auto async_copy = [this, promise, offset, transfer_size, stream, local_device,
                     device_buffer, usage_event = std::move(usage_event)](
                        absl::StatusOr<void*> dst) mutable {
    absl::StatusOr<EventPool::Handle> event =
        local_device->event_pool().AllocateEvent(stream->parent());
    if (!event.ok()) {
      promise.Set(event.status());
      return;
    }

    absl::Status defined_status =
        device_buffer->definition_events()[0]->GetDefinedStatus();
    if (!defined_status.ok()) {
      promise.Set(defined_status);
      return;
    }

    auto& device_memory = device_buffer->device_memory()[0];
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

    WaitForBufferDefinitionEventsOnStream(*device_buffer, stream);

    if (transfer_size != 0) {
      if (should_stage_host_to_device_transfers()) {
        if (host_memory_allocator() == nullptr) {
          promise.Set(InvalidArgument(
              "host_memory_allocator should be initialized for staging buffer "
              "transfer."));
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
        stream, [promise, device_buffer = std::move(device_buffer)]() mutable {
          promise.Set();
        });
    if (!callback_status.ok()) {
      promise.Set(std::move(callback_status));
      return;
    }
  };

  device_buffer->definition_events()[0]->ExecuteOrAddToFutureTasks(
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

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::Compile(const XlaComputation& computation,
                                 CompileOptions options) {
  options.executable_build_options.set_key_value_store(kv_store_);
  auto executable = PjRtStreamExecutorClient::Compile(computation, options);

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
  return PjRtStreamExecutorClient::DeserializeExecutable(serialized, options);
}

absl::StatusOr<std::unique_ptr<PjRtLoadedExecutable>>
StreamExecutorGpuClient::Load(std::unique_ptr<PjRtExecutable> executable) {
  auto se_executable = absl::WrapUnique(
      tensorflow::down_cast<StreamExecutorExecutable*>(executable.release()));

  CompileOptions compile_options = se_executable->compile_options();
  CompileOptions input_options = compile_options;
  TF_RETURN_IF_ERROR(compile_options.ApplyAllOptionOverrides());
  TF_ASSIGN_OR_RETURN(ExecutableExtras extras,
                      GetExecutableExtras(&compile_options));

  // Load Executable from AOT compilation result.
  std::vector<std::unique_ptr<LocalExecutable>> local_executables;
  local_executables.reserve(se_executable->aot_executables().size());
  for (std::unique_ptr<xla::AotCompilationResult>& aot_executable :
       se_executable->aot_executables()) {
    TF_ASSIGN_OR_RETURN(std::string serialized,
                        aot_executable->SerializeAsString());
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<LocalExecutable> local_executable,
        client()->Load(serialized, compile_options.executable_build_options));
    local_executables.push_back(std::move(local_executable));
  }
  bool parameter_is_tupled_arguments =
      compile_options.parameter_is_tupled_arguments;
  auto ret = std::make_unique<PjRtStreamExecutorLoadedExecutable>(
      std::move(local_executables), parameter_is_tupled_arguments,
      std::move(extras.device_assignment), std::move(input_options),
      std::move(extras.addressable_device_logical_ids),
      std::move(extras.addressable_devices), this);
  TF_RETURN_IF_ERROR(ret->SetUpDonation(parameter_is_tupled_arguments));
  return std::unique_ptr<PjRtLoadedExecutable>(std::move(ret));
}

namespace {

#if defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

absl::StatusOr<std::vector<se::MultiDeviceAdapter::AllocatorInfo>>
CreateCudaAsyncAllocator(
    se::Platform* platform,
    const std::map<int, std::unique_ptr<LocalDeviceState>>& addressable_devices,
    double memory_fraction, bool preallocate) {
  CHECK_GT(addressable_devices.size(), 0);
  std::vector<se::MultiDeviceAdapter::AllocatorInfo> allocators;

  for (auto& ordinal_and_device : addressable_devices) {
    se::StreamExecutor* executor = ordinal_and_device.second->executor();
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
    if (preallocate) {
      LOG(INFO) << "XLA backend allocating " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for CudaAsyncAllocator.";
    } else {
      LOG(INFO) << "XLA backend will use up to " << allocator_memory
                << " bytes on device " << device_ordinal
                << " for CudaAsyncAllocator.";
    }

    auto allocator = std::make_unique<se::GpuCudaMallocAsyncAllocator>(
        tsl::PlatformDeviceId(device_ordinal), allocator_memory, preallocate);
    allocator->SetStreamAndPreallocateMemory(
        ordinal_and_device.second->compute_stream()
            ->platform_specific_handle()
            .stream);
    allocators.emplace_back(std::move(allocator),
                            ordinal_and_device.second->compute_stream(),
                            /*memory_space=*/0);
  }
  return allocators;
}

#else  // defined(GOOGLE_CUDA) && CUDA_VERSION >= 11020

absl::StatusOr<std::vector<se::MultiDeviceAdapter::AllocatorInfo>>
CreateCudaAsyncAllocator(
    se::Platform* platform,
    const std::map<int, std::unique_ptr<LocalDeviceState>>& addressable_devices,
    double memory_fraction, bool preallocate) {
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
      auto allocators_or = CreateCudaAsyncAllocator(
          platform, addressable_devices, allocator_config.memory_fraction,
          allocator_config.preallocate);
      if (allocators_or.ok()) {
        LOG(INFO) << "Using CUDA async allocator.";
        allocators = std::move(allocators_or.value());
        break;
      }
      LOG(ERROR) << "Failed to initialize CUDA async allocator: "
                 << allocators_or.status() << "; falling back to BFC.";
      [[fallthrough]];
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
    auto host_allocator =
        GetGpuHostAllocator(ordinal_and_device.second->executor());
    allocators.emplace_back(std::move(host_allocator),
                            ordinal_and_device.second->compute_stream(),
                            /*memory_space=*/
                            static_cast<int>(se::MemoryType::kHost));
  }

  return std::make_unique<se::MultiDeviceAdapter>(platform,
                                                  std::move(allocators));
}

}  // namespace

absl::Status BuildDistributedDevices(
    std::string_view platform_name,
    std::map<int, std::unique_ptr<LocalDeviceState>> local_device_states,
    int node_id, int num_nodes,
    std::vector<std::unique_ptr<PjRtStreamExecutorDevice>>* devices,
    gpu::GpuExecutableRunOptions* gpu_executable_run_options,
    std::shared_ptr<KeyValueStoreInterface> kv_store, bool enable_mock_nccl,
    absl::Duration get_local_topology_timeout,
    absl::Duration get_global_topology_timeout) {
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
  for (const auto& ordinal_and_device : local_device_states) {
    const se::Platform* platform =
        ordinal_and_device.second->executor()->GetPlatform();
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::se::DeviceDescription> desc,
        platform->DescriptionForDevice(ordinal_and_device.first));
    DeviceProto* device_proto = local_topology.add_devices();
    device_proto->set_local_device_ordinal(ordinal_and_device.first);
    device_proto->set_name(desc->name());
    device_proto->set_vendor(desc->device_vendor());
    device_proto->set_compute_capability(
        MakeComputeCapabilityString(desc.get()));
    device_proto->set_core_count(desc->core_count());
  }

  GlobalTopologyProto global_topology;
  if (enable_mock_nccl) {
    std::vector<LocalTopologyProto> local_topologies(num_nodes, local_topology);
    for (int i = 0; i < num_nodes; ++i) {
      local_topologies[i].set_node_id(i);
      // Set a distinct boot_id for each local topology to change slice_index
      // for each node.
      local_topologies[i].set_boot_id(absl::StrCat(i));
    }
    global_topology = BuildGlobalTopology(absl::MakeSpan(local_topologies),
                                          /*assign_global_device_ids=*/true);
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
      }
      auto device = std::make_unique<StreamExecutorGpuDevice>(
          device_proto.global_device_id(), std::move(local_device),
          device_proto.name(), device_proto.vendor(),
          device_proto.compute_capability(), device_proto.core_count(),
          node.node_id(), device_proto.slice_index());
      devices->push_back(std::move(device));
    }
  }
  for (const auto& device : local_device_states) {
    TF_RET_CHECK(device.second == nullptr);
  }
  gpu_executable_run_options->set_gpu_global_device_ids(
      std::move(gpu_device_ids));
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
  if (num_nodes > 1) {
    auto nccl_id_store = std::make_shared<NcclIdStore>(node_id, device_to_node,
                                                       std::move(kv_store));
    gpu_executable_run_options->set_nccl_clique_id_callback(
        [nccl_id_store](const gpu::NcclCliqueKey& key) {
          return nccl_id_store->GetNcclUniqueId(key);
        });
  }
#endif  // GOOGLE_CUDA
  return absl::OkStatus();
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
                               std::move(device_kind), node_id),
      device_vendor_(std::move(device_vendor)),
      slice_index_(slice_index) {
  int64_t core_index = 0;
  description().SetCoreOnChip(core_index);
  std::array<int, 1> coords = {local_device_id().value()};
  description().SetCoords(coords);
  std::vector<int64_t> v_coords(description().coords().begin(),
                                description().coords().end());

  description().SetAttributes(
      {{"coords", xla::PjRtDeviceAttribute(v_coords)},
       {"core_on_chip", xla::PjRtDeviceAttribute(core_index)},
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

int StreamExecutorGpuDevice::core_on_chip() const {
  return description().core_on_chip();
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
  auto host_memory_allocator =
      GetGpuHostAllocator(local_device_states.begin()->second->executor());

  std::vector<std::unique_ptr<PjRtStreamExecutorDevice>> devices;
  auto gpu_run_options = std::make_unique<gpu::GpuExecutableRunOptions>();
  if (options.enable_mock_nccl) {
    gpu_run_options->set_enable_mock_nccl_collectives();
  }
  std::shared_ptr<KeyValueStoreInterface> kv_store = options.kv_store;
  if (options.enable_mock_nccl) {
    kv_store = std::make_shared<InMemoryKeyValueStore>();
  }
  TF_RET_CHECK(options.num_nodes == 1 || kv_store != nullptr);
  TF_RETURN_IF_ERROR(BuildDistributedDevices(
      pjrt_platform_name, std::move(local_device_states), options.node_id,
      options.num_nodes, &devices, gpu_run_options.get(), kv_store,
      options.enable_mock_nccl));

  return std::unique_ptr<PjRtClient>(std::make_unique<StreamExecutorGpuClient>(
      pjrt_platform_name, xla_client, std::move(devices), options.node_id,
      std::move(allocator), std::move(host_memory_allocator),
      options.should_stage_host_to_device_transfers, std::move(gpu_run_options),
      std::move(kv_store)));
}

absl::StatusOr<std::string> StreamExecutorGpuTopologyDescription::Serialize()
    const {
  std::string result;
  if (!tsl::SerializeToStringDeterministic(gpu_topology_.ToProto(), &result)) {
    return absl::InternalError("Failed to serialize gpu_topology");
  }
  return result;
}

absl::StatusOr<Layout> StreamExecutorGpuTopologyDescription::GetDefaultLayout(
    PrimitiveType element_type, absl::Span<const int64_t> dims) const {
  Shape shape = ShapeUtil::MakeShape(element_type, dims);
  return LayoutUtil::GetWithDefaultLayout(shape).layout();
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

}  // namespace xla
